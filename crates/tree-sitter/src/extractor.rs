use std::collections::HashSet;
use std::ops::Range;

use anyhow::{Context, Result};
use codeindex_core::{
    EntityKind, ExtractedEntity, LanguageId, Representation, RepresentationKind, SourceSpan,
};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Node, Parser, QueryCursor};

use crate::language::{LanguageDef, PendingUnit};
use crate::normalizer::{normalize_for_hash, sha256_hex, strip_ranges};

#[derive(Debug, Clone, Copy)]
pub struct ExtractOptions {
    pub body_node_count_threshold: usize,
    pub max_body_chars: usize,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            body_node_count_threshold: 10,
            max_body_chars: 10_000,
        }
    }
}

pub fn extract_units(
    def: &LanguageDef,
    source: &str,
    options: &ExtractOptions,
) -> Result<Vec<ExtractedEntity>> {
    let mut parser = Parser::new();
    parser
        .set_language(&def.language)
        .with_context(|| format!("loading grammar for {}", def.spec.id))?;
    let tree = parser
        .parse(source, None)
        .with_context(|| format!("parsing {} source", def.spec.id))?;

    let unit_idx = capture_index(def, "unit");
    let name_idx = capture_index(def, "unit.name");
    let body_idx = capture_index(def, "unit.body");
    let strip_idx = capture_index(def, "unit.strip");
    let scope_idx = capture_index(def, "unit.scope");

    let mut units = Vec::new();
    let mut seen_ranges: HashSet<(usize, usize)> = HashSet::new();
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&def.query, tree.root_node(), source.as_bytes());
    while let Some(query_match) = matches.next() {
        let mut unit_node = None;
        let mut pending = PendingUnit {
            node: tree.root_node(),
            body: None,
            kind: "function".to_string(),
            name: None,
            scope: None,
            strip: Vec::new(),
            doc: None,
        };
        for capture in query_match.captures {
            let index = Some(capture.index);
            if index == unit_idx {
                unit_node = Some(capture.node);
            } else if index == name_idx {
                pending.name = Some(source[capture.node.byte_range()].to_string());
            } else if index == body_idx {
                pending.body = Some(capture.node);
            } else if index == strip_idx {
                pending.strip.push(capture.node.byte_range());
            } else if index == scope_idx {
                pending.scope = Some(source[capture.node.byte_range()].to_string());
            }
        }
        let Some(node) = unit_node else { continue };
        pending.node = node;
        for property in def.query.property_settings(query_match.pattern_index) {
            if &*property.key == "unit.kind"
                && let Some(value) = &property.value
            {
                pending.kind = value.to_string();
            }
        }
        if let Some(adapter) = def.adapter {
            adapter.refine(source, &mut pending);
        }
        let range = pending.node.byte_range();
        if seen_ranges.contains(&(range.start, range.end)) {
            continue;
        }
        if let Some(unit) = build_unit(def, source, pending, options) {
            seen_ranges.insert((range.start, range.end));
            units.push(unit);
        }
    }
    units.sort_by_key(|u| (u.span.start_byte, u.span.end_byte));
    Ok(units)
}

/// One raw call site captured from a source file, before cross-corpus
/// resolution. `start_byte` attributes it to the enclosing unit (the unit whose
/// span contains it); `callee_symbol` is the raw callee expression text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawReference {
    pub callee_symbol: String,
    pub call_snippet: String,
    pub start_byte: usize,
    pub start_line: usize,
}

/// Capture every call site in `source` using the language's `references.scm`.
/// Returns an empty vector for languages without a reference query. The caller
/// attributes each reference to a unit by span containment and resolves the
/// symbol against the corpus (see the indexer's Usage pass).
pub fn extract_references(def: &LanguageDef, source: &str) -> Result<Vec<RawReference>> {
    let Some(query) = def.references.as_ref() else {
        return Ok(Vec::new());
    };
    let mut parser = Parser::new();
    parser
        .set_language(&def.language)
        .with_context(|| format!("loading grammar for {}", def.spec.id))?;
    let tree = parser
        .parse(source, None)
        .with_context(|| format!("parsing {} source", def.spec.id))?;

    let callee_idx = query
        .capture_names()
        .iter()
        .position(|n| *n == "ref.callee")
        .map(|i| i as u32);

    // One pass over the source instead of O(lines) per reference.
    let lines: Vec<&str> = source.lines().collect();
    let mut references = Vec::new();
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query, tree.root_node(), source.as_bytes());
    while let Some(query_match) = matches.next() {
        for capture in query_match.captures {
            if Some(capture.index) != callee_idx {
                continue;
            }
            let node = capture.node;
            let symbol = source[node.byte_range()].split_whitespace().collect();
            let line = node.start_position().row;
            let snippet = lines
                .get(line)
                .map(|l| l.trim())
                .unwrap_or("")
                .chars()
                .take(200)
                .collect();
            references.push(RawReference {
                callee_symbol: symbol,
                call_snippet: snippet,
                start_byte: node.start_byte(),
                start_line: line + 1,
            });
        }
    }
    Ok(references)
}

fn capture_index(def: &LanguageDef, name: &str) -> Option<u32> {
    def.query
        .capture_names()
        .iter()
        .position(|n| *n == name)
        .map(|i| i as u32)
}

fn build_unit(
    def: &LanguageDef,
    source: &str,
    pending: PendingUnit<'_>,
    options: &ExtractOptions,
) -> Option<ExtractedEntity> {
    let node = pending.node;
    let start_byte = node.start_byte();
    let end_byte = node.end_byte();
    let display_source = &source[start_byte..end_byte];
    let body_node = pending.body.unwrap_or(node);
    let body_node_count = count_named_nodes(body_node);
    if body_node_count < options.body_node_count_threshold {
        return None;
    }

    let mut strip: Vec<Range<usize>> = pending
        .strip
        .iter()
        .map(|r| r.start.saturating_sub(start_byte)..r.end.saturating_sub(start_byte))
        .collect();
    collect_comment_ranges(node, &def.spec.comment_nodes, start_byte, &mut strip);
    let embedding_text = strip_ranges(display_source, &strip);
    let normalized = normalize_for_hash(&embedding_text);
    if normalized.is_empty() || embedding_text.chars().count() > options.max_body_chars {
        return None;
    }
    let scope = pending
        .scope
        .clone()
        .or_else(|| recover_scope(def, source, node));
    let span = SourceSpan::new(
        start_byte,
        end_byte,
        node.start_position().row + 1,
        node.end_position().row + 1,
    );
    let body_span = Some(SourceSpan::new(
        body_node.start_byte(),
        body_node.end_byte(),
        body_node.start_position().row + 1,
        body_node.end_position().row + 1,
    ));
    let source_hash = sha256_hex(display_source);
    let normalized_body_hash = sha256_hex(&normalized);
    let name = pending.name.unwrap_or_else(|| "<anonymous>".to_string());

    let mut representations = vec![
        Representation::new(
            RepresentationKind::FullSource,
            display_source,
            source_hash.clone(),
        ),
        Representation::new(
            RepresentationKind::Implementation,
            embedding_text,
            normalized_body_hash.clone(),
        ),
    ];

    // Signature: the declaration up to the body — for a function, everything
    // before the `{ ... }` block (params, return type, generics). Empty when
    // there is no distinct body node (e.g. a bare closure body).
    if let Some(body) = pending.body
        && body.start_byte() > start_byte
    {
        let signature = source[start_byte..body.start_byte()].trim();
        if !signature.is_empty() {
            representations.push(Representation::new(
                RepresentationKind::Signature,
                signature,
                sha256_hex(signature),
            ));
        }
    }

    // Documentation: an adapter-identified documentation literal (Python
    // docstring) when present, otherwise the contiguous run of comment-node
    // siblings immediately preceding the unit. Comment text is stored raw —
    // markers carry meaning for embedding.
    let doc = pending
        .doc
        .as_ref()
        .and_then(|range| clean_docstring(&source[range.clone()]))
        .or_else(|| leading_documentation(def, source, node));
    if let Some(doc) = doc
        && !doc.is_empty()
    {
        representations.push(Representation::new(
            RepresentationKind::Documentation,
            doc.clone(),
            sha256_hex(&doc),
        ));
    }

    // Symbol: the qualified name, a short high-signal channel for
    // name-oriented retrieval.
    let symbol = match &scope {
        Some(scope) => format!("{scope}.{name}"),
        None => name.clone(),
    };
    representations.push(Representation::new(
        RepresentationKind::Symbol,
        symbol.clone(),
        sha256_hex(&symbol),
    ));

    Some(ExtractedEntity {
        language: LanguageId::from(def.spec.id.clone()),
        kind: EntityKind::from(pending.kind),
        name,
        scope,
        span,
        body_span,
        body_node_count,
        source_hash,
        normalized_body_hash,
        representations,
    })
}

/// Strip string prefixes and quotes from an adapter-identified docstring
/// literal (`"""…"""`, `'''…'''`, `r"""…"""`, plain quotes). Only raw/unicode
/// prefixes appear here — the adapter never marks f-string or bytes literals
/// as docstrings.
fn clean_docstring(literal: &str) -> Option<String> {
    let trimmed = literal.trim();
    let trimmed = trimmed.trim_start_matches(['r', 'R', 'u', 'U']);
    let unquoted = ["\"\"\"", "'''", "\"", "'"].iter().find_map(|quote| {
        trimmed
            .strip_prefix(quote)
            .and_then(|rest| rest.strip_suffix(quote))
    })?;
    let text = unquoted.trim();
    (!text.is_empty()).then(|| text.to_string())
}

/// The contiguous block of comment-node siblings directly above `node`, joined
/// top-to-bottom. Used as the `Documentation` channel. Attribute/annotation
/// nodes between the declaration and its comments are skipped — but only until
/// the comment block starts, so `/// doc` + `#[inline]` + `fn` yields the doc
/// comment while an unrelated comment sitting above the attributes of an
/// already-documented item is never merged in. When the declaration is wrapped
/// in a container node (Python's `decorated_definition`), the walk continues
/// from the container's preceding siblings.
fn leading_documentation(def: &LanguageDef, source: &str, node: Node<'_>) -> Option<String> {
    let comment_kinds = &def.spec.comment_nodes;
    if comment_kinds.is_empty() {
        return None;
    }
    let attribute_kinds = &def.spec.attribute_nodes;
    let container_kinds = &def.spec.doc_container_nodes;
    let mut comments: Vec<Range<usize>> = Vec::new();
    let mut anchor = node;
    let mut sibling = anchor.prev_sibling();
    loop {
        let Some(current) = sibling else {
            match anchor.parent() {
                Some(parent)
                    if comments.is_empty()
                        && container_kinds.iter().any(|kind| kind == parent.kind()) =>
                {
                    anchor = parent;
                    sibling = parent.prev_sibling();
                    continue;
                }
                _ => break,
            }
        };
        if comment_kinds.iter().any(|kind| kind == current.kind()) {
            comments.push(current.byte_range());
        } else if !comments.is_empty() || !attribute_kinds.iter().any(|kind| kind == current.kind())
        {
            break;
        }
        sibling = current.prev_sibling();
    }
    if comments.is_empty() {
        return None;
    }
    comments.reverse();
    let text = comments
        .into_iter()
        .map(|range| source[range].trim_end())
        .collect::<Vec<_>>()
        .join("\n");
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_string())
}

fn count_named_nodes(node: Node<'_>) -> usize {
    let mut count = 0usize;
    let mut cursor = node.walk();
    let mut done = false;
    while !done {
        if cursor.node().is_named() {
            count += 1;
        }
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() || cursor.node() == node {
                done = true;
                break;
            }
        }
    }
    count.saturating_sub(1)
}

fn collect_comment_ranges(
    node: Node<'_>,
    kinds: &[String],
    unit_start: usize,
    out: &mut Vec<Range<usize>>,
) {
    let mut stack = vec![node];
    while let Some(current) = stack.pop() {
        if kinds.iter().any(|kind| kind == current.kind()) {
            let range = current.byte_range();
            out.push(range.start - unit_start..range.end - unit_start);
            continue;
        }
        for i in 0..current.child_count() as u32 {
            if let Some(child) = current.child(i) {
                stack.push(child);
            }
        }
    }
}

fn recover_scope(def: &LanguageDef, source: &str, node: Node<'_>) -> Option<String> {
    let mut parts = Vec::new();
    let mut current = node.parent();
    while let Some(ancestor) = current {
        for rule in &def.spec.scopes {
            if ancestor.kind() == rule.kind
                && let Some(name) = ancestor.child_by_field_name(rule.field.as_str())
            {
                parts.push(source[name.byte_range()].to_string());
            }
        }
        current = ancestor.parent();
    }
    if parts.is_empty() {
        None
    } else {
        parts.reverse();
        Some(parts.join("."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LanguageRegistry;

    fn small_units(language: &str, source: &str) -> Vec<ExtractedEntity> {
        let def = LanguageRegistry::global().get(language).unwrap();
        extract_units(
            def,
            source,
            &ExtractOptions {
                body_node_count_threshold: 1,
                max_body_chars: 10_000,
            },
        )
        .unwrap()
    }

    #[test]
    fn cfg_test_detection_is_structural() {
        let units = small_units(
            "rust",
            r#"
#[cfg(test)]
fn direct() { do_it() }

#[cfg(all(test, not(windows)))]
fn nested() { do_it() }

#[cfg(feature = "testing")]
fn feature_gated() { do_it() }

#[cfg(not(test))]
fn negated() { do_it() }
"#,
        );
        let scope = |name: &str| {
            units
                .iter()
                .find(|u| u.name == name)
                .unwrap_or_else(|| panic!("unit {name} missing"))
                .scope
                .clone()
        };
        assert_eq!(scope("direct").as_deref(), Some("tests"));
        assert_eq!(scope("nested").as_deref(), Some("tests"));
        assert_eq!(scope("feature_gated"), None);
        assert_eq!(scope("negated"), None);
    }

    #[test]
    fn python_docstring_feeds_documentation_channel() {
        let units = small_units(
            "python",
            "def parse(data):\n    \"\"\"Parse data into an AST.\"\"\"\n    return build(data)\n",
        );
        assert_eq!(units.len(), 1);
        assert_eq!(
            units[0].representation_text(&RepresentationKind::Documentation),
            Some("Parse data into an AST.")
        );
        // The docstring stays stripped from the Implementation channel.
        assert!(
            !units[0]
                .representation_text(&RepresentationKind::Implementation)
                .unwrap()
                .contains("Parse data into an AST")
        );
    }

    #[test]
    fn leading_docs_survive_intervening_attributes() {
        let units = small_units(
            "rust",
            "/// Documented function.\n#[inline]\nfn documented() { do_it() }\n",
        );
        assert_eq!(units.len(), 1);
        assert_eq!(
            units[0].representation_text(&RepresentationKind::Documentation),
            Some("/// Documented function.")
        );
    }

    #[test]
    fn unrelated_comments_above_attributes_are_not_merged_into_docs() {
        let units = small_units(
            "rust",
            "// unrelated section note\n#[derive(Debug)]\n/// Documented.\nfn documented() { do_it() }\n",
        );
        assert_eq!(units.len(), 1);
        assert_eq!(
            units[0].representation_text(&RepresentationKind::Documentation),
            Some("/// Documented.")
        );
    }

    #[test]
    fn decorated_python_function_keeps_leading_comment_docs() {
        let units = small_units(
            "python",
            "# validates user input\n@lru_cache\ndef check(x):\n    return x + 1\n",
        );
        assert_eq!(units.len(), 1);
        assert_eq!(
            units[0].representation_text(&RepresentationKind::Documentation),
            Some("# validates user input")
        );
    }

    #[test]
    fn python_fstring_first_statement_is_not_a_docstring() {
        let units = small_units(
            "python",
            "def h(x):\n    f\"\"\"Doc {x}.\"\"\"\n    return x\n",
        );
        assert_eq!(units.len(), 1);
        assert_eq!(
            units[0].representation_text(&RepresentationKind::Documentation),
            None
        );
        // The runtime f-string expression must stay in the Implementation
        // channel rather than being stripped as a docstring.
        assert!(
            units[0]
                .representation_text(&RepresentationKind::Implementation)
                .unwrap()
                .contains("Doc {x}")
        );
    }

    #[test]
    fn extracts_parser_neutral_rust_entity() {
        let def = LanguageRegistry::global().get("rust").unwrap();
        let units = extract_units(
            def,
            "fn add(a: i32, b: i32) -> i32 { a + b }",
            &ExtractOptions {
                body_node_count_threshold: 1,
                max_body_chars: 10_000,
            },
        )
        .unwrap();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].name, "add");
        assert_eq!(units[0].kind.as_str(), "function");
        assert!(
            units[0]
                .representation_text(&RepresentationKind::Implementation)
                .unwrap()
                .contains("a + b")
        );
    }
}
