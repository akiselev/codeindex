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
    /// Bodies whose stripped text exceeds this also emit fragment entities
    /// split at AST boundaries (docs/oversized-units-plan.md).
    pub fragment_threshold_chars: usize,
    /// Greedy packing target for one fragment.
    pub fragment_target_chars: usize,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            body_node_count_threshold: 10,
            max_body_chars: 10_000,
            fragment_threshold_chars: 6_000,
            fragment_target_chars: 3_000,
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
    let mut candidates: Vec<PendingUnit<'_>> = Vec::new();
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
        seen_ranges.insert((range.start, range.end));
        candidates.push(pending);
    }

    // Size every candidate up front so nested oversized units can be
    // detected before deciding who fragments: an inner closure that itself
    // exceeds the threshold fragments the same lines as its enclosing
    // function, doubling fragment count for no extra coverage (measured on
    // `spawn_senders`/`walker.run(...)` — 6 fragments where 3 carried the
    // information; docs/oversized-units-plan.md track 3). A covering child
    // only disqualifies its parent from fragmenting if the child will
    // actually produce fragments itself (`would_fragment`) — an oversized
    // body that isn't AST-splittable (e.g. one giant string/byte literal, as
    // in generated code) must not suppress every ancestor's attempt too,
    // or the whole nested chain loses fragment-level addressability.
    let fragment_target = options.fragment_target_chars.max(256);
    let sizes: Vec<(Range<usize>, usize, bool)> = candidates
        .iter()
        .map(|pending| {
            let (_, _, stripped_chars) = strip_and_measure(def, source, pending);
            let fragmentable = stripped_chars > options.fragment_threshold_chars
                && would_fragment(pending, source, fragment_target);
            (pending.node.byte_range(), stripped_chars, fragmentable)
        })
        .collect();

    for (index, pending) in candidates.into_iter().enumerate() {
        let (range, stripped_chars, _) = &sizes[index];
        let covered_by_oversized_child = sizes.iter().enumerate().any(
            |(other_index, (other_range, other_chars, other_fragmentable))| {
                other_index != index
                    && other_range.start >= range.start
                    && other_range.end <= range.end
                    && (other_range.start, other_range.end) != (range.start, range.end)
                    && *other_fragmentable
                    && (*other_chars as f64) >= 0.9 * (*stripped_chars as f64)
            },
        );
        if let Some((unit, fragments)) =
            build_unit(def, source, pending, options, covered_by_oversized_child)
        {
            units.push(unit);
            units.extend(fragments);
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

/// Compute the stripped (comment/marker-free) embedding text for a pending
/// unit and its char count, ahead of deciding whether it fragments. Shared
/// between the sizing pass (which needs only the count) and `build_unit`
/// (which needs the text too), so the two never disagree on size.
fn strip_and_measure(
    def: &LanguageDef,
    source: &str,
    pending: &PendingUnit<'_>,
) -> (Vec<Range<usize>>, String, usize) {
    let node = pending.node;
    let start_byte = node.start_byte();
    let end_byte = node.end_byte();
    let display_source = &source[start_byte..end_byte];
    let mut strip: Vec<Range<usize>> = pending
        .strip
        .iter()
        .map(|r| r.start.saturating_sub(start_byte)..r.end.saturating_sub(start_byte))
        .collect();
    collect_comment_ranges(node, &def.spec.comment_nodes, start_byte, &mut strip);
    let embedding_text = strip_ranges(display_source, &strip);
    let stripped_chars = embedding_text.chars().count();
    (strip, embedding_text, stripped_chars)
}

fn build_unit(
    def: &LanguageDef,
    source: &str,
    pending: PendingUnit<'_>,
    options: &ExtractOptions,
    covered_by_oversized_child: bool,
) -> Option<(ExtractedEntity, Vec<ExtractedEntity>)> {
    let node = pending.node;
    let start_byte = node.start_byte();
    let end_byte = node.end_byte();
    let display_source = &source[start_byte..end_byte];
    let body_node = pending.body.unwrap_or(node);
    let body_node_count = count_named_nodes(body_node);
    if body_node_count < options.body_node_count_threshold {
        return None;
    }

    let (strip, mut embedding_text, stripped_chars) = strip_and_measure(def, source, &pending);
    // Oversized bodies truncate rather than vanish: interpreter dispatch
    // loops and generated builders are exactly the units agents ask about,
    // and a dropped unit is invisible to every retrieval mode (measured:
    // jq's `jq_next` was unfindable because it exceeded the cap). The
    // normalized hash is computed over the truncated text so identity
    // matches the stored representation. Coverage past the cap comes from
    // fragment entities emitted below.
    if stripped_chars > options.max_body_chars {
        embedding_text = embedding_text
            .chars()
            .take(options.max_body_chars)
            .collect();
    }
    let normalized = normalize_for_hash(&embedding_text);
    if normalized.is_empty() {
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
    let name = pending
        .name
        .clone()
        .unwrap_or_else(|| "<anonymous>".to_string());

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

    // A unit whose body is almost entirely one oversized child (an inner
    // closure or nested function that itself exceeds the threshold) skips
    // fragmenting: the child fragments the same lines on its own, and
    // fragmenting both just doubles the fragment count over identical text
    // (docs/oversized-units-plan.md track 3).
    let fragments =
        if stripped_chars > options.fragment_threshold_chars && !covered_by_oversized_child {
            build_fragments(def, source, &pending, &name, &strip, options)
        } else {
            Vec::new()
        };

    Some((
        ExtractedEntity {
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
        },
        fragments,
    ))
}

/// Whether a candidate's body would actually produce ≥2 AST-boundary
/// fragments — the same leaf-count gate `build_fragments` uses internally,
/// exposed so a covering ancestor can be trusted to skip fragmenting only
/// when the child is guaranteed to pick up the coverage itself. An
/// unsplittable oversized body (e.g. one giant string/byte literal — not
/// unheard of in generated code) has fewer than 2 leaves and must not
/// suppress every enclosing unit's own fragmenting attempt too.
fn would_fragment(pending: &PendingUnit<'_>, source: &str, target: usize) -> bool {
    let body_node = pending.body.unwrap_or(pending.node);
    let mut leaves: Vec<Node<'_>> = Vec::new();
    collect_fragment_leaves(body_node, source, target, &mut leaves);
    leaves.len() >= 2
}

/// Fragment an oversized body at AST boundaries: descend until nodes fit the
/// packing target, then greedily pack consecutive nodes into fragments.
/// Fragments are ordinary entities (`kind = fragment`, `scope = parent name`,
/// names `parent#N`) that flow through representation, embedding, and lexical
/// indexing unchanged — they give retrieval a handle on the *inside* of a
/// giant function (docs/oversized-units-plan.md).
fn build_fragments(
    def: &LanguageDef,
    source: &str,
    pending: &PendingUnit<'_>,
    parent_name: &str,
    parent_strip: &[Range<usize>],
    options: &ExtractOptions,
) -> Vec<ExtractedEntity> {
    let parent_node = pending.node;
    let body_node = pending.body.unwrap_or(parent_node);
    let parent_start = parent_node.start_byte();
    let target = options.fragment_target_chars.max(256);

    let mut leaves: Vec<Node<'_>> = Vec::new();
    collect_fragment_leaves(body_node, source, target, &mut leaves);
    if leaves.len() < 2 {
        return Vec::new();
    }

    // Greedy pack consecutive leaves toward the target size.
    let mut packs: Vec<(usize, usize)> = Vec::new(); // leaf index ranges
    let mut pack_start = 0;
    let mut pack_chars = 0usize;
    for (index, leaf) in leaves.iter().enumerate() {
        let leaf_chars = source[leaf.byte_range()].chars().count();
        if pack_chars > 0 && pack_chars + leaf_chars > target {
            packs.push((pack_start, index));
            pack_start = index;
            pack_chars = 0;
        }
        pack_chars += leaf_chars;
    }
    packs.push((pack_start, leaves.len()));
    if packs.len() < 2 {
        return Vec::new();
    }

    let mut fragments = Vec::new();
    for (ordinal, (from, to)) in packs.iter().enumerate() {
        let first = &leaves[*from];
        let last = &leaves[to - 1];
        let start_byte = first.start_byte();
        let end_byte = last.end_byte();
        let display_source = &source[start_byte..end_byte];
        // Reuse the parent's comment/strip ranges, re-based onto this slice.
        let strip: Vec<Range<usize>> = parent_strip
            .iter()
            .filter_map(|range| {
                let absolute = (range.start + parent_start)..(range.end + parent_start);
                let clipped = absolute.start.max(start_byte)..absolute.end.min(end_byte);
                (clipped.start < clipped.end)
                    .then(|| (clipped.start - start_byte)..(clipped.end - start_byte))
            })
            .collect();
        let mut fragment_text = strip_ranges(display_source, &strip);
        if fragment_text.chars().count() > options.max_body_chars {
            fragment_text = fragment_text.chars().take(options.max_body_chars).collect();
        }
        let normalized = normalize_for_hash(&fragment_text);
        if normalized.is_empty() {
            continue;
        }
        let name = format!("{parent_name}#{}", ordinal + 1);
        let symbol = format!("{parent_name}.{name}");
        let source_hash = sha256_hex(display_source);
        let normalized_body_hash = sha256_hex(&normalized);
        let span = SourceSpan::new(
            start_byte,
            end_byte,
            first.start_position().row + 1,
            last.end_position().row + 1,
        );
        fragments.push(ExtractedEntity {
            language: LanguageId::from(def.spec.id.clone()),
            kind: EntityKind::from("fragment".to_string()),
            name,
            scope: Some(parent_name.to_string()),
            span,
            body_span: Some(span),
            body_node_count: leaves[*from..*to]
                .iter()
                .map(|n| count_named_nodes(*n))
                .sum(),
            source_hash,
            normalized_body_hash: normalized_body_hash.clone(),
            representations: vec![
                Representation::new(
                    RepresentationKind::FullSource,
                    display_source,
                    sha256_hex(display_source),
                ),
                Representation::new(
                    RepresentationKind::Implementation,
                    fragment_text,
                    normalized_body_hash,
                ),
                Representation::new(
                    RepresentationKind::Symbol,
                    symbol.clone(),
                    sha256_hex(&symbol),
                ),
            ],
        });
    }
    fragments
}

/// Descend into nodes larger than the packing target so one giant child (a
/// match/switch statement, a long block) splits at its own boundaries
/// instead of forming one oversized fragment.
fn collect_fragment_leaves<'tree>(
    node: Node<'tree>,
    source: &str,
    target: usize,
    leaves: &mut Vec<Node<'tree>>,
) {
    let mut cursor = node.walk();
    let children: Vec<Node<'tree>> = node.named_children(&mut cursor).collect();
    for child in children {
        let chars = source[child.byte_range()].chars().count();
        if chars <= target || child.named_child_count() == 0 {
            leaves.push(child);
        } else {
            collect_fragment_leaves(child, source, target, leaves);
        }
    }
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
                ..ExtractOptions::default()
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
                ..ExtractOptions::default()
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

    fn monster_source(marker: &str) -> String {
        // Three distinct regions, each ~big enough to become its own
        // fragment at the default 3k target.
        let region = |tag: &str| -> String {
            (0..60)
                .map(|i| format!("    let {tag}_{i} = compute_{tag}({i});\n"))
                .collect::<String>()
        };
        format!(
            "fn dispatch_monster(input: u32) -> u32 {{\n{}{}{}    {marker}\n    input\n}}\n",
            region("alpha"),
            region("beta"),
            region("gamma"),
        )
    }

    #[test]
    fn oversized_bodies_emit_fragments_at_ast_boundaries() {
        let source = monster_source("let sentinel = 1;");
        let units = small_units("rust", &source);
        let parent = units
            .iter()
            .find(|unit| unit.name == "dispatch_monster")
            .expect("parent present");
        let fragments: Vec<_> = units
            .iter()
            .filter(|unit| unit.kind.as_str() == "fragment")
            .collect();
        assert!(
            fragments.len() >= 2,
            "expected multiple fragments, got {}",
            fragments.len()
        );
        for (index, fragment) in fragments.iter().enumerate() {
            assert_eq!(fragment.name, format!("dispatch_monster#{}", index + 1));
            assert_eq!(fragment.scope.as_deref(), Some("dispatch_monster"));
            assert!(fragment.span.start_byte >= parent.span.start_byte);
            assert!(fragment.span.end_byte <= parent.span.end_byte);
            let implementation = fragment
                .representation(&codeindex_core::RepresentationKind::Implementation)
                .expect("fragment implementation");
            assert!(!implementation.content.is_empty());
        }
        // Fragments tile the body in order without overlap.
        for pair in fragments.windows(2) {
            assert!(pair[0].span.end_byte <= pair[1].span.start_byte);
        }
    }

    #[test]
    fn editing_one_region_leaves_sibling_fragments_stable() {
        let before = small_units("rust", &monster_source("let sentinel = 1;"));
        let after = small_units("rust", &monster_source("let sentinel = 2;"));
        let hashes = |units: &[ExtractedEntity]| -> Vec<(String, String)> {
            units
                .iter()
                .filter(|unit| unit.kind.as_str() == "fragment")
                .map(|unit| (unit.name.clone(), unit.normalized_body_hash.clone()))
                .collect()
        };
        let before = hashes(&before);
        let after = hashes(&after);
        assert_eq!(before.len(), after.len());
        let changed: Vec<_> = before
            .iter()
            .zip(after.iter())
            .filter(|(b, a)| b.1 != a.1)
            .collect();
        assert_eq!(
            changed.len(),
            1,
            "exactly the edited region's fragment must change: {changed:?}"
        );
    }

    fn nested_oversized_closure_source() -> String {
        let body: String = (0..300)
            .map(|i| format!("        let step_{i} = compute_{i}({i});\n"))
            .collect();
        format!(
            "fn spawn_senders(input: u32) -> u32 {{\n    \
             let total = (0..input).map(|i| {{\n{body}        i\n    }}).sum();\n    \
             total\n}}\n"
        )
    }

    #[test]
    fn nested_oversized_child_suppresses_parent_fragmenting() {
        // The outer function's body is almost entirely one closure, and the
        // closure itself is oversized — both would fragment over the same
        // lines without the covered-by-oversized-child suppression.
        let source = nested_oversized_closure_source();
        let units = small_units("rust", &source);

        let parent = units
            .iter()
            .find(|unit| unit.name == "spawn_senders")
            .expect("outer function present");
        let closure = units
            .iter()
            .find(|unit| unit.kind.as_str() == "closure")
            .expect("nested closure present");

        let parent_fragments: Vec<_> = units
            .iter()
            .filter(|unit| {
                unit.kind.as_str() == "fragment"
                    && unit.scope.as_deref() == Some(parent.name.as_str())
            })
            .collect();
        let closure_fragments: Vec<_> = units
            .iter()
            .filter(|unit| {
                unit.kind.as_str() == "fragment"
                    && unit.scope.as_deref() == Some(closure.name.as_str())
            })
            .collect();

        assert!(
            parent_fragments.is_empty(),
            "parent must not double-fragment lines its oversized child already fragments"
        );
        assert!(
            closure_fragments.len() >= 2,
            "the oversized closure itself must still fragment"
        );
    }

    /// fd's real `WorkerState::spawn_senders` (docs/oversized-units-plan.md
    /// track 3 cost finding): three closures deep, each nested one entirely
    /// covering its parent's body. Only the innermost (`Box::new(...)`, the
    /// one AST-boundary fragmenting actually helps) should fragment; the two
    /// enclosing units are each ≥90% covered by an oversized descendant and
    /// must suppress.
    const SPAWN_SENDERS_TRIPLE_NESTED: &str = r#"
struct WorkerState;
impl WorkerState {
    fn spawn_senders(&self, walker: WalkParallel, tx: Sender<Batch>) {
        walker.run(|| {
            let patterns = &self.patterns;
            let config = &self.config;
            let quit_flag = self.quit_flag.as_ref();

            let mut limit = 0x100;
            if let Some(cmd) = &config.command
                && !cmd.in_batch_mode()
                && config.threads > 1
            {
                limit = 1;
            }
            let mut tx = BatchSender::new(tx.clone(), limit);

            Box::new(move |entry| {
                if quit_flag.load(Ordering::Relaxed) {
                    return WalkState::Quit;
                }

                if let Ok(e) = &entry {
                    if e.file_type().is_some_and(|t| t.is_dir()) {
                        let entry_path = e.path();
                        if config
                            .ignore_contain
                            .iter()
                            .any(|ic| entry_path.join(ic).exists())
                        {
                            return WalkState::Skip;
                        }
                    }
                    if e.depth() == 0 {
                        return WalkState::Continue;
                    }
                }
                let entry = match entry {
                    Ok(e) => DirEntry::normal(e),
                    Err(ignore::Error::WithPath {
                        path,
                        err: inner_err,
                    }) if inner_err
                        .io_error()
                        .is_some_and(|io_error| io_error.kind() == io::ErrorKind::NotFound)
                        && path
                            .symlink_metadata()
                            .ok()
                            .is_some_and(|m| m.file_type().is_symlink()) =>
                    {
                        DirEntry::broken_symlink(path)
                    }
                    Err(err) => {
                        return match tx.send(WorkerResult::Error(err)) {
                            Ok(_) => WalkState::Continue,
                            Err(_) => WalkState::Quit,
                        };
                    }
                };

                if let Some(min_depth) = config.min_depth
                    && entry.depth().is_none_or(|d| d < min_depth)
                {
                    return WalkState::Continue;
                }

                let entry_path = entry.path();
                let search_str = search_str_for_entry(entry_path, config.full_path_base.as_deref());

                if !patterns
                    .iter()
                    .all(|pat| pat.is_match(&filesystem::osstr_to_bytes(search_str.as_ref())))
                {
                    return WalkState::Continue;
                }

                if let Some(ref exts_regex) = config.extensions {
                    if let Some(path_str) = entry_path.file_name() {
                        if !exts_regex.is_match(&filesystem::osstr_to_bytes(path_str)) {
                            return WalkState::Continue;
                        }
                    } else {
                        return WalkState::Continue;
                    }
                }

                if let Some(ref file_types) = config.file_types
                    && file_types.should_ignore(&entry)
                {
                    return WalkState::Continue;
                }

                #[cfg(unix)]
                {
                    if let Some(ref owner_constraint) = config.owner_constraint {
                        if let Some(metadata) = entry.metadata() {
                            if !owner_constraint.matches(metadata) {
                                return WalkState::Continue;
                            }
                        } else {
                            return WalkState::Continue;
                        }
                    }
                }

                if !config.size_constraints.is_empty() {
                    if entry_path.is_file() {
                        if let Some(metadata) = entry.metadata() {
                            let file_size = metadata.len();
                            if config
                                .size_constraints
                                .iter()
                                .any(|sc| !sc.is_within(file_size))
                            {
                                return WalkState::Continue;
                            }
                        } else {
                            return WalkState::Continue;
                        }
                    } else {
                        return WalkState::Continue;
                    }
                }

                if !config.time_constraints.is_empty() {
                    let mut matched = false;
                    if let Some(metadata) = entry.metadata()
                        && let Ok(modified) = metadata.modified()
                    {
                        matched = config
                            .time_constraints
                            .iter()
                            .all(|tf| tf.applies_to(&modified));
                    }
                    if !matched {
                        return WalkState::Continue;
                    }
                }

                if config.is_printing()
                    && let Some(ls_colors) = &config.ls_colors
                {
                    entry.style(ls_colors);
                }

                let send_result = tx.send(WorkerResult::Entry(entry));

                if send_result.is_err() {
                    return WalkState::Quit;
                }

                if config.prune {
                    return WalkState::Skip;
                }

                WalkState::Continue
            })
        });
    }
}
"#;

    #[test]
    fn triple_nested_oversized_closures_only_fragment_the_innermost() {
        // fd's own per-project fragment_threshold_chars (docs above trim a
        // few comment lines versus the real file, so the default 6000
        // threshold no longer trips; 3000/1500 mirrors the actual config).
        let def = LanguageRegistry::global().get("rust").unwrap();
        let units = extract_units(
            def,
            SPAWN_SENDERS_TRIPLE_NESTED,
            &ExtractOptions {
                body_node_count_threshold: 1,
                max_body_chars: 10_000,
                fragment_threshold_chars: 3_000,
                fragment_target_chars: 1_500,
            },
        )
        .unwrap();
        let fragment_scopes: std::collections::HashSet<&str> = units
            .iter()
            .filter(|unit| unit.kind.as_str() == "fragment")
            .filter_map(|unit| unit.scope.as_deref())
            .collect();
        assert!(
            fragment_scopes.contains("Box::new(...)"),
            "the innermost oversized closure must still fragment: {fragment_scopes:?}"
        );
        assert!(
            !fragment_scopes.contains("spawn_senders"),
            "spawn_senders is ~100% covered by nested oversized closures and must not double-fragment"
        );
        assert!(
            !fragment_scopes.contains("walker.run(...)"),
            "walker.run(...)'s closure is ~99% covered by Box::new(...) and must not double-fragment"
        );
    }

    #[test]
    fn unsplittable_covering_child_does_not_suppress_the_parent() {
        // An oversized child that can't itself produce ≥2 AST-boundary
        // fragments (here: a closure whose entire body is one giant string
        // literal — a single leaf token, not unheard of in generated code)
        // must not disqualify its parent from fragmenting too. Without the
        // `would_fragment` gate, size/coverage alone would suppress the
        // parent, and neither level would fragment — a total loss of
        // fragment-level addressability for the whole region.
        let giant_literal: String = std::iter::repeat_n('A', 5_000).collect();
        let source = format!(
            "fn outer() -> String {{\n    \
             let setup_a = compute_a();\n    \
             let setup_b = compute_b();\n    \
             let closure = || {{\n        \"{giant_literal}\"\n    }};\n    \
             closure()\n}}\n"
        );
        let def = LanguageRegistry::global().get("rust").unwrap();
        let units = extract_units(
            def,
            &source,
            &ExtractOptions {
                body_node_count_threshold: 1,
                max_body_chars: 10_000,
                fragment_threshold_chars: 3_000,
                fragment_target_chars: 1_500,
            },
        )
        .unwrap();
        let closure = units
            .iter()
            .find(|unit| unit.kind.as_str() == "closure")
            .expect("closure present");
        assert!(
            !units.iter().any(|unit| unit.kind.as_str() == "fragment"
                && unit.scope.as_deref() == Some(closure.name.as_str())),
            "the unsplittable closure itself must not fragment"
        );
        assert!(
            units
                .iter()
                .any(|unit| unit.kind.as_str() == "fragment"
                    && unit.scope.as_deref() == Some("outer")),
            "outer must still fragment on its own boundaries since its covering child can't"
        );
    }

    #[test]
    fn small_bodies_do_not_fragment() {
        let units = small_units(
            "rust",
            "fn compact(input: u32) -> u32 { let a = input + 1; let b = a * 2; b }",
        );
        assert!(units.iter().all(|unit| unit.kind.as_str() != "fragment"));
    }

    #[test]
    fn oversized_bodies_truncate_instead_of_vanishing() {
        let huge_body: String = (0..2000)
            .map(|i| format!("    let variable_{i} = compute_{i}(input);\n"))
            .collect();
        let source = format!("fn interpreter_dispatch_loop(input: u32) {{\n{huge_body}}}\n");
        let units = small_units("rust", &source);
        let unit = units
            .iter()
            .find(|unit| unit.name == "interpreter_dispatch_loop")
            .expect("oversized function must stay indexed");
        let implementation = unit
            .representation(&codeindex_core::RepresentationKind::Implementation)
            .expect("implementation representation present");
        assert!(implementation.content.chars().count() <= 10_000);
    }
}
