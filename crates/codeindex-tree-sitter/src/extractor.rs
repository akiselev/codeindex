use std::ops::Range;

use anyhow::{Context, Result};
use codeindex_core::{
    EntityKind, ExtractedEntity, ExtractedFile, LanguageId, Representation, RepresentationKind,
    SourceSpan,
};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Node, Parser, QueryCursor};

use crate::language::{LanguageDef, PendingUnit};
use crate::normalizer::{merge_ranges, normalize_for_hash, sha256_hex, strip_ranges};

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

pub fn extract_file(
    def: &LanguageDef,
    source: &str,
    options: &ExtractOptions,
) -> Result<ExtractedFile> {
    Ok(ExtractedFile {
        entities: extract_units(def, source, options)?,
        diagnostics: Vec::new(),
    })
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
    let mut seen_ranges: Vec<Range<usize>> = Vec::new();
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
        if seen_ranges.contains(&range) {
            continue;
        }
        if let Some(unit) = build_unit(def, source, pending, options) {
            seen_ranges.push(range);
            units.push(unit);
        }
    }
    units.sort_by_key(|u| (u.span.start_byte, u.span.end_byte));
    Ok(units)
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
    let embedding_text = strip_ranges(display_source, &merge_ranges(&strip));
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
    Some(ExtractedEntity {
        language: LanguageId::from(def.spec.id.clone()),
        kind: EntityKind::from(pending.kind),
        name: pending.name.unwrap_or_else(|| "<anonymous>".to_string()),
        scope,
        span,
        body_span,
        body_node_count,
        source_hash: source_hash.clone(),
        normalized_body_hash: normalized_body_hash.clone(),
        representations: vec![
            Representation::new(RepresentationKind::FullSource, display_source, source_hash),
            Representation::new(
                RepresentationKind::Implementation,
                embedding_text,
                normalized_body_hash,
            ),
        ],
    })
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
