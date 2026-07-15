#![forbid(unsafe_code)]

//! Language-server-backed enrichment for a published codeindex corpus.
//!
//! Runs as a post-publish, generation-keyed analysis pass — never inside the
//! per-document staging loop, whose payloads must stay a pure function of one
//! document's content. For each indexed entity the pass asks the server for
//! hover information (persisted as a derived `typed_signature` representation
//! channel, embeddable like any other channel), and resolves the frontend's
//! raw call sites through `textDocument/definition` into typed, exact
//! `calls` relations.

pub mod client;

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use codeindex_core::{EntityId, RepresentationKind, RepresentationOrigin};
use codeindex_sqlite::storage::RelationRecord;
use codeindex_sqlite::{Db, FileRecord, NewRepresentation};
use serde_json::{Value, json};
use sha2::Digest;

use client::{LspClient, file_uri, uri_to_path};

/// The derived representation channel enriched hovers land in.
pub const TYPED_SIGNATURE_CHANNEL: &str = "typed_signature";

/// How to launch a language server for one language.
#[derive(Debug, Clone)]
pub struct LspServer {
    /// Language id the server covers (matches indexed `language_id`).
    pub language_id: String,
    pub command: String,
    pub args: Vec<String>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct EnrichmentReport {
    pub files_visited: usize,
    pub units_visited: usize,
    pub typed_signatures: usize,
    pub relations: usize,
}

/// Enrich one indexed project through a language server rooted at
/// `project_root` (which must be the same tree that was indexed).
pub fn enrich_project(
    db: &Db,
    project_label: &str,
    project_root: &Path,
    server: &LspServer,
) -> Result<EnrichmentReport> {
    let project = db
        .get_project(project_label)?
        .with_context(|| format!("project {project_label:?} is not indexed"))?;
    let generation = db.current_generation()?;
    let provenance = format!("lsp:{}", server.command);
    let files: Vec<FileRecord> = db
        .list_files(project.id)?
        .into_iter()
        .filter(|file| file.language_id == server.language_id)
        .collect();
    let root = project_root
        .canonicalize()
        .with_context(|| format!("canonicalizing {}", project_root.display()))?;

    let mut lsp = LspClient::start(&server.command, &server.args, &root)?;
    let mut report = EnrichmentReport::default();
    let mut relations: Vec<RelationRecord> = Vec::new();

    // Sources and open notifications first, so cross-file resolution sees the
    // complete workspace.
    let mut sources: HashMap<i64, String> = HashMap::new();
    let mut files_by_path: HashMap<String, &FileRecord> = HashMap::new();
    for file in &files {
        let path = root.join(&file.relative_path);
        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        lsp.notify(
            "textDocument/didOpen",
            json!({
                "textDocument": {
                    "uri": file_uri(&path),
                    "languageId": server.language_id,
                    "version": 1,
                    "text": source,
                }
            }),
        )?;
        sources.insert(file.id, source);
        files_by_path.insert(file.relative_path.clone(), file);
        report.files_visited += 1;
    }

    // Entities per file, for span-based attribution of definition targets.
    let mut units_by_file = HashMap::new();
    for file in &files {
        units_by_file.insert(file.id, db.list_units_for_file(file.id)?);
    }

    let mut first_request = true;
    for file in &files {
        let source = &sources[&file.id];
        let uri = file_uri(&root.join(&file.relative_path));
        let units = &units_by_file[&file.id];

        for unit in units {
            report.units_visited += 1;
            let Some(name_byte) =
                declared_name_byte(source, unit.start_byte, unit.end_byte, &unit.name)
            else {
                continue;
            };
            let (line, character) = position_of_byte(source, name_byte);
            let params = json!({
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
            });
            // The first request may hit the server mid-load; retry until the
            // workspace is ready rather than persisting an empty pass.
            let hover = if first_request {
                first_request = false;
                retry_until_some(|| lsp.request("textDocument/hover", params.clone()), 120)?
            } else {
                lsp.request("textDocument/hover", params.clone()).ok()
            };
            if let Some(text) = hover.as_ref().and_then(hover_text) {
                let content_hash = hex::encode(sha2::Sha256::digest(text.as_bytes()));
                db.upsert_representation(
                    unit.id,
                    &NewRepresentation {
                        kind: RepresentationKind::Custom(TYPED_SIGNATURE_CHANNEL.to_string()),
                        content_hash,
                        content: Some(text),
                        origin: RepresentationOrigin::Derived {
                            producer: provenance.clone(),
                            version: env!("CARGO_PKG_VERSION").to_string(),
                        },
                    },
                )?;
                report.typed_signatures += 1;
            }
        }

        // Call relations. Preferred source: the server's own call hierarchy —
        // fully language-agnostic, no per-language reference queries needed.
        // Fallback for servers without it: resolve the tree-sitter frontend's
        // raw call sites through textDocument/definition (coverage then
        // depends on the language having a populated references.scm).
        if lsp.supports("callHierarchyProvider") {
            for unit in units {
                let Some(name_byte) =
                    declared_name_byte(source, unit.start_byte, unit.end_byte, &unit.name)
                else {
                    continue;
                };
                let (line, character) = position_of_byte(source, name_byte);
                let prepared = lsp
                    .request(
                        "textDocument/prepareCallHierarchy",
                        json!({
                            "textDocument": {"uri": uri},
                            "position": {"line": line, "character": character},
                        }),
                    )
                    .unwrap_or(Value::Null);
                let Some(items) = prepared.as_array() else {
                    continue;
                };
                for item in items {
                    let outgoing = lsp
                        .request("callHierarchy/outgoingCalls", json!({"item": item}))
                        .unwrap_or(Value::Null);
                    let Some(calls) = outgoing.as_array() else {
                        continue;
                    };
                    for call in calls {
                        let Some(callee) = call.get("to") else {
                            continue;
                        };
                        let symbol = callee
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        // The selection range anchors on the identifier itself
                        // and always lies inside the definition; the full range
                        // may start at leading doc comments outside the span.
                        let target_line = callee
                            .pointer("/selectionRange/start/line")
                            .or_else(|| callee.pointer("/range/start/line"))
                            .and_then(Value::as_u64);
                        let target = callee
                            .get("uri")
                            .and_then(Value::as_str)
                            .zip(target_line)
                            .and_then(|(target_uri, target_line)| {
                                resolve_target(
                                    &root,
                                    target_uri,
                                    target_line as usize,
                                    &files_by_path,
                                    &units_by_file,
                                )
                            });
                        relations.push(RelationRecord {
                            from_entity_id: unit.entity_id.clone(),
                            to_entity_id: target,
                            to_symbol: symbol,
                            kind: "calls".to_string(),
                            resolution: "exact".to_string(),
                            provenance: provenance.clone(),
                        });
                    }
                }
            }
            continue;
        }

        let lines: Vec<&str> = source.lines().collect();
        let entity_of_unit: HashMap<i64, &EntityId> = units
            .iter()
            .map(|unit| (unit.id, &unit.entity_id))
            .collect();
        for (caller_unit_id, callee_symbol, start_line) in db.raw_references_for_file(file.id)? {
            let Some(caller_entity) = entity_of_unit.get(&caller_unit_id) else {
                continue;
            };
            let zero_line = (start_line - 1).max(0) as usize;
            let Some(line_text) = lines.get(zero_line) else {
                continue;
            };
            let needle = callee_symbol.rsplit("::").next().unwrap_or(&callee_symbol);
            let Some(column_byte) = line_text.find(needle) else {
                continue;
            };
            let character: usize = line_text[..column_byte].chars().map(char::len_utf16).sum();
            let definition = lsp
                .request(
                    "textDocument/definition",
                    json!({
                        "textDocument": {"uri": uri},
                        "position": {"line": zero_line, "character": character},
                    }),
                )
                .unwrap_or(Value::Null);
            let target = first_location(&definition).and_then(|(target_uri, target_line)| {
                resolve_target(
                    &root,
                    &target_uri,
                    target_line,
                    &files_by_path,
                    &units_by_file,
                )
            });
            relations.push(RelationRecord {
                from_entity_id: (*caller_entity).clone(),
                to_entity_id: target,
                to_symbol: callee_symbol,
                kind: "calls".to_string(),
                resolution: "exact".to_string(),
                provenance: provenance.clone(),
            });
        }
    }

    report.relations = db.replace_relations(generation, &provenance, &relations)?;
    lsp.shutdown()?;
    Ok(report)
}

/// First `(uri, line)` of a `Location | Location[] | LocationLink[]` result.
fn first_location(definition: &Value) -> Option<(String, usize)> {
    let location = match definition {
        Value::Array(items) => items.first()?,
        other => other,
    };
    if let Some(uri) = location.get("uri").and_then(Value::as_str) {
        let line = location.pointer("/range/start/line")?.as_u64()? as usize;
        return Some((uri.to_string(), line));
    }
    // LocationLink form.
    let uri = location.get("targetUri").and_then(Value::as_str)?;
    let line = location.pointer("/targetRange/start/line")?.as_u64()? as usize;
    Some((uri.to_string(), line))
}

/// Map a definition target onto the indexed entity containing it, when the
/// target lies inside the indexed project.
fn resolve_target(
    root: &Path,
    target_uri: &str,
    target_line: usize,
    files_by_path: &HashMap<String, &FileRecord>,
    units_by_file: &HashMap<i64, Vec<codeindex_sqlite::CodeUnit>>,
) -> Option<EntityId> {
    let target_path = uri_to_path(target_uri)?;
    let relative = target_path.strip_prefix(root).ok()?;
    let file = files_by_path.get(&relative.to_string_lossy().replace('\\', "/"))?;
    let one_based = target_line + 1;
    units_by_file[&file.id]
        .iter()
        .filter(|unit| unit.start_line <= one_based && one_based <= unit.end_line)
        .min_by_key(|unit| unit.end_line - unit.start_line)
        .map(|unit| unit.entity_id.clone())
}

/// Extract readable text from a hover result: the first fenced code block
/// when present (the signature), otherwise the plain contents, capped.
fn hover_text(hover: &Value) -> Option<String> {
    let contents = hover.get("contents")?;
    let raw = match contents {
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(marked_string)
            .collect::<Vec<_>>()
            .join("\n"),
        object => object.get("value")?.as_str()?.to_string(),
    };
    // Keep every fenced code block (rust-analyzer emits the module path and
    // the signature as separate blocks); fall back to the plain text.
    let blocks: Vec<String> = raw
        .split("```")
        .enumerate()
        .filter(|(index, _)| index % 2 == 1)
        .map(|(_, block)| {
            block
                .split_once('\n')
                .map(|(_, body)| body.trim().to_string())
                .unwrap_or_else(|| block.trim().to_string())
        })
        .filter(|block| !block.is_empty())
        .collect();
    let text = if blocks.is_empty() {
        raw.trim().to_string()
    } else {
        blocks.join("\n")
    };
    (!text.is_empty()).then(|| text.chars().take(2000).collect())
}

fn marked_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        object => Some(object.get("value")?.as_str()?.to_string()),
    }
}

/// Byte offset of the declared name inside a unit's span, word-bounded.
fn declared_name_byte(source: &str, start: usize, end: usize, name: &str) -> Option<usize> {
    if name.is_empty() || name == "<anonymous>" {
        return None;
    }
    let span = source.get(start..end)?;
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    let mut from = 0;
    while let Some(pos) = span[from..].find(name) {
        let at = from + pos;
        let boundary_before = span[..at].chars().next_back().is_none_or(|c| !is_ident(c));
        let boundary_after = span[at + name.len()..]
            .chars()
            .next()
            .is_none_or(|c| !is_ident(c));
        if boundary_before && boundary_after {
            return Some(start + at);
        }
        from = at + name.len();
    }
    None
}

/// LSP position (0-based line, UTF-16 column) of a byte offset.
fn position_of_byte(source: &str, byte: usize) -> (u32, u32) {
    let mut line = 0_u32;
    let mut column = 0_u32;
    for (index, character) in source.char_indices() {
        if index >= byte {
            break;
        }
        if character == '\n' {
            line += 1;
            column = 0;
        } else {
            column += character.len_utf16() as u32;
        }
    }
    (line, column)
}

/// Retry a request until it yields a non-null result, for the first request
/// against a server that is still loading the workspace.
fn retry_until_some(
    mut attempt: impl FnMut() -> Result<Value>,
    max_tries: usize,
) -> Result<Option<Value>> {
    for tries in 0..max_tries {
        match attempt() {
            Ok(Value::Null) => {}
            Ok(value) => return Ok(Some(value)),
            Err(_) if tries + 1 < max_tries => {}
            Err(error) => return Err(error),
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_are_utf16_aware() {
        let source = "let π = 1;\nfn beta() {}\n";
        let byte = source.find("beta").unwrap();
        assert_eq!(position_of_byte(source, byte), (1, 3));
        // π is one UTF-16 unit but two UTF-8 bytes.
        let byte = source.find('=').unwrap();
        assert_eq!(position_of_byte(source, byte), (0, 6));
    }

    #[test]
    fn hover_text_prefers_the_code_block() {
        let hover = serde_json::json!({
            "contents": {"kind": "markdown", "value": "intro\n```rust\nfn beta() -> u32\n```\nmore"}
        });
        assert_eq!(hover_text(&hover).as_deref(), Some("fn beta() -> u32"));
        let plain = serde_json::json!({"contents": "just text"});
        assert_eq!(hover_text(&plain).as_deref(), Some("just text"));
    }

    #[test]
    fn declared_names_respect_word_boundaries() {
        let source = "fn better() {}\nfn beta() {}\n";
        let at = declared_name_byte(source, 0, source.len(), "beta").unwrap();
        assert_eq!(&source[at..at + 4], "beta");
        assert_eq!(position_of_byte(source, at).0, 1);
    }
}
