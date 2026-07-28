#![forbid(unsafe_code)]

mod embed;
mod language;
mod run;
mod scanner;
pub mod source;
mod stage;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use anyhow::Result;
use codeindex_core::{
    EntityId, EntityVersionId, ExtractedEntity, Representation, RepresentationKind,
    RepresentationOrigin,
};
use codeindex_sqlite::{CodeUnit, Db, NewCodeUnit};
use codeindex_tree_sitter::normalizer::{normalize_for_hash, sha256_hex};

pub use codeindex_sqlite::index_publish::{IndexReport, ProjectIndexReport, PublishStep};
pub use codeindex_sqlite::index_runs::{IndexRunPhase, IndexRunState, IndexRunStatus};
pub use embed::{
    EmbedProgress, EmbedStats, LanguageTokens, embed_pending, embed_pending_with_progress,
    embed_space_pending, embed_space_pending_with_progress, find_or_create_model_id, token_report,
};
pub use language::{all_language_ids, is_prose, language_for_extension};
pub use run::{
    CancellationToken, DocumentCheckpointStep, IndexOutcome, IndexPausedError, IndexProgress,
    IndexRunBuilder, IndexRunFailure, RefreshMode, RefreshPolicy, ResumePolicy, RetryBackoff,
    RetryPolicy, RevisionTrust,
};
pub use scanner::{ScanOptions, ScannedFile, scan_files};
use serde::{Deserialize, Serialize};
pub use source::{
    FileSystemSource, MemorySource, RevisionSemantics, SourceDocument, SourceProject,
    SourceProvider, SourceProviderCatalog, SourceRevision, StableRead,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetentionMode {
    Full,
    Report,
    Minimal,
}

impl RetentionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Report => "report",
            Self::Minimal => "minimal",
        }
    }
}

/// Filesystem convenience configuration retained for the common case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectSpec {
    pub label: String,
    pub source_dir: PathBuf,
    pub exclude: Vec<String>,
}

/// Provider-independent indexing settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexSettings {
    pub enabled_languages: Vec<String>,
    pub body_node_count_threshold: usize,
    pub max_body_chars: usize,
    /// Bodies above this emit fragment entities (docs/oversized-units-plan.md).
    pub fragment_threshold_chars: usize,
    pub fragment_target_chars: usize,
    /// Chunk target for prose documents, independent of the code fragment
    /// target above (docs/prose-chunking-plan.md). Prose is chunked for
    /// embedding-window fit, code for AST-boundary fit; they are not the same
    /// question and sharing one knob was a wiring accident, not a decision.
    pub prose_target_chars: usize,
    /// Bytes of the previous prose chunk repeated at the head of the next.
    /// Applies to unstructured text only; Markdown sections do not overlap.
    pub prose_overlap_chars: usize,
    pub retention: RetentionMode,
}

impl IndexSettings {
    pub(crate) fn prose_options(&self) -> codeindex_prose::ProseOptions {
        codeindex_prose::ProseOptions::with_target(
            self.prose_target_chars,
            self.prose_overlap_chars,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexOptions {
    pub projects: Vec<ProjectSpec>,
    pub enabled_languages: Vec<String>,
    pub body_node_count_threshold: usize,
    pub max_body_chars: usize,
    pub retention: RetentionMode,
}

impl IndexOptions {
    pub fn settings(&self) -> IndexSettings {
        IndexSettings {
            enabled_languages: self.enabled_languages.clone(),
            body_node_count_threshold: self.body_node_count_threshold,
            max_body_chars: self.max_body_chars,
            fragment_threshold_chars: codeindex_tree_sitter::ExtractOptions::default()
                .fragment_threshold_chars,
            fragment_target_chars: codeindex_tree_sitter::ExtractOptions::default()
                .fragment_target_chars,
            prose_target_chars: codeindex_prose::ProseOptions::default()
                .chunking
                .target_chars,
            prose_overlap_chars: codeindex_prose::ProseOptions::default()
                .chunking
                .overlap_chars,
            retention: self.retention,
        }
    }
}

/// Optional producer for consumer-defined representations such as generated
/// descriptions. Producers run after deterministic frontend representations are
/// complete and before retention is applied.
pub trait RepresentationEnricher: Send + Sync {
    /// Stable producer/config identity used to decide whether staged work is
    /// safe to reuse across process invocations.
    fn identity(&self) -> EnricherIdentity;

    fn enrich(
        &self,
        document: &SourceDocument,
        source: &str,
        entity: &ExtractedEntity,
    ) -> Result<Vec<Representation>>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnricherIdentity {
    pub producer: String,
    pub version: String,
    pub config_fingerprint: String,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ProjectStats {
    pub label: String,
    pub indexed: usize,
    pub skipped: usize,
    pub removed: usize,
    pub failed: usize,
    pub units: usize,
    pub total_units: usize,
}

/// Index filesystem projects using the default provider implementation.
pub fn index(
    db: &Db,
    options: &IndexOptions,
    only_label: Option<&str>,
) -> Result<Vec<ProjectStats>> {
    let sources: Vec<(String, FileSystemSource)> = options
        .projects
        .iter()
        .map(|project| {
            (
                project.label.clone(),
                FileSystemSource::new(project.source_dir.clone())
                    .with_excludes(project.exclude.clone()),
            )
        })
        .collect();
    let projects: Vec<SourceProject<'_>> = sources
        .iter()
        .map(|(label, provider)| SourceProject {
            label: label.clone(),
            provider,
        })
        .collect();
    index_sources(db, &options.settings(), &projects, only_label)
}

/// Index arbitrary source providers with no representation enrichers.
pub fn index_sources(
    db: &Db,
    settings: &IndexSettings,
    projects: &[SourceProject<'_>],
    only_label: Option<&str>,
) -> Result<Vec<ProjectStats>> {
    index_sources_with_enrichers(db, settings, projects, only_label, &[])
}

/// Provider-neutral indexing entry point.
pub fn index_sources_with_enrichers(
    db: &Db,
    settings: &IndexSettings,
    projects: &[SourceProject<'_>],
    only_label: Option<&str>,
    enrichers: &[&dyn RepresentationEnricher],
) -> Result<Vec<ProjectStats>> {
    let selected: Vec<SourceProject<'_>> = projects
        .iter()
        .filter(|project| only_label.is_none_or(|label| project.label == label))
        .map(|project| SourceProject {
            label: project.label.clone(),
            provider: project.provider,
        })
        .collect();
    if let Some(label) = only_label
        && selected.is_empty()
    {
        anyhow::bail!("no configured project labeled {label:?}");
    }
    let outcome = IndexRunBuilder::new(db, settings, &selected)
        .with_enrichers(enrichers)
        .run()?;
    match outcome {
        IndexOutcome::Committed(report) => Ok(report
            .projects
            .into_iter()
            .map(|project| ProjectStats {
                label: project.label,
                indexed: project.indexed,
                skipped: project.skipped,
                removed: project.removed,
                failed: 0,
                units: project.units,
                total_units: project.total_units,
            })
            .collect()),
        IndexOutcome::Paused(status) => Err(IndexPausedError(status).into()),
    }
}

pub(crate) fn complete_representations(source: &str, entity: &mut ExtractedEntity) {
    if let Some(body_span) = entity.body_span {
        let body = source[body_span.start_byte..body_span.end_byte].to_string();
        let body_hash = sha256_hex(&normalize_for_hash(&body));
        upsert_representation(
            &mut entity.representations,
            Representation::new(RepresentationKind::Body, body, body_hash.clone()),
        );
        // Logical rename matching must ignore the declaration name. A body hash
        // is the strongest deterministic frontend signal currently available.
        entity.normalized_body_hash = body_hash;
    }

    let Some(full) = entity
        .representation(&RepresentationKind::FullSource)
        .map(|representation| representation.content.clone())
    else {
        return;
    };
    let body_offset = entity
        .body_span
        .map(|span| span.start_byte.saturating_sub(entity.span.start_byte))
        .unwrap_or(full.len())
        .min(full.len());
    let search_prefix = &full[..body_offset];
    if entity.name != "<anonymous>"
        && let Some(offset) = find_declared_name(search_prefix, &entity.name)
    {
        let mut without_name = String::with_capacity(full.len() + 8);
        without_name.push_str(&full[..offset]);
        without_name.push_str("<declared-name>");
        without_name.push_str(&full[offset + entity.name.len()..]);
        let hash = sha256_hex(&normalize_for_hash(&without_name));
        upsert_representation(
            &mut entity.representations,
            Representation::new(
                RepresentationKind::BodyWithoutDeclaredName,
                without_name,
                hash,
            ),
        );
    }
}

/// First occurrence of `name` in `text` bounded by non-identifier characters,
/// so a declared name that is also a substring of another identifier
/// (`add` inside a Go receiver type `adder`) never matches.
fn find_declared_name(text: &str, name: &str) -> Option<usize> {
    if name.is_empty() {
        return None;
    }
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    let mut from = 0;
    while let Some(pos) = text[from..].find(name) {
        let start = from + pos;
        let end = start + name.len();
        let before_ok = text[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !is_ident(c));
        let after_ok = text[end..].chars().next().is_none_or(|c| !is_ident(c));
        if before_ok && after_ok {
            return Some(start);
        }
        from = end;
    }
    None
}

pub(crate) fn apply_enrichers(
    document: &SourceDocument,
    source: &str,
    entity: &mut ExtractedEntity,
    enrichers: &[&dyn RepresentationEnricher],
) -> Result<()> {
    for enricher in enrichers {
        for representation in enricher.enrich(document, source, entity)? {
            upsert_representation(&mut entity.representations, representation);
        }
    }
    Ok(())
}

fn upsert_representation(representations: &mut Vec<Representation>, replacement: Representation) {
    if let Some(existing) = representations
        .iter_mut()
        .find(|representation| representation.kind == replacement.kind)
    {
        *existing = replacement;
    } else {
        representations.push(replacement);
    }
}

/// Assign logical identity by exact symbol first, then by a unique identical
/// body. Ambiguous duplicate bodies deliberately mint new ids rather than
/// guessing which historical entity survived.
pub(crate) fn assign_identity(
    project_label: &str,
    source_document_id: &str,
    prior: &[CodeUnit],
    entities: Vec<ExtractedEntity>,
    generation: i64,
) -> Vec<NewCodeUnit> {
    let mut by_key: HashMap<(&str, Option<&str>, &str), &EntityId> = HashMap::new();
    let mut by_body: HashMap<(&str, &str), Vec<&EntityId>> = HashMap::new();
    for unit in prior {
        by_key.insert(
            (
                unit.kind.as_str(),
                unit.scope.as_deref(),
                unit.name.as_str(),
            ),
            &unit.entity_id,
        );
        by_body
            .entry((unit.kind.as_str(), unit.normalized_body_hash.as_str()))
            .or_default()
            .push(&unit.entity_id);
    }

    let mut consumed: HashSet<EntityId> = HashSet::new();
    let mut out = Vec::with_capacity(entities.len());
    for entity in entities {
        let kind = entity.kind.as_str();
        let key = (kind, entity.scope.as_deref(), entity.name.as_str());
        let matched = by_key
            .get(&key)
            .copied()
            .filter(|id| !consumed.contains(*id))
            .or_else(|| {
                let candidates = by_body.get(&(kind, entity.normalized_body_hash.as_str()))?;
                let mut available = candidates
                    .iter()
                    .copied()
                    .filter(|id| !consumed.contains(*id));
                let first = available.next()?;
                available.next().is_none().then_some(first)
            });
        let entity_id = match matched {
            Some(id) => {
                consumed.insert(id.clone());
                id.clone()
            }
            None => mint_entity_id(project_label, source_document_id, &entity),
        };
        let version_ingredients = [
            entity_id.as_str(),
            entity.source_hash.as_str(),
            &entity.span.start_byte.to_string(),
            &entity.span.end_byte.to_string(),
        ]
        .join("\0");
        let entity_version_id =
            EntityVersionId::new(format!("ver:{}", &sha256_hex(&version_ingredients)[..16]));
        out.push(NewCodeUnit::from_entity(
            entity,
            entity_id,
            entity_version_id,
            generation,
        ));
    }
    out
}

fn mint_entity_id(
    project_label: &str,
    source_document_id: &str,
    entity: &ExtractedEntity,
) -> EntityId {
    let ingredients = [
        project_label,
        source_document_id,
        entity.kind.as_str(),
        entity.scope.as_deref().unwrap_or(""),
        entity.name.as_str(),
        &entity.span.start_byte.to_string(),
    ]
    .join("\0");
    EntityId::new(format!("ent:{}", &sha256_hex(&ingredients)[..16]))
}

pub(crate) fn apply_retention(units: &mut [NewCodeUnit], retention: RetentionMode) {
    for unit in units {
        for representation in &mut unit.representations {
            let extracted = matches!(
                &representation.origin,
                RepresentationOrigin::Extracted { .. }
            );
            match retention {
                RetentionMode::Full => {}
                RetentionMode::Report => {
                    if extracted && representation.kind != RepresentationKind::FullSource {
                        representation.content = None;
                    }
                }
                RetentionMode::Minimal => {
                    // Derived/imported channels cannot be re-created from one
                    // source document, so retain them. Extracted channels are
                    // recoverable through the provider catalog.
                    if extracted {
                        representation.content = None;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> IndexSettings {
        IndexSettings {
            enabled_languages: vec!["rust".into()],
            body_node_count_threshold: 1,
            max_body_chars: 10_000,
            fragment_threshold_chars: 6_000,
            fragment_target_chars: 3_000,
            prose_target_chars: 1_200,
            prose_overlap_chars: 150,
            retention: RetentionMode::Full,
        }
    }

    #[test]
    fn declared_name_erasure_respects_word_boundaries() {
        // `add` inside the receiver type `adder` must not match; the real
        // declared name after the receiver must.
        assert_eq!(find_declared_name("func (a *adder) add(", "add"), Some(16));
        assert_eq!(find_declared_name("fn addr(", "add"), None);
        assert_eq!(find_declared_name("fn add(", "add"), Some(3));
        assert_eq!(find_declared_name("", "add"), None);
    }

    #[test]
    fn indexes_memory_source_without_filesystem_assumptions() {
        let mut source = MemorySource::new("memory://workspace");
        source.insert(
            "src/lib.rs",
            "fn add(a: i32, b: i32) -> i32 { let value = a + b; value }",
        );
        let db = codeindex_sqlite::open_in_memory().unwrap();
        let projects = [SourceProject {
            label: "main".into(),
            provider: &source,
        }];
        let stats = index_sources(&db, &settings(), &projects, None).unwrap();
        assert_eq!(stats[0].indexed, 1);
        assert_eq!(stats[0].total_units, 1);
        assert_eq!(
            db.list_files(db.get_project("main").unwrap().unwrap().id)
                .unwrap()[0]
                .source_document_id,
            "src/lib.rs"
        );
    }

    #[test]
    fn rename_preserves_entity_identity_through_body_channel() {
        let mut source = MemorySource::new("memory://rename");
        source.insert(
            "lib.rs",
            "fn alpha(a: i32, b: i32) -> i32 { let sum = a + b; sum * 2 }",
        );
        let db = codeindex_sqlite::open_in_memory().unwrap();
        let projects = [SourceProject {
            label: "main".into(),
            provider: &source,
        }];
        index_sources(&db, &settings(), &projects, None).unwrap();
        let project_id = db.get_project("main").unwrap().unwrap().id;
        let before = db.list_units_for_project(project_id).unwrap()[0]
            .entity_id
            .clone();

        source.insert(
            "lib.rs",
            "fn beta(a: i32, b: i32) -> i32 { let sum = a + b; sum * 2 }",
        );
        let projects = [SourceProject {
            label: "main".into(),
            provider: &source,
        }];
        index_sources(&db, &settings(), &projects, None).unwrap();
        let after = db.list_units_for_project(project_id).unwrap()[0]
            .entity_id
            .clone();
        assert_eq!(before, after);
    }

    #[test]
    fn prose_chunk_settings_are_fixed_once_published() {
        // Chunk boundaries decide prose content hashes, and the embedding-time
        // recovery path re-chunks from the persisted values. Letting the target
        // change on a published corpus would silently orphan every stored hash,
        // so it must be rejected the way fragment_target_chars is.
        let mut source = MemorySource::new("memory://prose-settings");
        source.insert("note.md", "# Note\n\nSome prose body text here.\n");
        let db = codeindex_sqlite::open_in_memory().unwrap();
        let projects = [SourceProject {
            label: "main".into(),
            provider: &source,
        }];

        let mut first = settings();
        first.enabled_languages = vec!["markdown".into()];
        index_sources(&db, &first, &projects, None).unwrap();
        assert_eq!(
            db.get_setting("index.prose_target_chars")
                .unwrap()
                .as_deref(),
            Some("1200")
        );

        let mut drifted = first.clone();
        drifted.prose_target_chars = 900;
        let error = index_sources(&db, &drifted, &projects, None).unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains("index.prose_target_chars"),
            "expected the immutable-settings guard to name the key, got: {message}"
        );
    }

    #[test]
    fn prose_documents_index_through_the_real_pipeline() {
        // End-to-end proof that a grammar-free format reaches published units:
        // scanner resolves the extension, stage dispatches to the prose
        // frontend, and publication stores sections with their heading scope.
        let mut source = MemorySource::new("memory://prose-pipeline");
        source.insert(
            "note.md",
            "# Top\n\nlead paragraph\n\n## Findings\n\nthe finding body\n",
        );
        source.insert("scan.txt", "plain OCR-ish text body\n");
        let db = codeindex_sqlite::open_in_memory().unwrap();
        let projects = [SourceProject {
            label: "main".into(),
            provider: &source,
        }];
        let mut prose_settings = settings();
        prose_settings.enabled_languages = vec!["markdown".into(), "text".into()];
        let stats = index_sources(&db, &prose_settings, &projects, None).unwrap();
        assert_eq!(stats[0].indexed, 2, "both prose documents should index");
        assert!(stats[0].units >= 3, "expected sections plus a text chunk");
    }

    #[test]
    fn frontend_version_drift_refuses_reindex_instead_of_silent_reuse() {
        // Unchanged file content alone never triggers re-extraction
        // (`action_for` only looks at content hash/mtime/size), so an
        // extractor code change must be enforced the same way other
        // extraction-affecting settings are: fixed once published, loudly
        // rejected on drift. Simulates the drift by overwriting the
        // already-published `index.frontend_version` row with a stale value,
        // then reindexing identical content with the real (current) version.
        let mut source = MemorySource::new("memory://frontend-version");
        source.insert(
            "lib.rs",
            "fn add(a: i32, b: i32) -> i32 { let s = a + b; s }",
        );
        let db = codeindex_sqlite::open_in_memory().unwrap();
        let projects = [SourceProject {
            label: "main".into(),
            provider: &source,
        }];
        index_sources(&db, &settings(), &projects, None).unwrap();
        assert_eq!(
            db.get_setting("index.frontend_version").unwrap().as_deref(),
            Some(codeindex_tree_sitter::FRONTEND_VERSION)
        );

        db.set_setting("index.frontend_version", "0.0.0:stale")
            .unwrap();
        let error = index_sources(&db, &settings(), &projects, None).unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains("index.frontend_version"),
            "error must name the drifted setting: {message}"
        );
        assert!(
            message.contains("Delete the database file"),
            "error must point at the documented recovery path: {message}"
        );
    }

    #[test]
    fn deleting_a_document_removes_its_units_on_reindex() {
        let mut source = MemorySource::new("memory://delete");
        source.insert("a.rs", "fn alpha(a: i32) -> i32 { let x = a + 1; x * 2 }");
        source.insert("b.rs", "fn beta(b: i32) -> i32 { let y = b - 1; y * 3 }");
        let db = codeindex_sqlite::open_in_memory().unwrap();
        let projects = [SourceProject {
            label: "main".into(),
            provider: &source,
        }];
        let stats = index_sources(&db, &settings(), &projects, None).unwrap();
        assert_eq!(stats[0].total_units, 2);

        source.remove("b.rs");
        let projects = [SourceProject {
            label: "main".into(),
            provider: &source,
        }];
        let stats = index_sources(&db, &settings(), &projects, None).unwrap();
        assert_eq!(stats[0].removed, 1, "the deleted document must be reported");
        assert_eq!(
            stats[0].total_units, 1,
            "the deleted document's unit must go"
        );
        let project_id = db.get_project("main").unwrap().unwrap().id;
        assert_eq!(db.list_files(project_id).unwrap().len(), 1);
    }

    #[test]
    fn lexical_index_tracks_publication_and_deletion() {
        let mut source = MemorySource::new("memory://lexical");
        source.insert(
            "flags.rs",
            "fn parse_flags(input: &str) -> usize { let total = input.len(); total }",
        );
        source.insert(
            "sum.rs",
            "fn checksum(bytes: &[u8]) -> u32 { let mut acc = 0u32; acc += bytes.len() as u32; acc }",
        );
        let db = codeindex_sqlite::open_in_memory().unwrap();
        let projects = [SourceProject {
            label: "main".into(),
            provider: &source,
        }];
        index_sources(&db, &settings(), &projects, None).unwrap();

        let hits = db.lexical_search("parse_flags input", 10).unwrap();
        assert!(!hits.is_empty());
        let snapshot = db.snapshot(&[]).unwrap();
        let top = snapshot
            .units
            .iter()
            .find(|unit| unit.entity_version_id == hits[0].entity_version_id)
            .unwrap();
        assert_eq!(top.name, "parse_flags");

        // Code fragments with punctuation are safe FTS input.
        assert!(
            !db.lexical_search("acc += bytes.len() as u32", 10)
                .unwrap()
                .is_empty()
        );

        // A token deep inside an oversized body still matches: lexical
        // indexes the FULL text even though the embedding representation
        // truncates at max_body_chars (docs/oversized-units-plan.md).
        let deep_body: String = (0..1500)
            .map(|i| format!("    let filler_{i} = {i};\n"))
            .chain(std::iter::once(
                "    let zanzibar_sentinel = 42;\n".to_string(),
            ))
            .collect();
        source.insert(
            "huge.rs",
            format!("fn dispatch_monster(input: u32) -> u32 {{\n{deep_body}    input\n}}\n"),
        );
        let projects = [SourceProject {
            label: "main".into(),
            provider: &source,
        }];
        index_sources(&db, &settings(), &projects, None).unwrap();
        let hits = db.lexical_search("zanzibar_sentinel", 10).unwrap();
        assert!(
            !hits.is_empty(),
            "token past the embedding cap must match lexically"
        );

        // Deleting the document removes its lexical rows.
        source.remove("flags.rs");
        let projects = [SourceProject {
            label: "main".into(),
            provider: &source,
        }];
        index_sources(&db, &settings(), &projects, None).unwrap();
        let hits = db.lexical_search("parse_flags", 10).unwrap();
        assert!(
            hits.is_empty(),
            "deleted document must leave no lexical rows"
        );
    }

    #[test]
    fn body_and_name_erased_channels_are_materialized() {
        let mut source = MemorySource::new("memory://representations");
        source.insert(
            "lib.rs",
            "fn parse_flags(input: &str) -> usize { input.len() }",
        );
        let db = codeindex_sqlite::open_in_memory().unwrap();
        let projects = [SourceProject {
            label: "main".into(),
            provider: &source,
        }];
        index_sources(&db, &settings(), &projects, None).unwrap();
        let snapshot = db.snapshot(&[]).unwrap();
        let unit = &snapshot.units[0];
        assert!(unit.representation(&RepresentationKind::Body).is_some());
        assert!(
            unit.representation(&RepresentationKind::BodyWithoutDeclaredName)
                .unwrap()
                .content
                .as_deref()
                .unwrap()
                .contains("<declared-name>")
        );
    }
}
