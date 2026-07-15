#![forbid(unsafe_code)]

mod embed;
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
pub use run::{
    CancellationToken, DocumentCheckpointStep, IndexOutcome, IndexPausedError, IndexProgress,
    IndexRunBuilder, IndexRunFailure, RefreshMode, RefreshPolicy, ResumePolicy, RetryBackoff,
    RetryPolicy, RevisionTrust,
};
pub use scanner::{ScannedFile, scan_files};
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
    pub retention: RetentionMode,
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
