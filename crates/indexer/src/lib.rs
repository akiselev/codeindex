#![forbid(unsafe_code)]

mod embed;
mod scanner;
pub mod source;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use codeindex_core::{
    EntityId, EntityVersionId, ExtractedEntity, Representation, RepresentationKind,
    RepresentationOrigin,
};
use codeindex_source::{
    DocumentDescriptor, DocumentQuery, LanguageHint, RevisionGuarantee, SnapshotRequest,
};
use codeindex_sqlite::{CodeUnit, Db, NewCodeUnit, NewFile, NewReference, ProjectId};
use codeindex_tree_sitter::normalizer::{normalize_for_hash, sha256_hex};
use codeindex_tree_sitter::{ExtractOptions, LanguageRegistry, extract_references, extract_units};

pub use embed::{
    EmbedProgress, EmbedStats, LanguageTokens, embed_pending, embed_pending_with_progress,
    embed_space_pending, embed_space_pending_with_progress, find_or_create_model_id, token_report,
};
pub use scanner::{ScannedFile, scan_files};
pub use source::{
    ContentHash, DocumentIter, DocumentLocation, DocumentMetadata, DocumentVersion,
    FileSystemSource, FilesystemWorkspace, FilesystemWorkspaceBuilder, MemorySource,
    MemoryWorkspace, OverlayWorkspace, SnapshotConsistency, SnapshotId, SourceCapabilities,
    SourceCheckpoint, SourceContent, SourceDocument, SourceError, SourceKind, SourceProject,
    SourceProvider, SourceProviderCatalog, SourceRevision, SourceRootId, WorkspaceDescriptor,
    WorkspaceId, validate_snapshot,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevisionVerification {
    /// Trust equal provider revision tokens even when they are metadata hints.
    Fast,
    /// Read and hash documents whose provider cannot guarantee content identity.
    Verified,
}

/// Filesystem convenience configuration retained for the common case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectSpec {
    pub label: String,
    pub source_dir: PathBuf,
    pub exclude: Vec<String>,
}

/// Workspace-independent indexing settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexSettings {
    pub enabled_languages: Vec<String>,
    pub body_node_count_threshold: usize,
    pub max_body_chars: usize,
    pub retention: RetentionMode,
    pub revision_verification: RevisionVerification,
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
            revision_verification: RevisionVerification::Fast,
        }
    }
}

/// Optional producer for consumer-defined representations such as generated
/// descriptions. Producers run after deterministic frontend representations are
/// complete and before retention is applied.
pub trait RepresentationEnricher: Send + Sync {
    fn enrich(
        &self,
        document: &DocumentDescriptor,
        source: &str,
        entity: &ExtractedEntity,
    ) -> Result<Vec<Representation>>;
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

/// Index filesystem projects using the default workspace implementation.
pub fn index(
    db: &Db,
    options: &IndexOptions,
    only_label: Option<&str>,
) -> Result<Vec<ProjectStats>> {
    let workspaces: Vec<(String, FileSystemSource)> = options
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
    let projects: Vec<SourceProject<'_>> = workspaces
        .iter()
        .map(|(label, workspace)| SourceProject {
            label: label.clone(),
            workspace,
        })
        .collect();
    index_sources(db, &options.settings(), &projects, only_label)
}

/// Index arbitrary source workspaces with no representation enrichers.
pub fn index_sources(
    db: &Db,
    settings: &IndexSettings,
    projects: &[SourceProject<'_>],
    only_label: Option<&str>,
) -> Result<Vec<ProjectStats>> {
    index_sources_with_enrichers(db, settings, projects, only_label, &[])
}

/// Workspace-neutral indexing entry point.
pub fn index_sources_with_enrichers(
    db: &Db,
    settings: &IndexSettings,
    projects: &[SourceProject<'_>],
    only_label: Option<&str>,
    enrichers: &[&dyn RepresentationEnricher],
) -> Result<Vec<ProjectStats>> {
    db.check_or_set_immutable(
        "index.body_node_count_threshold",
        &settings.body_node_count_threshold.to_string(),
    )?;
    db.check_or_set_immutable("index.retention", settings.retention.as_str())?;
    db.check_or_set_immutable(
        "embedding.max_body_chars",
        &settings.max_body_chars.to_string(),
    )?;

    let generation = db.bump_generation()?;
    let mut stats = Vec::new();
    for project in projects {
        if only_label.is_some_and(|label| project.label != label) {
            continue;
        }
        stats.push(index_project(db, settings, project, generation, enrichers)?);
    }
    if let Some(label) = only_label
        && stats.is_empty()
    {
        anyhow::bail!("no configured project labeled {label:?}");
    }
    db.prune_orphan_entities()?;
    db.prune_orphan_embeddings()?;
    Ok(stats)
}

fn index_project(
    db: &Db,
    settings: &IndexSettings,
    project: &SourceProject<'_>,
    generation: i64,
    enrichers: &[&dyn RepresentationEnricher],
) -> Result<ProjectStats> {
    let workspace = project.workspace;
    let descriptor = workspace.descriptor();
    let project_id = db.upsert_project(&project.label, &descriptor.persisted_locator())?;
    let enabled: HashSet<String> = settings.enabled_languages.iter().cloned().collect();
    let mut query = DocumentQuery::all();
    query.language_ids = enabled.iter().cloned().collect();
    let snapshot = workspace
        .open_snapshot(&SnapshotRequest::default())
        .with_context(|| format!("failed to open source snapshot for {}", project.label))?;
    let extraction = ExtractOptions {
        body_node_count_threshold: settings.body_node_count_threshold,
        max_body_chars: settings.max_body_chars,
    };
    let registry = LanguageRegistry::global();

    let mut stats = ProjectStats {
        label: project.label.clone(),
        ..ProjectStats::default()
    };
    let mut document_ids = HashSet::new();
    let mut seen = HashSet::new();
    let mut paths = HashSet::new();
    let mut any_change = false;

    for document in snapshot.documents(&query)? {
        let document = match document {
            Ok(document) => document,
            Err(error) => {
                eprintln!(
                    "failed to enumerate source in project {}: {error}",
                    project.label
                );
                stats.failed += 1;
                continue;
            }
        };
        if !document_ids.insert(document.id.to_string()) {
            anyhow::bail!(
                "source workspace {} returned duplicate document id {}",
                descriptor.id,
                document.id
            );
        }
        if !paths.insert(document.location.logical_path.clone()) {
            anyhow::bail!(
                "source workspace {} returned duplicate logical path {:?}",
                descriptor.id,
                document.location.logical_path
            );
        }
        let Some(language_id) = resolve_language(&document, &enabled, registry) else {
            continue;
        };
        seen.insert(document.id.to_string());
        let source_document_id = document.id.to_string();
        let relative_path = document.location.logical_path.clone();
        let existing = db.get_file_by_source_id(project_id, &source_document_id)?;
        let mtime_ns = modified_ns(document.version.modified_at);
        let size = document.version.size.unwrap_or_default() as i64;

        let metadata_unchanged = existing.as_ref().is_some_and(|record| {
            record.source_revision == document.version.token.as_str()
                && record.relative_path == relative_path
                && record.language_id == language_id
        });
        let revision_is_trusted = document.version.guarantee == RevisionGuarantee::ContentIdentity
            || settings.revision_verification == RevisionVerification::Fast;
        if metadata_unchanged && revision_is_trusted {
            stats.skipped += 1;
            continue;
        }

        let content = match snapshot.read(&document) {
            Ok(content) => content,
            Err(error) => {
                eprintln!(
                    "failed to read {} from project {} snapshot {}: {error}",
                    relative_path,
                    project.label,
                    snapshot.id()
                );
                stats.failed += 1;
                continue;
            }
        };
        let source = match content.utf8() {
            Ok(source) => source,
            Err(error) => {
                eprintln!(
                    "failed to decode {} from project {}: {error}",
                    relative_path, project.label
                );
                stats.failed += 1;
                continue;
            }
        };
        let source_hash = sha256_hex(source);
        if let Some(record) = &existing
            && record.source_hash == source_hash
            && record.relative_path == relative_path
            && record.language_id == language_id
        {
            db.update_file_meta(
                record.id,
                document.version.token.as_str(),
                mtime_ns,
                content.observed_version.size.unwrap_or_default() as i64,
            )?;
            stats.skipped += 1;
            continue;
        }

        let def = registry
            .get(&language_id)
            .with_context(|| format!("unknown language {language_id}"))?;
        let mut entities = match extract_units(def, source, &extraction) {
            Ok(entities) => entities,
            Err(error) => {
                eprintln!("failed to parse {relative_path}: {error}");
                stats.failed += 1;
                continue;
            }
        };
        for entity in &mut entities {
            complete_representations(source, entity);
            apply_enrichers(&document, source, entity, enrichers)?;
        }
        let references = extract_references(def, source).unwrap_or_else(|error| {
            eprintln!("failed to extract references from {relative_path}: {error}");
            Vec::new()
        });

        let prior = match &existing {
            Some(record) => db.list_units_for_file(record.id)?,
            None => Vec::new(),
        };
        let mut units = assign_identity(
            &project.label,
            &source_document_id,
            &prior,
            entities,
            generation,
        );
        apply_retention(&mut units, settings.retention);

        let file_id = db.upsert_file(&NewFile {
            project_id,
            source_document_id,
            source_revision: document.version.token.to_string(),
            relative_path,
            language_id,
            mtime_ns,
            size: content.observed_version.size.unwrap_or(size.max(0) as u64) as i64,
            source_hash,
        })?;
        let unit_ids = db.insert_units(file_id, &units)?;
        stage_references(db, &units, &unit_ids, &references)?;
        stats.indexed += 1;
        stats.units += units.len();
        any_change = true;
    }

    for record in db.list_files(project_id)? {
        if !seen.contains(&record.source_document_id) {
            db.delete_file(record.id)?;
            stats.removed += 1;
            any_change = true;
        }
    }

    if any_change {
        resolve_usage(db, project_id, settings.retention)?;
    }
    stats.total_units = db.count_units_for_project(project_id)? as usize;
    Ok(stats)
}

fn resolve_language(
    document: &DocumentDescriptor,
    enabled: &HashSet<String>,
    registry: &'static LanguageRegistry,
) -> Option<String> {
    let hinted = match &document.language_hint {
        LanguageHint::Known(language) => Some(language.clone()),
        LanguageHint::FileExtension(extension) => registry
            .by_extension(&extension.to_ascii_lowercase())
            .map(|definition| definition.spec.id.clone()),
        LanguageHint::MediaType(media_type) => media_type
            .rsplit_once('/')
            .and_then(|(_, subtype)| registry.by_extension(subtype.trim_start_matches("x-")))
            .map(|definition| definition.spec.id.clone()),
        LanguageHint::Shebang(shebang) => language_from_shebang(shebang, registry),
        LanguageHint::Unknown => None,
    }
    .or_else(|| {
        Path::new(&document.location.logical_path)
            .extension()
            .and_then(|extension| extension.to_str())
            .and_then(|extension| registry.by_extension(&extension.to_ascii_lowercase()))
            .map(|definition| definition.spec.id.clone())
    })?;
    enabled.contains(&hinted).then_some(hinted)
}

fn language_from_shebang(shebang: &str, registry: &'static LanguageRegistry) -> Option<String> {
    let extension = if shebang.contains("python") {
        "py"
    } else if shebang.contains("node") || shebang.contains("deno") {
        "js"
    } else if shebang.contains("ruby") {
        "rb"
    } else {
        return None;
    };
    registry
        .by_extension(extension)
        .map(|definition| definition.spec.id.clone())
}

fn modified_ns(modified_at: Option<SystemTime>) -> i64 {
    modified_at
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos().min(i64::MAX as u128) as i64)
        .unwrap_or_default()
}

fn complete_representations(source: &str, entity: &mut ExtractedEntity) {
    if let Some(body_span) = entity.body_span {
        let body = source[body_span.start_byte..body_span.end_byte].to_string();
        let body_hash = sha256_hex(&normalize_for_hash(&body));
        upsert_representation(
            &mut entity.representations,
            Representation::new(RepresentationKind::Body, body, body_hash.clone()),
        );
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
        && let Some(offset) = search_prefix.find(&entity.name)
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

fn apply_enrichers(
    document: &DocumentDescriptor,
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
fn assign_identity(
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

fn apply_retention(units: &mut [NewCodeUnit], retention: RetentionMode) {
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
                    if extracted {
                        representation.content = None;
                    }
                }
            }
        }
    }
}

fn stage_references(
    db: &Db,
    units: &[NewCodeUnit],
    unit_ids: &[i64],
    references: &[codeindex_tree_sitter::RawReference],
) -> Result<()> {
    if references.is_empty() {
        return Ok(());
    }
    let mut staged = Vec::new();
    for reference in references {
        let mut best: Option<usize> = None;
        for (index, unit) in units.iter().enumerate() {
            if unit.start_byte <= reference.start_byte && reference.start_byte < unit.end_byte {
                match best {
                    Some(previous) if units[previous].start_byte >= unit.start_byte => {}
                    _ => best = Some(index),
                }
            }
        }
        let Some(index) = best else { continue };
        staged.push(NewReference {
            caller_unit_id: unit_ids[index],
            callee_symbol: reference.callee_symbol.clone(),
            call_snippet: reference.call_snippet.clone(),
            start_line: reference.start_line as i64,
        });
    }
    db.insert_references(&staged)
}

fn resolve_usage(db: &Db, project_id: ProjectId, _retention: RetentionMode) -> Result<()> {
    db.clear_channel_for_project(project_id, &RepresentationKind::Usage)?;
    let units = db.list_units_for_project(project_id)?;
    if units.is_empty() {
        return Ok(());
    }
    let mut definitions: HashMap<String, Vec<i64>> = HashMap::new();
    let mut qualified: HashMap<i64, String> = HashMap::new();
    for unit in &units {
        if let Some(name) = symbol_name(&unit.name) {
            definitions.entry(name).or_default().push(unit.id);
        }
        qualified.insert(
            unit.id,
            match &unit.scope {
                Some(scope) => format!("{scope}.{}", unit.name),
                None => unit.name.clone(),
            },
        );
    }

    let references = db.references_for_project(project_id)?;
    let mut usages: BTreeMap<i64, BTreeSet<String>> = BTreeMap::new();
    for (caller_id, _, _, callee_symbol, snippet, _) in references {
        let Some(name) = symbol_name(&callee_symbol) else {
            continue;
        };
        let Some(callee_ids) = definitions.get(&name) else {
            continue;
        };
        let caller = qualified
            .get(&caller_id)
            .cloned()
            .unwrap_or_else(|| "?".to_string());
        for &callee_id in callee_ids {
            if callee_id != caller_id {
                usages
                    .entry(callee_id)
                    .or_default()
                    .insert(format!("{caller}: {snippet}"));
            }
        }
    }

    let origin = RepresentationOrigin::Derived {
        producer: "codeindex-usage".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };
    for (unit_id, lines) in usages {
        let text = lines.into_iter().collect::<Vec<_>>().join("\n");
        db.set_representation_with_origin(
            unit_id,
            &RepresentationKind::Usage,
            &sha256_hex(&text),
            Some(&text),
            &origin,
        )?;
    }
    Ok(())
}

fn symbol_name(raw: &str) -> Option<String> {
    let trimmed = raw.trim().trim_end_matches('!');
    let mut without_generics = String::with_capacity(trimmed.len());
    let mut depth = 0usize;
    for character in trimmed.chars() {
        match character {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            _ if depth == 0 => without_generics.push(character),
            _ => {}
        }
    }
    let segment = without_generics
        .rsplit(['.', ':'])
        .find(|part| !part.trim().is_empty())
        .unwrap_or("")
        .trim();
    (!segment.is_empty() && segment != "<anonymous>").then(|| segment.to_string())
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
            revision_verification: RevisionVerification::Verified,
        }
    }

    #[test]
    fn indexes_memory_workspace_without_filesystem_assumptions() {
        let source = MemorySource::new("memory://workspace");
        source.insert(
            "src/lib.rs",
            "fn add(a: i32, b: i32) -> i32 { let value = a + b; value }",
        );
        let db = codeindex_sqlite::open_in_memory().unwrap();
        let projects = [SourceProject {
            label: "main".into(),
            workspace: &source,
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
        let source = MemorySource::new("memory://rename");
        source.insert(
            "lib.rs",
            "fn alpha(a: i32, b: i32) -> i32 { let sum = a + b; sum * 2 }",
        );
        let db = codeindex_sqlite::open_in_memory().unwrap();
        let projects = [SourceProject {
            label: "main".into(),
            workspace: &source,
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
        index_sources(&db, &settings(), &projects, None).unwrap();
        let after = db.list_units_for_project(project_id).unwrap()[0]
            .entity_id
            .clone();
        assert_eq!(before, after);
    }

    #[test]
    fn body_and_name_erased_channels_are_materialized() {
        let source = MemorySource::new("memory://representations");
        source.insert(
            "lib.rs",
            "fn parse_flags(input: &str) -> usize { input.len() }",
        );
        let db = codeindex_sqlite::open_in_memory().unwrap();
        let projects = [SourceProject {
            label: "main".into(),
            workspace: &source,
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
