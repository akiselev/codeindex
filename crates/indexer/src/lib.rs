#![forbid(unsafe_code)]

mod embed;
mod scanner;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result};
use codeindex_core::{ExtractedEntity, RepresentationKind};
use codeindex_sqlite::{CodeUnit, Db, NewCodeUnit, NewFile, NewReference, ProjectId};
use codeindex_tree_sitter::normalizer::sha256_hex;
use codeindex_tree_sitter::{
    ExtractOptions, LanguageRegistry, extract_references, extract_units,
};

pub use embed::{
    EmbedProgress, EmbedStats, LanguageTokens, embed_pending, embed_pending_with_progress,
    find_or_create_model_id, token_report,
};
pub use scanner::{ScannedFile, scan_files};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectSpec {
    pub label: String,
    pub source_dir: PathBuf,
    pub exclude: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexOptions {
    pub projects: Vec<ProjectSpec>,
    pub enabled_languages: Vec<String>,
    pub body_node_count_threshold: usize,
    pub max_body_chars: usize,
    pub retention: RetentionMode,
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

pub fn index(
    db: &Db,
    options: &IndexOptions,
    only_label: Option<&str>,
) -> Result<Vec<ProjectStats>> {
    db.check_or_set_immutable(
        "index.body_node_count_threshold",
        &options.body_node_count_threshold.to_string(),
    )?;
    db.check_or_set_immutable("index.retention", options.retention.as_str())?;
    db.check_or_set_immutable(
        "embedding.max_body_chars",
        &options.max_body_chars.to_string(),
    )?;

    // Every unit written in this run shares one generation number (M4).
    let generation = db.bump_generation()?;

    let mut stats = Vec::new();
    for project in &options.projects {
        if only_label.is_some_and(|label| project.label != label) {
            continue;
        }
        stats.push(index_project(db, options, project, generation)?);
    }
    if let Some(label) = only_label
        && stats.is_empty()
    {
        anyhow::bail!("no configured project labeled {label:?}");
    }
    db.prune_orphan_entities()?;
    let pruned = db.prune_orphan_embeddings()?;
    if pruned > 0 {
        eprintln!("pruned {pruned} orphaned embeddings");
    }
    Ok(stats)
}

fn index_project(
    db: &Db,
    options: &IndexOptions,
    project: &ProjectSpec,
    generation: i64,
) -> Result<ProjectStats> {
    let root = &project.source_dir;
    let project_id = db.upsert_project(&project.label, &root.to_string_lossy())?;
    let enabled: HashSet<String> = options.enabled_languages.iter().cloned().collect();
    let scanned = scan_files(root, &project.exclude, &enabled)?;
    let extraction = ExtractOptions {
        body_node_count_threshold: options.body_node_count_threshold,
        max_body_chars: options.max_body_chars,
    };
    let registry = LanguageRegistry::global();

    let mut stats = ProjectStats {
        label: project.label.clone(),
        ..ProjectStats::default()
    };
    let mut seen = HashSet::with_capacity(scanned.len());
    let mut any_change = false;
    for file in &scanned {
        seen.insert(file.relative_path.clone());
        let existing = db.get_file(project_id, &file.relative_path)?;
        let metadata = match std::fs::metadata(&file.absolute_path) {
            Ok(metadata) => metadata,
            Err(error) => {
                eprintln!("failed to stat {}: {error}", file.absolute_path.display());
                stats.failed += 1;
                continue;
            }
        };
        let mtime_ns = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos() as i64)
            .unwrap_or(0);
        let size = metadata.len() as i64;

        if existing
            .as_ref()
            .is_some_and(|record| record.mtime_ns == mtime_ns && record.size == size)
        {
            stats.skipped += 1;
            continue;
        }

        let source = match std::fs::read_to_string(&file.absolute_path) {
            Ok(source) => source,
            Err(error) => {
                eprintln!("failed to read {}: {error}", file.absolute_path.display());
                stats.failed += 1;
                continue;
            }
        };
        let source_hash = sha256_hex(&source);
        if let Some(record) = &existing
            && record.source_hash == source_hash
        {
            db.update_file_meta(record.id, mtime_ns, size)?;
            stats.skipped += 1;
            continue;
        }

        let def = registry
            .get(&file.language_id)
            .context("scanner produced an unregistered language")?;
        let entities = match extract_units(def, &source, &extraction) {
            Ok(entities) => entities,
            Err(error) => {
                eprintln!("failed to parse {}: {error}", file.absolute_path.display());
                stats.failed += 1;
                continue;
            }
        };
        let references = extract_references(def, &source).unwrap_or_default();

        // Carry entity identity forward from the file's prior units before we
        // replace them (M4).
        let prior = match &existing {
            Some(record) => db.list_units_for_file(record.id)?,
            None => Vec::new(),
        };
        let mut units = assign_identity(
            &project.label,
            &file.relative_path,
            &prior,
            entities,
            generation,
        );
        apply_retention(&mut units, options.retention);

        let file_id = db.upsert_file(&NewFile {
            project_id,
            relative_path: file.relative_path.clone(),
            language_id: file.language_id.clone(),
            mtime_ns,
            size,
            source_hash,
        })?;
        let unit_ids = db.insert_units(file_id, &units)?;
        stage_references(db, &units, &unit_ids, &references)?;
        stats.indexed += 1;
        stats.units += units.len();
        any_change = true;
    }

    for record in db.list_files(project_id)? {
        if !seen.contains(&record.relative_path) {
            db.delete_file(record.id)?;
            stats.removed += 1;
            any_change = true;
        }
    }

    // Usage channel: resolve staged call sites across the whole project into a
    // synthesized per-entity document (M4). Recomputed whenever anything in the
    // project changed, since a caller edit changes a callee's usage.
    if any_change {
        resolve_usage(db, project_id)?;
    }

    stats.total_units = db.count_units_for_project(project_id)? as usize;
    Ok(stats)
}

/// Assign each extracted entity a logical `entity_id` (stable across index
/// generations) and an exact `entity_version_id`. Matching is within-file:
/// first by `(kind, scope, name)`, then — to survive a rename — by identical
/// `Implementation` body hash and kind. Unmatched entities mint a fresh id.
fn assign_identity(
    project_label: &str,
    relative_path: &str,
    prior: &[CodeUnit],
    entities: Vec<ExtractedEntity>,
    generation: i64,
) -> Vec<NewCodeUnit> {
    let mut by_key: HashMap<(&str, Option<&str>, &str), &str> = HashMap::new();
    let mut by_body: HashMap<(&str, &str), &str> = HashMap::new();
    for unit in prior {
        by_key.insert(
            (unit.kind.as_str(), unit.scope.as_deref(), unit.name.as_str()),
            unit.entity_id.as_str(),
        );
        by_body
            .entry((unit.kind.as_str(), unit.normalized_body_hash.as_str()))
            .or_insert(unit.entity_id.as_str());
    }

    let mut consumed: HashSet<&str> = HashSet::new();
    let mut out = Vec::with_capacity(entities.len());
    for entity in entities {
        let kind = entity.kind.as_str();
        let key = (kind, entity.scope.as_deref(), entity.name.as_str());
        let matched = by_key
            .get(&key)
            .copied()
            .filter(|id| !consumed.contains(id))
            .or_else(|| {
                by_body
                    .get(&(kind, entity.normalized_body_hash.as_str()))
                    .copied()
                    .filter(|id| !consumed.contains(id))
            });
        let entity_id = match matched {
            Some(id) => {
                consumed.insert(id);
                id.to_string()
            }
            None => mint_entity_id(project_label, relative_path, &entity),
        };
        let version_ingredients = [
            entity_id.as_str(),
            entity.source_hash.as_str(),
            &entity.span.start_byte.to_string(),
            &entity.span.end_byte.to_string(),
        ]
        .join("\0");
        let entity_version_id = format!("ver:{}", &sha256_hex(&version_ingredients)[..16]);
        out.push(NewCodeUnit::from_entity(
            entity,
            entity_id,
            entity_version_id,
            generation,
        ));
    }
    out
}

fn mint_entity_id(project_label: &str, relative_path: &str, entity: &ExtractedEntity) -> String {
    let ingredients = [
        project_label,
        relative_path,
        entity.kind.as_str(),
        entity.scope.as_deref().unwrap_or(""),
        entity.name.as_str(),
        &entity.span.start_byte.to_string(),
    ]
    .join("\0");
    format!("ent:{}", &sha256_hex(&ingredients)[..16])
}

fn apply_retention(units: &mut [NewCodeUnit], retention: RetentionMode) {
    for unit in units {
        for repr in &mut unit.representations {
            match retention {
                RetentionMode::Full => {}
                // Report keeps display source (FullSource) but drops the
                // embed-only channels' text, which is recoverable from source.
                RetentionMode::Report => {
                    if repr.kind != RepresentationKind::FullSource {
                        repr.content = None;
                    }
                }
                RetentionMode::Minimal => repr.content = None,
            }
        }
    }
}

/// Attribute each raw reference to the innermost unit whose byte span contains
/// it, and stage it for the Usage pass.
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
        // Innermost containing unit = largest start_byte still covering it.
        let mut best: Option<usize> = None;
        for (index, unit) in units.iter().enumerate() {
            if unit.start_byte <= reference.start_byte && reference.start_byte < unit.end_byte {
                match best {
                    Some(prev) if units[prev].start_byte >= unit.start_byte => {}
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

/// Resolve staged call sites in a project into the `Usage` channel. For each
/// callee entity, assemble a deterministic document of its call sites (caller
/// qualified name + snippet) and store it as that unit's `Usage` representation.
/// Name-based, same-project resolution (best effort); ambiguous names resolve
/// to every candidate.
fn resolve_usage(db: &Db, project_id: ProjectId) -> Result<()> {
    db.clear_channel_for_project(project_id, &RepresentationKind::Usage)?;

    let units = db.list_units_for_project(project_id)?;
    if units.is_empty() {
        return Ok(());
    }
    // name -> unit ids defining it (last path segment of the symbol).
    let mut defs: HashMap<String, Vec<i64>> = HashMap::new();
    let mut qualified: HashMap<i64, String> = HashMap::new();
    for unit in &units {
        if let Some(name) = symbol_name(&unit.name) {
            defs.entry(name).or_default().push(unit.id);
        }
        let qual = match &unit.scope {
            Some(scope) => format!("{scope}.{}", unit.name),
            None => unit.name.clone(),
        };
        qualified.insert(unit.id, qual);
    }

    let references = db.references_for_project(project_id)?;
    // callee unit id -> sorted, de-duplicated usage lines.
    let mut usages: BTreeMap<i64, BTreeSet<String>> = BTreeMap::new();
    for (caller_unit_id, _caller_name, _caller_scope, callee_symbol, snippet, _line) in references {
        let Some(name) = symbol_name(&callee_symbol) else {
            continue;
        };
        let Some(callee_ids) = defs.get(&name) else {
            continue;
        };
        let caller_qual = qualified
            .get(&caller_unit_id)
            .cloned()
            .unwrap_or_else(|| "?".to_string());
        for &callee_id in callee_ids {
            if callee_id == caller_unit_id {
                continue; // skip trivial self-recursion
            }
            usages
                .entry(callee_id)
                .or_default()
                .insert(format!("{caller_qual}: {snippet}"));
        }
    }

    for (unit_id, lines) in usages {
        let text = lines.into_iter().collect::<Vec<_>>().join("\n");
        let hash = sha256_hex(&text);
        db.set_representation(unit_id, &RepresentationKind::Usage, &hash, Some(&text))?;
    }
    Ok(())
}

/// Reduce a raw callee expression to a bare symbol name: drop macro `!`,
/// balanced generic argument groups, and any `::`/`.` path prefix.
fn symbol_name(raw: &str) -> Option<String> {
    let trimmed = raw.trim().trim_end_matches('!');
    // Drop everything inside balanced `<...>` (turbofish and generics).
    let mut without_generics = String::with_capacity(trimmed.len());
    let mut depth = 0usize;
    for ch in trimmed.chars() {
        match ch {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            _ if depth == 0 => without_generics.push(ch),
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

    #[test]
    fn indexes_into_existing_sqlite_schema() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("lib.rs"),
            "fn add(a: i32, b: i32) -> i32 { let value = a + b; value }",
        )
        .unwrap();
        let db = codeindex_sqlite::open_in_memory().unwrap();
        let options = IndexOptions {
            projects: vec![ProjectSpec {
                label: "main".into(),
                source_dir: root.path().to_path_buf(),
                exclude: Vec::new(),
            }],
            enabled_languages: vec!["rust".into()],
            body_node_count_threshold: 1,
            max_body_chars: 10_000,
            retention: RetentionMode::Full,
        };
        let stats = index(&db, &options, None).unwrap();
        assert_eq!(stats[0].indexed, 1);
        assert_eq!(stats[0].total_units, 1);
    }

    #[test]
    fn symbol_name_reduces_paths_and_generics() {
        assert_eq!(symbol_name("foo").as_deref(), Some("foo"));
        assert_eq!(symbol_name("self.method").as_deref(), Some("method"));
        assert_eq!(symbol_name("Type::<T>::build").as_deref(), Some("build"));
        assert_eq!(symbol_name("vec!").as_deref(), Some("vec"));
        assert_eq!(symbol_name("").as_deref(), None);
    }
}
