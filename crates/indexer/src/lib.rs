#![forbid(unsafe_code)]

mod embed;
mod scanner;

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result};
use codeindex_sqlite::{Db, NewCodeUnit, NewFile};
use codeindex_tree_sitter::{ExtractOptions, LanguageRegistry, extract_units};

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

    let mut stats = Vec::new();
    for project in &options.projects {
        if only_label.is_some_and(|label| project.label != label) {
            continue;
        }
        stats.push(index_project(db, options, project)?);
    }
    if let Some(label) = only_label
        && stats.is_empty()
    {
        anyhow::bail!("no configured project labeled {label:?}");
    }
    let pruned = db.prune_orphan_embeddings()?;
    if pruned > 0 {
        eprintln!("pruned {pruned} orphaned embeddings");
    }
    Ok(stats)
}

fn index_project(db: &Db, options: &IndexOptions, project: &ProjectSpec) -> Result<ProjectStats> {
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
        let source_hash = codeindex_tree_sitter::normalizer::sha256_hex(&source);
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
        let mut units: Vec<NewCodeUnit> = entities.into_iter().map(NewCodeUnit::from).collect();
        apply_retention(&mut units, options.retention);

        let file_id = db.upsert_file(&NewFile {
            project_id,
            relative_path: file.relative_path.clone(),
            language_id: file.language_id.clone(),
            mtime_ns,
            size,
            source_hash,
        })?;
        db.insert_units(file_id, &units)?;
        stats.indexed += 1;
        stats.units += units.len();
    }

    for record in db.list_files(project_id)? {
        if !seen.contains(&record.relative_path) {
            db.delete_file(record.id)?;
            stats.removed += 1;
        }
    }
    stats.total_units = db.count_units_for_project(project_id)? as usize;
    Ok(stats)
}

fn apply_retention(units: &mut [NewCodeUnit], retention: RetentionMode) {
    for unit in units {
        match retention {
            RetentionMode::Full => {}
            RetentionMode::Report => unit.embedding_text = None,
            RetentionMode::Minimal => {
                unit.embedding_text = None;
                unit.display_source = None;
            }
        }
    }
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
}
