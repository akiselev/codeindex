//! The retrieval pipeline shared by the daemon and the direct CLI path, and
//! the background index/embed job run when a project is added or reindexed.
//!
//! Extracting this from the CLI keeps one implementation of the fusion
//! recipe: dense (+ compressed variant), lexical BM25, reciprocal-rank
//! fusion, then relation-graph expansion of the top seeds. Reranking stays
//! a caller-side post-step because it is feature-gated and needs unit text.

use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use codeindex_config::{OverrideFile, ProjectConfig};
use codeindex_core::{
    DocumentSideContract, EmbeddingSpaceId, EmbeddingSpaceIdentity, EmbeddingTask,
};
use codeindex_embedding::EmbeddingBackend;
use codeindex_indexer::{
    FileSystemSource, IndexOutcome, IndexRunBuilder, IndexSettings, RefreshPolicy, ResumePolicy,
    RetentionMode, RevisionTrust, SourceProject,
};
use codeindex_query::{
    FusedIndex, RankedList, TestsPolicy, WhereFilter, compress_query, reciprocal_rank_fusion,
};
use codeindex_search::SearchIndex;
use codeindex_sqlite::{Db, open_or_create};

use crate::protocol::{SearchHit, SearchResults};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retrieval {
    Hybrid,
    Dense,
    Lexical,
}

impl FromStr for Retrieval {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "hybrid" => Ok(Self::Hybrid),
            "dense" => Ok(Self::Dense),
            "lexical" => Ok(Self::Lexical),
            other => bail!("unknown retrieval mode {other:?} (hybrid|dense|lexical)"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compress {
    Auto,
    Off,
    Always,
}

impl FromStr for Compress {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            "always" => Ok(Self::Always),
            other => bail!("unknown compression mode {other:?} (auto|off|always)"),
        }
    }
}

pub struct HybridOptions {
    pub query: String,
    pub task: Option<EmbeddingTask>,
    /// Dense space; `None` forces lexical-only retrieval.
    pub space: Option<EmbeddingSpaceId>,
    pub filter: WhereFilter,
    pub limit: usize,
    pub retrieval: Retrieval,
    pub compress: Compress,
    pub graph: bool,
}

pub struct HybridOutcome {
    pub fused: Vec<FusedIndex>,
    pub compressed_query: Option<String>,
}

/// First-stage retrieval: candidate lists, fusion, graph expansion.
/// `embedder` may be `None` only for lexical-only retrieval.
pub fn hybrid_search<'e>(
    db: &Db,
    index: &SearchIndex,
    mut embedder: Option<&mut (dyn EmbeddingBackend + 'e)>,
    options: &HybridOptions,
) -> Result<HybridOutcome> {
    let candidates = (options.limit * 5).max(50);
    let dense_wanted = options.retrieval != Retrieval::Lexical && options.space.is_some();
    if dense_wanted && embedder.is_none() {
        bail!("dense retrieval requested but no embedding backend was provided");
    }

    let dense_hits = |embedder: &mut dyn EmbeddingBackend, text: &str| -> Result<Vec<usize>> {
        let space = options.space.as_ref().expect("dense implies a space");
        Ok(index
            .search_text(
                embedder,
                text,
                options.task.as_ref(),
                space,
                &options.filter,
                candidates,
            )?
            .hits
            .iter()
            .map(|hit| hit.index)
            .collect())
    };

    let dense_indices: Vec<usize> = match (dense_wanted, embedder.as_deref_mut()) {
        (true, Some(embedder)) => dense_hits(embedder, &options.query)?,
        _ => Vec::new(),
    };
    // A compressed variant recovers targets whose salient terms drown in
    // narrative phrasing; fused alongside the original, never instead.
    let compressed_query = match options.compress {
        Compress::Off => None,
        Compress::Auto => compress_query(&options.query, 24, 25),
        Compress::Always => compress_query(&options.query, 24, 0),
    };
    let compressed_indices: Vec<usize> = match (dense_wanted, embedder, compressed_query.as_deref())
    {
        (true, Some(embedder), Some(compressed)) => dense_hits(embedder, compressed)?,
        _ => Vec::new(),
    };

    let lexical_indices: Vec<usize> = if options.retrieval == Retrieval::Dense {
        Vec::new()
    } else {
        let by_version: HashMap<(&str, &str), usize> = index
            .units
            .iter()
            .enumerate()
            .map(|(position, unit)| {
                (
                    (unit.project_label.as_str(), unit.entity_version_id.as_str()),
                    position,
                )
            })
            .collect();
        db.lexical_search(&options.query, candidates * 2)?
            .iter()
            .filter_map(|hit| {
                by_version
                    .get(&(hit.project_label.as_str(), hit.entity_version_id.as_str()))
                    .copied()
            })
            .filter(|position| options.filter.matches(&index.units[*position]))
            .take(candidates)
            .collect()
    };

    let mut lists = Vec::new();
    if !dense_indices.is_empty() {
        lists.push(RankedList {
            source: "dense".into(),
            weight: 1.0,
            indices: dense_indices,
        });
    }
    if !compressed_indices.is_empty() {
        lists.push(RankedList {
            source: "dense-compressed".into(),
            weight: 0.7,
            indices: compressed_indices,
        });
    }
    if !lexical_indices.is_empty() {
        lists.push(RankedList {
            source: "lexical".into(),
            weight: 1.0,
            indices: lexical_indices,
        });
    }
    let mut fused = reciprocal_rank_fusion(&lists, 60);

    // Relation-graph expansion: 1-hop callers/callees of the top seeds join
    // as a low-weight list and everything re-fuses.
    if options.graph && !index.relations.is_empty() && !fused.is_empty() {
        let seeds: Vec<usize> = fused.iter().take(10).map(|hit| hit.index).collect();
        let expanded: Vec<usize> = index
            .expand_by_relations(&seeds, candidates)
            .into_iter()
            .filter(|position| options.filter.matches(&index.units[*position]))
            .collect();
        if !expanded.is_empty() {
            lists.push(RankedList {
                source: "graph".into(),
                weight: 0.5,
                indices: expanded,
            });
            fused = reciprocal_rank_fusion(&lists, 60);
        }
    }

    Ok(HybridOutcome {
        fused,
        compressed_query,
    })
}

/// Shape a fused outcome into the shared output payload.
pub fn shape_results(
    index: &SearchIndex,
    outcome: &HybridOutcome,
    options: &HybridOptions,
    rerank_scores: &HashMap<usize, f32>,
) -> SearchResults {
    let hits = outcome
        .fused
        .iter()
        .take(options.limit)
        .map(|hit| {
            let unit = &index.units[hit.index];
            SearchHit {
                selector: codeindex_query::unit_id(unit),
                score: rerank_scores.get(&hit.index).copied().unwrap_or(hit.score),
                rerank_score: rerank_scores.get(&hit.index).copied(),
                sources: hit
                    .contributions
                    .iter()
                    .map(|(source, rank)| format!("{source}#{rank}"))
                    .collect(),
                project: unit.project_label.clone(),
                path: unit.relative_path.clone(),
                lines: [unit.start_line, unit.end_line],
                language: unit.language_id.clone(),
                kind: unit.kind.clone(),
                name: unit.name.clone(),
                scope: unit.scope.clone(),
            }
        })
        .collect();
    SearchResults {
        query: options.query.clone(),
        compressed_query: outcome.compressed_query.clone(),
        space: options.space.as_ref().map(|space| space.to_string()),
        task: options.task.clone(),
        matched: outcome.fused.len(),
        hits,
    }
}

/// Apply a tests policy string (`include`/`exclude`/`only`) unless the
/// filter already carries an explicit `tests=` clause.
pub fn apply_tests_policy(filter: &mut WhereFilter, policy: Option<&str>) -> Result<()> {
    if filter.tests_policy().is_some() {
        return Ok(());
    }
    let policy = match policy {
        None | Some("exclude") => TestsPolicy::Exclude,
        Some("include") => return Ok(()),
        Some("only") => TestsPolicy::Only,
        Some(other) => bail!("unknown tests policy {other:?} (include|exclude|only)"),
    };
    filter.set_tests_policy(policy);
    Ok(())
}

// ---- background index/embed job ------------------------------------------

pub struct JobSummary {
    pub units: i64,
    pub embedded_spaces: Vec<(String, usize)>,
}

/// One full index pass for a registered project: tree-sitter indexing with
/// the effective excludes, then a projection of every configured embedding
/// space through a (warm) backend supplied per model reference.
pub fn run_index_job(
    db_path: &Path,
    label: &str,
    root: &Path,
    config: &ProjectConfig,
    overrides: &[OverrideFile],
    embedder_for: &mut dyn FnMut(&str) -> Result<Box<dyn EmbeddingBackend>>,
    mut progress: impl FnMut(&str, &str),
) -> Result<JobSummary> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let db = open_or_create(db_path)?;

    progress("indexing", "walking sources");
    let excludes = codeindex_config::effective_excludes(config, overrides);
    let source = FileSystemSource::new(root).with_excludes(excludes);
    let projects = [SourceProject {
        label: label.to_string(),
        provider: &source,
    }];
    let settings = IndexSettings {
        enabled_languages: config.index.languages.clone().unwrap_or_else(|| {
            codeindex_tree_sitter::BUNDLED_LANGUAGE_IDS
                .iter()
                .map(|language| (*language).to_string())
                .collect()
        }),
        body_node_count_threshold: 10,
        max_body_chars: 10_000,
        retention: RetentionMode::Full,
    };
    let outcome = IndexRunBuilder::new(&db, &settings, &projects)
        .resume_policy(ResumePolicy::Auto)
        .refresh_policy(RefreshPolicy::default())
        .revision_trust(RevisionTrust::VerifyContent)
        .run()?;
    if let IndexOutcome::Paused(status) = outcome {
        bail!(
            "index run {} paused: {}",
            status.run_id,
            status.pause_reason.as_deref().unwrap_or("unspecified")
        );
    }

    let mut embedded_spaces = Vec::new();
    for (space_id, space) in &config.spaces {
        progress("embedding", space_id);
        let mut embedder = embedder_for(&space.model)
            .with_context(|| format!("loading model for space {space_id:?}"))?;
        let identity = EmbeddingSpaceIdentity::new(
            EmbeddingSpaceId::new(space_id.clone()),
            space.channel.as_str().into(),
            embedder.contract().clone(),
        )
        .with_document_side(DocumentSideContract {
            prompt: space.document_prompt.clone(),
            output_dimensions: space.output_dimensions,
        });
        let run_config = codeindex_embedding::config::EmbeddingRunConfig {
            embedding: codeindex_embedding::config::EmbeddingConfig {
                model: space.model.clone(),
                ..Default::default()
            },
            source_recovery: codeindex_embedding::config::SourceRecoveryConfig {
                body_node_count_threshold: 10,
            },
        };
        let stats = codeindex_indexer::embed_space_pending_with_progress(
            &db,
            embedder.as_mut(),
            &run_config,
            &identity,
            None,
            &mut |_| {},
        )?;
        embedded_spaces.push((space_id.clone(), stats.embedded));
    }

    Ok(JobSummary {
        units: db.count_units()?,
        embedded_spaces,
    })
}
