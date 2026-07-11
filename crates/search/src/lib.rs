#![forbid(unsafe_code)]

//! The end-to-end semantic search service over a code index.
//!
//! The reusable crates below this one provide the *pieces* — SQLite storage
//! ([`codeindex_sqlite`]), embedding backends ([`codeindex_embedding`]), and
//! the ranking/filtering primitives ([`codeindex_query`]). This crate owns the
//! *operation* a consumer actually wants:
//!
//! > natural-language sentence → embed with the corpus's model → verify model
//! > compatibility → retrieve candidate vectors → rank/filter → resolve back to
//! > code-unit metadata → structured results
//!
//! A [`SearchIndex`] is the loaded corpus (projects, units, the single
//! embedding model's identity, and its dense vectors). [`SearchIndex::search_text`]
//! is the headline call; [`SearchIndex::search_vector`] and
//! [`SearchIndex::similar_to_unit`] cover the pre-embedded and unit-to-unit
//! cases. Presentation (JSON envelopes, text formatting) is left to the caller.

pub mod vector_store;

use std::collections::HashMap;

use anyhow::{Context as _, Result, bail};
use codeindex_embedding::{Embedder, normalize_in_place};
use codeindex_query::{UnitView, WhereFilter, identity_diff, rank_candidates, unit_id};
use codeindex_sqlite::{Db, ModelIdentity, Project, UnitId, blob_to_vector};
use rusqlite::params_from_iter;

pub use vector_store::{ScoredPair, VectorStore, dot};

/// Re-exported so consumers can name the DB handle the service loads from
/// without depending on `codeindex-sqlite` directly.
pub use codeindex_sqlite as sqlite;

/// A code unit joined with its file and project — the metadata a search result
/// resolves back to.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeUnitRef {
    pub id: UnitId,
    pub project_label: String,
    pub relative_path: String,
    pub language_id: String,
    pub kind: String,
    pub name: String,
    pub scope: Option<String>,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub body_node_count: usize,
    pub normalized_body_hash: String,
    pub display_source: Option<String>,
}

impl CodeUnitRef {
    /// `label:path` display location.
    pub fn location(&self) -> String {
        format!("{}:{}", self.project_label, self.relative_path)
    }
}

impl UnitView for CodeUnitRef {
    fn project_label(&self) -> &str {
        &self.project_label
    }
    fn relative_path(&self) -> &str {
        &self.relative_path
    }
    fn language_id(&self) -> &str {
        &self.language_id
    }
    fn kind(&self) -> &str {
        &self.kind
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn scope(&self) -> Option<&str> {
        self.scope.as_deref()
    }
    fn start_byte(&self) -> usize {
        self.start_byte
    }
    fn end_byte(&self) -> usize {
        self.end_byte
    }
    fn start_line(&self) -> usize {
        self.start_line
    }
    fn end_line(&self) -> usize {
        self.end_line
    }
    fn body_node_count(&self) -> usize {
        self.body_node_count
    }
    fn normalized_body_hash(&self) -> &str {
        &self.normalized_body_hash
    }
}

/// One ranked hit: an index into [`SearchIndex::units`] and its cosine score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchHit {
    pub index: usize,
    pub score: f32,
}

/// The outcome of a search: the hits returned (already truncated to the
/// caller's limit) plus how many candidates matched before truncation, so
/// callers can render "N of M" / `has_more`.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResults {
    pub matched: usize,
    pub hits: Vec<SearchHit>,
}

/// The loaded corpus: selected projects, their code units, the single
/// embedding model's identity, and its normalized vectors.
pub struct SearchIndex {
    pub model_id: i64,
    pub identity: ModelIdentity,
    pub projects: Vec<Project>,
    /// Sorted by (project label, path, start byte) for determinism.
    pub units: Vec<CodeUnitRef>,
    pub vectors: VectorStore,
}

/// Load selected projects (empty = all) and their code units without touching
/// embeddings. Used by metadata-only consumers (unit listing, inspection);
/// [`SearchIndex::load`] builds on it.
pub fn load_projects_and_units(
    db: &Db,
    project_labels: &[String],
) -> Result<(Vec<Project>, Vec<CodeUnitRef>)> {
    let all_projects = db.list_projects()?;
    let projects: Vec<Project> = if project_labels.is_empty() {
        all_projects
    } else {
        let mut selected = Vec::new();
        for label in project_labels {
            let project = all_projects
                .iter()
                .find(|p| &p.label == label)
                .with_context(|| format!("project {label:?} is not indexed"))?;
            selected.push(project.clone());
        }
        selected
    };
    if projects.is_empty() {
        bail!("no indexed projects; index a project first");
    }

    // Load units for the selected projects.
    let placeholders = projects.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT u.id, p.label, f.relative_path, u.language_id, u.kind, u.name, u.scope,
                    u.start_byte, u.end_byte, u.start_line, u.end_line,
                    u.body_node_count, u.normalized_body_hash, u.display_source
             FROM code_units u
             JOIN files f ON f.id = u.file_id
             JOIN projects p ON p.id = f.project_id
             WHERE p.id IN ({placeholders})
             ORDER BY p.label, f.relative_path, u.start_byte, u.end_byte"
    );
    let mut stmt = db.conn().prepare(&sql)?;
    let units: Vec<CodeUnitRef> = stmt
        .query_map(params_from_iter(projects.iter().map(|p| p.id)), |row| {
            Ok(CodeUnitRef {
                id: row.get(0)?,
                project_label: row.get(1)?,
                relative_path: row.get(2)?,
                language_id: row.get(3)?,
                kind: row.get(4)?,
                name: row.get(5)?,
                scope: row.get(6)?,
                start_byte: row.get::<_, i64>(7)? as usize,
                end_byte: row.get::<_, i64>(8)? as usize,
                start_line: row.get::<_, i64>(9)? as usize,
                end_line: row.get::<_, i64>(10)? as usize,
                body_node_count: row.get::<_, i64>(11)? as usize,
                normalized_body_hash: row.get(12)?,
                display_source: row.get(13)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    Ok((projects, units))
}

/// Projects + units without the vector store — for metadata-only operations
/// (listing, inspection) that never rank. A thin named wrapper over
/// [`load_projects_and_units`].
pub fn load_metadata(
    db: &Db,
    project_labels: &[String],
) -> Result<(Vec<Project>, Vec<CodeUnitRef>)> {
    load_projects_and_units(db, project_labels)
}

impl SearchIndex {
    /// Load the corpus for the given project labels (empty = all), including
    /// the single embedding model and its vectors.
    pub fn load(db: &Db, project_labels: &[String]) -> Result<SearchIndex> {
        let (projects, units) = load_projects_and_units(db, project_labels)?;

        // The search model: exactly one embedding model may exist per database
        // (enforced by the embed pipeline's immutable settings).
        let models = db.list_models()?;
        let model = match models.as_slice() {
            [] => bail!("no embeddings found; embed the corpus first"),
            [model] => model.clone(),
            _ => bail!("database contains multiple embedding models; this is unsupported"),
        };

        // Load this model's embeddings once, then assign per unit by hash.
        let mut by_hash: HashMap<String, Vec<f32>> = HashMap::new();
        let mut stmt = db.conn().prepare(
            "SELECT normalized_body_hash, vector_blob FROM embeddings WHERE model_id = ?1",
        )?;
        let rows = stmt.query_map([model.id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;
        for row in rows {
            let (hash, blob) = row?;
            by_hash.insert(hash, blob_to_vector(&blob));
        }

        let vectors = units
            .iter()
            .map(|unit| by_hash.get(&unit.normalized_body_hash).cloned())
            .collect();
        let vectors = VectorStore::from_unit_vectors(model.identity.dimensions, vectors);

        Ok(SearchIndex {
            model_id: model.id,
            identity: model.identity,
            projects,
            units,
            vectors,
        })
    }

    /// Unit indices belonging to one project label.
    pub fn unit_indices_for_project(&self, label: &str) -> Vec<usize> {
        self.units
            .iter()
            .enumerate()
            .filter(|(_, unit)| unit.project_label == label)
            .map(|(index, _)| index)
            .collect()
    }

    /// Embed `text` with `embedder`, verify it matches the corpus's model, then
    /// rank every embedded unit passing `filter`. This is the full
    /// sentence → results path: identity verification and query normalization
    /// happen here so no consumer has to reimplement them.
    pub fn search_text(
        &self,
        embedder: &mut dyn Embedder,
        text: &str,
        filter: &WhereFilter,
        limit: usize,
    ) -> Result<SearchResults> {
        let identity = embedder.identity();
        if *identity != self.identity {
            bail!(
                "search queries must be embedded with the same model identity as the indexed \
                 code units; the configured embedder differs from the database on: {}",
                identity_diff(&self.identity, identity).join(", ")
            );
        }
        let mut vectors = embedder.embed(std::slice::from_ref(&text.to_owned()))?;
        let mut query_vector = vectors.pop().context("embedder returned no vector")?;
        normalize_in_place(&mut query_vector);
        Ok(self.search_vector(&query_vector, filter, limit))
    }

    /// Rank every embedded unit passing `filter` against an already-normalized
    /// query vector. The embed-free core that [`Self::search_text`] delegates
    /// to; also usable directly by a consumer that owns its own vector.
    pub fn search_vector(
        &self,
        query: &[f32],
        filter: &WhereFilter,
        limit: usize,
    ) -> SearchResults {
        let candidates = (0..self.units.len()).filter_map(|index| {
            if !filter.matches(&self.units[index]) {
                return None;
            }
            let row = self.vectors.row_for_unit(index)?;
            Some((index, self.vectors.vector(row)))
        });
        self.finish(rank_candidates(query, candidates, -1.0), limit)
    }

    /// The top units most similar to `query_index`, excluding the query unit
    /// itself and keeping only hits at or above `threshold`. Errors if the
    /// query unit has no stored embedding.
    pub fn similar_to_unit(
        &self,
        query_index: usize,
        filter: &WhereFilter,
        limit: usize,
        threshold: f32,
    ) -> Result<SearchResults> {
        let query_row = self
            .vectors
            .row_for_unit(query_index)
            .context("query unit has no stored embedding")?;
        let query_vector = self.vectors.vector(query_row).to_vec();
        let candidates = (0..self.units.len()).filter_map(|index| {
            if index == query_index || !filter.matches(&self.units[index]) {
                return None;
            }
            let row = self.vectors.row_for_unit(index)?;
            Some((index, self.vectors.vector(row)))
        });
        Ok(self.finish(rank_candidates(&query_vector, candidates, threshold), limit))
    }

    /// Convert ranked candidates into results: record the pre-truncation match
    /// count, then keep the top `limit`.
    fn finish(&self, scored: Vec<codeindex_query::ScoredIndex>, limit: usize) -> SearchResults {
        let matched = scored.len();
        let hits = scored
            .into_iter()
            .take(limit)
            .map(|s| SearchHit {
                index: s.index,
                score: s.score,
            })
            .collect();
        SearchResults { matched, hits }
    }
}

/// Resolve a `unit:<id>` selector (as printed by query/report output) to an
/// index into `units`. The IDs are the deterministic [`unit_id`] hashes.
pub fn resolve_selector(units: &[CodeUnitRef], selector: &str) -> Result<usize> {
    if !selector.starts_with("unit:") {
        bail!(
            "selector {selector:?} is not a unit selector (expected `unit:<id>` as printed by \
             query/report output)"
        );
    }
    units
        .iter()
        .position(|unit| unit_id(unit) == selector)
        .with_context(|| {
            format!(
                "{selector} not found in the current index. Unit IDs are deterministic per index \
                 generation and change when code is re-indexed; re-run the query that produced \
                 the ID, or list units."
            )
        })
}
