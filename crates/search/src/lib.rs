#![forbid(unsafe_code)]

//! The end-to-end semantic search service over a code index.
//!
//! This crate owns the *operation* a consumer wants:
//!
//! > natural-language sentence → embed with the corpus's model → verify model
//! > compatibility → retrieve candidate vectors for a channel → rank/filter →
//! > resolve back to code-unit metadata → structured results
//!
//! It loads exclusively from a storage-neutral [`codeindex_storage::IndexSnapshot`]
//! ([`SearchIndex::from_snapshot`]); it never touches SQLite or any other store
//! directly. Any backend that can produce an `IndexSnapshot` — SQLite via
//! `Db::snapshot`, or any other database by deserializing its rows into the
//! public snapshot type — can be searched.
//!
//! Every representation channel (`Implementation`, `Signature`, `Documentation`,
//! `Symbol`, `Usage`, …) is embedded and searched independently: a query
//! targets one channel. Presentation (JSON envelopes, text formatting) is left
//! to the caller.

pub mod vector_store;

use std::collections::HashMap;

use anyhow::{Context as _, Result, bail};
use codeindex_core::{ModelIdentity, RepresentationKind};
use codeindex_embedding::{Embedder, normalize_in_place};
use codeindex_query::{UnitView, WhereFilter, identity_diff, rank_candidates, unit_id};
use codeindex_storage::{IndexSnapshot, ProjectRecord, RepresentationRef};

pub use vector_store::{ScoredPair, VectorStore, dot};

/// Re-export the snapshot types so consumers can name what the service loads
/// from without depending on `codeindex-storage` directly.
pub use codeindex_storage as storage;

/// A code unit joined with its file and project — the metadata a search result
/// resolves back to. Built from a [`codeindex_storage::UnitRecord`].
#[derive(Debug, Clone, PartialEq)]
pub struct CodeUnitRef {
    pub entity_id: String,
    pub entity_version_id: String,
    pub generation: u64,
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
    /// The `Implementation` channel content hash — the stable body identity used
    /// by [`unit_id`] and rename detection.
    pub normalized_body_hash: String,
    /// `FullSource` content, when retention kept it.
    pub display_source: Option<String>,
    /// Every representation channel of this unit (used to look up its vector in
    /// a given channel by content hash).
    pub representations: Vec<RepresentationRef>,
}

impl CodeUnitRef {
    /// `label:path` display location.
    pub fn location(&self) -> String {
        format!("{}:{}", self.project_label, self.relative_path)
    }

    /// The content hash for a channel, if this unit carries it.
    pub fn content_hash(&self, kind: &RepresentationKind) -> Option<&str> {
        self.representations
            .iter()
            .find(|r| &r.kind == kind)
            .map(|r| r.content_hash.as_str())
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
/// caller's limit) plus how many candidates matched before truncation.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResults {
    pub matched: usize,
    pub hits: Vec<SearchHit>,
}

/// The loaded corpus: projects, their code units, the single embedding model's
/// identity, and one [`VectorStore`] per embedded representation channel.
pub struct SearchIndex {
    pub identity: ModelIdentity,
    pub projects: Vec<ProjectRecord>,
    /// Sorted by (project label, path, start byte) for determinism.
    pub units: Vec<CodeUnitRef>,
    /// One vector store per channel, each aligned with `units`.
    pub channels: HashMap<RepresentationKind, VectorStore>,
}

impl SearchIndex {
    /// Build the search index from a storage-neutral snapshot. This is the only
    /// entry point: SQLite and every other backend feed search through here.
    pub fn from_snapshot(snapshot: IndexSnapshot) -> SearchIndex {
        let identity = snapshot.model;
        let projects = snapshot.projects;
        let units: Vec<CodeUnitRef> = snapshot
            .units
            .into_iter()
            .map(|unit| {
                let normalized_body_hash = unit
                    .content_hash(&RepresentationKind::Implementation)
                    .unwrap_or_default()
                    .to_string();
                let display_source = unit
                    .content(&RepresentationKind::FullSource)
                    .map(str::to_owned);
                CodeUnitRef {
                    entity_id: unit.entity_id,
                    entity_version_id: unit.entity_version_id,
                    generation: unit.generation,
                    project_label: unit.project_label,
                    relative_path: unit.relative_path,
                    language_id: unit.language_id,
                    kind: unit.kind,
                    name: unit.name,
                    scope: unit.scope,
                    start_byte: unit.span.start_byte,
                    end_byte: unit.span.end_byte,
                    start_line: unit.span.start_line,
                    end_line: unit.span.end_line,
                    body_node_count: unit.body_node_count,
                    normalized_body_hash,
                    display_source,
                    representations: unit.representations,
                }
            })
            .collect();

        // Build one vector store per channel, aligning each unit to its vector
        // in that channel by content hash.
        let mut channels = HashMap::new();
        for channel in &snapshot.channels {
            let by_hash = channel.by_hash();
            let vectors: Vec<Option<Vec<f32>>> = units
                .iter()
                .map(|unit| {
                    unit.content_hash(&channel.channel)
                        .and_then(|hash| by_hash.get(hash))
                        .map(|vector| vector.to_vec())
                })
                .collect();
            channels.insert(
                channel.channel.clone(),
                VectorStore::from_unit_vectors(channel.dimensions, vectors),
            );
        }

        SearchIndex {
            identity,
            projects,
            units,
            channels,
        }
    }

    fn channel_store(&self, channel: &RepresentationKind) -> Result<&VectorStore> {
        self.channels.get(channel).with_context(|| {
            format!(
                "channel {channel} has no embeddings in this index; embedded channels: {}",
                self.embedded_channels()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
    }

    /// The channels that carry vectors in this index.
    pub fn embedded_channels(&self) -> impl Iterator<Item = &RepresentationKind> {
        self.channels.keys()
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
    /// rank every unit embedded in `channel` that passes `filter`. This is the
    /// full sentence → results path.
    pub fn search_text(
        &self,
        embedder: &mut dyn Embedder,
        text: &str,
        channel: &RepresentationKind,
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
        self.search_vector(&query_vector, channel, filter, limit)
    }

    /// Rank every unit embedded in `channel` that passes `filter` against an
    /// already-normalized query vector.
    pub fn search_vector(
        &self,
        query: &[f32],
        channel: &RepresentationKind,
        filter: &WhereFilter,
        limit: usize,
    ) -> Result<SearchResults> {
        let store = self.channel_store(channel)?;
        let candidates = (0..self.units.len()).filter_map(|index| {
            if !filter.matches(&self.units[index]) {
                return None;
            }
            let row = store.row_for_unit(index)?;
            Some((index, store.vector(row)))
        });
        Ok(self.finish(rank_candidates(query, candidates, -1.0), limit))
    }

    /// The top units most similar to `query_index` in `channel`, excluding the
    /// query unit and keeping only hits at or above `threshold`.
    pub fn similar_to_unit(
        &self,
        query_index: usize,
        channel: &RepresentationKind,
        filter: &WhereFilter,
        limit: usize,
        threshold: f32,
    ) -> Result<SearchResults> {
        let store = self.channel_store(channel)?;
        let query_row = store
            .row_for_unit(query_index)
            .context("query unit has no stored embedding in this channel")?;
        let query_vector = store.vector(query_row).to_vec();
        let candidates = (0..self.units.len()).filter_map(|index| {
            if index == query_index || !filter.matches(&self.units[index]) {
                return None;
            }
            let row = store.row_for_unit(index)?;
            Some((index, store.vector(row)))
        });
        Ok(self.finish(rank_candidates(&query_vector, candidates, threshold), limit))
    }

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
