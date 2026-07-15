#![forbid(unsafe_code)]

//! Storage-neutral semantic search over independently modelled embedding spaces.

pub mod vector_store;

use std::collections::{BTreeMap, HashMap};

use anyhow::{Context as _, Result, bail};
use codeindex_core::{
    EmbeddingSpaceId, EmbeddingSpaceIdentity, EmbeddingTask, EntityId, EntityVersionId,
    RepresentationKind,
};
use codeindex_embedding::{
    EmbedRequest, EmbeddingBackend, apply_output_dimensions, normalize_in_place,
};
use codeindex_query::{UnitView, WhereFilter, contract_diff, rank_candidates, unit_id};
use codeindex_storage::{IndexSnapshot, ProjectRecord, RepresentationRef};

pub use codeindex_storage as storage;
pub use vector_store::{ScoredPair, VectorStore, dot};

#[derive(Debug, Clone, PartialEq)]
pub struct CodeUnitRef {
    pub entity_id: EntityId,
    pub entity_version_id: EntityVersionId,
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
    pub normalized_body_hash: String,
    pub display_source: Option<String>,
    pub representations: Vec<RepresentationRef>,
}

impl CodeUnitRef {
    pub fn location(&self) -> String {
        format!("{}:{}", self.project_label, self.relative_path)
    }

    pub fn content_hash(&self, kind: &RepresentationKind) -> Option<&str> {
        self.representations
            .iter()
            .find(|representation| &representation.kind == kind)
            .map(|representation| representation.content_hash.as_str())
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchHit {
    pub index: usize,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResults {
    pub matched: usize,
    pub hits: Vec<SearchHit>,
}

pub struct SearchSpace {
    pub identity: EmbeddingSpaceIdentity,
    pub vectors: VectorStore,
}

pub struct SearchIndex {
    pub projects: Vec<ProjectRecord>,
    pub units: Vec<CodeUnitRef>,
    /// Deterministic by space id.
    pub spaces: BTreeMap<EmbeddingSpaceId, SearchSpace>,
    /// Typed analyzer-produced relations between entities.
    pub relations: Vec<codeindex_storage::RelationRecord>,
}

#[derive(Debug, Clone, Copy)]
pub struct SpaceVectorQuery<'a> {
    pub space_id: &'a EmbeddingSpaceId,
    pub vector: &'a [f32],
    pub weight: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpaceContribution {
    pub space_id: EmbeddingSpaceId,
    pub rank: usize,
    pub raw_score: f32,
    pub contribution: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FusedSearchHit {
    pub index: usize,
    pub score: f32,
    pub contributions: Vec<SpaceContribution>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FusedSearchResults {
    pub matched: usize,
    pub hits: Vec<FusedSearchHit>,
}

impl SearchIndex {
    /// Validate and load a storage-neutral snapshot. Malformed external-backend
    /// data returns an error rather than panicking inside the vector store.
    pub fn from_snapshot(snapshot: IndexSnapshot) -> Result<SearchIndex> {
        let projects = snapshot.projects;
        let relations = snapshot.relations;
        let units: Vec<CodeUnitRef> = snapshot
            .units
            .into_iter()
            .map(|unit| {
                let normalized_body_hash = unit.normalized_body_hash.clone();
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

        let mut spaces = BTreeMap::new();
        for snapshot_space in snapshot.spaces {
            let identity = snapshot_space.identity;
            let dimensions = identity.effective_dimensions();
            if dimensions == 0 || identity.model.native_dimensions == 0 {
                bail!("embedding space {} has zero dimensions", identity.id);
            }
            if dimensions > identity.model.native_dimensions {
                bail!(
                    "embedding space {} projects to {dimensions} dimensions but its model is \
                     only {} wide",
                    identity.id,
                    identity.model.native_dimensions
                );
            }
            if spaces.contains_key(&identity.id) {
                bail!("duplicate embedding space id {}", identity.id);
            }
            let mut by_hash = HashMap::new();
            for (hash, vector) in snapshot_space.vectors {
                if vector.len() != dimensions {
                    bail!(
                        "embedding space {} vector {hash:?} has {} dimensions, expected \
                         {dimensions}",
                        identity.id,
                        vector.len(),
                    );
                }
                if !vector.iter().all(|value| value.is_finite()) {
                    bail!(
                        "embedding space {} vector {hash:?} contains non-finite values",
                        identity.id
                    );
                }
                if by_hash.insert(hash.clone(), vector).is_some() {
                    bail!(
                        "embedding space {} contains duplicate content hash {hash:?}",
                        identity.id
                    );
                }
            }
            let vectors = units
                .iter()
                .map(|unit| {
                    unit.content_hash(&identity.channel)
                        .and_then(|hash| by_hash.get(hash))
                        .cloned()
                })
                .collect();
            spaces.insert(
                identity.id.clone(),
                SearchSpace {
                    vectors: VectorStore::from_unit_vectors(dimensions, vectors),
                    identity,
                },
            );
        }

        Ok(SearchIndex {
            projects,
            units,
            spaces,
            relations,
        })
    }

    pub fn embedded_spaces(&self) -> impl Iterator<Item = &EmbeddingSpaceIdentity> {
        self.spaces.values().map(|space| &space.identity)
    }

    pub fn space(&self, id: &EmbeddingSpaceId) -> Result<&SearchSpace> {
        self.spaces
            .get(id)
            .with_context(|| format!("embedding space {id} is not loaded"))
    }

    pub fn unit_indices_for_project(&self, label: &str) -> Vec<usize> {
        self.units
            .iter()
            .enumerate()
            .filter(|(_, unit)| unit.project_label == label)
            .map(|(index, _)| index)
            .collect()
    }

    /// Embed `text` as a query — rendered through the model's prompt contract
    /// with the optional task instruction — and search the space. The task
    /// never has to match how documents were embedded: one document index
    /// serves many query intents.
    pub fn search_text(
        &self,
        embedder: &mut dyn EmbeddingBackend,
        text: &str,
        task: Option<&EmbeddingTask>,
        space_id: &EmbeddingSpaceId,
        filter: &WhereFilter,
        limit: usize,
    ) -> Result<SearchResults> {
        let space = self.space(space_id)?;
        let contract = embedder.contract();
        if contract != &space.identity.model {
            bail!(
                "search queries for space {} must use its model contract; differing fields: {}",
                space_id,
                contract_diff(&space.identity.model, contract).join(", ")
            );
        }
        let inputs = [text];
        let request = EmbedRequest::queries(&inputs, task);
        let mut vectors = embedder.embed(&request)?;
        let mut query = vectors.pop().context("embedder returned no vector")?;
        if space.identity.model.normalize {
            normalize_in_place(&mut query);
        }
        apply_output_dimensions(
            &mut query,
            space.identity.document_side.output_dimensions,
            space.identity.model.normalize,
        )?;
        self.search_vector(&query, space_id, filter, limit)
    }

    /// Filtered `(unit index, vector)` candidates of one space, optionally
    /// excluding a query unit.
    fn candidates<'a>(
        &'a self,
        space: &'a SearchSpace,
        filter: &'a WhereFilter,
        skip: Option<usize>,
    ) -> impl Iterator<Item = (usize, &'a [f32])> {
        (0..self.units.len()).filter_map(move |index| {
            if Some(index) == skip || !filter.matches(&self.units[index]) {
                return None;
            }
            let row = space.vectors.row_for_unit(index)?;
            Some((index, space.vectors.vector(row)))
        })
    }

    pub fn search_vector(
        &self,
        query: &[f32],
        space_id: &EmbeddingSpaceId,
        filter: &WhereFilter,
        limit: usize,
    ) -> Result<SearchResults> {
        let space = self.space(space_id)?;
        ensure_query_dimensions(query, &space.identity)?;
        Ok(finish(
            rank_candidates(
                query,
                self.candidates(space, filter, None),
                f32::NEG_INFINITY,
            ),
            limit,
        ))
    }

    pub fn similar_to_unit(
        &self,
        query_index: usize,
        space_id: &EmbeddingSpaceId,
        filter: &WhereFilter,
        limit: usize,
        threshold: f32,
    ) -> Result<SearchResults> {
        let space = self.space(space_id)?;
        let query_row = space
            .vectors
            .row_for_unit(query_index)
            .context("query unit has no stored embedding in this space")?;
        let query = space.vectors.vector(query_row).to_vec();
        Ok(finish(
            rank_candidates(
                &query,
                self.candidates(space, filter, Some(query_index)),
                threshold,
            ),
            limit,
        ))
    }

    /// Fuse independently ranked spaces with weighted reciprocal rank. Raw
    /// cosine values are retained as evidence but are never added across models.
    pub fn search_vectors_fused(
        &self,
        queries: &[SpaceVectorQuery<'_>],
        filter: &WhereFilter,
        limit: usize,
        rrf_k: usize,
    ) -> Result<FusedSearchResults> {
        anyhow::ensure!(
            !queries.is_empty(),
            "fusion requires at least one space query"
        );
        let mut fused: HashMap<usize, (f32, Vec<SpaceContribution>)> = HashMap::new();
        for query in queries {
            anyhow::ensure!(
                query.weight.is_finite() && query.weight > 0.0,
                "fusion weight for {} must be finite and positive",
                query.space_id
            );
            let space = self.space(query.space_id)?;
            ensure_query_dimensions(query.vector, &space.identity)?;
            for (zero_rank, scored) in rank_candidates(
                query.vector,
                self.candidates(space, filter, None),
                f32::NEG_INFINITY,
            )
            .into_iter()
            .enumerate()
            {
                let rank = zero_rank + 1;
                let contribution = query.weight / (rrf_k + rank) as f32;
                let entry = fused.entry(scored.index).or_default();
                entry.0 += contribution;
                entry.1.push(SpaceContribution {
                    space_id: query.space_id.clone(),
                    rank,
                    raw_score: scored.score,
                    contribution,
                });
            }
        }

        let matched = fused.len();
        let mut hits: Vec<FusedSearchHit> = fused
            .into_iter()
            .map(|(index, (score, mut contributions))| {
                contributions.sort_by(|left, right| left.space_id.cmp(&right.space_id));
                FusedSearchHit {
                    index,
                    score,
                    contributions,
                }
            })
            .collect();
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then(left.index.cmp(&right.index))
        });
        hits.truncate(limit);
        Ok(FusedSearchResults { matched, hits })
    }
}

fn ensure_query_dimensions(query: &[f32], identity: &EmbeddingSpaceIdentity) -> Result<()> {
    anyhow::ensure!(
        query.len() == identity.effective_dimensions(),
        "query for embedding space {} has {} dimensions, expected {}",
        identity.id,
        query.len(),
        identity.effective_dimensions()
    );
    anyhow::ensure!(
        query.iter().all(|value| value.is_finite()),
        "query for embedding space {} contains non-finite values",
        identity.id
    );
    Ok(())
}

fn finish(scored: Vec<codeindex_query::ScoredIndex>, limit: usize) -> SearchResults {
    let matched = scored.len();
    let hits = scored
        .into_iter()
        .take(limit)
        .map(|scored| SearchHit {
            index: scored.index,
            score: scored.score,
        })
        .collect();
    SearchResults { matched, hits }
}

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
        .with_context(|| format!("{selector} not found in the current index"))
}
