#![forbid(unsafe_code)]

//! Storage-neutral, serializable contracts between persistence backends and the
//! search engine.
//!
//! `codeindex-search` loads a corpus exclusively from [`IndexSnapshot`]. SQLite
//! produces one through `Db::snapshot`; any other store can construct or
//! deserialize the same public types. Embedding spaces are first-class: one
//! snapshot can carry different models for implementation, documentation,
//! usage, or any custom representation channel.

use codeindex_core::{
    EmbeddingSpaceId, EmbeddingSpaceIdentity, EntityId, EntityVersionId, RepresentationKind,
    RepresentationOrigin, SourceSpan,
};
use serde::{Deserialize, Serialize};

/// A complete, self-contained view of selected projects and all requested
/// embedding spaces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexSnapshot {
    /// Identity of the latest fully published indexing run.
    #[serde(default)]
    pub published_generation: u64,
    pub projects: Vec<ProjectRecord>,
    pub units: Vec<UnitRecord>,
    pub spaces: Vec<EmbeddingSpaceSnapshot>,
    /// Typed relations between entities (calls, references, implements, …),
    /// produced by resolved analyzers such as LSP servers after publication.
    #[serde(default)]
    pub relations: Vec<RelationRecord>,
}

/// One typed, provenance-carrying edge between entities. `to_entity_id` is
/// unset when the target resolves outside the indexed corpus; `to_symbol`
/// always carries the analyzer's name for the target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationRecord {
    pub from_entity_id: EntityId,
    pub to_entity_id: Option<EntityId>,
    pub to_symbol: String,
    /// Documented vocabulary: `calls`, `references`, `implements`, `type-of`,
    /// `defines`. Producers may add kinds; consumers must tolerate unknowns.
    pub kind: String,
    /// `exact` (resolved by a compiler-grade tool) or `heuristic`.
    pub resolution: String,
    /// Producer identity, e.g. `lsp:rust-analyzer`.
    pub provenance: String,
}

impl IndexSnapshot {
    pub fn space(&self, id: &EmbeddingSpaceId) -> Option<&EmbeddingSpaceSnapshot> {
        self.spaces.iter().find(|space| &space.identity.id == id)
    }

    pub fn embedding_spaces(&self) -> impl Iterator<Item = &EmbeddingSpaceIdentity> {
        self.spaces.iter().map(|space| &space.identity)
    }
}

/// A project in the snapshot. `source_dir` is retained for wire compatibility,
/// but semantically it is the provider's opaque project locator and need not be
/// a filesystem path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectRecord {
    pub label: String,
    pub source_dir: String,
    /// The run that most recently reconciled this project.
    #[serde(default)]
    pub last_index_run_id: Option<u64>,
}

/// One code unit: stable/version identity, location, and stored representations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnitRecord {
    pub entity_id: EntityId,
    pub entity_version_id: EntityVersionId,
    pub generation: u64,
    pub project_label: String,
    pub relative_path: String,
    pub language_id: String,
    pub kind: String,
    pub name: String,
    pub scope: Option<String>,
    pub span: SourceSpan,
    pub body_node_count: usize,
    /// Hash of the normalized body text, as used by the entity identity
    /// matcher.
    pub normalized_body_hash: String,
    pub representations: Vec<RepresentationRef>,
}

impl UnitRecord {
    pub fn representation(&self, kind: &RepresentationKind) -> Option<&RepresentationRef> {
        self.representations.iter().find(|repr| &repr.kind == kind)
    }

    pub fn content_hash(&self, kind: &RepresentationKind) -> Option<&str> {
        self.representation(kind)
            .map(|repr| repr.content_hash.as_str())
    }

    pub fn content(&self, kind: &RepresentationKind) -> Option<&str> {
        self.representation(kind)
            .and_then(|repr| repr.content.as_deref())
    }
}

/// One representation channel of a unit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentationRef {
    pub kind: RepresentationKind,
    pub content_hash: String,
    pub content: Option<String>,
    pub origin: RepresentationOrigin,
}

/// Dense vectors for one independently queryable embedding space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingSpaceSnapshot {
    pub identity: EmbeddingSpaceIdentity,
    /// `content_hash -> vector`. Identical representations share one vector.
    pub vectors: Vec<(String, Vec<f32>)>,
}

#[cfg(test)]
mod tests {
    use codeindex_core::{ModelContract, Pooling, PromptContract, RepresentationOrigin};

    use super::*;

    fn model(name: &str, dimensions: usize) -> ModelContract {
        ModelContract {
            model: name.into(),
            revision: None,
            model_hash: None,
            tokenizer_hash: None,
            pooling: Pooling::Mean,
            normalize: true,
            native_dimensions: dimensions,
            max_sequence_length: 512,
            prompts: PromptContract::Symmetric,
            quantization: None,
        }
    }

    #[test]
    fn snapshot_json_round_trips_multiple_spaces() {
        let snapshot = IndexSnapshot {
            published_generation: 1,
            relations: Vec::new(),
            projects: vec![ProjectRecord {
                label: "main".into(),
                source_dir: "memory://fixture".into(),
                last_index_run_id: Some(1),
            }],
            units: vec![UnitRecord {
                entity_id: EntityId::new("e1"),
                entity_version_id: EntityVersionId::new("v1"),
                generation: 1,
                project_label: "main".into(),
                relative_path: "lib.rs".into(),
                language_id: "rust".into(),
                kind: "function".into(),
                name: "parse".into(),
                scope: None,
                span: SourceSpan::new(0, 10, 1, 1),
                body_node_count: 3,
                normalized_body_hash: "body-hash".into(),
                representations: vec![RepresentationRef {
                    kind: RepresentationKind::Implementation,
                    content_hash: "h".into(),
                    content: Some("fn parse() {}".into()),
                    origin: RepresentationOrigin::default(),
                }],
            }],
            spaces: vec![
                EmbeddingSpaceSnapshot {
                    identity: EmbeddingSpaceIdentity::new(
                        "code",
                        RepresentationKind::Implementation,
                        model("code-model", 2),
                    ),
                    vectors: vec![("h".into(), vec![1.0, 0.0])],
                },
                EmbeddingSpaceSnapshot {
                    identity: EmbeddingSpaceIdentity::new(
                        "docs",
                        RepresentationKind::Documentation,
                        model("text-model", 3),
                    ),
                    vectors: Vec::new(),
                },
            ],
        };
        let json = serde_json::to_string(&snapshot).unwrap();
        let back: IndexSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(snapshot, back);
        assert_eq!(back.spaces.len(), 2);
        assert_eq!(
            back.units[0].content_hash(&RepresentationKind::Implementation),
            Some("h")
        );
    }
}
