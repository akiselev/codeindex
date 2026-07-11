#![forbid(unsafe_code)]

//! The storage-neutral, serializable contract between a persistence backend and
//! the search engine.
//!
//! `codeindex-search` loads a corpus exclusively from an [`IndexSnapshot`]; it
//! never touches SQLite (or any other store) directly. A backend's only
//! obligation is to *produce an `IndexSnapshot`* — `codeindex-sqlite` does it
//! with SQL in `Db::snapshot`, but any other database can do it by running its
//! own queries and deserializing the rows into these `serde` types. That is the
//! whole "support any database" story: it is deserialization into a public type,
//! not a store-specific trait the engine has to know about.
//!
//! The snapshot is the *canonical* contract. A streaming reader for corpora too
//! large to hold in memory is a possible future addition, but every backend must
//! be expressible as one of these values.

use std::collections::HashMap;

use codeindex_core::{ModelIdentity, RepresentationKind, SourceSpan};
use serde::{Deserialize, Serialize};

/// A complete, self-contained view of one embedding model's corpus: the model
/// identity, the selected projects, their code units (with every representation
/// channel), and the dense vectors for each embedded channel.
///
/// One snapshot binds to exactly one [`ModelIdentity`] — the same invariant the
/// stores enforce (a database holds a single embedding model).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexSnapshot {
    pub model: ModelIdentity,
    pub projects: Vec<ProjectRecord>,
    pub units: Vec<UnitRecord>,
    /// One entry per representation channel that has stored embeddings.
    pub channels: Vec<ChannelEmbeddings>,
}

impl IndexSnapshot {
    /// The embeddings for one channel, if any were stored for it.
    pub fn channel(&self, kind: &RepresentationKind) -> Option<&ChannelEmbeddings> {
        self.channels.iter().find(|c| &c.channel == kind)
    }

    /// The channels that actually carry vectors, in stored order.
    pub fn embedded_channels(&self) -> impl Iterator<Item = &RepresentationKind> {
        self.channels.iter().map(|c| &c.channel)
    }
}

/// A project (a labelled source root) in the snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectRecord {
    pub label: String,
    pub source_dir: String,
    pub role: Option<String>,
}

/// One code unit: its stable/version identity, location, and every stored
/// representation channel. Vectors live in [`ChannelEmbeddings`], keyed by the
/// per-channel `content_hash` found here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnitRecord {
    /// Stable logical identity carried across index generations (M4).
    pub entity_id: String,
    /// Exact identity of this indexed version of the entity (M4).
    pub entity_version_id: String,
    /// The generation (index run) this version was written in.
    pub generation: u64,
    pub project_label: String,
    pub relative_path: String,
    pub language_id: String,
    pub kind: String,
    pub name: String,
    pub scope: Option<String>,
    pub span: SourceSpan,
    pub body_node_count: usize,
    /// Every representation channel for this unit. `content` is `None` when
    /// retention dropped the text (it can be recovered from source); the
    /// `content_hash` is always present and is the embedding lookup key.
    pub representations: Vec<RepresentationRef>,
}

impl UnitRecord {
    /// The stored representation for a channel, if this unit has one.
    pub fn representation(&self, kind: &RepresentationKind) -> Option<&RepresentationRef> {
        self.representations.iter().find(|r| &r.kind == kind)
    }

    /// The content hash for a channel — the key into [`ChannelEmbeddings`].
    pub fn content_hash(&self, kind: &RepresentationKind) -> Option<&str> {
        self.representation(kind).map(|r| r.content_hash.as_str())
    }

    /// The stored text for a channel, when retention kept it.
    pub fn content(&self, kind: &RepresentationKind) -> Option<&str> {
        self.representation(kind).and_then(|r| r.content.as_deref())
    }
}

/// One representation channel of a unit: which channel, its content hash, and
/// optionally the text itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentationRef {
    pub kind: RepresentationKind,
    pub content_hash: String,
    pub content: Option<String>,
}

/// The dense vectors stored for one representation channel, keyed by content
/// hash. Units share a vector whenever their channel content hashes match, so
/// this holds one entry per *distinct* content, not per unit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelEmbeddings {
    pub channel: RepresentationKind,
    pub dimensions: usize,
    /// `content_hash -> vector`.
    pub vectors: Vec<(String, Vec<f32>)>,
}

impl ChannelEmbeddings {
    /// Index the vectors by content hash for lookup.
    pub fn by_hash(&self) -> HashMap<&str, &[f32]> {
        self.vectors
            .iter()
            .map(|(hash, vector)| (hash.as_str(), vector.as_slice()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_json_round_trips() {
        let snapshot = IndexSnapshot {
            model: ModelIdentity {
                backend: "hash".into(),
                backend_version: "0".into(),
                runtime_version: None,
                model: "test".into(),
                revision: None,
                dimensions: 2,
                tokenizer_hash: None,
                model_hash: None,
                normalize: true,
                execution_provider: "cpu".into(),
                quantization: None,
                cache_path: None,
            },
            projects: vec![ProjectRecord {
                label: "main".into(),
                source_dir: "/src".into(),
                role: None,
            }],
            units: vec![UnitRecord {
                entity_id: "e1".into(),
                entity_version_id: "v1".into(),
                generation: 1,
                project_label: "main".into(),
                relative_path: "lib.rs".into(),
                language_id: "rust".into(),
                kind: "function".into(),
                name: "parse".into(),
                scope: None,
                span: SourceSpan::new(0, 10, 1, 1),
                body_node_count: 3,
                representations: vec![RepresentationRef {
                    kind: RepresentationKind::Implementation,
                    content_hash: "h".into(),
                    content: Some("fn parse() {}".into()),
                }],
            }],
            channels: vec![ChannelEmbeddings {
                channel: RepresentationKind::Implementation,
                dimensions: 2,
                vectors: vec![("h".into(), vec![1.0, 0.0])],
            }],
        };
        let json = serde_json::to_string(&snapshot).unwrap();
        let back: IndexSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(snapshot, back);
        // Channels serialize by their canonical string token.
        assert!(json.contains("\"implementation\""));
        assert_eq!(
            back.units[0].content_hash(&RepresentationKind::Implementation),
            Some("h")
        );
    }
}
