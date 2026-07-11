use codeindex_core::{
    EmbeddingSpaceId, EmbeddingSpaceIdentity, EntityId, EntityVersionId, ExtractedEntity,
    RepresentationKind, RepresentationOrigin,
};

pub use codeindex_core::ModelIdentity;

pub type ProjectId = i64;
pub type FileId = i64;
pub type UnitId = i64;
pub type ModelId = i64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Project {
    pub id: ProjectId,
    pub label: String,
    /// Provider-defined project locator. The historical name is retained for API
    /// compatibility; values such as `memory://project` are valid.
    pub source_dir: String,
    pub role: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRecord {
    pub id: FileId,
    pub project_id: ProjectId,
    pub source_document_id: String,
    pub source_revision: String,
    pub relative_path: String,
    pub language_id: String,
    pub mtime_ns: i64,
    pub size: i64,
    pub source_hash: String,
}

/// A new source document row before it has an id.
#[derive(Debug, Clone)]
pub struct NewFile {
    pub project_id: ProjectId,
    pub source_document_id: String,
    pub source_revision: String,
    pub relative_path: String,
    pub language_id: String,
    pub mtime_ns: i64,
    pub size: i64,
    pub source_hash: String,
}

/// One representation channel to persist for a unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewRepresentation {
    pub kind: RepresentationKind,
    pub content_hash: String,
    pub content: Option<String>,
    pub origin: RepresentationOrigin,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodeUnit {
    pub id: UnitId,
    pub file_id: FileId,
    pub entity_id: EntityId,
    pub entity_version_id: EntityVersionId,
    pub generation: i64,
    pub language_id: String,
    pub kind: String,
    pub name: String,
    pub scope: Option<String>,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub body_node_count: usize,
    pub source_hash: String,
    pub normalized_body_hash: String,
}

/// A new code unit before it has an id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewCodeUnit {
    pub entity_id: EntityId,
    pub entity_version_id: EntityVersionId,
    pub generation: i64,
    pub language_id: String,
    pub kind: String,
    pub name: String,
    pub scope: Option<String>,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub body_node_count: usize,
    pub source_hash: String,
    pub normalized_body_hash: String,
    pub representations: Vec<NewRepresentation>,
}

impl NewCodeUnit {
    pub fn from_entity(
        entity: ExtractedEntity,
        entity_id: impl Into<EntityId>,
        entity_version_id: impl Into<EntityVersionId>,
        generation: i64,
    ) -> Self {
        let representations = entity
            .representations
            .into_iter()
            .map(|repr| NewRepresentation {
                kind: repr.kind,
                content_hash: repr.content_hash,
                content: Some(repr.content),
                origin: repr.origin,
            })
            .collect();
        NewCodeUnit {
            entity_id: entity_id.into(),
            entity_version_id: entity_version_id.into(),
            generation,
            language_id: entity.language.into_inner(),
            kind: entity.kind.as_str().to_owned(),
            name: entity.name,
            scope: entity.scope,
            start_byte: entity.span.start_byte,
            end_byte: entity.span.end_byte,
            start_line: entity.span.start_line,
            end_line: entity.span.end_line,
            body_node_count: entity.body_node_count,
            source_hash: entity.source_hash,
            normalized_body_hash: entity.normalized_body_hash,
            representations,
        }
    }

    pub fn content_hash(&self, kind: &RepresentationKind) -> Option<&str> {
        self.representations
            .iter()
            .find(|repr| &repr.kind == kind)
            .map(|repr| repr.content_hash.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingModelRecord {
    pub id: ModelId,
    pub identity: ModelIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingSpaceRecord {
    pub identity: EmbeddingSpaceIdentity,
    pub model_id: ModelId,
}

impl EmbeddingSpaceRecord {
    pub fn id(&self) -> &EmbeddingSpaceId {
        &self.identity.id
    }
}

/// Encode a vector as a little-endian f32 blob.
pub fn vector_to_blob(vector: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    blob
}

/// Decode a little-endian f32 blob back into a vector.
pub fn blob_to_vector(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_blob_roundtrip() {
        let vector = vec![0.25_f32, -1.5, 3.75, f32::MIN_POSITIVE];
        assert_eq!(blob_to_vector(&vector_to_blob(&vector)), vector);
    }
}