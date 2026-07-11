#![forbid(unsafe_code)]

use std::fmt;

use serde::{Deserialize, Serialize};

/// Language identifier independent of any parser implementation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LanguageId(String);

impl LanguageId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl From<&str> for LanguageId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for LanguageId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for LanguageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Logical identity for a source entity across index generations.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EntityId(String);

impl EntityId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl From<&str> for EntityId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for EntityId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Exact identity for one indexed version of a logical entity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EntityVersionId(String);

impl EntityVersionId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl From<&str> for EntityVersionId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for EntityVersionId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for EntityVersionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Stable identifier for one independently queryable embedding space.
///
/// A space binds one representation channel to one exact model identity and an
/// input transform. Human-readable ids such as `code`, `docs`, or
/// `implementation/coderank` are encouraged; stores enforce that an existing id
/// cannot silently change meaning.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EmbeddingSpaceId(String);

impl EmbeddingSpaceId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl From<&str> for EmbeddingSpaceId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for EmbeddingSpaceId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for EmbeddingSpaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Cross-language entity kind. Unknown frontend-specific kinds remain lossless.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum EntityKind {
    Module,
    Namespace,
    Type,
    Trait,
    Interface,
    Function,
    Method,
    Constructor,
    Closure,
    Constant,
    Static,
    Macro,
    Field,
    Test,
    Example,
    Other(String),
}

impl EntityKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Module => "module",
            Self::Namespace => "namespace",
            Self::Type => "type",
            Self::Trait => "trait",
            Self::Interface => "interface",
            Self::Function => "function",
            Self::Method => "method",
            Self::Constructor => "constructor",
            Self::Closure => "closure",
            Self::Constant => "constant",
            Self::Static => "static",
            Self::Macro => "macro",
            Self::Field => "field",
            Self::Test => "test",
            Self::Example => "example",
            Self::Other(value) => value,
        }
    }
}

impl From<&str> for EntityKind {
    fn from(value: &str) -> Self {
        match value {
            "module" => Self::Module,
            "namespace" => Self::Namespace,
            "type" => Self::Type,
            "trait" => Self::Trait,
            "interface" => Self::Interface,
            "function" => Self::Function,
            "method" => Self::Method,
            "constructor" => Self::Constructor,
            "closure" => Self::Closure,
            "constant" => Self::Constant,
            "static" => Self::Static,
            "macro" => Self::Macro,
            "field" => Self::Field,
            "test" => Self::Test,
            "example" => Self::Example,
            other => Self::Other(other.to_string()),
        }
    }
}

impl From<String> for EntityKind {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}

/// Half-open source range plus one-based human line numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceSpan {
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
}

impl SourceSpan {
    pub fn new(start_byte: usize, end_byte: usize, start_line: usize, end_line: usize) -> Self {
        debug_assert!(start_byte <= end_byte);
        debug_assert!(start_line <= end_line);
        Self {
            start_byte,
            end_byte,
            start_line,
            end_line,
        }
    }
}

/// A reproducible textual projection of a source entity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum RepresentationKind {
    FullSource,
    Implementation,
    Body,
    BodyWithoutDeclaredName,
    Signature,
    Symbol,
    Documentation,
    Usage,
    GeneratedDescription,
    Custom(String),
}

impl RepresentationKind {
    /// The canonical persisted/serialized token for this channel.
    pub fn as_str(&self) -> &str {
        match self {
            Self::FullSource => "full_source",
            Self::Implementation => "implementation",
            Self::Body => "body",
            Self::BodyWithoutDeclaredName => "body_without_declared_name",
            Self::Signature => "signature",
            Self::Symbol => "symbol",
            Self::Documentation => "documentation",
            Self::Usage => "usage",
            Self::GeneratedDescription => "generated_description",
            Self::Custom(value) => value,
        }
    }
}

impl From<&str> for RepresentationKind {
    fn from(value: &str) -> Self {
        match value {
            "full_source" => Self::FullSource,
            "implementation" => Self::Implementation,
            "body" => Self::Body,
            "body_without_declared_name" => Self::BodyWithoutDeclaredName,
            "signature" => Self::Signature,
            "symbol" => Self::Symbol,
            "documentation" => Self::Documentation,
            "usage" => Self::Usage,
            "generated_description" => Self::GeneratedDescription,
            other => Self::Custom(other.to_string()),
        }
    }
}

impl From<String> for RepresentationKind {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}

impl fmt::Display for RepresentationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for RepresentationKind {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for RepresentationKind {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let token = String::deserialize(deserializer)?;
        Ok(Self::from(token.as_str()))
    }
}

/// Provenance for a representation channel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RepresentationOrigin {
    /// Deterministically extracted from source by a language frontend.
    Extracted { frontend: String },
    /// Synthesized from other indexed facts, such as call sites.
    Derived { producer: String, version: String },
    /// Supplied by a consumer, such as an LLM-generated description.
    Imported { producer: String, version: String },
}

impl Default for RepresentationOrigin {
    fn default() -> Self {
        Self::Extracted {
            frontend: "tree-sitter".to_string(),
        }
    }
}

/// Text, identity, and provenance for one representation channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Representation {
    pub kind: RepresentationKind,
    pub content: String,
    pub content_hash: String,
    pub origin: RepresentationOrigin,
}

impl Representation {
    pub fn new(
        kind: RepresentationKind,
        content: impl Into<String>,
        content_hash: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            content: content.into(),
            content_hash: content_hash.into(),
            origin: RepresentationOrigin::default(),
        }
    }

    pub fn with_origin(mut self, origin: RepresentationOrigin) -> Self {
        self.origin = origin;
        self
    }
}

/// Parser-neutral entity emitted by a language frontend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedEntity {
    pub language: LanguageId,
    pub kind: EntityKind,
    pub name: String,
    pub scope: Option<String>,
    pub span: SourceSpan,
    pub body_span: Option<SourceSpan>,
    pub body_node_count: usize,
    pub source_hash: String,
    pub normalized_body_hash: String,
    pub representations: Vec<Representation>,
}

impl ExtractedEntity {
    pub fn representation(&self, kind: &RepresentationKind) -> Option<&Representation> {
        self.representations.iter().find(|item| &item.kind == kind)
    }

    pub fn representation_text(&self, kind: &RepresentationKind) -> Option<&str> {
        self.representation(kind).map(|item| item.content.as_str())
    }
}

/// One non-fatal frontend diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexDiagnostic {
    pub message: String,
    pub span: Option<SourceSpan>,
}

/// Complete parser-neutral result for one source file.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExtractedFile {
    pub entities: Vec<ExtractedEntity>,
    pub diagnostics: Vec<IndexDiagnostic>,
}

/// Everything that identifies an embedding model for reproducible runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub backend: String,
    pub backend_version: String,
    /// ONNX Runtime / `ort` version when applicable.
    pub runtime_version: Option<String>,
    pub model: String,
    pub revision: Option<String>,
    pub dimensions: usize,
    pub tokenizer_hash: Option<String>,
    pub model_hash: Option<String>,
    pub normalize: bool,
    pub execution_provider: String,
    pub quantization: Option<String>,
    pub cache_path: Option<String>,
}

/// Immutable meaning of one embedding space.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingSpaceIdentity {
    pub id: EmbeddingSpaceId,
    pub channel: RepresentationKind,
    pub model: ModelIdentity,
    /// Stable name for preprocessing applied before embedding. `identity` means
    /// the stored representation text is embedded verbatim.
    pub input_transform: String,
}

impl EmbeddingSpaceIdentity {
    pub fn new(
        id: impl Into<EmbeddingSpaceId>,
        channel: RepresentationKind,
        model: ModelIdentity,
    ) -> Self {
        Self {
            id: id.into(),
            channel,
            model,
            input_transform: "identity".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_entity_kinds_round_trip() {
        let kind = EntityKind::from("receiver-function");
        assert_eq!(kind.as_str(), "receiver-function");
    }

    #[test]
    fn representation_lookup_is_channel_specific() {
        let entity = ExtractedEntity {
            language: LanguageId::from("rust"),
            kind: EntityKind::Function,
            name: "parse".into(),
            scope: None,
            span: SourceSpan::new(0, 10, 1, 1),
            body_span: None,
            body_node_count: 2,
            source_hash: "source".into(),
            normalized_body_hash: "body".into(),
            representations: vec![Representation::new(
                RepresentationKind::Implementation,
                "fn parse() {}",
                "hash",
            )],
        };

        assert_eq!(
            entity.representation_text(&RepresentationKind::Implementation),
            Some("fn parse() {}")
        );
        assert_eq!(
            entity.representation_text(&RepresentationKind::Documentation),
            None
        );
    }

    #[test]
    fn embedding_space_ids_are_serializable_map_keys() {
        let id = EmbeddingSpaceId::new("docs");
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"docs\"");
        assert_eq!(serde_json::from_str::<EmbeddingSpaceId>(&json).unwrap(), id);
    }
}
