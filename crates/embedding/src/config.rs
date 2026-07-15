use std::path::PathBuf;

/// Fastembed catalog models supported by the local backend, together with
/// output dimensions and whether a quantized variant is available.
pub const SUPPORTED_MODELS: &[(&str, usize, bool)] = &[
    ("BGESmallENV15", 384, true),
    ("BGEBaseENV15", 768, true),
    ("JinaEmbeddingsV2BaseCode", 768, false),
    ("AllMiniLML6V2", 384, true),
    ("GTEBaseENV15", 768, true),
    ("SnowflakeArcticEmbedM", 768, true),
    ("SnowflakeArcticEmbedMLong", 768, true),
    ("NomicEmbedTextV15", 768, true),
];

/// Execution-provider names understood by the local ONNX backend.
pub const EXECUTION_PROVIDERS: &[&str] = &["cpu", "cuda", "coreml", "directml", "openvino"];

/// A model materialized and hash-verified by the embedding crate instead of
/// delegated to fastembed's built-in catalog.
#[derive(Debug, Clone, Copy)]
pub struct ManagedModel {
    pub name: &'static str,
    pub cache_id: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub onnx_file: &'static str,
    pub dimensions: usize,
    pub pooling: &'static str,
    pub max_length: usize,
    /// Literal prefix rendered before query text (`""` = none).
    pub query_prefix: &'static str,
    /// Literal prefix rendered before document text (`""` = none).
    pub document_prefix: &'static str,
    pub files: &'static [ManagedFile],
}

#[derive(Debug, Clone, Copy)]
pub struct ManagedFile {
    pub path: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

pub const MANAGED_MODELS: &[ManagedModel] = &[ManagedModel {
    name: "CodeRankEmbed",
    cache_id: "coderankembed",
    repo: "Zenabius/CodeRankEmbed-onnx",
    revision: "main",
    onnx_file: "onnx/model.onnx",
    dimensions: 768,
    pooling: "mean",
    max_length: 2048,
    // CodeRankEmbed is asymmetric: queries require this exact prefix
    // (see nomic-ai/CodeRankEmbed model card); documents embed raw.
    query_prefix: "Represent this query for searching relevant code: ",
    document_prefix: "",
    files: &[
        ManagedFile {
            path: "onnx/model.onnx",
            sha256: "87edaf9f6d544e9d46ed81e1e13610ac01b1c1904e3b26fcf1ce6744a0319ffa",
            size: 548_260_181,
        },
        ManagedFile {
            path: "tokenizer.json",
            sha256: "91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854",
            size: 711_649,
        },
        ManagedFile {
            path: "config.json",
            sha256: "5ff856a41d0f53ef2d74520627d464bd75c2efd8f26f381bd528654895c29b6c",
            size: 1_525,
        },
        ManagedFile {
            path: "special_tokens_map.json",
            sha256: "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
            size: 695,
        },
        ManagedFile {
            path: "tokenizer_config.json",
            sha256: "7809f768ee3614618b3f1b91dcbfab4f6a9d4b79fb1ad5d17feb65a7c1bb5b7a",
            size: 1_417,
        },
    ],
}];

pub fn managed_model(name: &str) -> Option<&'static ManagedModel> {
    MANAGED_MODELS.iter().find(|model| model.name == name)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderMode {
    Require,
    Auto,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomModelConfig {
    pub dir: PathBuf,
    pub onnx_file: PathBuf,
    pub dimensions: usize,
    pub pooling: String,
    pub max_length: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingConfig {
    pub backend: String,
    pub model: String,
    pub cache_dir: Option<PathBuf>,
    pub batch_size: usize,
    pub max_batch_chars: usize,
    pub max_batch_token_area: usize,
    pub max_body_chars: usize,
    pub pending_page_size: usize,
    pub normalize: bool,
    pub execution_provider: String,
    pub provider_mode: ProviderMode,
    pub quantized: bool,
    pub custom: Option<CustomModelConfig>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            backend: "fastembed".into(),
            // Deliberately empty: there is no default model. Callers must
            // pick an explicit reference (`hf:owner/name`, `dir:/path`,
            // `fastembed:Name`); an empty value fails with guidance.
            model: String::new(),
            cache_dir: None,
            batch_size: 256,
            max_batch_chars: 200_000,
            max_batch_token_area: 16_000_000,
            max_body_chars: 10_000,
            pending_page_size: 512,
            normalize: true,
            execution_provider: "cpu".into(),
            provider_mode: ProviderMode::Require,
            quantized: false,
            custom: None,
        }
    }
}

impl EmbeddingConfig {
    pub fn dimensions(&self) -> usize {
        if let Some(custom) = &self.custom {
            return custom.dimensions;
        }
        if let Some(managed) = managed_model(&self.model) {
            return managed.dimensions;
        }
        SUPPORTED_MODELS
            .iter()
            .find(|(name, _, _)| *name == self.model)
            .map(|(_, dimensions, _)| *dimensions)
            .unwrap_or(0)
    }
}

/// Extraction settings needed only when retained embedding text must be
/// reconstructed from the source corpus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRecoveryConfig {
    pub body_node_count_threshold: usize,
}

/// Complete configuration for an embedding projection or token report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingRunConfig {
    pub embedding: EmbeddingConfig,
    pub source_recovery: SourceRecoveryConfig,
}
