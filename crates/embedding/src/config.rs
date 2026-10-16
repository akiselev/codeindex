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

/// What a remote embedding provider may do with the code it receives.
/// Deliberately has no default: sending source off-machine is an explicit
/// choice, and the retention terms are part of that choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataPolicy {
    /// Zero data retention: route only to endpoints that keep neither
    /// prompts nor outputs, and never to ones that train on them.
    Zdr,
    /// Never route to endpoints that may store inputs for training; normal
    /// operational retention (abuse logs) may still apply.
    NoTraining,
    /// The provider's and account's defaults apply as-is. For public code.
    ProviderDefault,
}

impl DataPolicy {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Zdr => "zdr",
            Self::NoTraining => "no_training",
            Self::ProviderDefault => "provider_default",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "zdr" => Some(Self::Zdr),
            "no_training" => Some(Self::NoTraining),
            "provider_default" => Some(Self::ProviderDefault),
            _ => None,
        }
    }
}

/// A remotely hosted model reached through one of the
/// [`remote`](crate::embed::remote) providers, selected by
/// `EmbeddingConfig::backend`. Nothing here is secret: the API key is read
/// from the environment variable named by `api_key_env` at call time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteModelConfig {
    /// Required; the backend refuses to start without one.
    pub data_policy: Option<DataPolicy>,
    /// Endpoint root; `None` uses the provider's public API.
    pub base_url: Option<String>,
    /// `None` uses the provider's conventional variable
    /// (`OPENROUTER_API_KEY`).
    pub api_key_env: Option<String>,
    /// Expected output width; every response is checked against it.
    pub dimensions: usize,
    /// Send `dimensions` in the request (server-side Matryoshka truncation).
    /// Only for models whose API honours the parameter.
    pub request_dimensions: bool,
    /// Context length the remote model accepts. Inputs are not tokenized
    /// locally; `max_body_chars` keeps bodies under it.
    pub max_input_tokens: usize,
    /// Upstream providers to route to, in order, for aggregators that route
    /// (OpenRouter). Non-empty disables fallback routing, so one space never
    /// mixes providers' numerics. Recorded in the model contract.
    pub providers: Vec<String>,
    /// Upstream quantizations to accept (e.g. `bf16`, `fp8`). Non-empty
    /// keeps routing off endpoints serving the same model at lower precision.
    pub quantizations: Vec<String>,
    /// Provider-specific `input_type` per side (Voyage: `document`/`query`).
    /// Changes vectors, so the document side is part of the contract.
    pub document_input_type: Option<String>,
    pub query_input_type: Option<String>,
    pub max_inputs_per_request: usize,
    /// Cap on the summed input characters of one request, so long bodies
    /// split into more requests instead of one oversized one.
    pub max_request_chars: usize,
    /// Ceiling on requests in flight at once.
    pub concurrency: usize,
    /// Halve the in-flight limit on each rate limit and grow it back by one
    /// per `limit` successes. `false` holds `concurrency` fixed (429s still
    /// pause every worker).
    pub adaptive_concurrency: bool,
    /// Jittered exponential backoff for transient failures, and for 429s
    /// that carry no `Retry-After`.
    pub backoff_initial_ms: u64,
    pub backoff_max_ms: u64,
    /// Give up once one request has been rate limited this long in total.
    pub rate_limit_max_wait_secs: u64,
    pub query_prefix: String,
    pub document_prefix: String,
    pub timeout_secs: u64,
    pub max_retries: u32,
}

impl Default for RemoteModelConfig {
    fn default() -> Self {
        Self {
            data_policy: None,
            base_url: None,
            api_key_env: None,
            dimensions: 0,
            request_dimensions: false,
            max_input_tokens: 8192,
            providers: Vec::new(),
            quantizations: Vec::new(),
            document_input_type: None,
            query_input_type: None,
            max_inputs_per_request: 64,
            max_request_chars: 100_000,
            concurrency: 8,
            adaptive_concurrency: true,
            backoff_initial_ms: 1_000,
            backoff_max_ms: 60_000,
            rate_limit_max_wait_secs: 900,
            query_prefix: String::new(),
            document_prefix: String::new(),
            timeout_secs: 120,
            max_retries: 6,
        }
    }
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
    /// Required by remote backends (`openrouter`); ignored by local ones.
    pub remote: Option<RemoteModelConfig>,
    /// Pause the embed loop (releasing the model) while available system
    /// memory is low, instead of risking the OOM killer. `None` disables it.
    pub memory_guard: Option<crate::memory::MemoryGuard>,
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
            remote: None,
            memory_guard: None,
        }
    }
}

impl EmbeddingConfig {
    pub fn dimensions(&self) -> usize {
        if crate::embed::remote::is_remote_backend(&self.backend) {
            return self.remote.as_ref().map_or(0, |remote| remote.dimensions);
        }
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
