//! Remote embedding execution. Sending source code off-machine is opt-in in
//! two ways: a remote backend must be selected explicitly, and it refuses to
//! start without a [`DataPolicy`] stating what the provider may retain.
//!
//! The work is split in two:
//! - [`RemoteBackend`] is provider-independent: it splits a packer batch into
//!   requests of at most `max_inputs_per_request` inputs, sends `concurrency`
//!   at a time, retries rate limits and transient failures (honouring
//!   `Retry-After`), halves requests the server rejects as too large, checks
//!   widths, normalizes, reassembles input order, and meters usage.
//! - A [`RemoteProvider`] owns everything provider-specific: endpoint, auth
//!   headers, request body (including how the data policy and routing pins
//!   are expressed), response classification, and which settings belong in
//!   the model contract. New providers implement the trait, add a
//!   [`RemoteProviderSpec`] to [`REMOTE_PROVIDERS`], and get the rest free.
//!   OpenAI-compatible APIs can reuse [`openai_compat`].

pub mod openai_compat;
pub mod openrouter;

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};

use crate::config::{DataPolicy, EmbeddingConfig, RemoteModelConfig};
use crate::embed::{
    EmbedRequest, EmbeddingBackend, EmbeddingRole, normalize_in_place, render_inputs,
};
use codeindex_core::{ExecutionInfo, ModelContract, Pooling, PromptContract};

/// Cap on one response body. ureq's 10 MB default is too small: 64 inputs x
/// 4096 floats is ~6 MB of JSON, and larger requests scale linearly.
const MAX_RESPONSE_BYTES: u64 = 512 * 1024 * 1024;
/// Error bodies are quoted into messages up to this many characters.
const ERROR_EXCERPT_CHARS: usize = 400;

/// A registered remote provider: the `embedding.backend` name it answers to,
/// its public defaults, and its constructor.
pub struct RemoteProviderSpec {
    pub id: &'static str,
    pub default_base_url: &'static str,
    pub default_api_key_env: &'static str,
    pub build: fn(ProviderInit<'_>) -> Result<Box<dyn RemoteProvider>>,
}

pub const REMOTE_PROVIDERS: &[RemoteProviderSpec] = &[openrouter::SPEC];

pub fn remote_provider(id: &str) -> Option<&'static RemoteProviderSpec> {
    REMOTE_PROVIDERS.iter().find(|spec| spec.id == id)
}

pub fn is_remote_backend(backend: &str) -> bool {
    remote_provider(backend).is_some()
}

/// Everything a provider needs to build itself, already validated and with
/// defaults applied.
pub struct ProviderInit<'a> {
    /// The provider's own model id, e.g. `qwen/qwen3-embedding-8b`.
    pub model: &'a str,
    pub base_url: String,
    pub api_key: String,
    pub data_policy: DataPolicy,
    pub config: &'a RemoteModelConfig,
}

/// How a provider classified a completed HTTP exchange.
#[derive(Debug)]
pub enum Reply {
    /// `(input index, vector)` pairs, in any order, plus metered usage.
    Embeddings {
        data: Vec<(usize, Vec<f32>)>,
        usage: Usage,
    },
    /// Transient (rate limit, overload, upstream hiccup): retry with backoff.
    Retry(String),
    /// The request was too large: split it.
    TooLarge(String),
    /// Will not succeed on retry (auth, credits, bad model, bad params).
    Fatal(String),
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Usage {
    pub prompt_tokens: u64,
    /// Provider-reported charge, when the API reports one.
    pub cost: Option<f64>,
}

/// The provider-specific half of a remote backend. Implementations are pure
/// request/response translators; transport, retries and batching live in
/// [`RemoteBackend`].
pub trait RemoteProvider: Send + Sync {
    /// Backend name and model-reference prefix, e.g. `openrouter`.
    fn id(&self) -> &'static str;
    /// URL one embeddings request is POSTed to.
    fn endpoint(&self) -> String;
    /// Headers beyond `content-type` (authorization, attribution).
    fn headers(&self) -> Vec<(&'static str, String)>;
    /// JSON body for one request of already-rendered inputs. Must express
    /// the data policy and any routing pins.
    fn request_body(&self, role: EmbeddingRole, inputs: &[&str]) -> serde_json::Value;
    /// Classify a response the server completed, whatever its status.
    fn interpret(&self, status: u16, body: &str) -> Reply;
    /// Settings other than model id and width that change document vectors
    /// (routing pins, quantization, document input type). Becomes the
    /// contract's revision so differently configured runs never share a
    /// space.
    fn revision(&self) -> Option<String>;
    /// Human-readable routing description for execution provenance.
    fn routing(&self) -> String;
    /// One line shown before code is sent: where it goes, under what policy.
    fn disclosure(&self) -> String;
}

/// Build the remote backend named by `config.backend`.
pub fn backend(config: &EmbeddingConfig) -> Result<Box<dyn EmbeddingBackend>> {
    Ok(Box::new(RemoteBackend::new(config)?))
}

pub struct RemoteBackend {
    contract: ModelContract,
    execution: ExecutionInfo,
    provider: Box<dyn RemoteProvider>,
    max_inputs_per_request: usize,
    concurrency: usize,
    max_retries: u32,
    agent: ureq::Agent,
    meter: Mutex<Meter>,
}

#[derive(Debug, Default, Clone, Copy)]
struct Meter {
    requests: u64,
    retries: u64,
    prompt_tokens: u64,
    cost: f64,
    cost_reported: bool,
}

/// One request's outcome, `None` until a worker has run it.
type ChunkResult = Option<Result<Vec<Vec<f32>>>>;

enum Attempt {
    Done(Vec<Vec<f32>>),
    Retry(String, Option<Duration>),
    TooLarge(String),
}

impl RemoteBackend {
    pub fn new(config: &EmbeddingConfig) -> Result<Self> {
        let spec = remote_provider(&config.backend)
            .with_context(|| format!("{:?} is not a remote embedding backend", config.backend))?;
        let remote = config
            .remote
            .as_ref()
            .with_context(|| format!("the {} backend needs remote model settings", spec.id))?;
        let data_policy = remote.data_policy.with_context(|| {
            format!(
                "the {} backend sends every embedded code body to a third party; set a data \
                 policy to say what it may keep: `zdr` (zero data retention, for private \
                 code), `no_training` (no training on inputs; operational logs may remain), \
                 or `provider_default` (public code only)",
                spec.id
            )
        })?;
        let model = config.model.trim();
        anyhow::ensure!(
            !model.is_empty(),
            "no model configured; set the {} model id",
            spec.id
        );
        anyhow::ensure!(
            remote.dimensions > 0,
            "the {} backend needs the model's output width in `dimensions`",
            spec.id
        );
        let api_key_env = remote
            .api_key_env
            .clone()
            .unwrap_or_else(|| spec.default_api_key_env.to_string());
        let api_key = std::env::var(&api_key_env)
            .ok()
            .filter(|key| !key.trim().is_empty())
            .with_context(|| {
                format!(
                    "the {} backend reads its API key from ${api_key_env}, which is not set",
                    spec.id
                )
            })?;
        let base_url = remote
            .base_url
            .clone()
            .unwrap_or_else(|| spec.default_base_url.to_string())
            .trim_end_matches('/')
            .to_string();
        let provider = (spec.build)(ProviderInit {
            model,
            base_url,
            api_key,
            data_policy,
            config: remote,
        })?;
        Ok(Self::with_provider(config, remote, provider))
    }

    /// Assemble a backend around an already-built provider (tests, or
    /// callers constructing providers themselves).
    pub fn with_provider(
        config: &EmbeddingConfig,
        remote: &RemoteModelConfig,
        provider: Box<dyn RemoteProvider>,
    ) -> Self {
        eprintln!("{}", provider.disclosure());
        let prompts = if remote.query_prefix.is_empty() && remote.document_prefix.is_empty() {
            PromptContract::Symmetric
        } else {
            PromptContract::RolePrefixes {
                query: remote.query_prefix.clone(),
                document: remote.document_prefix.clone(),
            }
        };
        let contract = ModelContract {
            model: format!("{}:{}", provider.id(), config.model.trim()),
            revision: provider.revision(),
            model_hash: None,
            tokenizer_hash: None,
            pooling: Pooling::ModelDefined,
            normalize: config.normalize,
            native_dimensions: remote.dimensions,
            max_sequence_length: remote.max_input_tokens,
            prompts,
            quantization: None,
        };
        let execution = ExecutionInfo {
            backend: provider.id().into(),
            backend_version: env!("CARGO_PKG_VERSION").into(),
            runtime_version: None,
            execution_provider: provider.routing(),
            cache_path: Some(provider.endpoint()),
        };
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(remote.timeout_secs.max(1))))
            .build()
            .into();
        Self {
            contract,
            execution,
            provider,
            max_inputs_per_request: remote.max_inputs_per_request.max(1),
            concurrency: remote.concurrency.max(1),
            max_retries: remote.max_retries,
            agent,
            meter: Mutex::new(Meter::default()),
        }
    }

    /// Embed one request's worth of inputs, retrying transient failures and
    /// halving the request when the server says it is too large.
    fn embed_chunk(&self, role: EmbeddingRole, inputs: &[&str]) -> Result<Vec<Vec<f32>>> {
        let body = serde_json::to_string(&self.provider.request_body(role, inputs))?;
        let mut attempt = 0u32;
        loop {
            match self.attempt(&body, inputs.len())? {
                Attempt::Done(vectors) => return Ok(vectors),
                Attempt::TooLarge(reason) => {
                    anyhow::ensure!(
                        inputs.len() > 1,
                        "{} rejected a single input as too large ({reason}); lower \
                         embedding.max_body_chars",
                        self.provider.id()
                    );
                    let (left, right) = inputs.split_at(inputs.len() / 2);
                    let mut out = self.embed_chunk(role, left)?;
                    out.extend(self.embed_chunk(role, right)?);
                    return Ok(out);
                }
                Attempt::Retry(reason, delay) => {
                    if attempt >= self.max_retries {
                        bail!(
                            "{} request failed after {} attempts: {reason}",
                            self.provider.id(),
                            attempt + 1
                        );
                    }
                    let backoff = Duration::from_secs(1 << attempt.min(6));
                    std::thread::sleep(delay.unwrap_or(backoff).min(Duration::from_secs(120)));
                    attempt += 1;
                    self.meter.lock().unwrap().retries += 1;
                }
            }
        }
    }

    fn attempt(&self, body: &str, expected: usize) -> Result<Attempt> {
        let mut request = self
            .agent
            .post(&self.provider.endpoint())
            .header("content-type", "application/json");
        for (name, value) in self.provider.headers() {
            request = request.header(name, &value);
        }
        let mut response = match request.send(body) {
            Ok(response) => response,
            // Connection resets and timeouts are transient on long runs.
            Err(error) => return Ok(Attempt::Retry(format!("{error}"), None)),
        };
        let status = response.status().as_u16();
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.trim().parse::<u64>().ok())
            .map(Duration::from_secs);
        let text = match response
            .body_mut()
            .with_config()
            .limit(MAX_RESPONSE_BYTES)
            .read_to_string()
        {
            Ok(text) => text,
            Err(error) => return Ok(Attempt::Retry(format!("reading response: {error}"), None)),
        };
        self.meter.lock().unwrap().requests += 1;
        match self.provider.interpret(status, &text) {
            Reply::Embeddings { data, usage } => {
                let mut meter = self.meter.lock().unwrap();
                meter.prompt_tokens += usage.prompt_tokens;
                if let Some(cost) = usage.cost {
                    meter.cost += cost;
                    meter.cost_reported = true;
                }
                drop(meter);
                self.collect(data, expected).map(Attempt::Done)
            }
            Reply::Retry(reason) => Ok(Attempt::Retry(reason, retry_after)),
            Reply::TooLarge(reason) => Ok(Attempt::TooLarge(reason)),
            Reply::Fatal(reason) => bail!(
                "{} request for {} failed: {reason}",
                self.provider.id(),
                self.contract.model
            ),
        }
    }

    fn collect(&self, data: Vec<(usize, Vec<f32>)>, expected: usize) -> Result<Vec<Vec<f32>>> {
        let id = self.provider.id();
        anyhow::ensure!(
            data.len() == expected,
            "{id} returned {} embeddings for {expected} inputs",
            data.len()
        );
        let dimensions = self.contract.native_dimensions;
        let mut out: Vec<Option<Vec<f32>>> = vec![None; expected];
        for (index, mut vector) in data {
            let slot = out
                .get_mut(index)
                .with_context(|| format!("{id} returned out-of-range index {index}"))?;
            anyhow::ensure!(
                vector.len() == dimensions,
                "{} returned {}-dimensional vectors but `dimensions` is {dimensions}; fix the \
                 config (or set `request_dimensions` if the model supports truncation)",
                self.contract.model,
                vector.len(),
            );
            if self.contract.normalize {
                normalize_in_place(&mut vector);
            }
            *slot = Some(vector);
        }
        out.into_iter()
            .enumerate()
            .map(|(index, vector)| {
                vector.with_context(|| format!("{id} returned no embedding for input {index}"))
            })
            .collect()
    }
}

impl EmbeddingBackend for RemoteBackend {
    fn contract(&self) -> &ModelContract {
        &self.contract
    }

    fn execution(&self) -> &ExecutionInfo {
        &self.execution
    }

    fn padded_batches(&self) -> bool {
        false
    }

    fn embed(&mut self, request: &EmbedRequest<'_>) -> Result<Vec<Vec<f32>>> {
        let rendered = render_inputs(&self.contract, request)?;
        let inputs: Vec<&str> = rendered.iter().map(|text| text.as_ref()).collect();
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let chunks: Vec<&[&str]> = inputs.chunks(self.max_inputs_per_request).collect();
        let workers = self.concurrency.min(chunks.len());
        if workers == 1 {
            let mut out = Vec::with_capacity(inputs.len());
            for chunk in chunks {
                out.extend(self.embed_chunk(request.role, chunk)?);
            }
            return Ok(out);
        }

        let next = AtomicUsize::new(0);
        let results: Mutex<Vec<ChunkResult>> =
            Mutex::new((0..chunks.len()).map(|_| None).collect());
        let this = &*self;
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(chunk) = chunks.get(index) else {
                            break;
                        };
                        let result = this.embed_chunk(request.role, chunk);
                        let failed = result.is_err();
                        results.lock().unwrap()[index] = Some(result);
                        if failed {
                            // Stop handing out work; in-flight chunks finish.
                            next.store(chunks.len(), Ordering::Relaxed);
                            break;
                        }
                    }
                });
            }
        });
        let mut out = Vec::with_capacity(inputs.len());
        for result in results.into_inner().unwrap().into_iter().flatten() {
            out.extend(result?);
        }
        anyhow::ensure!(
            out.len() == inputs.len(),
            "{} batch incomplete: {} of {} embeddings",
            self.provider.id(),
            out.len(),
            inputs.len()
        );
        Ok(out)
    }

    fn usage_summary(&self) -> Option<String> {
        let meter = *self.meter.lock().unwrap();
        (meter.requests > 0).then(|| {
            let cost = if meter.cost_reported {
                format!(", ${:.4} reported cost", meter.cost)
            } else {
                String::new()
            };
            format!(
                "{}: {} requests ({} retries), {} prompt tokens{cost}",
                self.provider.id(),
                meter.requests,
                meter.retries,
                meter.prompt_tokens
            )
        })
    }
}

/// Statuses worth retrying on any HTTP API.
pub fn is_transient(status: u16) -> bool {
    matches!(status, 408 | 425 | 429 | 500 | 502 | 503 | 504 | 520..=529)
}

/// Bound an error body for inclusion in a message.
pub fn excerpt(text: &str) -> String {
    let mut out: String = text.chars().take(ERROR_EXCERPT_CHARS).collect();
    if text.chars().count() > ERROR_EXCERPT_CHARS {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    /// A provider that never talks to the network; exercises the generic
    /// collection path.
    struct Fake;

    impl RemoteProvider for Fake {
        fn id(&self) -> &'static str {
            "fake"
        }
        fn endpoint(&self) -> String {
            "http://127.0.0.1:9/embeddings".into()
        }
        fn headers(&self) -> Vec<(&'static str, String)> {
            Vec::new()
        }
        fn request_body(&self, _role: EmbeddingRole, inputs: &[&str]) -> Value {
            json!({ "input": inputs })
        }
        fn interpret(&self, _status: u16, _body: &str) -> Reply {
            Reply::Fatal("offline".into())
        }
        fn revision(&self) -> Option<String> {
            None
        }
        fn routing(&self) -> String {
            "fake".into()
        }
        fn disclosure(&self) -> String {
            "fake provider (tests)".into()
        }
    }

    fn fake_backend(dimensions: usize) -> RemoteBackend {
        let remote = RemoteModelConfig {
            dimensions,
            data_policy: Some(DataPolicy::Zdr),
            ..RemoteModelConfig::default()
        };
        let config = EmbeddingConfig {
            backend: "fake".into(),
            model: "m".into(),
            remote: Some(remote.clone()),
            ..EmbeddingConfig::default()
        };
        RemoteBackend::with_provider(&config, &remote, Box::new(Fake))
    }

    #[test]
    fn collection_reorders_normalizes_and_checks_width() {
        let backend = fake_backend(2);
        assert_eq!(backend.contract().model, "fake:m");
        assert!(!backend.padded_batches());
        let vectors = backend
            .collect(vec![(1, vec![0.0, 2.0]), (0, vec![3.0, 4.0])], 2)
            .unwrap();
        assert_eq!(vectors, vec![vec![0.6, 0.8], vec![0.0, 1.0]]);

        let error = backend
            .collect(vec![(0, vec![1.0, 0.0, 0.0])], 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("3-dimensional"), "{error}");
        assert!(backend.collect(vec![(0, vec![1.0, 0.0])], 2).is_err());
        assert!(backend.collect(vec![(5, vec![1.0, 0.0])], 1).is_err());
    }

    #[test]
    fn missing_data_policy_refuses_to_start() {
        let config = EmbeddingConfig {
            backend: "openrouter".into(),
            model: "qwen/qwen3-embedding-8b".into(),
            remote: Some(RemoteModelConfig {
                dimensions: 4096,
                ..RemoteModelConfig::default()
            }),
            ..EmbeddingConfig::default()
        };
        let error = RemoteBackend::new(&config).err().unwrap().to_string();
        assert!(error.contains("data policy"), "{error}");
        assert!(error.contains("zdr"), "{error}");
    }

    #[test]
    fn transient_statuses() {
        for status in [408, 429, 500, 502, 503, 504, 524] {
            assert!(is_transient(status), "{status}");
        }
        for status in [400, 401, 402, 403, 404, 413, 422] {
            assert!(!is_transient(status), "{status}");
        }
    }
}
