//! Remote embedding execution. Sending source code off-machine is opt-in in
//! two ways: a remote backend must be selected explicitly, and it refuses to
//! start without a [`DataPolicy`] stating what the provider may retain.
//!
//! The work is split in two:
//! - [`RemoteBackend`] is provider-independent: it splits a packer batch into
//!   requests bounded by `max_inputs_per_request` and `max_request_chars`,
//!   runs them on up to `concurrency` workers behind a shared
//!   [`throttle::RateGate`] (a 429 pauses every worker and, when adaptive,
//!   halves the in-flight limit), retries other transient failures with
//!   jittered exponential backoff, halves requests the server rejects as too
//!   large, checks widths, normalizes, reassembles input order, and meters
//!   usage. It sizes the indexer's batches (`batch_limits`) to keep every
//!   request slot busy.
//! - A [`RemoteProvider`] owns everything provider-specific: endpoint, auth
//!   headers, request body (including how the data policy and routing pins
//!   are expressed), response classification, and which settings belong in
//!   the model contract. New providers implement the trait, add a
//!   [`RemoteProviderSpec`] to [`REMOTE_PROVIDERS`], and get the rest free.
//!   OpenAI-compatible APIs can reuse [`openai_compat`].

pub mod openai_compat;
pub mod openrouter;
pub mod throttle;

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use throttle::{Backoff, RateGate, parse_retry_after};

use anyhow::{Context, Result, bail};

use crate::config::{DataPolicy, EmbeddingConfig, RemoteModelConfig};
use crate::embed::{
    BatchLimits, EmbedRequest, EmbeddingBackend, EmbeddingRole, normalize_in_place, render_inputs,
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
    /// HTTP 429 or equivalent: the account/upstream is over its limit.
    /// Pauses every worker, not just this request.
    RateLimited(String),
    /// Other transient failure (overload, upstream hiccup): retry this
    /// request with backoff.
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
    max_request_chars: usize,
    concurrency: usize,
    max_retries: u32,
    backoff: Backoff,
    rate_limit_max_wait: Duration,
    gate: RateGate,
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
    RateLimited(String, Option<Duration>),
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
        let backoff = Backoff {
            initial: Duration::from_millis(remote.backoff_initial_ms.max(1)),
            max: Duration::from_millis(remote.backoff_max_ms.max(remote.backoff_initial_ms)),
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
            max_request_chars: remote.max_request_chars.max(1),
            concurrency: remote.concurrency.max(1),
            max_retries: remote.max_retries,
            backoff,
            rate_limit_max_wait: Duration::from_secs(remote.rate_limit_max_wait_secs),
            gate: RateGate::new(remote.concurrency, remote.adaptive_concurrency, backoff),
            agent,
            meter: Mutex::new(Meter::default()),
        }
    }

    /// Embed one request's worth of inputs. Each send holds a gate permit;
    /// rate limits are reported to the gate (which pauses every worker) and
    /// bounded by total time spent limited, other transient failures retry
    /// with jittered backoff up to `max_retries`, and a 413 halves the
    /// request.
    fn embed_chunk(&self, role: EmbeddingRole, inputs: &[&str]) -> Result<Vec<Vec<f32>>> {
        let body = serde_json::to_string(&self.provider.request_body(role, inputs))?;
        let mut retries = 0u32;
        let mut limited_since: Option<Instant> = None;
        loop {
            let permit = self.gate.acquire();
            let attempt = self.attempt(&body, inputs.len())?;
            match attempt {
                Attempt::Done(vectors) => {
                    permit.success();
                    return Ok(vectors);
                }
                Attempt::RateLimited(reason, retry_after) => {
                    permit.rate_limited(retry_after);
                    let since = *limited_since.get_or_insert_with(Instant::now);
                    if since.elapsed() >= self.rate_limit_max_wait {
                        bail!(
                            "{} kept rate limiting this request for {}s (last: {reason}); \
                             lower `concurrency` or raise `rate_limit_max_wait_secs`",
                            self.provider.id(),
                            since.elapsed().as_secs()
                        );
                    }
                    self.meter.lock().unwrap().retries += 1;
                    // The wait happens in the next `acquire`, shared by all.
                }
                Attempt::TooLarge(reason) => {
                    drop(permit);
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
                Attempt::Retry(reason, retry_after) => {
                    drop(permit);
                    if retries >= self.max_retries {
                        bail!(
                            "{} request failed after {} attempts: {reason}",
                            self.provider.id(),
                            retries + 1
                        );
                    }
                    let delay = retry_after
                        .unwrap_or_else(|| self.backoff.delay(retries))
                        .min(self.backoff.max.max(Duration::from_secs(1)));
                    std::thread::sleep(delay);
                    retries += 1;
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
            .and_then(parse_retry_after);
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
            Reply::RateLimited(reason) => Ok(Attempt::RateLimited(reason, retry_after)),
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

    fn batch_limits(&self) -> Option<BatchLimits> {
        // Two waves of full requests per batch: work-stealing absorbs one
        // slow or backing-off request before the batch barrier.
        let waves = self.concurrency * 2;
        Some(BatchLimits {
            items: self.max_inputs_per_request * waves,
            chars: self.max_request_chars * waves,
        })
    }

    fn embed(&mut self, request: &EmbedRequest<'_>) -> Result<Vec<Vec<f32>>> {
        let rendered = render_inputs(&self.contract, request)?;
        let inputs: Vec<&str> = rendered.iter().map(|text| text.as_ref()).collect();
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let chunks = split_requests(&inputs, self.max_inputs_per_request, self.max_request_chars);
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
        let (rate_limited, floor) = self.gate.summary();
        (meter.requests > 0).then(|| {
            let cost = if meter.cost_reported {
                format!(", ${:.4} reported cost", meter.cost)
            } else {
                String::new()
            };
            let limits = if rate_limited > 0 {
                format!(
                    ", {rate_limited} rate-limited (concurrency fell to {floor}/{})",
                    self.concurrency
                )
            } else {
                String::new()
            };
            format!(
                "{}: {} requests ({} retries{limits}), {} prompt tokens{cost}",
                self.provider.id(),
                meter.requests,
                meter.retries,
                meter.prompt_tokens
            )
        })
    }
}

/// Split rendered inputs into contiguous requests of at most `max_items`
/// inputs and `max_chars` summed characters (a single oversized input still
/// gets its own request).
fn split_requests<'a, 'b>(
    inputs: &'b [&'a str],
    max_items: usize,
    max_chars: usize,
) -> Vec<&'b [&'a str]> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut chars = 0;
    for (index, input) in inputs.iter().enumerate() {
        let len = input.len();
        if index > start && (index - start >= max_items || chars + len > max_chars) {
            chunks.push(&inputs[start..index]);
            start = index;
            chars = 0;
        }
        chars += len;
    }
    if start < inputs.len() {
        chunks.push(&inputs[start..]);
    }
    chunks
}

/// Statuses worth retrying on any HTTP API (429 is classified separately as
/// a rate limit by providers).
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
    struct Fake {
        endpoint: String,
    }

    impl RemoteProvider for Fake {
        fn id(&self) -> &'static str {
            "fake"
        }
        fn endpoint(&self) -> String {
            self.endpoint.clone()
        }
        fn headers(&self) -> Vec<(&'static str, String)> {
            Vec::new()
        }
        fn request_body(&self, _role: EmbeddingRole, inputs: &[&str]) -> Value {
            json!({ "input": inputs })
        }
        fn interpret(&self, status: u16, body: &str) -> Reply {
            openai_compat::interpret(status, body)
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
        backend_at("http://127.0.0.1:9/embeddings", remote)
    }

    fn backend_at(endpoint: &str, remote: RemoteModelConfig) -> RemoteBackend {
        let config = EmbeddingConfig {
            backend: "fake".into(),
            model: "m".into(),
            remote: Some(remote.clone()),
            ..EmbeddingConfig::default()
        };
        let provider = Fake {
            endpoint: endpoint.into(),
        };
        RemoteBackend::with_provider(&config, &remote, Box::new(provider))
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
    fn requests_split_by_items_and_chars() {
        let inputs = ["aaaa", "bb", "cc", "dddddddd", "e"];
        let sizes = |chunks: Vec<&[&str]>| chunks.iter().map(|c| c.len()).collect::<Vec<_>>();
        assert_eq!(sizes(split_requests(&inputs, 2, 1000)), vec![2, 2, 1]);
        // 4+2 fits in 6; "cc" would make 8; the 8-char input stands alone.
        assert_eq!(sizes(split_requests(&inputs, 64, 6)), vec![2, 1, 1, 1]);
        assert!(split_requests(&[], 4, 4).is_empty());
    }

    #[test]
    fn batch_limits_cover_two_waves_of_requests() {
        let backend = fake_backend(2);
        assert_eq!(
            backend.batch_limits(),
            Some(BatchLimits {
                items: 64 * 16,
                chars: 100_000 * 16,
            })
        );
    }

    /// A one-thread-per-connection HTTP server: the first `limited`
    /// requests get 429 (`Retry-After: 0`), the rest get one `[1, 0]`
    /// embedding per input, tagged by index. Returns its URL and a request
    /// counter.
    fn mock_server(limited: usize) -> (String, std::sync::Arc<AtomicUsize>) {
        use std::io::{BufRead, BufReader, Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/embeddings", listener.local_addr().unwrap());
        let served = std::sync::Arc::new(AtomicUsize::new(0));
        let counter = served.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let counter = counter.clone();
                std::thread::spawn(move || {
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut length = 0;
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or(0) == 0 {
                            return;
                        }
                        if line == "\r\n" {
                            break;
                        }
                        if let Some((name, value)) = line.split_once(':')
                            && name.eq_ignore_ascii_case("content-length")
                        {
                            length = value.trim().parse().unwrap();
                        }
                    }
                    let mut body = vec![0; length];
                    reader.read_exact(&mut body).unwrap();
                    let request: Value = serde_json::from_slice(&body).unwrap();
                    let inputs = request["input"].as_array().unwrap().len();
                    let response = if counter.fetch_add(1, Ordering::SeqCst) < limited {
                        let body = r#"{"error":{"code":429,"message":"slow down"}}"#;
                        format!(
                            "HTTP/1.1 429 Too Many Requests\r\nretry-after: 0\r\n\
                             content-length: {}\r\nconnection: close\r\n\r\n{body}",
                            body.len()
                        )
                    } else {
                        let data: Vec<Value> = (0..inputs)
                            .map(|index| json!({"embedding": [1.0, 0.0], "index": index}))
                            .collect();
                        let body =
                            json!({"data": data, "usage": {"prompt_tokens": inputs}}).to_string();
                        format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                             content-length: {}\r\nconnection: close\r\n\r\n{body}",
                            body.len()
                        )
                    };
                    let _ = stream.write_all(response.as_bytes());
                });
            }
        });
        (url, served)
    }

    #[test]
    fn rate_limited_requests_back_off_together_and_complete() {
        let (url, served) = mock_server(3);
        let mut backend = backend_at(
            &url,
            RemoteModelConfig {
                dimensions: 2,
                data_policy: Some(DataPolicy::Zdr),
                max_inputs_per_request: 4,
                concurrency: 4,
                backoff_initial_ms: 1,
                backoff_max_ms: 5,
                ..RemoteModelConfig::default()
            },
        );
        let texts: Vec<String> = (0..40).map(|index| format!("fn f{index}() {{}}")).collect();
        let inputs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let vectors = backend
            .embed(&EmbedRequest::documents(&inputs, None))
            .unwrap();
        assert_eq!(vectors.len(), 40);
        assert!(vectors.iter().all(|vector| vector == &vec![1.0, 0.0]));
        // 10 requests succeeded after 3 rate-limited ones.
        assert_eq!(served.load(Ordering::SeqCst), 13);
        let (rate_limited, floor) = backend.gate.summary();
        assert_eq!(rate_limited, 3);
        assert!(floor < 4, "concurrency never adapted (floor {floor})");
        let summary = backend.usage_summary().unwrap();
        assert!(summary.contains("3 rate-limited"), "{summary}");
        assert!(summary.contains("40 prompt tokens"), "{summary}");
    }

    #[test]
    fn sustained_rate_limiting_gives_up_after_the_wait_budget() {
        let (url, _) = mock_server(usize::MAX);
        let mut backend = backend_at(
            &url,
            RemoteModelConfig {
                dimensions: 2,
                data_policy: Some(DataPolicy::Zdr),
                rate_limit_max_wait_secs: 0,
                ..RemoteModelConfig::default()
            },
        );
        let error = backend
            .embed(&EmbedRequest::documents(&["fn a() {}"], None))
            .unwrap_err()
            .to_string();
        assert!(error.contains("kept rate limiting"), "{error}");
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
