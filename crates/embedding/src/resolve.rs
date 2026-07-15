//! Generic model resolution: turn a model reference (`hf:org/name[@rev]`,
//! `dir:/path`, `fastembed:Name`) into a [`ModelContract`] by reading the
//! model's own machine-readable configuration — the sentence-transformers
//! trio (`modules.json`, `1_Pooling/config.json`,
//! `config_sentence_transformers.json`) plus `config.json` and
//! `tokenizer_config.json` — with an optional `codeindex.toml` override for
//! repos that lack them. Downloads are verified trust-on-first-use through a
//! lockfile recording per-file SHA-256 hashes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::Digest;

use codeindex_core::{ModelContract, Pooling, PromptContract};

/// A parsed model reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRef {
    /// `hf:owner/name[@revision]`, or a bare `owner/name` convenience form.
    HuggingFace {
        repo: String,
        revision: Option<String>,
    },
    /// `dir:/path/to/model` — a local directory laid out like an HF repo.
    Directory(PathBuf),
    /// `fastembed:Name` or a bare fastembed catalog / managed model name.
    Named(String),
}

impl ModelRef {
    pub fn parse(value: &str) -> Result<ModelRef> {
        let value = value.trim();
        if value.is_empty() {
            bail!(
                "no model configured; pass a model reference such as \
                 `hf:Qwen/Qwen3-Embedding-0.6B`, `dir:/path/to/model`, or \
                 `fastembed:BGESmallENV15`"
            );
        }
        if let Some(rest) = value.strip_prefix("hf:") {
            let (repo, revision) = match rest.split_once('@') {
                Some((repo, revision)) => (repo, Some(revision.to_string())),
                None => (rest, None),
            };
            anyhow::ensure!(
                repo.split('/').count() == 2 && !repo.starts_with('/') && !repo.ends_with('/'),
                "malformed HuggingFace reference {value:?}: expected hf:owner/name[@revision]"
            );
            return Ok(ModelRef::HuggingFace {
                repo: repo.to_string(),
                revision,
            });
        }
        if let Some(rest) = value.strip_prefix("dir:") {
            return Ok(ModelRef::Directory(PathBuf::from(rest)));
        }
        if let Some(rest) = value.strip_prefix("fastembed:") {
            return Ok(ModelRef::Named(rest.to_string()));
        }
        // Bare `owner/name` is HuggingFace shorthand; anything else is a
        // catalog/managed name resolved by the backend.
        if value.contains('/') {
            return ModelRef::parse(&format!("hf:{value}"));
        }
        Ok(ModelRef::Named(value.to_string()))
    }
}

/// Fetches one file of a model repository. Abstracted so resolution logic is
/// testable without the network.
pub trait RepoFetcher {
    /// `Ok(None)` when the file does not exist in the repository.
    fn fetch(&self, path: &str) -> Result<Option<Vec<u8>>>;
    /// At most the first `limit` bytes of a file — used to preflight large
    /// weight artifacts without downloading them. The default falls back to a
    /// full fetch; network fetchers should implement a bounded read.
    fn fetch_prefix(&self, path: &str, limit: usize) -> Result<Option<Vec<u8>>> {
        Ok(self.fetch(path)?.map(|mut bytes| {
            bytes.truncate(limit);
            bytes
        }))
    }
    /// Human-readable origin for error messages and the lockfile.
    fn origin(&self) -> String;
}

/// Fetches from huggingface.co over HTTPS. Honors `HF_TOKEN` for gated repos.
pub struct HuggingFaceFetcher {
    pub repo: String,
    pub revision: String,
}

impl HuggingFaceFetcher {
    fn get(&self, path: &str, range: Option<usize>) -> Result<Option<Vec<u8>>> {
        let url = format!(
            "https://huggingface.co/{}/resolve/{}/{path}",
            self.repo, self.revision
        );
        let mut request = ureq::get(&url);
        if let Ok(token) = std::env::var("HF_TOKEN") {
            request = request.header("authorization", &format!("Bearer {token}"));
        }
        if let Some(limit) = range {
            request = request.header("range", &format!("bytes=0-{}", limit.saturating_sub(1)));
        }
        match request.call() {
            Ok(response) => {
                // Bound the read even when the server ignores the Range
                // header, so a preflight can never turn into a full download.
                let reader = response.into_body().into_reader();
                let mut bytes = Vec::new();
                match range {
                    Some(limit) => std::io::Read::read_to_end(
                        &mut std::io::Read::take(reader, limit as u64),
                        &mut bytes,
                    ),
                    None => std::io::Read::read_to_end(&mut { reader }, &mut bytes),
                }
                .with_context(|| format!("reading {url}"))?;
                Ok(Some(bytes))
            }
            Err(ureq::Error::StatusCode(404)) => Ok(None),
            Err(error) => Err(error).with_context(|| format!("fetching {url}")),
        }
    }
}

impl RepoFetcher for HuggingFaceFetcher {
    fn fetch(&self, path: &str) -> Result<Option<Vec<u8>>> {
        self.get(path, None)
    }

    fn fetch_prefix(&self, path: &str, limit: usize) -> Result<Option<Vec<u8>>> {
        self.get(path, Some(limit))
    }

    fn origin(&self) -> String {
        format!("hf:{}@{}", self.repo, self.revision)
    }
}

/// Fetches from a local directory (the `dir:` scheme).
pub struct DirectoryFetcher {
    pub dir: PathBuf,
}

impl RepoFetcher for DirectoryFetcher {
    fn fetch(&self, path: &str) -> Result<Option<Vec<u8>>> {
        let full = self.dir.join(path);
        match std::fs::read(&full) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error).with_context(|| format!("reading {}", full.display())),
        }
    }

    fn origin(&self) -> String {
        format!("dir:{}", self.dir.display())
    }
}

// ---- machine-readable model configuration --------------------------------

#[derive(Debug, Default, Deserialize)]
struct HfConfig {
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
    #[serde(default)]
    hidden_size: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    model_max_length: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ModuleEntry {
    #[serde(default)]
    path: String,
    #[serde(rename = "type", default)]
    type_name: String,
}

#[derive(Debug, Default, Deserialize)]
struct PoolingConfig {
    #[serde(default)]
    word_embedding_dimension: Option<usize>,
    #[serde(default)]
    pooling_mode_cls_token: bool,
    #[serde(default)]
    pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pooling_mode_lasttoken: bool,
}

#[derive(Debug, Default, Deserialize)]
struct SentenceTransformersConfig {
    #[serde(default)]
    prompts: BTreeMap<String, String>,
}

/// Optional user-authored override for repositories without the
/// sentence-transformers configuration trio.
#[derive(Debug, Default, Deserialize)]
pub struct ManifestOverride {
    #[serde(default)]
    pub pooling: Option<String>,
    #[serde(default)]
    pub normalize: Option<bool>,
    #[serde(default)]
    pub native_dimensions: Option<usize>,
    #[serde(default)]
    pub max_sequence_length: Option<usize>,
    #[serde(default)]
    pub query_prefix: Option<String>,
    #[serde(default)]
    pub document_prefix: Option<String>,
}

/// One resolved repository file, hash-recorded for the lockfile.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LockedFile {
    pub sha256: String,
    pub size: u64,
}

/// Trust-on-first-use lockfile: hashes recorded on first fetch, verified on
/// every later one. Lives beside the cached model files.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ModelLockfile {
    pub origin: String,
    #[serde(default)]
    pub files: BTreeMap<String, LockedFile>,
}

pub const LOCKFILE_NAME: &str = "codeindex.lock.json";
pub const MANIFEST_OVERRIDE_NAME: &str = "codeindex.toml";

/// The resolved output: a semantic contract plus where the artifacts live.
#[derive(Debug)]
pub struct ResolvedModel {
    pub contract: ModelContract,
    /// Directory the metadata (and any downloaded artifacts) live in.
    pub local_dir: PathBuf,
    /// Repository-relative weight files that exist, in preference order
    /// (safetensors first, then ONNX).
    pub weight_files: Vec<String>,
    /// `config.json`'s declared architectures; backends dispatch on these.
    pub architectures: Vec<String>,
}

const SAFETENSORS_CANDIDATES: &[&str] = &["model.safetensors"];
const ONNX_CANDIDATES: &[&str] = &["onnx/model.onnx", "model.onnx"];

/// Resolve a HuggingFace/directory model reference into a contract by
/// fetching and parsing its configuration files into `cache_dir`.
pub fn resolve_model(
    reference: &ModelRef,
    fetcher: &dyn RepoFetcher,
    cache_dir: &Path,
) -> Result<ResolvedModel> {
    std::fs::create_dir_all(cache_dir)
        .with_context(|| format!("creating {}", cache_dir.display()))?;
    let mut lockfile = load_lockfile(cache_dir, fetcher)?;

    let config: HfConfig = fetch_json(fetcher, cache_dir, &mut lockfile, "config.json")?
        .with_context(|| format!("{} has no config.json", fetcher.origin()))?;
    let tokenizer_config: TokenizerConfig =
        fetch_json(fetcher, cache_dir, &mut lockfile, "tokenizer_config.json")?.unwrap_or_default();
    let modules: Option<Vec<ModuleEntry>> =
        fetch_json(fetcher, cache_dir, &mut lockfile, "modules.json")?;
    let st_config: SentenceTransformersConfig = fetch_json(
        fetcher,
        cache_dir,
        &mut lockfile,
        "config_sentence_transformers.json",
    )?
    .unwrap_or_default();
    let manifest: ManifestOverride = match fetcher.fetch(MANIFEST_OVERRIDE_NAME)? {
        Some(bytes) => toml::from_str(std::str::from_utf8(&bytes)?)
            .with_context(|| format!("parsing {MANIFEST_OVERRIDE_NAME}"))?,
        None => ManifestOverride::default(),
    };

    // Pooling + dimensions from the sentence-transformers Pooling module.
    let pooling_path = modules.as_ref().and_then(|modules| {
        modules
            .iter()
            .find(|entry| entry.type_name.ends_with("Pooling"))
            .map(|entry| entry.path.clone())
    });
    let pooling_config: PoolingConfig = match &pooling_path {
        Some(path) => fetch_json(
            fetcher,
            cache_dir,
            &mut lockfile,
            &format!("{path}/config.json"),
        )?
        .unwrap_or_default(),
        None => PoolingConfig::default(),
    };
    let has_normalize_module = modules
        .as_ref()
        .is_some_and(|modules| modules.iter().any(|m| m.type_name.ends_with("Normalize")));

    let pooling = if let Some(name) = &manifest.pooling {
        Pooling::parse(name).with_context(|| format!("unknown pooling override {name:?}"))?
    } else if pooling_config.pooling_mode_lasttoken {
        Pooling::LastToken
    } else if pooling_config.pooling_mode_cls_token {
        Pooling::Cls
    } else if pooling_config.pooling_mode_mean_tokens {
        Pooling::Mean
    } else {
        bail!(
            "{} does not declare a pooling strategy (no 1_Pooling/config.json); add a \
             {MANIFEST_OVERRIDE_NAME} with `pooling = \"mean\"|\"cls\"|\"last_token\"`",
            fetcher.origin()
        );
    };

    let native_dimensions = manifest
        .native_dimensions
        .or(pooling_config.word_embedding_dimension)
        .or(config.hidden_size)
        .with_context(|| {
            format!(
                "{} does not declare output dimensions; add `native_dimensions` to \
                 {MANIFEST_OVERRIDE_NAME}",
                fetcher.origin()
            )
        })?;

    // tokenizer_config's model_max_length is sometimes a giant float
    // sentinel; treat anything absurd as unset. When both the tokenizer and
    // the position embeddings declare a limit, the model attends to the
    // smaller one.
    let tokenizer_limit = tokenizer_config
        .model_max_length
        .filter(|value| value.is_finite() && *value >= 1.0 && *value <= 1e7)
        .map(|value| value as usize);
    let max_sequence_length = manifest
        .max_sequence_length
        .or(match (tokenizer_limit, config.max_position_embeddings) {
            (Some(tokenizer), Some(positions)) => Some(tokenizer.min(positions)),
            (tokenizer, positions) => tokenizer.or(positions),
        })
        .with_context(|| {
            format!(
                "{} does not declare a maximum sequence length; add `max_sequence_length` to \
                 {MANIFEST_OVERRIDE_NAME}",
                fetcher.origin()
            )
        })?;

    let prompts = prompt_contract(&st_config.prompts, &manifest);

    // Weight artifacts present in the repo, safetensors preferred.
    let mut weight_files = Vec::new();
    for candidate in SAFETENSORS_CANDIDATES.iter().chain(ONNX_CANDIDATES) {
        if file_exists(fetcher, cache_dir, candidate) {
            weight_files.push((*candidate).to_string());
        }
    }

    // The tokenizer participates in vector semantics; materialize and hash it
    // now (it is small and every backend needs it anyway).
    fetch_file(fetcher, cache_dir, &mut lockfile, "tokenizer.json")?;
    let tokenizer_hash = lockfile
        .files
        .get("tokenizer.json")
        .map(|f| f.sha256.clone());
    let contract = ModelContract {
        model: match reference {
            ModelRef::HuggingFace { repo, .. } => format!("hf:{repo}"),
            ModelRef::Directory(dir) => format!("dir:{}", dir.display()),
            ModelRef::Named(name) => name.clone(),
        },
        revision: match reference {
            ModelRef::HuggingFace { revision, .. } => {
                Some(revision.clone().unwrap_or_else(|| "main".to_string()))
            }
            _ => None,
        },
        // The weight hash is recorded when the executing backend downloads
        // the artifact; metadata-only resolution leaves it unset.
        model_hash: weight_files
            .first()
            .and_then(|file| lockfile.files.get(file))
            .map(|f| f.sha256.clone()),
        tokenizer_hash,
        pooling,
        normalize: manifest.normalize.unwrap_or(has_normalize_module),
        native_dimensions,
        max_sequence_length,
        prompts,
        quantization: None,
    };
    anyhow::ensure!(
        !config.architectures.is_empty() || manifest.pooling.is_some(),
        "{} config.json declares no architecture",
        fetcher.origin()
    );

    save_lockfile(cache_dir, &lockfile)?;
    Ok(ResolvedModel {
        contract,
        local_dir: cache_dir.to_path_buf(),
        weight_files,
        architectures: config.architectures,
    })
}

/// Map sentence-transformers prompts (+ overrides) onto a typed contract.
///
/// A query prompt shaped like Qwen3's `Instruct: <task>\nQuery:` upgrades to
/// the full instruction template with the shipped task as default, so callers
/// can substitute their own instructions; any other prompt pair becomes fixed
/// role prefixes.
fn prompt_contract(
    prompts: &BTreeMap<String, String>,
    manifest: &ManifestOverride,
) -> PromptContract {
    if manifest.query_prefix.is_some() || manifest.document_prefix.is_some() {
        return PromptContract::RolePrefixes {
            query: manifest.query_prefix.clone().unwrap_or_default(),
            document: manifest.document_prefix.clone().unwrap_or_default(),
        };
    }
    let query = prompts.get("query").cloned().unwrap_or_default();
    let document = prompts.get("document").cloned().unwrap_or_default();
    if query.is_empty() && document.is_empty() {
        return PromptContract::Symmetric;
    }
    if let Some(default_instruction) = query
        .strip_prefix("Instruct: ")
        .and_then(|rest| rest.strip_suffix("\nQuery:"))
        && document.is_empty()
    {
        return PromptContract::QueryInstruction {
            query_template: "Instruct: {instruction}\nQuery:{query}".to_string(),
            default_instruction: Some(default_instruction.to_string()),
        };
    }
    PromptContract::RolePrefixes { query, document }
}

/// Fetch a repo file into the cache with TOFU hash verification, returning
/// its parsed JSON (or `None` when absent).
fn fetch_json<T: serde::de::DeserializeOwned>(
    fetcher: &dyn RepoFetcher,
    cache_dir: &Path,
    lockfile: &mut ModelLockfile,
    path: &str,
) -> Result<Option<T>> {
    match fetch_file(fetcher, cache_dir, lockfile, path)? {
        Some(bytes) => {
            Ok(Some(serde_json::from_slice(&bytes).with_context(|| {
                format!("parsing {path} from {}", fetcher.origin())
            })?))
        }
        None => Ok(None),
    }
}

/// Fetch a repo file, verify or record its hash, and persist it in the cache.
pub fn fetch_file(
    fetcher: &dyn RepoFetcher,
    cache_dir: &Path,
    lockfile: &mut ModelLockfile,
    path: &str,
) -> Result<Option<Vec<u8>>> {
    let cached = cache_dir.join(path);
    if let Ok(bytes) = std::fs::read(&cached) {
        verify_or_record(lockfile, path, &bytes, fetcher)?;
        return Ok(Some(bytes));
    }
    let Some(bytes) = fetcher.fetch(path)? else {
        return Ok(None);
    };
    verify_or_record(lockfile, path, &bytes, fetcher)?;
    if let Some(parent) = cached.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cached, &bytes).with_context(|| format!("caching {}", cached.display()))?;
    Ok(Some(bytes))
}

fn file_exists(fetcher: &dyn RepoFetcher, cache_dir: &Path, path: &str) -> bool {
    cache_dir.join(path).exists() || matches!(fetcher.fetch_exists(path), Ok(true))
}

impl dyn RepoFetcher + '_ {
    /// Existence probe. Default falls back to a full fetch; HTTP fetchers
    /// could override with HEAD later.
    fn fetch_exists(&self, path: &str) -> Result<bool> {
        Ok(self.fetch(path)?.is_some())
    }
}

fn verify_or_record(
    lockfile: &mut ModelLockfile,
    path: &str,
    bytes: &[u8],
    fetcher: &dyn RepoFetcher,
) -> Result<()> {
    let sha256 = hex::encode(sha2::Sha256::digest(bytes));
    match lockfile.files.get(path) {
        Some(locked) if locked.sha256 != sha256 => bail!(
            "{path} from {} no longer matches the locked hash (locked {}, got {sha256}); the \
             upstream file changed. Delete {} to re-trust it.",
            fetcher.origin(),
            locked.sha256,
            LOCKFILE_NAME
        ),
        Some(_) => Ok(()),
        None => {
            lockfile.files.insert(
                path.to_string(),
                LockedFile {
                    sha256,
                    size: bytes.len() as u64,
                },
            );
            Ok(())
        }
    }
}

/// What a safetensors header declares, readable with a bounded range request
/// — no weight download required.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsPreflight {
    pub tensor_names: Vec<String>,
    pub dtypes: Vec<String>,
    /// Total artifact size implied by the last tensor's data offsets plus the
    /// header itself, in bytes.
    pub declared_size: u64,
}

/// Read only the header of a safetensors artifact: the first 8 bytes carry
/// the header length, the header itself lists every tensor's name, dtype,
/// shape, and offsets. Lets callers validate architecture fit and report the
/// download size *before* fetching gigabytes of weights.
pub fn preflight_safetensors(
    fetcher: &dyn RepoFetcher,
    path: &str,
) -> Result<Option<SafetensorsPreflight>> {
    const HEADER_CAP: usize = 16 * 1024 * 1024;
    let Some(prefix) = fetcher.fetch_prefix(path, 8)? else {
        return Ok(None);
    };
    anyhow::ensure!(prefix.len() >= 8, "{path} is too short to be safetensors");
    let header_len = u64::from_le_bytes(prefix[..8].try_into().unwrap()) as usize;
    anyhow::ensure!(
        header_len <= HEADER_CAP,
        "{path} declares an implausible {header_len}-byte safetensors header"
    );
    let full = fetcher
        .fetch_prefix(path, 8 + header_len)?
        .with_context(|| format!("{path} vanished between preflight reads"))?;
    anyhow::ensure!(
        full.len() >= 8 + header_len,
        "{path} returned a truncated safetensors header"
    );
    let header: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&full[8..])
        .with_context(|| format!("parsing safetensors header of {path}"))?;
    let mut tensor_names = Vec::new();
    let mut dtypes = Vec::new();
    let mut data_end = 0_u64;
    for (name, entry) in &header {
        if name == "__metadata__" {
            continue;
        }
        tensor_names.push(name.clone());
        if let Some(dtype) = entry.get("dtype").and_then(serde_json::Value::as_str)
            && !dtypes.iter().any(|existing| existing == dtype)
        {
            dtypes.push(dtype.to_string());
        }
        if let Some(end) = entry
            .get("data_offsets")
            .and_then(|offsets| offsets.get(1))
            .and_then(serde_json::Value::as_u64)
        {
            data_end = data_end.max(end);
        }
    }
    Ok(Some(SafetensorsPreflight {
        tensor_names,
        dtypes,
        declared_size: 8 + header_len as u64 + data_end,
    }))
}

/// Fetch (or verify) a set of required files into the cache, updating the
/// lockfile. Errors when any file is absent upstream.
pub fn materialize_files(
    fetcher: &dyn RepoFetcher,
    cache_dir: &Path,
    paths: &[&str],
) -> Result<()> {
    let mut lockfile = load_lockfile(cache_dir, fetcher)?;
    for path in paths {
        fetch_file(fetcher, cache_dir, &mut lockfile, path)?
            .with_context(|| format!("{} has no {path}", fetcher.origin()))?;
    }
    save_lockfile(cache_dir, &lockfile)
}

/// The locked hash of a previously materialized file, if any.
pub fn locked_hash(cache_dir: &Path, path: &str) -> Option<String> {
    let text = std::fs::read_to_string(cache_dir.join(LOCKFILE_NAME)).ok()?;
    let lockfile: ModelLockfile = serde_json::from_str(&text).ok()?;
    lockfile.files.get(path).map(|file| file.sha256.clone())
}

/// The fetcher matching a model reference. `Named` refs resolve through the
/// executing backend's own catalog instead.
pub fn fetcher_for(reference: &ModelRef) -> Result<Box<dyn RepoFetcher>> {
    match reference {
        ModelRef::HuggingFace { repo, revision } => Ok(Box::new(HuggingFaceFetcher {
            repo: repo.clone(),
            revision: revision.clone().unwrap_or_else(|| "main".to_string()),
        })),
        ModelRef::Directory(dir) => Ok(Box::new(DirectoryFetcher { dir: dir.clone() })),
        ModelRef::Named(name) => bail!(
            "model reference {name:?} is a backend catalog name and has no repository to resolve"
        ),
    }
}

/// The codeindex model cache root: `$XDG_CACHE_HOME/codeindex/models` (or the
/// equivalent under `$HOME`).
pub fn default_model_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(dir).join("codeindex").join("models");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home)
            .join(".cache")
            .join("codeindex")
            .join("models");
    }
    PathBuf::from(".codeindex-models")
}

fn load_lockfile(cache_dir: &Path, fetcher: &dyn RepoFetcher) -> Result<ModelLockfile> {
    let path = cache_dir.join(LOCKFILE_NAME);
    match std::fs::read_to_string(&path) {
        Ok(text) => {
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(ModelLockfile {
            origin: fetcher.origin(),
            files: BTreeMap::new(),
        }),
        Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
    }
}

fn save_lockfile(cache_dir: &Path, lockfile: &ModelLockfile) -> Result<()> {
    let path = cache_dir.join(LOCKFILE_NAME);
    std::fs::write(&path, serde_json::to_string_pretty(lockfile)?)
        .with_context(|| format!("writing {}", path.display()))
}

/// Cache directory for a resolved reference under the codeindex model root.
pub fn model_cache_dir(root: &Path, reference: &ModelRef) -> PathBuf {
    match reference {
        ModelRef::HuggingFace { repo, revision } => {
            let mut id = repo.replace('/', "--");
            if let Some(revision) = revision {
                id.push_str("--");
                id.push_str(revision);
            }
            root.join("hf").join(sanitize(&id))
        }
        ModelRef::Directory(dir) => dir.clone(),
        ModelRef::Named(name) => root.join("named").join(sanitize(name)),
    }
}

fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '-'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MapFetcher(BTreeMap<&'static str, &'static str>);

    impl RepoFetcher for MapFetcher {
        fn fetch(&self, path: &str) -> Result<Option<Vec<u8>>> {
            Ok(self.0.get(path).map(|text| text.as_bytes().to_vec()))
        }
        fn origin(&self) -> String {
            "test://repo".into()
        }
    }

    fn qwen3_like_repo() -> MapFetcher {
        // Mirrors the actual Qwen/Qwen3-Embedding-0.6B configuration files.
        MapFetcher(BTreeMap::from([
            (
                "config.json",
                r#"{"architectures":["Qwen3Model"],"hidden_size":1024,"max_position_embeddings":32768}"#,
            ),
            (
                "tokenizer_config.json",
                r#"{"model_max_length":32768,"padding_side":"left"}"#,
            ),
            (
                "modules.json",
                r#"[{"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"},
                    {"idx":1,"name":"1","path":"1_Pooling","type":"sentence_transformers.models.Pooling"},
                    {"idx":2,"name":"2","path":"2_Normalize","type":"sentence_transformers.models.Normalize"}]"#,
            ),
            (
                "1_Pooling/config.json",
                r#"{"word_embedding_dimension":1024,"pooling_mode_cls_token":false,
                    "pooling_mode_mean_tokens":false,"pooling_mode_lasttoken":true,
                    "include_prompt":true}"#,
            ),
            (
                "config_sentence_transformers.json",
                r#"{"prompts":{"query":"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:","document":""},"default_prompt_name":null,"similarity_fn_name":"cosine"}"#,
            ),
            ("tokenizer.json", r#"{"fake":"tokenizer"}"#),
            ("model.safetensors", "fake-weights"),
        ]))
    }

    #[test]
    fn model_ref_grammar() {
        assert_eq!(
            ModelRef::parse("hf:Qwen/Qwen3-Embedding-0.6B").unwrap(),
            ModelRef::HuggingFace {
                repo: "Qwen/Qwen3-Embedding-0.6B".into(),
                revision: None
            }
        );
        assert_eq!(
            ModelRef::parse("Qwen/Qwen3-Embedding-0.6B@abc123").unwrap(),
            ModelRef::HuggingFace {
                repo: "Qwen/Qwen3-Embedding-0.6B".into(),
                revision: Some("abc123".into())
            }
        );
        assert_eq!(
            ModelRef::parse("dir:/models/exported").unwrap(),
            ModelRef::Directory(PathBuf::from("/models/exported"))
        );
        assert_eq!(
            ModelRef::parse("fastembed:BGESmallENV15").unwrap(),
            ModelRef::Named("BGESmallENV15".into())
        );
        assert_eq!(
            ModelRef::parse("CodeRankEmbed").unwrap(),
            ModelRef::Named("CodeRankEmbed".into())
        );
        assert!(ModelRef::parse("").is_err());
        assert!(ModelRef::parse("hf:not-a-repo").is_err());
    }

    #[test]
    fn qwen3_configuration_resolves_to_instruction_contract() {
        let dir = tempfile::tempdir().unwrap();
        let reference = ModelRef::parse("hf:Qwen/Qwen3-Embedding-0.6B").unwrap();
        let resolved = resolve_model(&reference, &qwen3_like_repo(), dir.path()).unwrap();

        let contract = &resolved.contract;
        assert_eq!(contract.model, "hf:Qwen/Qwen3-Embedding-0.6B");
        assert_eq!(contract.pooling, Pooling::LastToken);
        assert!(contract.normalize);
        assert_eq!(contract.native_dimensions, 1024);
        assert_eq!(contract.max_sequence_length, 32768);
        match &contract.prompts {
            PromptContract::QueryInstruction {
                query_template,
                default_instruction,
            } => {
                assert_eq!(query_template, "Instruct: {instruction}\nQuery:{query}");
                assert_eq!(
                    default_instruction.as_deref(),
                    Some(
                        "Given a web search query, retrieve relevant passages that answer the \
                         query"
                    )
                );
            }
            other => panic!("expected QueryInstruction, got {other:?}"),
        }
        assert_eq!(resolved.weight_files, vec!["model.safetensors".to_string()]);
        assert!(contract.tokenizer_hash.is_some());
        assert!(dir.path().join(LOCKFILE_NAME).exists());
    }

    #[test]
    fn lockfile_detects_upstream_changes() {
        let dir = tempfile::tempdir().unwrap();
        let reference = ModelRef::parse("hf:Qwen/Qwen3-Embedding-0.6B").unwrap();
        resolve_model(&reference, &qwen3_like_repo(), dir.path()).unwrap();

        // Same content re-resolves fine (cache hit + hash match).
        resolve_model(&reference, &qwen3_like_repo(), dir.path()).unwrap();

        // Upstream (and cache) changing under the lock is rejected.
        let mut tampered = qwen3_like_repo();
        tampered.0.insert(
            "config.json",
            r#"{"architectures":["Evil"],"hidden_size":8}"#,
        );
        std::fs::remove_file(dir.path().join("config.json")).unwrap();
        let error = resolve_model(&reference, &tampered, dir.path())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("no longer matches the locked hash"),
            "{error}"
        );
    }

    #[test]
    fn missing_pooling_asks_for_manifest_override() {
        let dir = tempfile::tempdir().unwrap();
        let fetcher = MapFetcher(BTreeMap::from([(
            "config.json",
            r#"{"architectures":["BertModel"],"hidden_size":384}"#,
        )]));
        let reference = ModelRef::parse("hf:example/no-st-config").unwrap();
        let error = resolve_model(&reference, &fetcher, dir.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("codeindex.toml"), "{error}");
    }

    #[test]
    fn manifest_override_fills_gaps() {
        let dir = tempfile::tempdir().unwrap();
        let fetcher = MapFetcher(BTreeMap::from([
            (
                "config.json",
                r#"{"architectures":["BertModel"],"hidden_size":384,"max_position_embeddings":512}"#,
            ),
            (
                "codeindex.toml",
                "pooling = \"mean\"\nquery_prefix = \"query: \"\ndocument_prefix = \"passage: \"\n",
            ),
        ]));
        let reference = ModelRef::parse("hf:example/manifest-only").unwrap();
        let resolved = resolve_model(&reference, &fetcher, dir.path()).unwrap();
        assert_eq!(resolved.contract.pooling, Pooling::Mean);
        assert_eq!(resolved.contract.native_dimensions, 384);
        match &resolved.contract.prompts {
            PromptContract::RolePrefixes { query, document } => {
                assert_eq!(query, "query: ");
                assert_eq!(document, "passage: ");
            }
            other => panic!("expected RolePrefixes, got {other:?}"),
        }
    }
}
