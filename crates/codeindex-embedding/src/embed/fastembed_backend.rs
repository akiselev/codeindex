use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use fastembed::{
    EmbeddingModel, ExecutionProviderDispatch, InitOptionsUserDefined, Pooling, TextEmbedding,
    TextInitOptions, TokenizerFiles, UserDefinedEmbeddingModel,
};
use sha2::{Digest, Sha256};

use crate::config::{
    CustomModelConfig, EmbeddingConfig, ManagedModel, ProviderMode, managed_model,
};
use crate::embed::Embedder;
use codeindex_core::ModelIdentity;

/// Local ONNX inference through the `fastembed` crate. Downloads the model
/// into the cache directory on first use; no API credentials involved.
pub struct FastembedBackend {
    identity: ModelIdentity,
    model: TextEmbedding,
    max_sequence_length: usize,
    /// Encodes with truncation disabled, so a length past `max_sequence_length`
    /// is reported in full instead of clamped. Built once by cloning the
    /// inference tokenizer (avoids naming the `tokenizers` type directly).
    untruncated_len: LenCounter,
}

/// Counts a text's tokens with truncation disabled; `None` if encoding fails.
type LenCounter = Box<dyn Fn(&str) -> Option<usize> + Send + Sync>;

/// A cloned copy of the model's tokenizer with truncation switched off, wrapped
/// as a length-counting closure. The clone leaves the inference tokenizer's
/// truncation config untouched.
fn untruncated_len_fn(model: &TextEmbedding) -> LenCounter {
    let mut tokenizer = model.tokenizer.clone();
    let _ = tokenizer.with_truncation(None);
    Box::new(move |text| tokenizer.encode(text, true).ok().map(|e| e.len()))
}

/// Tokenizer truncation length fastembed applies to catalog models (its
/// `DEFAULT_MAX_LENGTH`); custom models use their configured `max_length`.
const CATALOG_MAX_LENGTH: usize = 512;
const TOKENIZER_FILE: &str = "tokenizer.json";
const CONFIG_FILE: &str = "config.json";
const SPECIAL_TOKENS_MAP_FILE: &str = "special_tokens_map.json";
const TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";
const CUSTOM_TOKENIZER_IDENTITY_FILES: &[&str] = &[
    TOKENIZER_FILE,
    CONFIG_FILE,
    SPECIAL_TOKENS_MAP_FILE,
    TOKENIZER_CONFIG_FILE,
];

/// Map a config model name (+ quantized flag) to the fastembed variant.
fn resolve_model(name: &str, quantized: bool) -> Result<EmbeddingModel> {
    Ok(match (name, quantized) {
        ("BGESmallENV15", false) => EmbeddingModel::BGESmallENV15,
        ("BGESmallENV15", true) => EmbeddingModel::BGESmallENV15Q,
        ("BGEBaseENV15", false) => EmbeddingModel::BGEBaseENV15,
        ("BGEBaseENV15", true) => EmbeddingModel::BGEBaseENV15Q,
        ("JinaEmbeddingsV2BaseCode", false) => EmbeddingModel::JinaEmbeddingsV2BaseCode,
        ("AllMiniLML6V2", false) => EmbeddingModel::AllMiniLML6V2,
        ("AllMiniLML6V2", true) => EmbeddingModel::AllMiniLML6V2Q,
        ("GTEBaseENV15", false) => EmbeddingModel::GTEBaseENV15,
        ("GTEBaseENV15", true) => EmbeddingModel::GTEBaseENV15Q,
        ("SnowflakeArcticEmbedM", false) => EmbeddingModel::SnowflakeArcticEmbedM,
        ("SnowflakeArcticEmbedM", true) => EmbeddingModel::SnowflakeArcticEmbedMQ,
        ("SnowflakeArcticEmbedMLong", false) => EmbeddingModel::SnowflakeArcticEmbedMLong,
        ("SnowflakeArcticEmbedMLong", true) => EmbeddingModel::SnowflakeArcticEmbedMLongQ,
        ("NomicEmbedTextV15", false) => EmbeddingModel::NomicEmbedTextV15,
        ("NomicEmbedTextV15", true) => EmbeddingModel::NomicEmbedTextV15Q,
        _ => bail!("unsupported fastembed model {name:?} (quantized={quantized})"),
    })
}

/// Load a locally exported ONNX model through fastembed's user-defined
/// model path. No download or cache: the files must already exist.
fn load_custom_model(
    custom: &CustomModelConfig,
    execution_providers: Vec<ExecutionProviderDispatch>,
) -> Result<TextEmbedding> {
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: read_custom_file(custom, Path::new(TOKENIZER_FILE))?,
        config_file: read_custom_file(custom, Path::new(CONFIG_FILE))?,
        special_tokens_map_file: read_custom_file(custom, Path::new(SPECIAL_TOKENS_MAP_FILE))?,
        tokenizer_config_file: read_custom_file(custom, Path::new(TOKENIZER_CONFIG_FILE))?,
    };
    let pooling = match custom.pooling.as_str() {
        "cls" => Pooling::Cls,
        _ => Pooling::Mean,
    };
    let model = UserDefinedEmbeddingModel::new(
        read_custom_file(custom, &custom.onnx_file)?,
        tokenizer_files,
    )
    .with_pooling(pooling);
    let options = InitOptionsUserDefined::new()
        .with_max_length(custom.max_length)
        .with_execution_providers(execution_providers);
    TextEmbedding::try_new_from_user_defined(model, options)
        .with_context(|| format!("loading custom ONNX model from {}", custom.dir.display()))
}

fn read_custom_file(custom: &CustomModelConfig, name: &Path) -> Result<Vec<u8>> {
    let path = custom.dir.join(name);
    std::fs::read(&path).with_context(|| format!("reading custom model file {}", path.display()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn custom_artifact_hashes(custom: &CustomModelConfig) -> Result<(String, String)> {
    let model_hash = sha256_hex(&read_custom_file(custom, &custom.onnx_file)?);

    let mut tokenizer_hasher = Sha256::new();
    for name in CUSTOM_TOKENIZER_IDENTITY_FILES {
        let bytes = read_custom_file(custom, Path::new(name))?;
        tokenizer_hasher.update(name.as_bytes());
        tokenizer_hasher.update([0]);
        tokenizer_hasher.update((bytes.len() as u64).to_le_bytes());
        tokenizer_hasher.update([0]);
        tokenizer_hasher.update(bytes);
    }
    Ok((hex::encode(tokenizer_hasher.finalize()), model_hash))
}

impl FastembedBackend {
    pub fn new(config: &EmbeddingConfig) -> Result<Self> {
        // Resolve the execution provider once; `provider` is what actually
        // took effect (never the requested accelerator when it fell back to
        // CPU) and is what the model identity records.
        let (execution_providers, provider) = resolve_providers(config)?;

        if let Some(custom) = &config.custom {
            let (tokenizer_hash, model_hash) = custom_artifact_hashes(custom)?;
            let model = load_custom_model(custom, execution_providers)?;
            let untruncated_len = untruncated_len_fn(&model);
            return Ok(Self {
                identity: ModelIdentity {
                    backend: "fastembed".into(),
                    backend_version: env!("DECOMBINE_FASTEMBED_VERSION").into(),
                    runtime_version: Some(format!("ort {}", env!("DECOMBINE_ORT_VERSION"))),
                    model: config.model.clone(),
                    revision: Some(format!(
                        "custom:{} pooling={} max_length={}",
                        custom.dir.join(&custom.onnx_file).display(),
                        custom.pooling,
                        custom.max_length
                    )),
                    dimensions: custom.dimensions,
                    tokenizer_hash: Some(tokenizer_hash),
                    model_hash: Some(model_hash),
                    normalize: config.normalize,
                    execution_provider: provider,
                    quantization: None,
                    cache_path: Some(custom.dir.to_string_lossy().into_owned()),
                },
                model,
                max_sequence_length: custom.max_length,
                untruncated_len,
            });
        }
        if let Some(managed) = managed_model(&config.model) {
            // A managed model with no explicit `custom` block: materialize the
            // pinned files into the cache (download + verify on first use) and
            // load them through the same custom ONNX path.
            let dir = managed_model_dir(config, managed);
            ensure_managed_files(managed, &dir)
                .with_context(|| format!("materializing managed model {}", managed.name))?;
            let custom = CustomModelConfig {
                dir: dir.clone(),
                onnx_file: PathBuf::from(managed.onnx_file),
                dimensions: managed.dimensions,
                pooling: managed.pooling.to_string(),
                max_length: managed.max_length,
            };
            let (tokenizer_hash, model_hash) = custom_artifact_hashes(&custom)?;
            let model = load_custom_model(&custom, execution_providers)?;
            let untruncated_len = untruncated_len_fn(&model);
            return Ok(Self {
                identity: ModelIdentity {
                    backend: "fastembed".into(),
                    backend_version: env!("DECOMBINE_FASTEMBED_VERSION").into(),
                    runtime_version: Some(format!("ort {}", env!("DECOMBINE_ORT_VERSION"))),
                    model: config.model.clone(),
                    // Path-independent so the identity is stable across machines,
                    // unlike an explicit custom block's absolute-path revision.
                    revision: Some(format!("managed:{}@{}", managed.repo, managed.revision)),
                    dimensions: managed.dimensions,
                    tokenizer_hash: Some(tokenizer_hash),
                    model_hash: Some(model_hash),
                    normalize: config.normalize,
                    execution_provider: provider,
                    quantization: None,
                    cache_path: Some(dir.to_string_lossy().into_owned()),
                },
                model,
                max_sequence_length: managed.max_length,
                untruncated_len,
            });
        }
        let model_name = resolve_model(&config.model, config.quantized)?;
        let info = TextEmbedding::get_model_info(&model_name)
            .context("fastembed has no metadata for the selected model")?;
        let dimensions = info.dim;
        let model_code = info.model_code.clone();

        let cache_dir = match &config.cache_dir {
            Some(dir) => dir.clone(),
            None => std::env::var_os("FASTEMBED_CACHE_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(default_cache_dir),
        };
        let options = TextInitOptions::new(model_name)
            .with_show_download_progress(true)
            .with_cache_dir(cache_dir.clone())
            .with_execution_providers(execution_providers);
        let model = TextEmbedding::try_new(options)
            .with_context(|| format!("loading fastembed model {}", config.model))?;
        let untruncated_len = untruncated_len_fn(&model);

        Ok(Self {
            identity: ModelIdentity {
                backend: "fastembed".into(),
                backend_version: env!("DECOMBINE_FASTEMBED_VERSION").into(),
                runtime_version: Some(format!("ort {}", env!("DECOMBINE_ORT_VERSION"))),
                model: config.model.clone(),
                revision: Some(model_code),
                dimensions,
                tokenizer_hash: None,
                model_hash: None,
                normalize: config.normalize,
                execution_provider: provider,
                quantization: config.quantized.then(|| "quantized".to_string()),
                cache_path: Some(cache_dir.to_string_lossy().into_owned()),
            },
            model,
            max_sequence_length: CATALOG_MAX_LENGTH,
            untruncated_len,
        })
    }
}

/// Outcome of trying to turn a requested accelerator name into an ONNX
/// Runtime execution provider. `Ready`/`Unavailable` are only constructed in
/// `accel` builds; the CPU-only build always resolves to `NotCompiled`.
#[cfg_attr(not(feature = "accel"), allow(dead_code))]
enum ProviderResolution {
    /// Ready to register on the session.
    Ready(ExecutionProviderDispatch),
    /// Compiled in, but ONNX Runtime lacks it or the platform can't run it.
    Unavailable,
    /// The EP feature is not compiled into this binary.
    NotCompiled,
}

/// Build the execution-provider list for the configured provider and report
/// which provider actually took effect. `cpu` yields an empty list (ONNX
/// Runtime's default). A requested accelerator that is missing or unavailable
/// either errors (`require`) or falls back to CPU with a warning (`auto`).
fn resolve_providers(config: &EmbeddingConfig) -> Result<(Vec<ExecutionProviderDispatch>, String)> {
    let want = config.execution_provider.as_str();
    if want == "cpu" {
        return Ok((Vec::new(), "cpu".to_string()));
    }
    let require = matches!(config.provider_mode, ProviderMode::Require);
    match build_accelerator(want, require) {
        ProviderResolution::Ready(dispatch) => Ok((vec![dispatch], want.to_string())),
        ProviderResolution::Unavailable => fallback_or_error(
            require,
            format!(
                "execution provider {want:?} is not available: ONNX Runtime was not built with \
                 it, or this platform cannot run it"
            ),
        ),
        ProviderResolution::NotCompiled => fallback_or_error(
            require,
            format!(
                "execution provider {want:?} is not compiled into this binary; rebuild with \
                 `cargo build --features {want}` or use the {want} release artifact"
            ),
        ),
    }
}

fn fallback_or_error(
    require: bool,
    message: String,
) -> Result<(Vec<ExecutionProviderDispatch>, String)> {
    if require {
        bail!(
            "{message} — set `embedding.execution_provider: cpu`, or \
             `embedding.provider_mode: auto` to fall back automatically"
        );
    }
    eprintln!("warning: {message}; falling back to cpu");
    Ok((Vec::new(), "cpu".to_string()))
}

/// Turn an accelerator name into an execution-provider dispatch when this
/// binary is built with it and ONNX Runtime can offer it. `require` marks the
/// dispatch `error_on_failure` so a failed registration surfaces as an error
/// rather than a silent CPU fallback at session-build time.
#[cfg(feature = "accel")]
fn build_accelerator(name: &str, require: bool) -> ProviderResolution {
    use ort::ep::{CUDA, CoreML, DirectML, ExecutionProvider, OpenVINO};

    macro_rules! resolve {
        ($ep:expr, $compiled:expr) => {{
            if !$compiled {
                return ProviderResolution::NotCompiled;
            }
            let ep = $ep;
            if ep.is_available().unwrap_or(false) && ep.supported_by_platform() {
                let mut dispatch = ep.build();
                if require {
                    dispatch = dispatch.error_on_failure();
                }
                ProviderResolution::Ready(dispatch)
            } else {
                ProviderResolution::Unavailable
            }
        }};
    }

    match name {
        "cuda" => resolve!(CUDA::default(), cfg!(feature = "cuda")),
        "directml" => resolve!(DirectML::default(), cfg!(feature = "directml")),
        "coreml" => resolve!(CoreML::default(), cfg!(feature = "coreml")),
        "openvino" => resolve!(OpenVINO::default(), cfg!(feature = "openvino")),
        _ => ProviderResolution::NotCompiled,
    }
}

#[cfg(not(feature = "accel"))]
fn build_accelerator(_name: &str, _require: bool) -> ProviderResolution {
    ProviderResolution::NotCompiled
}

/// Directory a managed model's files materialize into: a `custom/<id>` dir
/// beside fastembed's catalog cache so both share one decombine cache root.
fn managed_model_dir(config: &EmbeddingConfig, managed: &ManagedModel) -> PathBuf {
    let catalog = match &config.cache_dir {
        Some(dir) => dir.clone(),
        None => std::env::var_os("FASTEMBED_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(default_cache_dir),
    };
    let root = catalog.parent().map(Path::to_path_buf).unwrap_or(catalog);
    root.join("custom").join(managed.cache_id)
}

/// Ensure every pinned file is present and hash-matches, downloading any that
/// are missing or stale. Verification is by content hash, so a partially
/// written or tampered file is re-fetched rather than trusted.
fn ensure_managed_files(managed: &ManagedModel, dir: &Path) -> Result<()> {
    for file in managed.files {
        let dest = dir.join(file.path);
        if managed_file_ok(&dest, file)? {
            continue;
        }
        eprintln!(
            "downloading {} ({:.1} MB) for model {} ...",
            file.path,
            file.size as f64 / 1e6,
            managed.name
        );
        download_and_verify(managed, file, &dest)?;
    }
    Ok(())
}

/// Whether an on-disk file already matches the pinned size and hash.
fn managed_file_ok(dest: &Path, file: &crate::config::ManagedFile) -> Result<bool> {
    let Ok(meta) = std::fs::metadata(dest) else {
        return Ok(false);
    };
    if meta.len() != file.size {
        return Ok(false);
    }
    Ok(sha256_file(dest)? == file.sha256)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("reading {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Stream a file from Hugging Face to `<dest>.part`, hashing as it goes, and
/// promote it to `dest` only if the hash matches. A mismatch (moved branch,
/// corruption, tampering) removes the partial file and errors.
fn download_and_verify(
    managed: &ManagedModel,
    file: &crate::config::ManagedFile,
    dest: &Path,
) -> Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        managed.repo, managed.revision, file.path
    );
    let response = ureq::get(&url)
        .call()
        .with_context(|| format!("downloading {url}"))?;
    let mut reader = response.into_body().into_reader();

    let tmp = dest.with_extension("part");
    let mut hasher = Sha256::new();
    {
        let mut out = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        let mut buf = [0u8; 65536];
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            out.write_all(&buf[..n])?;
        }
        out.flush()?;
    }
    let got = hex::encode(hasher.finalize());
    if got != file.sha256 {
        let _ = std::fs::remove_file(&tmp);
        bail!(
            "downloaded {} has hash {got}, expected {} — refusing to use it",
            file.path,
            file.sha256
        );
    }
    std::fs::rename(&tmp, dest).with_context(|| format!("finalizing {}", dest.display()))?;
    Ok(())
}

fn default_cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(dir).join("decombine").join("models");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home)
            .join(".cache")
            .join("decombine")
            .join("models");
    }
    PathBuf::from(".decombine-models")
}

impl Embedder for FastembedBackend {
    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    fn max_sequence_length(&self) -> usize {
        self.max_sequence_length
    }

    fn count_tokens(&self, text: &str) -> Option<usize> {
        // fastembed configures truncation on this tokenizer, so the count
        // is already clamped to the model's max length.
        self.model
            .tokenizer
            .encode(text, true)
            .ok()
            .map(|encoding| encoding.len())
    }

    fn count_tokens_untruncated(&self, text: &str) -> Option<usize> {
        (self.untruncated_len)(text)
    }

    fn embed(&mut self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        // The caller packs batches to a padded-memory budget; pass each
        // through as a single fastembed batch so that budget is authoritative.
        self.model
            .embed(inputs, Some(inputs.len().max(1)))
            .context("fastembed inference failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, bytes: &[u8]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    fn custom(dir: PathBuf, onnx_file: &str) -> CustomModelConfig {
        CustomModelConfig {
            dir,
            onnx_file: PathBuf::from(onnx_file),
            dimensions: 768,
            pooling: "mean".into(),
            max_length: 2048,
        }
    }

    fn write_minimal_custom_files(dir: &Path, model: &[u8]) {
        write(&dir.join("onnx/model.onnx"), model);
        write(&dir.join(TOKENIZER_FILE), br#"{"tokenizer":true}"#);
        write(&dir.join(CONFIG_FILE), br#"{"model_type":"nomic_bert"}"#);
        write(&dir.join(SPECIAL_TOKENS_MAP_FILE), br#"{}"#);
        write(&dir.join(TOKENIZER_CONFIG_FILE), br#"{}"#);
    }

    #[test]
    fn custom_artifact_hashes_track_model_and_tokenizer_bytes() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_custom_files(dir.path(), b"fp32");
        let config = custom(dir.path().to_path_buf(), "onnx/model.onnx");

        let (tokenizer_hash, model_hash) = custom_artifact_hashes(&config).unwrap();

        write(&dir.path().join("onnx/model.onnx"), b"int8");
        let (same_tokenizer_hash, changed_model_hash) = custom_artifact_hashes(&config).unwrap();
        assert_eq!(same_tokenizer_hash, tokenizer_hash);
        assert_ne!(changed_model_hash, model_hash);

        write(
            &dir.path().join(TOKENIZER_CONFIG_FILE),
            br#"{"padding_side":"left"}"#,
        );
        let (changed_tokenizer_hash, _) = custom_artifact_hashes(&config).unwrap();
        assert_ne!(changed_tokenizer_hash, tokenizer_hash);
    }
}
