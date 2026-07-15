//! Native candle execution for decoder-style embedding models that
//! fastembed/ort structurally cannot run — currently the Qwen3-Embedding
//! family (last-token pooling, 32K context, Matryoshka dimensions).
//!
//! Correctness-first implementation: inputs are encoded one at a time
//! (candle's `qwen3` module builds causal masks internally and has no
//! padding-aware attention mask, so batched left-padded inference would be
//! wrong). Following the reference implementation, each input is truncated to
//! `max_sequence_length - 1` and terminated with the EOS token whose hidden
//! state is the embedding.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config as Qwen3Config, Model as Qwen3Model};
use tokenizers::Tokenizer;

use crate::config::EmbeddingConfig;
use crate::embed::{EmbedRequest, EmbeddingBackend, render_inputs};
use codeindex_core::{ExecutionInfo, ModelContract, Pooling};

/// Architectures this backend can instantiate, matched against
/// `config.json`'s `architectures` list.
pub const SUPPORTED_ARCHITECTURES: &[&str] = &["Qwen3Model", "Qwen3ForCausalLM"];

pub struct CandleBackend {
    contract: ModelContract,
    execution: ExecutionInfo,
    tokenizer: Tokenizer,
    /// Pristine model with an empty KV cache. Cloned per input — tensor
    /// clones share Arc-backed storage, so this is cheap — because the qwen3
    /// module's cache reset is private and the cache must never leak between
    /// unrelated texts.
    model: Qwen3Model,
    device: Device,
    eos_token_id: Option<u32>,
}

/// Whether the resolved model's architecture is one this backend implements.
pub fn supports_architectures(architectures: &[String]) -> bool {
    architectures
        .iter()
        .any(|architecture| SUPPORTED_ARCHITECTURES.contains(&architecture.as_str()))
}

/// Whether a safetensors tensor listing looks like a Qwen3 export this
/// backend can load — checked against the preflighted header so an
/// incompatible artifact is rejected before its download starts.
pub fn compatible_tensor_names(names: &[String]) -> bool {
    names
        .iter()
        .any(|name| name == "embed_tokens.weight" || name == "model.embed_tokens.weight")
}

fn select_device(config: &EmbeddingConfig) -> Result<(Device, &'static str)> {
    match config.execution_provider.as_str() {
        "cpu" => Ok((Device::Cpu, "cpu")),
        "cuda" => {
            #[cfg(feature = "candle-cuda")]
            {
                return Ok((Device::new_cuda(0)?, "cuda"));
            }
            #[cfg(not(feature = "candle-cuda"))]
            bail!(
                "execution provider \"cuda\" is not compiled into this binary; rebuild with \
                 `--features candle-cuda`"
            )
        }
        "metal" => {
            #[cfg(feature = "candle-metal")]
            {
                return Ok((Device::new_metal(0)?, "metal"));
            }
            #[cfg(not(feature = "candle-metal"))]
            bail!(
                "execution provider \"metal\" is not compiled into this binary; rebuild with \
                 `--features candle-metal`"
            )
        }
        other => bail!("the candle backend does not understand execution provider {other:?}"),
    }
}

impl CandleBackend {
    /// Load a generically resolved safetensors model. The weight file,
    /// `tokenizer.json`, and `config.json` must already be materialized in
    /// `resolved.local_dir`.
    pub fn from_resolved(
        contract: ModelContract,
        resolved_dir: &Path,
        weight_file: &str,
        config: &EmbeddingConfig,
    ) -> Result<Self> {
        anyhow::ensure!(
            contract.pooling == Pooling::LastToken,
            "the candle backend currently implements last-token pooling only; model {} \
             declares {}",
            contract.model,
            contract.pooling.as_str()
        );
        let (device, provider) = select_device(config)?;

        let config_bytes = std::fs::read(resolved_dir.join("config.json"))
            .with_context(|| format!("reading {}", resolved_dir.join("config.json").display()))?;
        let qwen_config: Qwen3Config =
            serde_json::from_slice(&config_bytes).context("parsing config.json for qwen3")?;

        let tokenizer = Tokenizer::from_file(resolved_dir.join("tokenizer.json"))
            .map_err(|error| anyhow::anyhow!("loading tokenizer.json: {error}"))?;
        let eos_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("</s>"));

        let weights = resolved_dir.join(weight_file);
        // Buffered (not mmapped) load: the crate forbids unsafe code, and the
        // one-time copy is acceptable for embedding-scale models.
        let bytes =
            std::fs::read(&weights).with_context(|| format!("reading {}", weights.display()))?;
        let vb = VarBuilder::from_buffered_safetensors(bytes, DType::F32, &device)
            .with_context(|| format!("loading {}", weights.display()))?;
        // Embedding exports save the bare `Qwen3Model` (`embed_tokens.…`,
        // `layers.…`), while candle's module addresses tensors under the
        // causal-LM `model.` prefix; strip it when the export is bare.
        let vb = if vb.contains_tensor("model.embed_tokens.weight") {
            vb
        } else {
            vb.rename_f(|name: &str| name.strip_prefix("model.").unwrap_or(name).to_string())
        };
        let model = Qwen3Model::new(&qwen_config, vb).context("instantiating qwen3 model")?;

        Ok(Self {
            contract,
            execution: ExecutionInfo {
                backend: "candle".into(),
                backend_version: env!("CARGO_PKG_VERSION").into(),
                runtime_version: Some(format!("candle {}", candle_version())),
                execution_provider: provider.into(),
                cache_path: Some(resolved_dir.to_string_lossy().into_owned()),
            },
            tokenizer,
            model,
            device,
            eos_token_id,
        })
    }

    /// Token ids for one rendered input: truncated to leave room for the
    /// terminal EOS token, whose hidden state is the embedding.
    fn input_ids(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| anyhow::anyhow!("tokenizing input: {error}"))?;
        let budget = self.contract.max_sequence_length.saturating_sub(1).max(1);
        let mut ids: Vec<u32> = encoding.get_ids().iter().copied().take(budget).collect();
        if let Some(eos) = self.eos_token_id {
            ids.push(eos);
        }
        anyhow::ensure!(!ids.is_empty(), "input tokenized to zero tokens");
        Ok(ids)
    }
}

fn candle_version() -> &'static str {
    // candle does not export a version constant; the crate version compiled
    // against is recorded at build time through cargo's metadata.
    env!("CODEINDEX_CANDLE_VERSION")
}

impl EmbeddingBackend for CandleBackend {
    fn contract(&self) -> &ModelContract {
        &self.contract
    }

    fn execution(&self) -> &ExecutionInfo {
        &self.execution
    }

    fn count_tokens(&self, text: &str) -> Option<usize> {
        let length = self.count_tokens_untruncated(text)?;
        Some(length.min(self.contract.max_sequence_length))
    }

    fn count_tokens_untruncated(&self, text: &str) -> Option<usize> {
        self.tokenizer
            .encode(text, false)
            .ok()
            .map(|encoding| encoding.len() + 1)
    }

    fn embed(&mut self, request: &EmbedRequest<'_>) -> Result<Vec<Vec<f32>>> {
        let rendered = render_inputs(&self.contract, request)?;
        let mut vectors = Vec::with_capacity(rendered.len());
        for text in &rendered {
            let ids = self.input_ids(text)?;
            let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
            // Fresh sequence per input: cloning the pristine template gives
            // an empty KV cache (the module's reset is private).
            let mut model = self.model.clone();
            let hidden = model.forward(&input, 0)?;
            // Last-token pooling: the final position's hidden state. With
            // batch size 1 and no padding this is simply the last position.
            let (_, seq_len, _) = hidden.dims3()?;
            let vector = hidden
                .narrow(1, seq_len - 1, 1)?
                .squeeze(1)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            anyhow::ensure!(
                vector.len() == self.contract.native_dimensions,
                "model produced {} dimensions, contract declares {}",
                vector.len(),
                self.contract.native_dimensions
            );
            vectors.push(vector);
        }
        Ok(vectors)
    }
}
