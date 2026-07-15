//! Cross-encoder reranking, a separate primitive from embedding: the model
//! jointly reads (instruction, query, document) and scores relevance, fixing
//! the flat score bands and near-miss orderings first-stage retrieval leaves
//! behind. First implementation: Qwen3-Reranker through candle — a causal LM
//! judged on the probability of answering "yes" vs "no".
#![cfg(feature = "candle")]

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{
    Config as Qwen3Config, ModelForCausalLM as Qwen3CausalLM,
};
use tokenizers::Tokenizer;

use crate::config::EmbeddingConfig;
use crate::resolve::{ModelRef, fetcher_for, materialize_files, model_cache_dir};
use codeindex_core::ExecutionInfo;

/// Identity of a reranker. Rerankers produce scores, not persisted vectors,
/// so this never gates space compatibility — it is recorded with results for
/// reproducibility.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RerankerIdentity {
    pub model: String,
    pub revision: Option<String>,
    pub model_hash: Option<String>,
    pub max_sequence_length: usize,
}

/// A second-stage relevance scorer over (instruction, query, document).
pub trait Reranker: Send {
    fn identity(&self) -> &RerankerIdentity;
    fn execution(&self) -> &ExecutionInfo;
    /// One relevance score in [0, 1] per document, higher is better.
    fn rerank(&mut self, instruction: &str, query: &str, documents: &[&str]) -> Result<Vec<f32>>;
}

const SYSTEM_PROMPT: &str = "Judge whether the Document meets the requirements based on the \
                             Query and the Instruct provided. Note that the answer can only be \
                             \"yes\" or \"no\".";

pub struct Qwen3Reranker {
    identity: RerankerIdentity,
    execution: ExecutionInfo,
    tokenizer: Tokenizer,
    model: Qwen3CausalLM,
    device: Device,
    yes_id: u32,
    no_id: u32,
    prefix_ids: Vec<u32>,
    suffix_ids: Vec<u32>,
    /// Total token budget per judgement; documents are truncated to fit.
    /// Bounded well below the model's 40K context for CPU latency.
    max_tokens: usize,
}

impl Qwen3Reranker {
    /// Load a Qwen3-Reranker checkpoint by model reference
    /// (e.g. `hf:Qwen/Qwen3-Reranker-0.6B`).
    pub fn from_reference(reference: &str, config: &EmbeddingConfig) -> Result<Qwen3Reranker> {
        let reference = ModelRef::parse(reference)?;
        let root = config
            .cache_dir
            .clone()
            .unwrap_or_else(crate::resolve::default_model_root);
        let cache_dir = model_cache_dir(&root, &reference);
        std::fs::create_dir_all(&cache_dir)
            .with_context(|| format!("creating {}", cache_dir.display()))?;
        let fetcher = fetcher_for(&reference)?;
        materialize_files(
            fetcher.as_ref(),
            &cache_dir,
            &["config.json", "tokenizer.json", "model.safetensors"],
        )?;

        let device = match config.execution_provider.as_str() {
            "cpu" => Device::Cpu,
            other => bail!("the reranker currently runs on cpu only (got {other:?})"),
        };
        let qwen_config: Qwen3Config =
            serde_json::from_slice(&std::fs::read(cache_dir.join("config.json"))?)
                .context("parsing reranker config.json")?;
        let tokenizer = Tokenizer::from_file(cache_dir.join("tokenizer.json"))
            .map_err(|error| anyhow::anyhow!("loading reranker tokenizer: {error}"))?;
        let yes_id = tokenizer
            .token_to_id("yes")
            .context("reranker tokenizer has no `yes` token")?;
        let no_id = tokenizer
            .token_to_id("no")
            .context("reranker tokenizer has no `no` token")?;

        let bytes = std::fs::read(cache_dir.join("model.safetensors"))?;
        let vb = VarBuilder::from_buffered_safetensors(bytes, DType::F32, &device)?;
        let vb = if vb.contains_tensor("model.embed_tokens.weight") {
            vb
        } else {
            vb.rename_f(|name: &str| name.strip_prefix("model.").unwrap_or(name).to_string())
        };
        let model = Qwen3CausalLM::new(&qwen_config, vb).context("instantiating qwen3 reranker")?;

        let encode = |text: &str, tokenizer: &Tokenizer| -> Result<Vec<u32>> {
            Ok(tokenizer
                .encode(text, false)
                .map_err(|error| anyhow::anyhow!("tokenizing: {error}"))?
                .get_ids()
                .to_vec())
        };
        let prefix = format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n");
        let suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let prefix_ids = encode(&prefix, &tokenizer)?;
        let suffix_ids = encode(suffix, &tokenizer)?;

        Ok(Qwen3Reranker {
            identity: RerankerIdentity {
                model: match &reference {
                    ModelRef::HuggingFace { repo, .. } => format!("hf:{repo}"),
                    ModelRef::Directory(dir) => format!("dir:{}", dir.display()),
                    ModelRef::Named(name) => name.clone(),
                },
                revision: match &reference {
                    ModelRef::HuggingFace { revision, .. } => {
                        Some(revision.clone().unwrap_or_else(|| "main".into()))
                    }
                    _ => None,
                },
                model_hash: crate::resolve::locked_hash(&cache_dir, "model.safetensors"),
                max_sequence_length: 2048,
            },
            execution: ExecutionInfo {
                backend: "candle".into(),
                backend_version: env!("CARGO_PKG_VERSION").into(),
                runtime_version: Some(format!("candle {}", env!("CODEINDEX_CANDLE_VERSION"))),
                execution_provider: "cpu".into(),
                cache_path: Some(cache_dir.to_string_lossy().into_owned()),
            },
            tokenizer,
            model,
            device,
            yes_id,
            no_id,
            prefix_ids,
            suffix_ids,
            max_tokens: 2048,
        })
    }

    fn judge(&mut self, instruction: &str, query: &str, document: &str) -> Result<f32> {
        let body = format!("<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}");
        let body_ids = self
            .tokenizer
            .encode(body.as_str(), false)
            .map_err(|error| anyhow::anyhow!("tokenizing judgement: {error}"))?
            .get_ids()
            .to_vec();
        let budget = self
            .max_tokens
            .saturating_sub(self.prefix_ids.len() + self.suffix_ids.len())
            .max(16);
        let mut ids = self.prefix_ids.clone();
        ids.extend(body_ids.into_iter().take(budget));
        ids.extend_from_slice(&self.suffix_ids);

        self.model.clear_kv_cache();
        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        // ModelForCausalLM::forward narrows to the final position: [1, 1, vocab].
        let logits = self
            .model
            .forward(&input, 0)?
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        let yes = logits.get(self.yes_id as usize)?.to_scalar::<f32>()?;
        let no = logits.get(self.no_id as usize)?.to_scalar::<f32>()?;
        // Stable two-way softmax: P(yes) / (P(yes) + P(no)).
        let peak = yes.max(no);
        let yes_exp = (yes - peak).exp();
        Ok(yes_exp / (yes_exp + (no - peak).exp()))
    }
}

impl Reranker for Qwen3Reranker {
    fn identity(&self) -> &RerankerIdentity {
        &self.identity
    }

    fn execution(&self) -> &ExecutionInfo {
        &self.execution
    }

    fn rerank(&mut self, instruction: &str, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        documents
            .iter()
            .map(|document| self.judge(instruction, query, document))
            .collect()
    }
}
