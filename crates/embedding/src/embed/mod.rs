#[cfg(feature = "candle")]
pub mod candle_backend;
#[cfg(feature = "fastembed")]
pub mod fastembed_backend;
pub mod hash;

use std::borrow::Cow;

use anyhow::{Context, Result};

use crate::config::EmbeddingConfig;
use codeindex_core::{EmbeddingTask, ExecutionInfo, ModelContract, PromptContract};

/// Which side of asymmetric retrieval an input belongs to. Instruction-aware
/// models render the two sides differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingRole {
    Query,
    Document,
}

/// One typed embedding call. Inputs are raw texts; rendering through the
/// model's prompt contract happens inside the backend via [`render_inputs`],
/// so persisted representation content never carries prompt text.
#[derive(Debug, Clone, Copy)]
pub struct EmbedRequest<'a> {
    pub role: EmbeddingRole,
    /// Query-side task instruction/profile. `None` uses the model's default
    /// query prompt when one exists.
    pub task: Option<&'a EmbeddingTask>,
    /// Document-side prompt from the embedding space's contract. `None`
    /// falls back to the model's own document prefix (or verbatim text).
    pub document_prompt: Option<&'a str>,
    pub inputs: &'a [&'a str],
}

impl<'a> EmbedRequest<'a> {
    pub fn documents(inputs: &'a [&'a str], document_prompt: Option<&'a str>) -> Self {
        Self {
            role: EmbeddingRole::Document,
            task: None,
            document_prompt,
            inputs,
        }
    }

    pub fn queries(inputs: &'a [&'a str], task: Option<&'a EmbeddingTask>) -> Self {
        Self {
            role: EmbeddingRole::Query,
            task,
            document_prompt: None,
            inputs,
        }
    }
}

/// An embedding backend. Implementations must be deterministic for a fixed
/// [`ModelContract`] and always return vectors at the contract's
/// `native_dimensions`; Matryoshka projection to a space's effective width is
/// applied by the caller via [`apply_output_dimensions`].
pub trait EmbeddingBackend: Send {
    /// The semantic identity of the loaded model. Gates space compatibility.
    fn contract(&self) -> &ModelContract;
    /// Execution provenance (backend, versions, device). Never compared.
    fn execution(&self) -> &ExecutionInfo;
    /// Tokens per input the model actually attends to (inputs are truncated
    /// beyond this). Bounds the padded-cost estimate in batch packing.
    fn max_sequence_length(&self) -> usize {
        self.contract().max_sequence_length
    }
    /// Exact token count for one *rendered* input using the model's own
    /// tokenizer, when the backend can provide it. Measured on the OSS eval
    /// corpora, the chars/4 fallback underestimates real token counts by
    /// 1.3–2.3x in padded area, which is exactly the margin by which embed
    /// runs overshot the token-area memory budget.
    fn count_tokens(&self, _text: &str) -> Option<usize> {
        None
    }
    /// True token count with truncation disabled. `count_tokens` clamps at
    /// the model's truncation length, which hides how far over the cap an
    /// input runs; this reveals what a capped model would silently drop.
    fn count_tokens_untruncated(&self, _text: &str) -> Option<usize> {
        None
    }
    /// Render the request through the prompt contract and encode. Returns one
    /// `native_dimensions`-width vector per input, in order.
    fn embed(&mut self, request: &EmbedRequest<'_>) -> Result<Vec<Vec<f32>>>;
}

/// Render one input for a role through the model's prompt contract.
///
/// Unknown or unsupported combinations error rather than silently embedding
/// unrendered text: a missing prompt is a retrieval-quality bug that should
/// fail loudly.
pub fn render_input<'t>(
    contract: &ModelContract,
    role: EmbeddingRole,
    task: Option<&EmbeddingTask>,
    document_prompt: Option<&str>,
    text: &'t str,
) -> Result<Cow<'t, str>> {
    match role {
        EmbeddingRole::Document => {
            if let Some(prompt) = document_prompt {
                return Ok(Cow::Owned(format!("{prompt}{text}")));
            }
            match &contract.prompts {
                PromptContract::RolePrefixes { document, .. } if !document.is_empty() => {
                    Ok(Cow::Owned(format!("{document}{text}")))
                }
                PromptContract::PairedTask { .. } => anyhow::bail!(
                    "model {} renders documents per task profile; the embedding space must set \
                     a document-side prompt",
                    contract.model
                ),
                _ => Ok(Cow::Borrowed(text)),
            }
        }
        EmbeddingRole::Query => match &contract.prompts {
            PromptContract::Symmetric => {
                anyhow::ensure!(
                    task.is_none(),
                    "model {} is symmetric and does not accept task instructions",
                    contract.model
                );
                Ok(Cow::Borrowed(text))
            }
            PromptContract::QueryInstruction {
                query_template,
                default_instruction,
            } => {
                let instruction = task
                    .map(|task| task.instruction.as_str())
                    .or(default_instruction.as_deref())
                    .with_context(|| {
                        format!(
                            "model {} requires a task instruction for queries and defines no \
                             default",
                            contract.model
                        )
                    })?;
                Ok(Cow::Owned(
                    query_template
                        .replace("{instruction}", instruction)
                        .replace("{query}", text),
                ))
            }
            PromptContract::RolePrefixes { query, .. } => {
                anyhow::ensure!(
                    task.is_none(),
                    "model {} has a fixed query prefix and does not accept task instructions",
                    contract.model
                );
                Ok(Cow::Owned(format!("{query}{text}")))
            }
            PromptContract::PairedTask { tasks } => {
                let task = task.with_context(|| {
                    format!(
                        "model {} requires a task profile for queries; available: {}",
                        contract.model,
                        tasks.keys().cloned().collect::<Vec<_>>().join(", ")
                    )
                })?;
                let pair = tasks.get(&task.id).with_context(|| {
                    format!("model {} has no task profile {:?}", contract.model, task.id)
                })?;
                Ok(Cow::Owned(format!("{}{}", pair.query, text)))
            }
        },
    }
}

/// Render every input of a request. Backends call this before encoding.
pub fn render_inputs<'t>(
    contract: &ModelContract,
    request: &EmbedRequest<'t>,
) -> Result<Vec<Cow<'t, str>>> {
    request
        .inputs
        .iter()
        .map(|text| {
            render_input(
                contract,
                request.role,
                request.task,
                request.document_prompt,
                text,
            )
        })
        .collect()
}

/// Project a native-width vector to a space's effective width: Matryoshka
/// truncation keeps the leading dimensions and re-normalizes when the model
/// normalizes. No-op when `output_dimensions` is `None` or the native width.
pub fn apply_output_dimensions(
    vector: &mut Vec<f32>,
    output_dimensions: Option<usize>,
    normalize: bool,
) -> Result<()> {
    let Some(dims) = output_dimensions else {
        return Ok(());
    };
    anyhow::ensure!(
        dims > 0 && dims <= vector.len(),
        "output_dimensions {dims} is outside 1..={}",
        vector.len()
    );
    if dims < vector.len() {
        vector.truncate(dims);
        if normalize {
            normalize_in_place(vector);
        }
    }
    Ok(())
}

/// Build a backend for the configured model reference. `fastembed:`/managed
/// names and explicit `custom` blocks run through the fastembed catalog path;
/// `hf:owner/name[@rev]` and `dir:/path` references are resolved generically
/// from the repository's own configuration and executed by whichever backend
/// can run the resolved contract (candle for last-token safetensors models,
/// fastembed for mean/cls ONNX exports).
pub fn embedder_from_config(config: &EmbeddingConfig) -> Result<Box<dyn EmbeddingBackend>> {
    anyhow::ensure!(
        config.backend == "fastembed",
        "unsupported embedding backend {:?}",
        config.backend
    );
    if config.custom.is_some() {
        return fastembed_new(config);
    }
    let reference = crate::resolve::ModelRef::parse(&config.model)?;
    if matches!(reference, crate::resolve::ModelRef::Named(_)) {
        return fastembed_new(config);
    }
    resolved_backend(config, &reference)
}

fn fastembed_new(config: &EmbeddingConfig) -> Result<Box<dyn EmbeddingBackend>> {
    #[cfg(feature = "fastembed")]
    {
        Ok(Box::new(fastembed_backend::FastembedBackend::new(config)?))
    }
    #[cfg(not(feature = "fastembed"))]
    {
        let _ = config;
        anyhow::bail!(
            "this binary was built without the `fastembed` feature; \
             rebuild with `cargo build --features fastembed`"
        )
    }
}

fn resolved_backend(
    config: &EmbeddingConfig,
    reference: &crate::resolve::ModelRef,
) -> Result<Box<dyn EmbeddingBackend>> {
    use crate::resolve::{fetcher_for, model_cache_dir, resolve_model};

    let root = config
        .cache_dir
        .clone()
        .unwrap_or_else(crate::resolve::default_model_root);
    let cache_dir = model_cache_dir(&root, reference);
    let fetcher = fetcher_for(reference)?;
    let resolved = resolve_model(reference, fetcher.as_ref(), &cache_dir)?;

    let safetensors = resolved
        .weight_files
        .iter()
        .find(|file| file.ends_with(".safetensors"))
        .cloned();
    let onnx = resolved
        .weight_files
        .iter()
        .find(|file| file.ends_with(".onnx"))
        .cloned();
    let _ = (&safetensors, &onnx);

    #[cfg(feature = "candle")]
    if let Some(weights) = safetensors.as_deref()
        && resolved.contract.pooling == codeindex_core::Pooling::LastToken
        && candle_backend::supports_architectures(&resolved.architectures)
    {
        // Preflight the safetensors header (a bounded range read) before
        // committing to a multi-gigabyte weight download: incompatible tensor
        // naming fails here, not after the fetch.
        if !cache_dir.join(weights).exists()
            && let Some(preflight) =
                crate::resolve::preflight_safetensors(fetcher.as_ref(), weights)?
        {
            anyhow::ensure!(
                candle_backend::compatible_tensor_names(&preflight.tensor_names),
                "model {} declares a supported architecture but its weight artifact {weights} \
                 has no recognizable embed_tokens tensor; refusing to download {:.2} GB of \
                 unusable weights",
                resolved.contract.model,
                preflight.declared_size as f64 / 1e9,
            );
            eprintln!(
                "downloading {weights} ({:.2} GB, {}) for model {} ...",
                preflight.declared_size as f64 / 1e9,
                preflight.dtypes.join("/"),
                resolved.contract.model
            );
        }
        crate::resolve::materialize_files(
            fetcher.as_ref(),
            &cache_dir,
            &[weights, "tokenizer.json", "config.json"],
        )?;
        let mut contract = resolved.contract;
        contract.model_hash = crate::resolve::locked_hash(&cache_dir, weights);
        return Ok(Box::new(candle_backend::CandleBackend::from_resolved(
            contract, &cache_dir, weights, config,
        )?));
    }

    #[cfg(feature = "fastembed")]
    if let Some(onnx) = onnx.as_deref()
        && matches!(
            resolved.contract.pooling,
            codeindex_core::Pooling::Mean | codeindex_core::Pooling::Cls
        )
    {
        crate::resolve::materialize_files(
            fetcher.as_ref(),
            &cache_dir,
            &[
                onnx,
                "tokenizer.json",
                "config.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
            ],
        )?;
        let mut contract = resolved.contract;
        contract.model_hash = crate::resolve::locked_hash(&cache_dir, onnx);
        return Ok(Box::new(
            fastembed_backend::FastembedBackend::from_resolved(contract, &cache_dir, onnx, config)?,
        ));
    }

    anyhow::bail!(
        "model {} resolved (pooling: {}, {} dims, architectures: [{}], weights: [{}]) but no \
         backend in this build can execute it. Last-token safetensors models need the `candle` \
         feature{}; mean/cls ONNX exports need the `fastembed` feature{}.",
        resolved.contract.model,
        resolved.contract.pooling.as_str(),
        resolved.contract.native_dimensions,
        resolved.architectures.join(", "),
        resolved.weight_files.join(", "),
        if cfg!(feature = "candle") {
            " (enabled, but this model's architecture or weights did not match)"
        } else {
            " (not enabled)"
        },
        if cfg!(feature = "fastembed") {
            " (enabled, but no compatible ONNX artifact was found)"
        } else {
            " (not enabled)"
        },
    )
}

/// Accelerator execution providers the local backend can request, in priority order.
pub const ACCELERATOR_PROVIDERS: &[&str] = &["cuda", "directml", "coreml", "openvino"];

/// One provider's readiness, as reported by `doctor`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderDiag {
    pub name: &'static str,
    /// This binary was built with the provider's cargo feature.
    pub compiled: bool,
    /// ONNX Runtime was built with support for the provider (`None` when not
    /// compiled in, so it cannot be queried).
    pub available: Option<bool>,
    /// The provider can run on this OS/arch at all.
    pub platform_supported: Option<bool>,
}

/// Compile- and runtime status of each accelerator provider. On a CPU-only
/// build every provider reports `compiled: false`; with `accel` features the
/// ONNX Runtime availability and platform support are queried live.
pub fn accelerator_diagnostics() -> Vec<ProviderDiag> {
    #[cfg(feature = "accel")]
    {
        use ort::ep::{CUDA, CoreML, DirectML, ExecutionProvider, OpenVINO};
        fn diag(name: &'static str, compiled: bool, ep: &dyn ExecutionProvider) -> ProviderDiag {
            ProviderDiag {
                name,
                compiled,
                available: Some(ep.is_available().unwrap_or(false)),
                platform_supported: Some(ep.supported_by_platform()),
            }
        }
        vec![
            diag("cuda", cfg!(feature = "cuda"), &CUDA::default()),
            diag("directml", cfg!(feature = "directml"), &DirectML::default()),
            diag("coreml", cfg!(feature = "coreml"), &CoreML::default()),
            diag("openvino", cfg!(feature = "openvino"), &OpenVINO::default()),
        ]
    }
    #[cfg(not(feature = "accel"))]
    {
        ACCELERATOR_PROVIDERS
            .iter()
            .map(|name| ProviderDiag {
                name,
                compiled: false,
                available: None,
                platform_supported: None,
            })
            .collect()
    }
}

/// Distribution of input token lengths plus padding/truncation costs, using
/// the same counts the batch packer budgets with.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TokenStats {
    /// Sorted at read time via `percentile`; append-only during the run.
    lengths: Vec<usize>,
    /// Inputs at the model's truncation length (content beyond it is lost).
    pub truncated: usize,
    /// Sum of real token positions across inputs.
    pub token_positions: u64,
    /// Sum of padded positions actually materialized per batch
    /// (`batch_len * longest_in_batch`).
    pub padded_positions: u64,
}

impl TokenStats {
    pub fn record_batch(&mut self, tokens: impl Iterator<Item = usize> + Clone, max_len: usize) {
        let longest = tokens.clone().max().unwrap_or(0);
        let mut count = 0_u64;
        for t in tokens {
            self.lengths.push(t);
            self.token_positions += t as u64;
            if t >= max_len {
                self.truncated += 1;
            }
            count += 1;
        }
        self.padded_positions += count * longest as u64;
    }

    /// Record one input's true (untruncated) token length for distribution
    /// reporting. Unlike `record_batch` this tracks no padding — the input is
    /// being measured, not batched — so `padding_waste`/`truncated` stay 0 and
    /// over-cap counts come from `over` instead.
    pub fn record_length(&mut self, tokens: usize) {
        self.lengths.push(tokens);
        self.token_positions += tokens as u64;
    }

    /// Fold another distribution into this one (for an all-languages total).
    pub fn merge(&mut self, other: &TokenStats) {
        self.lengths.extend_from_slice(&other.lengths);
        self.truncated += other.truncated;
        self.token_positions += other.token_positions;
        self.padded_positions += other.padded_positions;
    }

    /// Inputs strictly longer than `threshold` tokens — the content a model
    /// capped at `threshold` would silently drop.
    pub fn over(&self, threshold: usize) -> usize {
        self.lengths.iter().filter(|&&t| t > threshold).count()
    }

    pub fn percentile(&self, q: f64) -> usize {
        if self.lengths.is_empty() {
            return 0;
        }
        let mut sorted = self.lengths.clone();
        sorted.sort_unstable();
        sorted[((q * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1)]
    }

    pub fn max(&self) -> usize {
        self.lengths.iter().copied().max().unwrap_or(0)
    }

    pub fn count(&self) -> usize {
        self.lengths.len()
    }

    /// Fraction of materialized positions that are padding.
    pub fn padding_waste(&self) -> f64 {
        if self.padded_positions == 0 {
            return 0.0;
        }
        1.0 - self.token_positions as f64 / self.padded_positions as f64
    }
}

/// Rough tokens for a code body: ~4 chars/token, clamped to what the model
/// will actually attend to after truncation.
pub fn estimated_tokens(text: &str, max_sequence_length: usize) -> usize {
    (text.chars().count() / 4 + 1).min(max_sequence_length.max(1))
}

/// Split (hash, text, tokens) items into batches bounded by item count,
/// total characters, and padded token area. ONNX runtime pads every input
/// to the longest in the batch and materializes attention over it, so peak
/// memory scales with `count * longest_tokens^2`; that product is what
/// `max_token_area` caps. Items must be sorted longest-first so the running
/// maximum is the first item and long bodies batch together instead of
/// inflating short ones.
pub fn pack_batches(
    items: &[(String, String, usize)],
    max_items: usize,
    max_chars: usize,
    max_token_area: usize,
) -> Vec<&[(String, String, usize)]> {
    let mut batches = Vec::new();
    let mut start = 0;
    let mut chars = 0;
    let mut longest_tokens = 0;
    for (index, (_, text, tokens)) in items.iter().enumerate() {
        let len = text.chars().count();
        let longest = longest_tokens.max(*tokens);
        let at_capacity = index > start
            && (index - start >= max_items
                || chars + len > max_chars
                || (index - start + 1) * longest * longest > max_token_area);
        if at_capacity {
            batches.push(&items[start..index]);
            start = index;
            chars = 0;
            longest_tokens = 0;
        }
        chars += len;
        longest_tokens = longest_tokens.max(*tokens);
    }
    if start < items.len() {
        batches.push(&items[start..]);
    }
    batches
}

pub fn normalize_in_place(vector: &mut [f32]) {
    let norm = vector
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        for value in vector.iter_mut() {
            *value = (*value as f64 / norm) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codeindex_core::{PairedPrompts, Pooling};
    use std::collections::BTreeMap;

    fn contract(prompts: PromptContract) -> ModelContract {
        ModelContract {
            model: "test-model".into(),
            revision: None,
            model_hash: None,
            tokenizer_hash: None,
            pooling: Pooling::Mean,
            normalize: true,
            native_dimensions: 4,
            max_sequence_length: 512,
            prompts,
            quantization: None,
        }
    }

    #[test]
    fn symmetric_models_render_verbatim_and_reject_tasks() {
        let contract = contract(PromptContract::Symmetric);
        let rendered =
            render_input(&contract, EmbeddingRole::Document, None, None, "text").unwrap();
        assert_eq!(rendered, "text");
        let rendered = render_input(&contract, EmbeddingRole::Query, None, None, "text").unwrap();
        assert_eq!(rendered, "text");
        let task = EmbeddingTask::new("t", "instruction");
        assert!(render_input(&contract, EmbeddingRole::Query, Some(&task), None, "text").is_err());
    }

    #[test]
    fn query_instruction_renders_qwen_template() {
        let contract = contract(PromptContract::QueryInstruction {
            query_template: "Instruct: {instruction}\nQuery:{query}".into(),
            default_instruction: Some("Given a web search query, retrieve passages".into()),
        });
        let task = EmbeddingTask::new("edit", "Retrieve code needing edits");
        assert_eq!(
            render_input(
                &contract,
                EmbeddingRole::Query,
                Some(&task),
                None,
                "fix bug"
            )
            .unwrap(),
            "Instruct: Retrieve code needing edits\nQuery:fix bug"
        );
        // No task falls back to the model's shipped default instruction.
        assert_eq!(
            render_input(&contract, EmbeddingRole::Query, None, None, "fix bug").unwrap(),
            "Instruct: Given a web search query, retrieve passages\nQuery:fix bug"
        );
        // Documents embed raw for instruction models.
        assert_eq!(
            render_input(&contract, EmbeddingRole::Document, None, None, "fn f() {}").unwrap(),
            "fn f() {}"
        );
        let no_default = self::contract(PromptContract::QueryInstruction {
            query_template: "Instruct: {instruction}\nQuery:{query}".into(),
            default_instruction: None,
        });
        assert!(render_input(&no_default, EmbeddingRole::Query, None, None, "q").is_err());
    }

    #[test]
    fn role_prefixes_apply_per_side_and_space_prompt_wins() {
        let contract = contract(PromptContract::RolePrefixes {
            query: "QUERY: ".into(),
            document: "DOC: ".into(),
        });
        assert_eq!(
            render_input(&contract, EmbeddingRole::Query, None, None, "text").unwrap(),
            "QUERY: text"
        );
        assert_eq!(
            render_input(&contract, EmbeddingRole::Document, None, None, "text").unwrap(),
            "DOC: text"
        );
        // An explicit space-level document prompt overrides the model default.
        assert_eq!(
            render_input(
                &contract,
                EmbeddingRole::Document,
                None,
                Some("SPACE: "),
                "text"
            )
            .unwrap(),
            "SPACE: text"
        );
    }

    #[test]
    fn paired_task_requires_profile_and_document_prompt() {
        let mut tasks = BTreeMap::new();
        tasks.insert(
            "nl2code".to_string(),
            PairedPrompts {
                query: "Q[nl2code]: ".into(),
                document: "P[nl2code]: ".into(),
            },
        );
        let contract = contract(PromptContract::PairedTask { tasks });
        let task = EmbeddingTask::new("nl2code", "");
        assert_eq!(
            render_input(&contract, EmbeddingRole::Query, Some(&task), None, "q").unwrap(),
            "Q[nl2code]: q"
        );
        assert!(render_input(&contract, EmbeddingRole::Query, None, None, "q").is_err());
        // Documents require the space to have chosen a task profile prompt.
        assert!(render_input(&contract, EmbeddingRole::Document, None, None, "d").is_err());
        assert_eq!(
            render_input(
                &contract,
                EmbeddingRole::Document,
                None,
                Some("P[nl2code]: "),
                "d"
            )
            .unwrap(),
            "P[nl2code]: d"
        );
    }

    #[test]
    fn output_dimension_projection_truncates_and_renormalizes() {
        let mut vector = vec![3.0_f32, 4.0, 100.0, -7.0];
        apply_output_dimensions(&mut vector, Some(2), true).unwrap();
        assert_eq!(vector.len(), 2);
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);

        let mut untouched = vec![1.0_f32, 2.0];
        apply_output_dimensions(&mut untouched, None, true).unwrap();
        assert_eq!(untouched, vec![1.0, 2.0]);

        let mut too_wide = vec![1.0_f32, 2.0];
        assert!(apply_output_dimensions(&mut too_wide, Some(3), true).is_err());
    }

    /// Items sized in chars, tokens derived with the chars/4 estimate
    /// clamped at 500 — mirrors the fallback path.
    fn pairs(sizes: &[usize]) -> Vec<(String, String, usize)> {
        sizes
            .iter()
            .enumerate()
            .map(|(i, len)| {
                let text = "x".repeat(*len);
                let tokens = estimated_tokens(&text, 500);
                (format!("h{i}"), text, tokens)
            })
            .collect()
    }

    #[test]
    fn batches_respect_item_and_char_limits() {
        let items = pairs(&[10, 10, 10, 10, 10]);
        let by_count = pack_batches(&items, 2, 1000, usize::MAX);
        assert_eq!(by_count.len(), 3);
        assert_eq!(by_count[0].len(), 2);
        assert_eq!(by_count[2].len(), 1);

        let by_chars = pack_batches(&items, 100, 25, usize::MAX);
        assert_eq!(by_chars.len(), 3, "10+10 fits, third overflows 25");

        // A single oversized item still forms its own batch.
        let big = pairs(&[500]);
        assert_eq!(pack_batches(&big, 10, 25, usize::MAX).len(), 1);
    }

    #[test]
    fn batches_respect_token_area_budget() {
        // 2000 chars ~ 501 tokens, clamped to 500 -> area 250_000 per item.
        // Budget 1_000_000 fits four such items per batch.
        let items = pairs(&[2000, 2000, 2000, 2000, 2000, 2000]);
        let batches = pack_batches(&items, 100, usize::MAX, 1_000_000);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 4);
        assert_eq!(batches[1].len(), 2);

        // Longest-first input: one long item (500 tokens after clamping)
        // caps its batch at 4, then the short tail (26 tokens) packs densely.
        let mut sizes = vec![2000usize];
        sizes.extend(std::iter::repeat_n(100, 20));
        let mixed = pairs(&sizes);
        let batches = pack_batches(&mixed, 100, usize::MAX, 1_000_000);
        assert_eq!(batches[0].len(), 4, "long head limits the first batch");
        assert_eq!(batches.len(), 2, "short tail packs into one batch");

        // A single item over budget still embeds alone.
        let big = pairs(&[4000]);
        assert_eq!(pack_batches(&big, 10, usize::MAX, 1000).len(), 1);

        // Exact token counts override any char-based intuition: 100-char
        // items reported as 500 tokens each pack like long items.
        let dense: Vec<(String, String, usize)> = (0..6)
            .map(|i| (format!("d{i}"), "y".repeat(100), 500))
            .collect();
        let batches = pack_batches(&dense, 100, usize::MAX, 1_000_000);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 4);
    }

    #[test]
    fn token_stats_percentiles_truncation_and_waste() {
        let mut stats = TokenStats::default();
        // One batch padded to 500: 3 items -> 1500 padded, 1000 real.
        stats.record_batch([500usize, 300, 200].into_iter(), 500);
        // A dense batch with no padding.
        stats.record_batch([100usize, 100].into_iter(), 500);
        assert_eq!(stats.count(), 5);
        assert_eq!(stats.truncated, 1);
        assert_eq!(stats.max(), 500);
        assert_eq!(stats.percentile(0.5), 200);
        assert_eq!(stats.token_positions, 1200);
        assert_eq!(stats.padded_positions, 1700);
        assert!((stats.padding_waste() - (1.0 - 1200.0 / 1700.0)).abs() < 1e-9);
    }

    #[test]
    fn record_length_over_and_merge() {
        let mut a = TokenStats::default();
        for len in [100usize, 600, 2100, 300] {
            a.record_length(len);
        }
        assert_eq!(a.count(), 4);
        assert_eq!(a.max(), 2100);
        assert_eq!(a.over(512), 2, "600 and 2100 exceed 512");
        assert_eq!(a.over(2048), 1, "only 2100 exceeds 2048");
        // record_length tracks no padding, so waste stays zero.
        assert_eq!(a.padding_waste(), 0.0);

        let mut b = TokenStats::default();
        b.record_length(700);
        a.merge(&b);
        assert_eq!(a.count(), 5);
        assert_eq!(a.over(512), 3);
    }

    #[test]
    fn normalization() {
        let mut vector = vec![3.0, 4.0];
        normalize_in_place(&mut vector);
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
        let mut zero = vec![0.0, 0.0];
        normalize_in_place(&mut zero);
        assert_eq!(zero, vec![0.0, 0.0]);
    }
}
