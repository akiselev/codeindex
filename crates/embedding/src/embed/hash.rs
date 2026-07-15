use anyhow::Result;
use sha2::{Digest, Sha256};

use crate::embed::{EmbedRequest, EmbeddingBackend, render_inputs};
use codeindex_core::{ExecutionInfo, ModelContract, Pooling, PromptContract};

/// Deterministic, offline embedding backend for tests and development
/// builds. Hashes whitespace tokens into a fixed number of buckets, so
/// texts sharing vocabulary get similar vectors. Not a quality model.
pub struct HashEmbedder {
    contract: ModelContract,
    execution: ExecutionInfo,
}

impl HashEmbedder {
    pub fn new(dimensions: usize) -> Self {
        Self {
            contract: ModelContract {
                model: format!("token-hash-{dimensions}"),
                revision: None,
                model_hash: None,
                tokenizer_hash: None,
                pooling: Pooling::ModelDefined,
                normalize: true,
                native_dimensions: dimensions,
                max_sequence_length: 512,
                prompts: PromptContract::Symmetric,
                quantization: None,
            },
            execution: ExecutionInfo {
                backend: "hash-test".into(),
                backend_version: env!("CARGO_PKG_VERSION").into(),
                runtime_version: None,
                execution_provider: "cpu".into(),
                cache_path: None,
            },
        }
    }
}

impl EmbeddingBackend for HashEmbedder {
    fn contract(&self) -> &ModelContract {
        &self.contract
    }

    fn execution(&self) -> &ExecutionInfo {
        &self.execution
    }

    fn embed(&mut self, request: &EmbedRequest<'_>) -> Result<Vec<Vec<f32>>> {
        let dims = self.contract.native_dimensions;
        Ok(render_inputs(&self.contract, request)?
            .iter()
            .map(|text| {
                let mut vector = vec![0.0_f32; dims];
                for token in text.split(|c: char| !c.is_alphanumeric()) {
                    if token.is_empty() {
                        continue;
                    }
                    let digest = Sha256::digest(token.as_bytes());
                    let bucket = u64::from_le_bytes(digest[..8].try_into().unwrap());
                    vector[(bucket % dims as u64) as usize] += 1.0;
                }
                vector
            })
            .collect())
    }
}
