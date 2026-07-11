use anyhow::Result;
use sha2::{Digest, Sha256};

use crate::embed::Embedder;
use codeindex_core::ModelIdentity;

/// Deterministic, offline embedding backend for tests and development
/// builds. Hashes whitespace tokens into a fixed number of buckets, so
/// texts sharing vocabulary get similar vectors. Not a quality model.
pub struct HashEmbedder {
    identity: ModelIdentity,
}

impl HashEmbedder {
    pub fn new(dimensions: usize) -> Self {
        Self {
            identity: ModelIdentity {
                backend: "hash-test".into(),
                backend_version: env!("CARGO_PKG_VERSION").into(),
                runtime_version: None,
                model: format!("token-hash-{dimensions}"),
                revision: None,
                dimensions,
                tokenizer_hash: None,
                model_hash: None,
                normalize: true,
                execution_provider: "cpu".into(),
                quantization: None,
                cache_path: None,
            },
        }
    }
}

impl Embedder for HashEmbedder {
    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    fn embed(&mut self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let dims = self.identity.dimensions;
        Ok(inputs
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
