//! Golden conformance for the candle Qwen3-Embedding backend.
//!
//! Downloads Qwen3-Embedding-0.6B (~1.2 GB) and reproduces the similarity
//! matrix published in the model's README for its example queries/documents
//! (reference scores `[[0.7645, 0.1414], [0.1355, 0.5999]]` computed by the
//! official transformers implementation in bf16; fp32 execution here is
//! allowed a small tolerance). Run explicitly:
//!
//! ```sh
//! cargo test --release -p codeindex-embedding --features candle -- --ignored
//! ```
#![cfg(feature = "candle")]

use codeindex_embedding::config::EmbeddingConfig;
use codeindex_embedding::{EmbedRequest, embedder_from_config, normalize_in_place};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
#[ignore = "downloads Qwen3-Embedding-0.6B and runs real inference"]
fn qwen3_embedding_reproduces_published_similarities() {
    let config = EmbeddingConfig {
        model: "hf:Qwen/Qwen3-Embedding-0.6B".into(),
        ..EmbeddingConfig::default()
    };
    let mut embedder = embedder_from_config(&config).expect("backend construction");
    let contract = embedder.contract().clone();
    assert_eq!(contract.native_dimensions, 1024);
    assert_eq!(contract.pooling.as_str(), "last_token");

    // The model's shipped default instruction is exactly the one the README
    // example uses, so passing no task must reproduce the published scores.
    let queries = ["What is the capital of China?", "Explain gravity"];
    let documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to \
         physical objects and is responsible for the movement of planets around the sun.",
    ];

    let mut query_vectors = embedder
        .embed(&EmbedRequest::queries(&queries, None))
        .expect("query embedding");
    let mut document_vectors = embedder
        .embed(&EmbedRequest::documents(&documents, None))
        .expect("document embedding");
    for vector in query_vectors.iter_mut().chain(document_vectors.iter_mut()) {
        normalize_in_place(vector);
    }

    let reference = [[0.7645_f32, 0.1414], [0.1355, 0.5999]];
    let tolerance = 0.05;
    for (qi, query) in query_vectors.iter().enumerate() {
        for (di, document) in document_vectors.iter().enumerate() {
            let score = cosine(query, document);
            let expected = reference[qi][di];
            assert!(
                (score - expected).abs() < tolerance,
                "cosine(q{qi}, d{di}) = {score:.4}, published reference {expected:.4}"
            );
        }
    }
    // Ranking sanity independent of tolerance: each query prefers its own doc.
    assert!(
        cosine(&query_vectors[0], &document_vectors[0])
            > cosine(&query_vectors[0], &document_vectors[1])
    );
    assert!(
        cosine(&query_vectors[1], &document_vectors[1])
            > cosine(&query_vectors[1], &document_vectors[0])
    );
}
