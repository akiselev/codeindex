//! Embedding a *stored* corpus: resumable projection of indexed bodies into
//! vectors, source-text recovery under lean retention, and offline token
//! reports. This is the storage- and parser-coupled half of embedding; the
//! parser/storage-free primitives it builds on (the [`Embedder`] trait, batch
//! packing, normalization, token stats) live in `codeindex-embedding`.

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Context, Result};
use codeindex_embedding::config::EmbeddingRunConfig;
use codeindex_embedding::{
    Embedder, TokenStats, estimated_tokens, normalize_in_place, pack_batches,
};
use codeindex_sqlite::{Db, ModelId, ModelIdentity, NewCodeUnit};
use codeindex_tree_sitter::{ExtractOptions, LanguageRegistry, extract_units};

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct EmbedStats {
    /// Distinct body hashes pending at the start of this run.
    pub pending_total: usize,
    /// Distinct body hashes embedded in this run.
    pub embedded: usize,
    /// Pending hashes whose text could not be recovered (stale files in
    /// `minimal`/`report` retention).
    pub unresolved: usize,
    pub batches: usize,
    /// Token-length instrumentation over embedded inputs (post-truncation
    /// counts from the packer).
    pub tokens: TokenStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedProgress {
    pub pending_total: usize,
    pub embedded: usize,
    pub unresolved: usize,
    pub batches: usize,
    pub current_batch: usize,
}

/// Embed every distinct un-embedded body hash with this embedder, enforcing
/// model-identity immutability and resuming where a prior run stopped.
pub fn embed_pending(
    db: &Db,
    embedder: &mut dyn Embedder,
    config: &EmbeddingRunConfig,
) -> Result<EmbedStats> {
    embed_pending_with_progress(db, embedder, config, |_| {})
}

pub fn embed_pending_with_progress(
    db: &Db,
    embedder: &mut dyn Embedder,
    config: &EmbeddingRunConfig,
    mut progress: impl FnMut(EmbedProgress),
) -> Result<EmbedStats> {
    let identity = embedder.identity().clone();
    db.check_or_set_immutable("embedding.backend", &identity.backend)?;
    db.check_or_set_immutable("embedding.model", &identity.model)?;
    db.check_or_set_immutable("embedding.dimensions", &identity.dimensions.to_string())?;
    db.check_or_set_immutable("embedding.normalize", &identity.normalize.to_string())?;
    let model_id = db.find_or_create_model(&identity)?;

    let pending_total = db.count_unembedded_hashes(model_id)? as usize;
    let mut stats = EmbedStats {
        pending_total,
        ..EmbedStats::default()
    };
    let mut after_hash: Option<String> = None;
    loop {
        let pending = db.unembedded_hashes_page(
            model_id,
            after_hash.as_deref(),
            config.embedding.pending_page_size,
        )?;
        if pending.is_empty() {
            break;
        }
        after_hash = pending.last().map(|(hash, _)| hash.clone());

        let mut resolved: Vec<(String, String)> = Vec::with_capacity(pending.len());
        let mut missing: Vec<String> = Vec::new();
        for (hash, text) in pending {
            match text {
                Some(text) => resolved.push((hash, text)),
                None => missing.push(hash),
            }
        }
        if !missing.is_empty() {
            let recovered = recover_texts_from_source(db, config, &missing)?;
            for hash in missing {
                match recovered.get(&hash) {
                    Some(text) => resolved.push((hash, text.clone())),
                    None => stats.unresolved += 1,
                }
            }
        }
        // Pack length-sorted items so every batch's padded token area
        // (count x longest-item-tokens^2, the term ONNX attention memory
        // scales with) stays under budget. Token counts come from the
        // model's own tokenizer when available — the chars/4 estimate
        // undershoots real counts enough to blow the budget. Hash order
        // only matters for the page cursor above, not within a page.
        let max_sequence_length = embedder.max_sequence_length();
        let mut sized: Vec<(String, String, usize)> = resolved
            .into_iter()
            .map(|(hash, text)| {
                let tokens = embedder
                    .count_tokens(&text)
                    .unwrap_or_else(|| estimated_tokens(&text, max_sequence_length))
                    .min(max_sequence_length.max(1));
                (hash, text, tokens)
            })
            .collect();
        sized.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));

        for batch in pack_batches(
            &sized,
            config.embedding.batch_size,
            config.embedding.max_batch_chars,
            config.embedding.max_batch_token_area,
        ) {
            stats
                .tokens
                .record_batch(batch.iter().map(|(_, _, t)| *t), max_sequence_length);
            let texts: Vec<String> = batch.iter().map(|(_, text, _)| text.clone()).collect();
            let vectors = embedder.embed(&texts)?;
            anyhow::ensure!(
                vectors.len() == batch.len(),
                "embedder returned {} vectors for {} inputs",
                vectors.len(),
                batch.len()
            );
            for ((hash, _, _), mut vector) in batch.iter().zip(vectors) {
                anyhow::ensure!(
                    vector.len() == identity.dimensions,
                    "model {} returned {} dimensions, expected {}",
                    identity.model,
                    vector.len(),
                    identity.dimensions
                );
                if identity.normalize {
                    normalize_in_place(&mut vector);
                }
                db.insert_embedding(model_id, hash, &vector)?;
                stats.embedded += 1;
            }
            stats.batches += 1;
            progress(EmbedProgress {
                pending_total: stats.pending_total,
                embedded: stats.embedded,
                unresolved: stats.unresolved,
                batches: stats.batches,
                current_batch: batch.len(),
            });
        }
    }
    Ok(stats)
}

/// The stored model row for this identity, creating it if this is the first
/// run for the identity (mirrors the immutability-checked `embed_pending`
/// path, which also creates the row on first use).
pub fn find_or_create_model_id(db: &Db, identity: &ModelIdentity) -> Result<ModelId> {
    db.find_or_create_model(identity)
}

/// One language's untruncated token-length distribution.
#[derive(Debug, Clone)]
pub struct LanguageTokens {
    pub language: String,
    pub stats: TokenStats,
}

/// Measure the true (untruncated) token-length distribution of every indexed
/// unit body, bucketed by language, using the model's own tokenizer. Unlike
/// the embed-time stats this scans all units regardless of embedding state
/// (so it works on already-embedded DBs) and recovers text from source under
/// report/minimal retention. Units whose text cannot be recovered, or whose
/// backend cannot count tokens, are skipped.
pub fn token_report(
    db: &Db,
    config: &EmbeddingRunConfig,
    embedder: &dyn Embedder,
) -> Result<Vec<LanguageTokens>> {
    let rows = db.all_unit_texts()?;
    let missing: Vec<String> = rows
        .iter()
        .filter(|(_, _, text)| text.is_none())
        .map(|(hash, _, _)| hash.clone())
        .collect();
    let recovered = if missing.is_empty() {
        HashMap::new()
    } else {
        recover_texts_from_source(db, config, &missing)?
    };

    let mut by_language: BTreeMap<String, TokenStats> = BTreeMap::new();
    for (hash, language, text) in rows {
        let text = match text {
            Some(text) => text,
            None => match recovered.get(&hash) {
                Some(text) => text.clone(),
                None => continue,
            },
        };
        let Some(tokens) = embedder.count_tokens_untruncated(&text) else {
            continue;
        };
        by_language
            .entry(language)
            .or_default()
            .record_length(tokens);
    }
    Ok(by_language
        .into_iter()
        .map(|(language, stats)| LanguageTokens { language, stats })
        .collect())
}

/// In `report`/`minimal` retention the embedding text is not stored;
/// recover it by re-extracting the source files that contain the pending
/// hashes. Files that changed since indexing simply fail to recover their
/// hash and are picked up on the next index+embed cycle.
fn recover_texts_from_source(
    db: &Db,
    config: &EmbeddingRunConfig,
    hashes: &[String],
) -> Result<HashMap<String, String>> {
    let wanted: HashSet<&str> = hashes.iter().map(|s| s.as_str()).collect();
    let locations = db.locations_for_hashes(hashes)?;
    let options = ExtractOptions {
        body_node_count_threshold: config.source_recovery.body_node_count_threshold,
        max_body_chars: config.embedding.max_body_chars,
    };
    let registry = LanguageRegistry::global();
    let mut recovered = HashMap::new();
    for location in locations {
        let path = std::path::Path::new(&location.source_dir).join(&location.relative_path);
        let Ok(source) = std::fs::read_to_string(&path) else {
            continue;
        };
        let def = registry
            .get(&location.language_id)
            .with_context(|| format!("unknown language {}", location.language_id))?;
        for entity in extract_units(def, &source, &options)? {
            let unit = NewCodeUnit::from(entity);
            if wanted.contains(unit.normalized_body_hash.as_str())
                && let Some(text) = unit.embedding_text
            {
                recovered.insert(unit.normalized_body_hash, text);
            }
        }
    }
    Ok(recovered)
}
