//! Embedding a stored corpus into one or more independently queryable spaces.

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Context, Result};
use codeindex_core::{EmbeddingSpaceId, EmbeddingSpaceIdentity, ModelIdentity, RepresentationKind};
use codeindex_embedding::config::EmbeddingRunConfig;
use codeindex_embedding::{
    Embedder, TokenStats, estimated_tokens, normalize_in_place, pack_batches,
};
use codeindex_sqlite::{Db, ModelId};
use codeindex_tree_sitter::{ExtractOptions, LanguageRegistry, extract_units};

use crate::SourceProviderCatalog;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct EmbedStats {
    pub pending_total: usize,
    pub embedded: usize,
    pub unresolved: usize,
    pub batches: usize,
    pub spaces: usize,
    pub tokens: TokenStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedProgress {
    pub space_id: EmbeddingSpaceId,
    pub pending_total: usize,
    pub embedded: usize,
    pub unresolved: usize,
    pub batches: usize,
    pub current_batch: usize,
}

/// Convenience projection: create one `default/<channel>` space for every
/// embeddable representation channel using the same model.
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
    let mut total = EmbedStats::default();
    for channel in db.embeddable_channels()? {
        let identity = EmbeddingSpaceIdentity::new(
            EmbeddingSpaceId::new(format!("default/{channel}")),
            channel,
            embedder.identity().clone(),
        );
        let stats = embed_space_pending_with_progress(
            db,
            embedder,
            config,
            &identity,
            None,
            &mut progress,
        )?;
        merge_stats(&mut total, stats);
    }
    Ok(total)
}

/// Embed one explicit space. Different spaces may use different models even
/// when they target the same representation channel.
pub fn embed_space_pending(
    db: &Db,
    embedder: &mut dyn Embedder,
    config: &EmbeddingRunConfig,
    space: &EmbeddingSpaceIdentity,
) -> Result<EmbedStats> {
    embed_space_pending_with_progress(db, embedder, config, space, None, &mut |_| {})
}

/// Embed one explicit space with provider-backed source recovery and progress.
pub fn embed_space_pending_with_progress(
    db: &Db,
    embedder: &mut dyn Embedder,
    config: &EmbeddingRunConfig,
    space: &EmbeddingSpaceIdentity,
    sources: Option<&SourceProviderCatalog<'_>>,
    progress: &mut impl FnMut(EmbedProgress),
) -> Result<EmbedStats> {
    anyhow::ensure!(
        embedder.identity() == &space.model,
        "embedder identity does not match embedding space {}",
        space.id
    );
    anyhow::ensure!(
        space.input_transform == "identity",
        "embedding space {} requests unsupported input transform {:?}",
        space.id,
        space.input_transform
    );
    db.find_or_create_space(space)?;

    let mut stats = EmbedStats {
        pending_total: db.count_unembedded_hashes(&space.id)? as usize,
        spaces: 1,
        ..EmbedStats::default()
    };
    embed_space(db, embedder, config, space, sources, &mut stats, progress)?;
    Ok(stats)
}

fn embed_space(
    db: &Db,
    embedder: &mut dyn Embedder,
    config: &EmbeddingRunConfig,
    space: &EmbeddingSpaceIdentity,
    sources: Option<&SourceProviderCatalog<'_>>,
    stats: &mut EmbedStats,
    progress: &mut impl FnMut(EmbedProgress),
) -> Result<()> {
    let mut after_hash: Option<String> = None;
    loop {
        let pending = db.unembedded_hashes_page(
            &space.id,
            after_hash.as_deref(),
            config.embedding.pending_page_size,
        )?;
        if pending.is_empty() {
            break;
        }
        after_hash = pending.last().map(|(hash, _)| hash.clone());

        let mut resolved = Vec::with_capacity(pending.len());
        let mut missing = Vec::new();
        for (hash, text) in pending {
            match text {
                Some(text) => resolved.push((hash, text)),
                None => missing.push(hash),
            }
        }
        if !missing.is_empty() {
            let recovered =
                recover_channel_texts(db, config, &space.identity_channel(), &missing, sources)?;
            for hash in missing {
                match recovered.get(&hash) {
                    Some(text) => resolved.push((hash, text.clone())),
                    None => stats.unresolved += 1,
                }
            }
        }

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
        sized.sort_by(|left, right| right.2.cmp(&left.2).then(left.0.cmp(&right.0)));

        for batch in pack_batches(
            &sized,
            config.embedding.batch_size,
            config.embedding.max_batch_chars,
            config.embedding.max_batch_token_area,
        ) {
            stats.tokens.record_batch(
                batch.iter().map(|(_, _, tokens)| *tokens),
                max_sequence_length,
            );
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
                    vector.len() == space.model.dimensions,
                    "model {} returned {} dimensions, expected {}",
                    space.model.model,
                    vector.len(),
                    space.model.dimensions
                );
                if space.model.normalize {
                    normalize_in_place(&mut vector);
                }
                db.insert_embedding(&space.id, hash, &vector)?;
                stats.embedded += 1;
            }
            stats.batches += 1;
            progress(EmbedProgress {
                space_id: space.id.clone(),
                pending_total: stats.pending_total,
                embedded: stats.embedded,
                unresolved: stats.unresolved,
                batches: stats.batches,
                current_batch: batch.len(),
            });
        }
    }
    Ok(())
}

trait SpaceChannel {
    fn identity_channel(&self) -> RepresentationKind;
}

impl SpaceChannel for EmbeddingSpaceIdentity {
    fn identity_channel(&self) -> RepresentationKind {
        self.channel.clone()
    }
}

fn merge_stats(total: &mut EmbedStats, stats: EmbedStats) {
    total.pending_total += stats.pending_total;
    total.embedded += stats.embedded;
    total.unresolved += stats.unresolved;
    total.batches += stats.batches;
    total.spaces += stats.spaces;
    total.tokens.merge(&stats.tokens);
}

pub fn find_or_create_model_id(db: &Db, identity: &ModelIdentity) -> Result<ModelId> {
    db.find_or_create_model(identity)
}

#[derive(Debug, Clone)]
pub struct LanguageTokens {
    pub language: String,
    pub stats: TokenStats,
}

pub fn token_report(
    db: &Db,
    config: &EmbeddingRunConfig,
    embedder: &dyn Embedder,
) -> Result<Vec<LanguageTokens>> {
    token_report_channel(
        db,
        config,
        embedder,
        &RepresentationKind::Implementation,
        None,
    )
}

pub fn token_report_channel(
    db: &Db,
    config: &EmbeddingRunConfig,
    embedder: &dyn Embedder,
    channel: &RepresentationKind,
    sources: Option<&SourceProviderCatalog<'_>>,
) -> Result<Vec<LanguageTokens>> {
    let rows = db.channel_texts(channel)?;
    let missing: Vec<String> = rows
        .iter()
        .filter(|(_, _, text)| text.is_none())
        .map(|(hash, _, _)| hash.clone())
        .collect();
    let recovered = if missing.is_empty() {
        HashMap::new()
    } else {
        recover_channel_texts(db, config, channel, &missing, sources)?
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

fn recover_channel_texts(
    db: &Db,
    config: &EmbeddingRunConfig,
    channel: &RepresentationKind,
    hashes: &[String],
    sources: Option<&SourceProviderCatalog<'_>>,
) -> Result<HashMap<String, String>> {
    let wanted: HashSet<&str> = hashes.iter().map(String::as_str).collect();
    let locations = db.locations_for_content_hashes(channel, hashes)?;
    let options = ExtractOptions {
        body_node_count_threshold: config.source_recovery.body_node_count_threshold,
        max_body_chars: config.embedding.max_body_chars,
    };
    let registry = LanguageRegistry::global();
    let mut recovered = HashMap::new();
    for location in locations {
        let source =
            if let Some(blob) = db.get_source_blob(&location.source_hash)? {
                Some(String::from_utf8(blob.content).with_context(|| {
                    format!("cached source {} is not UTF-8", location.source_hash)
                })?)
            } else if let Some(catalog) = sources {
                catalog.read(
                    &location.project_label,
                    &location.source_document_id,
                    &location.source_revision,
                )?
            } else {
                None
            };
        let Some(source) = source else { continue };
        anyhow::ensure!(
            codeindex_tree_sitter::normalizer::sha256_hex(&source) == location.source_hash,
            "source cache hash mismatch for {}",
            location.source_document_id
        );
        let def = registry
            .get(&location.language_id)
            .with_context(|| format!("unknown language {}", location.language_id))?;
        for mut entity in extract_units(def, &source, &options)? {
            super::complete_representations(&source, &mut entity);
            if let Some(representation) = entity.representation(channel)
                && wanted.contains(representation.content_hash.as_str())
            {
                recovered.insert(
                    representation.content_hash.clone(),
                    representation.content.clone(),
                );
            }
        }
    }
    Ok(recovered)
}
