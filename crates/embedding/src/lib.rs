#![forbid(unsafe_code)]

//! Local embedding model execution and the reusable primitives around it:
//! the [`EmbeddingBackend`] trait, typed [`EmbedRequest`] rendering through
//! per-model prompt contracts, generic model resolution from HuggingFace
//! repositories, batch packing, normalization, and token instrumentation.
//! This crate is deliberately free of storage and parser dependencies so it
//! can back a lightweight notebook binding; the workflow that embeds a
//! *stored corpus* (source recovery, resumable projection, token reports)
//! lives in `codeindex-indexer`.

pub mod config;
pub mod rerank;
pub mod resolve;

pub mod embed;
pub use embed::*;
