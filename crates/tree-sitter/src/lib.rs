#![forbid(unsafe_code)]

pub use codeindex_core;

mod extractor;
pub mod language;
pub mod normalizer;

pub use extractor::{ExtractOptions, RawReference, extract_references, extract_units};
pub use language::{LanguageDef, LanguageRegistry, LanguageSpec, ScopeRule};

/// Stable identity of extraction semantics used by resumable indexing.
/// Increment the suffix whenever bundled queries, normalization, or adapters
/// change in a way that can alter a staged document payload.
pub const FRONTEND_VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), ":2");

/// Language identifiers compiled into this frontend crate.
pub const BUNDLED_LANGUAGE_IDS: &[&str] = &[
    "c",
    "cpp",
    "csharp",
    "go",
    "java",
    "javascript",
    "kotlin",
    "php",
    "python",
    "ruby",
    "rust",
    "typescript",
];
