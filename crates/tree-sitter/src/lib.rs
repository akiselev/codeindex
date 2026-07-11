#![forbid(unsafe_code)]

pub use codeindex_core;

mod extractor;
pub mod language;
pub mod normalizer;

pub use extractor::{ExtractOptions, RawReference, extract_file, extract_references, extract_units};
pub use language::{LanguageDef, LanguageRegistry, LanguageSpec, ScopeRule};

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

#[cfg(test)]
pub(crate) mod config {
    pub const KNOWN_LANGUAGE_IDS: &[&str] = super::BUNDLED_LANGUAGE_IDS;
}
