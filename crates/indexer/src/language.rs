//! Single point of truth for mapping a file extension to a language id.
//!
//! Two families of frontend exist: grammar-driven languages in
//! `codeindex-tree-sitter`, and grammar-free prose formats in
//! `codeindex-prose`. Every provider that turns paths into
//! [`crate::SourceDocument`]s must resolve extensions the same way, so the
//! lookup lives here rather than being repeated per provider.

use anyhow::{Context, Result};
use codeindex_core::ExtractedEntity;
use codeindex_prose::{ProseFormat, ProseOptions, extract_prose};
use codeindex_tree_sitter::{ExtractOptions, LanguageRegistry, extract_units};

/// Resolve a **lowercase** extension to a language id, preferring a compiled
/// grammar over prose when both would match.
pub fn language_for_extension(extension: &str) -> Option<String> {
    if let Some(def) = LanguageRegistry::global().by_extension(extension) {
        return Some(def.spec.id.clone());
    }
    ProseFormat::from_extension(extension).map(|format| format.id().to_string())
}

/// Every language id the indexer can handle, grammars plus prose.
pub fn all_language_ids() -> Vec<String> {
    LanguageRegistry::global()
        .ids()
        .map(str::to_string)
        .chain(ProseFormat::all().iter().map(|f| f.id().to_string()))
        .collect()
}

/// True when the id names a grammar-free prose format.
pub fn is_prose(language_id: &str) -> bool {
    ProseFormat::from_id(language_id).is_some()
}

/// Extract units from one document, dispatching to the grammar frontend or the
/// prose frontend by language id.
///
/// Every site that turns source text back into entities must go through here.
/// Indexing and the embedding-time source-recovery path both do, so a prose
/// document recovered under report/minimal retention re-extracts identically to
/// the way it was first indexed — which is what makes its content hashes match.
pub(crate) fn extract_entities(
    language_id: &str,
    source: &str,
    options: &ExtractOptions,
    prose: &ProseOptions,
) -> Result<Vec<ExtractedEntity>> {
    match ProseFormat::from_id(language_id) {
        Some(format) => extract_prose(format, source, prose),
        None => {
            let definition = LanguageRegistry::global()
                .get(language_id)
                .with_context(|| format!("unknown language {language_id}"))?;
            extract_units(definition, source, options)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grammars_resolve() {
        assert_eq!(language_for_extension("rs").as_deref(), Some("rust"));
        assert_eq!(language_for_extension("py").as_deref(), Some("python"));
    }

    #[test]
    fn prose_resolves() {
        assert_eq!(language_for_extension("md").as_deref(), Some("markdown"));
        assert_eq!(language_for_extension("txt").as_deref(), Some("text"));
    }

    #[test]
    fn unknown_extension_is_none() {
        assert!(language_for_extension("bin").is_none());
    }

    #[test]
    fn all_ids_include_both_families() {
        let ids = all_language_ids();
        assert!(ids.iter().any(|id| id == "rust"));
        assert!(ids.iter().any(|id| id == "markdown"));
        assert!(ids.iter().any(|id| id == "text"));
    }

    #[test]
    fn prose_predicate_matches_only_prose() {
        assert!(is_prose("markdown"));
        assert!(is_prose("text"));
        assert!(!is_prose("rust"));
    }
}
