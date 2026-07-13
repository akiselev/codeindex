use std::collections::{BTreeMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result, bail};
use codeindex_tree_sitter::LanguageRegistry;
use serde::{Deserialize, Serialize};

use crate::scanner::scan_files;

/// Provider-defined revision metadata used for cheap change detection before
/// source content is read and hashed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRevision {
    /// Opaque provider revision token. Equal tokens mean equal content only when
    /// the provider documents that guarantee; the indexer still hashes content
    /// after a revision change.
    pub opaque: String,
    pub modified_ns: Option<i64>,
    pub size: Option<u64>,
}

impl SourceRevision {
    pub fn new(opaque: impl Into<String>) -> Self {
        Self {
            opaque: opaque.into(),
            modified_ns: None,
            size: None,
        }
    }
}

/// Metadata for one logical source document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceDocument {
    /// Stable provider-local identity. It need not equal the display path.
    pub id: String,
    /// Forward-slash logical path used for display, filtering, and language
    /// detection. Providers may change this while preserving `id`.
    pub relative_path: String,
    pub language_id: String,
    pub revision: SourceRevision,
}

/// Whether equality of provider revisions proves equality of source bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RevisionSemantics {
    Authoritative,
    Advisory,
}

/// Result of a stable read. Mutable providers can ask the runner to refresh the
/// manifest instead of turning an ordinary concurrent save into an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StableRead {
    Content {
        source: String,
        revision: SourceRevision,
    },
    Changed,
}

/// Arbitrary source of source code. Implementations can read a filesystem,
/// database, object store, Git tree, archive, editor overlay, or generated
/// in-memory corpus.
pub trait SourceProvider: Send + Sync {
    /// Opaque project locator persisted with the project for diagnostics and
    /// default source recovery. It is not required to be a path.
    fn project_locator(&self) -> String;

    /// Stable identity of this provider implementation/configuration. The
    /// locator is a conservative default; providers with behavior-affecting
    /// configuration should override it.
    fn provider_fingerprint(&self) -> String {
        self.project_locator()
    }

    fn revision_semantics(&self) -> RevisionSemantics {
        RevisionSemantics::Advisory
    }

    /// Enumerate enabled source documents in deterministic path order.
    fn documents(&self, enabled_languages: &HashSet<String>) -> Result<Vec<SourceDocument>>;

    /// Read one enumerated document as UTF-8 source.
    fn read(&self, document: &SourceDocument) -> Result<String>;

    /// Read one observation. Providers that can change between enumeration and
    /// read should return `Changed` when they detect that race.
    fn stable_read(&self, document: &SourceDocument) -> Result<StableRead> {
        Ok(StableRead::Content {
            source: self.read(document)?,
            revision: document.revision.clone(),
        })
    }
}

/// A labelled provider passed to [`crate::index_sources`].
pub struct SourceProject<'a> {
    pub label: String,
    pub provider: &'a dyn SourceProvider,
}

/// Provider lookup used to recover representation text under report/minimal
/// retention during a later embedding run.
#[derive(Default)]
pub struct SourceProviderCatalog<'a> {
    providers: BTreeMap<String, &'a dyn SourceProvider>,
}

impl<'a> SourceProviderCatalog<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        project_label: impl Into<String>,
        provider: &'a dyn SourceProvider,
    ) -> Option<&'a dyn SourceProvider> {
        self.providers.insert(project_label.into(), provider)
    }

    pub fn provider(&self, project_label: &str) -> Option<&'a dyn SourceProvider> {
        self.providers.get(project_label).copied()
    }

    pub fn read(
        &self,
        project_label: &str,
        document_id: &str,
        relative_path: &str,
        language_id: &str,
    ) -> Result<Option<String>> {
        let Some(provider) = self.provider(project_label) else {
            return Ok(None);
        };
        let enabled = HashSet::from([language_id.to_string()]);
        let document = provider
            .documents(&enabled)?
            .into_iter()
            .find(|document| document.id == document_id)
            .or_else(|| {
                provider
                    .documents(&enabled)
                    .ok()?
                    .into_iter()
                    .find(|document| document.relative_path == relative_path)
            });
        match document {
            Some(document) => provider.read(&document).map(Some),
            None => Ok(None),
        }
    }
}

/// Default filesystem implementation used by the compatibility `index()` API.
#[derive(Debug, Clone)]
pub struct FileSystemSource {
    root: PathBuf,
    exclude: Vec<String>,
}

impl FileSystemSource {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            exclude: Vec::new(),
        }
    }

    pub fn with_excludes(mut self, exclude: Vec<String>) -> Self {
        self.exclude = exclude;
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl SourceProvider for FileSystemSource {
    fn project_locator(&self) -> String {
        self.root.to_string_lossy().into_owned()
    }

    fn provider_fingerprint(&self) -> String {
        format!("{}\0{}", self.project_locator(), self.exclude.join("\0"))
    }

    fn documents(&self, enabled_languages: &HashSet<String>) -> Result<Vec<SourceDocument>> {
        scan_files(&self.root, &self.exclude, enabled_languages)?
            .into_iter()
            .map(|file| {
                let metadata = std::fs::metadata(&file.absolute_path)
                    .with_context(|| format!("failed to stat {}", file.absolute_path.display()))?;
                let modified_ns = metadata
                    .modified()
                    .ok()
                    .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                    .map(|duration| duration.as_nanos() as i64);
                let size = metadata.len();
                Ok(SourceDocument {
                    id: file.relative_path.clone(),
                    relative_path: file.relative_path,
                    language_id: file.language_id,
                    revision: SourceRevision {
                        opaque: format!("{}:{}", modified_ns.unwrap_or_default(), size),
                        modified_ns,
                        size: Some(size),
                    },
                })
            })
            .collect()
    }

    fn read(&self, document: &SourceDocument) -> Result<String> {
        std::fs::read_to_string(self.root.join(&document.relative_path)).with_context(|| {
            format!(
                "failed to read {} from filesystem source {}",
                document.relative_path,
                self.root.display()
            )
        })
    }

    fn stable_read(&self, document: &SourceDocument) -> Result<StableRead> {
        let path = self.root.join(&document.relative_path);
        let mut file = match std::fs::File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(StableRead::Changed);
            }
            Err(error) => {
                return Err(error).with_context(|| format!("failed to open {}", path.display()));
            }
        };
        let before = file
            .metadata()
            .with_context(|| format!("failed to stat {}", path.display()))?;
        let mut source = String::new();
        file.read_to_string(&mut source)
            .with_context(|| format!("failed to read {} as UTF-8", path.display()))?;
        let after = file
            .metadata()
            .with_context(|| format!("failed to restat {}", path.display()))?;
        let before_revision = filesystem_revision(&before);
        let after_revision = filesystem_revision(&after);
        if before_revision != document.revision || before_revision != after_revision {
            return Ok(StableRead::Changed);
        }
        Ok(StableRead::Content {
            source,
            revision: after_revision,
        })
    }
}

/// Small public provider useful for tests, generated sources, and editor
/// overlays. Replacing an entry changes its deterministic content revision.
#[derive(Debug, Clone, Default)]
pub struct MemorySource {
    locator: String,
    documents: BTreeMap<String, String>,
}

impl MemorySource {
    pub fn new(locator: impl Into<String>) -> Self {
        Self {
            locator: locator.into(),
            documents: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, path: impl Into<String>, content: impl Into<String>) {
        self.documents.insert(path.into(), content.into());
    }

    pub fn remove(&mut self, path: &str) -> Option<String> {
        self.documents.remove(path)
    }
}

impl SourceProvider for MemorySource {
    fn project_locator(&self) -> String {
        self.locator.clone()
    }

    fn revision_semantics(&self) -> RevisionSemantics {
        RevisionSemantics::Authoritative
    }

    fn documents(&self, enabled_languages: &HashSet<String>) -> Result<Vec<SourceDocument>> {
        let registry = LanguageRegistry::global();
        let mut documents = Vec::new();
        for (path, content) in &self.documents {
            let extension = Path::new(path)
                .extension()
                .and_then(|extension| extension.to_str())
                .unwrap_or_default()
                .to_lowercase();
            let Some(language) = registry.by_extension(&extension) else {
                continue;
            };
            if !enabled_languages.contains(&language.spec.id) {
                continue;
            }
            documents.push(SourceDocument {
                id: path.clone(),
                relative_path: path.clone(),
                language_id: language.spec.id.clone(),
                revision: SourceRevision {
                    opaque: codeindex_tree_sitter::normalizer::sha256_hex(content),
                    modified_ns: None,
                    size: Some(content.len() as u64),
                },
            });
        }
        Ok(documents)
    }

    fn read(&self, document: &SourceDocument) -> Result<String> {
        self.documents
            .get(&document.id)
            .cloned()
            .or_else(|| self.documents.get(&document.relative_path).cloned())
            .with_context(|| format!("memory source has no document {:?}", document.id))
    }
}

fn filesystem_revision(metadata: &std::fs::Metadata) -> SourceRevision {
    let modified_ns = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos() as i64);
    let size = metadata.len();
    SourceRevision {
        opaque: format!("{}:{size}", modified_ns.unwrap_or_default()),
        modified_ns,
        size: Some(size),
    }
}

/// Validate the minimum invariants the indexer relies on for every provider.
pub(crate) fn validate_documents(documents: &[SourceDocument]) -> Result<()> {
    let mut ids = HashSet::new();
    let mut paths = HashSet::new();
    for document in documents {
        if document.id.is_empty() {
            bail!("source provider returned an empty document id");
        }
        if document.relative_path.is_empty() {
            bail!("source provider returned an empty relative path");
        }
        if !ids.insert(document.id.as_str()) {
            bail!(
                "source provider returned duplicate document id {:?}",
                document.id
            );
        }
        if !paths.insert(document.relative_path.as_str()) {
            bail!(
                "source provider returned duplicate relative path {:?}",
                document.relative_path
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_source_filters_languages_and_reads_by_stable_id() {
        let mut source = MemorySource::new("memory://test");
        source.insert("src/lib.rs", "fn answer() -> i32 { 42 }");
        source.insert("README.md", "ignored");
        let enabled = HashSet::from(["rust".to_string()]);
        let documents = source.documents(&enabled).unwrap();
        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].id, "src/lib.rs");
        assert!(source.read(&documents[0]).unwrap().contains("answer"));
    }
}
