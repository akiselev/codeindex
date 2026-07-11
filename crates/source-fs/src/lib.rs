#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs::Metadata;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use codeindex_source::{
    DocumentDescriptor, DocumentId, DocumentIter, DocumentLocation, DocumentMetadata,
    DocumentQuery, DocumentVersion, LanguageHint, RevisionGuarantee, RevisionToken,
    SnapshotConsistency, SnapshotId, SnapshotRequest, SourceCapabilities, SourceContent,
    SourceError, SourceErrorKind, SourceKind, SourceRootId, SourceSnapshot, SourceWorkspace,
    WorkspaceDescriptor, WorkspaceId,
};
use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;

#[derive(Clone, Debug)]
pub struct FilesystemWorkspace {
    root: PathBuf,
    exclude: Vec<String>,
    follow_symlinks: bool,
    descriptor: WorkspaceDescriptor,
}

impl FilesystemWorkspace {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self::builder(root).build()
    }

    pub fn builder(root: impl Into<PathBuf>) -> FilesystemWorkspaceBuilder {
        FilesystemWorkspaceBuilder {
            root: root.into(),
            exclude: Vec::new(),
            follow_symlinks: false,
            workspace_id: None,
            display_name: None,
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

#[derive(Clone, Debug)]
pub struct FilesystemWorkspaceBuilder {
    root: PathBuf,
    exclude: Vec<String>,
    follow_symlinks: bool,
    workspace_id: Option<WorkspaceId>,
    display_name: Option<String>,
}

impl FilesystemWorkspaceBuilder {
    pub fn excludes(mut self, exclude: Vec<String>) -> Self {
        self.exclude = exclude;
        self
    }

    pub fn follow_symlinks(mut self, follow: bool) -> Self {
        self.follow_symlinks = follow;
        self
    }

    pub fn workspace_id(mut self, id: impl Into<WorkspaceId>) -> Self {
        self.workspace_id = Some(id.into());
        self
    }

    pub fn display_name(mut self, display_name: impl Into<String>) -> Self {
        self.display_name = Some(display_name.into());
        self
    }

    pub fn build(self) -> FilesystemWorkspace {
        let locator = self.root.to_string_lossy().into_owned();
        FilesystemWorkspace {
            root: self.root,
            exclude: self.exclude,
            follow_symlinks: self.follow_symlinks,
            descriptor: WorkspaceDescriptor {
                id: self
                    .workspace_id
                    .unwrap_or_else(|| WorkspaceId::new(format!("fs:{locator}"))),
                display_name: self.display_name.unwrap_or_else(|| locator.clone()),
                source_kind: SourceKind::FileSystem,
                redacted_locator: Some(locator),
            },
        }
    }
}

impl SourceWorkspace for FilesystemWorkspace {
    fn descriptor(&self) -> WorkspaceDescriptor {
        self.descriptor.clone()
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            streaming_enumeration: true,
            batch_read: false,
            exact_revision_reads: false,
            change_feed: false,
        }
    }

    fn open_snapshot(
        &self,
        _request: &SnapshotRequest,
    ) -> Result<Arc<dyn SourceSnapshot>, SourceError> {
        let documents = scan_documents(&self.root, &self.exclude, self.follow_symlinks)?;
        let fingerprint = snapshot_fingerprint(documents.values());
        Ok(Arc::new(FilesystemSnapshot {
            id: SnapshotId::new(format!("{}@{fingerprint}", self.descriptor.id)),
            workspace_id: self.descriptor.id.clone(),
            root: self.root.clone(),
            documents,
        }))
    }
}

struct FilesystemSnapshot {
    id: SnapshotId,
    workspace_id: WorkspaceId,
    root: PathBuf,
    documents: BTreeMap<DocumentId, DocumentDescriptor>,
}

impl SourceSnapshot for FilesystemSnapshot {
    fn id(&self) -> &SnapshotId {
        &self.id
    }

    fn workspace_id(&self) -> &WorkspaceId {
        &self.workspace_id
    }

    fn consistency(&self) -> SnapshotConsistency {
        SnapshotConsistency::Validated
    }

    fn documents<'a>(&'a self, query: &'a DocumentQuery) -> Result<DocumentIter<'a>, SourceError> {
        Ok(Box::new(
            self.documents
                .values()
                .filter(move |&document| query.matches(document))
                .cloned()
                .map(Ok),
        ))
    }

    fn document(&self, id: &DocumentId) -> Result<Option<DocumentDescriptor>, SourceError> {
        Ok(self.documents.get(id).cloned())
    }

    fn read(&self, document: &DocumentDescriptor) -> Result<SourceContent, SourceError> {
        let indexed = self.documents.get(&document.id).ok_or_else(|| {
            SourceError::not_found(format!("unknown source document {}", document.id))
        })?;
        if indexed.version.token != document.version.token {
            return Err(SourceError::stale(format!(
                "source document {} was enumerated at revision {}, requested {}",
                document.id, indexed.version.token, document.version.token
            )));
        }
        let path = self.root.join(&indexed.location.logical_path);
        let before = std::fs::metadata(&path).map_err(|error| io_error(&path, error))?;
        let before_version = version_from_metadata(&before);
        if before_version.token != indexed.version.token {
            return Err(SourceError::stale(format!(
                "source document {} changed after snapshot enumeration",
                document.id
            )));
        }
        let bytes = std::fs::read(&path).map_err(|error| io_error(&path, error))?;
        let after = std::fs::metadata(&path).map_err(|error| io_error(&path, error))?;
        let after_version = version_from_metadata(&after);
        if before_version.token != after_version.token {
            return Err(SourceError::stale(format!(
                "source document {} changed while it was being read",
                document.id
            )));
        }
        Ok(SourceContent {
            bytes: Arc::from(bytes),
            observed_version: after_version,
            encoding_hint: None,
        })
    }
}

fn scan_documents(
    root: &Path,
    exclude: &[String],
    follow_symlinks: bool,
) -> Result<BTreeMap<DocumentId, DocumentDescriptor>, SourceError> {
    let mut overrides = OverrideBuilder::new(root);
    for pattern in exclude {
        overrides.add(&format!("!{pattern}")).map_err(|error| {
            SourceError::invalid(format!("bad exclude pattern {pattern:?}: {error}"))
        })?;
    }
    let overrides = overrides
        .build()
        .map_err(|error| SourceError::invalid(format!("invalid filesystem overrides: {error}")))?;
    let mut walk = WalkBuilder::new(root);
    walk.overrides(overrides)
        .require_git(false)
        .follow_links(follow_symlinks);
    let mut documents = BTreeMap::new();
    for entry in walk.build() {
        let entry = entry.map_err(|error| {
            SourceError::new(
                SourceErrorKind::Unavailable,
                format!("filesystem walk failed: {error}"),
            )
            .retryable(true)
        })?;
        if !entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
        {
            continue;
        }
        let path = entry.path();
        let relative = path.strip_prefix(root).map_err(|_| {
            SourceError::invalid(format!(
                "filesystem walker returned {} outside root {}",
                path.display(),
                root.display()
            ))
        })?;
        let logical_path = logical_path(relative);
        let id = DocumentId::new(logical_path.clone());
        let metadata = entry
            .metadata()
            .map_err(|error| SourceError::new(SourceErrorKind::Unavailable, error.to_string()))?;
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase());
        let descriptor = DocumentDescriptor {
            id: id.clone(),
            location: DocumentLocation {
                root: SourceRootId::new("root"),
                logical_path,
                display_uri: Some(path.to_string_lossy().into_owned()),
            },
            version: version_from_metadata(&metadata),
            language_hint: extension
                .map(LanguageHint::FileExtension)
                .unwrap_or(LanguageHint::Unknown),
            metadata: DocumentMetadata::default(),
        };
        if documents.insert(id.clone(), descriptor).is_some() {
            return Err(SourceError::invalid(format!(
                "filesystem snapshot produced duplicate document id {id}"
            )));
        }
    }
    Ok(documents)
}

fn logical_path(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn version_from_metadata(metadata: &Metadata) -> DocumentVersion {
    let modified_at = metadata.modified().ok();
    let modified_ns = modified_at
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let size = metadata.len();
    DocumentVersion {
        token: RevisionToken::new(format!("{modified_ns}:{size}")),
        guarantee: RevisionGuarantee::MetadataHint,
        modified_at,
        size: Some(size),
        content_hash: None,
    }
}

fn snapshot_fingerprint<'a>(documents: impl Iterator<Item = &'a DocumentDescriptor>) -> String {
    let mut count = 0_u64;
    let mut bytes = 0_u64;
    let mut latest = UNIX_EPOCH;
    for document in documents {
        count += 1;
        bytes = bytes.saturating_add(document.version.size.unwrap_or_default());
        if let Some(modified) = document.version.modified_at {
            latest = latest.max(modified);
        }
    }
    let latest_ns = latest
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("{count}:{bytes}:{latest_ns}")
}

fn io_error(path: &Path, error: std::io::Error) -> SourceError {
    let kind = match error.kind() {
        std::io::ErrorKind::NotFound => SourceErrorKind::NotFound,
        std::io::ErrorKind::PermissionDenied => SourceErrorKind::PermissionDenied,
        std::io::ErrorKind::WouldBlock
        | std::io::ErrorKind::Interrupted
        | std::io::ErrorKind::TimedOut => SourceErrorKind::Unavailable,
        _ => SourceErrorKind::Other,
    };
    SourceError::new(
        kind,
        format!("failed to access {}: {error}", path.display()),
    )
    .retryable(matches!(kind, SourceErrorKind::Unavailable))
}

#[cfg(test)]
mod tests {
    use super::*;
    use codeindex_source::{DocumentQuery, SnapshotRequest, SourceWorkspace, validate_snapshot};

    fn write(root: &Path, relative: &str, content: &str) {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    #[test]
    fn snapshots_respect_ignores_and_validate_reads() {
        let directory = tempfile::tempdir().unwrap();
        write(directory.path(), "src/lib.rs", "fn answer() -> i32 { 42 }");
        write(directory.path(), "vendor/skip.rs", "fn skip() {}");
        let workspace = FilesystemWorkspace::builder(directory.path())
            .excludes(vec!["vendor/**".to_string()])
            .build();
        let snapshot = workspace
            .open_snapshot(&SnapshotRequest::default())
            .unwrap();
        let documents: Vec<_> = snapshot
            .documents(&DocumentQuery::all())
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].location.logical_path, "src/lib.rs");
        validate_snapshot(snapshot.as_ref()).unwrap();
    }

    #[test]
    fn changed_files_are_rejected_by_old_snapshots() {
        let directory = tempfile::tempdir().unwrap();
        write(directory.path(), "src/lib.rs", "fn answer() -> i32 { 42 }");
        let workspace = FilesystemWorkspace::new(directory.path());
        let snapshot = workspace
            .open_snapshot(&SnapshotRequest::default())
            .unwrap();
        let document = snapshot
            .documents(&DocumentQuery::all())
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        std::fs::write(directory.path().join("src/lib.rs"), "different length").unwrap();
        assert_eq!(
            snapshot.read(&document).unwrap_err().kind(),
            SourceErrorKind::StaleRevision
        );
    }
}
