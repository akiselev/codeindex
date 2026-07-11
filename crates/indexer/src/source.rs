use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::Result;
use codeindex_source::{
    DocumentId, SnapshotRequest, SourceErrorKind, SourceSnapshot, SourceWorkspace,
};

pub use codeindex_source::{
    ContentHash, DocumentDescriptor, DocumentIter, DocumentLocation, DocumentMetadata,
    DocumentQuery, DocumentVersion, LanguageHint, MemoryWorkspace, OverlayWorkspace,
    RevisionGuarantee, SnapshotConsistency, SnapshotId, SourceCapabilities, SourceCheckpoint,
    SourceContent, SourceError, SourceKind, SourceRootId, WorkspaceDescriptor, WorkspaceId,
    validate_snapshot,
};
pub use codeindex_source::SourceWorkspace as SourceProvider;
pub use codeindex_source_fs::{FilesystemWorkspace, FilesystemWorkspaceBuilder};

pub type FileSystemSource = FilesystemWorkspace;
pub type MemorySource = MemoryWorkspace;
pub type SourceDocument = DocumentDescriptor;
pub type SourceRevision = DocumentVersion;

pub struct SourceProject<'a> {
    pub label: String,
    pub workspace: &'a dyn SourceWorkspace,
}

enum CatalogEntry<'a> {
    Workspace(&'a dyn SourceWorkspace),
    Snapshot(Arc<dyn SourceSnapshot>),
}

#[derive(Default)]
pub struct SourceProviderCatalog<'a> {
    providers: BTreeMap<String, CatalogEntry<'a>>,
}

impl<'a> SourceProviderCatalog<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        project_label: impl Into<String>,
        workspace: &'a dyn SourceWorkspace,
    ) {
        self.providers
            .insert(project_label.into(), CatalogEntry::Workspace(workspace));
    }

    pub fn insert_snapshot(
        &mut self,
        project_label: impl Into<String>,
        snapshot: Arc<dyn SourceSnapshot>,
    ) {
        self.providers
            .insert(project_label.into(), CatalogEntry::Snapshot(snapshot));
    }

    pub fn read(
        &self,
        project_label: &str,
        document_id: &str,
        _relative_path: &str,
        _language_id: &str,
    ) -> Result<Option<String>> {
        let Some(entry) = self.providers.get(project_label) else {
            return Ok(None);
        };
        let snapshot = match entry {
            CatalogEntry::Workspace(workspace) => {
                workspace.open_snapshot(&SnapshotRequest::default())?
            }
            CatalogEntry::Snapshot(snapshot) => snapshot.clone(),
        };
        let id = DocumentId::new(document_id);
        let Some(document) = snapshot.document(&id)? else {
            return Ok(None);
        };
        let content = match snapshot.read(&document) {
            Ok(content) => content,
            Err(error)
                if matches!(
                    error.kind(),
                    SourceErrorKind::NotFound | SourceErrorKind::StaleRevision
                ) =>
            {
                return Ok(None);
            }
            Err(error) => return Err(error.into()),
        };
        Ok(Some(content.utf8()?.to_owned()))
    }
}
