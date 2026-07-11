#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::str;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

macro_rules! string_id {
    ($name:ident) => {
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(Arc<str>);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(Arc::from(value.into()))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(self.as_str())
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self::new(value)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self::new(value)
            }
        }
    };
}

string_id!(WorkspaceId);
string_id!(SnapshotId);
string_id!(DocumentId);
string_id!(SourceRootId);
string_id!(RevisionToken);
string_id!(ContentHash);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceKind {
    FileSystem,
    Memory,
    Git,
    ObjectStore,
    Database,
    Archive,
    Editor,
    Generated,
    Custom(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RevisionGuarantee {
    ContentIdentity,
    MetadataHint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SnapshotConsistency {
    BestEffort,
    Validated,
    Transactional,
    Immutable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LanguageHint {
    Known(String),
    FileExtension(String),
    MediaType(String),
    Shebang(String),
    Unknown,
}

impl Default for LanguageHint {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DocumentMetadata {
    pub generated: bool,
    pub vendor: bool,
    pub test: bool,
    pub attributes: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceDescriptor {
    pub id: WorkspaceId,
    pub display_name: String,
    pub source_kind: SourceKind,
    pub redacted_locator: Option<String>,
}

impl WorkspaceDescriptor {
    pub fn persisted_locator(&self) -> String {
        self.redacted_locator
            .clone()
            .unwrap_or_else(|| self.id.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentLocation {
    pub root: SourceRootId,
    pub logical_path: String,
    pub display_uri: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentVersion {
    pub token: RevisionToken,
    pub guarantee: RevisionGuarantee,
    pub modified_at: Option<SystemTime>,
    pub size: Option<u64>,
    pub content_hash: Option<ContentHash>,
}

impl DocumentVersion {
    pub fn new(token: impl Into<RevisionToken>, guarantee: RevisionGuarantee) -> Self {
        Self {
            token: token.into(),
            guarantee,
            modified_at: None,
            size: None,
            content_hash: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentDescriptor {
    pub id: DocumentId,
    pub location: DocumentLocation,
    pub version: DocumentVersion,
    pub language_hint: LanguageHint,
    pub metadata: DocumentMetadata,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceCheckpoint {
    pub workspace: WorkspaceId,
    pub token: Arc<[u8]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotRequest {
    pub checkpoint: Option<SourceCheckpoint>,
}

impl Default for SnapshotRequest {
    fn default() -> Self {
        Self { checkpoint: None }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DocumentQuery {
    pub language_ids: BTreeSet<String>,
    pub path_prefixes: Vec<String>,
    pub include_generated: bool,
    pub include_vendor: bool,
}

impl DocumentQuery {
    pub fn all() -> Self {
        Self {
            include_generated: true,
            include_vendor: true,
            ..Self::default()
        }
    }

    pub fn matches(&self, document: &DocumentDescriptor) -> bool {
        if !self.include_generated && document.metadata.generated {
            return false;
        }
        if !self.include_vendor && document.metadata.vendor {
            return false;
        }
        if !self.path_prefixes.is_empty()
            && !self
                .path_prefixes
                .iter()
                .any(|prefix| document.location.logical_path.starts_with(prefix))
        {
            return false;
        }
        if self.language_ids.is_empty() {
            return true;
        }
        match &document.language_hint {
            LanguageHint::Known(language) => self.language_ids.contains(language),
            _ => true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SourceCapabilities {
    pub streaming_enumeration: bool,
    pub batch_read: bool,
    pub exact_revision_reads: bool,
    pub change_feed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceChange {
    Upsert(DocumentDescriptor),
    Remove(DocumentId),
    Move {
        id: DocumentId,
        new_location: DocumentLocation,
        version: DocumentVersion,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceErrorKind {
    NotFound,
    StaleRevision,
    PermissionDenied,
    RateLimited,
    Unavailable,
    InvalidData,
    Unsupported,
    Other,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceError {
    kind: SourceErrorKind,
    message: String,
    retryable: bool,
}

impl SourceError {
    pub fn new(kind: SourceErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            retryable: false,
        }
    }

    pub fn retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }

    pub fn kind(&self) -> SourceErrorKind {
        self.kind
    }

    pub fn is_retryable(&self) -> bool {
        self.retryable
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(SourceErrorKind::NotFound, message)
    }

    pub fn stale(message: impl Into<String>) -> Self {
        Self::new(SourceErrorKind::StaleRevision, message)
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(SourceErrorKind::InvalidData, message)
    }
}

impl fmt::Display for SourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for SourceError {}

#[derive(Clone, Debug)]
pub struct SourceContent {
    pub bytes: Arc<[u8]>,
    pub observed_version: DocumentVersion,
    pub encoding_hint: Option<String>,
}

impl SourceContent {
    pub fn utf8(&self) -> Result<&str, SourceError> {
        str::from_utf8(&self.bytes)
            .map_err(|error| SourceError::invalid(format!("source is not UTF-8: {error}")))
    }
}

pub type DocumentIter<'a> =
    Box<dyn Iterator<Item = Result<DocumentDescriptor, SourceError>> + Send + 'a>;
pub type ChangeIter<'a> = Box<dyn Iterator<Item = Result<SourceChange, SourceError>> + Send + 'a>;

pub trait SourceSnapshot: Send + Sync {
    fn id(&self) -> &SnapshotId;

    fn workspace_id(&self) -> &WorkspaceId;

    fn consistency(&self) -> SnapshotConsistency;

    fn checkpoint(&self) -> Option<&SourceCheckpoint> {
        None
    }

    fn documents<'a>(&'a self, query: &'a DocumentQuery) -> Result<DocumentIter<'a>, SourceError>;

    fn document(&self, id: &DocumentId) -> Result<Option<DocumentDescriptor>, SourceError> {
        let query = DocumentQuery::all();
        for document in self.documents(&query)? {
            let document = document?;
            if &document.id == id {
                return Ok(Some(document));
            }
        }
        Ok(None)
    }

    fn read(&self, document: &DocumentDescriptor) -> Result<SourceContent, SourceError>;

    fn read_by_id(
        &self,
        id: &DocumentId,
        expected: &RevisionToken,
    ) -> Result<SourceContent, SourceError> {
        let document = self
            .document(id)?
            .ok_or_else(|| SourceError::not_found(format!("unknown source document {id}")))?;
        if &document.version.token != expected {
            return Err(SourceError::stale(format!(
                "source document {id} is at revision {}, expected {expected}",
                document.version.token
            )));
        }
        self.read(&document)
    }

    fn read_batch(
        &self,
        documents: &[DocumentDescriptor],
    ) -> Vec<Result<SourceContent, SourceError>> {
        documents
            .iter()
            .map(|document| self.read(document))
            .collect()
    }
}

pub trait SourceWorkspace: Send + Sync {
    fn descriptor(&self) -> WorkspaceDescriptor;

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            streaming_enumeration: true,
            ..SourceCapabilities::default()
        }
    }

    fn open_snapshot(
        &self,
        request: &SnapshotRequest,
    ) -> Result<Arc<dyn SourceSnapshot>, SourceError>;

    fn changes_since<'a>(
        &'a self,
        _checkpoint: &'a SourceCheckpoint,
        _query: &'a DocumentQuery,
    ) -> Result<Option<ChangeIter<'a>>, SourceError> {
        Ok(None)
    }
}

#[derive(Clone)]
pub struct MemoryWorkspace {
    inner: Arc<MemoryWorkspaceInner>,
}

struct MemoryWorkspaceInner {
    descriptor: WorkspaceDescriptor,
    root: SourceRootId,
    sequence: AtomicU64,
    documents: RwLock<BTreeMap<DocumentId, MemoryDocument>>,
}

#[derive(Clone)]
struct MemoryDocument {
    descriptor: DocumentDescriptor,
    bytes: Arc<[u8]>,
}

impl MemoryWorkspace {
    pub fn new(id: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            inner: Arc::new(MemoryWorkspaceInner {
                descriptor: WorkspaceDescriptor {
                    id: WorkspaceId::new(id.clone()),
                    display_name: id.clone(),
                    source_kind: SourceKind::Memory,
                    redacted_locator: Some(id),
                },
                root: SourceRootId::new("root"),
                sequence: AtomicU64::new(0),
                documents: RwLock::new(BTreeMap::new()),
            }),
        }
    }

    pub fn insert(
        &self,
        logical_path: impl Into<String>,
        content: impl Into<Vec<u8>>,
    ) -> DocumentId {
        let logical_path = logical_path.into();
        self.insert_with_id(
            DocumentId::new(logical_path.clone()),
            logical_path,
            content,
            LanguageHint::Unknown,
        )
    }

    pub fn insert_with_language(
        &self,
        logical_path: impl Into<String>,
        content: impl Into<Vec<u8>>,
        language_hint: LanguageHint,
    ) -> DocumentId {
        let logical_path = logical_path.into();
        self.insert_with_id(
            DocumentId::new(logical_path.clone()),
            logical_path,
            content,
            language_hint,
        )
    }

    pub fn insert_with_id(
        &self,
        id: DocumentId,
        logical_path: impl Into<String>,
        content: impl Into<Vec<u8>>,
        language_hint: LanguageHint,
    ) -> DocumentId {
        let bytes: Arc<[u8]> = Arc::from(content.into());
        let sequence = self.inner.sequence.fetch_add(1, Ordering::SeqCst) + 1;
        let mut version = DocumentVersion::new(
            RevisionToken::new(format!("memory:{sequence}")),
            RevisionGuarantee::ContentIdentity,
        );
        version.size = Some(bytes.len() as u64);
        let descriptor = DocumentDescriptor {
            id: id.clone(),
            location: DocumentLocation {
                root: self.inner.root.clone(),
                logical_path: logical_path.into(),
                display_uri: None,
            },
            version,
            language_hint,
            metadata: DocumentMetadata::default(),
        };
        self.inner
            .documents
            .write()
            .expect("memory workspace lock poisoned")
            .insert(id.clone(), MemoryDocument { descriptor, bytes });
        id
    }

    pub fn remove(&self, id: &DocumentId) -> bool {
        self.inner
            .documents
            .write()
            .expect("memory workspace lock poisoned")
            .remove(id)
            .is_some()
    }

    pub fn move_document(
        &self,
        id: &DocumentId,
        logical_path: impl Into<String>,
    ) -> Result<(), SourceError> {
        let mut documents = self.inner.documents.write().map_err(|_| {
            SourceError::new(SourceErrorKind::Other, "memory workspace lock poisoned")
        })?;
        let document = documents
            .get_mut(id)
            .ok_or_else(|| SourceError::not_found(format!("unknown source document {id}")))?;
        document.descriptor.location.logical_path = logical_path.into();
        let sequence = self.inner.sequence.fetch_add(1, Ordering::SeqCst) + 1;
        document.descriptor.version.token = RevisionToken::new(format!("memory:{sequence}"));
        Ok(())
    }
}

impl SourceWorkspace for MemoryWorkspace {
    fn descriptor(&self) -> WorkspaceDescriptor {
        self.inner.descriptor.clone()
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            streaming_enumeration: true,
            batch_read: true,
            exact_revision_reads: true,
            change_feed: false,
        }
    }

    fn open_snapshot(
        &self,
        _request: &SnapshotRequest,
    ) -> Result<Arc<dyn SourceSnapshot>, SourceError> {
        let sequence = self.inner.sequence.load(Ordering::SeqCst);
        let documents = self
            .inner
            .documents
            .read()
            .map_err(|_| {
                SourceError::new(SourceErrorKind::Other, "memory workspace lock poisoned")
            })?
            .clone();
        let checkpoint = SourceCheckpoint {
            workspace: self.inner.descriptor.id.clone(),
            token: Arc::from(sequence.to_le_bytes()),
        };
        Ok(Arc::new(MemorySnapshot {
            id: SnapshotId::new(format!("{}@{sequence}", self.inner.descriptor.id)),
            workspace_id: self.inner.descriptor.id.clone(),
            documents,
            checkpoint,
        }))
    }
}

struct MemorySnapshot {
    id: SnapshotId,
    workspace_id: WorkspaceId,
    documents: BTreeMap<DocumentId, MemoryDocument>,
    checkpoint: SourceCheckpoint,
}

impl SourceSnapshot for MemorySnapshot {
    fn id(&self) -> &SnapshotId {
        &self.id
    }

    fn workspace_id(&self) -> &WorkspaceId {
        &self.workspace_id
    }

    fn consistency(&self) -> SnapshotConsistency {
        SnapshotConsistency::Immutable
    }

    fn checkpoint(&self) -> Option<&SourceCheckpoint> {
        Some(&self.checkpoint)
    }

    fn documents<'a>(&'a self, query: &'a DocumentQuery) -> Result<DocumentIter<'a>, SourceError> {
        Ok(Box::new(
            self.documents
                .values()
                .map(|document| document.descriptor.clone())
                .filter(move |document| query.matches(document))
                .map(Ok),
        ))
    }

    fn document(&self, id: &DocumentId) -> Result<Option<DocumentDescriptor>, SourceError> {
        Ok(self
            .documents
            .get(id)
            .map(|document| document.descriptor.clone()))
    }

    fn read(&self, document: &DocumentDescriptor) -> Result<SourceContent, SourceError> {
        let stored = self.documents.get(&document.id).ok_or_else(|| {
            SourceError::not_found(format!("unknown source document {}", document.id))
        })?;
        if stored.descriptor.version.token != document.version.token {
            return Err(SourceError::stale(format!(
                "source document {} is at revision {}, expected {}",
                document.id, stored.descriptor.version.token, document.version.token
            )));
        }
        Ok(SourceContent {
            bytes: stored.bytes.clone(),
            observed_version: stored.descriptor.version.clone(),
            encoding_hint: Some("utf-8".to_string()),
        })
    }
}

pub struct OverlayWorkspace {
    descriptor: WorkspaceDescriptor,
    base: Arc<dyn SourceWorkspace>,
    overlay: Arc<dyn SourceWorkspace>,
}

impl OverlayWorkspace {
    pub fn new(
        id: impl Into<String>,
        base: Arc<dyn SourceWorkspace>,
        overlay: Arc<dyn SourceWorkspace>,
    ) -> Self {
        let id = id.into();
        Self {
            descriptor: WorkspaceDescriptor {
                id: WorkspaceId::new(id.clone()),
                display_name: id,
                source_kind: SourceKind::Editor,
                redacted_locator: None,
            },
            base,
            overlay,
        }
    }
}

impl SourceWorkspace for OverlayWorkspace {
    fn descriptor(&self) -> WorkspaceDescriptor {
        self.descriptor.clone()
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            streaming_enumeration: true,
            batch_read: true,
            exact_revision_reads: self.base.capabilities().exact_revision_reads
                && self.overlay.capabilities().exact_revision_reads,
            change_feed: false,
        }
    }

    fn open_snapshot(
        &self,
        request: &SnapshotRequest,
    ) -> Result<Arc<dyn SourceSnapshot>, SourceError> {
        let base = self.base.open_snapshot(request)?;
        let overlay = self.overlay.open_snapshot(request)?;
        let mut by_path: BTreeMap<String, OverlayDocument> = BTreeMap::new();
        for document in base.documents(&DocumentQuery::all())? {
            let document = document?;
            by_path.insert(
                document.location.logical_path.clone(),
                OverlayDocument {
                    descriptor: document,
                    source: base.clone(),
                },
            );
        }
        for document in overlay.documents(&DocumentQuery::all())? {
            let document = document?;
            by_path.insert(
                document.location.logical_path.clone(),
                OverlayDocument {
                    descriptor: document,
                    source: overlay.clone(),
                },
            );
        }
        let consistency = base.consistency().min(overlay.consistency());
        Ok(Arc::new(OverlaySnapshot {
            id: SnapshotId::new(format!("overlay:{}+{}", base.id(), overlay.id())),
            workspace_id: self.descriptor.id.clone(),
            consistency,
            documents: by_path.into_values().collect(),
        }))
    }
}

#[derive(Clone)]
struct OverlayDocument {
    descriptor: DocumentDescriptor,
    source: Arc<dyn SourceSnapshot>,
}

struct OverlaySnapshot {
    id: SnapshotId,
    workspace_id: WorkspaceId,
    consistency: SnapshotConsistency,
    documents: Vec<OverlayDocument>,
}

impl SourceSnapshot for OverlaySnapshot {
    fn id(&self) -> &SnapshotId {
        &self.id
    }

    fn workspace_id(&self) -> &WorkspaceId {
        &self.workspace_id
    }

    fn consistency(&self) -> SnapshotConsistency {
        self.consistency
    }

    fn documents<'a>(&'a self, query: &'a DocumentQuery) -> Result<DocumentIter<'a>, SourceError> {
        Ok(Box::new(
            self.documents
                .iter()
                .map(|document| document.descriptor.clone())
                .filter(move |document| query.matches(document))
                .map(Ok),
        ))
    }

    fn document(&self, id: &DocumentId) -> Result<Option<DocumentDescriptor>, SourceError> {
        Ok(self
            .documents
            .iter()
            .find(|document| &document.descriptor.id == id)
            .map(|document| document.descriptor.clone()))
    }

    fn read(&self, document: &DocumentDescriptor) -> Result<SourceContent, SourceError> {
        let entry = self
            .documents
            .iter()
            .find(|entry| entry.descriptor.id == document.id)
            .ok_or_else(|| {
                SourceError::not_found(format!("unknown source document {}", document.id))
            })?;
        entry.source.read(&entry.descriptor)
    }
}

pub fn validate_snapshot(snapshot: &dyn SourceSnapshot) -> Result<(), SourceError> {
    let mut ids = HashSet::new();
    let mut paths = HashSet::new();
    for document in snapshot.documents(&DocumentQuery::all())? {
        let document = document?;
        if document.id.as_str().is_empty() {
            return Err(SourceError::invalid(
                "source snapshot returned an empty document id",
            ));
        }
        if document.location.logical_path.is_empty() {
            return Err(SourceError::invalid(
                "source snapshot returned an empty logical path",
            ));
        }
        if !ids.insert(document.id.clone()) {
            return Err(SourceError::invalid(format!(
                "source snapshot returned duplicate document id {}",
                document.id
            )));
        }
        if !paths.insert(document.location.logical_path.clone()) {
            return Err(SourceError::invalid(format!(
                "source snapshot returned duplicate logical path {:?}",
                document.location.logical_path
            )));
        }
        let content = snapshot.read(&document)?;
        if content.observed_version.token != document.version.token {
            return Err(SourceError::stale(format!(
                "source snapshot read revision {} for {}, expected {}",
                content.observed_version.token, document.id, document.version.token
            )));
        }
    }
    Ok(())
}

pub fn group_documents_by_root(
    documents: impl IntoIterator<Item = DocumentDescriptor>,
) -> HashMap<SourceRootId, Vec<DocumentDescriptor>> {
    let mut grouped: HashMap<SourceRootId, Vec<DocumentDescriptor>> = HashMap::new();
    for document in documents {
        grouped
            .entry(document.location.root.clone())
            .or_default()
            .push(document);
    }
    for group in grouped.values_mut() {
        group.sort_by(|left, right| left.location.logical_path.cmp(&right.location.logical_path));
    }
    grouped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_snapshots_are_immutable_and_revision_checked() {
        let workspace = MemoryWorkspace::new("memory://test");
        let id = workspace.insert_with_language(
            "src/lib.rs",
            b"fn answer() -> i32 { 42 }".to_vec(),
            LanguageHint::Known("rust".to_string()),
        );
        let snapshot = workspace
            .open_snapshot(&SnapshotRequest::default())
            .unwrap();
        let document = snapshot.document(&id).unwrap().unwrap();
        workspace.insert_with_language(
            "src/lib.rs",
            b"fn answer() -> i32 { 43 }".to_vec(),
            LanguageHint::Known("rust".to_string()),
        );
        assert!(
            snapshot
                .read(&document)
                .unwrap()
                .utf8()
                .unwrap()
                .contains("42")
        );
        validate_snapshot(snapshot.as_ref()).unwrap();
    }

    #[test]
    fn overlay_prefers_matching_logical_paths() {
        let base = MemoryWorkspace::new("memory://base");
        base.insert("src/lib.rs", b"base".to_vec());
        let overlay = MemoryWorkspace::new("memory://overlay");
        overlay.insert("src/lib.rs", b"overlay".to_vec());
        let workspace = OverlayWorkspace::new("overlay", Arc::new(base), Arc::new(overlay));
        let snapshot = workspace
            .open_snapshot(&SnapshotRequest::default())
            .unwrap();
        let documents: Vec<_> = snapshot
            .documents(&DocumentQuery::all())
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(documents.len(), 1);
        assert_eq!(
            snapshot.read(&documents[0]).unwrap().utf8().unwrap(),
            "overlay"
        );
    }
}
