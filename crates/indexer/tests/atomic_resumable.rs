use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use codeindex_core::{ExtractedEntity, Representation};
use codeindex_indexer::{
    CancellationToken, DocumentCheckpointStep, EnricherIdentity, IndexOutcome, IndexRunBuilder,
    IndexRunState, IndexSettings, MemorySource, PublishStep, RepresentationEnricher, RetentionMode,
    SourceDocument, SourceProject, SourceProvider, SourceRevision, StableRead,
};
use codeindex_sqlite::{Db, open_in_memory, open_or_create};
use codeindex_tree_sitter::normalizer::sha256_hex;

fn settings() -> IndexSettings {
    IndexSettings {
        enabled_languages: vec!["rust".into()],
        body_node_count_threshold: 1,
        max_body_chars: 10_000,
        retention: RetentionMode::Full,
    }
}

fn run_memory(db: &Db, source: &MemorySource) -> Result<IndexOutcome> {
    let projects = [SourceProject {
        label: "main".into(),
        provider: source,
    }];
    IndexRunBuilder::new(db, &settings(), &projects).run()
}

fn live_counts(db: &Db) -> (i64, i64, i64, i64, i64) {
    (
        db.conn()
            .query_row("SELECT COUNT(*) FROM projects", [], |row| row.get(0))
            .unwrap(),
        db.conn()
            .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))
            .unwrap(),
        db.conn()
            .query_row("SELECT COUNT(*) FROM code_units", [], |row| row.get(0))
            .unwrap(),
        db.conn()
            .query_row("SELECT COUNT(*) FROM representations", [], |row| row.get(0))
            .unwrap(),
        db.current_generation().unwrap(),
    )
}

#[test]
fn every_publish_fault_rolls_back_the_entire_live_corpus() {
    let steps = [
        PublishStep::Settings,
        PublishStep::Projects,
        PublishStep::DeleteLiveDocuments,
        PublishStep::Metadata,
        PublishStep::InsertDocument,
        PublishStep::Usage,
        PublishStep::Invariants,
        PublishStep::Generation,
        PublishStep::Commit,
    ];
    for fail_at in steps {
        let db = open_in_memory().unwrap();
        let mut source = MemorySource::new("memory://atomic");
        source.insert("lib.rs", "fn old_value() -> i32 { 1 }");
        run_memory(&db, &source).unwrap();
        let before_snapshot = db.snapshot(&[]).unwrap();
        let before_counts = live_counts(&db);

        source.insert("lib.rs", "fn new_value() -> i32 { 2 }");
        source.insert("added.rs", "fn added() -> i32 { 3 }");
        let projects = [SourceProject {
            label: "main".into(),
            provider: &source,
        }];
        let hook = |step| {
            if step == fail_at {
                anyhow::bail!("injected publish fault at {step:?}");
            }
            Ok(())
        };
        let error = IndexRunBuilder::new(&db, &settings(), &projects)
            .with_publish_fault_hook(&hook)
            .run()
            .unwrap_err();
        assert!(format!("{error:#}").contains("injected publish fault"));
        assert_eq!(live_counts(&db), before_counts, "fault at {fail_at:?}");
        assert_eq!(
            db.snapshot(&[]).unwrap(),
            before_snapshot,
            "fault at {fail_at:?}"
        );

        // The ready journal remains retryable and produces exactly one new
        // publication when the injected failure is removed.
        let outcome = IndexRunBuilder::new(&db, &settings(), &projects)
            .run()
            .unwrap();
        let IndexOutcome::Committed(report) = outcome else {
            panic!("retry did not commit")
        };
        assert!(report.generation > before_counts.4);
        assert!(
            db.snapshot(&[])
                .unwrap()
                .units
                .iter()
                .any(|unit| unit.name == "new_value")
        );
    }
}

#[derive(Clone)]
struct CountingEnricher {
    calls: Arc<AtomicUsize>,
}

impl RepresentationEnricher for CountingEnricher {
    fn identity(&self) -> EnricherIdentity {
        EnricherIdentity {
            producer: "test-counter".into(),
            version: "1".into(),
            config_fingerprint: "fixed".into(),
        }
    }

    fn enrich(
        &self,
        _document: &SourceDocument,
        _source: &str,
        _entity: &ExtractedEntity,
    ) -> Result<Vec<Representation>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Vec::new())
    }
}

#[test]
fn cancellation_preserves_ready_documents_and_resume_reuses_them() {
    let db = open_in_memory().unwrap();
    let mut source = MemorySource::new("memory://resume");
    source.insert("a.rs", "fn a() -> i32 { 1 }");
    source.insert("b.rs", "fn b() -> i32 { 2 }");
    let projects = [SourceProject {
        label: "main".into(),
        provider: &source,
    }];
    let cancellation = CancellationToken::new();
    let cancel_from_progress = cancellation.clone();
    let progress = move |event: codeindex_indexer::IndexProgress| {
        if event.source_document_id.as_deref() == Some("a.rs") {
            cancel_from_progress.cancel();
        }
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let enricher = CountingEnricher {
        calls: calls.clone(),
    };
    let enrichers: [&dyn RepresentationEnricher; 1] = [&enricher];
    let outcome = IndexRunBuilder::new(&db, &settings(), &projects)
        .with_enrichers(&enrichers)
        .with_cancellation(cancellation)
        .on_progress(&progress)
        .run()
        .unwrap();
    let IndexOutcome::Paused(status) = outcome else {
        panic!("cancellation unexpectedly committed")
    };
    assert_eq!(status.state, IndexRunState::Paused);
    assert_eq!(status.pause_reason.as_deref(), Some("user_interrupt"));
    assert_eq!(live_counts(&db), (0, 0, 0, 0, 0));
    assert_eq!(status.stats.documents_ready, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let outcome = IndexRunBuilder::new(&db, &settings(), &projects)
        .with_enrichers(&enrichers)
        .run()
        .unwrap();
    let IndexOutcome::Committed(report) = outcome else {
        panic!("resume did not commit")
    };
    assert_eq!(report.run_id, status.run_id);
    assert_eq!(calls.load(Ordering::SeqCst), 2, "ready a.rs was recomputed");
    assert_eq!(db.snapshot(&[]).unwrap().units.len(), 2);
}

#[test]
fn checkpoint_failure_never_leaks_a_partial_document() {
    let db = open_in_memory().unwrap();
    let mut source = MemorySource::new("memory://checkpoint");
    source.insert("lib.rs", "fn value() -> i32 { 7 }");
    let projects = [SourceProject {
        label: "main".into(),
        provider: &source,
    }];
    let hook = |step, _project: &str, _document: &str| {
        if step == DocumentCheckpointStep::Before {
            anyhow::bail!("process died before checkpoint")
        }
        Ok(())
    };
    assert!(
        IndexRunBuilder::new(&db, &settings(), &projects)
            .with_document_fault_hook(&hook)
            .run()
            .is_err()
    );
    assert_eq!(live_counts(&db), (0, 0, 0, 0, 0));
    let payloads: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM index_run_documents WHERE payload_json IS NOT NULL",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(payloads, 0);
}

#[test]
fn failed_first_publication_exposes_neither_project_nor_settings() {
    let db = open_in_memory().unwrap();
    let mut source = MemorySource::new("memory://new-project");
    source.insert("lib.rs", "fn hidden_until_commit() -> i32 { 1 }");
    let projects = [SourceProject {
        label: "new-project".into(),
        provider: &source,
    }];
    let hook = |step| {
        if step == PublishStep::InsertDocument {
            anyhow::bail!("stop before first commit")
        }
        Ok(())
    };
    assert!(
        IndexRunBuilder::new(&db, &settings(), &projects)
            .with_publish_fault_hook(&hook)
            .run()
            .is_err()
    );
    assert!(db.get_project("new-project").unwrap().is_none());
    assert_eq!(db.current_generation().unwrap(), 0);
    assert!(
        db.get_setting("index.body_node_count_threshold")
            .unwrap()
            .is_none()
    );
}

#[test]
fn incompatible_auto_resume_supersedes_only_the_overlapping_run() {
    let db = open_in_memory().unwrap();
    let mut source = MemorySource::new("memory://config-change");
    source.insert("lib.rs", "fn configured() -> i32 { 1 }");
    let projects = [SourceProject {
        label: "main".into(),
        provider: &source,
    }];
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let IndexOutcome::Paused(old) = IndexRunBuilder::new(&db, &settings(), &projects)
        .with_cancellation(cancellation)
        .run()
        .unwrap()
    else {
        panic!("pre-cancelled run committed")
    };
    let mut changed_settings = settings();
    changed_settings.body_node_count_threshold = 2;
    let IndexOutcome::Committed(new) = IndexRunBuilder::new(&db, &changed_settings, &projects)
        .run()
        .unwrap()
    else {
        panic!("replacement run did not commit")
    };
    assert_ne!(old.run_id, new.run_id);
    assert_eq!(
        db.run_status(old.run_id).unwrap().state,
        IndexRunState::Superseded
    );
}

#[derive(Default)]
struct MutableAdvisorySource {
    content: Mutex<String>,
}

impl MutableAdvisorySource {
    fn set(&self, content: &str) {
        *self.content.lock().unwrap() = content.to_string();
    }
}

impl SourceProvider for MutableAdvisorySource {
    fn project_locator(&self) -> String {
        "mutable://advisory".into()
    }

    fn documents(&self, enabled: &HashSet<String>) -> Result<Vec<SourceDocument>> {
        if !enabled.contains("rust") {
            return Ok(Vec::new());
        }
        let size = self.content.lock().unwrap().len() as u64;
        Ok(vec![SourceDocument {
            id: "stable-id".into(),
            relative_path: "lib.rs".into(),
            language_id: "rust".into(),
            // Deliberately useless metadata: default indexing must hash bytes.
            revision: SourceRevision {
                opaque: "constant".into(),
                modified_ns: None,
                size: Some(size),
            },
        }])
    }

    fn read(&self, _document: &SourceDocument) -> Result<String> {
        Ok(self.content.lock().unwrap().clone())
    }
}

#[test]
fn advisory_revisions_are_verified_by_content_hash() {
    let db = open_in_memory().unwrap();
    let source = MutableAdvisorySource::default();
    source.set("fn before() -> i32 { 1 }");
    let projects = [SourceProject {
        label: "main".into(),
        provider: &source,
    }];
    IndexRunBuilder::new(&db, &settings(), &projects)
        .run()
        .unwrap();
    source.set("fn after() -> i32 { 222 }");
    IndexRunBuilder::new(&db, &settings(), &projects)
        .run()
        .unwrap();
    let snapshot = db.snapshot(&[]).unwrap();
    assert_eq!(snapshot.units.len(), 1);
    assert_eq!(snapshot.units[0].name, "after");
}

#[derive(Default)]
struct SwappableSource {
    documents: Mutex<BTreeMap<String, (String, String)>>,
}

impl SwappableSource {
    fn insert(&self, id: &str, path: &str, source: &str) {
        self.documents
            .lock()
            .unwrap()
            .insert(id.into(), (path.into(), source.into()));
    }
}

impl SourceProvider for SwappableSource {
    fn project_locator(&self) -> String {
        "mutable://swap".into()
    }

    fn documents(&self, enabled: &HashSet<String>) -> Result<Vec<SourceDocument>> {
        if !enabled.contains("rust") {
            return Ok(Vec::new());
        }
        Ok(self
            .documents
            .lock()
            .unwrap()
            .iter()
            .map(|(id, (path, source))| SourceDocument {
                id: id.clone(),
                relative_path: path.clone(),
                language_id: "rust".into(),
                revision: SourceRevision::new(sha256_hex(source)),
            })
            .collect())
    }

    fn read(&self, document: &SourceDocument) -> Result<String> {
        Ok(self.documents.lock().unwrap()[&document.id].1.clone())
    }
}

#[test]
fn delete_first_publication_supports_path_swaps() {
    let db = open_in_memory().unwrap();
    let source = SwappableSource::default();
    source.insert("one", "a.rs", "fn one() -> i32 { 1 }");
    source.insert("two", "b.rs", "fn two() -> i32 { 2 }");
    let projects = [SourceProject {
        label: "main".into(),
        provider: &source,
    }];
    IndexRunBuilder::new(&db, &settings(), &projects)
        .run()
        .unwrap();
    source.insert("one", "b.rs", "fn one() -> i32 { 1 }");
    source.insert("two", "a.rs", "fn two() -> i32 { 2 }");
    IndexRunBuilder::new(&db, &settings(), &projects)
        .run()
        .unwrap();
    let snapshot = db.snapshot(&[]).unwrap();
    assert_eq!(snapshot.units.len(), 2);
    assert!(
        snapshot
            .units
            .iter()
            .any(|unit| unit.name == "one" && unit.relative_path == "b.rs")
    );
    assert!(
        snapshot
            .units
            .iter()
            .any(|unit| unit.name == "two" && unit.relative_path == "a.rs")
    );
}

#[test]
fn several_projects_publish_all_or_none() {
    let db = open_in_memory().unwrap();
    let mut first = MemorySource::new("memory://first");
    first.insert("a.rs", "fn first_old() -> i32 { 1 }");
    let mut second = MemorySource::new("memory://second");
    second.insert("b.rs", "fn second_old() -> i32 { 2 }");
    let initial = [
        SourceProject {
            label: "first".into(),
            provider: &first,
        },
        SourceProject {
            label: "second".into(),
            provider: &second,
        },
    ];
    IndexRunBuilder::new(&db, &settings(), &initial)
        .run()
        .unwrap();
    let before = db.snapshot(&[]).unwrap();
    first.insert("a.rs", "fn first_new() -> i32 { 10 }");
    second.insert("b.rs", "fn second_new() -> i32 { 20 }");
    let changed = [
        SourceProject {
            label: "first".into(),
            provider: &first,
        },
        SourceProject {
            label: "second".into(),
            provider: &second,
        },
    ];
    let inserted = AtomicUsize::new(0);
    let hook = |step| {
        if step == PublishStep::InsertDocument && inserted.fetch_add(1, Ordering::SeqCst) == 1 {
            anyhow::bail!("late project failure")
        }
        Ok(())
    };
    assert!(
        IndexRunBuilder::new(&db, &settings(), &changed)
            .with_publish_fault_hook(&hook)
            .run()
            .is_err()
    );
    assert_eq!(db.snapshot(&[]).unwrap(), before);
}

struct ChurningSource {
    stable_reads: AtomicUsize,
    version: AtomicUsize,
    churn_reads: AtomicUsize,
    churn: AtomicBool,
    triggered: AtomicBool,
    cancellation: CancellationToken,
}

impl ChurningSource {
    fn new(cancellation: CancellationToken) -> Self {
        Self {
            stable_reads: AtomicUsize::new(0),
            version: AtomicUsize::new(0),
            churn_reads: AtomicUsize::new(0),
            churn: AtomicBool::new(false),
            triggered: AtomicBool::new(false),
            cancellation,
        }
    }

    fn settle(&self) {
        self.churn.store(false, Ordering::SeqCst);
    }

    fn source_for(&self, id: &str, version: usize) -> String {
        if id == "a" {
            "fn unaffected() -> i32 { 1 }".into()
        } else {
            format!("fn changing() -> i32 {{ {} }}", version + 2)
        }
    }
}

impl SourceProvider for ChurningSource {
    fn project_locator(&self) -> String {
        "mutable://churning".into()
    }

    fn documents(&self, enabled: &HashSet<String>) -> Result<Vec<SourceDocument>> {
        if !enabled.contains("rust") {
            return Ok(Vec::new());
        }
        let version = self.version.load(Ordering::SeqCst);
        Ok(vec![
            SourceDocument {
                id: "a".into(),
                relative_path: "a.rs".into(),
                language_id: "rust".into(),
                revision: SourceRevision::new("a-stable"),
            },
            SourceDocument {
                id: "b".into(),
                relative_path: "b.rs".into(),
                language_id: "rust".into(),
                revision: SourceRevision::new(format!("b-{version}")),
            },
        ])
    }

    fn read(&self, document: &SourceDocument) -> Result<String> {
        Ok(self.source_for(&document.id, self.version.load(Ordering::SeqCst)))
    }

    fn stable_read(&self, document: &SourceDocument) -> Result<StableRead> {
        let call = self.stable_reads.fetch_add(1, Ordering::SeqCst) + 1;
        let version = self.version.load(Ordering::SeqCst);
        let source = self.source_for(&document.id, version);
        if document.id == "b" && self.churn.load(Ordering::SeqCst) {
            self.version.fetch_add(1, Ordering::SeqCst);
            if self.churn_reads.fetch_add(1, Ordering::SeqCst) + 1 >= 3 {
                self.cancellation.cancel();
            }
            return Ok(StableRead::Changed);
        }
        // The fourth stable read is processing b.rs after both initial
        // observations and a.rs. Change b immediately after returning its old
        // stable bytes so the refresh barrier, not the processing pass, finds
        // the save.
        if document.id == "b" && call == 4 && !self.triggered.swap(true, Ordering::SeqCst) {
            self.version.fetch_add(1, Ordering::SeqCst);
            self.churn.store(true, Ordering::SeqCst);
        }
        Ok(StableRead::Content {
            source,
            revision: document.revision.clone(),
        })
    }
}

#[test]
fn persistent_churn_pauses_without_losing_unaffected_staged_work() {
    let db = open_in_memory().unwrap();
    let cancellation = CancellationToken::new();
    let source = ChurningSource::new(cancellation.clone());
    let projects = [SourceProject {
        label: "main".into(),
        provider: &source,
    }];
    let calls = Arc::new(AtomicUsize::new(0));
    let enricher = CountingEnricher {
        calls: calls.clone(),
    };
    let enrichers: [&dyn RepresentationEnricher; 1] = [&enricher];
    let outcome = IndexRunBuilder::new(&db, &settings(), &projects)
        .with_enrichers(&enrichers)
        .with_cancellation(cancellation)
        .run()
        .unwrap();
    let IndexOutcome::Paused(paused) = outcome else {
        panic!("churning source unexpectedly published")
    };
    assert_eq!(paused.pause_reason.as_deref(), Some("user_interrupt"));
    assert_eq!(db.current_generation().unwrap(), 0);
    assert_eq!(calls.load(Ordering::SeqCst), 2);

    source.settle();
    let outcome = IndexRunBuilder::new(&db, &settings(), &projects)
        .with_enrichers(&enrichers)
        .run()
        .unwrap();
    let IndexOutcome::Committed(report) = outcome else {
        panic!("settled source did not publish")
    };
    assert_eq!(report.run_id, paused.run_id);
    assert!(report.refresh_rounds >= 1);
    assert!(report.reused_documents >= 1);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "unaffected a.rs was recomputed"
    );
}

#[test]
fn concurrent_snapshots_are_entirely_old_or_new() {
    let directory = tempfile::tempdir().unwrap();
    let database_path = directory.path().join("index.db");
    let db = open_or_create(&database_path).unwrap();
    let mut source = MemorySource::new("memory://snapshot-race");
    source.insert("lib.rs", "fn old_snapshot() -> i32 { 1 }");
    let initial = {
        let projects = [SourceProject {
            label: "main".into(),
            provider: &source,
        }];
        let IndexOutcome::Committed(initial) = IndexRunBuilder::new(&db, &settings(), &projects)
            .run()
            .unwrap()
        else {
            panic!("initial run did not commit")
        };
        initial
    };

    let stop = Arc::new(AtomicBool::new(false));
    let failures = Arc::new(Mutex::new(Vec::new()));
    let reader_stop = stop.clone();
    let reader_failures = failures.clone();
    let reader_path = database_path.clone();
    let initial_generation = initial.generation;
    let reader = std::thread::spawn(move || {
        let reader_db = open_or_create(&reader_path).unwrap();
        while !reader_stop.load(Ordering::SeqCst) {
            let snapshot = reader_db.snapshot(&[]).unwrap();
            let names: Vec<_> = snapshot
                .units
                .iter()
                .map(|unit| unit.name.as_str())
                .collect();
            let valid_old = snapshot.published_generation == initial_generation as u64
                && names == ["old_snapshot"]
                && snapshot.projects[0].last_index_run_id == Some(initial_generation as u64);
            let valid_new = names == ["new_snapshot"]
                && snapshot.projects[0].last_index_run_id == Some(snapshot.published_generation);
            if !valid_old && !valid_new {
                reader_failures
                    .lock()
                    .unwrap()
                    .push((snapshot.published_generation, names.join(",")));
            }
        }
    });

    source.insert("lib.rs", "fn new_snapshot() -> i32 { 2 }");
    let projects = [SourceProject {
        label: "main".into(),
        provider: &source,
    }];
    let hook = |step| {
        if step == PublishStep::InsertDocument {
            std::thread::sleep(std::time::Duration::from_millis(30));
        }
        Ok(())
    };
    let IndexOutcome::Committed(updated) = IndexRunBuilder::new(&db, &settings(), &projects)
        .with_publish_fault_hook(&hook)
        .run()
        .unwrap()
    else {
        panic!("updated run did not commit")
    };
    assert!(updated.generation > initial.generation);
    std::thread::sleep(std::time::Duration::from_millis(20));
    stop.store(true, Ordering::SeqCst);
    reader.join().unwrap();
    assert!(
        failures.lock().unwrap().is_empty(),
        "mixed snapshots observed"
    );
}
