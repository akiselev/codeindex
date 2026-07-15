# Structured diagnostics, typed errors, and failure-policy plan

Status: proposed. Scope: remove process output from reusable crates, make
recoverable failures machine-readable, replace public `anyhow::Result` APIs
with typed errors, and give the CLI a stable policy and exit-code contract.

This plan is based on `master` after `ed72dfa` (atomic, resumable indexing).
The issue description refers to the older counter-and-`eprintln!` indexing
loop. That loop is no longer present: document read/processing errors now pause
a durable run in `crates/indexer/src/run.rs`. The architectural problem remains,
but the implementation must extend the new run journal rather than restore the
old `ProjectStats.failed` behavior.

## 1. Goals and non-goals

Goals:

- reusable crates never write to stdout or stderr;
- recoverable problems are emitted as versioned, serializable diagnostics;
- callers choose strict or best-effort behavior explicitly;
- every public fallible API returns a crate-owned error type;
- error categories and CLI exit codes are stable enough for agents and scripts;
- JSON mode produces NDJSON on stdout and no unstructured library output;
- durable run status preserves the same diagnostic records emitted live;
- the final result states whether a committed generation is complete or partial.

Non-goals:

- do not expose `rusqlite`, tree-sitter, ONNX Runtime, or HTTP error types as the
  cross-crate error contract;
- do not create one exhaustive application error enum containing every backend
  implementation detail;
- do not make best-effort downgrade database corruption, invalid configuration,
  invariant violations, or a required unavailable model/accelerator;
- do not promise source-level recovery from an invalid `IndexSnapshot`; a
  corrupt snapshot is rejected as a whole.

## 2. Public contracts

### 2.1 Common classification and diagnostics

Add the dependency-light public vocabulary to `codeindex-core` in
`crates/core/src/diagnostic.rs` and re-export it from the facade:

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCategory {
    BadConfig,
    IncompatibleDatabase,
    UnavailableModel,
    UnsupportedAccelerator,
    PartialIndex,
    CorruptSnapshot,
    Interrupted,
    Io,
    Internal,
}

pub trait CategorizedError: std::error::Error {
    fn category(&self) -> ErrorCategory;
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity { Warning, Error }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub severity: DiagnosticSeverity,
    pub message: String,
    pub help: Option<String>,
    pub project_label: Option<String>,
    pub source_document_id: Option<String>,
    pub relative_path: Option<String>,
    pub language_id: Option<String>,
    pub span: Option<SourceSpan>,
    pub details: BTreeMap<String, String>,
}
```

`DiagnosticCode` is a non-exhaustive enum serialized in `snake_case`, initially
including `document_read`, `document_parse`, `reference_extract`,
`representation_enrichment`, `accelerator_fallback`, and
`model_artifact_refetch`. Codes, field names, and enum serialization are the
machine contract; `message` and `help` are human-facing and may evolve.

Do not serialize opaque error chains. Preserve backend errors as typed error
sources for fatal returns, and place only stable, intentionally public facts in
diagnostic `details`.

### 2.2 Event sinks

Use an event sink rather than a diagnostic-only callback so progress and
diagnostics have one ordered stream:

```rust
pub trait EventSink<E> {
    type Error: std::error::Error + Send + Sync + 'static;
    fn emit(&mut self, event: &E) -> Result<(), Self::Error>;
}

pub enum IndexEvent {
    Progress(IndexProgress),
    Diagnostic(Diagnostic),
}

pub enum EmbeddingEvent {
    Progress(EmbedProgress),
    Diagnostic(Diagnostic),
    ModelDownload(ModelDownloadProgress),
}
```

Provide `NoopEventSink` and blanket support for suitable closures. Builder
methods become `with_event_sink`; retain `on_progress` for one deprecation cycle
as a compatibility adapter if desired. Sink failure is a typed operation error,
never ignored. The CLI treats a closed stdout pipe as a clean early termination
and other rendering I/O failures as `Io`.

The sink is synchronous and receives borrowed events. Document that a sink must
return promptly and must not call back into the same `Db`; this preserves event
ordering and avoids reentrancy/deadlock surprises. Emit events only after their
corresponding durable checkpoint succeeds, so an observed diagnostic never
claims state that was not recorded.

### 2.3 Failure policy and outcomes

Add a serialized policy to `IndexSettings` and the durable run fingerprint:

```rust
pub enum FailurePolicy { Strict, BestEffort }

pub enum IndexCompletion { Complete, Partial }

pub enum IndexOutcome {
    Committed { report: IndexReport, completion: IndexCompletion },
    Paused(IndexRunStatus),
}
```

Default to `Strict`. A policy change makes an unfinished run incompatible and
therefore follows the existing supersede/new-run rules.

The policy matrix is:

| Failure | Strict | Best effort |
|---|---|---|
| document read failure after retries | checkpoint diagnostic; pause; do not publish | checkpoint diagnostic; continue |
| unit parse/extraction failure | checkpoint diagnostic; pause; do not publish | checkpoint diagnostic; continue |
| reference extraction failure | checkpoint diagnostic; pause; do not publish | index units without new reference data; continue |
| optional enrichment failure | checkpoint diagnostic; pause; do not publish | omit only that producer's representation; continue |
| source changes during stable read | reconcile/retry | reconcile/retry |
| bad config/provider contract | fail | fail |
| incompatible/corrupt DB or publish failure | fail/leave prior corpus live | same |
| unavailable required model/accelerator | fail | fail |
| explicit accelerator `auto` fallback | emit warning and use CPU | same |
| corrupt snapshot | reject snapshot | reject snapshot |
| user interrupt | pause; exit 130 | same |

Best-effort publication semantics must be deterministic:

- a failed new document is absent from the new generation;
- a failed update retains its last committed file, units, representations, and
  references;
- a document positively observed as deleted is still deleted;
- a reference-only failure retains the prior document's Usage representation
  if one exists; for a new document Usage is absent;
- all successfully staged changes still publish atomically in one transaction;
- any committed run with one or more error-severity diagnostics has
  `IndexCompletion::Partial`, even if the retained prior data makes the corpus
  searchable;
- a partial commit returns a report normally from the library but the CLI uses
  the dedicated partial-index exit code.

Warnings (for example an explicitly allowed CPU fallback) do not by themselves
make a generation partial. `--warnings-as-errors` is a CLI rendering/policy
overlay: the CLI requests strict handling for diagnostics that are otherwise
warnings, and must decide this before publication. It is not implemented by
changing an exit code after a complete generation has already committed.

## 3. Typed error ownership

Add `thiserror` directly to crates that define public error enums. Each crate
owns its domain errors and implements `CategorizedError`; callers classify at
crate boundaries without string matching.

### `codeindex-sqlite`

Introduce `SqliteError` and `type Result<T> = std::result::Result<T,
SqliteError>`. Required variants include `IncompatibleDatabase { found,
supported }`, `InvalidStoredValue { table, column, value }`, `CorruptData {
context, source }`, `ConfigConflict { key, stored, requested }`, `LeaseConflict`,
and a private-detail `Database(#[source] rusqlite::Error)` variant. Migration
version rejection maps directly to `IncompatibleDatabase`; malformed staged
payload/report rows map to corrupt data rather than generic context strings.

Keep raw `rusqlite::Result` inside row-mapping closures only. No public method
returns it or `anyhow::Result`.

### `codeindex-embedding`

Introduce `EmbeddingError` with distinct `BadConfig`, `UnavailableModel`,
`UnsupportedAccelerator`, `CorruptModelArtifact`, `Download`, and `Inference`
variants. `provider_mode=require` returns `UnsupportedAccelerator`; `auto`
emits `accelerator_fallback`. Missing/offline model material returns
`UnavailableModel`, while a downloaded artifact that fails its pinned hash is
`CorruptModelArtifact` with category `UnavailableModel` (the CLI may show the
more precise diagnostic code).

Change `Embedder::embed` and `embedder_from_config` to the typed result. For
third-party embedders, provide a constructor/boxed source variant so custom
implementations can retain their native cause without depending on `anyhow`.

### `codeindex-tree-sitter` and `codeindex-query`

Introduce `FrontendError` (`UnknownLanguage`, `GrammarLoad`, `InvalidQuery`,
`Parse`) and `QueryError` (`MalformedFilter`, `InvalidGlob`, `InvalidSelector`,
`DimensionMismatch`). These generally classify as `BadConfig` when caused by a
caller selector/config value and `Internal` or `Io` when bundled assets cannot
load. The indexer converts document-scoped frontend errors into diagnostics;
direct frontend callers still receive the typed error.

### `codeindex-search` and `codeindex-storage`

Introduce `SearchError`, separating `CorruptSnapshot` validation (zero or
mismatched dimensions, duplicate ids, missing vector/hash relationships) from
`BadConfig` query selection and embedding errors. Make snapshot validation an
explicit `IndexSnapshot::validate()` or `ValidatedIndexSnapshot::try_from`, so
every consumer shares the same corrupt-snapshot rules rather than duplicating
checks in `SearchIndex::from_snapshot`.

### `codeindex-indexer`

Introduce `IndexError` variants that transparently wrap the crate-owned storage,
frontend, and embedding errors, plus `BadConfig`, `Provider`, `Enricher`,
`EventSink`, `Paused`, `Partial`, `Interrupted`, and `Invariant`. Replace
`IndexRunFailure { source: anyhow::Error }` with a typed variant carrying
`run_id` and a boxed `IndexError` source.

Object-safe extension traits need explicit boundary errors:

- `SourceProvider::{documents,read,stable_read}` returns `SourceResult<T>` with
  a `SourceErrorKind` supplied by the provider;
- `RepresentationEnricher::enrich` returns `EnrichmentResult<T>`;
- fault hooks used only by tests may use a small `InjectedFault` type or remain
  behind `cfg(test)` instead of shaping production errors.

An unknown third-party error maps to `Internal` unless its wrapper explicitly
declares `Io`, `BadConfig`, or a recoverable document fault. This prevents
best-effort mode from guessing recoverability by inspecting messages or source
types.

Use `anyhow` only in binaries, tests, examples, or private transitional helpers.
No public signature or public trait method may mention `anyhow::Error` or
`anyhow::Result` when this work is complete.

## 4. Durable diagnostics and partial publication

Replace the ad hoc JSON strings in `index_runs.last_error_json` and
`index_run_documents.error_json` with the serialized common diagnostic schema.
Because a document may have both a recoverable reference and enrichment problem,
one error blob per document is insufficient. Add a journal table:

```sql
CREATE TABLE index_run_diagnostics (
  run_id              INTEGER NOT NULL REFERENCES index_runs(id) ON DELETE CASCADE,
  sequence            INTEGER NOT NULL,
  project_label       TEXT,
  source_document_id  TEXT,
  severity            TEXT NOT NULL,
  code                TEXT NOT NULL,
  diagnostic_json     TEXT NOT NULL,
  created_at          TEXT NOT NULL,
  PRIMARY KEY (run_id, sequence)
);
```

Allocate `sequence` and checkpoint a diagnostic in the same transaction as the
document state change. Add `Db::append_run_diagnostic`,
`Db::list_run_diagnostics`, and diagnostic counts to `IndexRunStats`. Keep
`last_error_json` only as a derived/cache field during the schema rewrite, or
remove it and query the last journal row. Since the database is pre-release,
rewrite the bootstrap schema and bump the rejected schema epoch; no migration
path is required.

Add a terminal staged-document state such as `skipped`. In best-effort mode it
is publishable but carries no replacement payload. Publication uses the live
row retention rules above and records complete/partial in `index_runs` and the
serialized `IndexReport`. Resuming must not retry `skipped` rows unless the
source revision, configuration fingerprint, retry command, or policy changes.
Expose an explicit retry/reset API rather than overloading ordinary resume.

Refactor `prepare_document` so unit extraction, reference extraction, and each
enricher are separate checkpoints. Today a single `Result` loses the distinction
needed for the policy matrix. It should return staged content plus zero or more
diagnostics when recovery is safe; invariant failures still return `IndexError`.

## 5. CLI contract

Move all rendering to `crates/cli` behind an `Output`/`Renderer` abstraction.
The renderer implements both event sinks and renders the terminal outcome/error.

Add global flags (available to every subcommand):

- `--json`: versioned NDJSON events on stdout;
- `--quiet`: suppress progress and warnings in human mode, but never suppress a
  terminal error; in JSON mode it suppresses progress only so diagnostics and
  the terminal result remain machine-observable;
- `--warnings-as-errors`: request strict treatment of warning diagnostics;
- `--failure-policy strict|best-effort` on indexing commands, default `strict`.

JSON event names are `progress`, `diagnostic`, `result`, and `error`, all using
the existing `{ "version": 1, "event": ..., "data": ... }` envelope. Exactly
one terminal `result` or `error` event is emitted. A partial commit is a `result`
with `completion: "partial"`, not an `error`, even though the process exit code
is nonzero. Human diagnostics go to stderr; human results go to stdout. JSON
mode writes the entire NDJSON protocol to stdout and leaves stderr empty except
for failures that occur before CLI parsing can establish the mode.

Use this stable mapping:

| Exit | Meaning |
|---:|---|
| 0 | complete success |
| 1 | unexpected/internal or uncategorized failure |
| 2 | bad CLI/config/query input (aligned with clap usage errors) |
| 3 | incompatible database/schema/configuration fixed by reindexing |
| 4 | requested model unavailable |
| 5 | requested accelerator unsupported or unavailable in required mode |
| 6 | indexing committed a partial generation |
| 7 | corrupt or internally inconsistent snapshot/database payload |
| 74 | operational I/O failure (database open, filesystem, network, output) |
| 130 | interrupted by SIGINT/SIGTERM through the current graceful path |

`ErrorCategory` maps in one exhaustive CLI function; individual commands must
not choose codes ad hoc. `PartialIndex` is produced from the committed outcome,
not by parsing an error. Database SQL corruption that prevents a snapshot maps
to 7; inability to open/read/write an otherwise valid database maps to 74.

Remove all reusable-crate output:

- replace accelerator fallback and managed-model download `eprintln!` calls in
  `crates/embedding/src/embed/fastembed_backend.rs` with embedding events;
- disable fastembed's direct download progress (`with_show_download_progress`)
  unless it can be redirected into the sink;
- keep `println!`/`eprintln!` only in the CLI renderer (Cargo build-script
  protocol output is exempt).

## 6. Implementation sequence

1. **Freeze contracts with tests.** Add compile-time/public API tests for the
   diagnostic JSON shape, error categories, policy defaults, and exit mapping.
   Add CLI golden tests proving JSON stdout contains only valid envelopes.
2. **Introduce common types.** Add `ErrorCategory`, `CategorizedError`,
   `Diagnostic`, diagnostic codes, the generic sink, and no-op/closure adapters
   in `codeindex-core`; re-export them from `codeindex`.
3. **Remove embedding output.** Add `EmbeddingEvent`, thread its sink through
   model construction/download, disable backend-owned progress, and convert the
   embedding public API to `EmbeddingError`.
4. **Type leaf-crate errors.** Convert storage/snapshot validation,
   tree-sitter, query, search, and SQLite public APIs. Convert callers one crate
   at a time; do not use string classification as a bridge.
5. **Add the diagnostic journal.** Rewrite/bump the bootstrap schema, add typed
   storage models and CRUD, and replace hand-built `error_json` in `run.rs` and
   `index_publish.rs`.
6. **Implement policy-aware staging.** Split document preparation stages, add
   skipped/partial state, retain prior live records on recoverable failures,
   compute completion, and emit checkpointed `IndexEvent::Diagnostic` records.
7. **Convert the indexer public surface.** Type provider/enricher boundaries,
   replace `IndexRunFailure`/`IndexPausedError` plumbing, and adapt legacy
   convenience entry points.
8. **Centralize CLI rendering.** Add flags, unified sinks, the exhaustive
   category-to-exit mapping, terminal JSON events, quiet behavior, and broken
   pipe handling. Remove direct output from command logic.
9. **Audit and document.** Require `rg 'eprintln!|println!' crates` to show only
   CLI/build-script sites and `rg 'anyhow::(Result|Error)|use anyhow' crates/*/src`
   to show no public-library use. Update `docs/architecture.md` and
   `docs/getting-started.md` with policy, JSON schema, and exit codes.

Do the public error conversion in small, compiling commits, but merge policy,
durable diagnostics, and partial publication together: exposing best-effort
before its diagnostics and retained-data semantics are durable would recreate
the unreliable counter-only behavior this work is meant to remove.

## 7. Acceptance tests

### Library behavior

- a library call with a recording sink produces no stdout/stderr output;
- read, parse, reference, and enrichment faults yield the expected diagnostic
  code, location, severity, and ordered sequence;
- sink failure returns a typed error and cannot be mistaken for successful
  indexing;
- every public error enum reports the expected category and preserves its
  source chain;
- malformed snapshot dimensions/ids/hashes return `CorruptSnapshot`;
- incompatible schema returns `IncompatibleDatabase`, not a message-classified
  generic error;
- required unavailable acceleration and automatic fallback take different
  typed paths.

### Policy and atomicity

- strict read/parse/reference/enricher failure pauses and leaves the generation
  unchanged;
- best-effort failure on a new document commits other documents, omits the new
  failed document, marks the report partial, and retains diagnostics;
- best-effort failure updating an existing document retains its prior units and
  references while successful sibling updates commit;
- a confirmed deletion is not retained merely because another document failed;
- warnings alone produce a complete result; `--warnings-as-errors` prevents
  publication under the strict overlay;
- partial publication rollback under an injected SQLite fault leaves the prior
  corpus and generation intact;
- resume does not silently retry durable skipped rows, while the explicit retry
  operation does.

### CLI/protocol

- every line from `codeindex index --json` parses as one version-1 envelope;
- JSON mode has empty stderr for progress, diagnostics, and classified errors;
- quiet human mode suppresses progress/warnings but prints terminal errors;
- one and only one terminal event is emitted;
- complete, partial, bad-config, incompatible-DB, unavailable-model,
  unsupported-accelerator, corrupt-snapshot, I/O, and interrupt fixtures return
  the documented codes;
- a closed pipe does not print a secondary panic/error or corrupt a preceding
  JSON line.

## 8. Completion gate

Run:

```text
cargo fmt --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Then enforce the output/API audits from step 9 and manually exercise human,
`--quiet`, `--json`, strict, best-effort, and warnings-as-errors runs against a
fixture containing one unreadable file and one injected reference failure.
