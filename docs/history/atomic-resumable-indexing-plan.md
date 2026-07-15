# Atomic, resumable indexing implementation plan

> **Historical design record — implemented.** Shipped in schema epoch 3
> (commit `ed72dfa`). Written pre-implementation, so it reads as pending work
> ("Add…", "Refactor…"); the implementation map below is accurate. Current
> contracts live in `docs/architecture.md`.

Status: implemented in schema epoch 3. Scope: replace the former live,
statement-by-statement index
mutation with a durable staging journal and one atomic publish transaction before
the `codeindex-cli` exposes `index`.

Implementation map:

- `crates/indexer/src/run.rs`: builder, resume/refresh/cancellation state machine;
- `crates/indexer/src/stage.rs`: document preparation and ordinal references;
- `crates/sqlite/src/index_runs.rs`: durable journal, leases, transitions, and GC;
- `crates/sqlite/src/index_publish.rs`: delete-first atomic corpus publication;
- `crates/cli/src/main.rs`: thin index/status/resume/restart/abandon consumer;
- `crates/indexer/tests/atomic_resumable.rs`: fault, churn, resume, and concurrency
  acceptance coverage.

This plan assumes the repository's existing pre-release schema policy: schema
epoch changes may rewrite the bootstrap schema and reject older databases rather
than carrying a migration path.

## 1. Decisions

1. **Atomicity covers one requested run scope.** A run that selects several
   projects publishes all selected projects or none. Unselected projects are not
   part of that transaction.
2. **Operational state is durable but not search-visible.** Run records,
   manifests, staged documents, progress, leases, and errors commit throughout
   indexing. Published corpus tables change only in the final transaction.
3. **Source refresh is automatic by default.** The indexer converges a mutable
   manifest toward the latest provider observations. Normal edits do not force
   users to discard a run manually.
4. **Refresh invalidates the smallest safe unit.** A changed document discards
   only that document's staged payload. Unchanged staged documents remain ready.
5. **Publication requires a quiescent refresh barrier.** Once no documents are
   pending, the indexer refreshes every project again. It publishes only when
   reconciliation makes no changes.
6. **Automatic convergence has no refresh budget in the first implementation.**
   Persistent source churn keeps the run in reconciliation until the source
   settles or the caller cancels. It never causes partial publication, and
   cancellation preserves all valid staged work for resume.
7. **The initial CLI is strict about actual failures.** Unresolved provider,
   read, extraction, enrichment, or storage errors prevent publication. Source
   changes are refresh events, not failures.
8. **Indexing and embedding remain separate operations.** Index publication
   includes files, entities, units, representations, raw references, and Usage.
   Content-addressed embeddings remain independently resumable and may lag a
   newly published corpus.
9. **SQLite remains the concrete write-side store.** Do not introduce the
   generalized write-store trait that the roadmap intentionally defers until a
   second backend needs it.
10. **Snapshot reads get transaction isolation too.** `Db::snapshot*` must read
    projects, units, representations, spaces, and vectors in one SQLite read
    transaction so readers observe one publication.
11. **Advisory source revisions are verified by hashing by default.** Metadata
    trust remains an explicit performance opt-in until measurements justify a
    different default.
12. **The first staging representation is versioned per-document JSON.** Measure
    its size, decode cost, memory use, and publish time before considering
    normalized staging tables.
13. **Automatic behavior does not remove low-level control.** The builder and
    storage API expose explicit run selection, resume, supersede, abandon,
    revision-trust, and convergence-policy controls.

## 2. User-visible contract

During an index run:

- query and snapshot consumers continue to see the previous committed corpus;
- progress and status consumers see durable run/document progress;
- repeated file saves cause automatic restaging of only the affected files;
- interruption loses at most the document currently being computed;
- resuming a compatible run reuses every still-valid staged document;
- a successful return means the complete requested scope was published;
- a paused or failed return means the published corpus and generation did not
  change.

The source snapshot is defined as the set of provider observations accepted by
the final no-change refresh barrier. This is not a promise that a mutable
filesystem remains unchanged after the barrier; providing that stronger
guarantee requires a snapshot-capable provider such as a Git tree.

### Default convergence behavior

The default runner should require no manual resume or refresh decisions:

1. Reuse the newest compatible unfinished run when one exists.
2. Reconcile its manifest with current provider observations.
3. Preserve ready documents whose relevant inputs still match.
4. Reset changed or newly discovered documents to pending.
5. Turn disappeared documents into staged deletions.
6. Process pending documents and repeat the refresh barrier.
7. Publish after a barrier makes no changes.

If the configured settings, project locators, enabled enrichers, or staged
payload format are incompatible with an unfinished run for the same overlapping
scope, default `Auto` resume marks that run `superseded` and starts a fresh run.
Unrelated project scopes are not superseded. Explicit APIs can instead inspect
or resume a chosen run.

## 3. State machine and interruption semantics

```text
planning -> running -> ready -- atomic publish --> committed
                |         |
                v         v
              paused    failed
                |
                +--------> running

planning/running/ready/paused/failed -> superseded | abandoned
```

`publishing` is an in-memory phase, not a separately committed status. The
`ready -> committed` transition is written inside the same transaction as the
live corpus changes. A crash during publish rolls the corpus back and leaves the
durable run `ready` and retryable.

- **SIGINT:** set a cancellation flag, finish or discard the current uncommitted
  computation, checkpoint `paused/user_interrupt`, and return an interrupted
  outcome suitable for CLI exit code 130.
- **SIGTERM:** use the same graceful path when time permits.
- **Process death/SIGKILL:** the durable row remains `running` with a stale
  lease. The next owner atomically records `paused/process_lost` before resume.
- **Transient source change:** reconcile and retry automatically.
- **Persistent churn:** continue reconciling without an arbitrary round/time
  limit. A caller or agent may cancel and resume after the source settles.
- **Permanent document error:** checkpoint the document error and run as
  `paused/document_error` by default. A later invocation retries according to
  policy. A caller may explicitly abandon it.
- **Invariant/configuration failure:** mark `failed`; do not publish.
- **Publish failure:** SQLite rollback leaves the run `ready`; store the publish
  error outside the failed transaction and allow retry.

## 4. Durable SQLite model

Add an index-run journal in the next bootstrap schema epoch. Exact SQL can be
tuned during implementation, but ownership and constraints should follow this
shape.

### `index_runs`

```sql
CREATE TABLE index_runs (
  id                    INTEGER PRIMARY KEY,
  base_generation       INTEGER NOT NULL,
  status                TEXT NOT NULL,
  phase                 TEXT NOT NULL,
  scope_json             TEXT NOT NULL,
  config_json            TEXT NOT NULL,
  config_fingerprint     TEXT NOT NULL,
  payload_schema_version INTEGER NOT NULL,
  refresh_round          INTEGER NOT NULL DEFAULT 0,
  owner_token            TEXT,
  heartbeat_at           TEXT,
  created_at             TEXT NOT NULL,
  updated_at             TEXT NOT NULL,
  committed_at           TEXT,
  last_error_json        TEXT,
  stats_json             TEXT NOT NULL
);
```

Use checked status/phase values or equivalent validation in the Rust model.
Index the status and configuration fingerprint used by automatic resume.

### `index_run_projects`

Persist project label, provider locator, provider identity/fingerprint when
available, manifest digest, last refresh time, and per-project counters. Use the
label rather than a live `project_id` because a new project must not become live
before publication.

### `index_run_documents`

```sql
CREATE TABLE index_run_documents (
  run_id                 INTEGER NOT NULL REFERENCES index_runs(id) ON DELETE CASCADE,
  project_label          TEXT NOT NULL,
  source_document_id     TEXT NOT NULL,
  relative_path          TEXT,
  language_id            TEXT,
  source_revision_json   TEXT,
  observed_source_hash   TEXT,
  action                 TEXT NOT NULL,
  state                  TEXT NOT NULL,
  input_fingerprint      TEXT,
  attempts               INTEGER NOT NULL DEFAULT 0,
  payload_schema_version INTEGER,
  payload_json           TEXT,
  error_json             TEXT,
  updated_at             TEXT NOT NULL,
  PRIMARY KEY (run_id, project_label, source_document_id)
);
```

Actions are `unchanged`, `metadata`, `upsert`, and `delete`. States are
`pending`, `processing`, `ready`, and `error`. Enforce the legal combinations,
for example `upsert/ready` requires a payload and `delete/ready` does not.

Start with one versioned JSON payload per changed document. It contains:

- the prospective `NewFile` values;
- all `NewCodeUnit` values and representations after retention;
- raw references attributed by unit ordinal rather than live `UnitId`;
- source/config/frontend/enricher fingerprints needed to validate reuse.

JSON is already available, inspectable during development, and keeps the first
implementation from duplicating every live table as a staging table. Put a
payload version in both the run and row. If profiling later shows publication
decode or memory cost is material, replace the payload with normalized staging
tables without changing the run lifecycle.

### Lease and writer coordination

Run acquisition and takeover happen in `BEGIN IMMEDIATE` transactions. A random
owner token must match on every checkpoint and publish. Heartbeats make stale
ownership diagnosable; takeover after expiry is explicit in the storage API even
when the high-level runner invokes it automatically.

Only one live-corpus maintenance operation should publish at a time. Search and
status reads remain allowed. Initially, serialize index publication with other
operations that prune or destructively change corpus state. Content-addressed
embedding inserts may interleave, but orphan-vector GC must not race an embedder.

## 5. Automatic refresh and manifest reconciliation

The manifest is a durable cache of desired state, not an immutable plan.

### Reconcile algorithm

For each project refresh:

1. Enumerate and validate the provider's current documents.
2. Build a map keyed by stable provider-local document ID.
3. Compare it with the journal and the currently published files.
4. For a new document, insert `upsert/pending`.
5. For a disappeared document, discard any payload and mark `delete/ready`.
6. For a matching document whose relevant input fingerprint is unchanged,
   preserve its state and payload.
7. For changed revision, path, language, provider input, or processing
   fingerprint, discard only that payload and mark it pending.
8. Recompute the project manifest digest and counters in the same transaction.

The input fingerprint should include document ID, logical path, language,
provider revision/observation, indexing settings, frontend version, retention,
and ordered enricher identities. Source hash joins the fingerprint after a read.

### Revision confidence

Make revision confidence explicit rather than silently trusting every opaque
token:

```rust
pub enum RevisionSemantics {
    /// Equal revisions guarantee equal source bytes.
    Authoritative,
    /// Revisions are scheduling hints; content must be verified by hashing.
    Advisory,
}
```

Add a default `SourceProvider::revision_semantics()` returning `Advisory`.

- `MemorySource` can use `Authoritative` because its revision is the content
  hash.
- Snapshot-addressed Git/object providers can use `Authoritative`.
- The conservative generic behavior for `Advisory` providers reads and hashes
  content before declaring it unchanged.
- `FileSystemSource` should provide a specialized stable-read implementation:
  open/read/stat, detect a revision change across the read, and ask the runner
  to refresh rather than surfacing a user-facing failure. The final barrier
  still rescans the project.

Provide a builder override for applications that deliberately trust advisory
metadata for performance, but do not make that the reliability-oriented
default. Record the chosen trust policy in the run configuration fingerprint.

### Convergence policy

Expose a policy value on the run builder rather than pushing orchestration into
each provider:

```rust
pub struct RefreshPolicy {
    pub mode: RefreshMode,             // default: Automatic
    pub settle_delay: Duration,
    pub retry_backoff: RetryBackoff,
}
```

The provider supplies observations and declares revision semantics; the runner
owns retry, backoff, invalidation, status, and UX. This keeps custom providers
simple and gives the filesystem and CLI the same reliable default behavior.

There is no refresh-round or elapsed-time budget in the first implementation.
After the first completed processing pass, a refresh barrier may discover that
source changed while indexing. For example:

1. The runner finishes staging the initial manifest.
2. The barrier discovers that `a.rs` was saved again, so only `a.rs` is reset and
   restaged. That is one refresh round.
3. The next barrier discovers that `b.rs` changed, so `b.rs` is restaged. That is
   a second refresh round.
4. A barrier that observes no changes publishes immediately.

The runner continues this loop until a barrier observes no changes, at which
point it publishes. A continuously changing generated file can therefore keep a
run alive; this is acceptable for the initial agent-oriented use case because
callers can wait for source activity to settle before expecting the new corpus.
Queries made during that time continue to use the previous committed corpus.

Use a short settling delay and backoff to avoid a hot loop. Cancellation remains
the escape hatch: it pauses the run with all valid payloads preserved, and the
next invocation resumes the same convergence work. The policy stays a builder
type so a future CLI or interactive application can add a time/round limit
without changing journal or publisher semantics, but no such limit is part of
the initial default.

## 6. Library API shape

Add a high-level builder in `codeindex-indexer`; keep journal/publish primitives
SQLite-specific in `codeindex-sqlite`.

```rust
let outcome = IndexRunBuilder::new(&db, settings, projects)
    .with_enrichers(enrichers)
    .resume_policy(ResumePolicy::Auto)       // default
    .refresh_policy(RefreshPolicy::default())
    .retry_policy(RetryPolicy::default())
    .on_progress(progress_callback)
    .with_cancellation(cancellation)
    .run()?;
```

Return a typed outcome rather than forcing operational interruption through an
`anyhow` string:

```rust
pub enum IndexOutcome {
    Committed(IndexReport),
    Paused(IndexRunStatus),
}
```

Programming/configuration/storage failures may still return `Result::Err`, but
the error should carry a run ID whenever one exists. Durable document errors and
interruptions return a status that the CLI can render consistently.

Preserve `index()` and `index_sources()` as convenience functions implemented
through the default builder. Their successful return continues to mean a fully
committed run; paused outcomes become a typed top-level error containing the run
ID and reason so existing callers cannot mistake them for success.

### Required supporting API changes

- Give `RepresentationEnricher` a stable identity containing producer, version,
  and configuration fingerprint. Resume reuse is unsafe without it.
- Add source observation/stable-read support without requiring every provider to
  implement retry orchestration.
- Add public run/status/report types with serde support for future versioned CLI
  JSON envelopes.
- Add SQLite operations such as `create_or_resume_run`, `reconcile_manifest`,
  `claim_run`, `checkpoint_document`, `mark_ready`, `publish_run`, `pause_run`,
  `fail_run`, `abandon_run`, `run_status`, and staged-run GC.
- Refactor live write helpers so the publisher can execute them through a
  `rusqlite::Transaction`, not only through `Db`'s owned connection.

Do not expose raw staging payloads as part of the stable high-level API.

### Expected source layout

Keep the existing large modules from absorbing the entire feature:

- `crates/sqlite/src/migrations.rs`: new bootstrap schema epoch only;
- `crates/sqlite/src/index_runs.rs`: journal models, transitions, leases,
  reconciliation persistence, status reads, and GC;
- `crates/sqlite/src/index_publish.rs`: transaction-scoped live merge helpers;
- `crates/sqlite/src/models.rs`: live and staged payload value types;
- `crates/sqlite/src/lib.rs`: narrow re-exports and existing store API;
- `crates/indexer/src/run.rs`: builder, state-machine orchestration, retry,
  cancellation, and progress;
- `crates/indexer/src/stage.rs`: one-document extraction/enrichment/identity
  preparation and staged-reference attribution;
- `crates/indexer/src/source.rs`: observation semantics and built-in provider
  implementations;
- `crates/indexer/src/lib.rs`: convenience APIs and public re-exports.

The names may shift during implementation, but the journal, publisher, runner,
and document preparation responsibilities should remain separate and testable.

## 7. Atomic publish algorithm

Prepare and validate staged payloads before taking the write transaction. Then:

1. `BEGIN IMMEDIATE`.
2. Verify owner token, `ready` status, manifest digest, payload versions, and
   `base_generation == current published generation`.
3. Check or initialize immutable index settings inside this transaction.
4. Validate existing project locators and insert new projects.
5. Delete every live file being replaced or removed before inserts. This avoids
   transient unique-path conflicts during moves and swaps.
6. Apply metadata-only rows.
7. Insert changed files, entities, units, representations, and raw references,
   translating staged unit ordinals to new live IDs.
8. Recompute Usage for every changed project inside the transaction.
9. Run cheap structural/count invariants needed to reject a malformed staged
   publication.
10. Update each selected project's `last_index_run_id`.
11. Set the published generation to the run ID. Gaps from abandoned runs are
    valid.
12. Mark the run committed with final stats and timestamp.
13. Commit.

Do not delete staged payloads, orphan vectors, or other operational history in
the publish transaction. Post-commit GC is retryable maintenance and cannot turn
a valid publication into a failed index command.

Unchanged units retain the generation in which their version was created. The
published generation means "latest successfully reconciled run," while a unit's
generation means "run that introduced this unit version."

## 8. Consistent snapshot reads

Refactor `snapshot_with_spaces` into an internal helper parameterized by a
connection-like executor. The public method opens one deferred read transaction,
runs every project/unit/representation/space/vector query through it, constructs
the snapshot, then commits or rolls back the read transaction.

Add publication identity to the snapshot contract. At minimum expose the
published generation and per-project last run so callers can diagnose freshness.
Because this changes the serde shape, do it in the same pre-release schema/API
epoch as the journal.

## 9. Implementation phases

### Phase 0 — Specify invariants and fault hooks

- Add the state/action enums, typed errors, reports, and transition tests.
- Add deterministic fault-injection hooks available to tests at document
  checkpoint and publish steps.
- Document exactly which tables are live corpus versus operational state.

Exit: invalid state transitions and malformed staged rows are rejected without
touching live corpus state.

### Phase 1 — Schema and journal storage API

- Rewrite the bootstrap schema to the next pre-release epoch.
- Add run, project, and document journal tables and Rust models.
- Implement create, claim, heartbeat, pause, fail, supersede, abandon, inspect,
  reconcile, and checkpoint operations using short transactions.
- Add payload serde/versioning and staged-reference unit ordinals.

Exit: a process can create a run, checkpoint documents, close the DB, reopen it,
claim the run, and recover identical staged payloads and progress.

### Phase 2 — Source observation and automatic reconciliation

- Add revision semantics and stable source-read results.
- Implement authoritative behavior for `MemorySource`.
- Implement filesystem stable reads and change detection.
- Implement manifest reconciliation, selective invalidation, retry/backoff, and
  the no-change refresh barrier.
- Add enricher identities and include all processing inputs in fingerprints.

Exit: edits, creates, deletes, renames, and repeated saves during indexing
converge automatically without discarding unaffected staged documents.

### Phase 3 — Stage instead of mutate

- Split current `index_project` into pure-ish document preparation and journal
  checkpointing.
- Keep reads, extraction, enrichment, identity assignment, retention, and raw
  reference attribution outside live writes.
- Make the legacy convenience APIs delegate to `IndexRunBuilder`.
- Remove generation bump and all live file mutation from the processing loop.

Exit: a complete indexing pass can reach `ready`, while a before/after snapshot
of every live corpus table and published generation is unchanged.

### Phase 4 — Transactional publisher

- Refactor SQLite write helpers to operate on the publish transaction.
- Implement delete-first merge, insertion, Usage rebuild, invariant checks,
  generation update, and committed transition.
- Make publish idempotent: committed runs report their existing result; ready
  runs may retry after rollback; no other state may publish.

Exit: the ready run produces the same semantic corpus as a clean run through the
old pipeline, and injected failure after every publish step leaves the old corpus
and generation intact.

### Phase 5 — Read isolation, cancellation, and concurrency

- Wrap snapshot export in a read transaction.
- Add owner tokens, heartbeat expiry, stale-owner takeover, cancellation, and
  maintenance-operation coordination.
- Define safe interleaving with embedding inserts and move orphan-vector cleanup
  to explicit/post-commit GC.

Exit: concurrent snapshot loops observe only complete old or new generations;
two indexers cannot publish conflicting runs; killed processes resume safely.

### Phase 6 — CLI integration and documentation

- Expose `index`, explicit resume/restart/abandon controls, and durable status.
- Default `index` to compatible auto-resume and automatic refresh.
- Emit stable versioned JSON events/reports as required by the roadmap.
- Report committed generation, run ID, refresh count, reused/restaged documents,
  warnings, and per-space pending embedding counts.
- Update architecture, getting-started, roadmap, and TODO documentation.

Exit: a user can interrupt indexing, edit files, rerun the same command, and get
one atomic publication without learning the journal internals.

## 10. Verification matrix

### Atomicity

- Inject failure after every live mutation in publish; live tables, Usage, and
  published generation remain byte-for-byte or semantically identical.
- A late failure in project N does not publish projects 1 through N-1.
- A new project is invisible until commit.
- Immutable settings are not initialized by a run that never commits.
- Path swaps and delete-plus-create cases publish without unique-key transients.

### Resume

- Kill after each document checkpoint and resume only pending/invalidated work.
- Kill while a document is processing; no partial payload survives.
- Kill during publish; run remains ready and retry produces one publication.
- Config, frontend, retention, payload-version, and enricher changes invalidate
  exactly the staged work they semantically affect.

### Refresh UX

- Modify a pending document, a ready document, and an unchanged live document.
- Create, delete, rename, delete-and-recreate, and rapidly rewrite documents.
- Change a file between enumeration and read.
- Continuously churn one document and verify the run neither publishes nor loses
  unaffected staged work; cancel it, stop changing the document, then verify
  resume converges and commits.
- Verify unaffected ready documents are never recomputed.
- Verify advisory providers take the conservative content-hash path.

### Read/write concurrency

- Repeated snapshots during publish are each entirely old or entirely new.
- Concurrent search never sees staging rows.
- A competing index owner cannot checkpoint or publish with the wrong token.
- Embedding and GC coordination cannot reintroduce orphan vectors after cleanup.

### Compatibility

- Existing filesystem and in-memory indexing acceptance tests pass through the
  default builder.
- Entity identity across rename remains unchanged relative to the current
  behavior.
- Usage and retention behavior match the current successful pipeline.
- A staged/published corpus has the same deterministic snapshot ordering as a
  clean single-process run.

## 11. Performance and follow-up thresholds

Measure, do not preemptively replace the design:

- manifest reconciliation time;
- advisory-provider bytes hashed;
- staged JSON size and encode/decode time;
- memory used to prepare a publish;
- time spent in `BEGIN IMMEDIATE` publication;
- WAL growth and reader latency;
- Usage rebuild share of publish time.

Consider normalized staging tables when JSON decode/memory materially dominates.
Consider append-only version tables plus an active-generation pointer when the
publish transaction itself becomes too long or historical generations become a
product requirement. Neither is required for the initial reliable CLI.

## 12. Plan completion criteria

Atomic resumable indexing is complete when all of the following are true:

- no processing-loop path mutates search-visible corpus tables;
- all selected projects publish in one transaction;
- every non-successful outcome preserves the prior published generation;
- interrupted work resumes from durable document checkpoints;
- normal source changes refresh and converge automatically by default;
- persistent churn cannot publish; cancellation remains resumable;
- snapshots are transactionally consistent with one publication;
- the CLI uses the library state machine rather than implementing its own
  transaction, refresh, or resume behavior.
