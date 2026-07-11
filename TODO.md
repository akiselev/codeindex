# TODO — codeindex 0.1

Goal: a coherent, consumable library substrate that `decombine`,
`codeindex-cli`, bindings, IDE integrations, and agents can use without copying
implementation logic or depending on application-specific modules.

## Completed foundation

- [x] Split the workspace into dependency/change boundaries.
- [x] Keep `codeindex-embedding` free of SQLite and language grammars.
- [x] Add high-level storage-neutral search and stable selectors.
- [x] Normalize multi-representation persistence with entity/version identity.
- [x] Add `Body`, name-erased body, signature, documentation, symbol, Usage, and
      custom/generated representation paths.
- [x] Persist representation provenance.
- [x] Add dependency-light source workspace and immutable snapshot contracts.
- [x] Separate stable workspace/document/root identity from logical paths and
      physical locators.
- [x] Add streamed enumeration, direct/batched reads, structured errors,
      capability declarations, checkpoints, revision guarantees, and consistency
      levels.
- [x] Add filesystem, memory, and editor-overlay workspace implementations.
- [x] Move language resolution out of source providers and into the indexer.
- [x] Add named embedding spaces with different models/dimensions in one corpus.
- [x] Add explicit-space search and reciprocal-rank fusion.
- [x] Filter storage-neutral snapshots to selected-project vectors and validate
      snapshots on load.
- [x] Reject the incompatible previous pre-release schema epoch.
- [x] CI: rustfmt, Clippy with warnings denied, workspace tests, and fastembed
      feature compile check.

## P0 — release blockers

- [ ] **Publish scope.** Decide which of the facade and component crates are
      public, reserve names, and remove `publish = false` where appropriate.
- [ ] **Compiled examples.** Convert source-workspace, explicit-space, fusion, and
      ordinary filesystem quickstarts into `examples/` compiled by CI.
- [ ] **Public API audit.** Review ownership, error types, naming, and future
      extensibility of `SourceWorkspace`, `SourceSnapshot`, `IndexSnapshot`,
      representation enrichment, and embedding-space APIs before external
      adoption.
- [ ] **Downstream migration.** Update `decombine` to the workspace and explicit
      space APIs and typed entity ids, then use it as the first compatibility
      proof.

## P1 — documentation and quality

- [ ] **Rustdoc coverage.** Document every public item and enable
      `#![deny(missing_docs)]` crate by crate.
- [ ] **Getting-started rewrite.** Compile the workspace quickstart and document
      delete/reindex behavior for schema epoch 2.
- [ ] **CHANGELOG and compatibility policy.** State schema, snapshot, and public
      Rust API guarantees for 0.x releases.
- [ ] **Source conformance expansion.** Turn `validate_snapshot` into a reusable
      test harness covering stale revisions, change feeds, batch alignment,
      redacted locators, overlays, and provider capability claims.
- [ ] **Real backend smoke test.** Add a scheduled or manually triggered test that
      downloads a small model and executes inference, rather than compile-only
      checking fastembed.

## P2 — next substrate capabilities

- [ ] **Relations.** Replace the Rust-only name-based Usage resolver with a
      parser-neutral relation model carrying provenance and resolution quality.
- [ ] **Language reference coverage.** Populate and test `references.scm` for
      bundled languages beyond Rust.
- [ ] **Cross-document entity moves.** Preserve logical identity across workspace
      document moves only when matching is unique and evidence is explicit.
- [ ] **Maintained remote providers.** Add Git-tree and editor protocol adapters,
      then evaluate an `object_store`-backed implementation with conditional and
      versioned reads.
- [ ] **Source-provider ergonomics.** Add a maintained Git-tree provider. Editor
       overlays, direct source lookup, bounded checkpoint feeds, and durable
       content-addressed recovery are implemented.
- [ ] **Write-side store seam.** Introduce one only when a second persistence
      backend needs to reuse incremental indexing and embedding projection.
- [ ] **Streaming search.** Add a streaming snapshot/index interface when measured
      corpora no longer fit the in-memory search model.
- [ ] **Retrieval evaluation.** Benchmark channels, models, generated
      descriptions, and fusion independently; move substrate-level evaluation
      out of `decombine`.

## Deliberately deferred

- ANN/vector indexes until exact-search benchmarks establish a concrete trigger.
- Hosted embedding as a required service; local-first remains the default.
- Duplicate, correctness, security, or migration conclusions in the core crate.
