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
- [x] Add provider-neutral indexing with filesystem and memory implementations.
- [x] Separate stable source document identity from logical path and revision.
- [x] Add named embedding spaces with different models/dimensions in one corpus.
- [x] Add explicit-space search and reciprocal-rank fusion.
- [x] Filter storage-neutral snapshots to selected-project vectors and validate
      snapshots on load.
- [x] Reject the incompatible previous pre-release schema epoch.
- [x] CI: rustfmt, Clippy with warnings denied, workspace tests, and fastembed
      feature compile check.

## P0 — release blockers

- [ ] **Publish scope.** Decide which of the facade and eight component crates are
      public, reserve names, and remove `publish = false` where appropriate.
- [ ] **Compiled examples.** Convert source-provider, explicit-space, fusion, and
      ordinary filesystem quickstarts into `examples/` compiled by CI.
- [ ] **Public API audit.** Review ownership, error types, naming, and future
      extensibility of `SourceProvider`, `IndexSnapshot`, representation
      enrichment, and embedding-space APIs before external adoption.
- [ ] **Downstream migration.** Update `decombine` to the explicit space APIs and
      typed entity ids, then use it as the first compatibility proof.

## P1 — documentation and quality

- [ ] **Rustdoc coverage.** Document every public item and enable
      `#![deny(missing_docs)]` crate by crate.
- [ ] **Getting-started rewrite.** Replace the pre-space workflow and document
      delete/reindex behavior for schema epoch 2.
- [ ] **CHANGELOG and compatibility policy.** State schema, snapshot, and public
      Rust API guarantees for 0.x releases.
- [ ] **Real backend smoke test.** Add a scheduled or manually triggered test that
      downloads a small model and executes inference, rather than compile-only
      checking fastembed.

## P2 — next substrate capabilities

- [ ] **Relations.** Replace the Rust-only name-based Usage resolver with a
      parser-neutral relation model carrying provenance and resolution quality.
- [ ] **Language reference coverage.** Populate and test `references.scm` for
      bundled languages beyond Rust.
- [ ] **Cross-document entity moves.** Preserve logical identity across provider
      document moves only when matching is unique and evidence is explicit.
- [ ] **Source-provider ergonomics.** Add maintained Git-tree and editor-overlay
      providers; optimize provider catalogs so source recovery does not need full
      enumeration on every lookup.
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
