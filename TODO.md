# TODO — codeindex 0.1

Goal of 0.1: a **coherent, consumable substrate** — the crates that were
extracted from decombine, cleaned up enough that a second consumer (a CLI, a
Python binding) can depend on them without reaching back into decombine. Ordered
by priority.

## P0 — release hygiene (blocks any external consumer)

- [ ] **License.** No crate declares a license and there is no `LICENSE` file.
      Decide the license (recommendation: dual `MIT OR Apache-2.0`, the Rust
      ecosystem norm — needs owner sign-off), add `LICENSE-MIT` /
      `LICENSE-APACHE`, and set `license = "MIT OR Apache-2.0"` on every crate.
- [ ] **Crate metadata.** Add `repository`, `authors`, `keywords`, `categories`,
      and a one-line `description` (present) to each `crates/*/Cargo.toml`.
      Consider a `[workspace.package]` block so the shared fields live in one
      place.
- [ ] **Publish scope.** All 7 crates are `publish = false`. Decide what ships to
      crates.io (facade + the six members) vs stays internal, and flip the flag
      once license + metadata land. Reserve the crate names early if publishing.

## P1 — make the substrate usable on its own

- [ ] **High-level search API.** The end-to-end "embed a sentence → rank → resolve
      `body_hash` back to unit metadata (path, line range, name, scope), with
      model-identity verification" flow lives only in decombine
      (`AnalysisContext`). A standalone consumer must reimplement it. Lift it into
      `codeindex-query` (or a new `codeindex-search`) so the stated goal — agents
      querying functionality by sentence — is a library call, not a copy-paste.
      This is the single most valuable 0.1 addition.
- [ ] **Workspace integration test.** Per-crate unit tests exist, but nothing in
      this repo exercises index → embed (hash backend) → query across the seams;
      that coverage stayed in decombine's golden/integration suite. Add a
      `tests/` round-trip using `codeindex-embedding::embed::hash`.
- [ ] **`examples/`.** Turn the `docs/getting-started.md` snippets into compiled
      examples so the docs cannot rot (`cargo test --examples` in CI).

## P2 — debt and polish

- [ ] **Dormant core vocabulary.** `codeindex-core` ships `EntityId`,
      `EntityVersionId`, and 8 of 10 `RepresentationKind` variants that no
      consumer uses yet (the pipeline wires only `FullSource` + `Implementation`).
      It is deliberate seeding for the multi-representation schema (see ROADMAP),
      but decide whether to feature-gate it behind `unstable` or trim it until
      that lands, so the public 0.1 surface reflects what actually works.
- [ ] **Rustdoc coverage.** Add crate- and item-level docs on the public API and
      turn on `#![deny(missing_docs)]` per crate for docs.rs readiness. (Crate
      root docs exist for `codeindex`, `codeindex-embedding`, and `-indexer`.)
- [ ] **CI depth.** CI compiles the fastembed backend but never runs it (model
      download + heavy link). Consider a scheduled job that actually embeds with a
      small model against a fixture, to catch backend regressions the
      compile-only check misses.

## Done in the extraction (2026-07-11)

- Split into 7 crates along dependency/change boundaries; `codeindex-embedding`
  is storage/parser-free (only `codeindex-core`), so a lean binding compiles
  neither bundled SQLite nor the grammars.
- Single `impl From<ExtractedEntity> for NewCodeUnit`; `ModelIdentity` moved to
  `codeindex-core`; language assets relocated into the `tree-sitter` crate
  (self-contained); build-script env vars renamed `DECOMBINE_*` → `CODEINDEX_*`.
- Docs: `README.md`, `docs/architecture.md`, `docs/getting-started.md`; CI
  (fmt, clippy, test, fastembed compile-check). All crates green.
