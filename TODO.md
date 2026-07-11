# TODO — codeindex 0.1

Goal of 0.1: a **coherent, consumable substrate** — the crates that were
extracted from decombine, cleaned up enough that a second consumer (a CLI, a
Python binding) can depend on them without reaching back into decombine. Ordered
by priority.

## P0 — release hygiene (blocks any external consumer)

- [x] **License.** *(done 2026-07-11)* Dual `MIT OR Apache-2.0`: `LICENSE-MIT` /
      `LICENSE-APACHE` at the repo root; `license` inherited from
      `[workspace.package]`.
- [x] **Crate metadata.** *(done 2026-07-11)* Shared `version` / `edition` /
      `authors` / `repository` / `keywords` / `categories` live in
      `[workspace.package]`; each crate inherits them and keeps its own
      one-line `description`.
- [ ] **Publish scope.** All 7 crates are `publish = false`. Decide what ships to
      crates.io (facade + the six members) vs stays internal, and flip the flag
      once license + metadata land. Reserve the crate names early if publishing.

## P1 — make the substrate usable on its own

- [x] **High-level search API.** *(done 2026-07-11)* The end-to-end
      "embed a sentence → verify model identity → rank → resolve `body_hash` back
      to unit metadata" flow now lives in the new **`codeindex-search`** crate
      (`SearchIndex::{load, search_text, search_vector, similar_to_unit}`,
      `resolve_selector`), depending on `sqlite` + `embedding` + `query`.
      `codeindex-query` stays a pure, embedding-free compute layer. decombine's
      `AnalysisContext`/`VectorStore`/`query::{search,similar}` are now thin
      shims over it (presentation-only). Agents query by sentence with one call.
- [x] **Workspace integration test.** *(done 2026-07-11)*
      `crates/search/tests/round_trip.rs` walks sqlite → embedding (hash backend)
      → query → search across the seams: load, `search_text` ranking, the
      identity-mismatch error, and `similar_to_unit`.
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
