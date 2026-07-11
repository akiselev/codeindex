# PLAN.md — M1 + M4 implementation (in progress)

Handoff doc for finishing the M1 (standalone search API) + M4 (multi-channel &
entity versions) work. Full design rationale is in `docs/m1-m4-plan.md`; this
file tracks **what is done and what remains**. Branch: `m1-m4-multichannel`.

## Overriding constraints (from maintainer)
1. **No migration story** — schema is pre-release; rewrite it optimally.
   `crates/sqlite/src/migrations.rs` is now a single bootstrap that *rejects*
   (not migrates) a database at another version.
2. **Any database via a public type** — search loads only from
   `codeindex_storage::IndexSnapshot` (serde). SQLite produces it via
   `Db::snapshot`; any other store deserializes its rows into the same type.
3. **Full Usage extraction** — resolve call sites across the corpus (not just
   reserve the channel).

## Architecture delta
- New crate **`codeindex-storage`** (serde-only leaf, depends on core): the
  `IndexSnapshot` / `UnitRecord` / `RepresentationRef` / `ChannelEmbeddings`
  public wire types. This is the storage↔search seam.
- `codeindex-core`: `RepresentationKind` now has canonical `as_str`/`From<&str>`
  + `Display` + string-based serde; serde on `LanguageId`/`EntityId`/
  `EntityVersionId`/`SourceSpan`.
- Schema (M4): `entities` ledger, `code_units` carries
  `entity_id`/`entity_version_id`/`generation` (no more inline
  `display_source`/`embedding_text`), `representations(unit_id, kind,
  content_hash, content)`, `embeddings` keyed by `(model_id, channel,
  content_hash)`, `references_raw` staging for Usage.
- `codeindex-query` repointed off `codeindex-sqlite` onto `codeindex-core`
  (`ModelIdentity`), so search no longer transitively pulls SQLite.

## Status by phase

| Phase | What | State |
|------|------|-------|
| 1 | `codeindex-storage` + `SearchIndex::from_snapshot` + `Db::snapshot`; drop search→sqlite dep | code done; search crate not yet compiled/tested |
| 2 | schema rewrite, `representations`/`entities`/`references_raw`, channel embeddings, `models.rs`, `Db` CRUD | DONE, 15 tests pass |
| 3 | `generation` counter + within-file entity matcher (`assign_identity`); populate ids | DONE (indexer builds) |
| 4a | Signature/Documentation/Symbol channels in tree-sitter `build_unit` | DONE (tree-sitter tests pass) |
| 4b | `references.scm` (Rust), `extract_references`, `references_raw`, `resolve_usage` pass | code done; needs an end-to-end test |
| 5 | channel-aware `embed_pending` (loops `embeddable_channels`); multi-`VectorStore` `SearchIndex`; `search_*(channel,…)` | embed DONE (indexer builds); search rewrite done, NOT yet compiled |
| 6 | tests + docs | TODO |

## What compiles right now
`cargo test` green for: `codeindex-core`, `codeindex-storage`,
`codeindex-tree-sitter`, `codeindex-sqlite` (15), `codeindex-indexer` (5).

## REMAINING WORK (do these next, in order)

1. **Compile `codeindex-query`** after the ModelIdentity repoint
   (`cargo test -p codeindex-query`). Its unit tests define a local `Unit`
   implementing `UnitView` — should be unaffected.

2. **Compile `codeindex-search`** (`cargo build -p codeindex-search`). The
   rewrite in `crates/search/src/lib.rs` is fresh — expect small fixups:
   - `CodeUnitRef` no longer has a numeric `id`; nothing internal uses it.
   - methods now take a `channel: &RepresentationKind` argument.

3. **Fix `crates/search/tests/round_trip.rs`** — it still uses the OLD API
   (`NewCodeUnit { display_source, embedding_text, ... }`, `SearchIndex::load`,
   `db.insert_embedding(model, hash, vec)`, `search_text` without a channel).
   Rewrite it to:
   - build units with the new `NewCodeUnit { entity_id, entity_version_id,
     generation, …, representations: vec![NewRepresentation{…}] }` OR (simpler)
     drive it through `codeindex_indexer::index` + `embed_pending` on a temp dir
     of real `.rs` files;
   - load via `SearchIndex::from_snapshot(db.snapshot(&[])?)`;
   - pass `&RepresentationKind::Implementation` to `search_text` /
     `similar_to_unit`.

4. **Compile the facade** `crates/codeindex/src/lib.rs` — add
   `pub use codeindex_storage as storage;` and confirm re-exports still resolve.
   Update `crates/codeindex/Cargo.toml` to depend on `codeindex-storage`.

5. **`cargo build --workspace` + `cargo test --workspace`** green. Then
   `cargo clippy --workspace --all-targets` and `cargo fmt`.

6. **Phase 6 tests to add** (design §10 of `docs/m1-m4-plan.md`):
   - `codeindex-storage`: snapshot JSON round-trip (already have one).
   - **"any database" proof**: hand-build an `IndexSnapshot` (no SQLite) and run
     `SearchIndex::from_snapshot(...).search_vector(...)`.
   - **per-channel search**: index Rust source, embed, assert a query ranks
     differently against `Signature` vs `Documentation` vs `Implementation`.
   - **entity identity**: index a fn, rename it, re-index; assert same
     `entity_id`, different `entity_version_id` (via `db.snapshot` units or a
     direct `entities`/`code_units` query).
   - **Usage**: two `.rs` files where A calls B; after index, assert B's unit has
     a `Usage` representation containing A's call site; embed + search it.
   - tree-sitter: assert `build_unit` emits Signature/Documentation/Symbol for a
     documented Rust fn; assert `extract_references` finds a call.

7. **Docs**: update `docs/architecture.md` (new crate, snapshot boundary,
   channel/entity schema, per-channel embeddings); tick M1 + M4 in `ROADMAP.md`;
   resolve the "dormant core vocabulary" P2 item in `TODO.md`.

## Known limitations / non-goals (documented, intentional)
- Entity matching is **within-file only** (rename ok; cross-file move mints a new
  id). Cross-file/-crate tracking is future work.
- Usage resolution is **same-project, name-based** (last path segment); ambiguous
  names resolve to every candidate. Import-/type-aware resolution is future work.
- Usage is **recomputed per index run** for any project that changed (no
  incremental invalidation yet).
- Only Rust has a populated `references.scm`; other languages have empty
  placeholders (their units still index; Usage stays empty). Adding a language =
  fill its `crates/tree-sitter/assets/languages/<id>/references.scm` with a
  `@ref.callee` capture.

## Gotchas discovered
- `embeddable_channels()` = every channel present in `representations` except
  `FullSource` (display-only). Embed loops over it, so new channels are picked up
  automatically.
- `Db::snapshot` requires exactly one embedding model (the corpus invariant) and
  errors on 0 or >1.
- Reference attribution picks the **innermost** unit whose byte span contains the
  call (largest `start_byte`), so calls inside nested closures attribute to the
  closure.
