# Architecture

`codeindex` is split by **dependency and change boundary**, not by command. The
guiding constraint is that the pure embedding primitives must never drag in
storage or parser code, so that a lightweight consumer (e.g. a Python/PyO3
binding for exploring embeddings in a notebook) can depend on just the embedder
without compiling bundled SQLite or twelve Tree-sitter grammars.

## The crates

- **`codeindex-core`** — application-neutral vocabulary with no dependencies of
  its own: `LanguageId`, `EntityKind`, `SourceSpan`, the `RepresentationKind`
  channels, `ExtractedEntity`/`ExtractedFile`, and the embedding `ModelIdentity`.
  `ModelIdentity` lives here (not in the store) because it is the shared contract
  between the backends that *produce* it and the persistence layer that *stores*
  it; keeping it in the leaf lets `codeindex-embedding` name it without depending
  on SQLite.

- **`codeindex-tree-sitter`** — the language frontend. Grammars are compiled in;
  each language is declared by `assets/languages/<id>.toml` (extensions, comment
  node kinds, scope rules, adapter name) plus `<id>/units.scm` (the Tree-sitter
  query that captures units, names, bodies, and strip ranges). Irregular cases
  that a query cannot express — anonymous-function naming, receiver scopes,
  docstring stripping — live in `LanguageAdapter` implementations in
  `language.rs`. Extraction produces parser-neutral `ExtractedEntity` values.

- **`codeindex-sqlite`** — the incremental schema, migrations, model-identity
  rows, vector blobs, immutable-setting guards, and the persistence API (`Db`).
  It owns the single `impl From<ExtractedEntity> for NewCodeUnit` — the one place
  the representation channels map onto the current schema's
  `display_source`/`embedding_text` columns.

- **`codeindex-indexer`** — orchestration over the store. `index()` walks a
  project root (honoring `.gitignore` and explicit excludes), detects unchanged
  files by mtime/size then content hash, extracts, applies a retention policy,
  and writes transactionally. Its `embed` module is the **stored-corpus
  embedding workflow**: `embed_pending` resumably projects every distinct
  un-embedded body into a vector, recovers embedding text from source when lean
  retention did not store it, and `token_report` measures token-length
  distributions. This is the storage- and parser-coupled half of embedding.

- **`codeindex-embedding`** — the storage/parser-free half. The `Embedder`
  trait, the fastembed/ONNX backend (feature-gated), provider diagnostics, the
  length-sorted **batch packer** (bounds a batch's padded token area — the term
  ONNX attention memory scales with), vector normalization, and `TokenStats`.
  Depends only on `codeindex-core`.

- **`codeindex-query`** — read-side primitives an agent-facing query layer needs:
  the `UnitView` trait, stable `unit:<hash>` selectors and one-line rendering,
  `WhereFilter` (`project=`, `language=`, `kind=`, `name=`, `scope=`, `path=`,
  `min_nodes=`), deterministic `rank_candidates`, and `identity_diff` for
  explaining a model-identity mismatch.

- **`codeindex`** — a facade that re-exports the six crates as `core`,
  `tree_sitter`, `sqlite`, `indexer`, `embedding`, and `query` for consumers
  that prefer a single dependency, forwarding the `fastembed`/`accel` features.

## Data flow

```
source files
   │  codeindex-tree-sitter: parse + query + adapters + normalize
   ▼
ExtractedEntity            (parser-neutral, N representation channels)
   │  From<ExtractedEntity>  (codeindex-sqlite)
   ▼
NewCodeUnit  ──▶  code_units table          (codeindex-indexer::index)
                     │  distinct normalized_body_hash
                     ▼
                 embed_pending               (codeindex-indexer::embed)
                     │  Embedder (codeindex-embedding), batch-packed
                     ▼
                 embeddings table  (vector blob keyed by model_id + body_hash)
                     │
                     ▼
                 rank_candidates             (codeindex-query)
```

## Design invariants

- **Determinism.** For a fixed `ModelIdentity`, embeddings are reproducible;
  `rank_candidates` breaks score ties by index; `unit_id` is a stable hash over
  `(project, path, byte range, body hash, name, scope, language)`.
- **Model identity is a key, not a label.** A database binds to one
  `ModelIdentity`; embeddings for different backends/models/providers never
  share a row. Immutable settings (body threshold, retention, normalization,
  model identity) are checked on every run.
- **Incremental by content.** Re-indexing skips files whose mtime/size are
  unchanged, then whose content hash is unchanged. Embeddings are keyed by
  `(model, body-hash)`, so they survive re-indexing and are shared across
  identical bodies.
- **Retention is a projection, not a parse-time decision.** `Full` keeps display
  source and embedding text; `Report` drops embedding text; `Minimal` drops both
  and re-derives them from source when needed. Extraction output is identical
  across modes.

## Extending

- **A new language** = a Tree-sitter grammar dependency + `assets/languages/
  <id>.toml` + `assets/languages/<id>/units.scm`, added to the `bundled!` list
  in `crates/tree-sitter/src/language.rs`. Reach for a `LanguageAdapter` only
  when a capture cannot be expressed in the query.
- **A new backend** = another `Embedder` implementation plus a branch in
  `embedder_from_config`. Backends must be deterministic for a fixed identity.
- **A new representation channel** = a `RepresentationKind` variant emitted by
  the frontend; persisting more than one channel is a schema migration in
  `codeindex-sqlite`, deliberately kept separate from extraction.
