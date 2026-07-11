# codeindex

A reusable Rust substrate for **code intelligence over local embeddings**:
parse source with Tree-sitter, project functions/methods/types into a compact
per-entity representation, embed them with a local ONNX model, store everything
in incremental SQLite, and rank by semantic similarity — all offline, no network
at query time.

`codeindex` is the engine extracted from [decombine](../decombine2) (a non-exact
duplication detector). It is split into small crates so you can take only what
you need: a notebook binding that just wants to embed strings pulls in neither
SQLite nor the grammars, while a full indexing CLI takes the whole stack.

## Crates

| Crate | Responsibility | Heavy deps |
|-------|----------------|-----------|
| `codeindex-core` | Parser/storage-neutral vocabulary: `LanguageId`, `EntityKind`, `SourceSpan`, `RepresentationKind`, `ExtractedEntity`, `ModelIdentity`. | none |
| `codeindex-tree-sitter` | Bundled grammars (12 languages), declarative language specs + adapters, normalization, and extraction into `ExtractedEntity`. | tree-sitter grammars |
| `codeindex-sqlite` | Incremental schema, migrations, model identities, vector blobs, and the persistence API. Owns the `ExtractedEntity → NewCodeUnit` projection. | bundled SQLite |
| `codeindex-indexer` | Filesystem scan → change detection → extract → retention → transactional store. Also the workflow that **embeds a stored corpus** (resumable projection, source recovery, token reports). | SQLite + grammars |
| `codeindex-embedding` | Local model execution (fastembed/ONNX), provider diagnostics, batch packing, normalization, token stats. **No storage or parser deps** — safe for a lean binding. | fastembed/ort (feature-gated) |
| `codeindex-query` | Stable `unit:` selectors, `--where` metadata filtering, identity diffing, deterministic vector ranking. | none |
| `codeindex` | A thin facade re-exporting all of the above under `core`, `tree_sitter`, `sqlite`, `indexer`, `embedding`, `query`. | — |

### Dependency graph

```
codeindex-core  (leaf: pure vocabulary, incl. ModelIdentity)
  ├── codeindex-tree-sitter   (+ grammars)
  ├── codeindex-sqlite        (+ bundled SQLite; From<ExtractedEntity>)
  └── codeindex-embedding     (+ fastembed, feature-gated)   ← storage/parser-free
        └── codeindex-indexer (core + sqlite + tree-sitter + embedding)
codeindex-query (→ sqlite)
codeindex       (facade → all)
```

The single deliberate rule: **`codeindex-embedding` never depends on SQLite or
the grammars.** The pure primitives (the `Embedder` trait, batch packer,
normalization, token stats) live there; anything that needs a stored corpus
lives in `codeindex-indexer`.

## Quickstart

Add the facade (path or git dependency) and enable a backend feature:

```toml
[dependencies]
codeindex = { git = "https://…/codeindex", features = ["fastembed"] }
# or, locally:
# codeindex = { path = "../codeindex/crates/codeindex", features = ["fastembed"] }
```

### Just embed text (lean path — no SQLite, no grammars)

Depend only on `codeindex-embedding` with the `fastembed` feature:

```rust
use codeindex_embedding::{config::EmbeddingConfig, embedder_from_config};

let cfg = EmbeddingConfig { model: "BGESmallENV15".into(), ..Default::default() };
let mut embedder = embedder_from_config(&cfg)?;          // downloads the model on first use
let vectors: Vec<Vec<f32>> = embedder.embed(&["fn parse(&self) {}".to_string()])?;
```

### Index a project and embed the corpus

```rust
use codeindex::{indexer, sqlite, tree_sitter};
use codeindex::indexer::{IndexOptions, ProjectSpec, RetentionMode};

let db = sqlite::open_or_create(std::path::Path::new("index.db"))?;

let options = IndexOptions {
    projects: vec![ProjectSpec {
        label: "main".into(),
        source_dir: "./src".into(),
        exclude: vec![],
    }],
    enabled_languages: tree_sitter::BUNDLED_LANGUAGE_IDS.iter().map(|s| s.to_string()).collect(),
    body_node_count_threshold: 10,
    max_body_chars: 10_000,
    retention: RetentionMode::Full,
};
indexer::index(&db, &options, None)?;   // incremental: unchanged files are skipped
```

Embedding the stored bodies (resumable; re-run to pick up new units) is the
`codeindex-indexer::embed_pending` half — see
[`docs/getting-started.md`](docs/getting-started.md) for the full flow including
`codeindex-query` ranking.

## Features

`codeindex-embedding` (and, transitively, the `codeindex` facade) gate the
model runtime behind cargo features so a build that only parses/stores never
compiles ONNX:

- `fastembed` — the local ONNX backend (fastembed). Required to run any model.
- `accel` and `cuda` / `directml` / `coreml` / `openvino` — hardware execution
  providers on top of `fastembed`.
- `load-dynamic` — link ONNX Runtime dynamically.

With no features, the crates parse, store, and rank pre-computed vectors, but
cannot produce embeddings.

## Building

```sh
cargo build --workspace                       # parse/store/rank; no model runtime
cargo build -p codeindex-embedding --features fastembed   # with the ONNX backend
cargo test --workspace
```

See [`docs/architecture.md`](docs/architecture.md) for the design rationale and
[`docs/getting-started.md`](docs/getting-started.md) for a complete example.
