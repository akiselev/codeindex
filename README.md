# codeindex

A reusable Rust substrate for semantic code intelligence. `codeindex` accepts
source from arbitrary providers, extracts parser-neutral entities and multiple
textual representations, projects those representations into independently
modelled embedding spaces, persists them incrementally, and exposes structured
search primitives. Local ONNX inference is supported, but consumers can provide
any deterministic embedder.

`codeindex` is the engine extracted from `decombine`. The workspace is primarily
a reusable library substrate. Its small `codeindex-cli` binary is the first
consumer and intentionally delegates indexing state, refresh, resume, and
publication to the library.

## Crates

| Crate | Responsibility | Heavy dependencies |
|---|---|---|
| `codeindex-core` | Application-neutral entity, representation, provenance, model, and embedding-space vocabulary. | none |
| `codeindex-tree-sitter` | Bundled grammars, declarative language definitions, entity extraction, normalization, and call-site capture. | Tree-sitter grammars |
| `codeindex-storage` | Serializable `IndexSnapshot` contract between persistence backends and search. | none |
| `codeindex-sqlite` | Default incremental store, schema, entity versions, representations, embedding spaces, vectors, and snapshot export. | bundled SQLite |
| `codeindex-indexer` | Provider-neutral incremental indexing, representation enrichment, Usage synthesis, source recovery, and resumable embedding projection. | SQLite + grammars |
| `codeindex-embedding` | Storage/parser-free `Embedder` trait, local ONNX backends, model management, batching, normalization, and token statistics. | fastembed/ort when enabled |
| `codeindex-query` | Stable selectors, metadata filtering, model identity diagnostics, and deterministic ranking kernels. | none |
| `codeindex-search` | Validated snapshot loading, per-space search, similarity search, and reciprocal-rank fusion across spaces. | none beyond embedding/query primitives |
| `codeindex` | Thin facade re-exporting the eight component crates. | selected features |
| `codeindex-cli` | Thin atomic index/status/abandon/supersede command-line consumer (resume via `index --resume`). | SQLite + grammars |

The major dependency boundaries are deliberate:

```text
codeindex-core
  ├── codeindex-tree-sitter
  ├── codeindex-storage
  └── codeindex-embedding

codeindex-query ───────────────→ codeindex-core
codeindex-sqlite ──────────────→ codeindex-core + codeindex-storage
codeindex-search ──────────────→ core + storage + embedding + query
codeindex-indexer ─────────────→ core + tree-sitter + sqlite + embedding
```

`codeindex-search` never touches SQLite. Any backend that can construct an
`IndexSnapshot` can use the complete search layer. `codeindex-embedding` never
pulls in SQLite or language grammars.

## Source providers

`index()` indexes ordinary filesystem projects; the underlying
`index_sources()` accepts any `SourceProvider` — database rows, Git objects,
editor buffers, object stores, archives, or generated code. Providers expose
stable document identities, logical paths, opaque revisions, and UTF-8
content. Indexing is atomic for the complete selected scope: extraction is
checkpointed in a durable journal while queries keep seeing the prior
generation, and a no-change refresh barrier publishes everything in one
transaction. See `docs/getting-started.md` §2–3 for runnable examples and
`docs/architecture.md` for the provider contract.

## Representations

Each entity carries multiple independently versioned representation channels
(`FullSource`, `Implementation`, `Body`, `BodyWithoutDeclaredName`,
`Signature`, `Symbol`, `Documentation`, derived `Usage`, imported
`GeneratedDescription`, and custom channels), each with provenance. The full
model is documented in `docs/architecture.md`.

## Embedding spaces

A space binds one representation channel to one semantic model contract and a
document-side contract (document prompt + Matryoshka output dimensions), so a
code model can embed implementations while a text model embeds documentation.
Search selects an explicit space; multi-model retrieval fuses independently
ranked lists with weighted reciprocal rank rather than averaging raw cosines
across models. Query-side task instructions render per request — one document
index serves many retrieval intents. Examples: `docs/getting-started.md` §5–7.

## Local model backends

Models are referenced generically — `hf:owner/name[@rev]`, `dir:/path`, or
`fastembed:Name` — and their semantic contract (pooling, prompts, dimensions,
max length) is resolved from the repository's own sentence-transformers
configuration, hash-locked trust-on-first-use. Two execution backends:

- `fastembed` — bundled ONNX/ort execution for mean/cls encoder models
  (BGE, MiniLM, Jina v2 code, the managed CodeRankEmbed), with CUDA,
  DirectML, CoreML, and OpenVINO accelerator features;
- `candle` — native execution for decoder-style last-token models
  (Qwen3-Embedding), with `candle-cuda`/`candle-metal` device features and
  instruction-aware queries (`Instruct:/Query:` rendered per request).

```toml
[dependencies]
codeindex = { git = "https://github.com/akiselev/codeindex", features = ["fastembed", "candle"] }
```

There is no implicit default model; every consumer picks an explicit
reference. A build without backend features can still extract, persist, load,
and rank externally computed vectors.

## Building

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo check -p codeindex --features fastembed
cargo run -p codeindex-cli -- --help
```

The database schema is pre-release (`codeindex_sqlite::SCHEMA_VERSION` is the
current epoch); incompatible epochs are rejected with an explicit
delete-and-reindex error rather than migrated.
