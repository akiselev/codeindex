# codeindex

A reusable Rust substrate for semantic code intelligence. `codeindex` accepts
source from arbitrary providers, extracts parser-neutral entities and multiple
textual representations, projects those representations into independently
modelled embedding spaces, persists them incrementally, and exposes structured
search primitives. Local ONNX inference is supported, but consumers can provide
any deterministic embedder.

`codeindex` is the engine extracted from `decombine`. It is a library workspace;
a future `codeindex-cli`, IDE integrations, agents, bindings, daemons, and
analysis applications are separate consumers.

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

The compatibility `index()` function still indexes ordinary filesystem
projects. The underlying operation is `index_sources()`, which accepts any
`SourceProvider`:

```rust
use codeindex::indexer::{
    IndexSettings, MemorySource, RetentionMode, SourceProject, index_sources,
};
use codeindex::sqlite;

let db = sqlite::open_in_memory()?;
let mut source = MemorySource::new("memory://workspace");
source.insert(
    "src/lib.rs",
    "fn answer() -> i32 { 42 }",
);

let settings = IndexSettings {
    enabled_languages: vec!["rust".into()],
    body_node_count_threshold: 1,
    max_body_chars: 10_000,
    retention: RetentionMode::Full,
};
index_sources(
    &db,
    &settings,
    &[SourceProject {
        label: "main".into(),
        provider: &source,
    }],
    None,
)?;
```

Providers expose stable document identities, logical paths, opaque revisions,
and UTF-8 content. Database, object-store, Git-tree, archive, generated-source,
and editor-overlay providers can reuse the same indexing pipeline.

## Representations

Each entity can carry multiple independently versioned representation channels:

- `FullSource`
- `Implementation`
- `Body`
- `BodyWithoutDeclaredName`
- `Signature`
- `Symbol`
- `Documentation`
- `Usage`
- `GeneratedDescription`
- consumer-defined custom channels

Representations include provenance. Deterministic frontend channels are marked
`Extracted`; corpus-derived channels such as `Usage` are `Derived`; external or
model-generated channels are `Imported`. Consumers can register
`RepresentationEnricher` implementations before retention is applied.

## Embedding spaces

An embedding space binds one representation channel to one exact model identity
and input transform. A corpus can therefore use a code model for implementations
and a text model for documentation or generated descriptions:

```rust
use codeindex::core::{EmbeddingSpaceIdentity, RepresentationKind};
use codeindex::indexer::embed_space_pending;

let code_space = EmbeddingSpaceIdentity::new(
    "code",
    RepresentationKind::Implementation,
    code_embedder.identity().clone(),
);
embed_space_pending(&db, &mut code_embedder, &run_config, &code_space)?;

let docs_space = EmbeddingSpaceIdentity::new(
    "docs",
    RepresentationKind::Documentation,
    text_embedder.identity().clone(),
);
embed_space_pending(&db, &mut text_embedder, &run_config, &docs_space)?;
```

`embed_pending()` remains a convenience operation. It creates one
`default/<channel>` space per embeddable channel using the same embedder.

Search selects an explicit space. For multi-model retrieval,
`SearchIndex::search_vectors_fused` combines independently ranked result lists
with weighted reciprocal rank; raw cosine values from incompatible models are
not averaged.

## Local model backend

Enable `fastembed` to run supported local ONNX models. Accelerator features are
available for CUDA, DirectML, CoreML, and OpenVINO.

```toml
[dependencies]
codeindex = { git = "https://github.com/akiselev/codeindex", features = ["fastembed"] }
```

A build without embedding backend features can still extract, persist, load,
and rank externally computed vectors.

## Building

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo check -p codeindex --features fastembed
```

The database schema is pre-release. Incompatible schema epochs are rejected with
an explicit delete-and-reindex error rather than migrated.
