# codeindex

A reusable Rust substrate for semantic code intelligence. `codeindex` accepts
source from arbitrary workspaces, extracts parser-neutral entities and multiple
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
| `codeindex-source` | Typed workspace, snapshot, document, revision, checkpoint, capability, content, error, memory, and overlay contracts. | none |
| `codeindex-source-fs` | Default filesystem workspace with ignore rules and revision-validated snapshot reads. | `ignore` |
| `codeindex-tree-sitter` | Bundled grammars, declarative language definitions, entity extraction, normalization, and call-site capture. | Tree-sitter grammars |
| `codeindex-storage` | Serializable `IndexSnapshot` contract between persistence backends and search. | none |
| `codeindex-sqlite` | Default incremental store, schema, entity versions, representations, embedding spaces, vectors, and snapshot export. | bundled SQLite |
| `codeindex-indexer` | Workspace-neutral incremental indexing, language resolution, representation enrichment, Usage synthesis, source recovery, and resumable embedding projection. | SQLite + grammars |
| `codeindex-embedding` | Storage/parser-free `Embedder` trait, local ONNX backends, model management, batching, normalization, and token statistics. | fastembed/ort when enabled |
| `codeindex-query` | Stable selectors, metadata filtering, model identity diagnostics, and deterministic ranking kernels. | none |
| `codeindex-search` | Validated snapshot loading, per-space search, similarity search, and reciprocal-rank fusion across spaces. | none beyond embedding/query primitives |
| `codeindex` | Thin facade re-exporting the component crates. | selected features |

The major dependency boundaries are deliberate:

```text
codeindex-source ──────────────→ std only
  └── codeindex-source-fs      → ignore

codeindex-core
  ├── codeindex-tree-sitter
  ├── codeindex-storage
  └── codeindex-embedding

codeindex-query ───────────────→ codeindex-core
codeindex-sqlite ──────────────→ codeindex-core + codeindex-storage
codeindex-search ──────────────→ core + storage + embedding + query
codeindex-indexer ─────────────→ source + source-fs + core + tree-sitter + sqlite + embedding
```

`codeindex-search` never touches SQLite. Any backend that can construct an
`IndexSnapshot` can use the complete search layer. `codeindex-embedding` never
pulls in SQLite or language grammars. Implementing a custom source workspace does
not require either dependency.

## Source workspaces

The compatibility `index()` function still indexes ordinary filesystem
projects. The underlying operation is `index_sources()`, which accepts any
`SourceWorkspace` through a `SourceProject`:

```rust
use codeindex::indexer::{
    IndexSettings, MemoryWorkspace, RetentionMode, RevisionVerification,
    SourceProject, index_sources,
};
use codeindex::sqlite;

let db = sqlite::open_in_memory()?;
let source = MemoryWorkspace::new("memory://workspace");
source.insert(
    "src/lib.rs",
    "fn answer() -> i32 { 42 }",
);

let settings = IndexSettings {
    enabled_languages: vec!["rust".into()],
    body_node_count_threshold: 1,
    max_body_chars: 10_000,
    retention: RetentionMode::Full,
    revision_verification: RevisionVerification::Verified,
};
index_sources(
    &db,
    &settings,
    &[SourceProject {
        label: "main".into(),
        workspace: &source,
    }],
    None,
)?;
```

A workspace opens a `SourceSnapshot`. One indexing run enumerates and reads only
through that snapshot, so a database transaction, Git commit, object-store
version, archive, editor buffer set, or validated filesystem view can provide a
coherent corpus. Snapshots expose:

- stable `WorkspaceId`, `SnapshotId`, `DocumentId`, and `SourceRootId` values;
- logical paths separate from provider identity and physical location;
- strong content revisions or explicitly weak metadata hints;
- streamed, fallible document enumeration;
- direct and batched reads with observed revision metadata;
- structured, retry-aware source errors;
- optional checkpoints and change feeds;
- declared consistency and capability levels.

Language resolution belongs to the indexer. Providers return a `LanguageHint`
such as a known language, file extension, media type, shebang, or unknown value.
This supports extensionless database rows, virtual documents, notebooks, and
future non-Tree-sitter frontends without coupling providers to the grammar
registry.

`RevisionVerification::Fast` trusts equal provider revision tokens. This is the
default used by the filesystem convenience API. `RevisionVerification::Verified`
reads and hashes documents when a provider marks its revision as a metadata hint.
Strong revisions such as Git blob IDs, immutable row versions, and versioned
object IDs are always eligible for a cheap skip.

`MemoryWorkspace` creates immutable snapshots for tests, generated sources, and
in-memory corpora. `OverlayWorkspace` composes an overlay over a base workspace,
which is the intended shape for unsaved editor buffers over filesystem or Git
content. Database, object-store, Git-tree, archive, and notebook integrations can
implement the dependency-light `codeindex-source` traits directly.

Workspaces expose stable document identities, logical paths, revisions, coherent
snapshots, and optional bounded change feeds. The indexer persists provider
checkpoints and consumes complete deltas when available, falling back to a full
reconciliation when a checkpoint expires. Indexed source is retained in a
deduplicated content-addressed cache so minimal/report representations can be
reconstructed after the original snapshot disappears. Cache entries are retained
while their indexed file version remains present and are pruned when no indexed
file references that source hash.

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
