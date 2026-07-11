# Getting started

This guide covers the current library pipeline:

```text
source workspace → entities/representations → embedding spaces → snapshot → search
```

Examples use the facade crate. Enable `fastembed` for local ONNX inference.

```toml
[dependencies]
anyhow = "1"
codeindex = { path = "../codeindex/crates/codeindex", features = ["fastembed"] }
```

## 1. Embed arbitrary text without storage or parsers

Depend directly on `codeindex-embedding` for the leanest path.

```rust
use codeindex_embedding::{
    Embedder,
    config::EmbeddingConfig,
    embedder_from_config,
};

fn main() -> anyhow::Result<()> {
    let config = EmbeddingConfig {
        model: "BGESmallENV15".into(),
        ..Default::default()
    };
    let mut embedder = embedder_from_config(&config)?;
    let vectors = embedder.embed(&[
        "fn parse(input: &str) -> Result<Ast>".to_string(),
        "def parse(input): return build_ast(input)".to_string(),
    ])?;
    println!("{} dimensions", vectors[0].len());
    Ok(())
}
```

This compiles neither SQLite nor bundled language grammars.

## 2. Index ordinary filesystem projects

The common filesystem API remains `indexer::index`. It delegates to the same
workspace-neutral pipeline used by custom sources.

```rust
use codeindex::{indexer, sqlite, tree_sitter};
use codeindex::indexer::{IndexOptions, ProjectSpec, RetentionMode};

let db = sqlite::open_or_create(std::path::Path::new("index.db"))?;
let options = IndexOptions {
    projects: vec![ProjectSpec {
        label: "main".into(),
        source_dir: "./src".into(),
        exclude: vec!["**/generated/**".into()],
    }],
    enabled_languages: tree_sitter::BUNDLED_LANGUAGE_IDS
        .iter()
        .map(|id| id.to_string())
        .collect(),
    body_node_count_threshold: 10,
    max_body_chars: 10_000,
    retention: RetentionMode::Full,
};

let stats = indexer::index(&db, &options, None)?;
for project in stats {
    println!("{}: {} units", project.label, project.total_units);
}
```

The indexer extracts multiple representations for each entity, including body,
signature, symbol, documentation, and—where reference extraction is available—
usage call sites.

## 3. Index a custom source workspace

A workspace does not need to emulate a filesystem. It opens coherent snapshots
with stable document identities, logical paths, revisions, and byte content.

```rust
use codeindex::{indexer, sqlite};
use codeindex::indexer::{
    IndexSettings, MemoryWorkspace, RetentionMode, RevisionVerification,
    SourceProject,
};

let db = sqlite::open_in_memory()?;
let source = MemoryWorkspace::new("memory://workspace");
source.insert(
    "src/lib.rs",
    "fn answer() -> i32 { let value = 42; value }",
);

let settings = IndexSettings {
    enabled_languages: vec!["rust".into()],
    body_node_count_threshold: 1,
    max_body_chars: 10_000,
    retention: RetentionMode::Full,
    revision_verification: RevisionVerification::Verified,
};
let projects = [SourceProject {
    label: "main".into(),
    workspace: &source,
}];
indexer::index_sources(&db, &settings, &projects, None)?;
```

Implement `SourceWorkspace` and `SourceSnapshot` for database rows, Git objects,
object-store keys, archives, editor buffers, notebooks, or generated code.
Preserve the same document id across a logical move when the provider can do so
reliably.

### Checkpoints and bounded change feeds

A workspace may expose a checkpoint on each snapshot and advertise
`change_feed`. After a successful reconciliation, the SQLite backend persists the
checkpoint for that project. On the next run, the indexer asks the workspace for
a complete `SourceDelta` bounded by the stored checkpoint and the newly opened
snapshot checkpoint.

The indexer applies only the coalesced document upserts, moves, and removals in
that delta. It advances the checkpoint only after every changed document is
processed successfully. Unsupported or expired checkpoints trigger a full
snapshot reconciliation; incomplete feeds must return an error rather than a
partial delta.

## 4. Retention, provenance, and durable recovery

Representations carry provenance:

- `Extracted`: deterministic frontend output, recoverable from source;
- `Derived`: synthesized from indexed facts, such as `Usage`;
- `Imported`: supplied by a consumer or external model.

Retention modes are provenance-aware:

- `Full` retains all representation text.
- `Report` may drop extracted embed-only text while retaining display and
  non-recoverable derived/imported text.
- `Minimal` may drop all recoverable extracted representation text.

Whenever a document is read for indexing, the SQLite backend stores its complete
source bytes in a deduplicated content-addressed cache keyed by the verified
source hash. Delayed embedding first reconstructs missing representation text
from that cache, so it does not depend on the original provider, working tree,
transaction, object version, or snapshot still being available. Provider-backed
recovery remains an exact-revision fallback when a cache entry is absent.

Source blobs are integrity-checked on insertion and read, and blobs no longer
referenced by indexed files are pruned with the rest of the index garbage.

## 5. Embed one explicit space

An embedding space binds a channel to an exact model identity. Spaces are
independent: a code model can embed implementations while a text model embeds
documentation.

```rust
use codeindex::core::{EmbeddingSpaceIdentity, RepresentationKind};
use codeindex::embedding::{
    Embedder,
    config::{EmbeddingConfig, EmbeddingRunConfig, SourceRecoveryConfig},
    embedder_from_config,
};
use codeindex::indexer;

let embedding_config = EmbeddingConfig {
    model: "BGESmallENV15".into(),
    ..Default::default()
};
let run_config = EmbeddingRunConfig {
    embedding: embedding_config.clone(),
    source_recovery: SourceRecoveryConfig {
        body_node_count_threshold: 10,
    },
};
let mut embedder = embedder_from_config(&embedding_config)?;
let space = EmbeddingSpaceIdentity::new(
    "docs",
    RepresentationKind::Documentation,
    embedder.identity().clone(),
);
let stats = indexer::embed_space_pending(
    &db,
    embedder.as_mut(),
    &run_config,
    &space,
)?;
println!("embedded {} representations", stats.embedded);
```

Reusing the id `docs` with a different channel, model identity, or input
transform is rejected.

`embed_pending` is a convenience operation that uses one embedder for every
present channel, creating spaces named `default/<channel>`.

## 6. Load a storage-neutral search index

SQLite exports the public `IndexSnapshot`; another backend can construct the
same type directly.

```rust
use codeindex::search::SearchIndex;

let snapshot = db.snapshot(&[])?; // empty project list = all projects
let index = SearchIndex::from_snapshot(snapshot)?;

for space in index.embedded_spaces() {
    println!("{}: {} via {}", space.id, space.channel, space.model.model);
}
```

Snapshot loading validates dimensions, duplicate ids/hashes, and finite vector
values rather than trusting external stores.

## 7. Search an explicit space

The query embedder must exactly match the selected space's model identity.

```rust
use codeindex::core::EmbeddingSpaceId;
use codeindex::query::WhereFilter;

let results = index.search_text(
    embedder.as_mut(),
    "parse command line flags",
    &EmbeddingSpaceId::new("docs"),
    &WhereFilter::default(),
    10,
)?;

for hit in results.hits {
    let unit = &index.units[hit.index];
    println!("{:.4}  {}  {}", hit.score, unit.location(), unit.name);
}
```

Use `similar_to_unit` for query-by-example retrieval in one space.

## 8. Fuse multiple spaces

Do not add cosine scores from different models. Embed each query with the
corresponding model, normalize according to that space, then fuse ranked lists
with weighted reciprocal rank:

```rust
use codeindex::core::EmbeddingSpaceId;
use codeindex::query::WhereFilter;
use codeindex::search::SpaceVectorQuery;

let code_id = EmbeddingSpaceId::new("code");
let docs_id = EmbeddingSpaceId::new("docs");
let fused = index.search_vectors_fused(
    &[
        SpaceVectorQuery {
            space_id: &code_id,
            vector: &code_query_vector,
            weight: 1.0,
        },
        SpaceVectorQuery {
            space_id: &docs_id,
            vector: &docs_query_vector,
            weight: 0.8,
        },
    ],
    &WhereFilter::default(),
    20,
    60,
)?;
```

Each fused hit preserves its rank, raw score, weight-derived contribution, and
space id for explanation.

## 9. Metadata filtering

`WhereFilter` is independent of storage and embedding space:

```rust
use codeindex::query::WhereFilter;

let filter = WhereFilter::parse(Some(
    "language=rust kind=function path=src/** min_nodes=20",
))?;
```

Supported keys are `project`, `language`, `kind`, `name`, `scope`, `path`, and
`min_nodes`.

## Schema compatibility

The schema is pre-release. Schema epoch 3 adds persisted source checkpoints and
the durable content-addressed source cache. Databases from incompatible epochs
are rejected with a delete-and-reindex message rather than partially migrated.
