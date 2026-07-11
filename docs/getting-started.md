# Getting started

This guide covers the current library pipeline:

```text
source workspace → immutable snapshot → entities/representations → embedding spaces → index snapshot → search
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

The common filesystem API remains `indexer::index`. It builds a
`FilesystemWorkspace`, opens one validated snapshot per indexing run, and then
delegates to the same pipeline used by custom sources.

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

The filesystem convenience API uses `RevisionVerification::Fast`, preserving
cheap metadata-based incremental scans. Build `IndexSettings` directly when
weak revisions must be read and hashed before they are trusted.

## 3. Index a custom source workspace

A workspace does not emulate a filesystem. It opens a snapshot containing stable
document identities, logical paths, revision metadata, language hints, and
revision-checked content reads.

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

Implement `codeindex::source::SourceWorkspace` and
`codeindex::source::SourceSnapshot` for database transactions, Git commits,
object-store versions, archives, notebooks, editor buffers, or generated code.
Preserve the same `DocumentId` across a logical move when the provider can do so
reliably.

A `SourceSnapshot` exposes streamed enumeration and direct reads. Providers may
override batch reads, direct lookup, checkpoints, and change feeds. The indexer
resolves final languages from `LanguageHint`, so providers can supply a known
language, extension, media type, shebang, or no hint.

## 4. Snapshot consistency and revisions

`SnapshotConsistency` describes the guarantees a provider offers:

- `Immutable`: Git commits, immutable archives, or pinned object versions;
- `Transactional`: a repeatable-read database transaction or provider snapshot;
- `Validated`: a live source whose reads are checked against enumerated revisions;
- `BestEffort`: a live view that can report stale reads.

`RevisionGuarantee::ContentIdentity` means equal tokens guarantee equal bytes.
`RevisionGuarantee::MetadataHint` means the token is only a cheap change hint.
`RevisionVerification::Verified` hashes weak revisions, while `Fast` trusts them.

`OverlayWorkspace` composes a higher-priority workspace over a base workspace.
The primary intended use is unsaved editor buffers over a filesystem or Git
snapshot.

## 5. Retention and source recovery

Representations carry provenance:

- `Extracted`: deterministic frontend output, recoverable from source;
- `Derived`: synthesized from indexed facts, such as `Usage`;
- `Imported`: supplied by a consumer or external model.

Retention modes are provenance-aware:

- `Full` retains all text.
- `Report` may drop extracted embed-only text while retaining display and
  non-recoverable derived/imported text.
- `Minimal` may drop all recoverable extracted text.

For non-filesystem workspaces, pass a `SourceProviderCatalog` to explicit-space
embedding when dropped text must be recovered. Registering the exact snapshot
with `insert_snapshot` prevents recovery from drifting to a newer live workspace
while that snapshot remains available.

## 6. Embed one explicit space

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
transform is rejected. `embed_pending` is a convenience operation that creates
one `default/<channel>` space per present channel.

## 7. Load and search a storage-neutral index

SQLite exports the public `IndexSnapshot`; another backend can construct the
same type directly.

```rust
use codeindex::core::EmbeddingSpaceId;
use codeindex::query::WhereFilter;
use codeindex::search::SearchIndex;

let snapshot = db.snapshot(&[])?;
let index = SearchIndex::from_snapshot(snapshot)?;
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

Snapshot loading validates dimensions, duplicate ids and hashes, and finite
vector values rather than trusting external stores.

## 8. Fuse multiple spaces

Do not add cosine scores from different models. Embed each query with the
corresponding model, normalize according to that space, then fuse ranked lists
with weighted reciprocal rank through `SearchIndex::search_vectors_fused`.
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

The schema is pre-release. Databases from an incompatible epoch are rejected
with a delete-and-reindex message. This avoids accepting an old layout and
failing later with unrelated SQL errors.
