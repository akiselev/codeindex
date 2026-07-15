# Getting started

This guide covers the current library pipeline:

```text
source provider → entities/representations → embedding spaces → snapshot → search
```

Examples use the facade crate. Enable `fastembed` for local ONNX inference.

```toml
[dependencies]
anyhow = "1"
codeindex = { path = "../codeindex/crates/codeindex", features = ["fastembed"] }
```

## 1. Embed arbitrary text without storage or parsers

Depend directly on `codeindex-embedding` for the leanest path. Its default
features are empty, so enable `fastembed` here too:

```toml
[dependencies]
codeindex-embedding = { path = "../codeindex/crates/embedding", features = ["fastembed"] }
```

```rust
use codeindex_embedding::{
    EmbedRequest,
    config::EmbeddingConfig,
    embedder_from_config,
};

fn main() -> anyhow::Result<()> {
    let config = EmbeddingConfig {
        // Any HuggingFace repo (`hf:owner/name`), local dir (`dir:/path`), or
        // fastembed catalog name works; there is no implicit default model.
        model: "fastembed:BGESmallENV15".into(),
        ..Default::default()
    };
    let mut embedder = embedder_from_config(&config)?;
    let vectors = embedder.embed(&EmbedRequest::documents(
        &[
            "fn parse(input: &str) -> Result<Ast>",
            "def parse(input): return build_ast(input)",
        ],
        None,
    ))?;
    println!("{} dimensions", vectors[0].len());
    Ok(())
}
```

This compiles neither SQLite nor bundled language grammars.

## 2. Index ordinary filesystem projects

The common filesystem API remains `indexer::index`. It delegates to the same
provider-neutral pipeline used by custom sources.

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

The call returns only after one atomic publication of every selected project.
During extraction, queries continue to see the previous committed generation.
If the process is interrupted, rerunning the same configuration automatically
resumes compatible document checkpoints and refreshes changed files.

For explicit progress and cancellation, use the builder:

```rust
use codeindex::indexer::{
    CancellationToken, FileSystemSource, IndexOutcome, IndexRunBuilder,
    SourceProject,
};

let cancellation = CancellationToken::new();
let source = FileSystemSource::new("./src");
let projects = [SourceProject {
    label: "main".into(),
    provider: &source,
}];
let settings = options.settings();
let outcome = IndexRunBuilder::new(&db, &settings, &projects)
    .with_cancellation(cancellation.clone())
    .on_progress(&|event| eprintln!("{event:?}"))
    .run()?;

match outcome {
    IndexOutcome::Committed(report) => println!("generation {}", report.generation),
    IndexOutcome::Paused(status) => println!("resume run {}", status.run_id),
}
```

`ResumePolicy::Auto` and content verification for advisory revisions are the
defaults. `ResumePolicy::Run(id)`, `ResumePolicy::New`, and
`RevisionTrust::TrustAdvisory` are explicit lower-level controls.

## 3. Index a custom source provider

A provider does not need to emulate a filesystem. It exposes stable document
identities, logical paths, revisions, and UTF-8 content.

```rust
use codeindex::{indexer, sqlite};
use codeindex::indexer::{
    IndexSettings, MemorySource, RetentionMode, SourceProject,
};

let db = sqlite::open_in_memory()?;
let mut source = MemorySource::new("memory://workspace");
source.insert(
    "src/lib.rs",
    "fn answer() -> i32 { let value = 42; value }",
);

let settings = IndexSettings {
    enabled_languages: vec!["rust".into()],
    body_node_count_threshold: 1,
    max_body_chars: 10_000,
    retention: RetentionMode::Full,
};
let projects = [SourceProject {
    label: "main".into(),
    provider: &source,
}];
indexer::index_sources(&db, &settings, &projects, None)?;
```

Implement `SourceProvider` for database rows, Git objects, object-store keys,
archives, editor buffers, or generated code. Preserve the same document id
across a logical move when the provider can do so reliably.

## 4. Retention and provenance

Representations carry provenance:

- `Extracted`: deterministic frontend output, recoverable from source;
- `Derived`: synthesized from indexed facts, such as `Usage`;
- `Imported`: supplied by a consumer or external model.

Retention modes are provenance-aware:

- `Full` retains all text.
- `Report` may drop extracted embed-only text while retaining display and
  non-recoverable derived/imported text.
- `Minimal` may drop all recoverable extracted text.

For non-filesystem providers, pass a `SourceProviderCatalog` to explicit-space
embedding when dropped text must be recovered.

## 5. Embed one explicit space

An embedding space binds a channel to an exact model identity. Spaces are
independent: a code model can embed implementations while a text model embeds
documentation.

```rust
use codeindex::core::{EmbeddingSpaceIdentity, RepresentationKind};
use codeindex::embedding::{
    config::{EmbeddingConfig, EmbeddingRunConfig, SourceRecoveryConfig},
    embedder_from_config,
};
use codeindex::indexer;

let embedding_config = EmbeddingConfig {
    model: "fastembed:BGESmallENV15".into(),
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
    embedder.contract().clone(),
);
let stats = indexer::embed_space_pending(
    &db,
    embedder.as_mut(),
    &run_config,
    &space,
)?;
println!("embedded {} representations", stats.embedded);
```

Reusing the id `docs` with a different channel, model contract, or
document-side contract is rejected.

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

The query embedder must match the selected space's semantic model contract
(execution environment — device, versions, cache paths — is provenance and
never compared). Instruction-aware models accept an optional task describing
the retrieval intent; documents never re-embed when the task changes.

```rust
use codeindex::core::{EmbeddingSpaceId, EmbeddingTask};
use codeindex::query::WhereFilter;

let task = EmbeddingTask::new(
    "code-search",
    "Given a question about repository behavior, retrieve code that answers it",
);
let results = index.search_text(
    embedder.as_mut(),
    "parse command line flags",
    Some(&task), // or None for the model's default query prompt
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

## 9. Command-line atomic indexing

The CLI is a thin consumer of `IndexRunBuilder`:

```sh
cargo run -p codeindex-cli -- index \
  --db ./codeindex.db \
  --project main=./src \
  --language rust

cargo run -p codeindex-cli -- status --db ./codeindex.db
```

Use `--resume RUN_ID` to select a compatible run, `--restart` to supersede
overlapping unfinished work, and the `abandon` or `supersede` subcommands for
explicit lifecycle control. `--json` emits versioned JSON progress and result
envelopes. SIGINT and SIGTERM take the graceful cancellation path and SIGINT
returns exit code 130.

## 10. Metadata filtering

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

The schema is pre-release (`codeindex_sqlite::SCHEMA_VERSION` is the current
epoch). Databases from an incompatible epoch are rejected with a
delete-and-reindex message. This avoids accepting an old layout and failing
later with unrelated SQL errors.
