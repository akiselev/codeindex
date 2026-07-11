# Getting started

This walks the full pipeline: index a project, embed its bodies, and run a
semantic query. All examples assume the `fastembed` feature is enabled so a real
model can run.

```toml
[dependencies]
codeindex = { path = "../codeindex/crates/codeindex", features = ["fastembed"] }
anyhow = "1"
```

## 1. Embed text (no store, no parser)

The leanest use is the `Embedder` itself. This path compiles neither SQLite nor
the grammars — depend only on `codeindex-embedding`.

```rust
use codeindex_embedding::{config::EmbeddingConfig, embedder_from_config};

fn main() -> anyhow::Result<()> {
    // BGESmallENV15 is small and downloads quickly; the default model is the
    // larger CodeRankEmbed (2048-token context, ~550 MB, fetched on first use).
    let cfg = EmbeddingConfig { model: "BGESmallENV15".into(), ..Default::default() };
    let mut embedder = embedder_from_config(&cfg)?;

    let vectors = embedder.embed(&[
        "fn parse(input: &str) -> Result<Ast>".to_string(),
        "def parse(input): return build_ast(input)".to_string(),
    ])?;

    println!("{} dims", vectors[0].len());
    Ok(())
}
```

## 2. Index a project

Indexing scans a root, extracts every unit above `body_node_count_threshold`,
and writes into SQLite. It is incremental — unchanged files are skipped on
re-runs.

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
        .map(|s| s.to_string())
        .collect(),
    body_node_count_threshold: 10,
    max_body_chars: 10_000,
    retention: RetentionMode::Full, // Full | Report | Minimal
    };

let stats = indexer::index(&db, &options, None)?;
for project in &stats {
    println!("{}: {} units", project.label, project.total_units);
}
```

`RetentionMode` trades storage for the ability to re-derive text:

- `Full` stores display source and embedding text.
- `Report` drops embedding text (recovered from source when embedding).
- `Minimal` stores only hashes and ranges; reports re-read source files.

## 3. Embed the indexed corpus

`embed_pending` embeds every distinct un-embedded body under the current model
identity. It is resumable: re-run it after indexing more code and it only
processes what is new.

```rust
use codeindex::indexer;
use codeindex::embedding::{
    config::{EmbeddingConfig, EmbeddingRunConfig, SourceRecoveryConfig},
    embedder_from_config,
};

let embedding = EmbeddingConfig { model: "BGESmallENV15".into(), ..Default::default() };
let run = EmbeddingRunConfig {
    embedding: embedding.clone(),
    // Only consulted under Report/Minimal retention, to re-derive dropped text.
    source_recovery: SourceRecoveryConfig { body_node_count_threshold: 10 },
};

let mut embedder = embedder_from_config(&embedding)?;
let stats = indexer::embed_pending(&db, embedder.as_mut(), &run)?;
println!("embedded {} of {} pending bodies", stats.embedded, stats.pending_total);
```

The database binds to the first model identity it sees; embedding again with a
different backend, model, dimension count, or normalization setting is rejected
(delete the DB to re-index under a new model).

## 4. Semantic search

Ranking is a pure operation over stored vectors. Embed the query with the *same*
model, then score it against every stored embedding.

```rust
use codeindex::{embedding, indexer, query};

let model_id = indexer::find_or_create_model_id(&db, embedder.identity())?;
let stored = db.all_embeddings(model_id)?; // Vec<(body_hash, Vec<f32>)>

let mut q = embedder.embed(&["retry an HTTP request with backoff".to_string()])?.remove(0);
embedding::normalize_in_place(&mut q);

let ranked = query::rank_candidates(
    &q,
    stored.iter().enumerate().map(|(i, (_, v))| (i, v.as_slice())),
    0.0, // score threshold
);

for scored in ranked.iter().take(5) {
    let (body_hash, _) = &stored[scored.index];
    println!("{:.4}  {}", scored.score, body_hash);
}
```

`rank_candidates` returns `ScoredIndex { index, score }` sorted by descending
score with deterministic tie-breaking. Resolving a `body_hash` back to a unit's
file, line range, name, and scope is a join the application layer performs over
`code_units` — decombine's `AnalysisContext` (in the `decombine` repo,
`src/query/mod.rs` and `src/analyze/context.rs`) is the reference implementation,
including model-identity verification via `query::identity_diff` and the
`unit:<id>` selectors from `query::unit_id`.

## Filtering

`codeindex-query::WhereFilter` parses the same `key=value` expression the CLI
uses and matches any `UnitView`:

```rust
use codeindex::query::WhereFilter;

let filter = WhereFilter::parse(Some("language=rust kind=function path=src/** min_nodes=20"))?;
// filter.matches(&unit)  where unit: impl UnitView
```

Supported keys: `project`, `language`, `kind` (exact); `name`, `scope`, `path`
(glob); `min_nodes` (integer).
