#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use codeindex_indexer::{
    CancellationToken, FileSystemSource, IndexOutcome, IndexProgress, IndexRunBuilder,
    IndexSettings, RefreshPolicy, ResumePolicy, RetentionMode, RevisionTrust, SourceProject,
};
use codeindex_sqlite::{Db, open_or_create};
use serde::Serialize;

#[derive(Parser)]
#[command(name = "codeindex", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Atomically index one or more filesystem projects.
    Index(IndexArgs),
    /// Inspect durable indexing runs.
    Status(StatusArgs),
    /// Permanently abandon an unfinished indexing run.
    Abandon(RunArgs),
    /// Mark an unfinished indexing run superseded without starting a replacement.
    Supersede(RunArgs),
    /// Inspect and resolve embedding model references.
    #[command(subcommand)]
    Models(ModelsCommand),
    /// Embed pending representations into an explicit embedding space.
    Embed(EmbedArgs),
    /// Search an embedding space with an optional task instruction.
    Search(SearchArgs),
    /// Enrich a published project through a language server: derived
    /// `typed_signature` representations plus exact `calls` relations.
    LspEnrich(LspEnrichArgs),
}

#[derive(Args)]
struct LspEnrichArgs {
    #[command(flatten)]
    database: DatabaseArgs,
    /// Indexed project label to enrich.
    #[arg(long)]
    project: String,
    /// Root directory the language server runs against (the indexed tree).
    #[arg(long)]
    root: PathBuf,
    /// Language server executable.
    #[arg(long, default_value = "rust-analyzer")]
    server: String,
    /// Extra arguments for the language server.
    #[arg(long = "server-arg")]
    server_args: Vec<String>,
    /// Language id the server covers.
    #[arg(long, default_value = "rust")]
    language: String,
}

#[derive(Args)]
struct EmbedArgs {
    #[command(flatten)]
    database: DatabaseArgs,
    /// Model reference: `hf:owner/name[@rev]`, `dir:/path`, `fastembed:Name`.
    #[arg(long)]
    model: String,
    /// Space as `ID=CHANNEL`, e.g. `code=implementation` or
    /// `docs=documentation`. Repeat to project several spaces in one run.
    #[arg(long = "space", required = true, value_parser = parse_space)]
    spaces: Vec<SpaceArgument>,
    /// Document-side prompt prefixed to every representation at embed time
    /// (part of the space's immutable contract).
    #[arg(long)]
    document_prompt: Option<String>,
    /// Matryoshka projection: store only this many leading dimensions.
    #[arg(long)]
    output_dimensions: Option<usize>,
    /// Model cache root override.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    /// Execution provider (cpu, cuda, metal, ...).
    #[arg(long, default_value = "cpu")]
    execution_provider: String,
}

#[derive(Args)]
struct SearchArgs {
    #[command(flatten)]
    database: DatabaseArgs,
    /// Query text.
    query: String,
    /// Embedding space to search.
    #[arg(long)]
    space: String,
    /// Named task preset (see `--list-tasks`) rendered as the query
    /// instruction on instruction-aware models.
    #[arg(long, conflicts_with = "instruction")]
    task: Option<String>,
    /// Raw task instruction (instruction-aware models only).
    #[arg(long)]
    instruction: Option<String>,
    /// Metadata filter, e.g. `language=rust kind=function path=src/**`.
    #[arg(long = "where")]
    filter: Option<String>,
    #[arg(long, default_value_t = 10)]
    limit: usize,
    /// Model cache root override.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    /// Execution provider (cpu, cuda, metal, ...).
    #[arg(long, default_value = "cpu")]
    execution_provider: String,
    /// List the built-in task presets and exit.
    #[arg(long)]
    list_tasks: bool,
    /// Include test entities. Excluded by default — trivial test chunks
    /// dominate degraded rankings; an explicit `--where tests=` clause wins.
    #[arg(long)]
    include_tests: bool,
    /// Retrieval mode. `hybrid` fuses the dense space with lexical BM25
    /// (SQLite FTS5) via weighted reciprocal rank.
    #[arg(long, value_enum, default_value_t = RetrievalArgument::Hybrid)]
    retrieval: RetrievalArgument,
    /// Rerank the top candidates with a cross-encoder (requires the
    /// `candle` feature; downloads the reranker on first use).
    #[arg(long)]
    rerank: bool,
    /// Reranker model reference.
    #[arg(long, default_value = "hf:Qwen/Qwen3-Reranker-0.6B")]
    rerank_model: String,
    /// How many fused candidates the reranker judges.
    #[arg(long, default_value_t = 12)]
    rerank_candidates: usize,
    /// Fuse a compressed variant of the query (function words stripped) as
    /// an extra dense list. `auto` compresses only paragraph-length queries,
    /// whose salient terms otherwise drown in narrative phrasing.
    #[arg(long, value_enum, default_value_t = CompressArgument::Auto)]
    compress: CompressArgument,
    /// Disable relation-graph expansion. By default, when analyzer relations
    /// are stored (`lsp-enrich`), 1-hop callers/callees of the top fused
    /// seeds join the fusion as a low-weight `graph` list.
    #[arg(long)]
    no_graph: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum CompressArgument {
    Auto,
    Off,
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum RetrievalArgument {
    Hybrid,
    Dense,
    Lexical,
}

#[derive(Debug, Clone)]
struct SpaceArgument {
    id: String,
    channel: String,
}

fn parse_space(value: &str) -> Result<SpaceArgument, String> {
    let (id, channel) = value
        .split_once('=')
        .ok_or_else(|| "expected ID=CHANNEL, e.g. code=implementation".to_string())?;
    if id.is_empty() || channel.is_empty() {
        return Err("space id and channel must both be non-empty".into());
    }
    Ok(SpaceArgument {
        id: id.into(),
        channel: channel.into(),
    })
}

/// Built-in retrieval intents (FEEDBACK.md §1): stable ids agents can pass as
/// `--task` instead of hand-writing instructions.
const TASK_PRESETS: &[(&str, &str)] = &[
    (
        "code-search",
        "Given a code search query, retrieve relevant code implementations",
    ),
    (
        "locate-edit-targets",
        "Given a software change request, retrieve code regions likely to require editing",
    ),
    (
        "explain-behavior",
        "Given a question about repository behavior, retrieve code that provides evidence for \
         the answer",
    ),
    (
        "find-analogues",
        "Given a code fragment, retrieve functionally equivalent implementations",
    ),
    (
        "diagnose-failure",
        "Given a failure report, retrieve implementation, tests, configuration, and \
         error-handling paths relevant to diagnosing it",
    ),
];

#[derive(Subcommand)]
enum ModelsCommand {
    /// Resolve a model reference into its semantic contract without loading
    /// any weights.
    Resolve(ModelResolveArgs),
}

#[derive(Args)]
struct ModelResolveArgs {
    /// Model reference: `hf:owner/name[@rev]`, `owner/name`, or `dir:/path`.
    reference: String,
    /// Model cache root (defaults to the codeindex cache).
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    /// Emit versioned newline-delimited JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct DatabaseArgs {
    /// SQLite database path.
    #[arg(long, default_value = "codeindex.db")]
    db: PathBuf,
    /// Emit versioned newline-delimited JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct IndexArgs {
    #[command(flatten)]
    database: DatabaseArgs,
    /// Project in LABEL=PATH form. Repeat for an atomic multi-project run.
    #[arg(long = "project", required = true, value_parser = parse_project)]
    projects: Vec<ProjectArgument>,
    /// Language id to enable. Defaults to every bundled language.
    #[arg(long = "language")]
    languages: Vec<String>,
    /// Glob excluded from every filesystem project. Repeat as needed.
    #[arg(long = "exclude")]
    excludes: Vec<String>,
    #[arg(long, default_value_t = 10)]
    body_node_count_threshold: usize,
    #[arg(long, default_value_t = 10_000)]
    max_body_chars: usize,
    #[arg(long, value_enum, default_value_t = RetentionArgument::Full)]
    retention: RetentionArgument,
    /// Resume this exact compatible run id.
    #[arg(long, conflicts_with = "restart")]
    resume: Option<i64>,
    /// Supersede overlapping unfinished work and start a fresh run.
    #[arg(long)]
    restart: bool,
    /// Trust opaque revisions from advisory providers instead of hashing bytes.
    #[arg(long)]
    trust_advisory_revisions: bool,
}

#[derive(Args)]
struct StatusArgs {
    #[command(flatten)]
    database: DatabaseArgs,
    /// Inspect one run rather than the newest runs.
    #[arg(long)]
    run: Option<i64>,
    #[arg(long, default_value_t = 20)]
    limit: usize,
}

#[derive(Args)]
struct RunArgs {
    #[command(flatten)]
    database: DatabaseArgs,
    #[arg(long)]
    run: i64,
}

#[derive(Debug, Clone)]
struct ProjectArgument {
    label: String,
    path: PathBuf,
}

fn parse_project(value: &str) -> Result<ProjectArgument, String> {
    let (label, path) = value
        .split_once('=')
        .ok_or_else(|| "expected LABEL=PATH".to_string())?;
    if label.is_empty() || path.is_empty() {
        return Err("project label and path must both be non-empty".into());
    }
    Ok(ProjectArgument {
        label: label.into(),
        path: path.into(),
    })
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum RetentionArgument {
    Full,
    Report,
    Minimal,
}

impl From<RetentionArgument> for RetentionMode {
    fn from(value: RetentionArgument) -> Self {
        match value {
            RetentionArgument::Full => Self::Full,
            RetentionArgument::Report => Self::Report,
            RetentionArgument::Minimal => Self::Minimal,
        }
    }
}

/// Version of the newline-delimited JSON envelope this binary emits.
const PROTOCOL_VERSION: u32 = 1;

#[derive(Serialize)]
struct Envelope<'a, T> {
    version: u32,
    event: &'a str,
    data: T,
}

fn envelope_line<T: Serialize>(event: &str, data: T) -> serde_json::Result<String> {
    serde_json::to_string(&Envelope {
        version: PROTOCOL_VERSION,
        event,
        data,
    })
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    match cli.command {
        Command::Index(arguments) => index_command(arguments),
        Command::Status(arguments) => {
            let db = open_or_create(&arguments.database.db)?;
            if let Some(run_id) = arguments.run {
                print_value(arguments.database.json, "status", &db.run_status(run_id)?)?;
            } else {
                print_value(
                    arguments.database.json,
                    "status",
                    &db.list_run_statuses(arguments.limit)?,
                )?;
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Abandon(arguments) => update_run(arguments, |db, run| db.abandon_run(run)),
        Command::Supersede(arguments) => update_run(arguments, |db, run| db.supersede_run(run)),
        Command::Models(ModelsCommand::Resolve(arguments)) => models_resolve(arguments),
        Command::Embed(arguments) => embed_command(arguments),
        Command::Search(arguments) => search_command(arguments),
        Command::LspEnrich(arguments) => lsp_enrich_command(arguments),
    }
}

fn lsp_enrich_command(arguments: LspEnrichArgs) -> Result<ExitCode> {
    let db = open_or_create(&arguments.database.db)?;
    let report = codeindex_lsp::enrich_project(
        &db,
        &arguments.project,
        &arguments.root,
        &codeindex_lsp::LspServer {
            language_id: arguments.language.clone(),
            command: arguments.server.clone(),
            args: arguments.server_args.clone(),
        },
    )?;
    print_value(
        arguments.database.json,
        "lsp-enrich",
        &LspEnrichOut {
            files_visited: report.files_visited,
            units_visited: report.units_visited,
            typed_signatures: report.typed_signatures,
            relations: report.relations,
        },
    )?;
    Ok(ExitCode::SUCCESS)
}

#[derive(Debug, Serialize)]
struct LspEnrichOut {
    files_visited: usize,
    units_visited: usize,
    typed_signatures: usize,
    relations: usize,
}

fn embed_command(arguments: EmbedArgs) -> Result<ExitCode> {
    use codeindex_core::{DocumentSideContract, EmbeddingSpaceId, EmbeddingSpaceIdentity};
    use codeindex_embedding::config::{EmbeddingConfig, EmbeddingRunConfig, SourceRecoveryConfig};

    let db = open_or_create(&arguments.database.db)?;
    let config = EmbeddingConfig {
        model: arguments.model.clone(),
        cache_dir: arguments.cache_dir.clone(),
        execution_provider: arguments.execution_provider.clone(),
        ..EmbeddingConfig::default()
    };
    let mut embedder = codeindex_embedding::embedder_from_config(&config)?;
    let run_config = EmbeddingRunConfig {
        embedding: config,
        source_recovery: SourceRecoveryConfig {
            body_node_count_threshold: 10,
        },
    };
    let document_side = DocumentSideContract {
        prompt: arguments.document_prompt.clone(),
        output_dimensions: arguments.output_dimensions,
    };
    let json = arguments.database.json;
    let mut totals = Vec::new();
    for space in &arguments.spaces {
        let identity = EmbeddingSpaceIdentity::new(
            EmbeddingSpaceId::new(space.id.clone()),
            space.channel.as_str().into(),
            embedder.contract().clone(),
        )
        .with_document_side(document_side.clone());
        let stats = codeindex_indexer::embed_space_pending_with_progress(
            &db,
            embedder.as_mut(),
            &run_config,
            &identity,
            None,
            &mut |progress| {
                if json
                    && let Ok(line) = envelope_line(
                        "progress",
                        &EmbedProgressOut {
                            space_id: progress.space_id.to_string(),
                            embedded: progress.embedded,
                            pending_total: progress.pending_total,
                        },
                    )
                {
                    println!("{line}");
                }
            },
        )?;
        totals.push(EmbedResultOut {
            space_id: space.id.clone(),
            channel: space.channel.clone(),
            embedded: stats.embedded,
            unresolved: stats.unresolved,
            batches: stats.batches,
        });
    }
    print_value(json, "embed", &totals)?;
    Ok(ExitCode::SUCCESS)
}

#[derive(Debug, Serialize)]
struct EmbedProgressOut {
    space_id: String,
    embedded: usize,
    pending_total: usize,
}

#[derive(Debug, Serialize)]
struct EmbedResultOut {
    space_id: String,
    channel: String,
    embedded: usize,
    unresolved: usize,
    batches: usize,
}

fn search_command(arguments: SearchArgs) -> Result<ExitCode> {
    use codeindex_core::{EmbeddingSpaceId, EmbeddingTask};
    use codeindex_embedding::config::EmbeddingConfig;
    use codeindex_query::{RankedList, TestsPolicy, WhereFilter, reciprocal_rank_fusion};
    use codeindex_search::SearchIndex;
    use std::collections::HashMap;

    if arguments.list_tasks {
        for (id, instruction) in TASK_PRESETS {
            println!("{id}: {instruction}");
        }
        return Ok(ExitCode::SUCCESS);
    }

    let db = open_or_create(&arguments.database.db)?;
    let space_id = EmbeddingSpaceId::new(arguments.space.clone());
    let space = db
        .get_space(&space_id)?
        .with_context(|| format!("embedding space {:?} is not stored", arguments.space))?;

    // The stored semantic contract's model field is itself a resolvable
    // reference, so the matching query embedder reconstructs automatically.
    let config = EmbeddingConfig {
        model: space.identity.model.model.clone(),
        cache_dir: arguments.cache_dir.clone(),
        execution_provider: arguments.execution_provider.clone(),
        ..EmbeddingConfig::default()
    };

    let task = match (&arguments.task, &arguments.instruction) {
        (Some(preset), _) => Some(EmbeddingTask::new(
            preset.clone(),
            TASK_PRESETS
                .iter()
                .find(|(id, _)| id == preset)
                .map(|(_, instruction)| (*instruction).to_string())
                .with_context(|| {
                    format!(
                        "unknown task preset {preset:?}; available: {}",
                        TASK_PRESETS
                            .iter()
                            .map(|(id, _)| *id)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?,
        )),
        (None, Some(instruction)) => Some(EmbeddingTask::new("custom", instruction.clone())),
        (None, None) => None,
    };

    let mut filter = WhereFilter::parse(arguments.filter.as_deref())?;
    if filter.tests_policy().is_none() && !arguments.include_tests {
        filter.set_tests_policy(TestsPolicy::Exclude);
    }
    let index = SearchIndex::from_snapshot(db.snapshot(&[])?)?;

    // First stage: dense and/or lexical candidate lists over one filter.
    let candidates = (arguments.limit * 5).max(50);
    let mut embedder = if arguments.retrieval == RetrievalArgument::Lexical {
        None
    } else {
        Some(codeindex_embedding::embedder_from_config(&config)?)
    };
    let dense_hits = |embedder: &mut Box<dyn codeindex_embedding::embed::EmbeddingBackend>,
                      text: &str|
     -> Result<Vec<usize>> {
        Ok(index
            .search_text(
                embedder.as_mut(),
                text,
                task.as_ref(),
                &space_id,
                &filter,
                candidates,
            )?
            .hits
            .iter()
            .map(|hit| hit.index)
            .collect())
    };
    let dense_indices: Vec<usize> = match embedder.as_mut() {
        Some(embedder) => dense_hits(embedder, &arguments.query)?,
        None => Vec::new(),
    };
    // A compressed variant recovers targets whose salient terms drown in
    // narrative phrasing (evals S13); fused alongside, never instead.
    let compressed_query = match arguments.compress {
        CompressArgument::Off => None,
        CompressArgument::Auto => codeindex_query::compress_query(&arguments.query, 24, 25),
        CompressArgument::Always => codeindex_query::compress_query(&arguments.query, 24, 0),
    };
    let compressed_indices: Vec<usize> = match (embedder.as_mut(), compressed_query.as_deref()) {
        (Some(embedder), Some(compressed)) => dense_hits(embedder, compressed)?,
        _ => Vec::new(),
    };
    let lexical_indices: Vec<usize> = if arguments.retrieval == RetrievalArgument::Dense {
        Vec::new()
    } else {
        let by_version: HashMap<(&str, &str), usize> = index
            .units
            .iter()
            .enumerate()
            .map(|(position, unit)| {
                (
                    (unit.project_label.as_str(), unit.entity_version_id.as_str()),
                    position,
                )
            })
            .collect();
        db.lexical_search(&arguments.query, candidates * 2)?
            .iter()
            .filter_map(|hit| {
                by_version
                    .get(&(hit.project_label.as_str(), hit.entity_version_id.as_str()))
                    .copied()
            })
            .filter(|position| filter.matches(&index.units[*position]))
            .take(candidates)
            .collect()
    };

    let mut lists = Vec::new();
    if !dense_indices.is_empty() {
        lists.push(RankedList {
            source: "dense".into(),
            weight: 1.0,
            indices: dense_indices,
        });
    }
    if !compressed_indices.is_empty() {
        lists.push(RankedList {
            source: "dense-compressed".into(),
            weight: 0.7,
            indices: compressed_indices,
        });
    }
    if !lexical_indices.is_empty() {
        lists.push(RankedList {
            source: "lexical".into(),
            weight: 1.0,
            indices: lexical_indices,
        });
    }
    let mut fused = reciprocal_rank_fusion(&lists, 60);

    // Relation-graph expansion: 1-hop callers/callees of the top seeds join
    // as a low-weight list and everything re-fuses. This reaches targets
    // whose own text never matches the query.
    if !arguments.no_graph && !index.relations.is_empty() && !fused.is_empty() {
        let seeds: Vec<usize> = fused.iter().take(10).map(|hit| hit.index).collect();
        let expanded: Vec<usize> = index
            .expand_by_relations(&seeds, candidates)
            .into_iter()
            .filter(|position| filter.matches(&index.units[*position]))
            .collect();
        if !expanded.is_empty() {
            lists.push(RankedList {
                source: "graph".into(),
                weight: 0.5,
                indices: expanded,
            });
            fused = reciprocal_rank_fusion(&lists, 60);
        }
    }

    // Second stage: cross-encoder judgement of the fused head.
    #[cfg_attr(not(feature = "candle"), allow(unused_mut))]
    let mut rerank_scores: HashMap<usize, f32> = HashMap::new();
    if arguments.rerank {
        #[cfg(not(feature = "candle"))]
        anyhow::bail!("--rerank needs the `candle` feature; rebuild with --features candle");
        #[cfg(feature = "candle")]
        {
            use codeindex_embedding::rerank::{Qwen3Reranker, Reranker as _};
            let head = arguments.rerank_candidates.min(fused.len());
            let instruction = task
                .as_ref()
                .map(|task| task.instruction.clone())
                .unwrap_or_else(|| {
                    "Given a code search query, retrieve relevant code implementations".to_string()
                });
            let mut judged: Vec<(usize, &str)> = Vec::new();
            for hit in fused.iter().take(head) {
                let unit = &index.units[hit.index];
                let text = unit
                    .representations
                    .iter()
                    .find(|repr| repr.kind == codeindex_core::RepresentationKind::Implementation)
                    .and_then(|repr| repr.content.as_deref())
                    .or(unit.display_source.as_deref());
                if let Some(text) = text {
                    judged.push((hit.index, text));
                }
            }
            if !judged.is_empty() {
                let mut reranker = Qwen3Reranker::from_reference(&arguments.rerank_model, &config)?;
                let documents: Vec<&str> = judged.iter().map(|(_, text)| *text).collect();
                let scores = reranker.rerank(&instruction, &arguments.query, &documents)?;
                for ((position, _), score) in judged.iter().zip(scores) {
                    rerank_scores.insert(*position, score);
                }
                // Judged candidates re-sort by relevance; unjudged keep their
                // fused order below them.
                fused.sort_by(|left, right| {
                    match (
                        rerank_scores.get(&left.index),
                        rerank_scores.get(&right.index),
                    ) {
                        (Some(a), Some(b)) => b.total_cmp(a),
                        (Some(_), None) => std::cmp::Ordering::Less,
                        (None, Some(_)) => std::cmp::Ordering::Greater,
                        (None, None) => right.score.total_cmp(&left.score),
                    }
                });
            }
        }
    }

    let matched = fused.len();
    let hits: Vec<SearchHitOut> = fused
        .iter()
        .take(arguments.limit)
        .map(|hit| {
            let unit = &index.units[hit.index];
            SearchHitOut {
                selector: codeindex_query::unit_id(unit),
                score: rerank_scores.get(&hit.index).copied().unwrap_or(hit.score),
                rerank_score: rerank_scores.get(&hit.index).copied(),
                sources: hit
                    .contributions
                    .iter()
                    .map(|(source, rank)| format!("{source}#{rank}"))
                    .collect(),
                project: unit.project_label.clone(),
                path: unit.relative_path.clone(),
                lines: [unit.start_line, unit.end_line],
                language: unit.language_id.clone(),
                kind: unit.kind.clone(),
                name: unit.name.clone(),
                scope: unit.scope.clone(),
            }
        })
        .collect();
    print_value(
        arguments.database.json,
        "search",
        &SearchResultsOut {
            query: arguments.query.clone(),
            compressed_query,
            space: arguments.space.clone(),
            task: task.clone(),
            matched,
            hits,
        },
    )?;
    Ok(ExitCode::SUCCESS)
}

#[derive(Debug, Serialize)]
struct SearchResultsOut {
    query: String,
    /// The compressed query variant fused in, when one was derived.
    #[serde(skip_serializing_if = "Option::is_none")]
    compressed_query: Option<String>,
    space: String,
    /// The exact task rendered into the query, for reproducibility.
    task: Option<codeindex_core::EmbeddingTask>,
    matched: usize,
    hits: Vec<SearchHitOut>,
}

#[derive(Debug, Serialize)]
struct SearchHitOut {
    selector: String,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    rerank_score: Option<f32>,
    /// `source#rank` contributions from each first-stage list.
    sources: Vec<String>,
    project: String,
    path: String,
    lines: [usize; 2],
    language: String,
    kind: String,
    name: String,
    scope: Option<String>,
}

fn models_resolve(arguments: ModelResolveArgs) -> Result<ExitCode> {
    use codeindex_embedding::resolve::{
        ModelRef, default_model_root, fetcher_for, model_cache_dir, resolve_model,
    };

    #[derive(Debug, Serialize)]
    struct WeightsOutput {
        file: String,
        declared_bytes: u64,
        dtypes: Vec<String>,
        tensors: usize,
    }

    #[derive(Debug, Serialize)]
    struct ResolvedOutput {
        contract: codeindex_core::ModelContract,
        local_dir: String,
        weight_files: Vec<String>,
        /// Safetensors preflight (header-only range read): size and dtypes
        /// known before any weight download.
        weights: Option<WeightsOutput>,
    }

    let reference = ModelRef::parse(&arguments.reference)?;
    let root = arguments.cache_dir.unwrap_or_else(default_model_root);
    let cache_dir = model_cache_dir(&root, &reference);
    let fetcher = fetcher_for(&reference)?;
    let resolved = resolve_model(&reference, fetcher.as_ref(), &cache_dir)?;
    let weights = resolved
        .weight_files
        .iter()
        .find(|file| file.ends_with(".safetensors"))
        .and_then(|file| {
            let preflight =
                codeindex_embedding::resolve::preflight_safetensors(fetcher.as_ref(), file)
                    .ok()
                    .flatten()?;
            Some(WeightsOutput {
                file: file.clone(),
                declared_bytes: preflight.declared_size,
                dtypes: preflight.dtypes,
                tensors: preflight.tensor_names.len(),
            })
        });
    print_value(
        arguments.json,
        "model",
        &ResolvedOutput {
            contract: resolved.contract,
            local_dir: resolved.local_dir.display().to_string(),
            weight_files: resolved.weight_files,
            weights,
        },
    )?;
    Ok(ExitCode::SUCCESS)
}

fn update_run(
    arguments: RunArgs,
    operation: impl FnOnce(&Db, i64) -> Result<codeindex_sqlite::index_runs::IndexRunStatus>,
) -> Result<ExitCode> {
    let db = open_or_create(&arguments.database.db)?;
    let status = operation(&db, arguments.run)?;
    print_value(arguments.database.json, "status", &status)?;
    Ok(ExitCode::SUCCESS)
}

fn index_command(arguments: IndexArgs) -> Result<ExitCode> {
    let db = open_or_create(&arguments.database.db)?;
    let sources: Vec<_> = arguments
        .projects
        .iter()
        .map(|project| {
            (
                project.label.clone(),
                FileSystemSource::new(&project.path).with_excludes(arguments.excludes.clone()),
            )
        })
        .collect();
    let projects: Vec<_> = sources
        .iter()
        .map(|(label, source)| SourceProject {
            label: label.clone(),
            provider: source,
        })
        .collect();
    let settings = IndexSettings {
        enabled_languages: if arguments.languages.is_empty() {
            codeindex_tree_sitter::BUNDLED_LANGUAGE_IDS
                .iter()
                .map(|language| (*language).to_string())
                .collect()
        } else {
            arguments.languages.clone()
        },
        body_node_count_threshold: arguments.body_node_count_threshold,
        max_body_chars: arguments.max_body_chars,
        retention: arguments.retention.into(),
    };
    let cancellation = CancellationToken::new();
    let signal_cancellation = cancellation.clone();
    ctrlc::set_handler(move || signal_cancellation.cancel())
        .context("installing SIGINT/SIGTERM handler")?;
    let json = arguments.database.json;
    let progress = |progress: IndexProgress| {
        if json {
            if let Ok(line) = envelope_line("progress", &progress) {
                println!("{line}");
            }
        } else if let (Some(project), Some(document)) =
            (&progress.project_label, &progress.source_document_id)
        {
            eprintln!(
                "run {}: {project}/{document} ({}/{})",
                progress.run_id, progress.ready_documents, progress.total_documents
            );
        }
    };
    let resume_policy = if let Some(run_id) = arguments.resume {
        ResumePolicy::Run(run_id)
    } else if arguments.restart {
        ResumePolicy::New
    } else {
        ResumePolicy::Auto
    };
    let outcome = IndexRunBuilder::new(&db, &settings, &projects)
        .resume_policy(resume_policy)
        .refresh_policy(RefreshPolicy::default())
        .revision_trust(if arguments.trust_advisory_revisions {
            RevisionTrust::TrustAdvisory
        } else {
            RevisionTrust::VerifyContent
        })
        .with_cancellation(cancellation)
        .on_progress(&progress)
        .run()?;
    print_value(json, "result", &outcome)?;
    match outcome {
        IndexOutcome::Committed(_) => Ok(ExitCode::SUCCESS),
        IndexOutcome::Paused(status)
            if status.pause_reason.as_deref() == Some("user_interrupt") =>
        {
            Ok(ExitCode::from(130))
        }
        IndexOutcome::Paused(status) => {
            if !json {
                eprintln!(
                    "index run {} paused: {}",
                    status.run_id,
                    status.pause_reason.as_deref().unwrap_or("unspecified")
                );
            }
            Ok(ExitCode::from(2))
        }
    }
}

fn print_value<T: Serialize + std::fmt::Debug>(json: bool, event: &str, value: &T) -> Result<()> {
    if json {
        println!("{}", envelope_line(event, value)?);
    } else {
        println!("{value:#?}");
    }
    Ok(())
}
