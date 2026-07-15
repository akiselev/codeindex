#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
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
    /// Register a project root (creates .codeindex.toml if missing) and
    /// start background indexing through the daemon.
    Add(AddArgs),
    /// Unregister a project (keeps .codeindex.toml; --purge deletes the db).
    Remove(RemoveArgs),
    /// Registered projects with index/job state.
    List(ProjectListArgs),
    /// Daemon and per-project status.
    Status(DaemonStatusArgs),
    /// Re-run indexing (and space embedding) for a registered project.
    Reindex(ReindexArgs),
    /// Control the background daemon.
    #[command(subcommand)]
    Daemon(DaemonCommand),
    /// Atomically index one or more filesystem projects.
    Index(IndexArgs),
    /// Inspect durable indexing runs.
    Runs(StatusArgs),
    /// Permanently abandon an unfinished indexing run.
    Abandon(RunArgs),
    /// Mark an unfinished indexing run superseded without starting a replacement.
    Supersede(RunArgs),
    /// Inspect and resolve embedding model references.
    #[command(subcommand)]
    Models(ModelsCommand),
    /// Embed pending representations into an explicit embedding space.
    Embed(EmbedArgs),
    /// Search a project (hybrid retrieval, optional task instruction).
    #[command(visible_alias = "query")]
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
    /// Explicit SQLite database path; bypasses project discovery and the
    /// daemon entirely.
    #[arg(long)]
    db: Option<PathBuf>,
    /// Emit versioned newline-delimited JSON.
    #[arg(long)]
    json: bool,
    /// Registered project label or path (default: discovered from the
    /// working directory by walking up to .codeindex.toml).
    #[arg(long, conflicts_with = "db")]
    project: Option<String>,
    /// Run the search in this process with a freshly loaded model instead
    /// of routing through the daemon.
    #[arg(long)]
    no_daemon: bool,
    /// Query text.
    query: String,
    /// Embedding space to search (default: the project's default_space or
    /// its single stored space; lexical-only when none exists).
    #[arg(long)]
    space: Option<String>,
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

use codeindex_query::TASK_PRESETS;

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
struct AddArgs {
    /// Directory to register; walking up locates an enclosing project root.
    path: Option<PathBuf>,
    /// Index in this process instead of the daemon.
    #[arg(long)]
    no_daemon: bool,
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct RemoveArgs {
    /// Project label or root path.
    needle: String,
    /// Also delete the project database (only when it lives in the managed
    /// data directory).
    #[arg(long)]
    purge: bool,
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct ProjectListArgs {
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct DaemonStatusArgs {
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct ReindexArgs {
    /// Project label or root path; defaults to the project containing the
    /// working directory.
    needle: Option<String>,
    #[arg(long)]
    json: bool,
}

#[derive(Subcommand)]
enum DaemonCommand {
    /// Start (or attach to) the daemon and report its identity.
    Start,
    /// Request graceful shutdown.
    Stop,
    /// Lifecycle state as seen by the daemonkit runtime.
    Status,
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
    // A daemonkit bootstrap child never parses CLI arguments: the private
    // channel in the environment decides, not argv (the `__daemon` argv
    // marker exists only for `ps` readability).
    match codeindex_daemon::bootstrap_entry() {
        Ok(true) => return ExitCode::SUCCESS,
        Ok(false) => {}
        Err(error) => {
            eprintln!("daemon error: {error:#}");
            return ExitCode::from(1);
        }
    }
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
        Command::Add(arguments) => add_command(arguments),
        Command::Remove(arguments) => remove_command(arguments),
        Command::List(arguments) => list_command(arguments.json),
        Command::Status(arguments) => status_command(arguments.json),
        Command::Reindex(arguments) => reindex_command(arguments),
        Command::Daemon(command) => daemon_command(command),
        Command::Index(arguments) => index_command(arguments),
        Command::Runs(arguments) => {
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
    if arguments.list_tasks {
        for (id, instruction) in TASK_PRESETS {
            println!("{id}: {instruction}");
        }
        return Ok(ExitCode::SUCCESS);
    }

    // Explicit --db bypasses discovery and the daemon (scripting/evals).
    if let Some(db_path) = arguments.db.clone() {
        let results = direct_search(&arguments, &db_path, None, None)?;
        print_value(arguments.json, "search", &results)?;
        return Ok(ExitCode::SUCCESS);
    }

    let cwd = std::env::current_dir()?;
    let registry = codeindex_config::Registry::load()?;
    let context = codeindex_config::discover(&cwd)?;
    let registered = match (&arguments.project, &context) {
        (Some(needle), _) => registry
            .find(needle)
            .cloned()
            .with_context(|| format!("no registered project matches {needle:?}"))?,
        (None, Some(context)) => registry
            .find(&context.root.to_string_lossy())
            .cloned()
            .with_context(|| {
                format!(
                    "{} is not registered; run `codeindex add {}`",
                    context.root.display(),
                    context.root.display()
                )
            })?,
        (None, None) => anyhow::bail!(
            "no .codeindex.toml found walking up from {}; run `codeindex add`, or pass \
             --project or --db",
            cwd.display()
        ),
    };
    let context_tests = arguments
        .include_tests
        .then(|| "include".to_string())
        .or_else(|| {
            context
                .as_ref()
                .and_then(|context| context.tests_policy().map(str::to_owned))
        });

    // Reranking needs local model access; everything else goes through the
    // warm daemon.
    if arguments.no_daemon || arguments.rerank {
        let results = direct_search(
            &arguments,
            &registered.db.clone(),
            context_tests.as_deref(),
            context.as_ref(),
        )?;
        print_value(arguments.json, "search", &results)?;
        return Ok(ExitCode::SUCCESS);
    }

    let mut connection = codeindex_daemon::client::Connection::ensure()?;
    let params = codeindex_daemon::protocol::SearchParams {
        project: Some(registered.label.clone()),
        cwd: Some(cwd),
        query: arguments.query.clone(),
        space: arguments.space.clone(),
        task: arguments.task.clone(),
        instruction: arguments.instruction.clone(),
        filter: arguments.filter.clone(),
        limit: arguments.limit,
        retrieval: Some(retrieval_name(arguments.retrieval).to_string()),
        compress: Some(compress_name(arguments.compress).to_string()),
        no_graph: arguments.no_graph,
        tests: context_tests,
    };
    let value = connection.call("search", &params)?;
    let results: codeindex_daemon::protocol::SearchResults =
        serde_json::from_value(value).context("daemon returned an unexpected search payload")?;
    print_value(arguments.json, "search", &results)?;
    Ok(ExitCode::SUCCESS)
}

fn retrieval_name(retrieval: RetrievalArgument) -> &'static str {
    match retrieval {
        RetrievalArgument::Hybrid => "hybrid",
        RetrievalArgument::Dense => "dense",
        RetrievalArgument::Lexical => "lexical",
    }
}

fn compress_name(compress: CompressArgument) -> &'static str {
    match compress {
        CompressArgument::Auto => "auto",
        CompressArgument::Off => "off",
        CompressArgument::Always => "always",
    }
}

/// The in-process search path: loads the model (if a space is involved) and
/// runs the shared pipeline directly against the database.
fn direct_search(
    arguments: &SearchArgs,
    db_path: &Path,
    tests_policy: Option<&str>,
    context: Option<&codeindex_config::ProjectContext>,
) -> Result<codeindex_daemon::protocol::SearchResults> {
    use codeindex_core::{EmbeddingSpaceId, EmbeddingTask};
    use codeindex_daemon::pipeline::{self, Compress, HybridOptions, Retrieval};
    use codeindex_embedding::config::EmbeddingConfig;
    use codeindex_query::WhereFilter;
    use codeindex_search::SearchIndex;
    use std::collections::HashMap;

    let db = open_or_create(db_path)?;
    let retrieval: Retrieval = retrieval_name(arguments.retrieval).parse()?;
    let compress: Compress = compress_name(arguments.compress).parse()?;

    let space_id = arguments
        .space
        .clone()
        .or_else(|| context.and_then(|context| context.config.search.default_space.clone()))
        .map(EmbeddingSpaceId::new)
        .or_else(|| {
            let spaces = db.list_spaces().ok()?;
            match spaces.as_slice() {
                [only] => Some(only.identity.id.clone()),
                _ => None,
            }
        });
    if retrieval == Retrieval::Dense && space_id.is_none() {
        anyhow::bail!(
            "--retrieval dense needs an embedding space (pass --space or define one in \
             .codeindex.toml)"
        );
    }

    let task = match (&arguments.task, &arguments.instruction) {
        (Some(preset), _) => {
            let from_config = context
                .and_then(|context| context.config.tasks.get(preset))
                .map(|task| task.instruction.clone());
            let instruction = match from_config {
                Some(instruction) => instruction,
                None => TASK_PRESETS
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
            };
            Some(EmbeddingTask::new(preset.clone(), instruction))
        }
        (None, Some(instruction)) => Some(EmbeddingTask::new("custom", instruction.clone())),
        (None, None) => None,
    };

    let mut filter = WhereFilter::parse(arguments.filter.as_deref())?;
    pipeline::apply_tests_policy(
        &mut filter,
        if arguments.include_tests {
            Some("include")
        } else {
            tests_policy
        },
    )?;

    let index = SearchIndex::from_snapshot(db.snapshot(&[])?)?;
    let options = HybridOptions {
        query: arguments.query.clone(),
        task,
        space: if retrieval == Retrieval::Lexical {
            None
        } else {
            space_id
        },
        filter,
        limit: arguments.limit,
        retrieval,
        compress,
        graph: !arguments.no_graph,
    };

    let mut local_embedder = None;
    if let Some(space_id) = &options.space {
        let space = db
            .get_space(space_id)?
            .with_context(|| format!("embedding space {space_id} is not stored"))?;
        // The stored semantic contract's model field is itself a resolvable
        // reference, so the matching query embedder reconstructs
        // automatically.
        let config = EmbeddingConfig {
            model: space.identity.model.model.clone(),
            cache_dir: arguments.cache_dir.clone(),
            execution_provider: arguments.execution_provider.clone(),
            ..EmbeddingConfig::default()
        };
        local_embedder = Some(codeindex_embedding::embedder_from_config(&config)?);
    }

    // `mut` is exercised only by the cfg-gated rerank arm below.
    #[cfg_attr(not(feature = "candle"), allow(unused_mut))]
    let mut outcome =
        pipeline::hybrid_search(&db, &index, local_embedder.as_deref_mut(), &options)?;

    // Second stage: cross-encoder judgement of the fused head.
    #[cfg_attr(not(feature = "candle"), allow(unused_mut))]
    let mut rerank_scores: HashMap<usize, f32> = HashMap::new();
    if arguments.rerank {
        #[cfg(not(feature = "candle"))]
        anyhow::bail!("--rerank needs the `candle` feature; rebuild with --features candle");
        #[cfg(feature = "candle")]
        {
            use codeindex_embedding::rerank::{Qwen3Reranker, Reranker as _};
            let head = arguments.rerank_candidates.min(outcome.fused.len());
            let instruction = options
                .task
                .as_ref()
                .map(|task| task.instruction.clone())
                .unwrap_or_else(|| {
                    "Given a code search query, retrieve relevant code implementations".to_string()
                });
            let mut judged: Vec<(usize, &str)> = Vec::new();
            for hit in outcome.fused.iter().take(head) {
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
                let config = EmbeddingConfig {
                    model: String::new(),
                    cache_dir: arguments.cache_dir.clone(),
                    execution_provider: arguments.execution_provider.clone(),
                    ..EmbeddingConfig::default()
                };
                let mut reranker = Qwen3Reranker::from_reference(&arguments.rerank_model, &config)?;
                let documents: Vec<&str> = judged.iter().map(|(_, text)| *text).collect();
                let scores = reranker.rerank(&instruction, &arguments.query, &documents)?;
                for ((position, _), score) in judged.iter().zip(scores) {
                    rerank_scores.insert(*position, score);
                }
                // Judged candidates re-sort by relevance; unjudged keep
                // their fused order below them.
                outcome.fused.sort_by(|left, right| {
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

    Ok(pipeline::shape_results(
        &index,
        &outcome,
        &options,
        &rerank_scores,
    ))
}

fn add_command(arguments: AddArgs) -> Result<ExitCode> {
    let path = match &arguments.path {
        Some(path) => path.clone(),
        None => std::env::current_dir()?,
    };
    let path = std::fs::canonicalize(&path)
        .with_context(|| format!("{} does not exist", path.display()))?;
    // Adding from inside an existing project registers the enclosing root.
    let root = match codeindex_config::discover(&path)? {
        Some(context) => context.root,
        None => path,
    };
    let config_path = root.join(codeindex_config::CONFIG_FILE);
    if !config_path.exists() {
        let label = root
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "project".into());
        std::fs::write(&config_path, codeindex_config::starter_config(&label))?;
        eprintln!("created {}", config_path.display());
    }

    if arguments.no_daemon {
        let context = codeindex_config::discover(&root)?
            .context("the project root just written failed discovery")?;
        codeindex_config::validate_root_config(&context.config)?;
        let mut registry = codeindex_config::Registry::load()?;
        let project = registry.register(&root, &context.label(), context.db_path())?;
        registry.save()?;
        let overrides = codeindex_config::collect_overrides(&root)?;
        let mut embedder_for =
            |model: &str| -> Result<Box<dyn codeindex_embedding::EmbeddingBackend>> {
                codeindex_embedding::embedder_from_config(
                    &codeindex_embedding::config::EmbeddingConfig {
                        model: model.to_string(),
                        ..Default::default()
                    },
                )
            };
        let summary = codeindex_daemon::pipeline::run_index_job(
            &project.db,
            &project.label,
            &root,
            &context.config,
            &overrides,
            &mut embedder_for,
            |phase, detail| eprintln!("{phase}: {detail}"),
        )?;
        print_value(
            arguments.json,
            "add",
            &serde_json::json!({
                "label": project.label,
                "root": project.root,
                "db": project.db,
                "units": summary.units,
                "embedded": summary.embedded_spaces,
            }),
        )?;
        return Ok(ExitCode::SUCCESS);
    }

    let mut connection = codeindex_daemon::client::Connection::ensure()?;
    let result = connection.call(
        "project.add",
        &codeindex_daemon::protocol::AddParams { root },
    )?;
    let result: codeindex_daemon::protocol::AddResult = serde_json::from_value(result)?;
    print_value(arguments.json, "add", &result)?;
    Ok(ExitCode::SUCCESS)
}

fn remove_command(arguments: RemoveArgs) -> Result<ExitCode> {
    if let Some(mut connection) = codeindex_daemon::client::Connection::attach()? {
        let result = connection.call(
            "project.remove",
            &codeindex_daemon::protocol::RemoveParams {
                needle: arguments.needle.clone(),
                purge: arguments.purge,
            },
        )?;
        let result: codeindex_daemon::protocol::RemoveResult = serde_json::from_value(result)?;
        print_value(arguments.json, "remove", &result)?;
        return Ok(ExitCode::SUCCESS);
    }
    // No daemon running: edit the registry directly with the same
    // owned-state purge rule the daemon applies.
    let mut registry = codeindex_config::Registry::load()?;
    let removed = registry.remove(&arguments.needle)?;
    registry.save()?;
    let mut purged = None;
    if arguments.purge {
        let default_dir = codeindex_config::default_db_path(&removed.root)
            .parent()
            .map(Path::to_path_buf);
        if let Some(dir) = default_dir
            && removed.db.starts_with(&dir)
            && dir.starts_with(codeindex_config::data_root())
            && dir.exists()
        {
            std::fs::remove_dir_all(&dir)?;
            purged = Some(dir);
        }
    }
    print_value(
        arguments.json,
        "remove",
        &codeindex_daemon::protocol::RemoveResult {
            label: removed.label,
            root: removed.root,
            purged,
        },
    )?;
    Ok(ExitCode::SUCCESS)
}

fn list_command(json: bool) -> Result<ExitCode> {
    status_command(json)
}

fn status_command(json: bool) -> Result<ExitCode> {
    if let Some(mut connection) = codeindex_daemon::client::Connection::attach()? {
        let value = connection.call("daemon.status", &serde_json::json!({}))?;
        let status: codeindex_daemon::protocol::StatusResult = serde_json::from_value(value)?;
        print_value(json, "status", &status)?;
        return Ok(ExitCode::SUCCESS);
    }
    // Daemon not running: report registry contents directly.
    let registry = codeindex_config::Registry::load()?;
    let mut projects = Vec::new();
    for project in &registry.projects {
        let (units, spaces) = if project.db.exists() {
            match open_or_create(&project.db) {
                Ok(db) => (
                    db.count_units().ok(),
                    db.list_spaces()
                        .map(|spaces| {
                            spaces
                                .into_iter()
                                .map(|space| space.identity.id.to_string())
                                .collect()
                        })
                        .unwrap_or_default(),
                ),
                Err(_) => (None, Vec::new()),
            }
        } else {
            (None, Vec::new())
        };
        projects.push(codeindex_daemon::protocol::ProjectSummary {
            label: project.label.clone(),
            root: project.root.clone(),
            db: project.db.clone(),
            units,
            spaces,
            job: None,
        });
    }
    if !json {
        eprintln!("daemon: not running (state read directly from the registry)");
    }
    print_value(
        json,
        "status",
        &serde_json::json!({ "daemon": "not-running", "projects": projects }),
    )?;
    Ok(ExitCode::SUCCESS)
}

fn reindex_command(arguments: ReindexArgs) -> Result<ExitCode> {
    let needle = match &arguments.needle {
        Some(needle) => needle.clone(),
        None => {
            let cwd = std::env::current_dir()?;
            codeindex_config::discover(&cwd)?
                .map(|context| context.root.to_string_lossy().into_owned())
                .with_context(|| {
                    format!("no .codeindex.toml found walking up from {}", cwd.display())
                })?
        }
    };
    let mut connection = codeindex_daemon::client::Connection::ensure()?;
    let result = connection.call(
        "project.reindex",
        &codeindex_daemon::protocol::NeedleParams { needle },
    )?;
    print_value(arguments.json, "reindex", &result)?;
    Ok(ExitCode::SUCCESS)
}

fn daemon_command(command: DaemonCommand) -> Result<ExitCode> {
    match command {
        DaemonCommand::Start => {
            let mut connection = codeindex_daemon::client::Connection::ensure()?;
            let ping = connection.call("daemon.ping", &serde_json::json!({}))?;
            println!("daemon running: {ping}");
            Ok(ExitCode::SUCCESS)
        }
        DaemonCommand::Stop => {
            println!("{}", codeindex_daemon::client::stop()?);
            Ok(ExitCode::SUCCESS)
        }
        DaemonCommand::Status => {
            println!("{}", codeindex_daemon::client::lifecycle_status()?);
            Ok(ExitCode::SUCCESS)
        }
    }
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
