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

#[derive(Serialize)]
struct Envelope<'a, T> {
    version: u32,
    event: &'a str,
    data: T,
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
    }
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
            let envelope = Envelope {
                version: 1,
                event: "progress",
                data: progress,
            };
            if let Ok(line) = serde_json::to_string(&envelope) {
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
        println!(
            "{}",
            serde_json::to_string(&Envelope {
                version: 1,
                event,
                data: value,
            })?
        );
    } else {
        println!("{value:#?}");
    }
    Ok(())
}
