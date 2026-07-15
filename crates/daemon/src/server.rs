//! The daemon service: accepts authenticated streams from daemonkit,
//! dispatches framed requests onto blocking worker threads, and runs
//! background index jobs. Connections are expected to be short-lived (one
//! per CLI command), which keeps daemonkit's drain semantics healthy.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use codeindex_core::{EmbeddingSpaceId, EmbeddingTask};
use codeindex_query::WhereFilter;
use codeindex_sqlite::open_or_create;
use serde_json::{Value, json};

use crate::pipeline::{self, Compress, HybridOptions, Retrieval};
use crate::protocol::{
    self, AddParams, AddResult, NeedleParams, ProjectSummary, RemoveParams, RemoveResult, Request,
    Response, SearchParams, StatusResult,
};
use crate::state::{DaemonState, ProjectHandle};

/// Entry point for the bootstrap path: serve until daemonkit requests
/// shutdown. In-flight connections finish because daemonkit drains
/// application streams before the process exits.
pub async fn run(bootstrap: daemonkit::Bootstrap) -> Result<()> {
    bootstrap
        .run_embedded_fn(|_context, mut incoming, mut shutdown| async move {
            let state = match DaemonState::new() {
                Ok(state) => Arc::new(state),
                Err(error) => {
                    eprintln!("codeindex daemon failed to load state: {error:#}");
                    return Err(std::io::Error::other(format!("{error:#}")));
                }
            };
            loop {
                tokio::select! {
                    _ = shutdown.requested() => break,
                    stream = next_stream(&mut incoming) => {
                        match stream {
                            Some(Ok(stream)) => {
                                let state = state.clone();
                                tokio::spawn(async move {
                                    if let Err(error) = serve_connection(state, stream).await {
                                        eprintln!("codeindex daemon connection error: {error:#}");
                                    }
                                });
                            }
                            Some(Err(error)) => {
                                eprintln!("codeindex daemon accept error: {error}");
                            }
                            None => break,
                        }
                    }
                }
            }
            Ok::<(), std::io::Error>(())
        })
        .await
        .map_err(|error| anyhow!("daemon lifecycle error: {error}"))
}

async fn next_stream(
    incoming: &mut daemonkit::Incoming,
) -> Option<Result<daemonkit::AuthenticatedStream, impl std::error::Error>> {
    use futures_core::Stream;
    std::future::poll_fn(|cx| std::pin::Pin::new(&mut *incoming).poll_next(cx)).await
}

async fn serve_connection(
    state: Arc<DaemonState>,
    mut stream: daemonkit::AuthenticatedStream,
) -> Result<()> {
    while let Some(payload) = protocol::read_frame(&mut stream).await? {
        let request: Request = match serde_json::from_slice(&payload) {
            Ok(request) => request,
            Err(error) => {
                let response = Response {
                    id: 0,
                    result: None,
                    error: Some(format!("malformed request: {error}")),
                };
                protocol::write_frame(&mut stream, &serde_json::to_vec(&response)?).await?;
                continue;
            }
        };
        let id = request.id;
        let state = state.clone();
        let result = tokio::task::spawn_blocking(move || dispatch(state, request)).await;
        let response = match result {
            Ok(Ok(value)) => Response {
                id,
                result: Some(value),
                error: None,
            },
            Ok(Err(error)) => Response {
                id,
                result: None,
                error: Some(format!("{error:#}")),
            },
            Err(join_error) => Response {
                id,
                result: None,
                error: Some(format!("request handler panicked: {join_error}")),
            },
        };
        protocol::write_frame(&mut stream, &serde_json::to_vec(&response)?).await?;
    }
    Ok(())
}

fn dispatch(state: Arc<DaemonState>, request: Request) -> Result<Value> {
    match request.method.as_str() {
        "daemon.ping" => Ok(json!({
            "version": env!("CARGO_PKG_VERSION"),
            "protocol": protocol::PROTOCOL_VERSION,
        })),
        "daemon.status" => daemon_status(&state),
        "project.add" => project_add(&state, serde_json::from_value(request.params)?),
        "project.remove" => project_remove(&state, serde_json::from_value(request.params)?),
        "project.list" => daemon_status(&state),
        "project.reindex" => project_reindex(&state, serde_json::from_value(request.params)?),
        "search" => search(&state, serde_json::from_value(request.params)?),
        other => bail!("unknown method {other:?}"),
    }
}

fn daemon_status(state: &DaemonState) -> Result<Value> {
    let mut projects = Vec::new();
    for project in state.registered_projects() {
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
        projects.push(ProjectSummary {
            label: project.label.clone(),
            root: project.root.clone(),
            db: project.db.clone(),
            units,
            spaces,
            job: state.job(&project.label),
        });
    }
    Ok(serde_json::to_value(StatusResult {
        version: env!("CARGO_PKG_VERSION").to_string(),
        protocol: protocol::PROTOCOL_VERSION,
        pid: std::process::id(),
        uptime_seconds: state.started.elapsed().as_secs(),
        projects,
    })?)
}

fn project_add(state: &Arc<DaemonState>, params: AddParams) -> Result<Value> {
    let root = params.root;
    if !root.is_dir() {
        bail!("{} is not a directory", root.display());
    }
    let config_path = root.join(codeindex_config::CONFIG_FILE);
    if !config_path.is_file() {
        bail!(
            "{} has no {} (the CLI creates one on `codeindex add`)",
            root.display(),
            codeindex_config::CONFIG_FILE
        );
    }
    let config: codeindex_config::ProjectConfig =
        toml::from_str(&std::fs::read_to_string(&config_path)?)
            .with_context(|| format!("parsing {}", config_path.display()))?;
    codeindex_config::validate_root_config(&config)?;

    let label_hint = config
        .label
        .clone()
        .or_else(|| {
            root.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "project".into());
    let db_path = match &config.storage.path {
        Some(path) if path.is_absolute() => path.clone(),
        Some(path) => root.join(path),
        None => codeindex_config::default_db_path(&root),
    };
    let (registered, existed) = state.register_project(&root, &label_hint, db_path)?;
    // Config may have changed since a previous registration.
    state.invalidate_project(&registered.label);

    let job = if state.job_running(&registered.label) {
        "already-indexing"
    } else {
        spawn_index_job(state, &registered.label)?;
        if existed { "reindexing" } else { "indexing" }
    };
    Ok(serde_json::to_value(AddResult {
        label: registered.label,
        root: registered.root,
        db: registered.db,
        job: job.to_string(),
    })?)
}

fn project_remove(state: &DaemonState, params: RemoveParams) -> Result<Value> {
    let removed = state.remove_project(&params.needle)?;
    let mut purged = None;
    if params.purge {
        // Only delete state this daemon owns: the default per-project
        // directory under the data root. Custom [storage] paths stay.
        let default_dir = codeindex_config::default_db_path(&removed.root)
            .parent()
            .map(Path::to_path_buf);
        if let Some(dir) = default_dir
            && removed.db.starts_with(&dir)
            && dir.starts_with(codeindex_config::data_root())
            && dir.exists()
        {
            std::fs::remove_dir_all(&dir).with_context(|| format!("purging {}", dir.display()))?;
            purged = Some(dir);
        }
    }
    Ok(serde_json::to_value(RemoveResult {
        label: removed.label,
        root: removed.root,
        purged,
    })?)
}

fn project_reindex(state: &Arc<DaemonState>, params: NeedleParams) -> Result<Value> {
    let project = state.resolve_project(Some(&params.needle), None)?;
    if state.job_running(&project.label) {
        bail!("project {:?} is already indexing", project.label);
    }
    state.invalidate_project(&project.label);
    spawn_index_job(state, &project.label)?;
    Ok(json!({ "label": project.label, "job": "reindexing" }))
}

fn spawn_index_job(state: &Arc<DaemonState>, label: &str) -> Result<()> {
    // The job thread re-resolves everything from the registry so it survives
    // handle invalidation; it owns its status entry from start to finish.
    let project = state.resolve_project(Some(label), None)?;
    let handle = state.project_handle(&project)?;
    let state = state.clone();
    state.set_job(&handle.label, "queued", "");
    std::thread::Builder::new()
        .name(format!("index:{}", handle.label))
        .spawn(move || {
            let label = handle.label.clone();
            let result = run_job(&state, &handle);
            state.finish_job(&label, result.err().map(|error| format!("{error:#}")));
        })
        .context("spawning index job thread")?;
    Ok(())
}

fn run_job(state: &Arc<DaemonState>, handle: &Arc<ProjectHandle>) -> Result<()> {
    let overrides = codeindex_config::collect_overrides(&handle.root)?;
    let state_for_embedders = state.clone();
    let mut embedder_for =
        move |model: &str| -> Result<Box<dyn codeindex_embedding::EmbeddingBackend>> {
            Ok(Box::new(state_for_embedders.embedder(model)?))
        };
    let label = handle.label.clone();
    let state_for_progress = state.clone();
    let summary = pipeline::run_index_job(
        &handle.db_path,
        &handle.label,
        &handle.root,
        &handle.config,
        &overrides,
        &mut embedder_for,
        move |phase, detail| state_for_progress.set_job(&label, phase, detail),
    )?;
    state.set_job(
        &handle.label,
        "done",
        &format!(
            "{} units{}",
            summary.units,
            summary
                .embedded_spaces
                .iter()
                .map(|(space, count)| format!(", {space}: {count} embedded"))
                .collect::<String>()
        ),
    );
    Ok(())
}

fn search(state: &Arc<DaemonState>, params: SearchParams) -> Result<Value> {
    let project = state.resolve_project(params.project.as_deref(), params.cwd.as_deref())?;
    let handle = state.project_handle(&project)?;
    let index = handle.search_index()?;
    let db = open_or_create(&handle.db_path)?;

    // Space: explicit, else the config default, else the single stored
    // space, else lexical-only.
    let space_id = params
        .space
        .clone()
        .or_else(|| handle.config.search.default_space.clone())
        .map(EmbeddingSpaceId::new)
        .or_else(|| {
            let spaces = db.list_spaces().ok()?;
            match spaces.as_slice() {
                [only] => Some(only.identity.id.clone()),
                _ => None,
            }
        });

    let task = resolve_task(
        &handle.config,
        params.task.as_deref(),
        params.instruction.as_deref(),
    )?;
    let mut filter = WhereFilter::parse(params.filter.as_deref())?;
    let tests = params
        .tests
        .as_deref()
        .or(handle.config.search.tests.as_deref());
    pipeline::apply_tests_policy(&mut filter, tests)?;

    let retrieval: Retrieval = params
        .retrieval
        .as_deref()
        .or(handle.config.search.retrieval.as_deref())
        .unwrap_or("hybrid")
        .parse()?;
    let compress: Compress = params.compress.as_deref().unwrap_or("auto").parse()?;

    let options = HybridOptions {
        query: params.query.clone(),
        task,
        space: if retrieval == Retrieval::Lexical {
            None
        } else {
            space_id
        },
        filter,
        limit: params.limit,
        retrieval,
        compress,
        graph: !params.no_graph,
    };

    let mut remote;
    let mut embedder: Option<&mut dyn codeindex_embedding::EmbeddingBackend> = match &options.space
    {
        Some(space_id) => {
            let space = db.get_space(space_id)?.with_context(|| {
                format!(
                    "embedding space {space_id} is not stored for {:?}",
                    project.label
                )
            })?;
            remote = state.embedder(&space.identity.model.model)?;
            Some(&mut remote)
        }
        None => None,
    };

    let outcome = pipeline::hybrid_search(&db, &index, embedder.take(), &options)?;
    let results = pipeline::shape_results(&index, &outcome, &options, &Default::default());
    Ok(serde_json::to_value(results)?)
}

fn resolve_task(
    config: &codeindex_config::ProjectConfig,
    preset: Option<&str>,
    instruction: Option<&str>,
) -> Result<Option<EmbeddingTask>> {
    if let Some(instruction) = instruction {
        return Ok(Some(EmbeddingTask::new("custom", instruction.to_string())));
    }
    let Some(preset) = preset else {
        return Ok(None);
    };
    if let Some(task) = config.tasks.get(preset) {
        return Ok(Some(EmbeddingTask::new(
            preset.to_string(),
            task.instruction.clone(),
        )));
    }
    let instruction = codeindex_query::TASK_PRESETS
        .iter()
        .find(|(id, _)| *id == preset)
        .map(|(_, instruction)| (*instruction).to_string())
        .with_context(|| {
            format!(
                "unknown task preset {preset:?}; available: {}",
                codeindex_query::TASK_PRESETS
                    .iter()
                    .map(|(id, _)| *id)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
    Ok(Some(EmbeddingTask::new(preset.to_string(), instruction)))
}
