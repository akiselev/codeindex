//! Resident daemon state: the registry of roots, per-project cached search
//! indexes, background index/embed jobs, and warm embedding backends.
//!
//! Embedding models are the expensive resident state and the reason the
//! daemon exists. Each distinct model reference gets one dedicated worker
//! thread that constructs the backend *inside* the thread and serves
//! embed jobs over a channel; [`RemoteEmbedder`] adapts that channel back
//! into the `EmbeddingBackend` trait so every existing pipeline works
//! unchanged against a warm model.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result, anyhow, bail};
use codeindex_config::{ProjectConfig, RegisteredProject, Registry};
use codeindex_core::{EmbeddingTask, ExecutionInfo, ModelContract};
use codeindex_embedding::config::EmbeddingConfig;
use codeindex_embedding::{EmbedRequest, EmbeddingBackend};
use codeindex_search::SearchIndex;
use codeindex_sqlite::open_or_create;

use crate::protocol::JobState;

pub struct DaemonState {
    pub started: Instant,
    registry_path: PathBuf,
    registry: Mutex<Registry>,
    projects: Mutex<HashMap<String, Arc<ProjectHandle>>>,
    embedders: Mutex<HashMap<String, Arc<Mutex<Option<EmbedderHandle>>>>>,
    jobs: Mutex<HashMap<String, JobState>>,
}

impl DaemonState {
    pub fn new() -> Result<DaemonState> {
        let registry_path = codeindex_config::registry_path();
        let registry = Registry::load_from(&registry_path)?;
        Ok(DaemonState {
            started: Instant::now(),
            registry_path,
            registry: Mutex::new(registry),
            projects: Mutex::new(HashMap::new()),
            embedders: Mutex::new(HashMap::new()),
            jobs: Mutex::new(HashMap::new()),
        })
    }

    pub fn registered_projects(&self) -> Vec<RegisteredProject> {
        self.registry
            .lock()
            .expect("registry lock")
            .projects
            .clone()
    }

    pub fn register_project(
        &self,
        root: &Path,
        label_hint: &str,
        db: PathBuf,
    ) -> Result<(RegisteredProject, bool)> {
        let mut registry = self.registry.lock().expect("registry lock");
        let existed = registry.projects.iter().any(|project| project.root == root);
        let project = registry.register(root, label_hint, db)?;
        registry.save_to(&self.registry_path)?;
        Ok((project, existed))
    }

    pub fn remove_project(&self, needle: &str) -> Result<RegisteredProject> {
        let mut registry = self.registry.lock().expect("registry lock");
        let removed = registry.remove(needle)?;
        registry.save_to(&self.registry_path)?;
        drop(registry);
        self.projects
            .lock()
            .expect("projects lock")
            .remove(&removed.label);
        self.jobs.lock().expect("jobs lock").remove(&removed.label);
        Ok(removed)
    }

    /// Resolve a project by explicit needle (label or path) or by working
    /// directory containment.
    pub fn resolve_project(
        &self,
        needle: Option<&str>,
        cwd: Option<&Path>,
    ) -> Result<RegisteredProject> {
        let registry = self.registry.lock().expect("registry lock");
        if let Some(needle) = needle {
            return registry
                .find(needle)
                .cloned()
                .with_context(|| format!("no registered project matches {needle:?}"));
        }
        if let Some(cwd) = cwd
            && let Some(project) = registry.find(&cwd.to_string_lossy())
        {
            return Ok(project.clone());
        }
        match registry.projects.as_slice() {
            [only] => Ok(only.clone()),
            [] => bail!("no projects registered; run `codeindex add` first"),
            _ => bail!(
                "cannot resolve a project from the working directory; pass --project \
                 (registered: {})",
                registry
                    .projects
                    .iter()
                    .map(|project| project.label.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

    pub fn project_handle(&self, project: &RegisteredProject) -> Result<Arc<ProjectHandle>> {
        let mut projects = self.projects.lock().expect("projects lock");
        if let Some(handle) = projects.get(&project.label) {
            return Ok(handle.clone());
        }
        let config_path = project.root.join(codeindex_config::CONFIG_FILE);
        let config: ProjectConfig = if config_path.is_file() {
            toml::from_str(&std::fs::read_to_string(&config_path)?)
                .with_context(|| format!("parsing {}", config_path.display()))?
        } else {
            ProjectConfig::default()
        };
        let handle = Arc::new(ProjectHandle {
            label: project.label.clone(),
            root: project.root.clone(),
            db_path: project.db.clone(),
            config,
            cache: Mutex::new(None),
        });
        projects.insert(project.label.clone(), handle.clone());
        Ok(handle)
    }

    /// Forget a cached project handle (config may have changed on disk).
    pub fn invalidate_project(&self, label: &str) {
        self.projects.lock().expect("projects lock").remove(label);
    }

    pub fn set_job(&self, label: &str, phase: &str, detail: &str) {
        self.jobs.lock().expect("jobs lock").insert(
            label.to_string(),
            JobState {
                phase: phase.to_string(),
                detail: detail.to_string(),
                finished: false,
                error: None,
            },
        );
    }

    pub fn finish_job(&self, label: &str, error: Option<String>) {
        let mut jobs = self.jobs.lock().expect("jobs lock");
        let entry = jobs.entry(label.to_string()).or_insert_with(|| JobState {
            phase: "done".into(),
            detail: String::new(),
            finished: true,
            error: None,
        });
        entry.finished = true;
        entry.error = error;
        if entry.error.is_some() {
            entry.phase = "failed".into();
        } else {
            entry.phase = "done".into();
        }
    }

    pub fn job(&self, label: &str) -> Option<JobState> {
        self.jobs.lock().expect("jobs lock").get(label).cloned()
    }

    pub fn job_running(&self, label: &str) -> bool {
        self.job(label).is_some_and(|job| !job.finished)
    }

    /// A warm embedding backend for a model reference. The first call for a
    /// reference spawns its worker thread and blocks until the model is
    /// loaded; later calls (and other models) are unaffected — the outer map
    /// lock is held only to fetch the per-model slot.
    pub fn embedder(&self, model: &str) -> Result<RemoteEmbedder> {
        let slot = {
            let mut embedders = self.embedders.lock().expect("embedders lock");
            embedders
                .entry(model.to_string())
                .or_insert_with(|| Arc::new(Mutex::new(None)))
                .clone()
        };
        let mut guard = slot.lock().expect("embedder slot lock");
        if guard.is_none() {
            *guard = Some(spawn_embed_worker(model)?);
        }
        let handle = guard.as_ref().expect("just initialized");
        Ok(RemoteEmbedder {
            contract: handle.contract.clone(),
            execution: handle.execution.clone(),
            sender: handle.sender.clone(),
        })
    }
}

pub struct ProjectHandle {
    pub label: String,
    pub root: PathBuf,
    pub db_path: PathBuf,
    pub config: ProjectConfig,
    cache: Mutex<Option<CachedIndex>>,
}

struct CachedIndex {
    index: Arc<SearchIndex>,
    stamp: DbStamp,
}

/// Cheap freshness stamp: the mtimes+sizes of the database and its WAL.
/// A stale stamp only costs a rebuild; a missed change cannot happen because
/// every publish rewrites the WAL.
#[derive(PartialEq, Eq, Clone, Debug)]
struct DbStamp(Vec<(PathBuf, SystemTime, u64)>);

fn db_stamp(db_path: &Path) -> DbStamp {
    let mut parts = Vec::new();
    for suffix in ["", "-wal"] {
        let path = PathBuf::from(format!("{}{suffix}", db_path.display()));
        if let Ok(meta) = std::fs::metadata(&path) {
            parts.push((
                path,
                meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                meta.len(),
            ));
        }
    }
    DbStamp(parts)
}

impl ProjectHandle {
    /// The warm `SearchIndex`, rebuilt only when the database changed.
    pub fn search_index(&self) -> Result<Arc<SearchIndex>> {
        if !self.db_path.exists() {
            bail!(
                "project {:?} has no index yet (db {} does not exist); \
                 indexing may still be running — check `codeindex status`",
                self.label,
                self.db_path.display()
            );
        }
        let stamp = db_stamp(&self.db_path);
        let mut cache = self.cache.lock().expect("index cache lock");
        if let Some(cached) = cache.as_ref()
            && cached.stamp == stamp
        {
            return Ok(cached.index.clone());
        }
        let db = open_or_create(&self.db_path)?;
        let index = Arc::new(SearchIndex::from_snapshot(db.snapshot(&[])?)?);
        *cache = Some(CachedIndex {
            index: index.clone(),
            stamp,
        });
        Ok(cache.as_ref().expect("just set").index.clone())
    }
}

// ---- warm embedding workers ---------------------------------------------

struct EmbedderHandle {
    contract: ModelContract,
    execution: ExecutionInfo,
    sender: mpsc::Sender<EmbedJob>,
}

struct EmbedJob {
    role: codeindex_embedding::EmbeddingRole,
    task: Option<EmbeddingTask>,
    document_prompt: Option<String>,
    inputs: Vec<String>,
    reply: mpsc::SyncSender<Result<Vec<Vec<f32>>, String>>,
}

/// Channel-backed [`EmbeddingBackend`] whose model lives on a dedicated
/// worker thread. Cloning is cheap; concurrent users serialize on the
/// worker's queue, which matches the batch=1 execution model underneath.
#[derive(Clone)]
pub struct RemoteEmbedder {
    contract: ModelContract,
    execution: ExecutionInfo,
    sender: mpsc::Sender<EmbedJob>,
}

impl EmbeddingBackend for RemoteEmbedder {
    fn contract(&self) -> &ModelContract {
        &self.contract
    }

    fn execution(&self) -> &ExecutionInfo {
        &self.execution
    }

    fn embed(&mut self, request: &EmbedRequest<'_>) -> Result<Vec<Vec<f32>>> {
        let (reply, receive) = mpsc::sync_channel(1);
        self.sender
            .send(EmbedJob {
                role: request.role,
                task: request.task.cloned(),
                document_prompt: request.document_prompt.map(str::to_owned),
                inputs: request.inputs.iter().map(|s| (*s).to_owned()).collect(),
                reply,
            })
            .map_err(|_| anyhow!("embedding worker exited"))?;
        receive
            .recv()
            .map_err(|_| anyhow!("embedding worker dropped the reply channel"))?
            .map_err(|message| anyhow!(message))
    }
}

fn spawn_embed_worker(model: &str) -> Result<EmbedderHandle> {
    let (sender, receiver) = mpsc::channel::<EmbedJob>();
    let (ready_sender, ready_receiver) =
        mpsc::sync_channel::<Result<(ModelContract, ExecutionInfo), String>>(1);
    let model_ref = model.to_string();
    std::thread::Builder::new()
        .name(format!("embed:{model_ref}"))
        .spawn(move || {
            let config = EmbeddingConfig {
                model: model_ref.clone(),
                ..EmbeddingConfig::default()
            };
            let mut backend = match codeindex_embedding::embedder_from_config(&config) {
                Ok(backend) => {
                    let _ = ready_sender.send(Ok((
                        backend.contract().clone(),
                        backend.execution().clone(),
                    )));
                    backend
                }
                Err(error) => {
                    let _ = ready_sender.send(Err(format!("{error:#}")));
                    return;
                }
            };
            while let Ok(job) = receiver.recv() {
                let inputs: Vec<&str> = job.inputs.iter().map(String::as_str).collect();
                let request = EmbedRequest {
                    role: job.role,
                    task: job.task.as_ref(),
                    document_prompt: job.document_prompt.as_deref(),
                    inputs: &inputs,
                };
                let result = backend
                    .embed(&request)
                    .map_err(|error| format!("{error:#}"));
                let _ = job.reply.send(result);
            }
        })
        .context("spawning embedding worker thread")?;
    // Model resolution may download weights on first use: allow a generous
    // readiness window, but fail loudly rather than hanging forever.
    let ready = ready_receiver
        .recv_timeout(Duration::from_secs(30 * 60))
        .map_err(|_| anyhow!("embedding worker for {model} did not become ready"))?;
    let (contract, execution) = ready.map_err(|message| anyhow!(message))?;
    Ok(EmbedderHandle {
        contract,
        execution,
        sender,
    })
}
