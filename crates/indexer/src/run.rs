//! High-level atomic/resumable indexing state machine.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use codeindex_sqlite::index_publish::{IndexReport, PublishStep};
use codeindex_sqlite::index_runs::{
    CreateRunSpec, DocumentAction, DocumentState, IndexRunPhase, IndexRunState, IndexRunStatus,
    ManifestDocument, RunProjectSpec, STAGED_PAYLOAD_SCHEMA_VERSION, manifest_digest,
};
use codeindex_sqlite::{Db, FileRecord};
use codeindex_tree_sitter::normalizer::sha256_hex;
use serde::{Deserialize, Serialize};

use crate::source::{RevisionSemantics, StableRead};
use crate::stage::{PrepareDocument, prepare_document};
use crate::{IndexSettings, RepresentationEnricher, SourceDocument, SourceProject};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResumePolicy {
    #[default]
    Auto,
    New,
    Run(i64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RevisionTrust {
    #[default]
    VerifyContent,
    TrustAdvisory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RefreshMode {
    #[default]
    Automatic,
    PauseOnChange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetryBackoff {
    pub initial_millis: u64,
    pub maximum_millis: u64,
}

impl Default for RetryBackoff {
    fn default() -> Self {
        Self {
            initial_millis: 25,
            maximum_millis: 1_000,
        }
    }
}

impl RetryBackoff {
    fn delay(self, attempt: u32) -> Duration {
        let factor = 1_u64.checked_shl(attempt.min(20)).unwrap_or(u64::MAX);
        Duration::from_millis(
            self.initial_millis
                .saturating_mul(factor)
                .min(self.maximum_millis),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefreshPolicy {
    pub mode: RefreshMode,
    pub settle_delay: Duration,
    pub retry_backoff: RetryBackoff,
}

impl Default for RefreshPolicy {
    fn default() -> Self {
        Self {
            mode: RefreshMode::Automatic,
            settle_delay: Duration::from_millis(20),
            retry_backoff: RetryBackoff::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Attempts for a failing document in one invocation. Durable attempts are
    /// retained for diagnostics, but a new invocation receives a fresh budget.
    pub document_attempts_per_invocation: u32,
    pub backoff: RetryBackoff,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            document_attempts_per_invocation: 1,
            backoff: RetryBackoff::default(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexProgress {
    pub run_id: i64,
    pub phase: IndexRunPhase,
    pub project_label: Option<String>,
    pub source_document_id: Option<String>,
    pub ready_documents: u64,
    pub total_documents: u64,
    pub refresh_round: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", content = "value", rename_all = "snake_case")]
pub enum IndexOutcome {
    Committed(IndexReport),
    Paused(IndexRunStatus),
}

#[derive(Debug)]
pub struct IndexPausedError(pub IndexRunStatus);

impl fmt::Display for IndexPausedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "index run {} paused ({})",
            self.0.run_id,
            self.0.pause_reason.as_deref().unwrap_or("unspecified")
        )
    }
}

impl std::error::Error for IndexPausedError {}

#[derive(Debug)]
pub struct IndexRunFailure {
    pub run_id: i64,
    pub source: anyhow::Error,
}

impl fmt::Display for IndexRunFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "index run {} failed: {:#}",
            self.run_id, self.source
        )
    }
}

impl std::error::Error for IndexRunFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.source()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentCheckpointStep {
    Before,
    After,
}

type DocumentFaultHook<'a> = dyn Fn(DocumentCheckpointStep, &str, &str) -> Result<()> + 'a;

#[derive(Serialize)]
struct RunConfig<'a> {
    settings: &'a IndexSettings,
    providers: Vec<(String, String, String)>,
    enrichers: Vec<crate::EnricherIdentity>,
    revision_trust: RevisionTrust,
    refresh_mode: RefreshMode,
    settle_delay_millis: u64,
    retry_backoff: RetryBackoff,
    payload_schema_version: i64,
    frontend_version: &'static str,
}

pub struct IndexRunBuilder<'a, 'provider> {
    db: &'a Db,
    settings: IndexSettings,
    projects: &'a [SourceProject<'provider>],
    enrichers: Vec<&'a dyn RepresentationEnricher>,
    resume_policy: ResumePolicy,
    refresh_policy: RefreshPolicy,
    retry_policy: RetryPolicy,
    revision_trust: RevisionTrust,
    cancellation: CancellationToken,
    progress: Option<&'a dyn Fn(IndexProgress)>,
    document_fault_hook: Option<&'a DocumentFaultHook<'a>>,
    publish_fault_hook: Option<&'a dyn Fn(PublishStep) -> Result<()>>,
    lease_seconds: u64,
}

impl<'a, 'provider> IndexRunBuilder<'a, 'provider> {
    pub fn new(
        db: &'a Db,
        settings: &IndexSettings,
        projects: &'a [SourceProject<'provider>],
    ) -> Self {
        Self {
            db,
            settings: settings.clone(),
            projects,
            enrichers: Vec::new(),
            resume_policy: ResumePolicy::Auto,
            refresh_policy: RefreshPolicy::default(),
            retry_policy: RetryPolicy::default(),
            revision_trust: RevisionTrust::VerifyContent,
            cancellation: CancellationToken::new(),
            progress: None,
            document_fault_hook: None,
            publish_fault_hook: None,
            lease_seconds: 30,
        }
    }

    pub fn with_enrichers(mut self, enrichers: &'a [&'a dyn RepresentationEnricher]) -> Self {
        self.enrichers = enrichers.to_vec();
        self
    }

    pub fn resume_policy(mut self, policy: ResumePolicy) -> Self {
        self.resume_policy = policy;
        self
    }

    pub fn refresh_policy(mut self, policy: RefreshPolicy) -> Self {
        self.refresh_policy = policy;
        self
    }

    pub fn retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    pub fn revision_trust(mut self, trust: RevisionTrust) -> Self {
        self.revision_trust = trust;
        self
    }

    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = cancellation;
        self
    }

    pub fn on_progress(mut self, callback: &'a dyn Fn(IndexProgress)) -> Self {
        self.progress = Some(callback);
        self
    }

    pub fn with_document_fault_hook(mut self, hook: &'a DocumentFaultHook<'a>) -> Self {
        self.document_fault_hook = Some(hook);
        self
    }

    pub fn with_publish_fault_hook(mut self, hook: &'a dyn Fn(PublishStep) -> Result<()>) -> Self {
        self.publish_fault_hook = Some(hook);
        self
    }

    pub fn lease_seconds(mut self, seconds: u64) -> Self {
        self.lease_seconds = seconds.max(1);
        self
    }

    pub fn run(self) -> Result<IndexOutcome> {
        ensure!(
            !self.projects.is_empty(),
            "no source projects were selected"
        );
        let mut labels = HashSet::new();
        for project in self.projects {
            ensure!(!project.label.is_empty(), "project labels cannot be empty");
            ensure!(
                labels.insert(project.label.as_str()),
                "duplicate source project label {:?}",
                project.label
            );
        }
        ensure!(
            self.retry_policy.document_attempts_per_invocation > 0,
            "document retry attempts must be positive"
        );
        let mut providers: Vec<_> = self
            .projects
            .iter()
            .map(|project| {
                (
                    project.label.clone(),
                    project.provider.project_locator(),
                    project.provider.provider_fingerprint(),
                )
            })
            .collect();
        providers.sort();
        let config = RunConfig {
            settings: &self.settings,
            providers,
            enrichers: self
                .enrichers
                .iter()
                .map(|enricher| enricher.identity())
                .collect(),
            revision_trust: self.revision_trust,
            refresh_mode: self.refresh_policy.mode,
            settle_delay_millis: self.refresh_policy.settle_delay.as_millis() as u64,
            retry_backoff: self.refresh_policy.retry_backoff,
            payload_schema_version: STAGED_PAYLOAD_SCHEMA_VERSION,
            frontend_version: codeindex_tree_sitter::FRONTEND_VERSION,
        };
        let config_json = serde_json::to_string(&config)?;
        let config_fingerprint = sha256_hex(&config_json);
        let mut scope: Vec<String> = self
            .projects
            .iter()
            .map(|project| project.label.clone())
            .collect();
        scope.sort();

        let status = match self.resume_policy {
            ResumePolicy::Run(run_id) => {
                let status = self.db.run_status(run_id)?;
                ensure!(
                    status.scope == scope,
                    "run {run_id} has a different project scope"
                );
                ensure!(
                    status.config_fingerprint == config_fingerprint,
                    "run {run_id} has incompatible indexing configuration"
                );
                status
            }
            policy => self.db.create_or_resume_run(&CreateRunSpec {
                scope,
                config_json,
                config_fingerprint,
                payload_schema_version: STAGED_PAYLOAD_SCHEMA_VERSION,
                projects: self
                    .projects
                    .iter()
                    .map(|project| RunProjectSpec {
                        label: project.label.clone(),
                        provider_locator: project.provider.project_locator(),
                        provider_fingerprint: project.provider.provider_fingerprint(),
                    })
                    .collect(),
                force_new: policy == ResumePolicy::New,
            })?,
        };
        if status.state == IndexRunState::Committed {
            return Ok(IndexOutcome::Committed(self.db.publish_run(
                status.run_id,
                "",
                &[],
                None,
            )?));
        }
        let owner_token = format!("{:032x}", rand::random::<u128>());
        self.db
            .claim_run(status.run_id, &owner_token, self.lease_seconds)
            .with_context(|| format!("claiming index run {}", status.run_id))?;
        match self.run_owned(status.run_id, &owner_token) {
            Ok(outcome) => Ok(outcome),
            Err(error) => {
                // Publish failures deliberately leave `ready` intact. Other
                // invariant/storage failures become durable failed runs.
                if self
                    .db
                    .run_status(status.run_id)
                    .is_ok_and(|status| status.state == IndexRunState::Running)
                {
                    let error_json = serde_json::json!({
                        "kind": "indexing_failure",
                        "message": error.to_string(),
                    })
                    .to_string();
                    let _ = self.db.fail_run(status.run_id, &owner_token, &error_json);
                }
                Err(IndexRunFailure {
                    run_id: status.run_id,
                    source: error,
                }
                .into())
            }
        }
    }

    fn run_owned(&self, run_id: i64, owner_token: &str) -> Result<IndexOutcome> {
        self.validate_immutable_settings()?;
        // Constant for the run's lifetime; the per-document fingerprint loop
        // must not re-query run status for it.
        let config_fingerprint = self.config_fingerprint(run_id)?;
        let enabled: HashSet<String> = self.settings.enabled_languages.iter().cloned().collect();
        let projects_by_label: HashMap<&str, &SourceProject<'provider>> = self
            .projects
            .iter()
            .map(|project| (project.label.as_str(), project))
            .collect();
        let mut completed_processing_pass = self
            .db
            .staged_documents(run_id)?
            .iter()
            .any(|document| document.state == DocumentState::Ready);
        let mut refresh_attempt = 0_u32;
        let mut document_attempts: HashMap<(String, String), u32> = HashMap::new();

        loop {
            if self.cancellation.is_cancelled() {
                return self.pause(run_id, owner_token, "user_interrupt", None);
            }
            self.db
                .set_run_phase(run_id, owner_token, IndexRunPhase::Refreshing)?;
            self.emit_progress(run_id, IndexRunPhase::Refreshing, None, None)?;

            let staged_before = self.db.staged_documents(run_id)?;
            let staged_lookup: HashMap<(&str, &str), _> = staged_before
                .iter()
                .map(|document| {
                    (
                        (
                            document.project_label.as_str(),
                            document.source_document_id.as_str(),
                        ),
                        document,
                    )
                })
                .collect();
            let mut refreshes = Vec::new();
            let mut unstable = false;
            for project in self.projects {
                let documents = match project.provider.documents(&enabled) {
                    Ok(documents) => documents,
                    Err(error) => {
                        return self.pause(
                            run_id,
                            owner_token,
                            "provider_error",
                            Some(&error.to_string()),
                        );
                    }
                };
                if let Err(error) = crate::source::validate_documents(&documents) {
                    let error_json = serde_json::json!({"message": error.to_string()}).to_string();
                    self.db.fail_run(run_id, owner_token, &error_json)?;
                    return Err(error);
                }
                let live_files = match self.db.get_project(&project.label)? {
                    Some(live) => self
                        .db
                        .list_files(live.id)?
                        .into_iter()
                        .map(|file| (file.source_document_id.clone(), file))
                        .collect(),
                    None => HashMap::new(),
                };
                let mut observations = Vec::with_capacity(documents.len());
                for document in &documents {
                    if self.cancellation.is_cancelled() {
                        return self.pause(run_id, owner_token, "user_interrupt", None);
                    }
                    let live = live_files.get(&document.id);
                    let prior_staged = staged_lookup
                        .get(&(project.label.as_str(), document.id.as_str()))
                        .copied();
                    let may_trust_revision = project.provider.revision_semantics()
                        == RevisionSemantics::Authoritative
                        || self.revision_trust == RevisionTrust::TrustAdvisory;
                    let trusted_hash = may_trust_revision
                        .then(|| {
                            live.filter(|file| same_live_observation(file, document))
                                .map(|file| file.source_hash.clone())
                                .or_else(|| {
                                    prior_staged
                                        .filter(|staged| {
                                            staged.source_revision_json.as_deref()
                                                == serde_json::to_string(&document.revision)
                                                    .ok()
                                                    .as_deref()
                                        })
                                        .and_then(|staged| staged.observed_source_hash.clone())
                                })
                        })
                        .flatten();
                    let source_hash = if let Some(hash) = trusted_hash {
                        hash
                    } else {
                        match project.provider.stable_read(document) {
                            Ok(StableRead::Content { source, revision })
                                if revision == document.revision =>
                            {
                                sha256_hex(&source)
                            }
                            Ok(StableRead::Content { .. }) | Ok(StableRead::Changed) => {
                                unstable = true;
                                break;
                            }
                            Err(_) => "<read-error>".to_string(),
                        }
                    };
                    let action = action_for(live, document, &source_hash);
                    let fingerprint = document_fingerprint(
                        &config_fingerprint,
                        project.provider.provider_fingerprint().as_str(),
                        document,
                        &source_hash,
                    )?;
                    observations.push(ManifestDocument {
                        source_document_id: document.id.clone(),
                        relative_path: document.relative_path.clone(),
                        language_id: document.language_id.clone(),
                        source_revision_json: serde_json::to_string(&document.revision)?,
                        observed_source_hash: source_hash,
                        input_fingerprint: fingerprint,
                        action,
                    });
                }
                // Live documents the provider no longer reports are deletions.
                // Without these rows the publish transaction has nothing to
                // remove and deleted files silently survive reindexing.
                let observed: HashSet<&str> = documents
                    .iter()
                    .map(|document| document.id.as_str())
                    .collect();
                let mut deleted_ids: Vec<&String> = live_files
                    .keys()
                    .filter(|id| !observed.contains(id.as_str()))
                    .collect();
                deleted_ids.sort();
                for id in deleted_ids {
                    let file = &live_files[id];
                    let fingerprint = sha256_hex(&serde_json::to_string(&serde_json::json!({
                        "config": config_fingerprint,
                        "provider": project.provider.provider_fingerprint(),
                        "id": id,
                        "deleted": true,
                    }))?);
                    observations.push(ManifestDocument {
                        source_document_id: id.clone(),
                        relative_path: file.relative_path.clone(),
                        language_id: file.language_id.clone(),
                        source_revision_json: serde_json::to_string(&file.source_revision)?,
                        observed_source_hash: String::new(),
                        input_fingerprint: fingerprint,
                        action: DocumentAction::Delete,
                    });
                }
                if unstable {
                    break;
                }
                // The manifest digest covers surviving documents only; the
                // publish transaction recomputes it the same way (delete rows
                // are journaled but excluded from the digest).
                let digest = manifest_digest(
                    &observations
                        .iter()
                        .filter(|document| document.action != DocumentAction::Delete)
                        .cloned()
                        .collect::<Vec<_>>(),
                );
                refreshes.push((project.label.clone(), documents, observations, digest));
            }
            if unstable {
                refresh_attempt += 1;
                std::thread::sleep(self.refresh_policy.retry_backoff.delay(refresh_attempt));
                continue;
            }
            refresh_attempt = 0;

            let mut refresh_changed = false;
            let mut latest_documents = HashMap::new();
            for (project_label, documents, observations, digest) in refreshes {
                for document in documents {
                    latest_documents.insert((project_label.clone(), document.id.clone()), document);
                }
                let result = self.db.reconcile_manifest(
                    run_id,
                    owner_token,
                    &project_label,
                    &digest,
                    &observations,
                )?;
                refresh_changed |= result.changed;
            }
            if refresh_changed && completed_processing_pass {
                self.db.increment_refresh_round(run_id, owner_token)?;
                if self.refresh_policy.mode == RefreshMode::PauseOnChange {
                    return self.pause(run_id, owner_token, "source_changed", None);
                }
                std::thread::sleep(self.refresh_policy.settle_delay);
            }

            let pending: Vec<_> = self
                .db
                .staged_documents(run_id)?
                .into_iter()
                .filter(|document| document.state == DocumentState::Pending)
                .collect();
            if pending.is_empty() {
                if refresh_changed || !completed_processing_pass {
                    completed_processing_pass = true;
                    continue;
                }
                if self.cancellation.is_cancelled() {
                    return self.pause(run_id, owner_token, "user_interrupt", None);
                }
                self.db.mark_run_ready(run_id, owner_token)?;
                self.emit_progress(run_id, IndexRunPhase::Ready, None, None)?;
                let immutable = self.immutable_settings();
                let report = self.db.publish_run(
                    run_id,
                    owner_token,
                    &immutable,
                    self.publish_fault_hook,
                )?;
                return Ok(IndexOutcome::Committed(report));
            }

            self.db
                .set_run_phase(run_id, owner_token, IndexRunPhase::Processing)?;
            for staged in pending {
                if self.cancellation.is_cancelled() {
                    return self.pause(run_id, owner_token, "user_interrupt", None);
                }
                let key = (
                    staged.project_label.clone(),
                    staged.source_document_id.clone(),
                );
                let Some(project) = projects_by_label.get(staged.project_label.as_str()) else {
                    anyhow::bail!(
                        "run refers to unconfigured project {:?}",
                        staged.project_label
                    );
                };
                let Some(document) = latest_documents.get(&key) else {
                    // It disappeared after reconciliation; the next refresh
                    // will turn this row into a staged deletion.
                    continue;
                };
                self.db.begin_document(
                    run_id,
                    owner_token,
                    &staged.project_label,
                    &staged.source_document_id,
                )?;
                let read = project.provider.stable_read(document);
                let (source, stable_revision) = match read {
                    Ok(StableRead::Content { source, revision }) => (source, revision),
                    Ok(StableRead::Changed) => {
                        self.db.reset_document_pending(
                            run_id,
                            owner_token,
                            &staged.project_label,
                            &staged.source_document_id,
                        )?;
                        break;
                    }
                    Err(error) => {
                        let attempts = document_attempts.entry(key.clone()).or_default();
                        *attempts += 1;
                        if *attempts < self.retry_policy.document_attempts_per_invocation {
                            self.db.reset_document_pending(
                                run_id,
                                owner_token,
                                &staged.project_label,
                                &staged.source_document_id,
                            )?;
                            std::thread::sleep(self.retry_policy.backoff.delay(*attempts));
                            break;
                        }
                        let error_json = serde_json::json!({
                            "kind": "document_read",
                            "message": error.to_string(),
                        })
                        .to_string();
                        self.db.checkpoint_document_error(
                            run_id,
                            owner_token,
                            &staged.project_label,
                            &staged.source_document_id,
                            &error_json,
                        )?;
                        return self.pause(
                            run_id,
                            owner_token,
                            "document_error",
                            Some(&error_json),
                        );
                    }
                };
                let source_hash = sha256_hex(&source);
                if stable_revision != document.revision
                    || staged.observed_source_hash.as_deref() != Some(source_hash.as_str())
                {
                    self.db.reset_document_pending(
                        run_id,
                        owner_token,
                        &staged.project_label,
                        &staged.source_document_id,
                    )?;
                    break;
                }
                let input_fingerprint = staged
                    .input_fingerprint
                    .as_deref()
                    .context("pending document has no input fingerprint")?;
                let preparation = prepare_document(PrepareDocument {
                    db: self.db,
                    settings: &self.settings,
                    project_label: &staged.project_label,
                    document,
                    source: &source,
                    source_hash: &source_hash,
                    generation: run_id,
                    input_fingerprint,
                    enrichers: &self.enrichers,
                });
                let payload = match preparation {
                    Ok(payload) => payload,
                    Err(error) => {
                        let error_json = serde_json::json!({
                            "kind": "document_processing",
                            "message": error.to_string(),
                        })
                        .to_string();
                        self.db.checkpoint_document_error(
                            run_id,
                            owner_token,
                            &staged.project_label,
                            &staged.source_document_id,
                            &error_json,
                        )?;
                        return self.pause(
                            run_id,
                            owner_token,
                            "document_error",
                            Some(&error_json),
                        );
                    }
                };
                if let Some(hook) = self.document_fault_hook {
                    hook(
                        DocumentCheckpointStep::Before,
                        &staged.project_label,
                        &staged.source_document_id,
                    )?;
                }
                self.db.checkpoint_document(
                    run_id,
                    owner_token,
                    &staged.project_label,
                    &staged.source_document_id,
                    &payload,
                )?;
                if let Some(hook) = self.document_fault_hook {
                    hook(
                        DocumentCheckpointStep::After,
                        &staged.project_label,
                        &staged.source_document_id,
                    )?;
                }
                self.emit_progress(
                    run_id,
                    IndexRunPhase::Processing,
                    Some(staged.project_label),
                    Some(staged.source_document_id),
                )?;
            }
            completed_processing_pass = true;
        }
    }

    fn pause(
        &self,
        run_id: i64,
        owner_token: &str,
        reason: &str,
        error: Option<&str>,
    ) -> Result<IndexOutcome> {
        Ok(IndexOutcome::Paused(self.db.pause_run(
            run_id,
            owner_token,
            reason,
            error,
        )?))
    }

    fn emit_progress(
        &self,
        run_id: i64,
        phase: IndexRunPhase,
        project_label: Option<String>,
        source_document_id: Option<String>,
    ) -> Result<()> {
        if let Some(callback) = self.progress {
            let status = self.db.run_status(run_id)?;
            callback(IndexProgress {
                run_id,
                phase,
                project_label,
                source_document_id,
                ready_documents: status.stats.documents_ready,
                total_documents: status.stats.documents_total,
                refresh_round: status.refresh_round,
            });
        }
        Ok(())
    }

    fn config_fingerprint(&self, run_id: i64) -> Result<String> {
        Ok(self.db.run_status(run_id)?.config_fingerprint)
    }

    /// Settings fixed once a corpus is first published. Enforced inside the
    /// publish transaction and validated against any existing database before
    /// a run starts; both sites must use this single list.
    fn immutable_settings(&self) -> [(&'static str, String); 3] {
        [
            (
                "index.body_node_count_threshold",
                self.settings.body_node_count_threshold.to_string(),
            ),
            (
                "index.retention",
                self.settings.retention.as_str().to_string(),
            ),
            (
                "embedding.max_body_chars",
                self.settings.max_body_chars.to_string(),
            ),
        ]
    }

    fn validate_immutable_settings(&self) -> Result<()> {
        for (key, expected) in self.immutable_settings() {
            if let Some(existing) = self.db.get_setting(key)? {
                ensure!(
                    existing == expected,
                    "setting `{key}` is fixed once the database is published: stored \
                     {existing:?}, config now says {expected:?}. Delete the database file to \
                     reindex with new settings."
                );
            }
        }
        for project in self.projects {
            if let Some(existing) = self.db.get_project(&project.label)? {
                let locator = project.provider.project_locator();
                ensure!(
                    existing.source_dir == locator,
                    "project {:?} is already indexed from {:?}; its source locator cannot \
                     change to {:?}",
                    project.label,
                    existing.source_dir,
                    locator
                );
            }
        }
        Ok(())
    }
}

fn same_live_observation(file: &FileRecord, document: &SourceDocument) -> bool {
    file.source_revision == document.revision.opaque
        && file.relative_path == document.relative_path
        && file.language_id == document.language_id
}

fn action_for(
    live: Option<&FileRecord>,
    document: &SourceDocument,
    source_hash: &str,
) -> DocumentAction {
    let Some(live) = live else {
        return DocumentAction::Upsert;
    };
    if live.relative_path != document.relative_path || live.language_id != document.language_id {
        return DocumentAction::Upsert;
    }
    if live.source_hash != source_hash {
        return DocumentAction::Upsert;
    }
    if live.source_revision != document.revision.opaque
        || live.mtime_ns != document.revision.modified_ns.unwrap_or_default()
        || live.size != document.revision.size.unwrap_or_default() as i64
    {
        DocumentAction::Metadata
    } else {
        DocumentAction::Unchanged
    }
}

fn document_fingerprint(
    config_fingerprint: &str,
    provider_fingerprint: &str,
    document: &SourceDocument,
    source_hash: &str,
) -> Result<String> {
    Ok(sha256_hex(&serde_json::to_string(&serde_json::json!({
        "config": config_fingerprint,
        "provider": provider_fingerprint,
        "id": document.id,
        "path": document.relative_path,
        "language": document.language_id,
        "revision": document.revision,
        "source_hash": source_hash,
    }))?))
}
