//! Durable operational journal for atomic indexing runs.
//!
//! Nothing in this module is search-visible. Short transactions checkpoint
//! manifests and document payloads; [`crate::index_publish`] is the only bridge
//! from this journal to the live corpus.

use std::collections::{BTreeSet, HashMap};
use std::fmt;

use anyhow::{Context, Result, bail, ensure};
use rusqlite::{OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};

use crate::{Db, StagedDocumentPayload};

pub const STAGED_PAYLOAD_SCHEMA_VERSION: i64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexRunState {
    Planning,
    Running,
    Ready,
    Committed,
    Paused,
    Failed,
    Superseded,
    Abandoned,
}

impl IndexRunState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Planning => "planning",
            Self::Running => "running",
            Self::Ready => "ready",
            Self::Committed => "committed",
            Self::Paused => "paused",
            Self::Failed => "failed",
            Self::Superseded => "superseded",
            Self::Abandoned => "abandoned",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        Ok(match value {
            "planning" => Self::Planning,
            "running" => Self::Running,
            "ready" => Self::Ready,
            "committed" => Self::Committed,
            "paused" => Self::Paused,
            "failed" => Self::Failed,
            "superseded" => Self::Superseded,
            "abandoned" => Self::Abandoned,
            _ => bail!("unknown index run state {value:?}"),
        })
    }
}

impl fmt::Display for IndexRunState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexRunPhase {
    Planning,
    Refreshing,
    Processing,
    Ready,
    Committed,
    Paused,
    Failed,
}

impl IndexRunPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Planning => "planning",
            Self::Refreshing => "refreshing",
            Self::Processing => "processing",
            Self::Ready => "ready",
            Self::Committed => "committed",
            Self::Paused => "paused",
            Self::Failed => "failed",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        Ok(match value {
            "planning" => Self::Planning,
            "refreshing" => Self::Refreshing,
            "processing" => Self::Processing,
            "ready" => Self::Ready,
            "committed" => Self::Committed,
            "paused" => Self::Paused,
            "failed" => Self::Failed,
            _ => bail!("unknown index run phase {value:?}"),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentAction {
    Unchanged,
    Metadata,
    Upsert,
    Delete,
}

impl DocumentAction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unchanged => "unchanged",
            Self::Metadata => "metadata",
            Self::Upsert => "upsert",
            Self::Delete => "delete",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        Ok(match value {
            "unchanged" => Self::Unchanged,
            "metadata" => Self::Metadata,
            "upsert" => Self::Upsert,
            "delete" => Self::Delete,
            _ => bail!("unknown staged document action {value:?}"),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentState {
    Pending,
    Processing,
    Ready,
    Error,
}

impl DocumentState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Processing => "processing",
            Self::Ready => "ready",
            Self::Error => "error",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        Ok(match value {
            "pending" => Self::Pending,
            "processing" => Self::Processing,
            "ready" => Self::Ready,
            "error" => Self::Error,
            _ => bail!("unknown staged document state {value:?}"),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexRunStatus {
    pub run_id: i64,
    pub base_generation: i64,
    pub state: IndexRunState,
    pub phase: IndexRunPhase,
    pub pause_reason: Option<String>,
    pub scope: Vec<String>,
    pub config_fingerprint: String,
    pub payload_schema_version: i64,
    pub refresh_round: u64,
    pub owner_token: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub committed_at: Option<String>,
    pub last_error_json: Option<String>,
    pub stats: IndexRunStats,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexRunStats {
    pub documents_total: u64,
    pub documents_ready: u64,
    pub documents_pending: u64,
    pub documents_error: u64,
    pub documents_reused: u64,
    pub documents_restaged: u64,
}

#[derive(Debug, Clone)]
pub struct RunProjectSpec {
    pub label: String,
    pub provider_locator: String,
    pub provider_fingerprint: String,
}

#[derive(Debug, Clone)]
pub struct CreateRunSpec {
    pub scope: Vec<String>,
    pub config_json: String,
    pub config_fingerprint: String,
    pub payload_schema_version: i64,
    pub projects: Vec<RunProjectSpec>,
    pub force_new: bool,
}

#[derive(Debug, Clone)]
pub struct ManifestDocument {
    pub source_document_id: String,
    pub relative_path: String,
    pub language_id: String,
    pub source_revision_json: String,
    pub observed_source_hash: String,
    pub input_fingerprint: String,
    pub action: DocumentAction,
}

/// One manifest row's digest line. A project's manifest digest is the sha256
/// of its sorted row lines joined with `\n`; `publish_run` recomputes the same
/// digest from the journal to verify consistency, so this row format must
/// never drift between the refresh and publish sides.
pub fn manifest_row_line(
    source_document_id: &str,
    relative_path: &str,
    language_id: &str,
    input_fingerprint: &str,
    action: &str,
) -> String {
    format!("{source_document_id}\0{relative_path}\0{language_id}\0{input_fingerprint}\0{action}")
}

/// Digest of a complete manifest given its row lines, in any order.
pub fn manifest_digest_from_lines(mut lines: Vec<String>) -> String {
    lines.sort();
    use sha2::Digest as _;
    sha2::Sha256::digest(lines.join("\n").as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Digest of a refreshed manifest observation set.
pub fn manifest_digest(documents: &[ManifestDocument]) -> String {
    manifest_digest_from_lines(
        documents
            .iter()
            .map(|document| {
                manifest_row_line(
                    &document.source_document_id,
                    &document.relative_path,
                    &document.language_id,
                    &document.input_fingerprint,
                    document.action.as_str(),
                )
            })
            .collect(),
    )
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReconcileResult {
    pub changed: bool,
    pub invalidated: usize,
    pub reused: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedDocumentRecord {
    pub project_label: String,
    pub source_document_id: String,
    pub relative_path: Option<String>,
    pub language_id: Option<String>,
    pub source_revision_json: Option<String>,
    pub observed_source_hash: Option<String>,
    pub action: DocumentAction,
    pub state: DocumentState,
    pub input_fingerprint: Option<String>,
    pub attempts: u64,
    pub payload: Option<StagedDocumentPayload>,
    pub error_json: Option<String>,
}

impl Db {
    /// Reuse the newest exactly compatible unfinished run, or create a new one.
    /// Incompatible unfinished runs whose scope overlaps this request are
    /// superseded atomically.
    pub fn create_or_resume_run(&self, spec: &CreateRunSpec) -> Result<IndexRunStatus> {
        ensure!(!spec.scope.is_empty(), "an index run scope cannot be empty");
        let requested: BTreeSet<&str> = spec.scope.iter().map(String::as_str).collect();
        ensure!(
            requested.len() == spec.scope.len(),
            "an index run scope contains duplicate project labels"
        );
        let transaction =
            rusqlite::Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        let current_generation = transaction
            .query_row(
                "SELECT value FROM settings WHERE key = 'index.generation'",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        let mut candidates = Vec::new();
        {
            let mut statement = transaction.prepare(
                "SELECT id, scope_json, config_fingerprint, payload_schema_version, status,
                        base_generation
                 FROM index_runs
                 WHERE status IN ('planning','running','ready','paused','failed')
                 ORDER BY id DESC",
            )?;
            let rows = statement.query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, i64>(5)?,
                ))
            })?;
            for row in rows {
                candidates.push(row?);
            }
        }

        let mut reusable = None;
        for (id, scope_json, fingerprint, payload_version, state, base_generation) in &candidates {
            let scope: Vec<String> = serde_json::from_str(scope_json)?;
            let candidate: BTreeSet<&str> = scope.iter().map(String::as_str).collect();
            let overlaps = requested.iter().any(|label| candidate.contains(label));
            if !overlaps {
                continue;
            }
            if !spec.force_new
                && reusable.is_none()
                && candidate == requested
                && fingerprint == &spec.config_fingerprint
                && *payload_version == spec.payload_schema_version
                && state != "failed"
                && *base_generation == current_generation
            {
                reusable = Some(*id);
            } else {
                transaction.execute(
                    "UPDATE index_runs
                     SET status = 'superseded', owner_token = NULL, heartbeat_at = NULL,
                         updated_at = datetime('now') WHERE id = ?1",
                    [id],
                )?;
            }
        }

        let run_id = if let Some(id) = reusable {
            id
        } else {
            transaction.execute(
                "INSERT INTO index_runs(
                   base_generation, status, phase, scope_json, config_json,
                   config_fingerprint, payload_schema_version, created_at, updated_at, stats_json)
                 VALUES (?1, 'planning', 'planning', ?2, ?3, ?4, ?5,
                         datetime('now'), datetime('now'), '{}')",
                params![
                    current_generation,
                    serde_json::to_string(&spec.scope)?,
                    spec.config_json,
                    spec.config_fingerprint,
                    spec.payload_schema_version,
                ],
            )?;
            let id = transaction.last_insert_rowid();
            for project in &spec.projects {
                transaction.execute(
                    "INSERT INTO index_run_projects(
                       run_id, project_label, provider_locator, provider_fingerprint)
                     VALUES (?1, ?2, ?3, ?4)",
                    params![
                        id,
                        project.label,
                        project.provider_locator,
                        project.provider_fingerprint
                    ],
                )?;
            }
            id
        };
        transaction.commit()?;
        self.run_status(run_id)
    }

    /// Claim a resumable run. A stale `running` owner is first recorded as a
    /// process loss so the interruption is visible in durable history.
    pub fn claim_run(&self, run_id: i64, owner_token: &str, lease_seconds: u64) -> Result<()> {
        let transaction =
            rusqlite::Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        let (state, current_owner, stale): (String, Option<String>, bool) = transaction
            .query_row(
                "SELECT status, owner_token,
                        heartbeat_at IS NULL OR
                        heartbeat_at <= datetime('now', '-' || ?2 || ' seconds')
                 FROM index_runs WHERE id = ?1",
                params![run_id, lease_seconds as i64],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .with_context(|| format!("unknown index run {run_id}"))?;
        let state = IndexRunState::parse(&state)?;
        if state == IndexRunState::Running
            && current_owner.as_deref() != Some(owner_token)
            && !stale
        {
            bail!("index run {run_id} is owned by an active writer");
        }
        ensure!(
            matches!(
                state,
                IndexRunState::Planning
                    | IndexRunState::Running
                    | IndexRunState::Ready
                    | IndexRunState::Paused
            ),
            "index run {run_id} in state {state} cannot be claimed"
        );
        if state == IndexRunState::Running && stale {
            transaction.execute(
                "UPDATE index_runs SET status = 'paused', phase = 'paused',
                    pause_reason = 'process_lost',
                    last_error_json = '{\"kind\":\"process_lost\"}',
                    updated_at = datetime('now') WHERE id = ?1",
                [run_id],
            )?;
        }
        transaction.execute(
            "UPDATE index_run_documents SET state = 'pending', error_json = NULL,
                    updated_at = datetime('now')
             WHERE run_id = ?1 AND state IN ('processing','error')",
            [run_id],
        )?;
        transaction.execute(
            "UPDATE index_run_documents SET reused = reused + 1
             WHERE run_id = ?1 AND action = 'upsert' AND state = 'ready'
               AND payload_json IS NOT NULL",
            [run_id],
        )?;
        transaction.execute(
            "UPDATE index_runs SET status = 'running',
                    phase = CASE WHEN phase = 'ready' THEN 'ready' ELSE 'refreshing' END,
                    pause_reason = NULL, owner_token = ?2, heartbeat_at = datetime('now'),
                    updated_at = datetime('now') WHERE id = ?1",
            params![run_id, owner_token],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn heartbeat_run(&self, run_id: i64, owner_token: &str) -> Result<()> {
        self.owned_update(
            run_id,
            owner_token,
            "UPDATE index_runs SET heartbeat_at = datetime('now'), updated_at = datetime('now')
             WHERE id = ?1 AND owner_token = ?2 AND status = 'running'",
        )
    }

    pub fn set_run_phase(
        &self,
        run_id: i64,
        owner_token: &str,
        phase: IndexRunPhase,
    ) -> Result<()> {
        let changed = self.conn.execute(
            "UPDATE index_runs SET phase = ?3, heartbeat_at = datetime('now'),
                    updated_at = datetime('now')
             WHERE id = ?1 AND owner_token = ?2 AND status = 'running'",
            params![run_id, owner_token, phase.as_str()],
        )?;
        ensure!(
            changed == 1,
            "index run {run_id} is not owned by this writer"
        );
        Ok(())
    }

    pub fn increment_refresh_round(&self, run_id: i64, owner_token: &str) -> Result<()> {
        let changed = self.conn.execute(
            "UPDATE index_runs SET refresh_round = refresh_round + 1,
                    heartbeat_at = datetime('now'), updated_at = datetime('now')
             WHERE id = ?1 AND owner_token = ?2 AND status = 'running'",
            params![run_id, owner_token],
        )?;
        ensure!(
            changed == 1,
            "index run {run_id} is not owned by this writer"
        );
        Ok(())
    }

    pub fn reconcile_manifest(
        &self,
        run_id: i64,
        owner_token: &str,
        project_label: &str,
        manifest_digest: &str,
        documents: &[ManifestDocument],
    ) -> Result<ReconcileResult> {
        let transaction =
            rusqlite::Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        ensure_owner(&transaction, run_id, owner_token)?;
        let old_digest: String = transaction.query_row(
            "SELECT manifest_digest FROM index_run_projects
             WHERE run_id = ?1 AND project_label = ?2",
            params![run_id, project_label],
            |row| row.get(0),
        )?;
        let mut existing = HashMap::new();
        {
            let mut statement = transaction.prepare(
                "SELECT source_document_id, relative_path, language_id, action, state,
                        input_fingerprint FROM index_run_documents
                 WHERE run_id = ?1 AND project_label = ?2",
            )?;
            let rows = statement.query_map(params![run_id, project_label], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, Option<String>>(5)?,
                ))
            })?;
            for row in rows {
                let (id, path, language, action, state, fingerprint) = row?;
                existing.insert(
                    id,
                    (
                        path,
                        language,
                        DocumentAction::parse(&action)?,
                        DocumentState::parse(&state)?,
                        fingerprint,
                    ),
                );
            }
        }

        let mut result = ReconcileResult {
            changed: old_digest != manifest_digest,
            ..ReconcileResult::default()
        };
        let mut seen = BTreeSet::new();
        for document in documents {
            seen.insert(document.source_document_id.as_str());
            let keep = existing.get(&document.source_document_id).is_some_and(
                |(path, language, action, state, fingerprint)| {
                    path.as_deref() == Some(document.relative_path.as_str())
                        && language.as_deref() == Some(document.language_id.as_str())
                        && *action == document.action
                        && *state != DocumentState::Processing
                        && fingerprint.as_deref() == Some(document.input_fingerprint.as_str())
                },
            );
            if keep {
                result.reused += 1;
                continue;
            }
            if existing.contains_key(&document.source_document_id) {
                result.invalidated += 1;
            }
            result.changed = true;
            let initial_state = if document.action == DocumentAction::Upsert {
                DocumentState::Pending
            } else {
                DocumentState::Ready
            };
            transaction.execute(
                "INSERT INTO index_run_documents(
                   run_id, project_label, source_document_id, relative_path, language_id,
                   source_revision_json, observed_source_hash, action, state,
                   input_fingerprint, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, datetime('now'))
                 ON CONFLICT(run_id, project_label, source_document_id) DO UPDATE SET
                   relative_path = excluded.relative_path,
                   language_id = excluded.language_id,
                   source_revision_json = excluded.source_revision_json,
                   observed_source_hash = excluded.observed_source_hash,
                   action = excluded.action,
                   state = excluded.state,
                   input_fingerprint = excluded.input_fingerprint,
                   payload_schema_version = NULL, payload_json = NULL, error_json = NULL,
                   updated_at = excluded.updated_at",
                params![
                    run_id,
                    project_label,
                    document.source_document_id,
                    document.relative_path,
                    document.language_id,
                    document.source_revision_json,
                    document.observed_source_hash,
                    document.action.as_str(),
                    initial_state.as_str(),
                    document.input_fingerprint,
                ],
            )?;
        }
        for (document_id, (_, _, action, state, _)) in existing {
            if seen.contains(document_id.as_str()) {
                continue;
            }
            if action != DocumentAction::Delete || state != DocumentState::Ready {
                result.changed = true;
                result.invalidated += 1;
                transaction.execute(
                    "UPDATE index_run_documents
                     SET relative_path = NULL, language_id = NULL,
                         source_revision_json = NULL, observed_source_hash = NULL,
                         action = 'delete', state = 'ready', input_fingerprint = NULL,
                         payload_schema_version = NULL, payload_json = NULL, error_json = NULL,
                         updated_at = datetime('now')
                     WHERE run_id = ?1 AND project_label = ?2 AND source_document_id = ?3",
                    params![run_id, project_label, document_id],
                )?;
            }
        }
        transaction.execute(
            "UPDATE index_run_projects SET manifest_digest = ?3,
                    last_refresh_at = datetime('now')
             WHERE run_id = ?1 AND project_label = ?2",
            params![run_id, project_label, manifest_digest],
        )?;
        transaction.execute(
            "UPDATE index_runs SET phase = 'refreshing',
                    heartbeat_at = datetime('now'), updated_at = datetime('now')
             WHERE id = ?1 AND owner_token = ?2",
            params![run_id, owner_token],
        )?;
        transaction.commit()?;
        Ok(result)
    }

    pub fn staged_documents(&self, run_id: i64) -> Result<Vec<StagedDocumentRecord>> {
        let mut statement = self.conn.prepare(
            "SELECT project_label, source_document_id, relative_path, language_id,
                    source_revision_json, observed_source_hash, action, state,
                    input_fingerprint, attempts, payload_json, error_json
             FROM index_run_documents WHERE run_id = ?1
             ORDER BY project_label, source_document_id",
        )?;
        let rows = statement.query_map([run_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, Option<String>>(8)?,
                row.get::<_, i64>(9)?,
                row.get::<_, Option<String>>(10)?,
                row.get::<_, Option<String>>(11)?,
            ))
        })?;
        let mut documents = Vec::new();
        for row in rows {
            let (
                project_label,
                source_document_id,
                relative_path,
                language_id,
                source_revision_json,
                observed_source_hash,
                action,
                state,
                input_fingerprint,
                attempts,
                payload_json,
                error_json,
            ) = row?;
            documents.push(StagedDocumentRecord {
                project_label,
                source_document_id,
                relative_path,
                language_id,
                source_revision_json,
                observed_source_hash,
                action: DocumentAction::parse(&action)?,
                state: DocumentState::parse(&state)?,
                input_fingerprint,
                attempts: attempts as u64,
                payload: payload_json
                    .map(|json| serde_json::from_str(&json))
                    .transpose()?,
                error_json,
            });
        }
        Ok(documents)
    }

    pub fn begin_document(
        &self,
        run_id: i64,
        owner_token: &str,
        project_label: &str,
        document_id: &str,
    ) -> Result<()> {
        let changed = self.conn.execute(
            "UPDATE index_run_documents SET state = 'processing', attempts = attempts + 1,
                    error_json = NULL, updated_at = datetime('now')
             WHERE run_id = ?1 AND project_label = ?2 AND source_document_id = ?3
               AND state = 'pending' AND EXISTS(
                 SELECT 1 FROM index_runs r WHERE r.id = ?1 AND r.owner_token = ?4
                   AND r.status = 'running')",
            params![run_id, project_label, document_id, owner_token],
        )?;
        ensure!(
            changed == 1,
            "document checkpoint rejected: run is not owned or pending"
        );
        Ok(())
    }

    pub fn checkpoint_document(
        &self,
        run_id: i64,
        owner_token: &str,
        project_label: &str,
        document_id: &str,
        payload: &StagedDocumentPayload,
    ) -> Result<()> {
        ensure!(
            payload.payload_schema_version == STAGED_PAYLOAD_SCHEMA_VERSION,
            "unsupported staged payload version {}",
            payload.payload_schema_version
        );
        let changed = self.conn.execute(
            "UPDATE index_run_documents SET state = 'ready',
                    observed_source_hash = ?5, input_fingerprint = ?6,
                    payload_schema_version = ?7, payload_json = ?8, error_json = NULL,
                    updated_at = datetime('now')
             WHERE run_id = ?1 AND project_label = ?2 AND source_document_id = ?3
               AND state = 'processing' AND action = 'upsert' AND EXISTS(
                 SELECT 1 FROM index_runs r WHERE r.id = ?1 AND r.owner_token = ?4
                   AND r.status = 'running')",
            params![
                run_id,
                project_label,
                document_id,
                owner_token,
                payload.source_hash,
                payload.input_fingerprint,
                payload.payload_schema_version,
                serde_json::to_string(payload)?,
            ],
        )?;
        ensure!(
            changed == 1,
            "document checkpoint rejected: run is not owned or processing"
        );
        Ok(())
    }

    pub fn checkpoint_document_error(
        &self,
        run_id: i64,
        owner_token: &str,
        project_label: &str,
        document_id: &str,
        error_json: &str,
    ) -> Result<()> {
        let changed = self.conn.execute(
            "UPDATE index_run_documents SET state = 'error', error_json = ?5,
                    updated_at = datetime('now')
             WHERE run_id = ?1 AND project_label = ?2 AND source_document_id = ?3
               AND state = 'processing' AND EXISTS(
                 SELECT 1 FROM index_runs r WHERE r.id = ?1 AND r.owner_token = ?4
                   AND r.status = 'running')",
            params![run_id, project_label, document_id, owner_token, error_json],
        )?;
        ensure!(changed == 1, "document error checkpoint rejected");
        Ok(())
    }

    pub fn reset_document_pending(
        &self,
        run_id: i64,
        owner_token: &str,
        project_label: &str,
        document_id: &str,
    ) -> Result<()> {
        let changed = self.conn.execute(
            "UPDATE index_run_documents SET state = 'pending', error_json = NULL,
                    updated_at = datetime('now')
             WHERE run_id = ?1 AND project_label = ?2 AND source_document_id = ?3
               AND state = 'processing' AND EXISTS(
                 SELECT 1 FROM index_runs r WHERE r.id = ?1 AND r.owner_token = ?4
                   AND r.status = 'running')",
            params![run_id, project_label, document_id, owner_token],
        )?;
        ensure!(changed == 1, "document reset rejected");
        Ok(())
    }

    pub fn mark_run_ready(&self, run_id: i64, owner_token: &str) -> Result<()> {
        let transaction =
            rusqlite::Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        ensure_owner(&transaction, run_id, owner_token)?;
        let incomplete: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM index_run_documents
             WHERE run_id = ?1 AND state != 'ready'",
            [run_id],
            |row| row.get(0),
        )?;
        ensure!(
            incomplete == 0,
            "run {run_id} still has {incomplete} incomplete documents"
        );
        transaction.execute(
            "UPDATE index_runs SET status = 'ready', phase = 'ready',
                    heartbeat_at = datetime('now'), updated_at = datetime('now')
             WHERE id = ?1 AND owner_token = ?2",
            params![run_id, owner_token],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn pause_run(
        &self,
        run_id: i64,
        owner_token: &str,
        reason: &str,
        error_json: Option<&str>,
    ) -> Result<IndexRunStatus> {
        let changed = self.conn.execute(
            "UPDATE index_runs SET status = 'paused', phase = 'paused', pause_reason = ?3,
                    last_error_json = ?4, owner_token = NULL, heartbeat_at = NULL,
                    updated_at = datetime('now')
             WHERE id = ?1 AND owner_token = ?2 AND status IN ('running','ready')",
            params![run_id, owner_token, reason, error_json],
        )?;
        ensure!(
            changed == 1,
            "pause rejected: index run {run_id} is not owned"
        );
        self.run_status(run_id)
    }

    pub fn fail_run(
        &self,
        run_id: i64,
        owner_token: &str,
        error_json: &str,
    ) -> Result<IndexRunStatus> {
        let changed = self.conn.execute(
            "UPDATE index_runs SET status = 'failed', phase = 'failed',
                    last_error_json = ?3, owner_token = NULL, heartbeat_at = NULL,
                    updated_at = datetime('now')
             WHERE id = ?1 AND owner_token = ?2 AND status IN ('running','ready')",
            params![run_id, owner_token, error_json],
        )?;
        ensure!(
            changed == 1,
            "fail rejected: index run {run_id} is not owned"
        );
        self.run_status(run_id)
    }

    pub fn abandon_run(&self, run_id: i64) -> Result<IndexRunStatus> {
        let changed = self.conn.execute(
            "UPDATE index_runs SET status = 'abandoned', owner_token = NULL,
                    heartbeat_at = NULL, updated_at = datetime('now')
             WHERE id = ?1 AND status IN ('planning','running','ready','paused','failed')",
            [run_id],
        )?;
        ensure!(
            changed == 1,
            "index run {run_id} cannot be abandoned from its current state"
        );
        self.run_status(run_id)
    }

    pub fn supersede_run(&self, run_id: i64) -> Result<IndexRunStatus> {
        let changed = self.conn.execute(
            "UPDATE index_runs SET status = 'superseded', owner_token = NULL,
                    heartbeat_at = NULL, updated_at = datetime('now')
             WHERE id = ?1 AND status IN ('planning','running','ready','paused','failed')",
            [run_id],
        )?;
        ensure!(changed == 1, "index run {run_id} cannot be superseded");
        self.run_status(run_id)
    }

    pub fn run_status(&self, run_id: i64) -> Result<IndexRunStatus> {
        let status = self
            .conn
            .query_row(
                "SELECT id, base_generation, status, phase, pause_reason, scope_json,
                        config_fingerprint, payload_schema_version, refresh_round,
                        owner_token, created_at, updated_at, committed_at,
                        last_error_json
                 FROM index_runs WHERE id = ?1",
                [run_id],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, Option<String>>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, String>(6)?,
                        row.get::<_, i64>(7)?,
                        row.get::<_, i64>(8)?,
                        row.get::<_, Option<String>>(9)?,
                        row.get::<_, String>(10)?,
                        row.get::<_, String>(11)?,
                        row.get::<_, Option<String>>(12)?,
                        row.get::<_, Option<String>>(13)?,
                    ))
                },
            )
            .with_context(|| format!("unknown index run {run_id}"))?;
        let stats = self.document_stats(run_id)?;
        Ok(IndexRunStatus {
            run_id: status.0,
            base_generation: status.1,
            state: IndexRunState::parse(&status.2)?,
            phase: IndexRunPhase::parse(&status.3)?,
            pause_reason: status.4,
            scope: serde_json::from_str(&status.5)?,
            config_fingerprint: status.6,
            payload_schema_version: status.7,
            refresh_round: status.8 as u64,
            owner_token: status.9,
            created_at: status.10,
            updated_at: status.11,
            committed_at: status.12,
            last_error_json: status.13,
            stats,
        })
    }

    pub fn list_run_statuses(&self, limit: usize) -> Result<Vec<IndexRunStatus>> {
        let mut statement = self
            .conn
            .prepare("SELECT id FROM index_runs ORDER BY id DESC LIMIT ?1")?;
        let ids = statement
            .query_map([limit.max(1) as i64], |row| row.get::<_, i64>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        ids.into_iter().map(|id| self.run_status(id)).collect()
    }

    pub fn gc_staged_runs(&self, keep_newest: usize) -> Result<usize> {
        Ok(self.conn.execute(
            "DELETE FROM index_runs WHERE id IN (
               SELECT id FROM index_runs
               WHERE status IN ('committed','superseded','abandoned')
               ORDER BY id DESC LIMIT -1 OFFSET ?1)",
            [keep_newest as i64],
        )?)
    }

    fn document_stats(&self, run_id: i64) -> Result<IndexRunStats> {
        self.conn
            .query_row(
                "SELECT COUNT(*),
                    SUM(state = 'ready'), SUM(state = 'pending' OR state = 'processing'),
                    SUM(state = 'error'), SUM(reused > 0), SUM(attempts > 1)
             FROM index_run_documents WHERE run_id = ?1",
                [run_id],
                |row| {
                    Ok(IndexRunStats {
                        documents_total: row.get::<_, i64>(0)? as u64,
                        documents_ready: row.get::<_, Option<i64>>(1)?.unwrap_or(0) as u64,
                        documents_pending: row.get::<_, Option<i64>>(2)?.unwrap_or(0) as u64,
                        documents_error: row.get::<_, Option<i64>>(3)?.unwrap_or(0) as u64,
                        documents_reused: row.get::<_, Option<i64>>(4)?.unwrap_or(0) as u64,
                        documents_restaged: row.get::<_, Option<i64>>(5)?.unwrap_or(0) as u64,
                    })
                },
            )
            .map_err(Into::into)
    }

    fn owned_update(&self, run_id: i64, owner_token: &str, sql: &str) -> Result<()> {
        let changed = self.conn.execute(sql, params![run_id, owner_token])?;
        ensure!(
            changed == 1,
            "index run {run_id} is not owned by this writer"
        );
        Ok(())
    }
}

fn ensure_owner(
    transaction: &rusqlite::Transaction<'_>,
    run_id: i64,
    owner_token: &str,
) -> Result<()> {
    let owned: bool = transaction.query_row(
        "SELECT status = 'running' AND owner_token = ?2 FROM index_runs WHERE id = ?1",
        params![run_id, owner_token],
        |row| row.get(0),
    )?;
    ensure!(owned, "index run {run_id} is not owned by this writer");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> CreateRunSpec {
        CreateRunSpec {
            scope: vec!["main".into()],
            config_json: "{}".into(),
            config_fingerprint: "config".into(),
            payload_schema_version: STAGED_PAYLOAD_SCHEMA_VERSION,
            projects: vec![RunProjectSpec {
                label: "main".into(),
                provider_locator: "memory://main".into(),
                provider_fingerprint: "provider".into(),
            }],
            force_new: false,
        }
    }

    #[test]
    fn wrong_owner_cannot_checkpoint_or_pause() {
        let db = crate::open_in_memory().unwrap();
        let run = db.create_or_resume_run(&spec()).unwrap();
        db.claim_run(run.run_id, "owner", 30).unwrap();
        assert!(db.heartbeat_run(run.run_id, "intruder").is_err());
        assert!(
            db.pause_run(run.run_id, "intruder", "user_interrupt", None)
                .is_err()
        );
        assert_eq!(
            db.run_status(run.run_id).unwrap().owner_token.as_deref(),
            Some("owner")
        );
    }

    #[test]
    fn malformed_ready_upsert_is_rejected_by_schema() {
        let db = crate::open_in_memory().unwrap();
        let run = db.create_or_resume_run(&spec()).unwrap();
        let result = db.conn().execute(
            "INSERT INTO index_run_documents(
               run_id, project_label, source_document_id, action, state, updated_at)
             VALUES (?1, 'main', 'lib.rs', 'upsert', 'ready', datetime('now'))",
            [run.run_id],
        );
        assert!(result.is_err());
        assert_eq!(db.current_generation().unwrap(), 0);
    }

    #[test]
    fn stale_owner_takeover_records_process_loss() {
        let db = crate::open_in_memory().unwrap();
        let run = db.create_or_resume_run(&spec()).unwrap();
        db.claim_run(run.run_id, "dead-owner", 30).unwrap();
        db.conn()
            .execute(
                "UPDATE index_runs SET heartbeat_at = datetime('now', '-1 hour') WHERE id = ?1",
                [run.run_id],
            )
            .unwrap();
        db.claim_run(run.run_id, "new-owner", 30).unwrap();
        let status = db.run_status(run.run_id).unwrap();
        assert_eq!(status.owner_token.as_deref(), Some("new-owner"));
        assert!(
            status
                .last_error_json
                .as_deref()
                .unwrap()
                .contains("process_lost")
        );
    }
}
