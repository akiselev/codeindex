//! Atomic journal-to-corpus publication.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Context, Result, bail, ensure};
use codeindex_core::{RepresentationKind, RepresentationOrigin};
use rusqlite::{OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::index_runs::{DocumentAction, IndexRunState, STAGED_PAYLOAD_SCHEMA_VERSION};
use crate::{Db, StagedDocumentPayload};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishStep {
    Settings,
    Projects,
    DeleteLiveDocuments,
    Metadata,
    InsertDocument,
    Usage,
    Invariants,
    Generation,
    Commit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectIndexReport {
    pub label: String,
    pub indexed: usize,
    pub skipped: usize,
    pub removed: usize,
    pub units: usize,
    pub total_units: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexReport {
    pub run_id: i64,
    pub generation: i64,
    pub refresh_rounds: u64,
    pub reused_documents: usize,
    pub restaged_documents: usize,
    pub projects: Vec<ProjectIndexReport>,
    pub warnings: Vec<String>,
    pub pending_embeddings: BTreeMap<String, u64>,
}

#[derive(Debug, Deserialize)]
struct StoredRevision {
    opaque: String,
    modified_ns: Option<i64>,
    size: Option<u64>,
}

impl Db {
    /// Publish all selected projects in one immediate transaction. The hook is
    /// intended for deterministic crash/failure tests; returning an error at
    /// any point rolls back every live mutation and leaves the run ready.
    pub fn publish_run(
        &self,
        run_id: i64,
        owner_token: &str,
        immutable_settings: &[(&str, String)],
        fault_hook: Option<&dyn Fn(PublishStep) -> Result<()>>,
    ) -> Result<IndexReport> {
        let status = self.run_status(run_id)?;
        if status.state == IndexRunState::Committed {
            return self.committed_report(run_id);
        }
        ensure!(
            status.state == IndexRunState::Ready,
            "index run {run_id} must be ready before publication"
        );

        // Decode and validate potentially large JSON before acquiring the
        // corpus write lock.
        let documents = self.staged_documents(run_id)?;
        ensure!(
            documents.iter().all(|document| {
                document.state == crate::index_runs::DocumentState::Ready
                    && (document.action != DocumentAction::Upsert
                        || document.payload.as_ref().is_some_and(|payload| {
                            payload.payload_schema_version == STAGED_PAYLOAD_SCHEMA_VERSION
                        }))
            }),
            "index run {run_id} has incomplete or incompatible staged documents"
        );
        for document in &documents {
            if let Some(payload) = &document.payload {
                ensure!(
                    document.input_fingerprint.as_deref()
                        == Some(payload.input_fingerprint.as_str()),
                    "staged payload fingerprint does not match its manifest row"
                );
                ensure!(
                    document.observed_source_hash.as_deref() == Some(payload.source_hash.as_str()),
                    "staged payload hash does not match its manifest row"
                );
                ensure!(
                    document.relative_path.as_deref() == Some(payload.file.relative_path.as_str())
                        && document.language_id.as_deref()
                            == Some(payload.file.language_id.as_str()),
                    "staged payload metadata does not match its manifest row"
                );
            }
        }

        let publish_result = (|| -> Result<IndexReport> {
            let transaction =
                rusqlite::Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
            let (state, owner, base_generation, refresh_round): (String, Option<String>, i64, i64) =
                transaction.query_row(
                    "SELECT status, owner_token, base_generation, refresh_round
                 FROM index_runs WHERE id = ?1",
                    [run_id],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )?;
            ensure!(state == "ready", "index run {run_id} is no longer ready");
            ensure!(
                owner.as_deref() == Some(owner_token),
                "index run {run_id} is not owned by this publisher"
            );
            let current_document_count: i64 = transaction.query_row(
                "SELECT COUNT(*) FROM index_run_documents
                 WHERE run_id = ?1 AND state = 'ready'
                   AND (action != 'upsert' OR payload_schema_version = ?2)",
                params![run_id, STAGED_PAYLOAD_SCHEMA_VERSION],
                |row| row.get(0),
            )?;
            ensure!(
                current_document_count == documents.len() as i64,
                "staged documents changed while publication was being prepared"
            );
            let current_generation = transaction
                .query_row(
                    "SELECT value FROM settings WHERE key = 'index.generation'",
                    [],
                    |row| row.get::<_, String>(0),
                )
                .optional()?
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            ensure!(
                current_generation == base_generation,
                "index run {run_id} was based on generation {base_generation}, but generation \
                 {current_generation} is now published"
            );
            let missing_manifests: i64 = transaction.query_row(
                "SELECT COUNT(*) FROM index_run_projects
                 WHERE run_id = ?1 AND (manifest_digest = '' OR last_refresh_at IS NULL)",
                [run_id],
                |row| row.get(0),
            )?;
            ensure!(
                missing_manifests == 0,
                "index run has an unrefreshed project manifest"
            );
            {
                let mut projects = transaction.prepare(
                    "SELECT project_label, manifest_digest FROM index_run_projects
                     WHERE run_id = ?1 ORDER BY project_label",
                )?;
                let rows = projects.query_map([run_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?;
                for row in rows {
                    let (label, expected_digest) = row?;
                    let mut statement = transaction.prepare(
                        "SELECT source_document_id, relative_path, language_id,
                                input_fingerprint, action
                         FROM index_run_documents
                         WHERE run_id = ?1 AND project_label = ?2 AND action != 'delete'",
                    )?;
                    let manifest_rows = statement.query_map(params![run_id, label], |row| {
                        Ok(crate::index_runs::manifest_row_line(
                            &row.get::<_, String>(0)?,
                            &row.get::<_, String>(1)?,
                            &row.get::<_, String>(2)?,
                            &row.get::<_, String>(3)?,
                            &row.get::<_, String>(4)?,
                        ))
                    })?;
                    let manifest: Vec<String> = manifest_rows.collect::<rusqlite::Result<_>>()?;
                    ensure!(
                        crate::index_runs::manifest_digest_from_lines(manifest) == expected_digest,
                        "project {label:?} manifest digest is inconsistent"
                    );
                }
            }

            for (key, expected) in immutable_settings {
                let existing: Option<String> = transaction
                    .query_row("SELECT value FROM settings WHERE key = ?1", [key], |row| {
                        row.get(0)
                    })
                    .optional()?;
                match existing {
                    None => {
                        transaction.execute(
                            "INSERT INTO settings(key, value) VALUES (?1, ?2)",
                            params![key, expected],
                        )?;
                    }
                    Some(existing) if existing == *expected => {}
                    Some(existing) => bail!(
                        "setting `{key}` is fixed once the database is published: stored \
                         {existing:?}, config now says {expected:?}. Delete the database file to \
                         reindex with new settings."
                    ),
                }
            }
            call_hook(fault_hook, PublishStep::Settings)?;

            let mut project_ids = BTreeMap::new();
            {
                let mut statement = transaction.prepare(
                    "SELECT project_label, provider_locator FROM index_run_projects
                     WHERE run_id = ?1 ORDER BY project_label",
                )?;
                let rows = statement.query_map([run_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?;
                for row in rows {
                    let (label, locator) = row?;
                    let existing: Option<(i64, String)> = transaction
                        .query_row(
                            "SELECT id, source_dir FROM projects WHERE label = ?1",
                            [&label],
                            |row| Ok((row.get(0)?, row.get(1)?)),
                        )
                        .optional()?;
                    let project_id = match existing {
                        Some((id, stored)) if stored == locator => id,
                        Some((_, stored)) => bail!(
                            "project {label:?} is already indexed from {stored:?}; its source \
                             locator cannot change to {locator:?}"
                        ),
                        None => {
                            transaction.execute(
                                "INSERT INTO projects(label, source_dir, created_at)
                                 VALUES (?1, ?2, datetime('now'))",
                                params![label, locator],
                            )?;
                            transaction.last_insert_rowid()
                        }
                    };
                    project_ids.insert(label, project_id);
                }
            }
            call_hook(fault_hook, PublishStep::Projects)?;

            // Delete every replacement first. This is required for path swaps:
            // inserting either side while the old rows remain can violate the
            // unique project/path constraint.
            for document in &documents {
                if matches!(
                    document.action,
                    DocumentAction::Upsert | DocumentAction::Delete
                ) {
                    let project_id = project_ids[&document.project_label];
                    transaction.execute(
                        "DELETE FROM files WHERE project_id = ?1 AND source_document_id = ?2",
                        params![project_id, document.source_document_id],
                    )?;
                }
            }
            call_hook(fault_hook, PublishStep::DeleteLiveDocuments)?;

            for document in &documents {
                if document.action != DocumentAction::Metadata {
                    continue;
                }
                let revision: StoredRevision = serde_json::from_str(
                    document
                        .source_revision_json
                        .as_deref()
                        .context("metadata action has no source revision")?,
                )?;
                let project_id = project_ids[&document.project_label];
                let changed = transaction.execute(
                    "UPDATE files SET source_revision = ?3, mtime_ns = ?4, size = ?5
                     WHERE project_id = ?1 AND source_document_id = ?2",
                    params![
                        project_id,
                        document.source_document_id,
                        revision.opaque,
                        revision.modified_ns.unwrap_or_default(),
                        revision.size.unwrap_or_default() as i64,
                    ],
                )?;
                ensure!(
                    changed == 1,
                    "metadata target disappeared before publication"
                );
            }
            call_hook(fault_hook, PublishStep::Metadata)?;

            for document in &documents {
                let Some(payload) = document.payload.as_ref() else {
                    continue;
                };
                validate_payload(document.source_document_id.as_str(), payload)?;
                let project_id = project_ids[&document.project_label];
                let file = &payload.file;
                transaction.execute(
                    "INSERT INTO files(project_id, source_document_id, source_revision,
                       relative_path, language_id, mtime_ns, size, source_hash)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![
                        project_id,
                        file.source_document_id,
                        file.source_revision,
                        file.relative_path,
                        file.language_id,
                        file.mtime_ns,
                        file.size,
                        file.source_hash,
                    ],
                )?;
                let file_id = transaction.last_insert_rowid();
                let unit_ids = insert_units(&transaction, project_id, file_id, &payload.units)?;
                for reference in &payload.references {
                    let caller = *unit_ids
                        .get(reference.caller_unit_ordinal)
                        .context("staged reference has an invalid caller ordinal")?;
                    transaction.execute(
                        "INSERT INTO references_raw(
                           caller_unit_id, callee_symbol, call_snippet, start_line)
                         VALUES (?1, ?2, ?3, ?4)",
                        params![
                            caller,
                            reference.callee_symbol,
                            reference.call_snippet,
                            reference.start_line
                        ],
                    )?;
                }
                call_hook(fault_hook, PublishStep::InsertDocument)?;
            }

            for &project_id in project_ids.values() {
                rebuild_usage(&transaction, project_id)?;
                transaction.execute(
                    "UPDATE projects SET last_index_run_id = ?2 WHERE id = ?1",
                    params![project_id, run_id],
                )?;
                call_hook(fault_hook, PublishStep::Usage)?;
            }
            transaction.execute(
                "DELETE FROM entities WHERE entity_id NOT IN
                   (SELECT DISTINCT entity_id FROM code_units)",
                [],
            )?;

            let dangling: i64 = transaction.query_row(
                "SELECT
                   (SELECT COUNT(*) FROM code_units u LEFT JOIN files f ON f.id = u.file_id
                    WHERE f.id IS NULL) +
                   (SELECT COUNT(*) FROM representations r
                    LEFT JOIN code_units u ON u.id = r.unit_id WHERE u.id IS NULL) +
                   (SELECT COUNT(*) FROM references_raw r
                    LEFT JOIN code_units u ON u.id = r.caller_unit_id WHERE u.id IS NULL)",
                [],
                |row| row.get(0),
            )?;
            ensure!(
                dangling == 0,
                "publication produced dangling live corpus rows"
            );
            call_hook(fault_hook, PublishStep::Invariants)?;

            // Vectors whose content hash no longer exists in any representation
            // of their space's channel are unreachable by snapshot exports;
            // drop them so they cannot accumulate across reindexes. Only runs
            // when documents actually changed — unchanged/metadata publishes
            // cannot create orphans, and the sweep probes the whole table.
            if documents.iter().any(|document| {
                matches!(
                    document.action,
                    DocumentAction::Upsert | DocumentAction::Delete
                )
            }) {
                transaction.execute(
                    "DELETE FROM embeddings
                     WHERE NOT EXISTS (
                       SELECT 1 FROM embedding_spaces s
                       JOIN representations r
                         ON r.kind = s.channel AND r.content_hash = embeddings.content_hash
                       WHERE s.space_id = embeddings.space_id
                     )",
                    [],
                )?;
            }

            transaction.execute(
                "INSERT INTO settings(key, value) VALUES ('index.generation', ?1)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [run_id.to_string()],
            )?;
            call_hook(fault_hook, PublishStep::Generation)?;

            let mut reports = Vec::new();
            for (label, &project_id) in &project_ids {
                let indexed = documents
                    .iter()
                    .filter(|doc| {
                        doc.project_label == *label && doc.action == DocumentAction::Upsert
                    })
                    .count();
                let removed = documents
                    .iter()
                    .filter(|doc| {
                        doc.project_label == *label && doc.action == DocumentAction::Delete
                    })
                    .count();
                let skipped = documents
                    .iter()
                    .filter(|doc| {
                        doc.project_label == *label
                            && matches!(
                                doc.action,
                                DocumentAction::Unchanged | DocumentAction::Metadata
                            )
                    })
                    .count();
                let units: usize = documents
                    .iter()
                    .filter(|doc| doc.project_label == *label)
                    .filter_map(|doc| doc.payload.as_ref())
                    .map(|payload| payload.units.len())
                    .sum();
                let total_units: i64 = transaction.query_row(
                    "SELECT COUNT(*) FROM code_units u JOIN files f ON f.id = u.file_id
                     WHERE f.project_id = ?1",
                    [project_id],
                    |row| row.get(0),
                )?;
                reports.push(ProjectIndexReport {
                    label: label.clone(),
                    indexed,
                    skipped,
                    removed,
                    units,
                    total_units: total_units as usize,
                });
            }
            let report = IndexReport {
                run_id,
                generation: run_id,
                refresh_rounds: refresh_round as u64,
                reused_documents: transaction.query_row(
                    "SELECT COUNT(*) FROM index_run_documents
                     WHERE run_id = ?1 AND reused > 0",
                    [run_id],
                    |row| row.get::<_, i64>(0),
                )? as usize,
                restaged_documents: transaction.query_row(
                    "SELECT COUNT(*) FROM index_run_documents
                     WHERE run_id = ?1 AND attempts > 1",
                    [run_id],
                    |row| row.get::<_, i64>(0),
                )? as usize,
                projects: reports,
                warnings: Vec::new(),
                pending_embeddings: BTreeMap::new(),
            };
            let report_json = serde_json::to_string(&report)?;
            transaction.execute(
                "UPDATE index_runs SET status = 'committed', phase = 'committed',
                    committed_at = datetime('now'), updated_at = datetime('now'),
                    owner_token = NULL, heartbeat_at = NULL, last_error_json = NULL,
                    stats_json = ?2 WHERE id = ?1",
                params![run_id, report_json],
            )?;
            call_hook(fault_hook, PublishStep::Commit)?;
            transaction.commit()?;
            Ok(report)
        })();

        match publish_result {
            Ok(mut report) => {
                report.pending_embeddings = self.pending_embedding_counts()?;
                // Pending-vector diagnostics are post-commit and therefore do
                // not affect validity of the publication.
                let _ = self.conn.execute(
                    "UPDATE index_runs SET stats_json = ?2 WHERE id = ?1 AND status = 'committed'",
                    params![run_id, serde_json::to_string(&report)?],
                );
                Ok(report)
            }
            Err(error) => {
                let error_json = serde_json::json!({"kind":"publish", "message":error.to_string()});
                let _ = self.conn.execute(
                    "UPDATE index_runs SET last_error_json = ?2, updated_at = datetime('now')
                     WHERE id = ?1 AND status = 'ready'",
                    params![run_id, error_json.to_string()],
                );
                Err(error)
            }
        }
    }

    fn committed_report(&self, run_id: i64) -> Result<IndexReport> {
        let report: String = self.conn.query_row(
            "SELECT stats_json FROM index_runs WHERE id = ?1 AND status = 'committed'",
            [run_id],
            |row| row.get(0),
        )?;
        serde_json::from_str(&report).context("committed run has an invalid report")
    }

    fn pending_embedding_counts(&self) -> Result<BTreeMap<String, u64>> {
        let mut counts = BTreeMap::new();
        for space in self.list_spaces()? {
            counts.insert(
                space.identity.id.to_string(),
                self.count_unembedded_hashes(&space.identity.id)? as u64,
            );
        }
        Ok(counts)
    }
}

fn validate_payload(document_id: &str, payload: &StagedDocumentPayload) -> Result<()> {
    ensure!(
        payload.file.source_document_id == document_id,
        "staged payload document identity does not match its journal row"
    );
    ensure!(
        payload.file.source_hash == payload.source_hash,
        "staged source hashes disagree"
    );
    ensure!(
        payload
            .references
            .iter()
            .all(|reference| { reference.caller_unit_ordinal < payload.units.len() }),
        "staged reference caller ordinal is out of range"
    );
    ensure!(
        payload
            .units
            .iter()
            .all(|unit| unit.start_byte < unit.end_byte),
        "staged unit has an invalid byte range"
    );
    Ok(())
}

fn insert_units(
    transaction: &rusqlite::Transaction<'_>,
    project_id: i64,
    file_id: i64,
    units: &[crate::NewCodeUnit],
) -> Result<Vec<i64>> {
    let mut ids = Vec::with_capacity(units.len());
    for unit in units {
        transaction.execute(
            "INSERT INTO entities(entity_id, project_id, kind, first_generation, last_generation)
             VALUES (?1, ?2, ?3, ?4, ?4)
             ON CONFLICT(entity_id) DO UPDATE SET
               last_generation = MAX(last_generation, excluded.last_generation)",
            params![
                unit.entity_id.as_str(),
                project_id,
                unit.kind,
                unit.generation
            ],
        )?;
        transaction.execute(
            "INSERT INTO code_units(
               file_id, entity_id, entity_version_id, generation, language_id, kind, name, scope,
               start_byte, end_byte, start_line, end_line, body_node_count, source_hash,
               normalized_body_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                file_id,
                unit.entity_id.as_str(),
                unit.entity_version_id.as_str(),
                unit.generation,
                unit.language_id,
                unit.kind,
                unit.name,
                unit.scope,
                unit.start_byte as i64,
                unit.end_byte as i64,
                unit.start_line as i64,
                unit.end_line as i64,
                unit.body_node_count as i64,
                unit.source_hash,
                unit.normalized_body_hash,
            ],
        )?;
        let unit_id = transaction.last_insert_rowid();
        for representation in &unit.representations {
            transaction.execute(
                "INSERT INTO representations(unit_id, kind, content_hash, content, origin_json)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    unit_id,
                    representation.kind.as_str(),
                    representation.content_hash,
                    representation.content,
                    serde_json::to_string(&representation.origin)?,
                ],
            )?;
        }
        ids.push(unit_id);
    }
    Ok(ids)
}

fn rebuild_usage(transaction: &rusqlite::Transaction<'_>, project_id: i64) -> Result<()> {
    transaction.execute(
        "DELETE FROM representations WHERE kind = ?1 AND unit_id IN
           (SELECT u.id FROM code_units u JOIN files f ON f.id = u.file_id
            WHERE f.project_id = ?2)",
        params![RepresentationKind::Usage.as_str(), project_id],
    )?;
    let mut definitions: HashMap<String, Vec<i64>> = HashMap::new();
    let mut qualified: HashMap<i64, String> = HashMap::new();
    {
        let mut statement = transaction.prepare(
            "SELECT u.id, u.name, u.scope FROM code_units u
             JOIN files f ON f.id = u.file_id WHERE f.project_id = ?1 ORDER BY u.id",
        )?;
        let rows = statement.query_map([project_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })?;
        for row in rows {
            let (id, name, scope) = row?;
            if let Some(symbol) = symbol_name(&name) {
                definitions.entry(symbol).or_default().push(id);
            }
            qualified.insert(
                id,
                scope.map_or_else(|| name.clone(), |scope| format!("{scope}.{name}")),
            );
        }
    }
    let mut usages: BTreeMap<i64, BTreeSet<String>> = BTreeMap::new();
    {
        let mut statement = transaction.prepare(
            "SELECT r.caller_unit_id, r.callee_symbol, r.call_snippet
             FROM references_raw r JOIN code_units u ON u.id = r.caller_unit_id
             JOIN files f ON f.id = u.file_id WHERE f.project_id = ?1
             ORDER BY r.caller_unit_id, r.start_line",
        )?;
        let rows = statement.query_map([project_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        for row in rows {
            let (caller_id, callee_symbol, snippet) = row?;
            let Some(name) = symbol_name(&callee_symbol) else {
                continue;
            };
            let Some(callee_ids) = definitions.get(&name) else {
                continue;
            };
            let caller = qualified.get(&caller_id).map_or("?", String::as_str);
            for &callee_id in callee_ids {
                if callee_id != caller_id {
                    usages
                        .entry(callee_id)
                        .or_default()
                        .insert(format!("{caller}: {snippet}"));
                }
            }
        }
    }
    let origin = RepresentationOrigin::Derived {
        producer: "codeindex-usage".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };
    for (unit_id, lines) in usages {
        let text = lines.into_iter().collect::<Vec<_>>().join("\n");
        transaction.execute(
            "INSERT INTO representations(unit_id, kind, content_hash, content, origin_json)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                unit_id,
                RepresentationKind::Usage.as_str(),
                sha256_hex(&text),
                text,
                serde_json::to_string(&origin)?,
            ],
        )?;
    }
    Ok(())
}

fn symbol_name(raw: &str) -> Option<String> {
    let trimmed = raw.trim().trim_end_matches('!');
    let mut without_generics = String::with_capacity(trimmed.len());
    let mut depth = 0usize;
    for character in trimmed.chars() {
        match character {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            _ if depth == 0 => without_generics.push(character),
            _ => {}
        }
    }
    let segment = without_generics
        .rsplit(['.', ':'])
        .find(|part| !part.trim().is_empty())
        .unwrap_or("")
        .trim();
    (!segment.is_empty() && segment != "<anonymous>").then(|| segment.to_string())
}

fn sha256_hex(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn call_hook(hook: Option<&dyn Fn(PublishStep) -> Result<()>>, step: PublishStep) -> Result<()> {
    if let Some(hook) = hook {
        hook(step)?;
    }
    Ok(())
}
