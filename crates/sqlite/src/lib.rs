pub mod index_publish;
pub mod index_runs;
pub mod migrations;
pub mod models;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail};
use codeindex_core::{
    EmbeddingSpaceId, EmbeddingSpaceIdentity, EntityId, EntityVersionId, Pooling,
    RepresentationKind,
};
use codeindex_storage::{
    EmbeddingSpaceSnapshot, IndexSnapshot, ProjectRecord, RepresentationRef, UnitRecord,
};
use rusqlite::{Connection, OptionalExtension, params, params_from_iter};

pub use models::{
    CodeUnit, EmbeddingModelRecord, EmbeddingSpaceRecord, ExecutionInfo, FileId, FileRecord,
    ModelContract, ModelId, NewCodeUnit, NewFile, NewRepresentation, Project, ProjectId,
    StagedDocumentPayload, StagedReference, UnitId, blob_to_vector, vector_to_blob,
};

pub use codeindex_storage as storage;

#[derive(Debug)]
pub struct Db {
    conn: Connection,
}

/// Where a representation can be re-derived from its source provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashLocation {
    pub project_label: String,
    pub source_dir: String,
    pub source_document_id: String,
    pub relative_path: String,
    pub language_id: String,
}

pub fn open_or_create(path: &Path) -> Result<Db> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let conn = Connection::open(path)
        .with_context(|| format!("cannot open database {}", path.display()))?;
    Db::from_connection(conn)
}

pub fn open_in_memory() -> Result<Db> {
    Db::from_connection(Connection::open_in_memory()?)
}

impl Db {
    fn from_connection(conn: Connection) -> Result<Db> {
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.pragma_update(None, "journal_mode", "WAL").ok();
        migrations::migrate(&conn)?;
        Ok(Db { conn })
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    // ----- settings -----

    pub fn get_setting(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT value FROM settings WHERE key = ?1", [key], |row| {
                row.get(0)
            })
            .optional()?)
    }

    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO settings(key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [key, value],
        )?;
        Ok(())
    }

    pub fn current_generation(&self) -> Result<i64> {
        Ok(self
            .get_setting("index.generation")?
            .and_then(|value| value.parse().ok())
            .unwrap_or(0))
    }

    // ----- projects -----

    pub fn upsert_project(&self, label: &str, source_dir: &str) -> Result<ProjectId> {
        if let Some(existing) = self.get_project(label)? {
            if existing.source_dir != source_dir {
                bail!(
                    "project {label:?} is already indexed from {:?}; its source locator cannot \
                     change to {source_dir:?}. Delete the database file to reindex.",
                    existing.source_dir
                );
            }
            return Ok(existing.id);
        }
        self.conn.execute(
            "INSERT INTO projects(label, source_dir, created_at) VALUES (?1, ?2, datetime('now'))",
            [label, source_dir],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn get_project(&self, label: &str) -> Result<Option<Project>> {
        Ok(self
            .conn
            .query_row(
                "SELECT id, label, source_dir, last_index_run_id
                 FROM projects WHERE label = ?1",
                [label],
                row_to_project,
            )
            .optional()?)
    }

    pub fn list_projects(&self) -> Result<Vec<Project>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, label, source_dir, last_index_run_id
                 FROM projects ORDER BY label",
        )?;
        Ok(stmt
            .query_map([], row_to_project)?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }

    // ----- source documents -----

    pub fn get_file_by_source_id(
        &self,
        project_id: ProjectId,
        source_document_id: &str,
    ) -> Result<Option<FileRecord>> {
        self.get_file_where(project_id, "source_document_id", source_document_id)
    }

    fn get_file_where(
        &self,
        project_id: ProjectId,
        column: &str,
        value: &str,
    ) -> Result<Option<FileRecord>> {
        let sql = format!(
            "SELECT id, project_id, source_document_id, source_revision, relative_path, \
                    language_id, mtime_ns, size, source_hash
             FROM files WHERE project_id = ?1 AND {column} = ?2"
        );
        Ok(self
            .conn
            .query_row(&sql, params![project_id, value], row_to_file)
            .optional()?)
    }

    pub fn list_files(&self, project_id: ProjectId) -> Result<Vec<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, project_id, source_document_id, source_revision, relative_path,
                    language_id, mtime_ns, size, source_hash
             FROM files WHERE project_id = ?1 ORDER BY relative_path",
        )?;
        Ok(stmt
            .query_map([project_id], row_to_file)?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Insert or update by stable provider document id. Updating a document may
    /// also move its display path; prior units are deleted before reinsertion.
    pub fn upsert_file(&self, file: &NewFile) -> Result<FileId> {
        if let Some(existing) =
            self.get_file_by_source_id(file.project_id, &file.source_document_id)?
        {
            self.conn.execute(
                "UPDATE files SET source_revision = ?1, relative_path = ?2, language_id = ?3,
                                  mtime_ns = ?4, size = ?5, source_hash = ?6
                 WHERE id = ?7",
                params![
                    file.source_revision,
                    file.relative_path,
                    file.language_id,
                    file.mtime_ns,
                    file.size,
                    file.source_hash,
                    existing.id
                ],
            )?;
            self.conn
                .execute("DELETE FROM code_units WHERE file_id = ?1", [existing.id])?;
            return Ok(existing.id);
        }
        self.conn.execute(
            "INSERT INTO files(project_id, source_document_id, source_revision, relative_path,
                               language_id, mtime_ns, size, source_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                file.project_id,
                file.source_document_id,
                file.source_revision,
                file.relative_path,
                file.language_id,
                file.mtime_ns,
                file.size,
                file.source_hash
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    // ----- units and representations -----

    pub fn insert_units(&self, file_id: FileId, units: &[NewCodeUnit]) -> Result<Vec<UnitId>> {
        let project_id: ProjectId = self.conn.query_row(
            "SELECT project_id FROM files WHERE id = ?1",
            [file_id],
            |row| row.get(0),
        )?;
        let mut unit_stmt = self.conn.prepare_cached(
            "INSERT INTO code_units(
               file_id, entity_id, entity_version_id, generation,
               language_id, kind, name, scope,
               start_byte, end_byte, start_line, end_line,
               body_node_count, source_hash, normalized_body_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
        )?;
        let mut entity_stmt = self.conn.prepare_cached(
            "INSERT INTO entities(entity_id, project_id, kind, first_generation, last_generation)
             VALUES (?1, ?2, ?3, ?4, ?4)
             ON CONFLICT(entity_id) DO UPDATE SET
               last_generation = MAX(last_generation, excluded.last_generation)",
        )?;
        let mut repr_stmt = self.conn.prepare_cached(
            "INSERT INTO representations(unit_id, kind, content_hash, content, origin_json)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        let mut ids = Vec::with_capacity(units.len());
        for unit in units {
            entity_stmt.execute(params![
                unit.entity_id.as_str(),
                project_id,
                unit.kind,
                unit.generation,
            ])?;
            unit_stmt.execute(params![
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
            ])?;
            let unit_id = self.conn.last_insert_rowid();
            for repr in &unit.representations {
                repr_stmt.execute(params![
                    unit_id,
                    repr.kind.as_str(),
                    repr.content_hash,
                    repr.content,
                    serde_json::to_string(&repr.origin)?,
                ])?;
            }
            ids.push(unit_id);
        }
        Ok(ids)
    }

    pub fn list_units_for_file(&self, file_id: FileId) -> Result<Vec<CodeUnit>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, file_id, entity_id, entity_version_id, generation,
                    language_id, kind, name, scope, start_byte, end_byte,
                    start_line, end_line, body_node_count, source_hash, normalized_body_hash
             FROM code_units WHERE file_id = ?1 ORDER BY start_byte",
        )?;
        Ok(stmt
            .query_map([file_id], row_to_unit)?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn list_units_for_project(&self, project_id: ProjectId) -> Result<Vec<CodeUnit>> {
        let mut stmt = self.conn.prepare(
            "SELECT u.id, u.file_id, u.entity_id, u.entity_version_id, u.generation,
                    u.language_id, u.kind, u.name, u.scope, u.start_byte, u.end_byte,
                    u.start_line, u.end_line, u.body_node_count, u.source_hash,
                    u.normalized_body_hash
             FROM code_units u JOIN files f ON f.id = u.file_id
             WHERE f.project_id = ?1 ORDER BY u.id",
        )?;
        Ok(stmt
            .query_map([project_id], row_to_unit)?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn count_units(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM code_units", [], |row| row.get(0))?)
    }

    pub fn count_units_for_project(&self, project_id: ProjectId) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM code_units u JOIN files f ON f.id = u.file_id
             WHERE f.project_id = ?1",
            [project_id],
            |row| row.get(0),
        )?)
    }

    /// Raw frontend call sites attributed to units of one file:
    /// `(caller_unit_id, callee_symbol, start_line)`, ordered by line.
    pub fn raw_references_for_file(&self, file_id: FileId) -> Result<Vec<(UnitId, String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT r.caller_unit_id, r.callee_symbol, r.start_line
             FROM references_raw r
             JOIN code_units u ON u.id = r.caller_unit_id
             WHERE u.file_id = ?1
             ORDER BY r.start_line, r.caller_unit_id",
        )?;
        Ok(stmt
            .query_map([file_id], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }

    // ----- relations -----

    /// Replace one producer's relations for a generation in a single
    /// transaction. Post-publish analyzers (LSP, SCIP) call this after
    /// resolving edges against the published corpus.
    pub fn replace_relations(
        &self,
        generation: i64,
        provenance: &str,
        relations: &[storage::RelationRecord],
    ) -> Result<usize> {
        let transaction = rusqlite::Transaction::new_unchecked(
            &self.conn,
            rusqlite::TransactionBehavior::Immediate,
        )?;
        self.conn.execute(
            "DELETE FROM relations WHERE generation = ?1 AND provenance = ?2",
            params![generation, provenance],
        )?;
        let mut inserted = 0;
        {
            let mut stmt = self.conn.prepare_cached(
                "INSERT INTO relations(
                   generation, from_entity_id, to_entity_id, to_symbol, kind,
                   resolution, provenance)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;
            for relation in relations {
                inserted += stmt.execute(params![
                    generation,
                    relation.from_entity_id.as_str(),
                    relation.to_entity_id.as_ref().map(|id| id.as_str()),
                    relation.to_symbol,
                    relation.kind,
                    relation.resolution,
                    relation.provenance,
                ])?;
            }
        }
        transaction.commit()?;
        Ok(inserted)
    }

    /// Every relation recorded for a generation, deterministic order.
    pub fn relations_for_generation(
        &self,
        generation: i64,
    ) -> Result<Vec<storage::RelationRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT from_entity_id, to_entity_id, to_symbol, kind, resolution, provenance
             FROM relations WHERE generation = ?1
             ORDER BY from_entity_id, kind, to_symbol, id",
        )?;
        Ok(stmt
            .query_map([generation], |row| {
                Ok(storage::RelationRecord {
                    from_entity_id: EntityId::new(row.get::<_, String>(0)?),
                    to_entity_id: row.get::<_, Option<String>>(1)?.map(EntityId::new),
                    to_symbol: row.get(2)?,
                    kind: row.get(3)?,
                    resolution: row.get(4)?,
                    provenance: row.get(5)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Insert or replace one derived/imported representation on a live unit.
    /// Post-publish analyzers use this for channels like `typed_signature`;
    /// deterministic extracted channels always come from the staging pipeline.
    pub fn upsert_representation(
        &self,
        unit_id: UnitId,
        representation: &NewRepresentation,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO representations(unit_id, kind, content_hash, content, origin_json)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(unit_id, kind) DO UPDATE SET
               content_hash = excluded.content_hash,
               content = excluded.content,
               origin_json = excluded.origin_json",
            params![
                unit_id,
                representation.kind.as_str(),
                representation.content_hash,
                representation.content,
                serde_json::to_string(&representation.origin)?
            ],
        )?;
        Ok(())
    }

    // ----- models and spaces -----

    pub fn find_or_create_model(&self, contract: &ModelContract) -> Result<ModelId> {
        let prompts_json = serde_json::to_string(&contract.prompts)?;
        let existing: Option<ModelId> = self
            .conn
            .query_row(
                "SELECT id FROM embedding_models
                 WHERE model = ?1 AND revision IS ?2 AND model_hash IS ?3
                   AND tokenizer_hash IS ?4 AND pooling = ?5 AND normalize = ?6
                   AND native_dimensions = ?7 AND max_sequence_length = ?8
                   AND prompts_json = ?9 AND quantization IS ?10",
                params![
                    contract.model,
                    contract.revision,
                    contract.model_hash,
                    contract.tokenizer_hash,
                    contract.pooling.as_str(),
                    contract.normalize as i64,
                    contract.native_dimensions as i64,
                    contract.max_sequence_length as i64,
                    prompts_json,
                    contract.quantization,
                ],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(id) = existing {
            return Ok(id);
        }
        self.conn.execute(
            "INSERT INTO embedding_models(
               model, revision, model_hash, tokenizer_hash, pooling, normalize,
               native_dimensions, max_sequence_length, prompts_json, quantization)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                contract.model,
                contract.revision,
                contract.model_hash,
                contract.tokenizer_hash,
                contract.pooling.as_str(),
                contract.normalize as i64,
                contract.native_dimensions as i64,
                contract.max_sequence_length as i64,
                prompts_json,
                contract.quantization,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Record that a model executed in a given environment. Append-only
    /// provenance: one row per distinct environment, `last_seen_at` refreshed
    /// on repeats.
    pub fn record_model_execution(
        &self,
        model_id: ModelId,
        execution: &ExecutionInfo,
    ) -> Result<()> {
        let existing: Option<i64> = self
            .conn
            .query_row(
                "SELECT id FROM model_executions
                 WHERE model_id = ?1 AND backend = ?2 AND backend_version = ?3
                   AND runtime_version IS ?4 AND execution_provider = ?5 AND cache_path IS ?6",
                params![
                    model_id,
                    execution.backend,
                    execution.backend_version,
                    execution.runtime_version,
                    execution.execution_provider,
                    execution.cache_path,
                ],
                |row| row.get(0),
            )
            .optional()?;
        match existing {
            Some(id) => {
                self.conn.execute(
                    "UPDATE model_executions SET last_seen_at = datetime('now') WHERE id = ?1",
                    [id],
                )?;
            }
            None => {
                self.conn.execute(
                    "INSERT INTO model_executions(
                       model_id, backend, backend_version, runtime_version,
                       execution_provider, cache_path, first_seen_at, last_seen_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, datetime('now'), datetime('now'))",
                    params![
                        model_id,
                        execution.backend,
                        execution.backend_version,
                        execution.runtime_version,
                        execution.execution_provider,
                        execution.cache_path,
                    ],
                )?;
            }
        }
        Ok(())
    }

    /// Every execution environment recorded for a model, oldest first.
    pub fn model_executions(&self, model_id: ModelId) -> Result<Vec<ExecutionInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT backend, backend_version, runtime_version, execution_provider, cache_path
             FROM model_executions WHERE model_id = ?1 ORDER BY id",
        )?;
        Ok(stmt
            .query_map([model_id], |row| {
                Ok(ExecutionInfo {
                    backend: row.get(0)?,
                    backend_version: row.get(1)?,
                    runtime_version: row.get(2)?,
                    execution_provider: row.get(3)?,
                    cache_path: row.get(4)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn list_models(&self) -> Result<Vec<EmbeddingModelRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, model, revision, model_hash, tokenizer_hash, pooling, normalize,
                    native_dimensions, max_sequence_length, prompts_json, quantization
             FROM embedding_models ORDER BY id",
        )?;
        Ok(stmt
            .query_map([], row_to_model)?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn find_or_create_space(
        &self,
        identity: &EmbeddingSpaceIdentity,
    ) -> Result<EmbeddingSpaceId> {
        if let Some(existing) = self.get_space(&identity.id)? {
            if existing.identity != *identity {
                bail!(
                    "embedding space {:?} already exists with a different model, channel, or \
                     document-side contract",
                    identity.id.as_str()
                );
            }
            return Ok(identity.id.clone());
        }
        let model_id = self.find_or_create_model(&identity.model)?;
        self.conn.execute(
            "INSERT INTO embedding_spaces(
               space_id, model_id, channel, document_side_json, created_at)
             VALUES (?1, ?2, ?3, ?4, datetime('now'))",
            params![
                identity.id.as_str(),
                model_id,
                identity.channel.as_str(),
                serde_json::to_string(&identity.document_side)?,
            ],
        )?;
        Ok(identity.id.clone())
    }

    pub fn get_space(&self, id: &EmbeddingSpaceId) -> Result<Option<EmbeddingSpaceRecord>> {
        Ok(self
            .conn
            .query_row(
                &format!("{SPACE_SELECT} WHERE s.space_id = ?1"),
                [id.as_str()],
                row_to_space,
            )
            .optional()?)
    }

    pub fn list_spaces(&self) -> Result<Vec<EmbeddingSpaceRecord>> {
        let mut stmt = self
            .conn
            .prepare(&format!("{SPACE_SELECT} ORDER BY s.space_id"))?;
        Ok(stmt
            .query_map([], row_to_space)?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }

    // ----- embeddings -----

    pub fn embeddable_channels(&self) -> Result<Vec<RepresentationKind>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT kind FROM representations WHERE kind != ?1 ORDER BY kind")?;
        Ok(stmt
            .query_map([RepresentationKind::FullSource.as_str()], |row| {
                Ok(RepresentationKind::from(row.get::<_, String>(0)?.as_str()))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn insert_embedding(
        &self,
        space_id: &EmbeddingSpaceId,
        content_hash: &str,
        vector: &[f32],
    ) -> Result<()> {
        self.insert_embeddings(space_id, &[(content_hash.to_string(), vector.to_vec())])
            .map(|_| ())
    }

    /// Insert one page of vectors in a single transaction. The space is
    /// resolved and dimension-checked once per call; each row is still guarded
    /// against re-introducing an orphan content hash.
    pub fn insert_embeddings(
        &self,
        space_id: &EmbeddingSpaceId,
        vectors: &[(String, Vec<f32>)],
    ) -> Result<usize> {
        if vectors.is_empty() {
            return Ok(0);
        }
        let space = self
            .get_space(space_id)?
            .with_context(|| format!("unknown embedding space {space_id}"))?;
        let transaction = rusqlite::Transaction::new_unchecked(
            &self.conn,
            rusqlite::TransactionBehavior::Immediate,
        )?;
        let mut inserted = 0;
        {
            let mut stmt = self.conn.prepare_cached(
                "INSERT OR IGNORE INTO embeddings(space_id, content_hash, vector_blob, norm, created_at)
                 SELECT ?1, ?2, ?3, ?4, datetime('now')
                 WHERE EXISTS (
                   SELECT 1 FROM embedding_spaces s
                   JOIN representations r ON r.kind = s.channel
                   WHERE s.space_id = ?1 AND r.content_hash = ?2
                 )",
            )?;
            let expected_dimensions = space.identity.effective_dimensions();
            for (content_hash, vector) in vectors {
                anyhow::ensure!(
                    vector.len() == expected_dimensions,
                    "embedding space {space_id} expects {expected_dimensions} dimensions, got {}",
                    vector.len()
                );
                let norm = vector
                    .iter()
                    .map(|value| (*value as f64) * (*value as f64))
                    .sum::<f64>()
                    .sqrt();
                // Rows the orphan guard rejects (or that already exist) count
                // zero changed rows and must not be reported as stored.
                inserted += stmt.execute(params![
                    space_id.as_str(),
                    content_hash,
                    vector_to_blob(vector),
                    norm
                ])?;
            }
        }
        transaction.commit()?;
        Ok(inserted)
    }

    /// Every stored vector of one space as `(content_hash, vector)`,
    /// hash-ordered. Read API for drift gates and export tooling.
    pub fn embeddings_for_space(
        &self,
        space_id: &EmbeddingSpaceId,
    ) -> Result<Vec<(String, Vec<f32>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT content_hash, vector_blob FROM embeddings
             WHERE space_id = ?1 ORDER BY content_hash",
        )?;
        Ok(stmt
            .query_map([space_id.as_str()], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    blob_to_vector(&row.get::<_, Vec<u8>>(1)?),
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn count_embeddings(&self, space_id: &EmbeddingSpaceId) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM embeddings WHERE space_id = ?1",
            [space_id.as_str()],
            |row| row.get(0),
        )?)
    }

    pub fn count_unembedded_hashes(&self, space_id: &EmbeddingSpaceId) -> Result<i64> {
        let space = self
            .get_space(space_id)?
            .with_context(|| format!("unknown embedding space {space_id}"))?;
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM (
               SELECT r.content_hash FROM representations r
               LEFT JOIN embeddings e
                 ON e.content_hash = r.content_hash AND e.space_id = ?1
               WHERE r.kind = ?2 AND e.content_hash IS NULL
               GROUP BY r.content_hash
             )",
            params![space_id.as_str(), space.identity.channel.as_str()],
            |row| row.get(0),
        )?)
    }

    pub fn unembedded_hashes_page(
        &self,
        space_id: &EmbeddingSpaceId,
        after_hash: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(String, Option<String>)>> {
        let space = self
            .get_space(space_id)?
            .with_context(|| format!("unknown embedding space {space_id}"))?;
        let mut stmt = self.conn.prepare(
            "SELECT r.content_hash, MAX(r.content)
             FROM representations r
             LEFT JOIN embeddings e
               ON e.content_hash = r.content_hash AND e.space_id = ?1
             WHERE r.kind = ?2 AND e.content_hash IS NULL
               AND (?3 IS NULL OR r.content_hash > ?3)
             GROUP BY r.content_hash ORDER BY r.content_hash LIMIT ?4",
        )?;
        Ok(stmt
            .query_map(
                params![
                    space_id.as_str(),
                    space.identity.channel.as_str(),
                    after_hash,
                    limit as i64
                ],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn channel_texts(
        &self,
        channel: &RepresentationKind,
    ) -> Result<Vec<(String, String, Option<String>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT r.content_hash, MAX(u.language_id), MAX(r.content)
             FROM representations r JOIN code_units u ON u.id = r.unit_id
             WHERE r.kind = ?1 GROUP BY r.content_hash ORDER BY r.content_hash",
        )?;
        Ok(stmt
            .query_map([channel.as_str()], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn locations_for_content_hashes(
        &self,
        channel: &RepresentationKind,
        hashes: &[String],
    ) -> Result<Vec<HashLocation>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT p.label, p.source_dir, f.source_document_id, f.relative_path, f.language_id
             FROM representations r
             JOIN code_units u ON u.id = r.unit_id
             JOIN files f ON f.id = u.file_id
             JOIN projects p ON p.id = f.project_id
             WHERE r.kind = ?1 AND r.content_hash = ?2 LIMIT 1",
        )?;
        let mut locations = Vec::new();
        let mut seen = HashSet::new();
        for hash in hashes {
            let location = stmt
                .query_row(params![channel.as_str(), hash], |row| {
                    Ok(HashLocation {
                        project_label: row.get(0)?,
                        source_dir: row.get(1)?,
                        source_document_id: row.get(2)?,
                        relative_path: row.get(3)?,
                        language_id: row.get(4)?,
                    })
                })
                .optional()?;
            if let Some(location) = location
                && seen.insert((
                    location.project_label.clone(),
                    location.source_document_id.clone(),
                ))
            {
                locations.push(location);
            }
        }
        Ok(locations)
    }

    // ----- analysis provenance -----

    pub fn create_analysis_run(
        &self,
        analysis_kind: &str,
        model_id: ModelId,
        project_scope: &[String],
        config_json: &str,
    ) -> Result<i64> {
        self.create_analysis_run_inner(analysis_kind, model_id, None, project_scope, config_json)
    }

    pub fn create_analysis_run_in_space(
        &self,
        analysis_kind: &str,
        space_id: &EmbeddingSpaceId,
        project_scope: &[String],
        config_json: &str,
    ) -> Result<i64> {
        let space = self
            .get_space(space_id)?
            .with_context(|| format!("unknown embedding space {space_id}"))?;
        self.create_analysis_run_inner(
            analysis_kind,
            space.model_id,
            Some(space_id.as_str()),
            project_scope,
            config_json,
        )
    }

    fn create_analysis_run_inner(
        &self,
        analysis_kind: &str,
        model_id: ModelId,
        space_id: Option<&str>,
        project_scope: &[String],
        config_json: &str,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO analysis_runs(
               analysis_kind, model_id, space_id, project_scope_json, config_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, datetime('now'))",
            params![
                analysis_kind,
                model_id,
                space_id,
                serde_json::to_string(project_scope)?,
                config_json
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    // ----- storage-neutral snapshot -----

    pub fn snapshot(&self, project_labels: &[String]) -> Result<IndexSnapshot> {
        self.snapshot_with_spaces(project_labels, &[])
    }

    /// Export selected projects and spaces. An empty space list means every
    /// stored space. Vectors not referenced by the selected units are omitted.
    pub fn snapshot_with_spaces(
        &self,
        project_labels: &[String],
        space_ids: &[EmbeddingSpaceId],
    ) -> Result<IndexSnapshot> {
        // A deferred read transaction pins one SQLite snapshot across project,
        // unit, representation, space, and vector queries. Publication may
        // commit concurrently, but this export is entirely old or entirely new.
        let transaction = rusqlite::Transaction::new_unchecked(
            &self.conn,
            rusqlite::TransactionBehavior::Deferred,
        )?;
        let snapshot = self.snapshot_with_spaces_in_transaction(project_labels, space_ids)?;
        transaction.commit()?;
        Ok(snapshot)
    }

    fn snapshot_with_spaces_in_transaction(
        &self,
        project_labels: &[String],
        space_ids: &[EmbeddingSpaceId],
    ) -> Result<IndexSnapshot> {
        let all_projects = self.list_projects()?;
        let projects: Vec<Project> = if project_labels.is_empty() {
            all_projects
        } else {
            let mut selected = Vec::new();
            for label in project_labels {
                let project = all_projects
                    .iter()
                    .find(|project| &project.label == label)
                    .with_context(|| format!("project {label:?} is not indexed"))?;
                selected.push(project.clone());
            }
            selected
        };
        if projects.is_empty() {
            bail!("no indexed projects; index a project first");
        }

        let placeholders = projects.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT u.id, u.entity_id, u.entity_version_id, u.generation,
                    p.label, f.relative_path, u.language_id, u.kind, u.name, u.scope,
                    u.start_byte, u.end_byte, u.start_line, u.end_line, u.body_node_count,
                    u.normalized_body_hash
             FROM code_units u
             JOIN files f ON f.id = u.file_id
             JOIN projects p ON p.id = f.project_id
             WHERE p.id IN ({placeholders})
             ORDER BY p.label, f.relative_path, u.start_byte, u.end_byte"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut unit_rows: Vec<(UnitId, UnitRecord)> = stmt
            .query_map(
                params_from_iter(projects.iter().map(|project| project.id)),
                |row| {
                    Ok((
                        row.get::<_, UnitId>(0)?,
                        UnitRecord {
                            entity_id: EntityId::new(row.get::<_, String>(1)?),
                            entity_version_id: EntityVersionId::new(row.get::<_, String>(2)?),
                            generation: row.get::<_, i64>(3)? as u64,
                            project_label: row.get(4)?,
                            relative_path: row.get(5)?,
                            language_id: row.get(6)?,
                            kind: row.get(7)?,
                            name: row.get(8)?,
                            scope: row.get(9)?,
                            span: codeindex_core::SourceSpan::new(
                                row.get::<_, i64>(10)? as usize,
                                row.get::<_, i64>(11)? as usize,
                                row.get::<_, i64>(12)? as usize,
                                row.get::<_, i64>(13)? as usize,
                            ),
                            body_node_count: row.get::<_, i64>(14)? as usize,
                            normalized_body_hash: row.get(15)?,
                            representations: Vec::new(),
                        },
                    ))
                },
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let unit_index: HashMap<UnitId, usize> = unit_rows
            .iter()
            .enumerate()
            .map(|(index, (id, _))| (*id, index))
            .collect();
        let repr_sql = format!(
            "SELECT r.unit_id, r.kind, r.content_hash, r.content, r.origin_json
             FROM representations r
             JOIN code_units u ON u.id = r.unit_id
             JOIN files f ON f.id = u.file_id
             WHERE f.project_id IN ({placeholders})"
        );
        let mut repr_stmt = self.conn.prepare(&repr_sql)?;
        let mut rows =
            repr_stmt.query(params_from_iter(projects.iter().map(|project| project.id)))?;
        while let Some(row) = rows.next()? {
            let unit_id: UnitId = row.get(0)?;
            if let Some(&index) = unit_index.get(&unit_id) {
                let kind: String = row.get(1)?;
                let origin_json: String = row.get(4)?;
                unit_rows[index].1.representations.push(RepresentationRef {
                    kind: RepresentationKind::from(kind.as_str()),
                    content_hash: row.get(2)?,
                    content: row.get(3)?,
                    origin: serde_json::from_str(&origin_json)?,
                });
            }
        }
        for (_, unit) in &mut unit_rows {
            unit.representations
                .sort_by(|left, right| left.kind.cmp(&right.kind));
        }
        let units: Vec<UnitRecord> = unit_rows.into_iter().map(|(_, unit)| unit).collect();

        let selected_spaces: Vec<EmbeddingSpaceRecord> = if space_ids.is_empty() {
            self.list_spaces()?
        } else {
            let mut spaces = Vec::new();
            for id in space_ids {
                spaces.push(
                    self.get_space(id)?
                        .with_context(|| format!("embedding space {id} is not stored"))?,
                );
            }
            spaces
        };

        let mut hashes_by_channel: BTreeMap<RepresentationKind, BTreeSet<String>> = BTreeMap::new();
        for unit in &units {
            for repr in &unit.representations {
                hashes_by_channel
                    .entry(repr.kind.clone())
                    .or_default()
                    .insert(repr.content_hash.clone());
            }
        }

        let mut spaces = Vec::with_capacity(selected_spaces.len());
        for space in selected_spaces {
            let relevant = hashes_by_channel
                .get(&space.identity.channel)
                .cloned()
                .unwrap_or_default();
            let mut vectors = Vec::new();
            let mut stmt = self.conn.prepare(
                "SELECT content_hash, vector_blob FROM embeddings
                 WHERE space_id = ?1 ORDER BY content_hash",
            )?;
            let rows = stmt.query_map([space.identity.id.as_str()], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
            })?;
            for row in rows {
                let (hash, blob) = row?;
                if relevant.contains(&hash) {
                    vectors.push((hash, blob_to_vector(&blob)));
                }
            }
            spaces.push(EmbeddingSpaceSnapshot {
                identity: space.identity,
                vectors,
            });
        }

        let published_generation = self.current_generation()?;
        Ok(IndexSnapshot {
            published_generation: published_generation as u64,
            relations: self.relations_for_generation(published_generation)?,
            projects: projects
                .into_iter()
                .map(|project| ProjectRecord {
                    label: project.label,
                    source_dir: project.source_dir,
                    last_index_run_id: project.last_index_run_id.map(|id| id as u64),
                })
                .collect(),
            units,
            spaces,
        })
    }
}

fn row_to_project(row: &rusqlite::Row<'_>) -> rusqlite::Result<Project> {
    Ok(Project {
        id: row.get(0)?,
        label: row.get(1)?,
        source_dir: row.get(2)?,
        last_index_run_id: row.get(3)?,
    })
}

fn row_to_file(row: &rusqlite::Row<'_>) -> rusqlite::Result<FileRecord> {
    Ok(FileRecord {
        id: row.get(0)?,
        project_id: row.get(1)?,
        source_document_id: row.get(2)?,
        source_revision: row.get(3)?,
        relative_path: row.get(4)?,
        language_id: row.get(5)?,
        mtime_ns: row.get(6)?,
        size: row.get(7)?,
        source_hash: row.get(8)?,
    })
}

fn row_to_unit(row: &rusqlite::Row<'_>) -> rusqlite::Result<CodeUnit> {
    Ok(CodeUnit {
        id: row.get(0)?,
        file_id: row.get(1)?,
        entity_id: EntityId::new(row.get::<_, String>(2)?),
        entity_version_id: EntityVersionId::new(row.get::<_, String>(3)?),
        generation: row.get(4)?,
        language_id: row.get(5)?,
        kind: row.get(6)?,
        name: row.get(7)?,
        scope: row.get(8)?,
        start_byte: row.get::<_, i64>(9)? as usize,
        end_byte: row.get::<_, i64>(10)? as usize,
        start_line: row.get::<_, i64>(11)? as usize,
        end_line: row.get::<_, i64>(12)? as usize,
        body_node_count: row.get::<_, i64>(13)? as usize,
        source_hash: row.get(14)?,
        normalized_body_hash: row.get(15)?,
    })
}

/// Shared select list for spaces joined with their semantic model contract.
const SPACE_SELECT: &str = "SELECT s.space_id, s.model_id, s.channel, s.document_side_json,
        m.model, m.revision, m.model_hash, m.tokenizer_hash, m.pooling, m.normalize,
        m.native_dimensions, m.max_sequence_length, m.prompts_json, m.quantization
 FROM embedding_spaces s JOIN embedding_models m ON m.id = s.model_id";

/// Decode a JSON column, surfacing malformed persisted data as a conversion
/// error instead of a panic.
fn json_column<T: serde::de::DeserializeOwned>(index: usize, value: &str) -> rusqlite::Result<T> {
    serde_json::from_str(value).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(index, rusqlite::types::Type::Text, error.into())
    })
}

fn pooling_column(index: usize, value: &str) -> rusqlite::Result<Pooling> {
    Pooling::parse(value).ok_or_else(|| {
        rusqlite::Error::FromSqlConversionFailure(
            index,
            rusqlite::types::Type::Text,
            format!("unknown pooling {value:?}").into(),
        )
    })
}

fn model_contract_from_row(
    row: &rusqlite::Row<'_>,
    offset: usize,
) -> rusqlite::Result<ModelContract> {
    Ok(ModelContract {
        model: row.get(offset)?,
        revision: row.get(offset + 1)?,
        model_hash: row.get(offset + 2)?,
        tokenizer_hash: row.get(offset + 3)?,
        pooling: pooling_column(offset + 4, &row.get::<_, String>(offset + 4)?)?,
        normalize: row.get::<_, i64>(offset + 5)? != 0,
        native_dimensions: row.get::<_, i64>(offset + 6)? as usize,
        max_sequence_length: row.get::<_, i64>(offset + 7)? as usize,
        prompts: json_column(offset + 8, &row.get::<_, String>(offset + 8)?)?,
        quantization: row.get(offset + 9)?,
    })
}

fn row_to_model(row: &rusqlite::Row<'_>) -> rusqlite::Result<EmbeddingModelRecord> {
    Ok(EmbeddingModelRecord {
        id: row.get(0)?,
        contract: model_contract_from_row(row, 1)?,
    })
}

fn row_to_space(row: &rusqlite::Row<'_>) -> rusqlite::Result<EmbeddingSpaceRecord> {
    Ok(EmbeddingSpaceRecord {
        identity: EmbeddingSpaceIdentity {
            id: EmbeddingSpaceId::new(row.get::<_, String>(0)?),
            channel: RepresentationKind::from(row.get::<_, String>(2)?.as_str()),
            document_side: json_column(3, &row.get::<_, String>(3)?)?,
            model: model_contract_from_row(row, 4)?,
        },
        model_id: row.get(1)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(name: &str, dimensions: usize) -> ModelContract {
        ModelContract {
            model: name.into(),
            revision: None,
            model_hash: None,
            tokenizer_hash: None,
            pooling: Pooling::Mean,
            normalize: true,
            native_dimensions: dimensions,
            max_sequence_length: 512,
            prompts: codeindex_core::PromptContract::Symmetric,
            quantization: None,
        }
    }

    #[test]
    fn distinct_embedding_spaces_can_use_different_models() {
        let db = open_in_memory().unwrap();
        let code = EmbeddingSpaceIdentity::new(
            "code",
            RepresentationKind::Implementation,
            model("code", 2),
        );
        let docs = EmbeddingSpaceIdentity::new(
            "docs",
            RepresentationKind::Documentation,
            model("text", 3),
        );
        db.find_or_create_space(&code).unwrap();
        db.find_or_create_space(&docs).unwrap();
        assert_eq!(db.list_spaces().unwrap().len(), 2);
        assert!(db.find_or_create_space(&code).is_ok());
        let conflicting = EmbeddingSpaceIdentity::new(
            "code",
            RepresentationKind::Implementation,
            model("other", 2),
        );
        assert!(db.find_or_create_space(&conflicting).is_err());
    }

    #[test]
    fn old_schema_epoch_is_rejected() {
        let conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "user_version", 1).unwrap();
        let error = Db::from_connection(conn).unwrap_err().to_string();
        assert!(error.contains("schema version 1"));
    }

    #[test]
    fn stale_embedding_insert_cannot_reintroduce_an_orphan_vector() {
        let db = open_in_memory().unwrap();
        let space = EmbeddingSpaceIdentity::new(
            "code",
            RepresentationKind::Implementation,
            model("code", 2),
        );
        db.find_or_create_space(&space).unwrap();
        db.conn
            .execute(
                "INSERT INTO projects(label, source_dir, created_at)
                 VALUES ('main', 'memory://main', datetime('now'))",
                [],
            )
            .unwrap();
        let project_id = db.conn.last_insert_rowid();
        db.conn
            .execute(
                "INSERT INTO files(project_id, source_document_id, source_revision,
                   relative_path, language_id, mtime_ns, size, source_hash)
                 VALUES (?1, 'lib', 'r1', 'lib.rs', 'rust', 0, 1, 'source')",
                [project_id],
            )
            .unwrap();
        let file_id = db.conn.last_insert_rowid();
        db.conn
            .execute(
                "INSERT INTO entities(entity_id, project_id, kind, first_generation, last_generation)
                 VALUES ('entity', ?1, 'function', 1, 1)",
                [project_id],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO code_units(
                   file_id, entity_id, entity_version_id, generation, language_id, kind,
                   name, start_byte, end_byte, start_line, end_line, body_node_count,
                   source_hash, normalized_body_hash)
                 VALUES (?1, 'entity', 'version', 1, 'rust', 'function', 'f',
                         0, 1, 1, 1, 1, 'source', 'body')",
                [file_id],
            )
            .unwrap();
        let unit_id = db.conn.last_insert_rowid();
        db.conn
            .execute(
                "INSERT INTO representations(unit_id, kind, content_hash, content, origin_json)
                 VALUES (?1, 'implementation', 'content', 'x', '{}')",
                [unit_id],
            )
            .unwrap();
        // Simulate an embedder that discovered `content` immediately before a
        // publication or GC removed the last representation using that hash.
        db.conn
            .execute("DELETE FROM representations WHERE unit_id = ?1", [unit_id])
            .unwrap();
        db.insert_embedding(&space.id, "content", &[1.0, 0.0])
            .unwrap();
        assert_eq!(db.count_embeddings(&space.id).unwrap(), 0);
    }
}
