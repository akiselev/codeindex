pub mod migrations;
pub mod models;

use std::path::Path;

use anyhow::{Context, Result, bail};
use codeindex_core::RepresentationKind;
use codeindex_storage::{
    ChannelEmbeddings, IndexSnapshot, ProjectRecord, RepresentationRef, UnitRecord,
};
use rusqlite::{Connection, OptionalExtension, params, params_from_iter};

pub use models::{
    CodeUnit, EmbeddingModelRecord, FileId, FileRecord, ModelId, ModelIdentity, NewCodeUnit,
    NewFile, NewRepresentation, Project, ProjectId, UnitId, blob_to_vector, vector_to_blob,
};

/// Re-export so consumers can name the snapshot types the store produces.
pub use codeindex_storage as storage;

pub struct Db {
    conn: Connection,
}

/// Where a channel's content hash can be found on disk, for re-deriving text
/// that retention did not store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashLocation {
    pub source_dir: String,
    pub relative_path: String,
    pub language_id: String,
}

/// A staged call site to persist. Mirrors the frontend's `RawReference` without
/// depending on the parser crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewReference {
    pub caller_unit_id: UnitId,
    pub callee_symbol: String,
    pub call_snippet: String,
    pub start_line: i64,
}

/// Open (or create) the database file and bring the schema up to date.
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

/// An in-memory database for tests.
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

    pub fn transaction(&mut self) -> Result<rusqlite::Transaction<'_>> {
        Ok(self.conn.transaction()?)
    }

    // ----- settings / immutable checks -----

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

    /// Record `value` for `key` on first use; on later runs, fail if the
    /// stored value differs. Used for settings that must not change once the
    /// database has content (project roots, body threshold, embedding model
    /// identity, normalization).
    pub fn check_or_set_immutable(&self, key: &str, value: &str) -> Result<()> {
        match self.get_setting(key)? {
            None => self.set_setting(key, value),
            Some(existing) if existing == value => Ok(()),
            Some(existing) => bail!(
                "setting `{key}` is fixed once the database is created: \
                 stored {existing:?}, config now says {value:?}. \
                 Delete the database file to reindex with new settings."
            ),
        }
    }

    /// The current index generation (0 before the first run).
    pub fn current_generation(&self) -> Result<i64> {
        Ok(self
            .get_setting("index.generation")?
            .and_then(|value| value.parse().ok())
            .unwrap_or(0))
    }

    /// Increment and return the index generation. Called once per index run so
    /// every unit written in the run shares one generation number.
    pub fn bump_generation(&self) -> Result<i64> {
        let next = self.current_generation()? + 1;
        self.set_setting("index.generation", &next.to_string())?;
        Ok(next)
    }

    // ----- projects -----

    /// Insert the project or return the existing row. A label that exists
    /// with a different source root is an error (roots are immutable).
    pub fn upsert_project(&self, label: &str, source_dir: &str) -> Result<ProjectId> {
        if let Some(existing) = self.get_project(label)? {
            if existing.source_dir != source_dir {
                bail!(
                    "project {label:?} is already indexed from {:?}; \
                     its source root cannot change to {source_dir:?}. \
                     Delete the database file to reindex.",
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
                "SELECT id, label, source_dir, role FROM projects WHERE label = ?1",
                [label],
                row_to_project,
            )
            .optional()?)
    }

    pub fn list_projects(&self) -> Result<Vec<Project>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, label, source_dir, role FROM projects ORDER BY label")?;
        let projects = stmt
            .query_map([], row_to_project)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(projects)
    }

    pub fn delete_project(&self, label: &str) -> Result<bool> {
        let deleted = self
            .conn
            .execute("DELETE FROM projects WHERE label = ?1", [label])?;
        Ok(deleted > 0)
    }

    // ----- files -----

    pub fn get_file(
        &self,
        project_id: ProjectId,
        relative_path: &str,
    ) -> Result<Option<FileRecord>> {
        Ok(self
            .conn
            .query_row(
                "SELECT id, project_id, relative_path, language_id, mtime_ns, size, source_hash
                 FROM files WHERE project_id = ?1 AND relative_path = ?2",
                params![project_id, relative_path],
                row_to_file,
            )
            .optional()?)
    }

    pub fn list_files(&self, project_id: ProjectId) -> Result<Vec<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, project_id, relative_path, language_id, mtime_ns, size, source_hash
             FROM files WHERE project_id = ?1 ORDER BY relative_path",
        )?;
        let files = stmt
            .query_map([project_id], row_to_file)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(files)
    }

    /// Insert or update a file row and return its id. On update, existing
    /// code units for the file are deleted (cascading to representations and
    /// references) so the caller can reinsert them.
    pub fn upsert_file(&self, file: &NewFile) -> Result<FileId> {
        if let Some(existing) = self.get_file(file.project_id, &file.relative_path)? {
            self.conn.execute(
                "UPDATE files SET language_id = ?1, mtime_ns = ?2, size = ?3, source_hash = ?4
                 WHERE id = ?5",
                params![
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
            "INSERT INTO files(project_id, relative_path, language_id, mtime_ns, size, source_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                file.project_id,
                file.relative_path,
                file.language_id,
                file.mtime_ns,
                file.size,
                file.source_hash
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Refresh mtime/size without touching the file's code units. Used when
    /// a file was touched but its content hash is unchanged.
    pub fn update_file_meta(&self, file_id: FileId, mtime_ns: i64, size: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE files SET mtime_ns = ?1, size = ?2 WHERE id = ?3",
            params![mtime_ns, size, file_id],
        )?;
        Ok(())
    }

    pub fn delete_file(&self, file_id: FileId) -> Result<()> {
        self.conn
            .execute("DELETE FROM files WHERE id = ?1", [file_id])?;
        Ok(())
    }

    // ----- code units, entities, representations -----

    /// Insert units for a file, creating/refreshing their entity-ledger rows
    /// and writing every representation channel. Returns the new unit ids in
    /// input order (aligned with `units`) so the caller can attribute
    /// references to the enclosing unit.
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
            "INSERT INTO representations(unit_id, kind, content_hash, content)
             VALUES (?1, ?2, ?3, ?4)",
        )?;
        let mut ids = Vec::with_capacity(units.len());
        for unit in units {
            entity_stmt.execute(params![
                unit.entity_id,
                project_id,
                unit.kind,
                unit.generation,
            ])?;
            unit_stmt.execute(params![
                file_id,
                unit.entity_id,
                unit.entity_version_id,
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
                ])?;
            }
            ids.push(unit_id);
        }
        Ok(ids)
    }

    pub fn list_units_for_file(&self, file_id: FileId) -> Result<Vec<CodeUnit>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, file_id, entity_id, entity_version_id, generation,
                    language_id, kind, name, scope,
                    start_byte, end_byte, start_line, end_line,
                    body_node_count, source_hash, normalized_body_hash
             FROM code_units WHERE file_id = ?1 ORDER BY start_byte",
        )?;
        let units = stmt
            .query_map([file_id], row_to_unit)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(units)
    }

    /// Every unit in a project, for the whole-corpus Usage resolution pass.
    pub fn list_units_for_project(&self, project_id: ProjectId) -> Result<Vec<CodeUnit>> {
        let mut stmt = self.conn.prepare(
            "SELECT u.id, u.file_id, u.entity_id, u.entity_version_id, u.generation,
                    u.language_id, u.kind, u.name, u.scope,
                    u.start_byte, u.end_byte, u.start_line, u.end_line,
                    u.body_node_count, u.source_hash, u.normalized_body_hash
             FROM code_units u
             JOIN files f ON f.id = u.file_id
             WHERE f.project_id = ?1
             ORDER BY u.id",
        )?;
        let units = stmt
            .query_map([project_id], row_to_unit)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(units)
    }

    pub fn count_units(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM code_units", [], |row| row.get(0))?)
    }

    pub fn count_units_for_project(&self, project_id: ProjectId) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*)
             FROM code_units u
             JOIN files f ON f.id = u.file_id
             WHERE f.project_id = ?1",
            [project_id],
            |row| row.get(0),
        )?)
    }

    /// Set (insert or replace) one representation channel for a unit. Used by
    /// the Usage pass to attach the synthesized `Usage` channel after units are
    /// already stored.
    pub fn set_representation(
        &self,
        unit_id: UnitId,
        kind: &RepresentationKind,
        content_hash: &str,
        content: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO representations(unit_id, kind, content_hash, content)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(unit_id, kind) DO UPDATE SET
               content_hash = excluded.content_hash, content = excluded.content",
            params![unit_id, kind.as_str(), content_hash, content],
        )?;
        Ok(())
    }

    /// Delete every representation of one channel across a project. Used to
    /// clear a derived channel (e.g. `Usage`) before the whole-corpus pass
    /// recomputes it, so units that lost all their inputs do not keep stale
    /// content.
    pub fn clear_channel_for_project(
        &self,
        project_id: ProjectId,
        kind: &RepresentationKind,
    ) -> Result<()> {
        self.conn.execute(
            "DELETE FROM representations WHERE kind = ?1 AND unit_id IN
               (SELECT u.id FROM code_units u JOIN files f ON f.id = u.file_id
                WHERE f.project_id = ?2)",
            params![kind.as_str(), project_id],
        )?;
        Ok(())
    }

    /// Remove entity-ledger rows that no longer back any code unit.
    pub fn prune_orphan_entities(&self) -> Result<usize> {
        Ok(self.conn.execute(
            "DELETE FROM entities WHERE entity_id NOT IN
               (SELECT DISTINCT entity_id FROM code_units)",
            [],
        )?)
    }

    // ----- references -----

    pub fn insert_references(&self, references: &[NewReference]) -> Result<()> {
        let mut stmt = self.conn.prepare_cached(
            "INSERT INTO references_raw(caller_unit_id, callee_symbol, call_snippet, start_line)
             VALUES (?1, ?2, ?3, ?4)",
        )?;
        for reference in references {
            stmt.execute(params![
                reference.caller_unit_id,
                reference.callee_symbol,
                reference.call_snippet,
                reference.start_line,
            ])?;
        }
        Ok(())
    }

    /// All staged call sites in a project, with the caller unit's location, for
    /// the Usage resolution pass. Returns
    /// `(caller_unit_id, caller_name, caller_scope, callee_symbol, call_snippet, start_line)`.
    #[allow(clippy::type_complexity)]
    pub fn references_for_project(
        &self,
        project_id: ProjectId,
    ) -> Result<Vec<(UnitId, String, Option<String>, String, String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT r.caller_unit_id, u.name, u.scope, r.callee_symbol, r.call_snippet, r.start_line
             FROM references_raw r
             JOIN code_units u ON u.id = r.caller_unit_id
             JOIN files f ON f.id = u.file_id
             WHERE f.project_id = ?1
             ORDER BY r.caller_unit_id, r.start_line",
        )?;
        let rows = stmt
            .query_map([project_id], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    // ----- embedding models -----

    /// Find the model row matching this identity, or insert one.
    pub fn find_or_create_model(&self, identity: &ModelIdentity) -> Result<ModelId> {
        let existing: Option<ModelId> = self
            .conn
            .query_row(
                "SELECT id FROM embedding_models
                 WHERE backend = ?1 AND backend_version = ?2 AND model = ?3
                   AND revision IS ?4 AND dimensions = ?5
                   AND tokenizer_hash IS ?6 AND model_hash IS ?7
                   AND normalize = ?8 AND execution_provider = ?9 AND quantization IS ?10",
                params![
                    identity.backend,
                    identity.backend_version,
                    identity.model,
                    identity.revision,
                    identity.dimensions as i64,
                    identity.tokenizer_hash,
                    identity.model_hash,
                    identity.normalize as i64,
                    identity.execution_provider,
                    identity.quantization,
                ],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(id) = existing {
            return Ok(id);
        }
        self.conn.execute(
            "INSERT INTO embedding_models(
               backend, backend_version, runtime_version, model, revision, dimensions,
               tokenizer_hash, model_hash, normalize, execution_provider, quantization, cache_path)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                identity.backend,
                identity.backend_version,
                identity.runtime_version,
                identity.model,
                identity.revision,
                identity.dimensions as i64,
                identity.tokenizer_hash,
                identity.model_hash,
                identity.normalize as i64,
                identity.execution_provider,
                identity.quantization,
                identity.cache_path,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn get_model(&self, id: ModelId) -> Result<Option<EmbeddingModelRecord>> {
        Ok(self
            .conn
            .query_row(
                "SELECT id, backend, backend_version, runtime_version, model, revision,
                        dimensions, tokenizer_hash, model_hash, normalize,
                        execution_provider, quantization, cache_path
                 FROM embedding_models WHERE id = ?1",
                [id],
                row_to_model,
            )
            .optional()?)
    }

    pub fn list_models(&self) -> Result<Vec<EmbeddingModelRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, backend, backend_version, runtime_version, model, revision,
                    dimensions, tokenizer_hash, model_hash, normalize,
                    execution_provider, quantization, cache_path
             FROM embedding_models ORDER BY id",
        )?;
        let models = stmt
            .query_map([], row_to_model)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(models)
    }

    // ----- embeddings (per channel) -----

    /// The representation channels present in the corpus that are eligible for
    /// embedding (every channel except the display-only `FullSource`), in a
    /// deterministic order.
    pub fn embeddable_channels(&self) -> Result<Vec<RepresentationKind>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT kind FROM representations WHERE kind != ?1 ORDER BY kind",
        )?;
        let channels = stmt
            .query_map([RepresentationKind::FullSource.as_str()], |row| {
                Ok(RepresentationKind::from(row.get::<_, String>(0)?.as_str()))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(channels)
    }

    pub fn insert_embedding(
        &self,
        model_id: ModelId,
        channel: &RepresentationKind,
        content_hash: &str,
        vector: &[f32],
    ) -> Result<()> {
        let norm = vector
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            .sqrt();
        self.conn.execute(
            "INSERT OR IGNORE INTO embeddings(model_id, channel, content_hash, vector_blob, norm, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, datetime('now'))",
            params![model_id, channel.as_str(), content_hash, vector_to_blob(vector), norm],
        )?;
        Ok(())
    }

    pub fn get_embedding(
        &self,
        model_id: ModelId,
        channel: &RepresentationKind,
        content_hash: &str,
    ) -> Result<Option<Vec<f32>>> {
        Ok(self
            .conn
            .query_row(
                "SELECT vector_blob FROM embeddings
                 WHERE model_id = ?1 AND channel = ?2 AND content_hash = ?3",
                params![model_id, channel.as_str(), content_hash],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?
            .map(|blob| blob_to_vector(&blob)))
    }

    pub fn count_embeddings(&self, model_id: ModelId) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM embeddings WHERE model_id = ?1",
            [model_id],
            |row| row.get(0),
        )?)
    }

    /// Distinct content hashes for one channel that this model has not embedded
    /// yet, with one representative content per hash (`None` under retention).
    pub fn count_unembedded_hashes(
        &self,
        model_id: ModelId,
        channel: &RepresentationKind,
    ) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM (
               SELECT r.content_hash
               FROM representations r
               LEFT JOIN embeddings e
                 ON e.content_hash = r.content_hash AND e.channel = r.kind AND e.model_id = ?1
               WHERE r.kind = ?2 AND e.content_hash IS NULL
               GROUP BY r.content_hash
             )",
            params![model_id, channel.as_str()],
            |row| row.get(0),
        )?)
    }

    pub fn unembedded_hashes_page(
        &self,
        model_id: ModelId,
        channel: &RepresentationKind,
        after_hash: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(String, Option<String>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT r.content_hash, MAX(r.content)
             FROM representations r
             LEFT JOIN embeddings e
               ON e.content_hash = r.content_hash AND e.channel = r.kind AND e.model_id = ?1
             WHERE r.kind = ?2 AND e.content_hash IS NULL
               AND (?3 IS NULL OR r.content_hash > ?3)
             GROUP BY r.content_hash
             ORDER BY r.content_hash
             LIMIT ?4",
        )?;
        let rows = stmt
            .query_map(
                params![model_id, channel.as_str(), after_hash, limit as i64],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// One `(content_hash, language, content)` per distinct hash for a channel,
    /// for offline token measurement. `content` is NULL under report/minimal
    /// retention and must be recovered from source.
    pub fn channel_texts(
        &self,
        channel: &RepresentationKind,
    ) -> Result<Vec<(String, String, Option<String>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT r.content_hash, MAX(u.language_id), MAX(r.content)
             FROM representations r
             JOIN code_units u ON u.id = r.unit_id
             WHERE r.kind = ?1
             GROUP BY r.content_hash
             ORDER BY r.content_hash",
        )?;
        let rows = stmt
            .query_map([channel.as_str()], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// One source location per requested content hash within a channel, for
    /// re-deriving text when retention did not store it.
    pub fn locations_for_content_hashes(
        &self,
        channel: &RepresentationKind,
        hashes: &[String],
    ) -> Result<Vec<HashLocation>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT p.source_dir, f.relative_path, f.language_id
             FROM representations r
             JOIN code_units u ON u.id = r.unit_id
             JOIN files f ON f.id = u.file_id
             JOIN projects p ON p.id = f.project_id
             WHERE r.kind = ?1 AND r.content_hash = ?2
             LIMIT 1",
        )?;
        let mut locations = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for hash in hashes {
            let location = stmt
                .query_row(params![channel.as_str(), hash], |row| {
                    Ok(HashLocation {
                        source_dir: row.get(0)?,
                        relative_path: row.get(1)?,
                        language_id: row.get(2)?,
                    })
                })
                .optional()?;
            if let Some(location) = location
                && seen.insert((location.source_dir.clone(), location.relative_path.clone()))
            {
                locations.push(location);
            }
        }
        Ok(locations)
    }

    /// Remove embeddings whose (channel, content_hash) no longer appears in any
    /// representation.
    pub fn prune_orphan_embeddings(&self) -> Result<usize> {
        let deleted = self.conn.execute(
            "DELETE FROM embeddings WHERE (channel, content_hash) NOT IN
               (SELECT DISTINCT kind, content_hash FROM representations)",
            [],
        )?;
        Ok(deleted)
    }

    // ----- analysis runs -----

    pub fn create_analysis_run(
        &self,
        analysis_kind: &str,
        model_id: ModelId,
        project_scope: &[String],
        config_json: &str,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO analysis_runs(analysis_kind, model_id, project_scope_json, config_json, created_at)
             VALUES (?1, ?2, ?3, ?4, datetime('now'))",
            params![
                analysis_kind,
                model_id,
                serde_json::to_string(project_scope)?,
                config_json
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    // ----- snapshot (the storage-neutral export) -----

    /// Export the selected projects (empty = all) as a storage-neutral
    /// [`IndexSnapshot`] the search engine loads from. Requires exactly one
    /// embedding model (the corpus invariant); its vectors are grouped by
    /// channel. This is the sole coupling point between SQLite and search —
    /// any other backend produces the same type by its own means.
    pub fn snapshot(&self, project_labels: &[String]) -> Result<IndexSnapshot> {
        let all_projects = self.list_projects()?;
        let projects: Vec<Project> = if project_labels.is_empty() {
            all_projects
        } else {
            let mut selected = Vec::new();
            for label in project_labels {
                let project = all_projects
                    .iter()
                    .find(|p| &p.label == label)
                    .with_context(|| format!("project {label:?} is not indexed"))?;
                selected.push(project.clone());
            }
            selected
        };
        if projects.is_empty() {
            bail!("no indexed projects; index a project first");
        }

        let model = match self.list_models()?.as_slice() {
            [] => bail!("no embeddings found; embed the corpus first"),
            [model] => model.clone(),
            _ => bail!("database contains multiple embedding models; this is unsupported"),
        };

        let placeholders = projects.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let ids = params_from_iter(projects.iter().map(|p| p.id));

        // Units, ordered deterministically.
        let sql = format!(
            "SELECT u.id, u.entity_id, u.entity_version_id, u.generation,
                    p.label, f.relative_path, u.language_id, u.kind, u.name, u.scope,
                    u.start_byte, u.end_byte, u.start_line, u.end_line, u.body_node_count
             FROM code_units u
             JOIN files f ON f.id = u.file_id
             JOIN projects p ON p.id = f.project_id
             WHERE p.id IN ({placeholders})
             ORDER BY p.label, f.relative_path, u.start_byte, u.end_byte"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut unit_rows: Vec<(UnitId, UnitRecord)> = stmt
            .query_map(ids, |row| {
                Ok((
                    row.get::<_, UnitId>(0)?,
                    UnitRecord {
                        entity_id: row.get(1)?,
                        entity_version_id: row.get(2)?,
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
                        representations: Vec::new(),
                    },
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        // Representations for those units, attached by unit id.
        {
            let unit_index: std::collections::HashMap<UnitId, usize> = unit_rows
                .iter()
                .enumerate()
                .map(|(i, (id, _))| (*id, i))
                .collect();
            let sql = format!(
                "SELECT r.unit_id, r.kind, r.content_hash, r.content
                 FROM representations r
                 JOIN code_units u ON u.id = r.unit_id
                 JOIN files f ON f.id = u.file_id
                 WHERE f.project_id IN ({placeholders})",
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let mut rows =
                stmt.query(params_from_iter(projects.iter().map(|p| p.id)))?;
            while let Some(row) = rows.next()? {
                let unit_id: UnitId = row.get(0)?;
                let kind: String = row.get(1)?;
                if let Some(&index) = unit_index.get(&unit_id) {
                    unit_rows[index].1.representations.push(RepresentationRef {
                        kind: RepresentationKind::from(kind.as_str()),
                        content_hash: row.get(2)?,
                        content: row.get(3)?,
                    });
                }
            }
        }
        for (_, unit) in unit_rows.iter_mut() {
            unit.representations.sort_by(|a, b| a.kind.cmp(&b.kind));
        }
        let units: Vec<UnitRecord> = unit_rows.into_iter().map(|(_, unit)| unit).collect();

        // Embeddings grouped by channel.
        let mut channel_map: std::collections::BTreeMap<String, Vec<(String, Vec<f32>)>> =
            std::collections::BTreeMap::new();
        {
            let mut stmt = self.conn.prepare(
                "SELECT channel, content_hash, vector_blob FROM embeddings
                 WHERE model_id = ?1 ORDER BY channel, content_hash",
            )?;
            let mut rows = stmt.query([model.id])?;
            while let Some(row) = rows.next()? {
                let channel: String = row.get(0)?;
                let hash: String = row.get(1)?;
                let blob: Vec<u8> = row.get(2)?;
                channel_map
                    .entry(channel)
                    .or_default()
                    .push((hash, blob_to_vector(&blob)));
            }
        }
        let channels = channel_map
            .into_iter()
            .map(|(channel, vectors)| ChannelEmbeddings {
                channel: RepresentationKind::from(channel.as_str()),
                dimensions: model.identity.dimensions,
                vectors,
            })
            .collect();

        Ok(IndexSnapshot {
            model: model.identity,
            projects: projects
                .into_iter()
                .map(|p| ProjectRecord {
                    label: p.label,
                    source_dir: p.source_dir,
                    role: p.role,
                })
                .collect(),
            units,
            channels,
        })
    }
}

fn row_to_project(row: &rusqlite::Row<'_>) -> rusqlite::Result<Project> {
    Ok(Project {
        id: row.get(0)?,
        label: row.get(1)?,
        source_dir: row.get(2)?,
        role: row.get(3)?,
    })
}

fn row_to_file(row: &rusqlite::Row<'_>) -> rusqlite::Result<FileRecord> {
    Ok(FileRecord {
        id: row.get(0)?,
        project_id: row.get(1)?,
        relative_path: row.get(2)?,
        language_id: row.get(3)?,
        mtime_ns: row.get(4)?,
        size: row.get(5)?,
        source_hash: row.get(6)?,
    })
}

fn row_to_unit(row: &rusqlite::Row<'_>) -> rusqlite::Result<CodeUnit> {
    Ok(CodeUnit {
        id: row.get(0)?,
        file_id: row.get(1)?,
        entity_id: row.get(2)?,
        entity_version_id: row.get(3)?,
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

fn row_to_model(row: &rusqlite::Row<'_>) -> rusqlite::Result<EmbeddingModelRecord> {
    Ok(EmbeddingModelRecord {
        id: row.get(0)?,
        identity: ModelIdentity {
            backend: row.get(1)?,
            backend_version: row.get(2)?,
            runtime_version: row.get(3)?,
            model: row.get(4)?,
            revision: row.get(5)?,
            dimensions: row.get::<_, i64>(6)? as usize,
            tokenizer_hash: row.get(7)?,
            model_hash: row.get(8)?,
            normalize: row.get::<_, i64>(9)? != 0,
            execution_provider: row.get(10)?,
            quantization: row.get(11)?,
            cache_path: row.get(12)?,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repr(kind: RepresentationKind, hash: &str, content: &str) -> NewRepresentation {
        NewRepresentation {
            kind,
            content_hash: hash.to_string(),
            content: Some(content.to_string()),
        }
    }

    fn test_unit(hash: &str) -> NewCodeUnit {
        NewCodeUnit {
            entity_id: format!("ent-{hash}"),
            entity_version_id: format!("ver-{hash}"),
            generation: 1,
            language_id: "rust".into(),
            kind: "function".into(),
            name: "example".into(),
            scope: None,
            start_byte: 0,
            end_byte: 100,
            start_line: 1,
            end_line: 10,
            body_node_count: 12,
            source_hash: format!("src-{hash}"),
            normalized_body_hash: hash.to_string(),
            representations: vec![
                repr(RepresentationKind::FullSource, &format!("src-{hash}"), "fn example() {}"),
                repr(RepresentationKind::Implementation, hash, "fn example() {}"),
            ],
        }
    }

    fn test_identity() -> ModelIdentity {
        ModelIdentity {
            backend: "fastembed".into(),
            backend_version: "5.0".into(),
            runtime_version: None,
            model: "BGESmallENV15".into(),
            revision: None,
            dimensions: 384,
            tokenizer_hash: None,
            model_hash: None,
            normalize: true,
            execution_provider: "cpu".into(),
            quantization: None,
            cache_path: None,
        }
    }

    fn seed_file(db: &Db, label: &str, path: &str) -> (ProjectId, FileId) {
        let project = db.upsert_project(label, "/src").unwrap();
        let file_id = db
            .upsert_file(&NewFile {
                project_id: project,
                relative_path: path.into(),
                language_id: "rust".into(),
                mtime_ns: 1,
                size: 1,
                source_hash: "h".into(),
            })
            .unwrap();
        (project, file_id)
    }

    #[test]
    fn migrates_from_empty() {
        let db = open_in_memory().unwrap();
        let version: i64 = db
            .conn()
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, migrations::SCHEMA_VERSION);
    }

    #[test]
    fn migration_is_idempotent_on_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        {
            let db = open_or_create(&path).unwrap();
            db.upsert_project("main", "/src").unwrap();
        }
        let db = open_or_create(&path).unwrap();
        assert_eq!(db.list_projects().unwrap().len(), 1);
    }

    #[test]
    fn project_crud_and_immutable_root() {
        let db = open_in_memory().unwrap();
        let id = db.upsert_project("main", "/src").unwrap();
        assert_eq!(db.upsert_project("main", "/src").unwrap(), id);
        assert!(db.upsert_project("main", "/other").is_err());
        assert!(db.delete_project("main").unwrap());
        assert!(!db.delete_project("main").unwrap());
    }

    #[test]
    fn unit_insert_stores_representations_and_returns_ids() {
        let db = open_in_memory().unwrap();
        let (_, file_id) = seed_file(&db, "main", "lib.rs");
        let ids = db
            .insert_units(file_id, &[test_unit("a"), test_unit("b")])
            .unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(db.list_units_for_file(file_id).unwrap().len(), 2);
        // Each unit carries its FullSource + Implementation representations.
        let channels = db.embeddable_channels().unwrap();
        assert_eq!(channels, vec![RepresentationKind::Implementation]);

        // Re-upserting the file clears its prior units (and representations).
        let file_id_again = db
            .upsert_file(&NewFile {
                project_id: db.get_project("main").unwrap().unwrap().id,
                relative_path: "lib.rs".into(),
                language_id: "rust".into(),
                mtime_ns: 2,
                size: 2,
                source_hash: "h2".into(),
            })
            .unwrap();
        assert_eq!(file_id, file_id_again);
        assert_eq!(db.list_units_for_file(file_id).unwrap().len(), 0);
    }

    #[test]
    fn entity_ledger_tracks_generations() {
        let db = open_in_memory().unwrap();
        let (_, file_id) = seed_file(&db, "main", "lib.rs");
        let mut unit = test_unit("a");
        unit.generation = 3;
        db.insert_units(file_id, &[unit]).unwrap();
        let (first, last): (i64, i64) = db
            .conn()
            .query_row(
                "SELECT first_generation, last_generation FROM entities WHERE entity_id = 'ent-a'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!((first, last), (3, 3));
    }

    #[test]
    fn invalid_unit_range_rejected() {
        let db = open_in_memory().unwrap();
        let (_, file_id) = seed_file(&db, "main", "lib.rs");
        let mut unit = test_unit("a");
        unit.start_byte = 100;
        unit.end_byte = 50;
        assert!(db.insert_units(file_id, &[unit]).is_err());
    }

    #[test]
    fn embedding_dedup_by_model_channel_and_hash() {
        let db = open_in_memory().unwrap();
        let model = db.find_or_create_model(&test_identity()).unwrap();
        let impl_ = RepresentationKind::Implementation;
        db.insert_embedding(model, &impl_, "hash-a", &[1.0, 0.0])
            .unwrap();
        db.insert_embedding(model, &impl_, "hash-a", &[0.0, 1.0])
            .unwrap(); // ignored
        assert_eq!(db.count_embeddings(model).unwrap(), 1);
        assert_eq!(
            db.get_embedding(model, &impl_, "hash-a").unwrap().unwrap(),
            vec![1.0, 0.0]
        );
        // Same hash on a different channel is a distinct row.
        db.insert_embedding(model, &RepresentationKind::Signature, "hash-a", &[0.5, 0.5])
            .unwrap();
        assert_eq!(db.count_embeddings(model).unwrap(), 2);
    }

    #[test]
    fn model_identity_uniqueness() {
        let db = open_in_memory().unwrap();
        let id1 = db.find_or_create_model(&test_identity()).unwrap();
        let id2 = db.find_or_create_model(&test_identity()).unwrap();
        assert_eq!(id1, id2);
        let mut quantized = test_identity();
        quantized.quantization = Some("q8".into());
        let id3 = db.find_or_create_model(&quantized).unwrap();
        assert_ne!(id1, id3);
        assert_eq!(db.list_models().unwrap().len(), 2);
    }

    #[test]
    fn orphan_embedding_cleanup_by_channel() {
        let db = open_in_memory().unwrap();
        let (_, file_id) = seed_file(&db, "main", "lib.rs");
        db.insert_units(file_id, &[test_unit("live"), test_unit("dead")])
            .unwrap();
        let model = db.find_or_create_model(&test_identity()).unwrap();
        let impl_ = RepresentationKind::Implementation;
        db.insert_embedding(model, &impl_, "live", &[1.0]).unwrap();
        db.insert_embedding(model, &impl_, "dead", &[1.0]).unwrap();

        // Replace units so "dead" no longer appears in any representation.
        db.upsert_file(&NewFile {
            project_id: db.get_project("main").unwrap().unwrap().id,
            relative_path: "lib.rs".into(),
            language_id: "rust".into(),
            mtime_ns: 2,
            size: 1,
            source_hash: "h2".into(),
        })
        .unwrap();
        db.insert_units(file_id, &[test_unit("live")]).unwrap();

        assert_eq!(db.prune_orphan_embeddings().unwrap(), 1);
        assert!(db.get_embedding(model, &impl_, "dead").unwrap().is_none());
        assert!(db.get_embedding(model, &impl_, "live").unwrap().is_some());
    }

    #[test]
    fn unembedded_hashes_and_resume_per_channel() {
        let db = open_in_memory().unwrap();
        let (_, file_id) = seed_file(&db, "main", "lib.rs");
        let impl_ = RepresentationKind::Implementation;
        // Two units share hash "x": only one embedding is needed.
        let mut ux = test_unit("x");
        ux.entity_id = "ex1".into();
        let mut ux2 = test_unit("x");
        ux2.entity_id = "ex2".into();
        ux2.start_byte = 200;
        ux2.end_byte = 300;
        db.insert_units(file_id, &[ux, ux2, test_unit("y")]).unwrap();
        let model = db.find_or_create_model(&test_identity()).unwrap();
        assert_eq!(db.count_unembedded_hashes(model, &impl_).unwrap(), 2);
        let pending = db.unembedded_hashes_page(model, &impl_, None, 100).unwrap();
        assert_eq!(
            pending.iter().map(|(h, _)| h.as_str()).collect::<Vec<_>>(),
            vec!["x", "y"]
        );
        db.insert_embedding(model, &impl_, "x", &[1.0]).unwrap();
        let pending = db.unembedded_hashes_page(model, &impl_, None, 100).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].0, "y");
    }

    #[test]
    fn immutable_settings() {
        let db = open_in_memory().unwrap();
        db.check_or_set_immutable("embedding.model", "BGESmallENV15")
            .unwrap();
        db.check_or_set_immutable("embedding.model", "BGESmallENV15")
            .unwrap();
        assert!(
            db.check_or_set_immutable("embedding.model", "BGEBaseENV15")
                .is_err()
        );
    }

    #[test]
    fn generation_counter_increments() {
        let db = open_in_memory().unwrap();
        assert_eq!(db.current_generation().unwrap(), 0);
        assert_eq!(db.bump_generation().unwrap(), 1);
        assert_eq!(db.bump_generation().unwrap(), 2);
        assert_eq!(db.current_generation().unwrap(), 2);
    }

    #[test]
    fn cascade_from_project_removes_files_units_and_entities() {
        let db = open_in_memory().unwrap();
        let (project, file_id) = seed_file(&db, "main", "lib.rs");
        db.insert_units(file_id, &[test_unit("a")]).unwrap();
        db.delete_project("main").unwrap();
        assert_eq!(db.count_units().unwrap(), 0);
        assert_eq!(db.list_files(project).unwrap().len(), 0);
        let entity_count: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM entities", [], |row| row.get(0))
            .unwrap();
        assert_eq!(entity_count, 0);
    }

    #[test]
    fn snapshot_exports_units_representations_and_channels() {
        let db = open_in_memory().unwrap();
        let (_, file_id) = seed_file(&db, "main", "lib.rs");
        db.insert_units(file_id, &[test_unit("a"), test_unit("b")])
            .unwrap();
        let model = db.find_or_create_model(&test_identity()).unwrap();
        db.insert_embedding(model, &RepresentationKind::Implementation, "a", &[1.0])
            .unwrap();

        let snapshot = db.snapshot(&[]).unwrap();
        assert_eq!(snapshot.projects.len(), 1);
        assert_eq!(snapshot.units.len(), 2);
        assert!(
            snapshot.units[0]
                .content_hash(&RepresentationKind::Implementation)
                .is_some()
        );
        let channel = snapshot
            .channel(&RepresentationKind::Implementation)
            .unwrap();
        assert_eq!(channel.vectors.len(), 1);
        assert_eq!(channel.vectors[0].0, "a");
    }
}
