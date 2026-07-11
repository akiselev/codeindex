pub mod migrations;
pub mod models;

use std::path::Path;

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, params};

pub use models::{
    CodeUnit, EmbeddingModelRecord, FileId, FileRecord, ModelId, ModelIdentity, NewCodeUnit,
    NewFile, Project, ProjectId, UnitId, blob_to_vector, vector_to_blob,
};

pub struct Db {
    conn: Connection,
}

/// Where a normalized body hash can be found on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashLocation {
    pub source_dir: String,
    pub relative_path: String,
    pub language_id: String,
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
                |row| {
                    Ok(Project {
                        id: row.get(0)?,
                        label: row.get(1)?,
                        source_dir: row.get(2)?,
                        role: row.get(3)?,
                    })
                },
            )
            .optional()?)
    }

    pub fn list_projects(&self) -> Result<Vec<Project>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, label, source_dir, role FROM projects ORDER BY label")?;
        let projects = stmt
            .query_map([], |row| {
                Ok(Project {
                    id: row.get(0)?,
                    label: row.get(1)?,
                    source_dir: row.get(2)?,
                    role: row.get(3)?,
                })
            })?
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
                |row| {
                    Ok(FileRecord {
                        id: row.get(0)?,
                        project_id: row.get(1)?,
                        relative_path: row.get(2)?,
                        language_id: row.get(3)?,
                        mtime_ns: row.get(4)?,
                        size: row.get(5)?,
                        source_hash: row.get(6)?,
                    })
                },
            )
            .optional()?)
    }

    pub fn list_files(&self, project_id: ProjectId) -> Result<Vec<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, project_id, relative_path, language_id, mtime_ns, size, source_hash
             FROM files WHERE project_id = ?1 ORDER BY relative_path",
        )?;
        let files = stmt
            .query_map([project_id], |row| {
                Ok(FileRecord {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    relative_path: row.get(2)?,
                    language_id: row.get(3)?,
                    mtime_ns: row.get(4)?,
                    size: row.get(5)?,
                    source_hash: row.get(6)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(files)
    }

    /// Insert or update a file row and return its id. On update, existing
    /// code units for the file are deleted so the caller can reinsert them.
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

    // ----- code units -----

    pub fn insert_units(&self, file_id: FileId, units: &[NewCodeUnit]) -> Result<()> {
        let mut stmt = self.conn.prepare_cached(
            "INSERT INTO code_units(
               file_id, language_id, kind, name, scope,
               start_byte, end_byte, start_line, end_line,
               body_node_count, source_hash, normalized_body_hash,
               display_source, embedding_text)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        )?;
        for unit in units {
            stmt.execute(params![
                file_id,
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
                unit.display_source,
                unit.embedding_text,
            ])?;
        }
        Ok(())
    }

    pub fn list_units_for_file(&self, file_id: FileId) -> Result<Vec<CodeUnit>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, file_id, language_id, kind, name, scope,
                    start_byte, end_byte, start_line, end_line,
                    body_node_count, source_hash, normalized_body_hash,
                    display_source, embedding_text
             FROM code_units WHERE file_id = ?1 ORDER BY start_byte",
        )?;
        let units = stmt
            .query_map([file_id], row_to_unit)?
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

    // ----- embeddings -----

    pub fn insert_embedding(
        &self,
        model_id: ModelId,
        normalized_body_hash: &str,
        vector: &[f32],
    ) -> Result<()> {
        let norm = vector
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            .sqrt();
        self.conn.execute(
            "INSERT OR IGNORE INTO embeddings(model_id, normalized_body_hash, vector_blob, norm, created_at)
             VALUES (?1, ?2, ?3, ?4, datetime('now'))",
            params![model_id, normalized_body_hash, vector_to_blob(vector), norm],
        )?;
        Ok(())
    }

    pub fn get_embedding(
        &self,
        model_id: ModelId,
        normalized_body_hash: &str,
    ) -> Result<Option<Vec<f32>>> {
        Ok(self
            .conn
            .query_row(
                "SELECT vector_blob FROM embeddings
                 WHERE model_id = ?1 AND normalized_body_hash = ?2",
                params![model_id, normalized_body_hash],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?
            .map(|blob| blob_to_vector(&blob)))
    }

    /// Every `(body_hash, vector)` embedded under `model_id`, hash-ordered for
    /// deterministic alignment across databases.
    pub fn all_embeddings(&self, model_id: ModelId) -> Result<Vec<(String, Vec<f32>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT normalized_body_hash, vector_blob FROM embeddings
             WHERE model_id = ?1 ORDER BY normalized_body_hash",
        )?;
        let rows = stmt
            .query_map([model_id], |row| {
                let hash: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((hash, blob_to_vector(&blob)))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn count_embeddings(&self, model_id: ModelId) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM embeddings WHERE model_id = ?1",
            [model_id],
            |row| row.get(0),
        )?)
    }

    /// Distinct body hashes of indexed units that this model has not
    /// embedded yet, with one representative embedding text per hash.
    pub fn unembedded_hashes(&self, model_id: ModelId) -> Result<Vec<(String, Option<String>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT u.normalized_body_hash, MAX(u.embedding_text)
             FROM code_units u
             LEFT JOIN embeddings e
               ON e.normalized_body_hash = u.normalized_body_hash AND e.model_id = ?1
             WHERE e.normalized_body_hash IS NULL
             GROUP BY u.normalized_body_hash
             ORDER BY u.normalized_body_hash",
        )?;
        let rows = stmt
            .query_map([model_id], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn count_unembedded_hashes(&self, model_id: ModelId) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM (
               SELECT u.normalized_body_hash
               FROM code_units u
               LEFT JOIN embeddings e
                 ON e.normalized_body_hash = u.normalized_body_hash AND e.model_id = ?1
               WHERE e.normalized_body_hash IS NULL
               GROUP BY u.normalized_body_hash
             )",
            [model_id],
            |row| row.get(0),
        )?)
    }

    pub fn unembedded_hashes_page(
        &self,
        model_id: ModelId,
        after_hash: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(String, Option<String>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT u.normalized_body_hash, MAX(u.embedding_text)
             FROM code_units u
             LEFT JOIN embeddings e
               ON e.normalized_body_hash = u.normalized_body_hash AND e.model_id = ?1
             WHERE e.normalized_body_hash IS NULL
               AND (?2 IS NULL OR u.normalized_body_hash > ?2)
             GROUP BY u.normalized_body_hash
             ORDER BY u.normalized_body_hash
             LIMIT ?3",
        )?;
        let rows = stmt
            .query_map(params![model_id, after_hash, limit as i64], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// One `(hash, language, embedding_text)` per distinct body hash across
    /// all projects, for offline token measurement. `embedding_text` is NULL
    /// under report/minimal retention and must be recovered from source. When
    /// a hash spans languages, `MAX` picks one deterministically.
    pub fn all_unit_texts(&self) -> Result<Vec<(String, String, Option<String>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT u.normalized_body_hash, MAX(u.language_id), MAX(u.embedding_text)
             FROM code_units u
             GROUP BY u.normalized_body_hash
             ORDER BY u.normalized_body_hash",
        )?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// One source location per requested hash, for re-deriving embedding
    /// text when retention did not store it.
    pub fn locations_for_hashes(&self, hashes: &[String]) -> Result<Vec<HashLocation>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT p.source_dir, f.relative_path, f.language_id
             FROM code_units u
             JOIN files f ON f.id = u.file_id
             JOIN projects p ON p.id = f.project_id
             WHERE u.normalized_body_hash = ?1
             LIMIT 1",
        )?;
        let mut locations = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for hash in hashes {
            let location = stmt
                .query_row([hash], |row| {
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

    /// Remove embeddings whose body hash no longer appears in any code unit.
    pub fn prune_orphan_embeddings(&self) -> Result<usize> {
        let deleted = self.conn.execute(
            "DELETE FROM embeddings WHERE normalized_body_hash NOT IN
               (SELECT DISTINCT normalized_body_hash FROM code_units)",
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
}

fn row_to_unit(row: &rusqlite::Row<'_>) -> rusqlite::Result<CodeUnit> {
    Ok(CodeUnit {
        id: row.get(0)?,
        file_id: row.get(1)?,
        language_id: row.get(2)?,
        kind: row.get(3)?,
        name: row.get(4)?,
        scope: row.get(5)?,
        start_byte: row.get::<_, i64>(6)? as usize,
        end_byte: row.get::<_, i64>(7)? as usize,
        start_line: row.get::<_, i64>(8)? as usize,
        end_line: row.get::<_, i64>(9)? as usize,
        body_node_count: row.get::<_, i64>(10)? as usize,
        source_hash: row.get(11)?,
        normalized_body_hash: row.get(12)?,
        display_source: row.get(13)?,
        embedding_text: row.get(14)?,
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

    fn test_unit(hash: &str) -> NewCodeUnit {
        NewCodeUnit {
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
            display_source: Some("fn example() {}".into()),
            embedding_text: Some("fn example() {}".into()),
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

    #[test]
    fn migrates_from_empty() {
        let db = open_in_memory().unwrap();
        let version: i64 = db
            .conn()
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 1);
        let schema_version: i64 = db
            .conn()
            .query_row("SELECT schema_version FROM metadata", [], |row| row.get(0))
            .unwrap();
        assert_eq!(schema_version, 1);
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
    fn file_crud_and_unit_replacement() {
        let db = open_in_memory().unwrap();
        let project = db.upsert_project("main", "/src").unwrap();
        let file = NewFile {
            project_id: project,
            relative_path: "lib.rs".into(),
            language_id: "rust".into(),
            mtime_ns: 100,
            size: 10,
            source_hash: "h1".into(),
        };
        let file_id = db.upsert_file(&file).unwrap();
        db.insert_units(file_id, &[test_unit("a"), test_unit("b")])
            .unwrap();
        assert_eq!(db.list_units_for_file(file_id).unwrap().len(), 2);

        // Re-upserting the file clears its prior units.
        let file_id_again = db.upsert_file(&file).unwrap();
        assert_eq!(file_id, file_id_again);
        assert_eq!(db.list_units_for_file(file_id).unwrap().len(), 0);

        db.insert_units(file_id, &[test_unit("c")]).unwrap();
        db.delete_file(file_id).unwrap();
        assert_eq!(db.count_units().unwrap(), 0);
    }

    #[test]
    fn file_path_uniqueness_scoped_by_project() {
        let db = open_in_memory().unwrap();
        let p1 = db.upsert_project("v1", "/a").unwrap();
        let p2 = db.upsert_project("v2", "/b").unwrap();
        for project_id in [p1, p2] {
            db.upsert_file(&NewFile {
                project_id,
                relative_path: "same/path.rs".into(),
                language_id: "rust".into(),
                mtime_ns: 1,
                size: 1,
                source_hash: "h".into(),
            })
            .unwrap();
        }
        // Same relative path in two projects: two distinct rows.
        let count: i64 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM files WHERE relative_path = 'same/path.rs'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 2);

        // Direct duplicate insert within one project violates the constraint.
        let err = db.conn().execute(
            "INSERT INTO files(project_id, relative_path, language_id, mtime_ns, size, source_hash)
             VALUES (?1, 'same/path.rs', 'rust', 1, 1, 'h')",
            [p1],
        );
        assert!(err.is_err());
    }

    #[test]
    fn invalid_unit_range_rejected() {
        let db = open_in_memory().unwrap();
        let project = db.upsert_project("main", "/src").unwrap();
        let file_id = db
            .upsert_file(&NewFile {
                project_id: project,
                relative_path: "lib.rs".into(),
                language_id: "rust".into(),
                mtime_ns: 1,
                size: 1,
                source_hash: "h".into(),
            })
            .unwrap();
        let mut unit = test_unit("a");
        unit.start_byte = 100;
        unit.end_byte = 50;
        assert!(db.insert_units(file_id, &[unit]).is_err());
    }

    #[test]
    fn embedding_dedup_by_model_and_hash() {
        let db = open_in_memory().unwrap();
        let model = db.find_or_create_model(&test_identity()).unwrap();
        db.insert_embedding(model, "hash-a", &[1.0, 0.0]).unwrap();
        db.insert_embedding(model, "hash-a", &[0.0, 1.0]).unwrap(); // ignored
        assert_eq!(db.count_embeddings(model).unwrap(), 1);
        assert_eq!(
            db.get_embedding(model, "hash-a").unwrap().unwrap(),
            vec![1.0, 0.0]
        );

        // A different model identity gets its own row for the same hash.
        let mut other = test_identity();
        other.model = "BGEBaseENV15".into();
        other.dimensions = 768;
        let model2 = db.find_or_create_model(&other).unwrap();
        db.insert_embedding(model2, "hash-a", &[0.0, 1.0]).unwrap();
        assert_eq!(db.count_embeddings(model2).unwrap(), 1);
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
        assert_eq!(
            db.get_model(id1).unwrap().unwrap().identity,
            test_identity()
        );
    }

    #[test]
    fn orphan_embedding_cleanup() {
        let db = open_in_memory().unwrap();
        let project = db.upsert_project("main", "/src").unwrap();
        let file_id = db
            .upsert_file(&NewFile {
                project_id: project,
                relative_path: "lib.rs".into(),
                language_id: "rust".into(),
                mtime_ns: 1,
                size: 1,
                source_hash: "h".into(),
            })
            .unwrap();
        db.insert_units(file_id, &[test_unit("live"), test_unit("dead")])
            .unwrap();
        let model = db.find_or_create_model(&test_identity()).unwrap();
        db.insert_embedding(model, "live", &[1.0]).unwrap();
        db.insert_embedding(model, "dead", &[1.0]).unwrap();

        // Replace the file's units so "dead" no longer exists.
        db.upsert_file(&NewFile {
            project_id: project,
            relative_path: "lib.rs".into(),
            language_id: "rust".into(),
            mtime_ns: 2,
            size: 1,
            source_hash: "h2".into(),
        })
        .unwrap();
        db.insert_units(file_id, &[test_unit("live")]).unwrap();

        assert_eq!(db.prune_orphan_embeddings().unwrap(), 1);
        assert!(db.get_embedding(model, "dead").unwrap().is_none());
        assert!(db.get_embedding(model, "live").unwrap().is_some());
    }

    #[test]
    fn unembedded_hashes_and_resume() {
        let db = open_in_memory().unwrap();
        let project = db.upsert_project("main", "/src").unwrap();
        let file_id = db
            .upsert_file(&NewFile {
                project_id: project,
                relative_path: "lib.rs".into(),
                language_id: "rust".into(),
                mtime_ns: 1,
                size: 1,
                source_hash: "h".into(),
            })
            .unwrap();
        // Two units share hash "x": only one embedding is needed.
        db.insert_units(file_id, &[test_unit("x"), test_unit("x"), test_unit("y")])
            .unwrap();
        let model = db.find_or_create_model(&test_identity()).unwrap();
        let pending = db.unembedded_hashes(model).unwrap();
        assert_eq!(
            pending.iter().map(|(h, _)| h.as_str()).collect::<Vec<_>>(),
            vec!["x", "y"]
        );
        db.insert_embedding(model, "x", &[1.0]).unwrap();
        let pending = db.unembedded_hashes(model).unwrap();
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
    fn cascade_from_project_removes_files_and_units() {
        let db = open_in_memory().unwrap();
        let project = db.upsert_project("main", "/src").unwrap();
        let file_id = db
            .upsert_file(&NewFile {
                project_id: project,
                relative_path: "lib.rs".into(),
                language_id: "rust".into(),
                mtime_ns: 1,
                size: 1,
                source_hash: "h".into(),
            })
            .unwrap();
        db.insert_units(file_id, &[test_unit("a")]).unwrap();
        db.delete_project("main").unwrap();
        assert_eq!(db.count_units().unwrap(), 0);
        assert_eq!(db.list_files(project).unwrap().len(), 0);
    }
}
