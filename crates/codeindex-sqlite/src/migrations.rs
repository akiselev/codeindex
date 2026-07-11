use anyhow::{Context, Result};
use rusqlite::Connection;

/// Schema migrations applied in order. `PRAGMA user_version` records how many
/// have run; new migrations are appended, never edited.
const MIGRATIONS: &[&str] = &[
    // 1: initial schema
    r#"
    CREATE TABLE metadata(
      id INTEGER PRIMARY KEY CHECK (id = 1),
      schema_version INTEGER NOT NULL,
      created_at TEXT NOT NULL
    );

    CREATE TABLE settings(
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );

    CREATE TABLE projects(
      id INTEGER PRIMARY KEY,
      label TEXT NOT NULL UNIQUE,
      source_dir TEXT NOT NULL,
      role TEXT,
      created_at TEXT NOT NULL
    );

    CREATE TABLE files(
      id INTEGER PRIMARY KEY,
      project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
      relative_path TEXT NOT NULL,
      language_id TEXT NOT NULL,
      mtime_ns INTEGER NOT NULL,
      size INTEGER NOT NULL,
      source_hash TEXT NOT NULL,
      UNIQUE (project_id, relative_path)
    );
    CREATE INDEX idx_files_source_hash ON files(source_hash);

    CREATE TABLE code_units(
      id INTEGER PRIMARY KEY,
      file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
      language_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      name TEXT NOT NULL,
      scope TEXT,
      start_byte INTEGER NOT NULL,
      end_byte INTEGER NOT NULL,
      start_line INTEGER NOT NULL,
      end_line INTEGER NOT NULL,
      body_node_count INTEGER NOT NULL,
      source_hash TEXT NOT NULL,
      normalized_body_hash TEXT NOT NULL,
      display_source TEXT,
      embedding_text TEXT,
      CHECK (start_byte < end_byte),
      CHECK (start_line <= end_line)
    );
    CREATE INDEX idx_code_units_file ON code_units(file_id);
    CREATE INDEX idx_code_units_body_hash ON code_units(normalized_body_hash);
    CREATE INDEX idx_code_units_file_range ON code_units(file_id, start_byte, end_byte);

    CREATE TABLE embedding_models(
      id INTEGER PRIMARY KEY,
      backend TEXT NOT NULL,
      backend_version TEXT NOT NULL,
      runtime_version TEXT,
      model TEXT NOT NULL,
      revision TEXT,
      dimensions INTEGER NOT NULL CHECK (dimensions > 0),
      tokenizer_hash TEXT,
      model_hash TEXT,
      normalize INTEGER NOT NULL,
      execution_provider TEXT NOT NULL,
      quantization TEXT,
      cache_path TEXT,
      UNIQUE (
        backend, backend_version, model, revision, dimensions,
        tokenizer_hash, model_hash, normalize, execution_provider, quantization
      )
    );

    CREATE TABLE embeddings(
      model_id INTEGER NOT NULL REFERENCES embedding_models(id) ON DELETE CASCADE,
      normalized_body_hash TEXT NOT NULL,
      vector_blob BLOB NOT NULL,
      norm REAL NOT NULL,
      created_at TEXT NOT NULL,
      PRIMARY KEY (model_id, normalized_body_hash)
    );

    CREATE TABLE analysis_runs(
      id INTEGER PRIMARY KEY,
      analysis_kind TEXT NOT NULL,
      model_id INTEGER NOT NULL REFERENCES embedding_models(id),
      project_scope_json TEXT NOT NULL,
      config_json TEXT NOT NULL,
      created_at TEXT NOT NULL
    );

    CREATE TABLE analysis_artifacts(
      id INTEGER PRIMARY KEY,
      run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
      artifact_kind TEXT NOT NULL,
      method TEXT NOT NULL,
      params_json TEXT NOT NULL,
      metrics_json TEXT NOT NULL,
      blob BLOB,
      created_at TEXT NOT NULL
    );

    INSERT INTO metadata(id, schema_version, created_at) VALUES (1, 1, datetime('now'));
    "#,
];

pub fn migrate(conn: &Connection) -> Result<()> {
    let mut version: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    while (version as usize) < MIGRATIONS.len() {
        let sql = MIGRATIONS[version as usize];
        conn.execute_batch(&format!("BEGIN;\n{sql}\nCOMMIT;"))
            .with_context(|| format!("applying migration {}", version + 1))?;
        version += 1;
        conn.pragma_update(None, "user_version", version)?;
        conn.execute(
            "UPDATE metadata SET schema_version = ?1 WHERE id = 1",
            [version],
        )?;
    }
    Ok(())
}
