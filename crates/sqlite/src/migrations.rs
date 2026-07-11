use anyhow::{Context, Result};
use rusqlite::Connection;

/// The current schema version. The project is pre-release, so the schema is a
/// single clean bootstrap rather than an append-only migration ledger: opening
/// a database at a different version is a hard error (delete and reindex), not
/// a migration. `PRAGMA user_version` records the applied version.
pub const SCHEMA_VERSION: i64 = 1;

const SCHEMA: &str = r#"
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

-- Logical-identity ledger: one row per entity that persists across index
-- generations (M4). A renamed function keeps its entity_id.
CREATE TABLE entities(
  entity_id        TEXT PRIMARY KEY,
  project_id       INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind             TEXT NOT NULL,
  first_generation INTEGER NOT NULL,
  last_generation  INTEGER NOT NULL
);
CREATE INDEX idx_entities_project ON entities(project_id);

CREATE TABLE code_units(
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
  entity_version_id TEXT NOT NULL,
  generation INTEGER NOT NULL,
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
  CHECK (start_byte < end_byte),
  CHECK (start_line <= end_line)
);
CREATE INDEX idx_code_units_file ON code_units(file_id);
CREATE INDEX idx_code_units_body_hash ON code_units(normalized_body_hash);
CREATE INDEX idx_code_units_entity ON code_units(entity_id);
CREATE INDEX idx_code_units_file_range ON code_units(file_id, start_byte, end_byte);

-- One row per representation channel of a unit (M4). `content` is NULL under
-- report/minimal retention; `content_hash` is the embedding key.
CREATE TABLE representations(
  unit_id INTEGER NOT NULL REFERENCES code_units(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  content TEXT,
  PRIMARY KEY (unit_id, kind)
);
CREATE INDEX idx_repr_channel_hash ON representations(kind, content_hash);

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

-- Vectors keyed by (model, channel, content_hash): every representation channel
-- is embedded and searched independently (M4).
CREATE TABLE embeddings(
  model_id INTEGER NOT NULL REFERENCES embedding_models(id) ON DELETE CASCADE,
  channel TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  vector_blob BLOB NOT NULL,
  norm REAL NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (model_id, channel, content_hash)
);

-- Raw call sites staged during indexing; resolved into the Usage channel by
-- the indexer's whole-corpus pass (M4). caller_unit_id is the enclosing unit.
CREATE TABLE references_raw(
  caller_unit_id INTEGER NOT NULL REFERENCES code_units(id) ON DELETE CASCADE,
  callee_symbol TEXT NOT NULL,
  call_snippet TEXT NOT NULL,
  start_line INTEGER NOT NULL
);
CREATE INDEX idx_refs_callee ON references_raw(callee_symbol);
CREATE INDEX idx_refs_caller ON references_raw(caller_unit_id);

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
"#;

/// Bring a freshly opened connection up to the current schema. Because there is
/// no migration path yet, a non-empty database at a different `user_version` is
/// rejected rather than migrated.
pub fn migrate(conn: &Connection) -> Result<()> {
    let version: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if version == SCHEMA_VERSION {
        return Ok(());
    }
    if version != 0 {
        anyhow::bail!(
            "database schema version {version} is not the supported version {SCHEMA_VERSION}; \
             the schema is pre-release and has no migration path. Delete the database file and \
             reindex."
        );
    }
    conn.execute_batch(&format!("BEGIN;\n{SCHEMA}\nCOMMIT;"))
        .context("applying initial schema")?;
    conn.execute(
        "INSERT INTO metadata(id, schema_version, created_at) VALUES (1, ?1, datetime('now'))",
        [SCHEMA_VERSION],
    )?;
    conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
    Ok(())
}
