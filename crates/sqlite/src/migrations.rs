use anyhow::{Context, Result};
use rusqlite::Connection;

/// Schema epoch. Older pre-release prototypes are intentionally rejected rather
/// than migrated. Epoch 4 dropped the unused `metadata` table and
/// `projects.role` column; epoch 5 split model identity into a semantic
/// contract (`embedding_models`) plus append-only execution provenance
/// (`model_executions`) and replaced `input_transform` with a typed
/// document-side contract; epoch 6 added analyzer-produced `relations`;
/// epoch 7 added the `lexical_index` FTS5 table for hybrid retrieval.
pub const SCHEMA_VERSION: i64 = 7;

const SCHEMA: &str = r#"
CREATE TABLE settings(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Operational indexing state. These rows are durable progress, but are never
-- joined by search or snapshot reads.
CREATE TABLE index_runs(
  id INTEGER PRIMARY KEY,
  base_generation INTEGER NOT NULL,
  status TEXT NOT NULL CHECK(status IN
    ('planning','running','ready','committed','paused','failed','superseded','abandoned')),
  phase TEXT NOT NULL CHECK(phase IN
    ('planning','refreshing','processing','ready','committed','paused','failed')),
  pause_reason TEXT,
  scope_json TEXT NOT NULL,
  config_json TEXT NOT NULL,
  config_fingerprint TEXT NOT NULL,
  payload_schema_version INTEGER NOT NULL,
  refresh_round INTEGER NOT NULL DEFAULT 0,
  owner_token TEXT,
  heartbeat_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  committed_at TEXT,
  last_error_json TEXT,
  stats_json TEXT NOT NULL
);
CREATE INDEX idx_index_runs_status_config
  ON index_runs(status, config_fingerprint, id DESC);

CREATE TABLE index_run_projects(
  run_id INTEGER NOT NULL REFERENCES index_runs(id) ON DELETE CASCADE,
  project_label TEXT NOT NULL,
  provider_locator TEXT NOT NULL,
  provider_fingerprint TEXT NOT NULL,
  manifest_digest TEXT NOT NULL DEFAULT '',
  last_refresh_at TEXT,
  stats_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY(run_id, project_label)
);

CREATE TABLE index_run_documents(
  run_id INTEGER NOT NULL REFERENCES index_runs(id) ON DELETE CASCADE,
  project_label TEXT NOT NULL,
  source_document_id TEXT NOT NULL,
  relative_path TEXT,
  language_id TEXT,
  source_revision_json TEXT,
  observed_source_hash TEXT,
  action TEXT NOT NULL CHECK(action IN ('unchanged','metadata','upsert','delete')),
  state TEXT NOT NULL CHECK(state IN ('pending','processing','ready','error')),
  input_fingerprint TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  reused INTEGER NOT NULL DEFAULT 0,
  payload_schema_version INTEGER,
  payload_json TEXT,
  error_json TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(run_id, project_label, source_document_id),
  FOREIGN KEY(run_id, project_label)
    REFERENCES index_run_projects(run_id, project_label) ON DELETE CASCADE,
  CHECK(
    (action = 'upsert' AND state = 'ready' AND payload_json IS NOT NULL
      AND payload_schema_version IS NOT NULL)
    OR (action = 'delete' AND state = 'ready' AND payload_json IS NULL)
    OR (action IN ('unchanged','metadata') AND state = 'ready' AND payload_json IS NULL)
    OR (state IN ('pending','processing','error') AND payload_json IS NULL)
  )
);
CREATE INDEX idx_index_run_documents_state
  ON index_run_documents(run_id, state, project_label, source_document_id);

CREATE TABLE projects(
  id INTEGER PRIMARY KEY,
  label TEXT NOT NULL UNIQUE,
  source_dir TEXT NOT NULL,
  created_at TEXT NOT NULL,
  last_index_run_id INTEGER REFERENCES index_runs(id)
);

CREATE TABLE files(
  id INTEGER PRIMARY KEY,
  project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  source_document_id TEXT NOT NULL,
  source_revision TEXT NOT NULL,
  relative_path TEXT NOT NULL,
  language_id TEXT NOT NULL,
  mtime_ns INTEGER NOT NULL,
  size INTEGER NOT NULL,
  source_hash TEXT NOT NULL,
  UNIQUE (project_id, source_document_id),
  UNIQUE (project_id, relative_path)
);
CREATE INDEX idx_files_source_hash ON files(source_hash);

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

CREATE TABLE representations(
  unit_id INTEGER NOT NULL REFERENCES code_units(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  content TEXT,
  origin_json TEXT NOT NULL,
  PRIMARY KEY (unit_id, kind)
);
CREATE INDEX idx_repr_channel_hash ON representations(kind, content_hash);

-- Semantic model contracts: every column changes vector meaning. Execution
-- environment lives in model_executions and is never part of identity.
CREATE TABLE embedding_models(
  id INTEGER PRIMARY KEY,
  model TEXT NOT NULL,
  revision TEXT,
  model_hash TEXT,
  tokenizer_hash TEXT,
  pooling TEXT NOT NULL,
  normalize INTEGER NOT NULL,
  native_dimensions INTEGER NOT NULL CHECK (native_dimensions > 0),
  max_sequence_length INTEGER NOT NULL,
  prompts_json TEXT NOT NULL,
  quantization TEXT,
  UNIQUE (
    model, revision, model_hash, tokenizer_hash, pooling, normalize,
    native_dimensions, max_sequence_length, prompts_json, quantization
  )
);

-- Append-only record of the environments a model has executed in.
-- Diagnostics only; never compared for space compatibility.
CREATE TABLE model_executions(
  id INTEGER PRIMARY KEY,
  model_id INTEGER NOT NULL REFERENCES embedding_models(id) ON DELETE CASCADE,
  backend TEXT NOT NULL,
  backend_version TEXT NOT NULL,
  runtime_version TEXT,
  execution_provider TEXT NOT NULL,
  cache_path TEXT,
  first_seen_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL
);
CREATE INDEX idx_model_executions_model ON model_executions(model_id);

CREATE TABLE embedding_spaces(
  space_id TEXT PRIMARY KEY,
  model_id INTEGER NOT NULL REFERENCES embedding_models(id) ON DELETE CASCADE,
  channel TEXT NOT NULL,
  document_side_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX idx_spaces_channel ON embedding_spaces(channel);
CREATE INDEX idx_spaces_model ON embedding_spaces(model_id);

CREATE TABLE embeddings(
  space_id TEXT NOT NULL REFERENCES embedding_spaces(space_id) ON DELETE CASCADE,
  content_hash TEXT NOT NULL,
  vector_blob BLOB NOT NULL,
  norm REAL NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (space_id, content_hash)
);

-- Typed, provenance-carrying relations produced by resolved analyzers (LSP,
-- SCIP) after publication, keyed by the generation they describe.
CREATE TABLE relations(
  id INTEGER PRIMARY KEY,
  generation INTEGER NOT NULL,
  from_entity_id TEXT NOT NULL,
  to_entity_id TEXT,
  to_symbol TEXT NOT NULL,
  kind TEXT NOT NULL,
  resolution TEXT NOT NULL CHECK(resolution IN ('exact','heuristic')),
  provenance TEXT NOT NULL
);
CREATE INDEX idx_relations_generation ON relations(generation);
CREATE INDEX idx_relations_from ON relations(from_entity_id);
CREATE INDEX idx_relations_to ON relations(to_entity_id);

-- Lexical retrieval over unit metadata and implementation text. rowid is
-- the code_units id; maintained explicitly by publication (FTS tables do
-- not participate in foreign-key cascades). tokenchars keeps identifiers
-- with underscores whole.
CREATE VIRTUAL TABLE lexical_index USING fts5(
  name, scope, path, content,
  tokenize = "unicode61 tokenchars '_'"
);

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
  space_id TEXT REFERENCES embedding_spaces(space_id),
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

/// Initialize a new database or reject any incompatible pre-release epoch.
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
    conn.execute_batch(&format!(
        "BEGIN;\n{SCHEMA}\nPRAGMA user_version = {SCHEMA_VERSION};\nCOMMIT;"
    ))
    .context("applying initial schema")?;
    Ok(())
}
