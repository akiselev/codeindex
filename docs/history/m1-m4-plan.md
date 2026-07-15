# Implementation plan — M1 (standalone search API) & M4 (multi-representation & entity versions)

> **Historical design record — superseded.** This work shipped in commits
> `22817c3`…`e9aa8df`. The wire types, SQL keys, and search signatures shown
> below no longer match the code (e.g. `IndexSnapshot` now carries named
> `spaces`, embeddings are keyed by `(space_id, content_hash)`, and search takes
> a `space_id`, not a channel). Current contracts live in
> `docs/architecture.md`.

Status: proposed. Scope: implement M1 and M4 **in full**, with two overriding
constraints from the maintainer:

1. **No migration story.** Nothing is released; the SQLite schema is rewritten
   to whatever is optimal. `migrations.rs` becomes a single clean bootstrap, not
   an append-only ledger.
2. **Storage independence.** Any store other than SQLite must be supportable by
   *deserializing a public API type* — no store-specific code in the search
   engine. This is the through-line that ties M1 and M4 together.

Decisions taken (maintainer, 2026-07-11):
- **Usage channel: full extraction now** — resolve call sites across the corpus,
  not just reserve the channel.
- **Snapshot types live in a new `codeindex-storage` crate** (serde-only leaf).

---

## 1. The unifying idea

Both milestones push on the same seam — the boundary between *storage* and the
*search engine*. Today `codeindex-search` reaches through that seam with raw SQL
(`SearchIndex::load(&Db, …)`, `load_projects_and_units`) and depends on
`codeindex-sqlite` + `rusqlite`. M4's richer data model would otherwise deepen
that coupling.

Instead: define a storage-neutral, `serde`-serializable **`IndexSnapshot`** that
the engine consumes. SQLite becomes one producer of it; any other backend
(Postgres, a JSON file, an in-memory fixture, a remote service) produces the same
type by its own means and hands it to `SearchIndex::from_snapshot`. M4's
multi-channel + entity-version shape is designed into that public type from the
start.

```
                     ┌──────────────────────┐
 codeindex-sqlite ──▶│                      │
 (Db::snapshot)      │   IndexSnapshot      │──▶ SearchIndex::from_snapshot
 any other backend ─▶│  (codeindex-storage) │        (codeindex-search)
 (serde deserialize) │                      │
                     └──────────────────────┘
```

---

## 2. New crate: `codeindex-storage`

A serde-only leaf (depends on `codeindex-core` + `serde`; **no** `rusqlite`,
no parsers). Holds the public wire types the engine loads from.

```rust
#[derive(Serialize, Deserialize)]
pub struct IndexSnapshot {
    pub model: ModelIdentity,
    pub projects: Vec<ProjectRecord>,
    pub units: Vec<UnitRecord>,
    /// One entry per representation channel that has embeddings.
    pub channels: Vec<ChannelEmbeddings>,
}

#[derive(Serialize, Deserialize)]
pub struct ProjectRecord {
    pub label: String,
    pub source_dir: String,
    pub role: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct UnitRecord {
    pub entity_id: String,          // stable logical identity across generations (M4)
    pub entity_version_id: String,  // exact identity of THIS indexed version    (M4)
    pub generation: u64,
    pub project_label: String,
    pub relative_path: String,
    pub language_id: String,
    pub kind: String,
    pub name: String,
    pub scope: Option<String>,
    pub span: SpanRecord,           // start/end byte + line
    pub body_node_count: usize,
    /// All representation channels for this unit; `content` is None under
    /// report/minimal retention (recoverable from source).
    pub representations: Vec<RepresentationRef>,
}

#[derive(Serialize, Deserialize)]
pub struct RepresentationRef {
    pub kind: RepresentationKind,   // core enum, now Serialize + as_str/FromStr
    pub content_hash: String,
    pub content: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct ChannelEmbeddings {
    pub channel: RepresentationKind,
    pub dimensions: usize,
    pub vectors: Vec<(String /* content_hash */, Vec<f32>)>,
}
```

Loading the full corpus into memory matches what `SearchIndex::load` already
does, so there is no regression. A streaming `IndexReader` trait for very large
corpora is a **later** option, explicitly out of scope here; `IndexSnapshot` is
the canonical contract.

`RepresentationKind` gains `as_str`/`FromStr`/`Serialize`/`Deserialize` in
`codeindex-core` (mirroring `EntityKind`) so it is a stable persisted/serialized
key, not just an in-memory enum.

---

## 3. Data model (M4) — SQLite rewritten clean

Drop the append-only migration array; write one bootstrap schema. Replace the two
flat text columns (`display_source`, `embedding_text`) with a representations
table, key embeddings by channel, and add the entity ledger.

```sql
-- unchanged in spirit: projects, files

CREATE TABLE entities (          -- logical identity ledger (M4)
  entity_id        TEXT PRIMARY KEY,
  project_id       INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind             TEXT NOT NULL,
  first_generation INTEGER NOT NULL,
  last_generation  INTEGER NOT NULL
);

CREATE TABLE code_units (
  id                 INTEGER PRIMARY KEY,
  file_id            INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  entity_id          TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
  entity_version_id  TEXT NOT NULL,
  generation         INTEGER NOT NULL,
  language_id TEXT NOT NULL, kind TEXT NOT NULL, name TEXT NOT NULL, scope TEXT,
  start_byte INTEGER NOT NULL, end_byte INTEGER NOT NULL,
  start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
  body_node_count INTEGER NOT NULL,
  source_hash TEXT NOT NULL,
  CHECK (start_byte < end_byte), CHECK (start_line <= end_line)
);

CREATE TABLE representations (   -- N channels per unit (M4)
  unit_id      INTEGER NOT NULL REFERENCES code_units(id) ON DELETE CASCADE,
  kind         TEXT NOT NULL,    -- RepresentationKind::as_str
  content_hash TEXT NOT NULL,
  content      TEXT,             -- NULL under retention; re-derive from source
  PRIMARY KEY (unit_id, kind)
);
CREATE INDEX idx_repr_hash ON representations(kind, content_hash);

CREATE TABLE embeddings (        -- rekeyed by channel (M4)
  model_id     INTEGER NOT NULL REFERENCES embedding_models(id) ON DELETE CASCADE,
  channel      TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  vector_blob  BLOB NOT NULL, norm REAL NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY (model_id, channel, content_hash)
);

CREATE TABLE references_raw (    -- Usage staging (see §6)
  caller_unit_id INTEGER NOT NULL REFERENCES code_units(id) ON DELETE CASCADE,
  callee_symbol  TEXT NOT NULL,
  call_snippet   TEXT NOT NULL,
  start_line     INTEGER NOT NULL
);
CREATE INDEX idx_refs_callee ON references_raw(callee_symbol);
```

`embedding_models`, `settings`, `analysis_runs`/`analysis_artifacts` carry over.

Model-layer changes (`crates/sqlite/src/models.rs`):
- `NewCodeUnit` carries `entity`/`version`/`generation` + a `Vec<Representation>`
  (all channels) instead of `display_source`/`embedding_text`.
- Replace `impl From<ExtractedEntity> for NewCodeUnit` (which hard-codes the two
  channels) with a mapping that copies **every** representation the frontend
  emitted. This stays the single place extraction maps onto rows.
- New `Db` methods: `insert_representations`, `all_channel_embeddings(model, channel)`,
  `unembedded_hashes(model, channel)`, `insert_embedding(model, channel, hash, vec)`,
  `snapshot(labels) -> IndexSnapshot`, plus the entity/reference helpers below.

---

## 4. Entity identity & versions (M4)

Add a monotonic `generation` (one per index run; stored in `settings`,
incremented at run start). On re-indexing a file, before its old units are
cleared, match new units to prior ones **within the same file**:

1. **Exact carry-forward** — same `(kind, scope, name)` → reuse `entity_id`.
2. **Rename detection** — same `Implementation` `content_hash` + same `kind`
   (body identical, name changed) → reuse `entity_id`.
3. Otherwise **mint** a fresh `entity_id` (`uuid`/hash of
   `project|path|kind|scope|name|generation`).

`entity_version_id = sha256(entity_id | source_hash | span)` — changes whenever
the code changes. `entities.last_generation` is bumped on every sighting;
`first_generation` is set at mint. This satisfies M4's exit: re-indexing a
renamed function preserves its `entity_id` while assigning a new
`entity_version_id`.

Cross-**file** moves are out of scope for v1 (within-file matching only) and
noted as a follow-up (would need corpus-level body-hash matching).

---

## 5. Definition-site channels (M4, Phase 4a)

Extend the tree-sitter frontend so `build_unit` emits, in addition to
`FullSource` + `Implementation`:

- **`Signature`** — declaration text with the body span removed (Rust: the
  `function_item` minus its `block`). Purely a span subtraction we already have
  (`body_span`), plus comment stripping.
- **`Documentation`** — leading doc comments (`///`, `/** */`, `#[doc]`). Rust
  gathers preceding `line_comment`/`block_comment` doc siblings; Python already
  isolates docstrings in `PythonAdapter` (reuse that range as the Documentation
  channel instead of only stripping it).
- **`Symbol`** — the qualified name string `scope::name` (cheap; no parse).

Each channel gets its own `content_hash`. Channels a language cannot yet produce
are simply absent for that language (graceful degradation). Rust is the reference
implementation; `Signature`/`Symbol` generalize trivially, `Documentation` per
language as adapters allow.

---

## 6. Usage channel — full cross-corpus extraction (M4, Phase 4b)

Usage is a synthesized record of **where an entity is called**, so it is a
whole-corpus, two-pass operation (the callee's Usage text depends on caller files
elsewhere in the corpus).

**Pass 1 (per file, extends existing extraction).** Add a reference capture —
either extra patterns in `units.scm` or a sibling `references.scm` — that
captures `call_expression` callees and argument snippets. For each reference,
record `(caller_unit, callee_symbol, call_snippet, line)` into `references_raw`.
Also build the definition symbol table: `symbol_key -> entity_id`, where
`symbol_key` is the resolvable name (qualified where possible, bare name
otherwise). Rust first; other languages contribute no references until their
query is written (empty Usage, not an error).

**Pass 2 (whole corpus, after all files in the run are indexed).** Resolve each
`references_raw.callee_symbol` to defining `entity_id`(s) via the symbol table
(same-project, name-based; ambiguous names resolve to all candidates or are
dropped — resolution is explicitly best-effort/heuristic for v1). For each
resolved callee entity, assemble its **Usage** representation: a deterministic
document of its call sites (`caller qualified name` + `call_snippet`, sorted).
Hash the assembled text → `content_hash`; write it as the entity's `Usage`
representation; it embeds like any other channel.

**Recompute policy.** Because changing a caller changes a callee's Usage, Pass 2
recomputes Usage for the touched projects at the end of each index run (bounded,
deterministic). Incremental invalidation (only recompute callees whose caller set
changed) is a follow-up optimization, not v1.

Import/use-aware and cross-crate resolution are future refinements; v1 documents
the name-based limitation.

---

## 7. Embedding pipeline (channel-aware)

`crates/indexer/src/embed.rs`:
- `embed_pending` iterates the set of channels to embed (default: all channels
  present; configurable). For each channel it pages `unembedded_hashes(model,
  channel)`, recovers missing text from source per channel (Signature/Symbol are
  cheap; Usage text is stored, so no recovery needed), packs, embeds, and writes
  `insert_embedding(model, channel, hash, vec)`.
- Immutable settings (`embedding.model`, `dimensions`, `normalize`) unchanged.
- The batch packer already handles short channels (`Signature`, `Symbol`) fine.

---

## 8. Search API (channel-aware) — M1 completion

`crates/search/src/lib.rs`:
- **Decouple from SQLite:** drop the `codeindex-sqlite`/`rusqlite` deps; depend on
  `codeindex-storage`. `CodeUnitRef` is built from `UnitRecord`; keep the
  `UnitView` impl and `unit_id`/selectors unchanged (M1's selector work stands).
- `SearchIndex::from_snapshot(&IndexSnapshot) -> SearchIndex`, holding **one
  `VectorStore` per embedded channel** (keyed by `RepresentationKind`).
- `search_text`/`search_vector`/`similar_to_unit` take a
  `channel: RepresentationKind` (default `Implementation`) selecting which
  `VectorStore` to rank against — this is M4's "a query can target a channel."
- Keep `SearchIndex::load(&Db, labels)` as a thin convenience wrapper
  (`Db::snapshot` → `from_snapshot`) behind a `sqlite` feature / re-export, so
  existing callers and the round-trip test keep working.
- `WhereFilter`, `rank_candidates`, `identity_diff` unchanged.

Result: the "any database" claim is real — a backend implements nothing but
"produce an `IndexSnapshot`."

---

## 9. Phases & sequencing

| Phase | Work | Exit |
|------|------|------|
| 1 | `codeindex-storage` crate; `IndexSnapshot` + records; `RepresentationKind` serde/as_str; `SearchIndex::from_snapshot`; `Db::snapshot`; drop search→sqlite dep. | Search runs from a snapshot; round-trip green. |
| 2 | Rewrite `migrations.rs` (single bootstrap); `representations`/`entities`/`references_raw` tables; channel-keyed `embeddings`; `models.rs` + `Db` CRUD. | New schema; sqlite tests green. |
| 3 | `generation` counter + within-file entity matcher; populate `entity_id`/`entity_version_id`. | Rename preserves `entity_id`. |
| 4a | Signature/Documentation/Symbol extraction (Rust ref impl). | 3 new channels stored for Rust. |
| 4b | Reference capture (`references.scm`), `references_raw`, Pass-2 resolver, Usage representations. | Usage channel populated & searchable. |
| 5 | Channel-aware `embed_pending`; multi-`VectorStore` `SearchIndex`; `search_*(channel,…)`. | `query --channel signature` works. |
| 6 | Tests + docs (below); tick M1/M4 in ROADMAP. | CI green; docs current. |

---

## 10. Testing

- **Snapshot serde round-trip** — `IndexSnapshot` → JSON → back is identical.
- **"Any database" proof** — build an `IndexSnapshot` by hand (no SQLite) and run
  `search_text` against it. This is the executable form of the M1 constraint.
- **Per-channel search** — a corpus where a query ranks differently by channel
  (signature vs. documentation vs. implementation).
- **Entity identity** — index a function, rename it, re-index; assert same
  `entity_id`, different `entity_version_id`.
- **Usage** — two files, A calls B; assert B's Usage representation contains A's
  call site and embeds/ranks.
- Extend `crates/search/tests/round_trip.rs` to the snapshot path + channels.

## 11. Docs to update

- `docs/architecture.md` — new crate, snapshot boundary, channel/entity schema.
- `ROADMAP.md` / `TODO.md` — mark M1 & M4 done; move the "dormant core
  vocabulary" P2 item to resolved.

## 12. Explicit non-goals / follow-ups

- Streaming `IndexReader` for corpora too large to hold in memory.
- Cross-file / cross-crate entity move tracking.
- Import-aware, cross-crate Usage resolution (v1 is same-project, name-based).
- Incremental Usage invalidation (v1 recomputes per run for touched projects).
