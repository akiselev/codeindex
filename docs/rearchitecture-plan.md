# Rearchitecture plan — typed embedding contract, generic models, candle/Qwen3, LSP relations

Status: **Phases 0–5 implemented at v1 depth** (2026-07-14). Phase 4 shipped
the `embed`/`search`/`models resolve`/`lsp-enrich` CLI commands with built-in
task presets; Phase 5 shipped `codeindex-lsp` (blocking stdio JSON-RPC client,
post-publish enrichment: hover → derived `typed_signature` channel,
`textDocument/definition` → exact `calls` relations in a new generation-keyed
`relations` table, surfaced through `IndexSnapshot.relations` and
`SearchIndex.relations`; integration-tested against a real rust-analyzer).
Follow-ups tracked in ROADMAP: masked batched inference for candle, GGUF
quantization, lexical FTS + RRF fusion in the CLI, the `Reranker` trait,
relation-aware query filters, and the decombine migration. Companion to `FEEDBACK.md` (the
model-landscape research brief). Implementation notes:

- Phase 0 shipped as schema epoch 4; Phase 1's identity split shipped as
  epoch 5 (`ModelContract` + `model_executions` provenance table,
  `document_side_json` replacing `input_transform`).
- Generic resolution (`crates/embedding/src/resolve.rs`) maps any
  sentence-transformers-configured repo onto a typed contract; a query prompt
  matching `Instruct: …\nQuery:` upgrades to the full instruction template
  automatically, which is how Qwen3-Embedding gets arbitrary task
  instructions with zero model-specific code.
- The candle backend (`embed/candle_backend.rs`, feature `candle`) executes
  Qwen3-architecture safetensors models with last-token pooling, batch=1
  (candle's qwen3 module has no padding-aware mask; masked batching is the
  next optimization), EOS-terminated inputs per the reference implementation,
  and `candle-cuda`/`candle-metal` device features.
- Conformance: `crates/embedding/tests/qwen3_conformance.rs` (`--ignored`)
  reproduces the README similarity matrix of Qwen3-Embedding-0.6B.

This plan was grounded in a full-workspace review; every claim below was
verified against `ed72dfa`. Sections 2–3 describe that pre-implementation
state and the design as built.

## 1. Goals

1. **Typed embedding contract.** Replace `Embedder::embed(&[String])` with a
   role-aware (Query vs Document), task-instruction-aware request type, so
   asymmetric instruction models (Qwen3-Embedding, Jina Code, CodeRankEmbed's
   own query prefix) are expressible. Agent interrogation features get dynamic
   per-query instructions ("Given a failure report, retrieve implementation and
   error-handling paths…") without re-embedding documents.
2. **Generic model support.** A HuggingFace repo path (or local dir) should be
   enough to run a model — contract (pooling, prompts, dims, max length)
   resolved from the repo's own machine-readable config, not from hand-written
   Rust enums and pinned const tables.
3. **Candle backend** for Qwen3-Embedding (last-token pooling, left padding,
   Matryoshka dims) — models fastembed/ort structurally cannot run (fastembed's
   `Pooling` enum is Cls/Mean only).
4. **Generic LSP support.** Drive language servers (and/or SCIP indexers) to
   extract typed relations and type-enriched representations, exposed to agent
   queries and embedded into parallel spaces.
5. **Streamlining.** Fix the bugs the review found, remove genuinely dead
   surface, and consolidate the documentation.

## 2. Review findings that force the design

### 2.1 The two structural flaws

**F1 — role-blind, string-typed embedding.** `Embedder::embed(&mut self,
&[String])` (`crates/embedding/src/embed/mod.rs:37`) is the single choke point.
`SearchIndex::search_text` (`crates/search/src/lib.rs:302`) embeds raw query
text identically to documents; `embed_space_pending` hard-rejects any
`input_transform != "identity"` (`crates/indexer/src/embed.rs:97`); the
`input_transform` string is otherwise never interpreted anywhere. Today this is
already a live quality bug: CodeRankEmbed (the default model) expects the query
prefix `Represent this query for searching relevant code:` and nothing in the
workspace applies it.

**F2 — model identity conflates vector semantics with execution provenance.**
`ModelIdentity` carries `cache_path`, `execution_provider`, `backend_version`,
`runtime_version` alongside the fields that actually determine vector meaning.
Three different comparison rules exist:

- `search_text` and `embed_space_pending` demand full-struct equality
  (`crates/search/src/lib.rs:295`, `crates/indexer/src/embed.rs:92`) — so
  upgrading fastembed/ort, switching cpu↔cuda, or moving the model cache makes
  every existing space unqueryable and un-resumable;
- `embedding_models` UNIQUE and `find_or_create_model` match on a 10-field
  subset *without* `runtime_version`/`cache_path`
  (`crates/sqlite/src/migrations.rs:170`, `crates/sqlite/src/lib.rs:499`) —
  the dedup key disagrees with the equality gates;
- meanwhile pooling and prompt contracts — which *do* change vector meaning —
  are not part of identity at all.

### 2.2 Bugs to fix regardless of the rearchitecture

| # | Location | Bug |
|---|---|---|
| B1 | `embedding/src/embed/fastembed_backend.rs:90` | Unknown pooling strings silently fall back to Mean (`"last_token"`, `"CLS"`, typos → wrong vectors, no error). |
| B2 | `embedding/src/embed/fastembed_backend.rs:139,168` | `quantized: true` silently ignored for managed and custom models; fp32 loads, identity records `quantization: None`. |
| B3 | `embedding/src/embed/fastembed_backend.rs:44,219` | Catalog models truncated at hardcoded 512 tokens; SnowflakeArcticEmbedMLong (2048) and NomicEmbedTextV15 (8192) are silently crippled. |
| B4 | `indexer/src/lib.rs:232` | `BodyWithoutDeclaredName` erases the first plain substring match of the name — `func (a *adder) add()` corrupts the receiver type instead of the name. Needs word-boundary/position anchoring. |
| B5 | `tree-sitter/src/language.rs:226` | `has_cfg_test_attribute` substring heuristic: `#[cfg(feature = "testing")]` → false positive; `#[cfg(all(test, not(windows)))]` → false negative. |
| B6 | `embedding/src/embed/fastembed_backend.rs:38` | `let _ = tokenizer.with_truncation(None)` — on failure `count_tokens_untruncated` silently returns truncated counts. |
| B7 | `embedding/src/embed/fastembed_backend.rs:149` | Custom-model identity `revision` embeds an absolute local path — same model at a different mount is a different identity. |
| B8 | `sqlite/src/index_runs.rs:419` | `heartbeat_run` exists but is never called; a document slower than the 30 s lease (guaranteed once LSP/LLM enrichers exist) gets stolen by a concurrent claim and the original run durably fails. |
| B9 | `sqlite/src/index_runs.rs:309` | `create_or_resume_run` supersedes overlapping *live* runs (fresh heartbeat) without an ownership check. |
| B10 | `search/src/lib.rs:154` | `CodeUnitRef.normalized_body_hash` is actually the Body/Implementation content hash, not the stored normalized body hash — selector stability quietly depends on channel presence. |
| B11 | `indexer/src/embed.rs:307` | Source recovery re-extracts with caller-supplied thresholds instead of the persisted index-time settings; drift silently counts rows `unresolved`. |
| B12 | `tree-sitter/src/language.rs:56` + `extractor.rs:300` | Python docstrings are stripped from `Implementation` but never captured into `Documentation` — the docs channel is empty for conventional Python. Same harvester loses Rust/Java docs separated from the declaration by attributes/annotations. |
| B13 | `embedding/build.rs:16` | `locked_version` textually scans Cargo.lock and can report the wrong `ort` version into the persisted identity when two versions are in the lock. |

Also worth noting (design gaps, not quick fixes): parse errors never surface
(`extract_units` ignores `root_node().has_error()` — the structured-diagnostics
plan already covers this); `prune_orphan_embeddings` is never called so orphan
vectors accumulate unboundedly; the RunConfig fingerprint includes operational
tuning (backoff, settle delay) so tuning changes discard staged work.

### 2.3 The decombine constraint

`decombine` consumes every crate by path (`/home/dev/projects/decombine`) and
uses much of what looks dead inside this workspace: `similar_pairs`,
`top_k_between`, `unit_line`, `WhereFilter::clauses`, `token_report`,
`accelerator_diagnostics`, `EmbeddingConfig::dimensions`, `SUPPORTED_MODELS`,
`EXECUTION_PROVIDERS`, `ManagedModel`, `list_models`, `create_analysis_run`,
`find_or_create_model_id`, `unit_indices_for_project`. **Do not delete these
without a decombine migration PR.** Genuinely dead in both trees:
`unique_space_for_channel`/`spaces_for_channel`,
`EmbeddingSpaceSnapshot::by_hash`, `ProjectRecord.role` (+ sqlite column),
`extract_file`/`ExtractedFile` diagnostics plumbing, the pre-journal sqlite
write API (`check_or_set_immutable`, `delete_project`, `get_file`,
`update_file_meta`, `delete_file`, `set_representation*`,
`clear_channel_for_project`, `insert_references`, `references_for_project`,
`get_model`, `get_embedding`, `prune_orphan_entities` free variant),
`SpaceChannel` trait, `metadata` table, `LanguageSpec.name`.

## 3. Target design

### 3.1 Typed embedding contract (codeindex-embedding)

```rust
pub enum EmbeddingRole { Query, Document }

/// A named retrieval intent plus the instruction text a model renders for it.
pub struct EmbeddingTask {
    pub id: String,               // "code-search", "locate-edit-targets", ...
    pub instruction: String,      // "Given a software change request, retrieve ..."
}

pub struct EmbedRequest<'a> {
    pub role: EmbeddingRole,
    /// Query-side task. None => the model's default query prompt (if any).
    pub task: Option<&'a EmbeddingTask>,
    pub inputs: &'a [&'a str],
    /// Matryoshka truncation; None => model's native dims.
    pub output_dimensions: Option<usize>,
}

pub trait EmbeddingBackend: Send {
    fn contract(&self) -> &ModelContract;     // semantic (see 3.2)
    fn execution(&self) -> &ExecutionInfo;    // provenance (see 3.2)
    fn count_tokens(&self, text: &str) -> Option<usize>;
    fn count_tokens_untruncated(&self, text: &str) -> Option<usize>;
    fn embed(&mut self, request: &EmbedRequest<'_>) -> Result<Vec<Vec<f32>>>;
}
```

Prompt rendering is a shared library function driven by the model contract —
not per-backend ad-hoc code — so fastembed, candle, and future HTTP backends
render identically:

```rust
pub enum PromptContract {
    /// Symmetric encoders (BGE, MiniLM): no role prompts.
    Symmetric,
    /// Qwen3 style: "Instruct: {task}\nQuery:{text}" on queries, documents raw.
    QueryInstruction { query_template: String, default_task: Option<String> },
    /// CodeRankEmbed/E5 style: fixed prefixes per role.
    RolePrefixes { query: String, document: String },
    /// Jina Code style: paired per-task templates on both sides.
    PairedTask { tasks: BTreeMap<String, (String, String)> },
}
```

Rules that keep the content-addressed vector store intact:

- **Document-side prompts are applied at embed time, never baked into stored
  representation content.** Vectors stay keyed by `(space_id, content_hash)` of
  the *representation* text; the rendered form is recomputed deterministically.
  The document-side prompt (usually empty for Qwen3) is hashed into the space's
  semantic identity, so a space cannot silently change document rendering.
- **Query-side instructions never enter space identity.** They vary per query
  and are reported in results for reproducibility. One document index serves
  many query intents — this is the Qwen3 payoff for interrogation features.
- Token accounting (`count_tokens`, batch packing area) measures the *rendered*
  input, not the raw representation (Qwen3 instructions are long).
- `pack_batches`, `TokenStats`, `normalize_in_place` are reused as-is; they are
  padding-side-agnostic.

Reranking (`Qwen3-Reranker`) is a separate `Reranker` trait added when Phase 4
lands — never modeled as an embedder (see FEEDBACK.md §5 for the shape).

### 3.2 Identity split (codeindex-core + schema epoch 5)

```rust
/// Determines vector meaning. Gates space compatibility, resume, and querying.
pub struct ModelContract {
    pub model: String,                 // "Qwen/Qwen3-Embedding-0.6B" or catalog name
    pub revision: Option<String>,      // resolved commit, not "main"
    pub model_hash: Option<String>,
    pub tokenizer_hash: Option<String>,
    pub pooling: Pooling,              // Mean | Cls | LastToken
    pub normalize: bool,
    pub native_dimensions: usize,
    pub max_sequence_length: usize,
    pub prompts: PromptContract,
    pub quantization: Option<String>,  // quantization changes vectors → semantic
}

/// Provenance only. Persisted for diagnostics, never compared for compatibility.
pub struct ExecutionInfo {
    pub backend: String,               // "fastembed" | "candle" | ...
    pub backend_version: String,
    pub runtime_version: Option<String>,
    pub execution_provider: String,    // "cpu" | "cuda" | "metal" | ...
    pub cache_path: Option<String>,
}
```

`EmbeddingSpaceIdentity` becomes:

```rust
pub struct EmbeddingSpaceIdentity {
    pub id: EmbeddingSpaceId,
    pub channel: RepresentationKind,
    pub model: ModelContract,
    /// Replaces `input_transform: String`.
    pub document_side: DocumentSideContract, // document prompt/task + output_dimensions
}
```

`output_dimensions` lives on the *space* (Matryoshka is a projection choice),
`native_dimensions` on the model. Vector validation and
`ensure_query_dimensions` check the space's effective dims; truncation
re-normalizes (Qwen3 requirement).

Storage changes (pre-release schema: bump to epoch 5 — Phase 0 already took 4, reject-and-reindex per
existing policy):

- `embedding_models` splits into semantic columns (UNIQUE over all of them)
  plus a `model_executions` provenance table (or a JSON provenance column);
- `embedding_spaces.input_transform` → `document_side_json`;
- comparison rule everywhere becomes: **semantic equality only**, and
  `identity_diff` is derived mechanically (serde field diff) so new fields
  can't be silently omitted (`query/src/lib.rs:166` is hand-maintained today).

Snapshot wire format changes accordingly (`IndexSnapshot` is serde; additive
where possible, but the identity split is a breaking change we take now, while
pre-release).

### 3.3 Generic model resolution (new module `codeindex-embedding::resolve`)

Model references become a small grammar instead of an enum:

```text
hf:Qwen/Qwen3-Embedding-0.6B          # HuggingFace repo, default revision
hf:Qwen/Qwen3-Embedding-0.6B@<rev>    # pinned revision
dir:/path/to/exported-model           # local directory
fastembed:BGESmallENV15               # legacy catalog passthrough (decombine compat)
```

Resolution pipeline for `hf:`/`dir:` refs — fetch (hf-hub 1.0, `HFClientSync`)
and parse, in priority order:

1. `config_sentence_transformers.json` → `prompts.query` / `prompts.document`,
   similarity fn (Qwen3-0.6B ships exactly this: query = `"Instruct: Given a
   web search query, …\nQuery:"`, document = `""`);
2. `modules.json` + `1_Pooling/config.json` → pooling
   (`pooling_mode_lasttoken: true` for Qwen3) and Normalize module presence —
   this is the same source TEI uses in production, it is reliable;
3. `config.json` / `tokenizer_config.json` → architecture, hidden size,
   max length, padding side;
4. an optional `codeindex.toml` override (user-authored manifest) for models
   whose repos lack the sentence-transformers trio, plus per-space task
   defaults.

The output is a `ModelContract` + artifact list. A **lockfile**
(`<cache>/models/<sanitized-ref>/codeindex.lock.json`) records the resolved
revision and per-file SHA256 on first download (trust-on-first-use), replacing
today's compile-time pinned `MANAGED_MODELS` table. The existing
`ensure_managed_files`/`download_and_verify` machinery
(`fastembed_backend.rs:361-450`) is already generic over repo/revision/files —
it moves into the resolver nearly unchanged, gains hf-hub as transport, and
keeps hash verification.

Backend selection is capability-based: candle takes safetensors +
supported-architecture models; fastembed takes ONNX exports with Cls/Mean
pooling; a model neither can run fails with a *capability* error, not a
"model not in enum" error. `SUPPORTED_MODELS`, the 14-arm `resolve_model`
match, and `MANAGED_MODELS` all disappear behind the `fastembed:` passthrough.

Cache root renames `decombine` → `codeindex` (`XDG_CACHE_HOME/codeindex/models`)
with a read-through fallback to the old path so existing downloads survive.

### 3.4 Candle backend (feature `candle`)

Ecosystem facts (verified July 2026):

- `candle-transformers` 0.11 ships `qwen3` and `quantized_qwen3` (GGUF);
  `Model::forward` returns final-RmsNorm hidden states → last-token pooling is
  implementable. **Caveat:** the module builds causal masks only — no
  padding-aware attention mask — so batched left-padded embedding needs a
  custom mask (or batch=1 initially).
- Reference implementations to crib from: HF `text-embeddings-inference`
  (Rust/candle, supports Qwen3-Embedding with last-token pooling, parses
  `1_Pooling/config.json`), fastembed-rs's candle-backed `qwen3` feature, and
  the `gte-qwen` candle example (decoder-based embedder).
- hf-hub 1.0.0 (2026-07-10) — note breaking API vs the 0.3/0.4 `Api` style in
  older candle examples. `tokenizers` 0.23 is the same crate fastembed uses, so
  `count_tokens*` ports unchanged.

Implementation order inside the backend:

1. fp32/bf16 safetensors, batch=1 last-token pooling, CPU — correctness first;
2. left-padded batching with an explicit attention mask (port TEI's approach)
   so `pack_batches` budgets stay meaningful;
3. `cuda`/`metal` cargo features (accelerator diagnostics stay ort-scoped;
   candle acceleration is compile-time, reported via `ExecutionInfo` only);
4. GGUF/quantized Qwen3 later — quantization enters `ModelContract` so
   quantized vectors never mix with fp32 spaces.

**Conformance gate:** golden-vector tests. Check in reference vectors produced
by Python sentence-transformers for a small fixture corpus (queries with
instructions + documents); the candle backend must match within tolerance and
must reproduce ranking order. Run in CI behind a `-- --ignored`/scheduled job
(downloads the 0.6B model). This is also the acceptance test for any future
backend (TEI/OpenAI-compatible HTTP, llama.cpp).

An HTTP backend (TEI / OpenAI-compatible) is deliberately *not* in scope for
the first pass but the contract is designed so it drops in later — it would
give early end-to-end Qwen3 validation on machines where candle compile times
or GPU setup are a problem.

### 3.5 Search and query surface

- `SearchIndex::search_text(embedder, text, space_id, filter, limit)` grows a
  `QueryOptions { task: Option<&EmbeddingTask>, output_dimensions: … }`
  parameter (or a `SearchRequest` struct); it renders the query through the
  space's model contract, and the result envelope records the instruction used.
- The equality gate becomes `ModelContract` equality; `ExecutionInfo`
  differences are at most a warning.
- `from_snapshot` validates the space's `document_side` contract (today
  `input_transform` is silently ignored — F1) and rejects unknown variants.
- CLI (this is ROADMAP M2, unblocked by generic resolution): `codeindex embed
  --space code=implementation --model hf:Qwen/Qwen3-Embedding-0.6B`,
  `codeindex search "<text>" --space code --task locate-edit-targets --json`,
  `codeindex models resolve|inspect|doctor`. Stable JSON envelopes as today.
- Named task presets ship as data (a small built-in table + user config), so
  agents can say `--task locate-edit-targets` instead of hand-writing
  instructions; arbitrary `--instruction "…"` stays available.

### 3.6 LSP relations and type-enriched spaces (new crate `codeindex-lsp`)

Ecosystem choice: **async-lsp 0.2** (actively maintained, explicitly supports
the client side, tower middleware, stdio transport; tower-lsp is dead and its
fork is server-only; Zed hand-rolls — viable but more code). **SCIP** (`scip`
crate 0.9 + `rust-analyzer scip .`) is the batch-mode complement: whole-repo
symbols/relationships without server lifecycle management. Kiro's agent-facing
code intelligence (spawn stock servers per language, translate agent queries to
LSP requests) is the closest prior art for the interrogation surface.

Architecture (keeps determinism/resume invariants intact):

1. **Do not put LSP inside the per-document staging loop.** Staged payloads are
   a pure function of (config, single document content); LSP facts are
   cross-file. Follow the Usage precedent: LSP extraction runs as a
   **post-publish, generation-keyed analysis pass** (`codeindex-lsp` walks the
   published corpus, talks to servers, writes derived facts for generation G).
   The `RepresentationEnricher` seam stays for per-document deterministic
   enrichment only.
2. **Products:**
   - a `TypedSignature` representation channel (origin
     `Derived { producer: "lsp:<server-id>", version: <server version> }`)
     — hover/type-of-symbol enriched signatures. Parallel embedding space
     `types = TypedSignature × <model>` comes free via `embed_space_pending`;
   - typed relations in a new `relations` table:
     `{from_entity_id, to (entity_id | external symbol), kind
     (defines/references/implements/type-of/calls), provenance, resolution
     (exact/heuristic), generation}` — superseding the Rust-only name-based
     `references_raw` resolver as servers become available (11 of 12
     `references.scm` files are empty placeholders today, so LSP mostly
     *replaces* rather than duplicates per-language query work);
   - `IndexSnapshot` gains a `relations` section keyed by stable entity ids;
     `codeindex-query` gains relation-aware filters (`calls:`, `implements:`)
     — the current whitespace-split `key=value` grammar has no room for this
     and will need a small extension.
3. **Server management:** config-driven registry (`language id → command,
   args, init options`), spawn on demand per project root, health/timeout
   handling, graceful degradation to tree-sitter facts when no server exists.
   SCIP ingestion is an alternative provider of the same relation records.
4. **Prerequisite fixes** (from the review): B8 heartbeats (long analysis
   passes must not get leases stolen), byte↔UTF-16 position mapping helper
   (`SourceSpan` has no columns; LSP speaks line+UTF-16), and revisiting the
   drop-small-entities policy (`extractor.rs:195` — entities below the node
   threshold don't exist, so type facts would have nothing to attach to).

The retrieval pipeline vision (lexical FTS5, RRF over lexical+dense, rerankers,
graph expansion, context packing — FEEDBACK.md §6-8) layers on top of these
relations; it is out of scope here but nothing in this plan blocks it.

## 4. Phasing

**Phase 0 — cleanup + bug fixes (no schema change).**
Fix B1-B7, B10-B13 (B8/B9 land with Phase 5 if not sooner). Delete the
both-trees-dead surface (§2.3 list). Extract duplicated helpers (manifest
digest, immutable-settings list, cache-dir resolution, CLI envelope). Hoist
`config_fingerprint` out of the per-document loop; batch `insert_embedding` per
page in one transaction. Doc consolidation (already applied alongside this
plan). Exit: workspace green, decombine still compiles.

**Phase 1 — contract + identity split (schema epoch 5).**
New `ModelContract`/`ExecutionInfo`/`PromptContract`/`EmbedRequest`; rendering
layer; `EmbeddingSpaceIdentity.document_side`; sqlite epoch 5; semantic-only
comparisons; mechanical `identity_diff`; fastembed backend ported (gains
RolePrefixes so CodeRankEmbed queries finally get their prefix). decombine
migration PR for the renamed types. Exit: existing models work through the new
contract; a query embedded with a task instruction hits an unchanged document
index; golden tests for rendering.

**Phase 2 — generic resolution.**
`ModelRef` grammar, hf-hub fetch, sentence-transformers config parsing,
manifest override, TOFU lockfile; `SUPPORTED_MODELS`/`resolve_model`/
`MANAGED_MODELS` collapse into the resolver + `fastembed:` passthrough; cache
root rename. Exit: `codeindex models resolve hf:Qwen/Qwen3-Embedding-0.6B`
prints a correct contract (last-token pooling, 1024 dims, query prompt) without
any code change for that model.

**Phase 3 — candle backend.**
As §3.4. Exit: Qwen3-Embedding-0.6B embeds a corpus and answers
instruction-tasked queries end-to-end on CPU; golden-vector conformance passes;
cuda/metal features compile.

**Phase 4 — query surface (M2).**
`QueryOptions`/task presets, CLI `embed`/`search`/`models` commands, stable
JSON result envelopes with instruction + space contributions recorded.
(Reranker trait + Qwen3-Reranker can ride along here or immediately after.)

**Phase 5 — LSP relations.**
As §3.6, starting with rust-analyzer (richest server, plus `rust-analyzer scip`
for the batch path) and one non-Rust server (pyright or gopls) to keep the
abstraction honest. Exit: `relations` in snapshots, one type-enriched space
fused into search, agent-visible `calls`/`implements` filters.

Order rationale: FEEDBACK.md's closing warning is correct — once document
vectors exist under an underspecified prompt/pooling contract, fixing the
abstraction is an index migration. Contract first, runtimes second, surfaces
third, relations fourth.

## 5. Decisions (resolved with maintainer, 2026-07-14)

1. **decombine timing: lockstep migration.** Each breaking phase lands with a
   matching decombine PR; no compat shims.
2. **fastembed: kept as the zero-config CPU/ONNX tier** behind the
   `fastembed:` model-ref scheme.
3. **Default model: none.** The library API requires an explicit model ref —
   `EmbeddingConfig`'s implicit `CodeRankEmbed` default goes away in Phase 1/2
   and the CLI requires `--model`. decombine will choose its own default
   downstream later.
4. **`ExecutionInfo` persistence: separate provenance table**
   (`model_executions`), append-only; `embedding_models` holds semantic
   contract columns only, UNIQUE over all of them.
5. **Doc dedup: after Phase 1**, so the owner-per-topic rewrite happens once,
   against the renamed types.
