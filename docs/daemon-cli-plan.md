# Daemon architecture and final CLI design

Status: M-daemon-1 implemented (2026-07-15) — registry, add/remove/list/status/reindex, daemon-routed search with per-role model lanes, .codeindex.toml discovery. M2+: fs-watch, LSP pool, folder-override fingerprinting, MCP. Companion to `docs/rearchitecture-plan.md`;
this document designs the end-state user/agent interface: a single stateful
background daemon owning indexing, embedding, LSP servers, and query
serving, with a thin CLI that resolves projects by walking up to
`.codeindex` and layered per-folder configuration.

## 1. Why a daemon

The one-shot CLI pays fixed costs on every invocation that a resident
process pays once:

| Cost | One-shot today | Daemon |
|---|---|---|
| Embedding model load | 10-20 s per `search` | once, kept warm |
| Reranker load | ~15 s per `--rerank` | once, kept warm |
| LSP server startup + workspace load | per `lsp-enrich` run | persistent server pool |
| Snapshot load + validation | per query | warm `SearchIndex`, invalidated by generation |
| Freshness | manual `index` runs | filesystem watch → debounced incremental reindex |

The LSP angle is decisive: language servers are *designed* to be
long-lived. A daemon keeps one rust-analyzer/clangd/pyright per project
warm, so enrichment happens continuously (post-publish, generation-keyed,
as today) instead of paying workspace load on every pass — and agent
queries like "callers of X" can be answered live against the warm server
when the stored relations are stale.

## 2. Process model

One **user-level daemon** (`codeindex daemon`, hosted by the same binary), many registered project
roots. Not one daemon per project: models are the expensive resident state
and are shared across projects; SQLite write serialization stays
per-project (one DB per root, as today).

- **Lifecycle runtime**: daemonkit (`Daemon::embedded` in the parent,
  `Bootstrap::detect()` + `run_embedded_fn` in the same binary's daemon
  path). daemonkit owns instance identity, keyed private state, startup
  serialization, endpoint discovery, socket authentication, and shutdown;
  codeindex owns only the application protocol on the authenticated
  streams it is handed.
- **Protocol**: JSON-RPC 2.0 with `Content-Length` framing over the
  authenticated stream — the same framing as LSP, trivially bridgeable to
  MCP for agents.
- **Autostart**: `Daemon::ensure()` attaches to a live compatible instance
  or transactionally starts one; explicit `codeindex daemon
  start|stop|status` for control.
- **Registry**: `~/.local/share/codeindex/registry.json` — the list of
  added roots with their DB paths and watch state. The daemon rebuilds all
  runtime state from the registry + per-project DBs at startup; the
  registry is the only global mutable file.
- **Versioning**: the daemon embeds the workspace version; a CLI with a
  different version asks the daemon to restart itself (single-user, so
  drain-and-exec is fine). Schema-epoch mismatches surface per project as
  status, never crash the daemon.

## 3. Project resolution: `.codeindex.toml`

There is exactly one config file name — `.codeindex.toml` — and no state
directory in the tree. The file is both the project marker and the
configuration; the database lives outside the repository:

```text
myrepo/
  .codeindex.toml      # project root: marker + root configuration
  src/
    .codeindex.toml    # optional per-folder override (committed)
  vendor/
    .codeindex.toml    # e.g. index = false

~/.local/share/codeindex/
  registry.json                  # registered roots -> project state
  projects/<key>/index.db        # per-project database (key = root path hash)
```

- **Resolution**: commands walk up from the CWD collecting every
  `.codeindex.toml` on the ancestor path. The **topmost** file found is the
  project root; the files below it form the override chain. A file with
  `root = true` stops the upward walk early (nested independent projects,
  EditorConfig-style). Nearest-file-wins would misanchor a project at a
  subfolder override like `vendor/`, which is why the walk continues to the
  filesystem root. `--project <path|label>` overrides discovery.
- **Layered config**: effective config for a file = root config merged with
  every override on the path from root to the file, nearest-wins per key.
  Overrides may only *narrow or tune* (exclude, language set, retention,
  chunking thresholds, test policy, LSP server choice); identity-level keys
  (space definitions, models) are root-only so one project cannot fragment
  into incompatible vector spaces.
- The override chain is hashed into the per-document input fingerprint, so
  editing a folder's `.codeindex.toml` invalidates exactly that subtree's
  staged work — the journal machinery already supports this via
  `config_fingerprint`.
- **No repository pollution**: the database (and its WAL/SHM siblings) never
  sit in the tree, so there is nothing to gitignore. `[storage] path` in the
  root `.codeindex.toml` opts back into an explicit location.

### root `.codeindex.toml` sketch

```toml
[index]
languages = ["rust", "python"]        # default: all bundled
exclude = ["target/**", "vendor/**"]
retention = "full"

[spaces.code]                          # root-only
channel = "implementation"
model = "hf:Qwen/Qwen3-Embedding-0.6B"

[spaces.types]
channel = "typed_signature"
model = "hf:Qwen/Qwen3-Embedding-0.6B"

[search]
default_space = "code"
tests = "exclude"                      # include|exclude|only
retrieval = "hybrid"
rerank = false                         # opt-in; model below
rerank_model = "hf:Qwen/Qwen3-Reranker-0.6B"

[lsp.rust]
server = "rust-analyzer"

[lsp.c]
server = "clangd"

[tasks.locate-edit-targets]            # extend/override built-in presets
instruction = "Given a software change request, retrieve code regions likely to require editing"
```

Per-folder `.codeindex.toml` example:

```toml
index = false            # or:
[index]
exclude = ["fixtures/**"]
[search]
tests = "include"        # this subtree is a test suite; tests are the point
```

## 4. Daemon responsibilities

1. **Watch → incremental index.** A filesystem watcher per root feeds a
   debounced (default 500 ms) reindex through the existing atomic journal;
   the refresh convergence barrier already handles hot trees. Watching is
   an optimization only — every query path revalidates against the
   published generation, so a missed event degrades to staleness, never
   corruption.
2. **Embedding queue.** One background embedder per distinct model
   contract, shared across projects; spaces re-project after each publish
   (content-addressing makes this cheap: touched hashes only).
3. **LSP pool.** One server per (project, language) with idle shutdown
   (default 10 min). Enrichment re-runs post-publish for changed files
   (hover + callHierarchy deltas keyed by generation). The pool also
   serves *live* relation queries when an agent asks for
   callers/callees of a specific entity.
4. **Query service.** Warm `SearchIndex` per project, swapped atomically
   when the generation advances; warm embedder/reranker. Target: <100 ms
   dense+lexical query after warmup (vs 15-20 s today).
5. **Agent surface.** The same JSON-RPC methods exposed over the socket
   are exposed as an MCP server (`codeindex daemon --mcp` or a stdio
   bridge `codeindex mcp`), so agents get `search`, `context`, `similar`,
   `relations`, and `status` as tools with the JSON envelopes the CLI
   already emits.

## 5. Final CLI surface

Thin client over the socket; every command works from anywhere inside a
registered root.

```text
codeindex add [path]              # register root, create .codeindex.toml, start indexing
codeindex remove [path|label]     # unregister (keeps .codeindex.toml; --purge deletes the db)
codeindex list                    # registered roots + generation/freshness
codeindex status [--watch]        # per-project: index, spaces, LSP, queue depth

codeindex search "<text>" [--task T|--instruction I] [--space S]
                 [--where ...] [--retrieval hybrid|dense|lexical]
                 [--rerank] [--limit N] [--json]
codeindex similar <selector> [...]
codeindex context "<question>" [--budget-tokens N] [--include tests,callers,config]
codeindex relations <selector> [--kind calls] [--direction in|out] [--live]

codeindex models resolve|list|doctor
codeindex tasks list
codeindex daemon start|stop|status|logs [--mcp]
```

- `query` is an alias of `search`; `context` is the budgeted evidence-pack
  command (FEEDBACK §8) built on search + relations + packing.
- Every command keeps `--json` versioned envelopes; the daemon protocol
  uses the same payloads, so CLI and MCP results are byte-identical.
- The current one-shot paths remain available via `--no-daemon` (CI,
  containers, scripting against a copied `.codeindex/index.db`).

## 6. Concurrency and consistency

- One writer per project DB: the daemon serializes index/embed/enrich per
  project internally (the journal's owner-token machinery already guards
  against a rogue second writer, e.g. a `--no-daemon` CLI run).
- Queries never block on writers: snapshot export pins one read
  transaction (existing invariant); the warm index swaps on generation
  advance.
- `add` is idempotent; `remove` never deletes user data without `--purge`.
- The daemon itself is owned by a lifecycle runtime (daemonkit): instance
  identity, cross-process startup locks, authenticated sockets, bootstrap
  transaction, graceful shutdown, and stale-state repair come from the
  runtime rather than hand-rolled pid/socket files.

## 7. Migration plan

1. **M-daemon-1**: `codeindex-daemon` crate — socket server, registry,
   project resolution, `add/list/status`, query serving with warm models.
   CLI grows the socket client + autostart; existing commands keep their
   flags and JSON shapes.
2. **M-daemon-2**: filesystem watch + debounced incremental reindex +
   embedding queue.
3. **M-daemon-3**: LSP pool with generation-keyed enrichment deltas and
   live `relations --live`.
4. **M-daemon-4**: `.codeindex.toml` folder overrides wired into the
   config fingerprint; `context` command; MCP bridge.

Open questions (deliberately deferred): multi-user/system daemons (out of
scope — user-level only), remote daemons (the socket protocol should not
preclude TCP+auth later), and Windows named-pipe support (needed before a
Windows release).
