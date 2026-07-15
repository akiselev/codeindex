# Daemon-era eval: five real corpora, four languages

Setup: every project registered through `codeindex add` (daemon M1), one
shared warm Qwen3-Embedding-0.6B worker, hybrid retrieval with compression
and graph expansion at defaults. Corpora: fd (Rust, 365 units), ripgrep
(Rust, 1728), click (Python, 462), jq (C, 458), zod (TypeScript, 1453).
jq's yacc/flex output and zod's test files are excluded through
`.codeindex.toml` — the first real use of per-project config for corpus
hygiene.

Scoring: **hit** = expected code in top 5; **partial** = adjacent code in
top 5 (same file/subsystem) or expected code in 6-10; **miss** otherwise.
Expected targets written down *before* running anything.

## ripgrep (Rust, biggest corpus)

- **R1 behavior**: "Why does ripgrep skip files listed in .gitignore even
  when I'm not inside a git repository?" → ignore crate dir/gitignore
  matcher wiring (`crates/ignore/src/dir.rs`, `gitignore.rs`).
- **R2 strategy**: "Where does rg decide to use memory maps versus
  incremental reading when searching a file?" → searcher strategy
  (`crates/searcher/src/searcher/{glue,mmap}.rs`).
- **R3 diagnose**: "I get 'binary file matches' on a text-ish file — where
  is binary detection implemented and what makes a file count as binary?"
  → NUL-byte detection in grep-searcher (`binary_detection`).
- **R4 deep**: "How does ripgrep turn a Unicode-aware regex into a fast
  literal prefilter?" → `crates/regex/src/literal.rs` (inner literal
  extraction).
- **R5 narrative/compression**: 50+ word paragraph about wanting to search
  only recently-modified files and asking where a metadata predicate would
  hook into the directory walk → `filter_entry` / walk builder plumbing in
  the ignore crate.
- **R6 graph (after lsp-enrich)**: "What code runs between a match being
  found and the colored line appearing on my terminal?" → printer
  standard/color path; graph expansion should pull the call chain.

## click (Python)

- **C1 mechanics**: "How does click figure out the parameter name and
  whether an option is a flag from decorator strings like '--verbose/-v'?"
  → `Option._parse_decls`.
- **C2 architecture**: "When commands are nested in groups, how does a
  subcommand get access to objects the parent command created?" →
  `Context.ensure_object` / `make_pass_decorator` / `ctx.obj`.
- **C3 behavior**: "Why does click sometimes page long output instead of
  printing it, and where is that decided?" → `echo_via_pager` /
  `_tempfilepager` heuristics.
- **C4 subsystem**: "Where does shell tab-completion get its suggestions
  for an option declared with Choice?" → `shell_completion.py`,
  `Choice.shell_complete`.
- **C5 diagnose**: "My option with prompt=True keeps prompting even though
  its environment variable is set — what's the resolution order between
  envvar, default, and prompt?" → `Parameter.consume_value` /
  `resolve_envvar_value`.

## jq (C)

- **J1 core**: "Where is the bytecode interpreter's main dispatch loop?"
  → `execute.c` (`jq_next`, the opcode switch).
- **J2 deep semantics**: "How does jq implement path expressions so that
  an assignment like `.a.b = 1` knows where to write back?" → path
  tracking / PATH opcodes in `execute.c`.
- **J3 orientation**: "Where are builtins like `map` and `select`
  defined?" → `builtin.c` + the embedded jq-source builtins.
- **J4 memory**: "Where is reference counting for JSON values implemented
  and when does a value actually get freed?" → `jv.c` (refcounts,
  `jv_free`).
- **J5 diagnose**: "When a filter hits an error like dividing by zero,
  how does the error propagate and where would a try/catch intercept it?"
  → backtracking / `jv_invalid` machinery, FORK_OPT/TRY opcodes.

## zod (TypeScript)

(zod master is the v4 layout — `packages/zod/src/v4/core/` — targets
below adjusted after a lexical smoke test confirmed it.)

- **Z1 mechanics**: "When I call z.string().email(), where does the email
  regex live and where does the check actually run?" → `core/regexes.ts`
  + the check machinery in `core/checks.ts` / `schemas.ts`.
- **Z2 architecture**: "How does .refine() attach a custom validation and
  when does it run relative to the base type's own checks?" → refinement
  plumbing (`core/checks.ts`, `.check()`/`superRefine` in schemas/api).
- **Z3 async**: "Where does zod decide between the synchronous and
  asynchronous parsing paths?" → `core/parse.ts` (sync/async parse
  entrypoints, Promise detection in the run loop `core/core.ts`).
- **Z4 performance**: "How does discriminatedUnion pick the right branch
  without trying every option?" → `$ZodDiscriminatedUnion` in
  `core/schemas.ts` (discriminator value map / propValues).
- **Z5 errors**: "How does an individual failed check turn into the final
  human-readable error message, and where would I plug in translations?"
  → issue creation in `core/errors.ts` → locale error maps
  (`locales/en.ts`, `config`).

## Cross-corpus probes

- **X1 same question, three languages**: "How are command line flags that
  take values parsed and validated?" against ripgrep, click, jq — the
  instruction contract should surface each project's own flag machinery,
  not generic string code.
- **X2 multilingual**: German query to zod ("Wo wird entschieden, ob ein
  unbekannter Schlüssel in einem Objekt einen Fehler auslöst oder
  entfernt wird?" → strict/strip/passthrough in ZodObject).
- **X3 agent flow**: from inside `ripgrep/crates/ignore`, run a bare
  `codeindex query` with no --project — discovery must anchor the right
  repo; then the same query with `--project zod` must switch corpora.

Results appended below as runs complete.

## Latency results (fd, 365 units, Qwen3-0.6B on CPU)

| Path | Latency |
|---|---|
| lexical via daemon | **0.09 s** |
| dense via daemon, idle | 2.1–2.4 s |
| dense via daemon, while 4 projects bulk-embed | **1.9–2.1 s** |
| dense `--no-daemon` (model load every call) | 13.3 s |

The dense floor is the query's own forward pass on CPU; the daemon adds
~0.1 s. GPU execution is the next lever for the floor itself.

### The scheduling saga (why per-role model instances)

Three designs were measured against "search while projects embed":

1. **Shared FIFO worker**: a query embed queued behind token-packed
   document batches — observed >10 min. Unusable.
2. **Priority lane + preemption between documents**: helps, but nothing
   interrupts a forward pass, and one 10k-char function costs 30–80 s on
   CPU — observed 26–79 s per query. Still unusable.
3. **Separate backend instance per role** (query lane eager, bulk lane
   lazy, ~weights-sized extra RAM when both active): queries at idle
   latency under full bulk load — observed 1.9–2.1 s. Shipped.

Also observed live: the daemonkit 0.1.0 drain wedge from the audit — an
abandoned drain left the daemon generation permanently QUIESCING and
`ensure()` refused to replace it. Fixed upstream on daemonkit's
`first-consumer-hardening` branch; codeindex additionally recovers
client-side (force-stop + re-ensure on BusyQuiescing) so it works on
0.1.0 as published.
