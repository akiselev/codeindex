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

## Results (2026-07-15, post oversized-units fixes for jq/fd)

Scores against the pre-registered targets. click/ripgrep/zod ran on
pre-fragment indexes (their giant units are truncated-visible only);
jq and fd were rebuilt with fragments.

| battery | score | notes |
|---|---|---|
| click C1–C5 | **5 hit** | `_parse_decls` #1; Context obj-inheritance top-4; `Choice.shell_complete` #1; `resolve_envvar_value` top-5 |
| jq J1–J5 | **5 hit** (2 via fragments) | see gates below |
| ripgrep R1–R5 | **4 hit, 1 partial** | mmap decision #1–2; `detect_binary` #2; literal-prefilter subsystem sweeps R4; R5 found alternate (defensible) hook points |
| zod Z1–Z5 | **5 hit** | discriminator `propValues` #5; "plug in translations" put locale files at dense#1–3 |
| X1 3-language | 2 hit, 1 partial | ripgrep `flags/parse.rs:parse` dense#1; jq's argv loop under-surfaced |
| X2 German→zod | hit (diluted) | dense#1 = v3 `strip` across the language barrier; German locale strings dominated lexical |
| X3 discovery | pass | bare `query` from `crates/ignore` anchored ripgrep; `--project` switched corpora |

### Oversized-units gates (the fix validating itself)

- **P1 (lexical, minutes after reindex)**: "FORK backtracking opcode
  dispatch" → `jq_next#1/#4/#5/#7` hold ranks 1–4; two arrived via
  `graph#3/#4` — part-of family expansion from lexically-matched
  siblings. Yesterday `jq_next` did not exist in any index.
- **J1 dense**: `jq_next` itself fused **#1**
  (`dense#7,lexical#50,graph#1` — its own fragments corroborated the
  parent to the top). Pre-registered target, previously a hard miss.
- **J5 dense**: `gen_try` #1, `jq_next#6` #2 (the error/backtrack
  region, lexical#1), `set_error` #3 — the TRY/FORK machinery is now
  addressable at fragment granularity.
- **S03 (fd ColorWhen)**: still missed at threshold 6000 —
  `construct_config` is ~5.8k chars and slipped under. fd rebuilt at
  `fragment_threshold_chars = 3000` (the `[index]` knob, per-project):
  retest pending re-embed.

### Recurring pattern: lexical dilution of a correct dense #1

Seven separate cases this battery (click C3/C5, jq J4, ripgrep R5, zod
Z3, X1-ripgrep, X2): dense ranks the exact target #1 and OR-token
lexical noise dilutes it to fused #4–6. Adaptive down-weighting of the
lexical list when its top ranks disagree with dense agreement signals —
or IDF-aware token filtering — is now the highest-leverage retrieval
item left.

### Cost finding: nested oversized units double-fragment

`spawn_senders` and its inner `walker.run(...)` closure both exceeded
the threshold and fragmented over the same lines (6 fragments where 3
carry the information). Content addressing dedupes identical text, but
the one-line span offset produces distinct hashes. Fix candidate:
suppress fragmenting a unit whose body is ≥90% covered by an oversized
child unit that will itself fragment.

## Follow-up (2026-07-23): three fixes from this battery's own findings

Implemented the extractor-level fix from the cost finding above, the
retrieval-level fix from the lexical-dilution pattern, ran the pending
S03 retest, and spent the resulting fresh binary on the chunk-everything
question Track 3 flagged as needing data before more code gets built
(`docs/oversized-units-plan.md`).

### Nested-oversized-unit suppression (the cost finding, fixed)

`spawn_senders` turned out to be **three** closures deep — `spawn_senders`
⊃ `walker.run(|| {...})` ⊃ `Box::new(move |entry| {...})` — with the
inner two each covering ~99-100% of their parent's body. The extractor
now sizes every candidate unit before deciding who fragments; a unit
≥90% covered by an oversized descendant skips fragmenting since the
descendant covers the same lines on its own. Verified against fd's real
source (inlined as a regression test,
`triple_nested_oversized_closures_only_fragment_the_innermost`): only
`Box::new(...)` fragments now; `spawn_senders` and `walker.run(...)`
correctly produce zero. Confirmed live on the reindexed fd corpus — the
search result for S03 (below) shows `spawn_senders`/`walker.run(...)`
with no `#N` fragments at all, down from the double-fragmenting
observed in the original battery.

**Caveat surfaced while validating this**: reindexing an unchanged file
does *not* re-run extraction — revision tracking is purely content-hash
based (`action_for` in `crates/indexer/src/run.rs`), so an extractor
*algorithm* change on unchanged source is silently ignored by a normal
`reindex`; only a fresh database (or a changed file) picks it up. This
is a real gap for anyone upgrading codeindex without also touching
source — worth a tracked follow-up (e.g. fold an extractor version into
the document fingerprint) but out of scope for this battery.

### Adaptive lexical down-weighting (the recurring-pattern fix)

`adaptive_lexical_weight` (`codeindex-query`): when dense's top pick
isn't corroborated anywhere in lexical's own top-5, lexical's fusion
weight drops from 1.0 to 0.4 for that query, so OR-token noise can't
out-vote a confident dense hit through RRF's rank-only scoring. Unit
tests reproduce the exact documented failure mode (a confident dense #1
diluted to a fused loser at equal weight, restored to fused #1 once
down-weighted) and cover the corroboration/no-signal edge cases. Applies
to every hybrid query going forward; not yet re-run across the full
five-corpus battery (that's the natural next eval pass).

### S03 retest (fd, `construct_config`'s `ColorWhen::Auto` decision)

Reindexed fd fresh (forced past the revision-trust caveat above) at the
already-configured `fragment_threshold_chars = 3000` /
`fragment_target_chars = 1500` (380 units, up from the old
non-fragmenting index; down from 390 pre-fix thanks to the
double-fragment suppression). Query: *"why is colored output disabled
when the output is piped to another command"*, `explain-behavior`,
`--project fd-daemon`.

**Still a miss by the hit/partial/miss rubric, now a partial by subsystem
adjacency**: `construct_config` no longer slips under the cap — it
correctly emits two fragments (`construct_config#1` covers the
`ColorWhen::Auto` decision lines directly) — but the fused rank is
`construct_config#2` at #17, `#1` at #21, parent at #28. None reach
top-10. `determine_ls_command` (same file, same color-decision
subsystem) lands at fused **#3**, which qualifies as a partial under
this eval's own "same file/subsystem in top 5" rule, an improvement
over the prior hard miss. The fragment mechanism is doing its job
(the target text is indexed and addressable); the remaining gap is
fusion ranking, not coverage — consistent with the lexical-dilution
finding above still being only partially mitigated for this specific
query.

### Chunk-everything experiment (Track 3, first item)

Track 3 named this the cheapest deferred experiment and asked for data
before committing more code. Ran fd at `fragment_threshold_chars = 400`
/ `fragment_target_chars = 300` (fragmenting almost every function,
not just oversized ones):

| | threshold 3000/1500 (shipped) | threshold 400/300 (experiment) |
|---|---|---|
| units | 380 | **889** (2.3x) |
| `construct_config`'s `ColorWhen` fragment | dense#25 (fused #21) | dense#7 (fused #35) |
| S03 top-5 | miss | miss |

The target region's own dense rank improved noticeably (dense#25 →
dense#7) — smaller, more topically coherent fragments do help dense
retrieval find the exact region. But the fused rank got *worse* (#21 →
#35): fragmenting everything multiplies the number of competing
candidates corpus-wide, and RRF's rank-only fusion doesn't reward the
individual dense-rank improvement enough to overcome having ~2.3x more
things to rank against. **Net finding: chunk-everything is not a clean
win at this scale** — it trades a real 2.3x cost increase for a
worse fused outcome on the one case it was tested against. Reverted fd
to the shipped 3000/1500 config. This makes the RRF-fusion-of-many-small-
candidates problem (not raw coverage) the load-bearing question before
lowering the default threshold further; it's the same problem the
lexical-dilution finding points at, just for dense-vs-dense competition
instead of dense-vs-lexical.

### Deferred: summary representations, late-interaction pooling

The other two Track 3 items are genuine design decisions, not
mechanical fixes, and weren't attempted here:

- **Summary representations** (model-generated abstracts as an extra
  channel for containers) needs a choice of which model foots the
  generation cost and a generation pipeline that doesn't exist yet.
- **Late-interaction pooling** (multi-vector per input at the model
  layer) is candle-side research or on the model/pooling architecture,
  separate from the fragment mechanism entirely.

Both stay queued behind a design decision rather than folded into this
battery.

## Follow-up (2026-07-23, part 2): extractor-version gap + adversarial review

### Extractor-version drift now fails loud instead of silently reusing stale entities

Fixed the gap the S03 retest above surfaced: `action_for`
(`crates/indexer/src/run.rs`) decides per-document reindex work purely
from content hash/mtime/size, so a document whose file hasn't changed
is never re-extracted — an extractor *algorithm* change (like the
nested-suppression fix above) was silently invisible to `reindex` on
unchanged files, requiring a manual database wipe to actually observe.
`codeindex_tree_sitter::FRONTEND_VERSION` already existed for exactly
this ("increment whenever bundled queries, normalization, or adapters
change in a way that can alter a staged payload") but was never wired
into anything that enforces it — it fed the resumed-run fingerprint and
nothing else. Added it to `immutable_settings()`, the same gate that
already fixes `fragment_threshold_chars` etc. once a database is
published: a version bump now raises the existing "setting is fixed
once published... delete the database file" error instead of quietly
serving stale extraction results. Bumped `FRONTEND_VERSION` to `:3` for
the nested-suppression change. New regression test
(`frontend_version_drift_refuses_reindex_instead_of_silent_reuse`)
seeds a stale version and confirms reindex refuses rather than reusing.
**Known limitation**: this only enforces going forward — a database
published before this fix has no `index.frontend_version` row to
compare against, so upgrading past this point adopts the current
version silently on first reindex rather than forcing a rebuild. A
one-time bootstrapping gap, not something worth a migration for a
pre-1.0 project.

### Adversarial review of this session's own fixes found a real regression risk

Reviewing the adaptive-lexical-weight fix (item #2 above) adversarially
rather than trusting the passing tests: the heuristic assumes
disagreement between dense's top pick and lexical means lexical is
noise. That assumption inverts for short/exact-identifier queries,
where lexical's exact-match precision is naturally *high* and
disagreement more often means dense picked a semantically similar
decoy. A sweep over realistic dense/lexical rank combinations found 930
distinct rank shapes where down-weighting lexical flips a case
reciprocal-rank fusion would otherwise get right —
concretely, a correct answer at dense-rank 35+/lexical-rank ≤9 loses to
a lexical-absent decoy at dense#1 once lexical is down-weighted, when
equal weighting would have ranked the correct answer first by a healthy
margin. All 7 cases that motivated the original fix were long,
multi-word behavioral questions (16+ words); no documented case was a
short query. Fixed by gating the down-weighting on query word count (< 6
words: never down-weight) — preserves the fix for the cases it was
built for while closing the demonstrated regression for identifier-style
searches. New tests: `short_queries_never_get_lexical_down_weighted` and
`short_identifier_query_is_not_worsened_by_a_dense_decoy` (the latter
reproduces the exact flip-triggering rank shape and confirms the gate
prevents it).

Reviewing the nested-oversized-suppression fix (item #3 above)
adversarially: it assumes a covering oversized child will actually
succeed at fragmenting (`build_fragments` needs ≥2 AST-boundary leaves).
An unsplittable oversized body — one giant string/byte literal, not
unheard of in generated code — produces exactly 1 leaf and fragments
into nothing. Before this fix, suppressing the parent based on size
alone meant a body like this would suppress *every* enclosing unit's
fragmenting attempt too, so the whole nested region would lose
fragment-level addressability entirely — worse than the pre-suppression
double-fragmenting bug it was meant to fix. Added `would_fragment`, a
cheap predicate mirroring `build_fragments`'s own leaf-count check, and
require it before treating a covering child as disqualifying its
parent. New test: `unsplittable_covering_child_does_not_suppress_the_parent`.
