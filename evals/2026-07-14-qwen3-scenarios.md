# Qwen3-Embedding scenario evaluation — 2026-07-14

Live evaluation of the rearchitected pipeline (typed contracts, candle
backend, instruction-tasked queries) against real open-source corpora.
Scenario runner: `evals/qwen3-scenarios.sh`; raw hits:
`2026-07-14-qwen3-scenarios-results.json`. Verdicts were produced by
independent judges reading the actual repository sources, not hit names.

## Setup

- Model: `hf:Qwen/Qwen3-Embedding-0.6B`, candle backend, CPU, resolved
  entirely from the repo's own sentence-transformers configuration.
- Corpora: `fd` (Rust, 228 units) + `flask` (Python, 337 units) indexed as
  two projects in one database; codeindex itself (605 units) with
  rust-analyzer enrichment (458 `typed_signature` representations, 2,919
  exact `calls` relations).
- Spaces: `code` (implementation, 1024d), `docs` (documentation),
  `code256` (implementation, 256-dim Matryoshka), `types`
  (typed_signature, codeindex corpus).

## Scorecard: 13 pass / 7 partial / 1 fail (21 scenarios)

Standouts:

- **Instruction mixing on one index** (S01a-c): one document embedding,
  three retrieval intents; all landed on fd's real walker
  (`build_walker` / `spawn_senders` closures in walk.rs) with sensible
  intent-driven reordering (config-builder first under
  locate-edit-targets).
- **Issue-to-edit** (S02): a paraphrased real fd issue retrieved
  `ensure_use_hidden_option_for_leading_dot_pattern` — the exact function —
  at 0.749 with a +0.16 margin, plus the detection helper it delegates to.
- **Failure diagnosis** (S04): "panicked while sending on a closed channel
  during the parallel scan" → `spawn_senders` (the literal send sites with
  the is_err/Quit handling), `scan` (channel creation + join), `poll` (the
  receiver-side early-stop that closes the channel — the root cause path).
- **Channel semantics** (S06): docs space surfaced the documented cookie
  attribute helpers; code space surfaced `open_session`/`save_session` —
  the actual itsdangerous signing/verification. The signing methods have no
  docstrings, so the docs channel structurally cannot see them: channels
  behave exactly as designed.
- **typed_signature beats implementation** (S14): on a type-shaped query
  the LSP-hover-derived space scored higher with a wider truth-to-noise
  margin and ranked the substantive loader body first. First evidence the
  enriched parallel space earns its keep.
- **Exact identifier** (S07): bare "make_response" found both canonical
  definitions at ranks 1-2 (0.777) — dense-only retrieval handled an
  identifier probe expected to need lexical search.
- **Matryoshka** (S08): 256-dim space kept 4/5 top-5 overlap and identical
  top-3 order vs 1024d — 4× smaller vectors, no meaningful quality loss on
  this query.
- **Incrementality** (S16): editing one file re-embedded exactly 1 vector
  out of a 565-hash space (content-addressed reuse working end to end).

## Weaknesses found (all actionable, mapping to planned roadmap items)

1. **Trivial-test gravity.** When a query degrades (Spanish S09, emoji
   S10, absurd instruction S12, dogfood S15), ranking collapses onto
   short, low-information unit-test chunks rather than random files. The
   S12 "pirate instruction" failure is the clean signature: all five hits
   were 3-7-line asserts. Mitigations: default search profiles that
   exclude `kind=test`/`scope=tests` (the `--where` grammar already can),
   a separate tests space, and `min_nodes` floors at query time.
2. **Lexical gap persists for behavior constants** (S03): the
   `ColorWhen::Auto && is_terminal()` decision was missed while a
   same-file distractor topped the list. The planned SQLite FTS + RRF
   fusion is the fix.
3. **Flat score bands under some instructions** (S01b spread 0.006): the
   ordering was right but barely discriminated — the planned
   Qwen3-Reranker second stage addresses exactly this.
4. **Long narrative queries drift generic** (S13): a paragraph describing
   the app-factory pattern retrieved five `__init__` constructors and
   missed `find_best_app`. Candidate fixes: reranking, or intent presets
   that compress narrative queries.
5. **Cross-lingual works but degrades** (S09): the right function still
   appeared (rank 2) from a Spanish query, at visibly lower confidence
   than its English twin (S02).

## decombine

Migrated to the new API on branch `codeindex-rearchitecture` (82/82 tests
pass): AnalysisContext now focuses one space of the multi-space index;
drift gates read execution provenance from `model_executions`. Migrating
its indexing tests exposed a real codeindex bug — file deletions were
silently ignored by the journal indexer (no Delete observations were ever
synthesized) — fixed with a regression test in
`Fix silent loss of file deletions in the journal indexer`.
