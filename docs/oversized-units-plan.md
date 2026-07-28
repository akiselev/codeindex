# Oversized units: from truncation to fragments

Status: P1+P2 implemented (2026-07-15); P3 presentation shipped with the `parent` hit field. Nested-oversized-unit suppression (the "cost finding" below) shipped 2026-07-23. The chunk-everything experiment (Track 3, first item) ran on fd 2026-07-23: 2.3x unit-count cost for a *worse* fused rank on the one target tested — not a clean win at this scale; RRF fusion-of-many-small-candidates is the load-bearing question before lowering the default threshold further, not raw coverage. Summary representations and late-interaction pooling remain future work pending a design decision. Motivated by the big-corpus eval
(`evals/2026-07-15-daemon-big-corpus.md`): jq's `jq_next` — the literal
answer to "where is the bytecode interpreter's dispatch loop" — was
absent from every retrieval mode because bodies over `max_body_chars`
were silently dropped. The hotfix (truncate instead of drop) makes the
head of a giant function findable; everything past the cap is still
invisible, and the eval history says the *interesting* parts of giant
functions live deep inside them (jq's FORK/TRY opcode arms, fd's
`ColorWhen` decision inside `construct_config`).

## The actual architectural error

The index equates three things that only coincide for small code:

    one syntactic entity  =  one retrieval unit  =  one embedding

For a 700-line dispatch loop that identity fails twice:

1. **Coverage**: anything past the representation cap contributes zero
   signal — dense or lexical. (Was: dropped. Now: truncated. Both lose
   the tail.)
2. **Resolution**: even under the cap, one vector averages the whole
   body; a 15-line decision inside a 9k-char function is diluted below
   retrievability. The reranker cannot rescue what retrieval never
   surfaces, and graph expansion only reaches it via neighbors.

Two facts make the fix cheap:

- `FullSource` representations already retain the **complete** text
  (untruncated) under the default retention mode.
- `lexical_index` currently indexes the *Implementation* representation
  — i.e. the truncated text — purely by choice of source column.

## Track 1 — full-text lexical (immediate, no schema change)

Index `FullSource` content (fallback: Implementation) into
`lexical_index`. FTS5 handles large documents without complaint;
comments and docstrings in FullSource *help* BM25. This alone makes
`jq_next` findable today by "FORK", "backtrack", or "TRY" tokens, with
zero new machinery. One-line change in `index_publish.rs::insert_units`
plus the mirror in `upsert_file`; a regression test asserting a token
past the 10k boundary matches.

Cost: slightly larger FTS index. Nothing else changes.

## Track 2 — AST-boundary fragments (the real fix)

When a stripped body exceeds a threshold, the extractor emits the parent
entity **plus fragment entities** split at the largest child AST
boundaries (top-level statements, match/switch arms, if/else chains),
greedily packed toward a target size (~3k chars). Fragments are ordinary
entities:

- `kind = "fragment"`, `scope = <parent name>`, real byte/line spans
  inside the file, names like `jq_next#2` (ordinal) — agents get a
  pointer into the *interesting region* of the monster, which is better
  than pointing at line 340 of 1100.
- They flow through the existing pipeline untouched: representations,
  content-addressed embedding (fragments dedupe across projects like
  everything else), FTS rows, tests-policy, `--where` filters.
- The parent keeps its (truncated) representation so whole-function
  queries still land on the function itself.

**Why AST boundaries, not text windows**: coherent fragments embed
better (a switch arm is a topic; an arbitrary 3k window is two half
topics), spans are meaningful to agents, and tree-sitter is already
holding the tree when the decision is made. Overlap comes free from
packing at statement granularity instead of characters.

**Relations**: `part-of` edges (fragment → parent, provenance
`extractor`, resolution `exact`) are synthesized **in memory** when the
`SearchIndex` loads, not stored: the relations table is generation-
replaced wholesale by analyzer enrichment, which would silently clobber
extractor-written rows. Derivation is exact and cheap (same project and
file, name prefix before `#`, span containment). Graph expansion then unifies the family: a hit on the parent
pulls its fragments into candidates and vice versa — the same lever
that lifted `handle_cmd_error`, applied within one function.

**Identity**: entity ids derive from span + normalized hash exactly as
today, so fragments version with their content. Editing one arm of the
switch invalidates that fragment (and the parent), not its siblings —
content addressing keeps re-embeds proportional to the edit.

**Search presentation**: fragments compete as ordinary hits. The JSON
grows `parent: Option<selector>` on hits (fragment → its function) so
agents and the future `context` command can group families; the CLI
prints `jq_next#2 (fragment of jq_next)`. No collapsing in v1 — measure
first whether duplicate families actually crowd the top-k before adding
dedup logic.

**Config** (layered, root or per-folder):

```toml
[index]
fragment_threshold_chars = 6000   # bodies above this also fragment
fragment_target_chars = 3000      # greedy packing target
```

Defaults conservative: only oversized bodies fragment. Lowering the
threshold to fragment *everything* is a quality experiment for later —
it multiplies vector count and embed time, so it needs eval data first
(the levers doc pattern: measure, then turn the knob).

**What does not change**: schema (kinds are strings; relations and
representations tables already fit), the journal, content addressing,
space identity, the daemon, decombine's consumption surface. A schema
epoch bump is *not* required; a reindex is (pre-release: rebuild).

## Track 3 — deferred, in dependency order

- **Chunk-everything experiment**: fragment at ~2-3k regardless of body
  size; compare recall on the S03/J5-class deep-target questions vs
  cost. Only after Track 2 ships and the eval battery is re-runnable.
- **Summary representations**: model-generated abstracts as an extra
  channel for containers (files, classes, giant functions) — the
  parallel-embeddings architecture already supports the channel; the
  generation pipeline is the new part.
- **Late-interaction pooling** (multi-vector per input at the model
  layer): candle-side research; not a substitute for fragments because
  it does not fix lexical coverage or span-level presentation.

## Phasing and eval gates

- **P1** Track 1 + regression test. Gate: J1 lexical finds `jq_next`
  by opcode tokens without reindexing embeddings.
- **P2** Track 2 extractor + `part-of` edges + tests (fragmentation
  determinism, identity stability under sibling edits, threshold
  respected, packing bounds). Rebuild jq + fd. Gate: J1 dense top-5;
  J5 surfaces the TRY/FORK region; fd S03 finally surfaces the
  `ColorWhen` decision inside `construct_config` — the oldest open miss
  in the eval log.
- **P3** presentation (`parent` field, fragment naming), re-run the full
  five-corpus battery, record before/after in the eval doc.
