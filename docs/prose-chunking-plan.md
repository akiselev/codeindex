# Prose support and chunking

Status: design + P1 implementation (`codeindex-prose`), 2026-07-27.

Motivated by a corpus that is not code: a civic-research repository of ~705,000
words — 81 hand-written Markdown notes (~474k words), 176 OCR'd city-council
minutes, 21 court opinions (one 148 KB), 32 OCR'd staff reports. The consuming
task is **verification**: finding passages that contradict each other or have
gone stale, before a claim is repeated to a real person.

That task changes what chunking is for. Generic RAG chunking optimizes "does the
answer survive retrieval." Verification optimizes "are these two passages about
the same fact, and do they disagree." The second wants tighter, more atomic
units than the first — but this project already has evidence that shrinking
units naively makes fused ranking *worse* (`oversized-units-plan.md`: the
chunk-everything experiment on fd, 2.3x unit count for a worse fused rank). So
the design has to shrink units without flooding the candidate pool.

## Why the tree-sitter path does not cover this

Extraction is grammar-driven: `LanguageRegistry::load()` iterates a hardcoded
`bundled_languages()` and each language needs a compiled grammar plus
`units.scm`. Adding `tree-sitter-md` would handle the 81 Markdown files and
nothing else — the OCR'd half of this corpus has no grammar, and no reliable
heading structure to hang one on. Representative strings from it:

    Case No.19BBCV00118          (no space after the period)
    Section J¡.(f)               (OCR corruption of "4(f)")
    54956. 9( d)( 2)             (spaces injected inside a statute cite)
    City of Gardena v. Los A ngeles Regional Water Qu ality Control Board

Sentence segmentation fails on this text, and so does anything that assumes
well-formed structure. Prose extraction therefore needs its own path.

## The methods, and what each is actually good for

Nine approaches, assessed against *this* corpus and *this* task rather than in
the abstract.

**1. Fixed-size windows (characters or tokens), optional overlap.** The
baseline. Immune to malformed input, trivially fast, no dependencies. Splits
mid-sentence and mid-argument; a claim severed from its subject becomes
unretrievable and, worse, becomes a false contradiction candidate. Overlap of
10–20% recovers most boundary losses at proportional index cost.
*Verdict: correct fallback, wrong default.*

**2. Sentence windows.** Chunk = N sentences, stride < N. Respects the natural
unit of a claim, which is exactly right for verification. Requires sentence
segmentation, and on this corpus that is the weak link: `v.`, `No.`, `§
54956.9(d)(1)`, `Cal. App. 4th`, and OCR-injected periods all defeat naive
splitters. Legal and civic prose is close to the worst case for it.
*Verdict: valuable, but only behind a segmenter that knows legal abbreviations.*

**3. Structural / heading-aware.** Split on document structure; nest sections.
For `docs/` this is close to free and close to ideal, because the corpus follows
an enforced house convention — every file carries `## Findings`, `## Assessment`,
`## Verification queue`. Those headings are semantic boundaries, not just
typographic ones: a claim in Findings and a claim in Assessment have different
epistemic status, and the index should know that. Useless on the OCR half.
*Verdict: the default wherever structure exists.*

**4. Recursive splitting with a separator hierarchy.** Try separators in
priority order — paragraph break, line break, sentence end, space — descending
only when a piece still exceeds the target. Degrades gracefully to (1) on
garbage input while behaving like (3) on clean input. It is the pragmatic
workhorse of the field for good reason.
*Verdict: the engine underneath everything else.*

**5. Semantic chunking (embedding-based boundary detection).** Embed sentences,
cut where adjacent similarity dips below a percentile threshold. Intuitively
appealing; empirically mixed — it frequently fails to beat (4) while costing a
full embedding pass *before* indexing, and it inherits (2)'s segmentation
problem. *Verdict: not worth it here.*

**6. Late chunking.** Embed the whole document with a long-context model, then
mean-pool token embeddings per chunk, so every chunk vector carries
whole-document context. Directly fixes the antecedent problem — "the City" in
paragraph 40 resolves to South Pasadena. Needs a long-context embedder exposing
token-level output; `codeindex-embedding`'s current backends do not. This
project already has "late-interaction pooling" listed as future work.
*Verdict: the most principled upgrade, blocked on the embedder.*

**7. Contextual retrieval.** Prepend a short generated blurb to each chunk
before embedding — "This passage is from the 1999 Slater preliminary injunction,
section ordering Caltrans to maintain corridor properties." Reported to cut
retrieval failure substantially. Costs one generation per chunk, cacheable.
**This project already has the seam**: `RepresentationKind::GeneratedDescription`
plus the `RepresentationEnricher` hook, applied in `stage.rs` right after
extraction. No new machinery is required — only a populated enricher.
*Verdict: highest value-per-unit-of-work upgrade, and architecturally free.*

**8. Proposition / atomic-fact chunking.** Decompose passages into standalone
assertions. This is the *theoretically correct* unit for contradiction
detection, because a contradiction holds between two atomic claims, not between
two paragraphs. Costs a generation pass per passage and discards prose context.
*Verdict: right idea, defer; revisit once (7) is in and measured.*

**9. Small-to-big / parent-document retrieval.** Embed small units for
precision, return the enclosing parent for context. **Already implemented here**
— the fragment mechanism plus the `parent` hit field shipped 2026-07-23. Prose
chunking should reuse it rather than reinvent it, and doing so is what makes
small units affordable despite the fd experiment's warning.
*Verdict: adopt as-is; it is the mitigation for over-chunking.*

## Design

Structure where it exists, recursion where it does not, parents throughout.

    ProseFormat::Markdown  ->  heading sections  ->  recursive split if oversized
    ProseFormat::Text      ->  whole document    ->  recursive split, with overlap

- **Unit** = one Markdown section (heading + body, at the deepest heading level)
  or one recursive chunk of plain text.
- **Oversized units fragment** rather than truncate, mirroring Track 2 of
  `oversized-units-plan.md`, and fragments carry `scope` pointing at the parent
  section so the existing parent presentation works unchanged.
- **Separator hierarchy** for recursion: `\n\n+` → `\n` → sentence-ish
  (`. `/`? `/`! ` not preceded by a known abbreviation) → ` `. The abbreviation
  guard is deliberately small and legal-flavoured (`v.`, `No.`, `Cal.`, `App.`,
  `Supp.`, `Stat.`, `§`, single capitals) because that is what this corpus is
  full of; it is a heuristic, not a parser.
- **Overlap** applies to plain text only. Markdown sections are semantically
  closed and duplicating across them mostly adds noise.
- **Representations**: `FullSource` (the untruncated unit, so full-text lexical
  search keeps working per Track 1) and `Body` (the chunk text used for
  embedding). `Signature`, `Symbol`, `Usage`, `Implementation` are meaningless
  for prose and are not emitted.
- **`EntityKind::Other("section" | "chunk")`**, and `LanguageId` of `markdown`
  or `text`. No new core vocabulary is required — `Other` and `Custom` already
  exist for exactly this.

Deliberately **not** in P1: semantic chunking (5), late chunking (6, blocked),
generated context (7, follow-up), propositions (8). The extraction path is built
so that (7) needs no changes to this crate at all.

## Where it plugs in

`codeindex-prose` depends only on `codeindex-core` — no grammars, no SQLite —
matching the boundary `codeindex-storage` already keeps. Two dispatch sites in
`codeindex-indexer`:

- `scanner.rs`: after the tree-sitter registry misses on an extension, consult
  the prose registry before skipping the file.
- `stage.rs`: if the document's language is a prose format, call
  `extract_prose` instead of `extract_units`, and skip reference extraction
  (prose has no call sites; `Usage` stays empty, which the schema already
  tolerates).

## Measured on the target corpus, 2026-07-27

First full runs of `codeindex index --language markdown` / `--language text`:

| Project | Documents | Units | Notes |
| --- | ---: | ---: | --- |
| `docs/` (Markdown) | 82 | 1,770 | avg 2,145 B/unit; 0 skipped, 0 warnings |
| council minutes (OCR text) | 176 | 405 | |
| court opinions (text) | 21 | 219 | includes one 148 KB opinion |
| staff reports (OCR text) | 32 | 117 | |

229 text documents indexed in ~7 s. Two defects surfaced and were fixed:

- **Container sections.** 5.2% of Markdown units were under 30 bytes — bare
  `## Findings` headings whose content lives entirely in sub-headings. They
  embed to a vector that matches everything and means nothing. Sections with no
  prose of their own are now skipped; their text survives as their children's
  `scope`. Sub-30-byte units fell from 100 to 1 (5.2% → 0.1%).
- **Runt leading chunk.** The runt-merge pass could only merge *backwards*, so a
  tiny first chunk survived at position zero. Now folded forward.

Two things this immediately caught in the consuming corpus, which is the point:

- Lexical search over the Markdown index returned exactly the three files
  discussing Cloudflare Turnstile, and `54956` (the Brown Act closed-session
  cite) matched 28 units across the OCR corpus.
- **88 of 166 council minutes are zero-byte OCR failures** with intact source
  PDFs. The consuming repo had been describing that corpus as "176 files
  covering 2017–2024"; it is really 42 readable files clustered in 2020–21.
  Nothing in the OCR pipeline had reported an error, and no grep would reveal
  it. That is the verification failure mode this work exists to catch, found on
  the first run.

## Why 1,200, and how to check it

The target is bounded above by something concrete rather than by taste: **the
embedder truncates**. `embed.rs` clamps each input's token count with
`.min(max_sequence_length)`, and `TokenStats::record_batch` counts an input as
`truncated` whenever it reaches the cap. Anything past the window contributes
no signal — the same failure `oversized-units-plan.md` was written to fix,
reappearing for prose.

There is deliberately **no default embedding model** (`config.rs`: "there is no
default model. Callers must…"), so the safe target depends on what is chosen:

| Model family | Window | Rough prose budget |
| --- | ---: | ---: |
| BGE-small/base, MiniLM, GTE — BERT-family | 512 tok | ~2,000 chars |
| CodeRankEmbed (the one managed model) | 2,048 tok | ~8,000 chars |
| Arctic-Embed-M-Long, Nomic v1.5 | 8,192 tok | very large |

1,200 bytes is chosen to sit under the **most constrained common case** with
headroom, because (a) OCR'd legal text tokenizes far worse than clean English —
case numbers, statute cites and corrupted glyphs fragment into many tokens, so
the chars-per-token ratio is well below 4, and (b) contextual retrieval
(method 7) would prepend a generated prefix to every chunk, which has to fit
too. On a long-context model 1,200 is conservative and 2,000–3,000 would likely
be better.

Measured cost of the choice, `docs/` at four targets (sections are structural
and constant at 1,516; only fragments vary):

| `--prose-target-chars` | units | chunks | avg | p90 |
| ---: | ---: | ---: | ---: | ---: |
| 800 | 5,677 | 4,161 | 1,027 B | 2,145 B |
| **1,200 (default)** | **3,782** | **2,266** | **1,384 B** | **2,750 B** |
| 2,000 | 2,247 | 731 | 1,917 B | 3,574 B |
| 3,000 | 1,771 | 255 | 2,145 B | 3,933 B |

**1,200 costs 2.1x the units of 3,000** — squarely in the range where the fd
chunk-everything experiment produced a *worse* fused rank. So this is a
defensible upper-bound argument, **not** a demonstrated ranking win, and the
two questions should not be conflated. The truncation argument fixes a ceiling;
where to sit under that ceiling is unresolved.

The check that would settle it, in order: run `token_report` after embedding at
each target and read `truncated` — any nonzero count at a given target rules it
out for that model. Then, and only then, compare retrieval quality on the eval
set below.

## Open questions

- ~~Prose inherits the code tuning knob.~~ **Fixed.** `IndexSettings` now
  carries `prose_target_chars` (default 1,200) and `prose_overlap_chars`
  (default 150), separate from `fragment_target_chars`, exposed as
  `--prose-target-chars` / `--prose-overlap-chars` and as `prose_target_chars` /
  `prose_overlap_chars` in daemon config. Both are **immutable settings**: they
  determine chunk boundaries, chunk boundaries determine content hashes, and the
  embedding-time recovery path re-chunks from the persisted values. Changing
  them on a published database is rejected, exactly as `fragment_target_chars`
  is. `section_fragment_threshold` is derived as 2x the target rather than
  exposed, since fragmenting a section below that yields a single chunk
  identical to the section.
- No prose eval set exists. The obvious seed is the four documented
  contradictions in the consuming repo's `contradiction-sweep.js` — real
  failures with known ground truth (AD 41 vs AD 49; "9,000+ parcels" vs 3,179;
  a $1.2M repair cap superseded by SB 169's $15M; a Maddy Act timing claim).
- Whether OCR normalization belongs at ingest instead of at chunk time. Current
  answer: at ingest, because normalizing inside the indexer would make the
  stored text diverge from the source document.
