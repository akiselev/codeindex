# Roadmap — codeindex toward 1.0

`codeindex` is the reusable library engine behind `decombine` and future tools.
The first architectural milestones establish a provider-neutral, multi-view,
multi-model code index. Product surfaces such as `codeindex-cli`, IDE plugins,
daemons, and bindings remain separate consumers that prove the public library
boundaries.

## Completed foundation

### M1 — Standalone search API

Complete. `codeindex-search` loads a storage-neutral `IndexSnapshot`, validates
it, verifies model identities, resolves stable unit selectors, and exposes text,
vector, and unit-similarity search without depending on `decombine` or SQLite.

### M4 — Multi-representation, entity versions, and embedding spaces

Complete at the 0.1 foundation level:

- logical `EntityId` and exact `EntityVersionId`;
- `FullSource`, `Implementation`, `Body`, `BodyWithoutDeclaredName`, `Signature`,
  `Symbol`, `Documentation`, and derived `Usage` representations;
- explicit representation provenance and an enrichment hook for generated or
  imported channels;
- provider-neutral source documents with stable identities and revisions;
- named embedding spaces binding a channel to an exact model identity;
- different models and dimensions in one corpus;
- explicit-space search and reciprocal-rank fusion across spaces;
- storage-neutral multi-space snapshots.

Current identity tracking is conservative and within one stable source document.
Cross-document move tracking remains a later extension.

### Atomic resumable indexing foundation

Complete. Index processing now checkpoints a durable, refreshable manifest and
versioned document payloads without touching live corpus tables. Compatible
runs resume automatically, mutable sources converge through a no-change barrier,
selected projects publish in one transaction, and snapshots use one read
transaction. The initial CLI exposes this indexing/status lifecycle as a thin
library consumer.

## Remaining milestones

### M9 — Typed embedding contract, generic models, candle/Qwen3 (next up)

Full design in `docs/rearchitecture-plan.md`: replace `embed(&[String])` with a
role/instruction-aware `EmbedRequest`, split semantic model identity from
execution provenance (schema epoch 5), resolve models from HuggingFace repo
paths (sentence-transformers config parsing + TOFU lockfile) instead of
hardcoded enums, and add a candle backend for Qwen3-Embedding (last-token
pooling, left padding, Matryoshka dims). Precedes and unblocks the M2 query
surface.

*Exit:* `hf:Qwen/Qwen3-Embedding-0.6B` resolves without code changes, embeds a
corpus, and answers instruction-tasked queries end-to-end; golden-vector
conformance passes.

### M2 — remaining `codeindex-cli` surface

Extend the existing atomic `index`/status lifecycle consumer with `embed`, `query`, `search`,
`similar`, `capabilities`, and model/source diagnostics. Machine output should
use stable, versioned JSON envelopes. The CLI must contain presentation and
orchestration only; indexing, embedding-space management, and search remain
library calls.

*Exit:* `codeindex search "retry with backoff" --space code --where language=rust`
returns ranked units as JSON.

### M3 — Python bindings

PyO3 + maturin wheels over the storage/parser-free embedding crate and the
snapshot/search API. Notebook users should be able to embed arbitrary text,
inspect spaces, run retrieval experiments, and load serialized snapshots without
compiling bundled SQLite or all language grammars.

*Exit:* published Linux/macOS wheels and a notebook example using two spaces.

### M5 — Public API stabilization and publication

- decide crate publication scope and reserve names;
- compile and test examples;
- semver-audit public types and error contracts (`SourceProvider`,
  `IndexSnapshot`, representation enrichment, embedding-space APIs);
- migrate `decombine` to the current explicit-space APIs and typed entity ids
  as the first compatibility proof;
- complete rustdoc and enable `deny(missing_docs)`;
- add a changelog and deprecation policy;
- publish supported crates to crates.io.

*Exit:* `cargo add codeindex` from crates.io with a documented support policy.

### M6 — Platform and accelerator matrix

CI-tested embedding on Linux/CUDA, macOS/CoreML, and Windows/DirectML, plus
provider drift gates. Add a scheduled/manually-triggered smoke test that
downloads a small model and runs real inference rather than compile-only
checking the backend features. Drift gates move from pinned artifact hashes to
golden-vector conformance once M9's generic model resolution lands.

*Exit:* documented support tier and reproducibility results per provider.

### M7 — Relations and context planning

Foundation shipped with M9: `codeindex-lsp` drives any stdio language server
(server-agnostic; integration-tested against rust-analyzer and clangd) as a
post-publish pass producing a derived `typed_signature` channel and exact
`calls` relations via LSP `callHierarchy` — no per-language query work — with
a `textDocument/definition` fallback seeded from tree-sitter call sites.
Relations are generation-keyed rows surfaced through `IndexSnapshot.relations`
and the `lsp-enrich` CLI command.

Remaining for M7: more relation kinds (implements, type-of, references),
SCIP batch ingestion as an alternative provider, relation-aware query filters,
and token-budgeted context-pack planning over semantic seeds, relations,
tests, examples, and diversity constraints (design:
`docs/rearchitecture-plan.md` §3.6).

*Exit:* a consumer can request an implementation/debug/review context pack
without implementing graph expansion or token selection itself.

### M8 — Write-side storage abstraction and large-corpus serving

The read side already accepts any store through `IndexSnapshot`. Introduce a
write-side persistence interface only when a second backend needs to reuse the
incremental indexer and embedding projection. Add streaming snapshots or indexed
serving when measured corpora no longer fit the in-memory search model.

*Exit:* a non-SQLite backend reuses indexing and embedding without application
code depending on `codeindex-sqlite`.

## Continuous tracks

- **Retrieval evaluation.** Move model, channel, fusion, and provider-drift
  benchmarks into this repository. Evaluate use cases independently rather than
  assuming one universal similarity threshold.
- **Language coverage.** Populate call/reference extraction for languages beyond
  Rust and support runtime-registered grammars when packaging and trust concerns
  are resolved.
- **Source providers.** Add maintained Git revision, editor overlay, archive, and
  structured-import adapters as real consumers demand them. Optimize provider
  catalogs so lean-retention source recovery does not re-enumerate the full
  corpus per lookup.
- **Cross-document entity moves.** Preserve logical identity across provider
  document moves only when matching is unique and evidence is explicit.
- **Serving.** A daemon and/or MCP server can maintain a live index once the CLI
  contract and invalidation model are stable.

## Non-goals

- Duplicate, comparison, concern, security, or migration conclusions. Those are
  application-level interpretations; `codeindex` returns structured semantic
  evidence.
- A mandatory hosted embedding service. The project remains local-first while
  allowing consumers to implement other deterministic embedding backends.
- ANN by default before exact-search benchmarks demonstrate a concrete limit.
