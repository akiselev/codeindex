# Architecture

`codeindex` is split by dependency and change boundary, not by command or user
interface. The workspace is a reusable library substrate. `codeindex-cli` is
the first consumer and deliberately thin; IDE extensions, daemons, bindings,
and analyzers are likewise consumers of these crates rather than part of the
core architecture.

## Crate boundaries

### `codeindex-core`

Application-neutral vocabulary:

- source languages and spans;
- logical `EntityId` and exact `EntityVersionId`;
- `RepresentationKind`, `Representation`, and `RepresentationOrigin`;
- the semantic `ModelContract` and provenance-only `ExecutionInfo`;
- named `EmbeddingSpaceId` and `EmbeddingSpaceIdentity`.

The model and space contracts live in the leaf crate because embedding backends,
stores, search, and consumers all need to name them without depending on one
another.

### `codeindex-tree-sitter`

The bundled language frontend. Language metadata and Tree-sitter queries extract
parser-neutral entities, signatures, documentation, symbols, and raw call sites.
Language adapters handle cases queries cannot express cleanly, such as anonymous
function naming, receiver scopes, and docstring stripping.

### `codeindex-storage`

The serializable read-side seam. `IndexSnapshot` contains selected projects,
units with all representations and provenance, and zero or more named embedding
spaces. `codeindex-search` loads only this type and has no SQLite dependency.
Alternative stores need only construct a valid snapshot to reuse search.

### `codeindex-sqlite`

The default incremental store. It persists:

- projects and provider-defined source document identities/revisions;
- a logical entity ledger and current entity versions;
- normalized representation rows with provenance;
- exact embedding models;
- named embedding spaces;
- vectors keyed by `(space_id, content_hash)`;
- staged call references and analysis provenance.

It also owns the operational index-run journal. `index_runs`,
`index_run_projects`, and `index_run_documents` are durable progress state and
are never joined by search reads. Versioned per-document JSON payloads are the
boundary between resumable processing and live publication. One
`BEGIN IMMEDIATE` transaction validates the journal, applies delete-first live
merges, rebuilds Usage, advances the generation, and marks the run committed.

The schema is pre-release. Incompatible epochs are rejected and require a
reindex rather than being silently accepted or partially migrated.

### `codeindex-indexer`

The orchestration layer. Its primary entry point is provider-neutral
`index_sources`; filesystem `index` is a compatibility convenience built over
`FileSystemSource`.

The indexer:

1. enumerates `SourceDocument` values from each `SourceProvider`;
2. performs cheap revision checks, then verifies changed content by hash;
3. extracts entities and deterministic representations;
4. completes `Body` and `BodyWithoutDeclaredName` channels;
5. runs optional `RepresentationEnricher` implementations;
6. carries logical entity identity across unambiguous edits and renames;
7. checkpoints a versioned document payload without mutating live tables;
8. repeats a full manifest refresh until a barrier observes no changes;
9. atomically publishes the complete selected scope and derives `Usage`
   (call sites attribute to the innermost unit whose byte span contains them,
   so calls inside nested closures attribute to the closure);
10. projects representation content into explicit embedding spaces separately.

`IndexRunBuilder` defaults to compatible auto-resume and automatic convergence.
Its cancellation token produces a durable paused outcome, and explicit policies
select a run or supersede overlapping unfinished work. Provider/read/document
errors prevent publication rather than being counted as partial success.

`SourceProviderCatalog` supplies the same source abstraction during embedding
text recovery under lean retention.

### `codeindex-embedding`

Storage- and parser-free embedding primitives:

- the role-aware `EmbeddingBackend` trait and typed `EmbedRequest`
  (Query vs Document, task instructions, document-side prompts);
- prompt rendering driven by each model's `PromptContract`;
- generic model resolution: `hf:owner/name[@rev]` / `dir:/path` references
  resolved from the repository's own sentence-transformers configuration,
  verified trust-on-first-use through a lockfile;
- fastembed/ONNX execution for mean/cls models, candle execution for
  decoder-style last-token models (Qwen3-Embedding);
- exact tokenizer accounting;
- length-sorted token-area batch packing;
- vector normalization, Matryoshka projection, and token statistics.

### `codeindex-query`

Embedding-free read-side kernels: metadata filters, stable selectors,
deterministic vector ranking, and model identity diagnostics.

### `codeindex-search`

The high-level search service. It validates an `IndexSnapshot`, aligns each
space's vectors with units through representation hashes, and exposes:

- text search in an explicit embedding space;
- vector search;
- unit-to-unit similarity;
- weighted reciprocal-rank fusion across spaces.

Raw cosine values from different models are never directly averaged. Fusion
combines ranks and retains each space's raw score as evidence.

### `codeindex`

A thin facade re-exporting all component crates.

## Source-provider boundary

A provider exposes four concepts:

```text
project locator
  └── SourceDocument
        ├── stable provider-local id
        ├── logical relative path
        ├── language id
        ├── opaque revision metadata
        └── UTF-8 content
```

The provider is not a virtual filesystem. Database rows, Git objects, editor
buffers, object-store keys, archives, generated code, and actual files can all
implement the same contract without inventing filesystem operations they do not
support.

Document identity and display path are separate. A provider can preserve the
same document id across a move, allowing the indexer to retain entity identity
while updating location metadata.

## Representation model

An entity version owns N representations:

```text
EntityId
  └── EntityVersionId
        ├── FullSource
        ├── Implementation
        ├── Body
        ├── BodyWithoutDeclaredName
        ├── Signature
        ├── Symbol
        ├── Documentation
        ├── Usage
        ├── GeneratedDescription
        └── Custom(...)
```

Each representation stores a content hash, optional retained content, and
provenance:

- `Extracted`: deterministic frontend output recoverable from source;
- `Derived`: synthesized from indexed facts, such as Usage;
- `Imported`: supplied by a consumer, model, or external system.

Retention is provenance-aware. Lean modes may drop extracted text that can be
recovered through the source provider. Derived/imported text is retained unless
its producer supplies another recovery mechanism.

## Entity identity

The current identity matcher is deliberately conservative and within one stable
source document:

1. exact `(kind, scope, name)` match;
2. otherwise, a unique matching normalized body and kind;
3. otherwise, mint a new logical entity id.

Duplicate identical bodies are not assigned by incidental ordering. Ambiguous
matches mint new identities. Cross-document move tracking remains future work.

## Embedding spaces

A representation channel and an embedding model are different axes. An
`EmbeddingSpaceIdentity` binds:

```text
space id
+ representation channel
+ semantic model contract (weights, tokenizer, pooling, prompts, dims)
+ document-side contract (document prompt, Matryoshka output dimensions)
```

This permits, for example:

```text
code       = Implementation × CodeRankEmbed
body       = BodyWithoutDeclaredName × CodeRankEmbed
docs       = Documentation × text embedding model
usage      = Usage × text/code model
description = GeneratedDescription × text embedding model
```

Identical representation hashes share one vector inside a space. The same
content can be embedded in multiple spaces without collisions. Space ids are
immutable semantic keys: an existing id cannot silently change channel, model,
or input transform. Every representation channel present in the store except
`FullSource` (display-only) is embeddable; the `embed_pending` convenience
creates one `default/<channel>` space per embeddable channel.

## Data flow

```text
SourceProvider
    │ enumerate/read
    ▼
SourceDocument observations
    │ stable read + content verification
    ▼
durable manifest
    │ Tree-sitter frontend + adapters
    ▼
ExtractedEntity
    │ representation completion + enrichers + identity assignment
    ▼
versioned staged document JSON
    │ no-change refresh barrier
    ▼
one SQLite publish transaction
    │ entities / code_units / representations + call-site resolution
    ├──────────────► Usage representation
    │
    │ embed_space_pending(space identity, embedder)
    ▼
embedding_spaces / embeddings
    │ Db::snapshot or another backend
    ▼
IndexSnapshot
    │ SearchIndex::from_snapshot (validated)
    ▼
per-space search / similarity / rank fusion
```

## Invariants

- **Determinism.** Providers return deterministic ordering; persisted and
  snapshot rows are ordered; ranking breaks ties deterministically.
- **Stable semantic keys.** Model identities and embedding-space identities are
  persisted contracts, not display labels.
- **Incremental by provider revision and content.** Equal revisions avoid reads;
  only for authoritative providers. Advisory revisions are verified with source
  hashes by default.
- **Atomic publication.** Processing never mutates search-visible tables; all
  selected projects publish together or the prior generation remains intact.
- **Durable resume.** At most the currently computed document is lost. Ready
  payloads survive interruption and are selectively invalidated on refresh.
- **Consistent reads.** Snapshot export pins one SQLite read transaction and
  carries the published generation plus each project's last run identity.
- **Content-addressed representations.** Embeddings survive reindexing whenever
  a representation's content hash remains present.
- **Storage-neutral search.** Search validates public snapshot data and returns
  errors for malformed dimensions, duplicate space ids/hashes, or non-finite
  vectors rather than panicking.
- **Application neutrality.** The core reports semantic evidence. Duplicate,
  security, correctness, ownership, and migration conclusions belong to
  consumers such as `decombine`.

## Extension points

- New source: implement `SourceProvider`.
- New derived/imported channel: implement `RepresentationEnricher` or write a
  representation with explicit provenance.
- New language: add a grammar, language metadata, unit/reference queries, and
  fixtures; use an adapter only when necessary.
- New embedding backend: implement `EmbeddingBackend` with a reproducible
  `ModelContract` (semantic identity) and `ExecutionInfo` (provenance).
- New persistence backend: construct `IndexSnapshot` for search; a generalized
  incremental write-side store interface may be added when a second backend
  needs to reuse the indexer.
