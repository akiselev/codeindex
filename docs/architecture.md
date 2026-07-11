# Architecture

`codeindex` is split by dependency and change boundary, not by command or user
interface. The workspace is a reusable library substrate. A future
`codeindex-cli`, IDE extension, daemon, binding, or analyzer is a consumer of
these crates rather than part of the core architecture.

## Crate boundaries

### `codeindex-core`

Application-neutral vocabulary:

- source languages and spans;
- logical `EntityId` and exact `EntityVersionId`;
- `RepresentationKind`, `Representation`, and `RepresentationOrigin`;
- exact `ModelIdentity`;
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
7. persists units and representations;
8. resolves call sites into the derived `Usage` channel;
9. projects representation content into explicit embedding spaces.

`SourceProviderCatalog` supplies the same source abstraction during embedding
text recovery under lean retention.

### `codeindex-embedding`

Storage- and parser-free embedding primitives:

- `Embedder`;
- fastembed/custom ONNX execution;
- managed models and accelerator diagnostics;
- exact tokenizer accounting;
- length-sorted token-area batch packing;
- vector normalization and token statistics.

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
+ exact model identity
+ input transform
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
or input transform.

## Data flow

```text
SourceProvider
    │ enumerate/read
    ▼
SourceDocument
    │ Tree-sitter frontend + adapters
    ▼
ExtractedEntity
    │ representation completion + enrichers + identity assignment
    ▼
entities / code_units / representations
    │ call-site resolution
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
  changed revisions are verified with source hashes.
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
- New embedding backend: implement `Embedder` with a reproducible
  `ModelIdentity`.
- New persistence backend: construct `IndexSnapshot` for search; a generalized
  incremental write-side store interface may be added when a second backend
  needs to reuse the indexer.
