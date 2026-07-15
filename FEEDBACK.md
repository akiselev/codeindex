## Conclusion

`codeindex` should not become “a vector-search wrapper.” It should become a **repository exploration engine** that accepts an agent’s intent, plans a hybrid search over code, documentation, usages, symbols, tests, configuration, and dependency relationships, then returns a ranked, budgeted evidence package.

The current repository already has much of the correct substrate:

* multiple representation channels;
* independently identified embedding spaces;
* storage-neutral snapshots;
* weighted reciprocal-rank fusion;
* stable entity identities and provenance.

The principal missing pieces are:

1. An embedding API that understands **query versus document**, task instructions, paired prompts, pooling, truncation, and flexible dimensions.
2. Runtime-independent model definitions.
3. Lexical retrieval and reranking.
4. Structural graph expansion.
5. A higher-level `query/context` service designed for coding agents.
6. A repository-level evaluation and fine-tuning loop.

My primary model recommendation is:

* **Qwen3-Embedding-0.6B** as the default open local model.
* **Qwen3-Embedding-4B** as the quality-oriented local/server model.
* **Qwen3-Reranker-0.6B or 4B** for second-stage ranking.
* **Jina Code Embeddings 1.5B** as an important code-task reference model, but not the default because of its noncommercial license.
* **Codestral Embed and Voyage Code 3** as managed quality baselines.
* A future **codeindex-trained Qwen3-Embedding-0.6B/4B checkpoint** specialized for repository exploration and issue localization.

The last item may ultimately matter more than moving from Qwen 4B to 8B.

---

# 1. What is missing in the current model abstraction

Your existing abstraction is:

```rust
pub trait Embedder {
    fn identity(&self) -> &ModelIdentity;
    fn dimensions(&self) -> usize;
    fn max_sequence_length(&self) -> usize;
    fn count_tokens(&self, text: &str) -> Result<usize>;
    fn embed(&mut self, inputs: &[String]) -> Result<Vec<Vec<f32>>>;
}
```

That works for symmetric encoders such as MiniLM and older BGE models. It is insufficient for current instruction-aware models.

`SearchIndex::search_text` currently sends the raw text directly to `embed()`. Although an embedding space records an `input_transform` string, the query path does not interpret it or distinguish query-side from document-side behavior.

This creates several concrete problems.

### Query and document roles

Many retrieval models intentionally embed queries and documents differently:

* Qwen3 adds an instruction to queries but normally leaves documents unprefixed.
* Jina Code applies a task-specific query prefix and a corresponding passage prefix.
* Voyage accepts explicit `query` and `document` input roles.
* Gemini Embedding 2 recommends different query and document structures for asymmetric retrieval. ([Hugging Face][1])

A bare `embed(&[String])` cannot express this.

### Dynamic task instructions

Qwen3 expects a structure equivalent to:

```text
Instruct: <description of retrieval task>
Query: <actual query>
```

Its authors report that task instructions commonly improve retrieval by 1–5%. Documents require no matching instruction, so one document index can potentially support several query intents. ([Hugging Face][1])

This is especially valuable for your use case. The agent can tell the model what relationship it wants:

```text
Given a software change request, retrieve code regions likely to require editing.

Given a question about repository behavior, retrieve code that provides evidence for the answer.

Given a code fragment, retrieve functionally equivalent implementations.

Given a failure report, retrieve implementation, tests, configuration, and error-handling paths relevant to diagnosing it.
```

These should not be hard-coded into a Qwen-specific backend. They belong in a query profile or embedding task contract.

### Paired task transforms

Jina Code is materially different. Its supported modes include:

* natural language to code;
* code to code;
* code to natural language;
* code to completion;
* technical question answering.

Each mode has both a query prefix and a passage prefix. Changing from `nl2code` to `code2code` therefore changes the document vectors and requires a distinct embedding space. ([Hugging Face][2])

That means `input_transform: String` should become a typed, role-aware contract.

### Pooling strategy

Your FastEmbed custom-model path currently handles CLS or mean pooling. Qwen3 and Jina Code use last-token pooling, left padding, and decoder-style architectures. Their official examples explicitly extract the last non-padding token.  ([Hugging Face][1])

Consequently, treating arbitrary ONNX models as interchangeable FastEmbed models will not be robust enough.

---

# 2. Model landscape

## Qwen3 Embedding: best overall fit

The Qwen3 family has 0.6B, 4B, and 8B embedding and reranking models. All have 32K context, flexible Matryoshka dimensions, user-defined task instructions, multilingual support, and explicit code-retrieval training. The models are Apache 2.0 licensed. ([Hugging Face][1])

| Model                | Maximum dimensions | Recommended codeindex role                                     |
| -------------------- | -----------------: | -------------------------------------------------------------- |
| Qwen3-Embedding-0.6B |              1,024 | Default local model, CI benchmarking, eventual fine-tuning     |
| Qwen3-Embedding-4B   |              2,560 | Best quality/performance local server tier                     |
| Qwen3-Embedding-8B   |              4,096 | Maximum-quality open baseline; probably excessive as a default |
| Qwen3-Reranker-0.6B  |                N/A | Local top-50 or top-100 reranking                              |
| Qwen3-Reranker-4B    |                N/A | Quality-oriented reranking                                     |
| Qwen3-Reranker-8B    |                N/A | Research or high-resource deployment                           |

The new CORE-Bench results are highly relevant because they evaluate the exact problem you described: an agent receives a repository state and a request, then must locate edit targets and broader supporting context rather than match an isolated docstring to a function. ([arXiv][3])

Selected results:

| Model                      | Traditional code understanding NDCG@10 | Issue-to-edit NDCG@10 / Recall@100 | Broader context NDCG@10 / Recall@100 |
| -------------------------- | -------------------------------------: | ---------------------------------: | -----------------------------------: |
| Current CodeRankEmbed      |                                   47.4 |                        12.1 / 32.9 |                          22.5 / 28.6 |
| Qwen3 0.6B                 |                                   66.9 |                        17.0 / 45.5 |                          32.6 / 40.2 |
| Jina Code 1.5B             |                                   56.2 |                        17.0 / 48.5 |                          31.6 / 42.7 |
| Qwen3 4B                   |                                   72.7 |                        18.3 / 46.9 |                          32.8 / 40.8 |
| Qwen3 8B                   |                                   71.7 |                        20.3 / 48.0 |                          34.4 / 41.5 |
| Qwen3 0.6B, repository SFT |                                   58.1 |                        26.5 / 59.4 |                          44.5 / 54.4 |
| Qwen3 4B, repository SFT   |                                   59.8 |                        30.3 / 66.0 |                          49.2 / 61.6 |
| Qwen3 8B, repository SFT   |                                   63.0 |                        32.8 / 66.4 |                          50.2 / 61.4 |

The important conclusion is not merely that Qwen3 is strong. It is that **repository-specific supervision is more valuable than model scale**. A fine-tuned 0.6B model substantially outperforms an off-the-shelf 8B model on agent-oriented retrieval, while the 4B fine-tune nearly reaches the 8B fine-tune. ([arXiv][3])

I would therefore make Qwen3-0.6B the reference implementation and design the training pipeline early.

## Jina Code Embeddings

Jina’s code models are unusually well matched to codeindex’s multiple representation channels. The 1.5B model supports explicit task profiles for NL-to-code, code-to-code, code-to-comment, code completion, and technical QA, with 32K context, last-token pooling, and Matryoshka dimensions from 128 through 1,536. ([Hugging Face][2])

Potential mappings:

| Jina task         | codeindex operation                                                |
| ----------------- | ------------------------------------------------------------------ |
| `nl2code`         | “Where is authentication enforced?”                                |
| `code2code`       | Similar implementations and duplicate logic                        |
| `code2nl`         | Match implementations to documentation or generated descriptions   |
| `code2completion` | Retrieve analogues for partially written code                      |
| `qa`              | Search documentation, usages, comments, and generated descriptions |

Its main limitation is licensing: the published 1.5B model is CC-BY-NC-4.0, so it cannot be the unrestricted default for a general-purpose crate. It remains valuable as a benchmark, research integration, and optional user-selected model. ([Hugging Face][2])

## SweRank

SweRank is specialized for natural-language issue descriptions and identifying files, classes, and functions likely to require modification. Its training data consists of real GitHub issues paired with corresponding code changes, and it uses a retrieve-and-rerank design. ([arXiv][4])

This is a better semantic match for:

```text
“Implement durable content recovery after snapshots expire.”

“Fix the race when two indexing runs overlap.”

“Add another embedding provider without breaking model identity.”
```

than conventional docstring-to-function training.

I would treat SweRank as:

* a specialized `locate-edit` profile;
* an evaluation baseline;
* a source of training methodology;
* not necessarily the universal embedding model.

Its strongest contribution is showing that **issue localization deserves its own task distribution**.

## Codestral Embed

Codestral Embed is a proprietary managed code-embedding service. Mistral positions it specifically for repository retrieval, coding-agent RAG, semantic code search, duplicate detection, clustering, and code analytics. Its published evaluations include SWE-Bench file localization, commit-message-to-code retrieval, code-to-code retrieval, and text-to-code tasks. It supports flexible dimensions and output precision, with an 8,192-token context window. ([Mistral AI][5])

This should be one of codeindex’s cloud quality baselines. It is especially useful for testing whether your open Qwen pipeline is leaving substantial quality on the table.

Its benchmark comparisons are vendor-reported, so codeindex should reproduce the relevant evaluation on its own corpus rather than encode “Codestral is best” as an assumption.

## Voyage Code 3

`voyage-code-3` is a managed model optimized for code retrieval, with 32K context and selectable dimensions of 256, 512, 1,024, or 2,048. ([Voyage AI][6])

Voyage exposes query/document roles rather than arbitrary task instructions. It is a good production cloud baseline for:

* natural-language-to-code retrieval;
* code RAG;
* code-to-code similarity;
* comparison against local deployments.

It is less expressive than Qwen’s arbitrary query instruction but operationally simple.

## Gemini Embedding 2

Gemini Embedding 2 supports explicit textual task instructions, including a documented `code retrieval` task structure. It also supports multimodal inputs, making it interesting for repositories containing screenshots, architecture figures, PDFs, design documents, or UI reference images. ([Google AI for Developers][7])

Example contract:

```text
Query:
task: code retrieval | query: where is resumable indexing committed atomically?

Document:
title: IndexRunBuilder::run | text: <implementation>
```

It is not the model I would choose as the primary source-code backend, but it could eventually power a **mixed repository-artifact space**.

One adapter-specific complication is that Gemini Embedding 2 aggregates multiple contents into one embedding in its ordinary API; codeindex would need per-item calls or its batch API rather than assuming every API accepts a string array and returns one vector per string. ([Google AI for Developers][7])

## Other useful models

**E5-Mistral-7B-Instruct** remains competitive on repository-context recall in CORE-Bench, despite not being code-specific. It is a useful instruction-aware baseline but too large for the default tier. ([arXiv][3])

**C2LLM** uses coder backbones and trainable pooling rather than relying solely on an EOS bottleneck. It is useful research for future native model support and pooler abstraction, although Qwen3 is currently a more practical initial target. ([arXiv][3])

**CodeXEmbed** and **F2LLM-v2** are worth including in the benchmark harness, but I would not implement dedicated integrations before Qwen3, Jina, and generic Hugging Face serving work.

**OpenAI `text-embedding-3-small` and `-large`** support flexible dimensions and generic code search but have no explicit query/document or task-instruction contract. They are useful generic API compatibility tests, not the best architectural model for codeindex. ([OpenAI Platform][8])

**Cohere Embed v4** has 128K context, selectable dimensions, and multimodal document support, but is general-purpose rather than code-specialized. It is more relevant to mixed documentation than primary code search. ([Cohere Documentation][9])

---

# 3. Backends to implement

Models and backends must be separate concepts.

A `QwenBackend` or `JinaBackend` would couple model semantics to one runtime. The same Qwen model can run through Transformers, SentenceTransformers, TEI, vLLM, and potentially a validated GGUF runtime. Its prompt and pooling contract must remain identical across them. Qwen officially documents both vLLM and TEI deployment. ([Hugging Face][1])

## Priority 1: generic HTTP service backends

Implement three small protocol clients.

### OpenAI-compatible embeddings and reranking

This supports:

* vLLM;
* llama.cpp;
* many hosted inference providers;
* OpenAI-compatible internal services.

vLLM exposes `/v1/embeddings`, Cohere-compatible embedding endpoints, scoring, and reranking endpoints. llama.cpp exposes OpenAI-compatible embeddings and dedicated reranking endpoints. ([vLLM][10])

The adapter should accept already-rendered inputs. It should not decide whether Qwen receives an instruction or Jina receives a passage prefix.

### Hugging Face TEI

TEI provides dynamic batching, token-based scheduling, Candle-based optimized inference, metrics, and production serving for open embedding models. Qwen publishes an official TEI deployment recipe. ([Hugging Face][11])

This is probably the best first-class server backend for Qwen3-0.6B and 4B.

### Ollama

Ollama’s embedding API accepts batches, optional truncation behavior, and an output dimension. It does not expose a general task field, so codeindex should render role/task prompts itself before making the request. ([Ollama][12])

Ollama is valuable for user ergonomics but should be an adapter over the same model contract, not a source of model semantics.

## Priority 2: SentenceTransformers subprocess worker

A managed Python JSONL worker gives you immediate support for nearly any research model:

```text
Rust codeindex
    │ JSONL over stdio or local socket
    ▼
Python worker
    ├── SentenceTransformer
    ├── Transformers
    ├── model prompt definitions
    ├── custom pooling
    └── CUDA / MPS / CPU
```

This should be the **reference correctness backend**.

Advantages:

* handles last-token and custom pooling;
* supports prompt names;
* loads arbitrary Hugging Face revisions;
* makes new research models testable without modifying Rust;
* gives you a reference against which native and quantized runtimes can be compared.

It is not the cleanest distribution story, but it prevents codeindex’s model support from being bounded by FastEmbed’s catalog.

## Priority 3: retain and generalize FastEmbed

The current FastEmbed backend is still useful for:

* CodeRankEmbed;
* BGE small/base;
* Nomic;
* Snowflake Arctic Embed;
* Jina Embeddings v2 Base Code;
* low-memory CPU installations.

Keep it as the bundled zero-configuration backend. Add explicit capabilities and typed pooling rather than trying to force decoder-based models through it immediately.

## Priority 4: native Candle

A native Candle backend could eventually provide:

* one-binary Rust deployments;
* CUDA, Metal, and CPU execution;
* direct safetensors loading;
* model-specific optimized implementations.

It is a larger maintenance commitment because codeindex would own architecture implementations, poolers, tokenizer behavior, padding, quantization support, and compatibility testing. Implement it after the service and subprocess contracts are stable.

## Priority 5: llama.cpp/GGUF

llama.cpp now exposes embeddings, configurable pooling, token embeddings, and reranking endpoints. It is attractive for quantized local Qwen deployments. ([GitHub][13])

Treat it as experimental until you validate:

* exact prompt rendering;
* last-token pooling;
* truncation side;
* output dimensions;
* normalization;
* ranking correlation against the BF16 reference;
* performance degradation under each quantization.

Quantization changes the produced vectors, so the quantization artifact must be part of vector semantics or at least a strict compatibility key.

---

# 4. Redesign the embedding contract

I would replace `Embedder::embed()` with something closer to:

```rust
pub trait EmbeddingBackend: Send {
    fn execution_identity(&self) -> &ExecutionIdentity;

    fn capabilities(&self) -> &EmbeddingBackendCapabilities;

    fn embed(
        &mut self,
        model: &ResolvedEmbeddingModel,
        request: EmbeddingRequest<'_>,
    ) -> Result<EmbeddingBatch>;
}

pub struct EmbeddingRequest<'a> {
    pub role: EmbeddingRole,
    pub task: &'a EmbeddingTask,
    pub inputs: &'a [EmbeddingInput<'a>],
    pub output_dimensions: Option<usize>,
    pub truncation: TruncationPolicy,
}

pub enum EmbeddingRole {
    Query,
    Document,
    Symmetric,
}

pub struct EmbeddingInput<'a> {
    pub text: &'a str,
    pub title: Option<&'a str>,
    pub metadata: &'a BTreeMap<String, String>,
}
```

The model definition should describe semantics independently of execution:

```rust
pub struct EmbeddingModelContract {
    pub artifact: ModelArtifactIdentity,
    pub tokenizer: TokenizerIdentity,
    pub max_sequence_length: usize,
    pub padding_side: PaddingSide,
    pub truncation_side: TruncationSide,
    pub pooling: PoolingStrategy,
    pub normalization: Normalization,
    pub dimensions: DimensionContract,
    pub prompting: PromptContract,
}
```

Prompt contracts need several variants:

```rust
pub enum PromptContract {
    Symmetric,

    // Qwen3: documents remain unchanged; query instruction may vary.
    QueryInstruction {
        template: PromptTemplate,
    },

    // Jina Code, E5: both sides depend on a task profile.
    PairedTask {
        tasks: BTreeMap<TaskId, PairedPromptTemplates>,
    },

    // Provider-owned semantic task such as input_type or task_type.
    ProviderTask {
        supported_tasks: BTreeSet<TaskId>,
    },
}
```

## Split semantic identity from execution provenance

Your present `ModelIdentity` contains:

* model and revision;
* tokenizer and model hashes;
* dimensions and normalization;
* backend/runtime version;
* execution provider;
* quantization;
* cache path.

These are not all the same kind of identity.

`cache_path` certainly does not change vector meaning. CPU versus CUDA usually should not make two spaces incompatible. Conversely, prompt template, pooling, output dimension, tokenizer revision, and quantization can change every vector.

Use two identities:

```rust
pub struct VectorSemanticsIdentity {
    pub model_artifact_hash: String,
    pub tokenizer_hash: String,
    pub pooling: PoolingStrategy,
    pub normalization: Normalization,
    pub dimensions: usize,
    pub document_transform_hash: String,
    pub prompt_contract_version: String,
    pub quantization: Option<QuantizationIdentity>,
}

pub struct ExecutionIdentity {
    pub backend: String,
    pub backend_version: String,
    pub runtime_version: Option<String>,
    pub device: String,
    pub execution_provider: String,
    pub cache_path: Option<PathBuf>,
}
```

Persist both, but use `VectorSemanticsIdentity` for search-space compatibility.

For query-only instruction models, log the exact query instruction and template hash with each search result. It affects retrieval reproducibility but does not require re-embedding documents.

---

# 5. Add reranking as a separate primitive

Do not model rerankers as embedders.

```rust
pub trait Reranker: Send {
    fn identity(&self) -> &RerankerIdentity;

    fn rerank(
        &mut self,
        request: RerankRequest<'_>,
    ) -> Result<Vec<RerankScore>>;
}

pub struct RerankRequest<'a> {
    pub task: &'a RerankingTask,
    pub query: &'a str,
    pub candidates: &'a [RerankCandidate<'a>],
}
```

Embedding retrieval should select perhaps 50–200 candidates cheaply. A reranker then jointly examines the query and each candidate.

Initial implementations:

* Qwen3 Reranker through vLLM or a Python worker;
* SweRank for issue localization;
* llama.cpp `/v1/rerank`;
* managed provider rerank APIs.

CORE-Bench and CoREB both indicate that repository-oriented retrieval is not solved by selecting a single universally best embedding model. Task-specific reranking and fine-tuning remain important, particularly for realistic developer queries and issue localization. ([arXiv][3])

---

# 6. The agent-facing retrieval pipeline

The strongest current research says that embeddings alone are not enough.

CORE-Bench shows a severe quality drop when moving from snippet search to issue-to-edit and broader-context retrieval. SWE-Explore finds that interactive agentic explorers form a distinct tier above classical retrieval, and that line-level coverage, early useful evidence, and context efficiency correlate with downstream repair success. ([arXiv][3])

I would implement this pipeline:

```text
Agent question
    │
    ▼
Intent classification / explicit profile
    │
    ├── exact identifiers and symbols
    ├── lexical BM25/FTS
    ├── one or more dense embedding spaces
    └── metadata/path/language filters
    │
    ▼
Weighted reciprocal-rank fusion
    │
    ▼
Structural graph expansion
    │ callers / callees / imports / implementations / tests / config
    ▼
Cross-encoder or generative reranker
    │
    ▼
Line-aware context packing and deduplication
    │
    ▼
Structured evidence package for the coding agent
```

## Lexical retrieval is mandatory

Embedding models are weak at:

* exact identifiers;
* error messages;
* filenames;
* issue numbers;
* configuration keys;
* unusual literals;
* dependency versions.

Add SQLite FTS5 or an equivalent storage-neutral lexical result provider. Fuse lexical results with your existing RRF implementation rather than trying to normalize BM25 and cosine scores directly.

## Use multiple representations intentionally

Your current channels are a strong foundation.

Suggested spaces:

| Space                 | Representation            | Intended questions                        |
| --------------------- | ------------------------- | ----------------------------------------- |
| `code/implementation` | `Implementation`          | General NL-to-code retrieval              |
| `code/body-anonymous` | `BodyWithoutDeclaredName` | Functional analogues and duplicate logic  |
| `code/signature`      | `Signature`               | API and type-shape search                 |
| `symbol/lexical`      | `Symbol`                  | Exact and fuzzy identifiers               |
| `docs`                | `Documentation`           | Concepts, contracts, intended behavior    |
| `usage`               | `Usage`                   | How an API is actually consumed           |
| `description`         | `GeneratedDescription`    | Higher-level conceptual questions         |
| `tests`               | Test entities and usages  | Behavior, invariants, regression coverage |
| `configuration`       | Custom representation     | Features, flags, schemas, build behavior  |

Do not embed every channel with every model. Define named search profiles that select a small set.

## Structural graph expansion

LocAgent’s result is directly relevant: it represents files, classes, and functions as a heterogeneous graph with imports, invocations, inheritance, and containment, then performs multi-hop localization. ([arXiv][14])

Your planned LSP/SCIP work should therefore be treated as part of retrieval, not merely extra metadata.

After initial semantic hits, expand to:

* callers and callees;
* definitions and references;
* trait/interface implementations;
* imported or re-exported symbols;
* enclosing types and modules;
* tests that reference the hit;
* configuration and registration sites;
* sibling implementations.

The graph should be used to gather candidate evidence. It should not replace embeddings or lexical search.

## Context packing

Agents do not need the ten highest-scoring functions if eight are near-duplicates. They need a useful, diverse context package within a line or token budget.

The packer should optimize for:

* relevance;
* coverage of distinct evidence roles;
* dependency support;
* diversity;
* compact line ranges;
* early placement of high-confidence evidence.

SWE-Explore explicitly evaluates ranked code regions under a fixed line budget and finds this more predictive than simple file-level hits. ([arXiv][15])

---

# 7. Search profiles for agents

Expose intent as a first-class concept rather than expecting every agent to invent prompts.

```rust
pub enum RepositoryQueryIntent {
    LocateDefinition,
    LocateEditTargets,
    ExplainBehavior,
    TraceDataFlow,
    FindAnalogues,
    FindUsages,
    FindTests,
    FindConfiguration,
    DiagnoseFailure,
    AssessImpact,
    FindQualityExamples,
    BroadContext,
}
```

Examples:

### `LocateEditTargets`

Question:

```text
Add checkpoint recovery when the original source snapshot disappears.
```

Plan:

* Qwen instruction specialized for software change requests;
* implementation, usage, tests, configuration, and generated-description spaces;
* lexical extraction of `checkpoint`, `source snapshot`, `recovery`;
* graph expansion around source-workspace and cache entities;
* issue-localization reranker;
* favor high recall.

### `ExplainBehavior`

Question:

```text
How does codeindex ensure that partially completed indexing runs never become visible?
```

Plan:

* documentation and generated descriptions;
* implementation search;
* Usage search;
* expansion to transactions and run-status state machine;
* favor evidence diversity rather than probable edit targets.

### `FindAnalogues`

Question:

```text
Where else do we validate a persisted semantic identity before loading data?
```

Plan:

* query `BodyWithoutDeclaredName`;
* code-to-code and NL-to-code spaces;
* omit declared names;
* rerank for behavioral equivalence.

### `AssessImpact`

Question:

```text
What would break if EmbeddingSpaceIdentity stopped including execution_provider?
```

Plan:

* symbol search;
* definitions and references;
* serialization/storage call graph;
* tests touching identity comparison;
* downstream crates and public API surfaces.

---

# 8. Agent API and CLI

The current CLI only supports indexing lifecycle operations: `index`, `status`, `abandon`, and `supersede`. There is no search or agent-facing query command yet.

Keep the reusable service in library crates, then expose it through `codeindex-cli`.

Recommended commands:

```text
codeindex query "How is embedding-space compatibility validated?"
    --intent explain-behavior
    --budget 1200
    --json

codeindex query "Add support for Qwen embeddings"
    --intent locate-edit-targets
    --profile agent-change
    --json

codeindex similar entity:<id>
    --intent find-analogues
    --space code/body-anonymous

codeindex context "Why can an index run be resumed?"
    --budget-tokens 12000
    --include-related tests,callers,configuration

codeindex explain-search <query-id>

codeindex models list
codeindex models inspect qwen3-embedding-0.6b
codeindex models doctor
codeindex models benchmark --suite repository-exploration
```

The JSON result should contain more than paths and scores:

```json
{
  "query": {
    "text": "...",
    "intent": "locate_edit_targets",
    "instruction": "...",
    "profile": "agent-change-v1"
  },
  "index": {
    "generation": 42,
    "projects": ["codeindex"],
    "spaces": ["code/implementation", "usage", "docs"]
  },
  "hits": [
    {
      "entity_id": "...",
      "path": "crates/search/src/lib.rs",
      "lines": [287, 310],
      "representation": "implementation",
      "snippet": "...",
      "scores": {
        "lexical": 8.4,
        "dense_rank": 2,
        "reranker": 0.91,
        "fused": 0.044
      },
      "contributions": [
        {"space": "code/implementation", "rank": 2},
        {"space": "usage", "rank": 11}
      ],
      "relationships": [
        {"kind": "calls", "target": "..."}
      ]
    }
  ],
  "context": {
    "lines": 876,
    "estimated_tokens": 6310
  }
}
```

The agent should be able to distinguish:

* why a result was retrieved;
* which model and task produced it;
* whether it was an exact lexical hit;
* which related entities were added structurally;
* whether evidence was omitted because of the budget.

A CLI plus stable JSON is the first integration. A stdio JSON-RPC or MCP server can be layered on the same query service later.

---

# 9. Benchmarking and training strategy

Do not choose the default model from MTEB or CodeSearchNet alone. Those benchmarks do not adequately represent repository-level agent questions. CORE-Bench was created specifically because traditional snippet-oriented evaluation misses edit localization and broader context. ([arXiv][3])

Build three evaluation layers.

## Public benchmarks

Use:

* CORE-Bench for code understanding, edit localization, and broader context;
* SweLoc/SweRank evaluations for issue localization;
* CoIR for conventional code retrieval;
* CoQuIR for quality-aware retrieval;
* SWE-Explore for line-level coverage and context efficiency. ([arXiv][4])

## Repository-history benchmark

Mine repositories with known changes:

```text
query:
    issue text / PR description / commit message

strong positives:
    modified entities and lines

broader positives:
    tests, callers, declarations, config, and documentation consulted or
    structurally connected to the patch

hard negatives:
    semantically similar entities from the same repository that were not relevant
```

Repository-local hard negatives are critical because the agent must distinguish similar implementations inside one project, not unrelated snippets from a global corpus. CORE-Bench uses repository-local negatives for the same reason. ([arXiv][3])

## Agent-trajectory benchmark

Log what successful agents read before producing a passing patch:

* files;
* line ranges;
* query sequence;
* graph traversals;
* order of reads;
* final used context.

This follows the trajectory-grounded approach in SWE-Explore. It allows codeindex to optimize not merely for patched files, but for the context that made a correct patch possible. ([arXiv][15])

Metrics should include:

* NDCG@10 for precise edit targets;
* Recall@100 for broader context;
* file and entity hit rate;
* line-level recall and precision;
* first useful hit;
* nDCG under a line/token budget;
* context efficiency;
* downstream patch success with only retrieved context.

## Fine-tuning

The first training target should be Qwen3-Embedding-0.6B, followed by 4B.

Training examples should cover distinct tasks:

```text
instruction: retrieve code likely to require modification for this request
query: PR or issue text
positive: patched function/class/file
negative: repository-local similar entity
```

```text
instruction: retrieve supporting context needed to understand and safely implement this change
query: PR or issue text
positive: tests, callers, configuration, related interfaces
negative: plausible but unused neighboring code
```

```text
instruction: retrieve a functionally equivalent implementation
query: source body or generated description
positive: known analogue
negative: same vocabulary but different behavior
```

Keep task instructions in the training set. The model should learn your stable agent intents rather than one generic “retrieve relevant code” instruction.

---

# 10. Recommended implementation sequence

## Phase 1: correct the model API

1. Introduce `EmbeddingRole`, `EmbeddingTask`, and `EmbeddingRequest`.
2. Replace string `input_transform` with a typed prompt/transform contract.
3. Split vector-semantic identity from execution identity.
4. Add pooling, padding, truncation, and dimension capabilities.
5. Add model manifests for Qwen3 0.6B, 4B, and 8B.
6. Add golden-vector conformance tests across runtimes.

## Phase 2: add practical runtimes

1. TEI backend.
2. OpenAI-compatible embedding backend.
3. SentenceTransformers JSONL worker.
4. Ollama backend.
5. Provider adapters for Voyage, Codestral, Gemini, OpenAI, and Cohere.
6. Preserve FastEmbed as the bundled lightweight implementation.

## Phase 3: build the query service

1. Add lexical FTS/BM25 retrieval.
2. Generalize RRF to lexical and dense result sets.
3. Add query profiles and explicit intents.
4. Add `query`, `context`, and `explain-search`.
5. Return structured evidence and stable JSON.

## Phase 4: reranking and graph retrieval

1. Add the `Reranker` trait.
2. Integrate Qwen3 Reranker.
3. Add relationship graph storage and expansion from LSP/SCIP.
4. Add diversity-aware, line-budgeted context packing.
5. Add test/configuration-specific retrieval.

## Phase 5: evaluation and specialization

1. Implement CORE-Bench and SWE-Explore adapters.
2. Mine PR/issue-based training and evaluation data.
3. Record agent read trajectories.
4. Fine-tune Qwen3-0.6B.
5. Evaluate Qwen3-4B only after the 0.6B training pipeline is stable.
6. Ship model recommendations based on measured latency, memory, recall, and downstream repair performance.

The most important near-term design decision is the role/task contract. Once persisted document vectors have been generated with the wrong or underspecified prompt semantics, correcting the abstraction becomes an index migration. The runtime adapters themselves are comparatively straightforward.

[1]: https://huggingface.co/Qwen/Qwen3-Embedding-8B "Qwen/Qwen3-Embedding-8B · Hugging Face"
[2]: https://huggingface.co/jinaai/jina-code-embeddings-1.5b "jinaai/jina-code-embeddings-1.5b · Hugging Face"
[3]: https://arxiv.org/html/2606.11864 "CORE-Bench: A Comprehensive Benchmark for Code Retrieval in the Era of Agentic Coding"
[4]: https://arxiv.org/abs/2505.07849?utm_source=chatgpt.com "SweRank: Software Issue Localization with Code Ranking"
[5]: https://mistral.ai/news/codestral-embed "Codestral Embed | Mistral AI"
[6]: https://docs.voyageai.com/docs/embeddings "Text Embeddings"
[7]: https://ai.google.dev/gemini-api/docs/embeddings "Embeddings  |  Gemini API  |  Google AI for Developers"
[8]: https://platform.openai.com/docs/guides/embeddings "
  Vector embeddings | OpenAI API
"
[9]: https://docs.cohere.com/docs/cohere-embed "Cohere's Embed Models (Details and Application) | Cohere"
[10]: https://docs.vllm.ai/en/latest/models/pooling_models/ "Pooling Models - vLLM"
[11]: https://huggingface.co/docs/text-embeddings-inference/index "Text Embeddings Inference · Hugging Face"
[12]: https://docs.ollama.com/api/embed "Generate embeddings - Ollama"
[13]: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md "llama.cpp/tools/server/README.md at master · ggml-org/llama.cpp · GitHub"
[14]: https://arxiv.org/abs/2503.09089?utm_source=chatgpt.com "LocAgent: Graph-Guided LLM Agents for Code Localization"
[15]: https://arxiv.org/abs/2606.07297 "SWE-Explore: Benchmarking How Coding Agents Explore Repositories">

