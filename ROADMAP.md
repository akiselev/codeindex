# Roadmap — codeindex toward 1.0

`codeindex` starts life as the reusable engine behind
[decombine](../decombine2). The path to 1.0 is about turning that engine into a
substrate two first-class consumers can stand on — an **agent-facing query CLI**
and a **Python binding for embedding experiments** — and then hardening the
public API enough to publish and support it.

1.0 means: stable public APIs under semver, published to crates.io, with the CLI
and Python surfaces built *on the libraries* (proving the boundaries hold), and a
schema that supports multi-channel retrieval rather than a single body vector.

## Milestones

### M1 — Standalone search API (foundation)
Lift the embed → rank → resolve-to-unit flow (today only in decombine's
`AnalysisContext`) into the library, with model-identity verification and the
`unit:<id>` selectors. Everything below depends on this being a library call.
*Exit:* a consumer can answer "where is the code that does X?" in a handful of
lines, no decombine code copied.

### M2 — `codeindex` CLI
A binary with `index`, `embed`, `query`, `search`, and `capabilities`
subcommands emitting stable JSON envelopes for agents. Generalize decombine's
`query` command family (`docs/query-interface.md` in that repo) into a
project-agnostic tool. This is the "agents find functionality before it becomes
a dedupe problem" use case.
*Exit:* `codeindex search "retry with backoff" --where language=rust` returns
ranked units as JSON.

### M3 — Python bindings
PyO3 + maturin wheels over `codeindex-embedding` (embed arbitrary text, inspect
vectors, token stats) and the M1 search API. The embedding crate was
deliberately kept free of SQLite and the grammars for exactly this — a notebook
user should `pip install` and embed without compiling a C SQLite or twelve
parsers.
*Exit:* `import codeindex; codeindex.embed([...])` works in a notebook; a
published wheel for Linux/macOS.

### M4 — Multi-representation & entity versions
Wire the dormant `codeindex-core` vocabulary: multiple `RepresentationKind`
channels (`Signature`, `Documentation`, `Symbol`, `Usage`), and
`EntityId`/`EntityVersionId` identity that tracks an entity across index
generations. Requires a `codeindex-sqlite` schema migration (kept intentionally
separate from the extraction). Unlocks channel-specific retrieval — search
signatures vs bodies vs docstrings — and change-tracking over time.
*Exit:* a query can target a channel; re-indexing a renamed function preserves
its logical identity.

### M5 — Publish & stabilize
Semver-audit and freeze the public APIs, achieve rustdoc coverage
(`deny(missing_docs)`), write a CHANGELOG, and publish the crates to crates.io.
Establish a deprecation policy for the pre-1.0 churn.
*Exit:* `cargo add codeindex` from crates.io; 1.0.0 tagged.

### M6 — Platform & accelerator matrix
CI-tested embedding on Linux/CUDA, macOS/CoreML, and Windows/DirectML, plus
per-platform managed-model distribution (including the fp16 variant) with
hash verification. Track and close the macOS/Windows gaps carried over from
decombine.
*Exit:* documented, tested support tier per OS/accelerator.

## Supporting tracks (continuous)

- **Retrieval quality & benchmarks.** Bring an eval harness into the repo (the
  calibration/OSS-eval work currently lives in `decombine/runs`) so model and
  threshold changes are measured against the substrate directly, not only
  through decombine.
- **Language coverage.** More bundled grammars, and a path for consumers to
  register non-bundled grammars at runtime rather than only the compiled-in set.
- **Serving.** A long-running index/query daemon (and/or an MCP server) so agents
  query a live, incrementally-updated index without reloading — the natural home
  once M1/M2 exist.

## Non-goals (for now)

- Reimplementing decombine's duplication/comparison analyzers here — those stay
  application-level; codeindex provides the index, embeddings, and ranking they
  build on.
- A hosted/remote embedding service — codeindex is local-first by design.
