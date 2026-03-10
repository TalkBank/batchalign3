# Pipeline System

**Status:** Current
**Last verified:** 2026-03-17

The pipeline system processes CHAT files through per-command orchestrators
that follow a parse → cache → infer → inject → validate → serialize
lifecycle. Each command has a dedicated Rust orchestrator module.

## Core Data Model

**`ChatFile`** is the typed Rust AST (from `talkbank-model`) representing a
parsed CHAT file. On the server path, orchestrators operate directly on
`ChatFile` using functions from `batchalign-chat-ops`. On the Python API
path, `ParsedChat` (a PyO3 `#[pyclass]`) wraps a `ChatFile` and exposes
mutation methods.

**Commands** define processing tasks. Available commands: `transcribe`
(ASR), `align` (forced alignment), `morphotag` (morphosyntax), `utseg`
(utterance segmentation), `translate`, `coref`, `compare`, `opensmile`,
`avqi`, and `benchmark`.

The low-level `speaker` infer task still exists for typed worker execution, but
it is not a standalone CLI command. This matches batchalign2, where diarization
was part of `transcribe_s`.

## Command Classification

Commands are classified by input/output type:

- **Generation**: Creates CHAT from media (e.g., `transcribe`). Builds a
  `ChatFile` from ASR output via `build_chat()`.
- **Processing** (default): Transforms existing CHAT in-place (e.g.,
  `morphotag`, `align`, `translate`). Parses, mutates, serializes.
- **Analysis**: Produces metrics or non-CHAT output (e.g., `opensmile`,
  `avqi`, `benchmark`). Returns structured results.

## Processing Lifecycle

Every CHAT-mutating command follows this pattern:

1. **Parse**: `parse_lenient()` produces a `ChatFile` AST.
2. **Pre-validate**: Check input quality against a command-specific
   `ValidityLevel` (e.g., `MainTierValid` for morphotag).
3. **Collect payloads**: Extract per-utterance data from the AST
   (word lists, text, language metadata).
4. **Cache check**: Hash payloads with BLAKE3. Partition into hits and
   misses.
5. **Infer**: Send misses to Python workers via typed worker IPC
   (`execute_v2` on the live infer surfaces). Workers return raw ML output.
6. **Inject**: Insert results (cache hits + infer results) into the AST.
7. **Cache put**: Persist new results for future reuse.
8. **Post-validate**: Alignment checks + semantic validation.
9. **Serialize**: `to_chat_string()` produces final CHAT output.

For generation commands (`transcribe`), step 1 is replaced by ASR inference
followed by `build_chat()` to construct the initial AST.

## Pre-Serialization Validation

The server runs validation gates before writing CHAT output:

1. **Pre-validation** — rejects malformed input early based on the
   command's required `ValidityLevel`.
2. **Alignment validation** — checks tier word counts (MOR/GRA/WOR must
   match the main tier). ParseHealth-aware: utterances flagged as
   unparseable are excluded.
3. **Semantic validation** — full CHAT validation (E362 monotonicity,
   E701/E704 temporal, header correctness). Only blocks on errors, not
   warnings.

Validation failures trigger bug reports to `~/.batchalign3/bug-reports/`
and self-correcting cache purges (deleting entries that produced invalid
output).

## Batched Inference

Text-only commands (morphotag, utseg, translate, coref) use
`dispatch_batched_infer()` to pool utterances across multiple files into a
single worker `execute_v2` request backed by one prepared-text artifact. This
improves throughput and model reuse compared to per-file dispatch without
re-expanding the Python control plane.

The morphosyntax orchestrator uses three phases for cache interaction:

1. `collect_payloads()` — extract per-utterance payloads with positions
2. `inject_from_cache()` — inject cached %mor/%gra strings
3. `inject_results()` — inject freshly inferred results

All cache logic is in Rust. Python workers receive only structured NLP
payloads and return raw model output.

## Multi-Step Pipelines

The `transcribe` command can chain multiple steps:

```
ASR inference → post-processing → CHAT assembly → utseg → morphosyntax
```

Each step is a separate orchestrator call (`process_transcribe` →
`process_utseg` → `process_morphosyntax`). Between steps, CHAT text is
serialized and re-parsed, which is not wasteful — each step operates on a
different version of the file.

## Worker Concurrency

Worker parallelism is capped based on available memory, not scaled linearly.
Each worker loads ~4-12 GB of ML models. The server computes per-job file
parallelism via `compute_job_workers()`, and a memory gate
(`memory_gate()`) defers jobs when system memory is low. Workers with
matching `(command, lang)` that are already loaded bypass the memory gate.
See [Worker Memory Architecture](worker-memory-architecture.md) for the
auto-tuning formula, memory gate internals, and pool concurrency model.

## Key Patterns

- **Times** throughout the pipeline are in **milliseconds**.
- **Language codes** use 3-letter ISO 639-3 format (`"eng"`, `"spa"`, `"jpn"`).
- **Files are sorted largest-first** before dispatch to avoid stragglers.
- **Heavy imports** (`stanza`, `torch`) are lazy — CLI startup must stay fast.
