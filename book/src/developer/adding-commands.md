# Adding a New Command

**Status:** Current
**Last updated:** 2026-03-21 13:00

This guide walks through adding a new batchalign3 command end-to-end.

**Reference implementations by workflow family:**

| Family | Best example | Files to read |
|--------|-------------|---------------|
| `PerFileTransform` (one file → one output) | **`align`** | `workflow/fa.rs` (54 lines), `fa/mod.rs` |
| `CrossFileBatchTransform` (batch-infer pool) | `morphotag` | `workflow/morphosyntax.rs`, `morphosyntax/` |
| `ReferenceProjection` (compare against gold) | `compare` | `workflow/compare.rs`, `compare.rs` |
| `Composite` (orchestrates sub-workflows) | `benchmark` | `workflow/benchmark.rs`, `benchmark.rs` |

**Start with `align`** — it's the simplest complete example (54 lines for the
workflow wrapper). If your command takes CHAT text in and produces modified CHAT
out, follow `align`. If your command needs to batch multiple files through one
ML call, follow `morphotag`.

## Quick start

```bash
make check    # after each file edit (~6s)
make test     # verify nothing broke (~6s)
```

## Architecture overview

Every command flows through these layers:

```
CLI args → CommandProfile → DispatchRequest → JobSubmission → Runner → Workflow → Worker → Output
```

The key files, in the order you'll edit them:

| Step | File | What you add |
|------|------|-------------|
| 1 | `batchalign-types/src/domain.rs` | `ReleasedCommand::YourCommand` variant |
| 2 | `batchalign-app/src/workflow/registry.rs` | `CommandWorkflowDescriptor` entry |
| 3 | `batchalign-app/src/workflow/your_command.rs` | Workflow trait impl |
| 4 | `batchalign-app/src/your_command.rs` | Core logic (ML dispatch, post-processing) |
| 5 | `batchalign-cli/src/args/commands.rs` | CLI arg struct |
| 6 | `batchalign-cli/src/args/mod.rs` | `CommandProfile` match arm |
| 7 | `batchalign-cli/src/args/options.rs` | `CommandOptions` variant + `build_typed_options` arm |

## Step 1: Add the ReleasedCommand variant

```rust
// crates/batchalign-types/src/domain.rs
pub enum ReleasedCommand {
    // ... existing ...
    YourCommand,  // ← add here
}
```

Update the `ALL` array, `as_str()`, `TryFrom<&str>`, and `From<ReleasedCommand> for CommandName`.

## Step 2: Register the workflow descriptor

```rust
// crates/batchalign-app/src/workflow/registry.rs
const RELEASED_COMMAND_WORKFLOWS: &[CommandWorkflowDescriptor] = &[
    // ... existing ...
    CommandWorkflowDescriptor {
        command: ReleasedCommand::YourCommand,
        family: WorkflowFamily::CrossFileBatchTransform,  // pick one of 4 families
        infer_task: InferTask::YourTask,                  // or reuse existing
        capability_kind: CommandCapabilityKind::DirectInfer,
        uses_local_audio: false,
        output_path_kind: CommandOutputPathKind::PreserveInputName,
        runner_dispatch_kind: RunnerDispatchKind::BatchedTextInfer,
    },
];
```

### Which workflow family?

| Family | Use when | Example |
|--------|----------|---------|
| `PerFileTransform` | One file in → one output | `align`, `transcribe` |
| `CrossFileBatchTransform` | Pool files, batch-infer, fan out | `morphotag`, `utseg`, `translate`, `coref` |
| `ReferenceProjection` | Compare against a reference | `compare` |
| `Composite` | Orchestrate sub-workflows | `benchmark` |

### Which runner dispatch kind?

| Kind | Use when |
|------|----------|
| `BatchedTextInfer` | Text-only commands (CHAT in → CHAT out, no audio) |
| `ForcedAlignment` | Per-file audio alignment |
| `TranscribeAudioInfer` | ASR transcription |
| `BenchmarkAudioInfer` | Benchmark orchestration |
| `MediaAnalysisV2` | Audio feature extraction (openSMILE, AVQI) |

Most text-only commands use `BatchedTextInfer`.

## Step 3: Implement the workflow

Create `crates/batchalign-app/src/workflow/your_command.rs`.

**For a per-file command** (like `align` — the common case):

```rust
// workflow/your_command.rs
use async_trait::async_trait;
use crate::api::ChatText;
use crate::error::ServerError;
use crate::pipeline::PipelineServices;
use super::PerFileWorkflow;

/// Borrowed request bundle for one execution.
pub(crate) struct YourCommandRequest<'a> {
    pub chat_text: ChatText<'a>,
    pub services: PipelineServices<'a>,
    // ... command-specific params ...
}

/// Per-file workflow.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct YourCommandWorkflow;

#[async_trait]
impl PerFileWorkflow for YourCommandWorkflow {
    type Output = String;  // modified CHAT text
    type Request<'a> = YourCommandRequest<'a> where Self: 'a;

    async fn run(&self, request: Self::Request<'_>) -> Result<Self::Output, ServerError> {
        run_your_command_impl(request.chat_text.as_ref(), request.services).await
    }
}
```

See `workflow/fa.rs` (54 lines) for the real working version.

**For a batch command** (like `morphotag`): use `CrossFileBatchWorkflow` instead.
See `workflow/morphosyntax.rs` (66 lines).

Register the new module in `workflow/mod.rs`:

```rust
pub(crate) mod your_command;
```

## Step 4: Core logic

Create `crates/batchalign-app/src/your_command.rs` with the actual ML dispatch:

```rust
pub(crate) async fn run_your_command_impl(
    chat_text: &str,
    services: PipelineServices<'_>,
    params: &YourCommandParams<'_>,
) -> Result<String, ServerError> {
    // 1. Parse CHAT text
    // 2. Build infer request
    // 3. Dispatch to worker pool
    // 4. Post-process response
    // 5. Return modified CHAT text
}
```

See `morphosyntax.rs` or `translate.rs` for complete examples.

## Step 5: CLI args

Add to `crates/batchalign-cli/src/args/commands.rs`:

```rust
#[derive(Args, Debug, Clone)]
pub struct YourCommandArgs {
    #[command(flatten)]
    pub common: CommonOpts,

    #[arg(long, default_value = "eng")]
    pub lang: String,

    // ... command-specific flags ...
}
```

Add to the `Commands` enum:

```rust
pub enum Commands {
    // ...
    YourCommand(YourCommandArgs),
}
```

## Step 6: Command profile

In `crates/batchalign-cli/src/args/mod.rs`, add a match arm:

```rust
Commands::YourCommand(a) => CommandProfile {
    command: ReleasedCommand::YourCommand,
    lang: &a.lang,
    num_speakers: 1,
    extensions: &["cha"],
},
```

## Step 7: Typed options

In `crates/batchalign-app/src/types/options.rs`, add:

```rust
pub enum CommandOptions {
    // ...
    YourCommand(YourCommandOptions),
}
```

And in `crates/batchalign-cli/src/args/options.rs`, add the `build_typed_options` arm.

## Step 8: Verify

```bash
make check          # compiles?
make test           # 1,273 tests still pass?
./target/debug/batchalign3 your-command --help   # CLI works?
```

## Python worker side

If your command needs a new ML model:

1. Add an `InferTask` variant in `crates/batchalign-app/src/worker/mod.rs`
2. Add a `WorkerProfile` mapping in `crates/batchalign-app/src/worker/registry.rs`
3. Implement the Python worker handler in `batchalign/worker/`

If reusing an existing model (e.g., Stanza for morphosyntax), you only need
to wire the Rust side — the worker already knows how to handle the infer task.

---

## Worked example: Compare (ReferenceProjection)

Compare is the most instructive example because it uses the `ReferenceProjection`
family — the workflow produces typed intermediate artifacts, then a swappable
`Materializer` turns them into the final output. This is how BA2's
`CompareEngine` + `CompareAnalysisEngine` pair maps to BA3.

### BA2 Python → BA3 Rust mapping

| BA2 Python (`compare.py`) | BA3 Rust | File |
|---------------------------|----------|------|
| `conform()` — contraction/filler normalization | `batchalign_chat_ops::compare::conform` | `batchalign-chat-ops/src/compare.rs` |
| `match_fn()` — word equality with paren stripping | `batchalign_chat_ops::compare::match_fn` | same |
| `_find_best_segment()` — bag-of-words window search | `batchalign_chat_ops::compare::find_best_segment` | same |
| `CompareEngine.process()` — align main vs gold, annotate | `CompareWorkflow::build_artifacts()` | `workflow/compare.rs:157` |
| `CompareAnalysisEngine.analyze()` — metrics CSV | `MainAnnotatedCompareMaterializer::materialize()` | `workflow/compare.rs:82` |
| `Document` / `Utterance` / `Form` model | `ChatFile` via `batchalign_chat_ops::parse` | `batchalign-chat-ops/` |
| `batchalign.utils.dp.align()` — Levenshtein DP | `batchalign_chat_ops::compare::compare()` | same |

### Architecture sketch

```
                    ┌─────────────────────────────────────────────┐
                    │  CompareWorkflow<M: Materializer>           │
                    │                                             │
  CompareWorkflow   │  build_artifacts(request)                   │
  Request {         │    1. morphotag main_text (reuses Stanza)   │
    main_text,      │    2. parse main + gold (parse_lenient)     │
    gold_text,      │    3. compare(&main, &gold) → Bundle        │
    lang,           │    4. return ComparisonArtifacts             │
    services,       │                                             │
    cache_policy,   │  run(request)                               │
    mwt             │    artifacts = build_artifacts(request)      │
  }                 │    materializer.materialize(artifacts)       │
                    └──────────────┬──────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────────────────────┐
                    │         Materializer (swappable)            │
                    │                                             │
                    │  MainAnnotatedCompareMaterializer (released)│
                    │    → annotated CHAT + metrics CSV           │
                    │                                             │
                    │  GoldProjectedCompareSkeletonMaterializer   │
                    │    → gold-shaped CHAT + metrics CSV         │
                    │    (scaffold for Houjun's full projection)  │
                    └─────────────────────────────────────────────┘
```

### Key types

```rust
// Intermediate artifacts — produced by build_artifacts(), consumed by materializer
struct ComparisonArtifacts {
    main_file: ChatFile,        // parsed morphotagged main
    gold_file: ChatFile,        // parsed gold
    bundle: ComparisonBundle,   // alignment + metrics from DP
}

// Released output shape
struct CompareMaterializedOutputs {
    annotated_main_chat: String,  // CHAT with %xsrep annotations
    metrics_csv: String,          // WER, accuracy, per-POS breakdown
}

// Gold projection output (Houjun extends this)
struct GoldProjectedCompareOutputs {
    projected_gold_chat: String,
    metrics_csv: String,
    projection_mode: GoldProjectionMode,  // SkeletalPassthrough → full projection
}
```

### How the BA2 `conform()` + `_find_best_segment()` + DP align maps

BA2's `CompareEngine.process()` does everything in one 250-line method:
extract words → conform → find windows → DP align → annotate gold → set timing.

BA3 splits this into layers:

1. **`batchalign_chat_ops::compare`** — pure functions, no ML, no IO:
   - `conform()` — same normalization rules as BA2
   - `compare(&main, &gold)` → `ComparisonBundle` with alignment + metrics
   - `inject_comparison(&mut gold, &bundle)` — annotate gold file
   - `format_metrics_csv(&metrics)` — CSV output

2. **`workflow/compare.rs`** — orchestration:
   - `build_artifacts()` — morphotag main, parse both, call `compare()`
   - `materialize()` — inject annotations, format outputs

3. **`runner/dispatch/compare_pipeline.rs`** — server integration:
   - Resolves gold file from `*.gold.cha` companion
   - Submits to the workflow
   - Writes output files

### What Houjun changes to extend gold projection

The `GoldProjectedCompareSkeletonMaterializer` is a stub that currently just
passes through the gold file unchanged. To implement real projection:

1. Edit `GoldProjectedCompareSkeletonMaterializer::materialize()` in
   `workflow/compare.rs` — use `ComparisonArtifacts.bundle` to project
   timing, morphology, and dependency from main onto gold tokens.

2. Change `GoldProjectionMode::SkeletalPassthrough` to a new variant
   reflecting the real projection strategy.

3. The workflow, runner, CLI, and registry are already wired — no plumbing
   needed. Just change the materializer.

### Files to read (in order)

1. `crates/batchalign-chat-ops/src/compare.rs` (711 lines) — pure compare logic
2. `crates/batchalign-app/src/workflow/compare.rs` (213 lines) — workflow + materializers
3. `crates/batchalign-app/src/runner/dispatch/compare_pipeline.rs` (332 lines) — server dispatch
4. `crates/batchalign-app/src/compare.rs` (149 lines) — shared helpers
5. BA2 reference: `~/batchalign2-master/batchalign/pipelines/analysis/compare.py` (470 lines)
