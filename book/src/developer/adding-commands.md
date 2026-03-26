# Adding a New Command

**Status:** Current
**Last updated:** 2026-03-26 14:05 EDT

This guide walks through adding a new batchalign3 command end-to-end.

**Reference implementations by workflow family:**

| Family | Best example | Files to read |
|--------|-------------|---------------|
| `PerFileTransform` (one file → one output) | **`align`** | `commands/align.rs`, `runner/dispatch/fa_pipeline.rs` |
| `CrossFileBatchTransform` (batch-infer pool) | `morphotag` | `commands/morphotag.rs`, `runner/dispatch/infer_batched.rs` |
| `ReferenceProjection` (compare against gold) | `compare` | `commands/compare.rs`, `compare.rs` |
| `Composite` (orchestrates sub-workflows) | `benchmark` | `commands/benchmark.rs`, `runner/dispatch/benchmark_pipeline.rs` |

**Start with `align`** — it is still the simplest command-owned example. If
your command takes CHAT text in and produces modified CHAT out, follow `align`.
If your command needs to batch multiple files through one ML call, follow
`morphotag`.

## Quick start

```bash
make check    # after each file edit (~6s)
make test     # verify nothing broke (~6s)
```

## Architecture overview

Every command flows through these layers:

```
CLI args → CommandOptions → JobSubmission → Runner → commands::<name>::build_plan()
        → shared kernel / worker pool → output materialization
```

The key files, in the order you'll edit them:

| Step | File | What you add |
|------|------|-------------|
| 1 | `batchalign-types/src/domain.rs` | `ReleasedCommand::YourCommand` variant |
| 2 | `batchalign-app/src/commands/your_command.rs` | `CommandModuleSpec` + command-owned wrapper |
| 3 | `batchalign-app/src/commands/catalog.rs` and `commands/mod.rs` | Register/export the module |
| 4 | `batchalign-app/src/your_command.rs` or shared runner code | Core logic (ML dispatch, post-processing) |
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

## Step 2: Add the command-owned spec

```rust
// crates/batchalign-app/src/commands/your_command.rs
pub(crate) const YOUR_COMMAND_SPEC: CommandModuleSpec = CommandModuleSpec {
    descriptor: CommandWorkflowDescriptor {
        command: ReleasedCommand::YourCommand,
        family: WorkflowFamily::CrossFileBatchTransform,  // pick one of 4 families
        infer_task: InferTask::YourTask,                  // or reuse existing
        capability_kind: CommandCapabilityKind::DirectInfer,
        uses_local_audio: false,
        output_path_kind: CommandOutputPathKind::PreserveInputName,
        runner_dispatch_kind: RunnerDispatchKind::BatchedTextInfer,
    },
    performance: CommandPerformanceProfile {
        // Copy the nearest existing command, then narrow it deliberately.
        ..MORPHOTAG_SPEC.performance
    },
};
```

Then export the module from `commands/mod.rs` and add the spec to
`commands/catalog.rs`. Do **not** create a second registry layer; the
command-owned catalog is the source of truth for released command metadata.

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

## Step 3: Implement the command-owned wrapper

Create `crates/batchalign-app/src/commands/your_command.rs`.

**For a command that reuses an existing runner family**:

```rust
// commands/your_command.rs
use std::sync::Arc;

use crate::config::ServerConfig;
use crate::pipeline::PipelineServices;
use crate::runner::{dispatch_batched_infer, BatchedInferDispatchPlan, RunnerJobSnapshot};
use crate::store::JobStore;

pub(crate) fn build_plan(
    job: &RunnerJobSnapshot,
    config: &ServerConfig,
) -> Option<BatchedInferDispatchPlan> {
    BatchedInferDispatchPlan::from_job(job, config)
}

pub(crate) async fn run(
    job: &RunnerJobSnapshot,
    store: &Arc<JobStore>,
    services: PipelineServices<'_>,
) {
    let Some(plan) = build_plan(job, store.config()) else {
        return;
    };
    dispatch_batched_infer(job, store, services, plan).await;
}
```

Only touch `runner/dispatch/*` when your command shape is genuinely new. Most
new commands should reuse an existing dispatch family and keep the obvious
top-level ownership in `src/commands/your_command.rs`.

**If you truly need a new family**, add or extend the shared runner kernel and
keep the command module as the contributor-facing entrypoint:

```rust
use crate::pipeline::PipelineServices;
pub(crate) async fn run(
    job: &RunnerJobSnapshot,
    store: &Arc<JobStore>,
    services: PipelineServices<'_>,
) {
    let Some(plan) = build_plan(job, store.config()) else {
        return;
    };
    dispatch_your_family(job, store, services, plan).await;
}
```

See `commands/align.rs`, `commands/morphotag.rs`, and `commands/benchmark.rs`
for the current real examples.

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
`CompareEngine` + `CompareAnalysisEngine` pair maps to BA3 without falling back
to string-level projection or ad hoc string assembly at the serialization
boundary.

### BA2 Python → BA3 Rust mapping

| BA2 Python (`compare.py`) | BA3 Rust | File |
|---------------------------|----------|------|
| `_find_best_segment()` — bag-of-words window search | `batchalign_chat_ops::compare::find_best_segment` | same |
| `CompareEngine.process()` — local window alignment + token status | `batchalign_chat_ops::compare::compare()` | `batchalign-chat-ops/src/compare.rs` |
| `CompareAnalysisEngine.analyze()` — metrics CSV | `CompareMetricsCsvTable` / `format_metrics_csv()` via compare materializers | `batchalign-chat-ops/src/compare.rs` / `compare.rs` |
| gold document projection | `project_gold_structurally()` | `batchalign-chat-ops/src/compare.rs` |
| `Document` / `Utterance` / `Form` model | `ChatFile` AST + dependent tiers | `talkbank-model` / `batchalign-chat-ops` |
| CLI dispatch `morphosyntax -> compare -> compare_analysis` | `build_comparison_artifacts()` + released/main-annotated materializers | `compare.rs` |

### Architecture sketch

```mermaid
flowchart TD
    request["compare.rs orchestration\nmain_text + gold_text"] --> morph["Morphotag main only\nreuses morphosyntax worker"]
    request --> gold["Parse raw gold"]
    morph --> main["Parse morphotagged main"]
    main --> bundle["compare(&main, &gold)\nComparisonBundle:\nmain_utterances + gold_utterances\n+ gold_word_matches + metrics"]
    gold --> bundle
    bundle --> tiers["XsrepTierContent / XsmorTierContent"]
    bundle --> csv["CompareMetricsCsvTable"]
    tiers --> released["materialize_released()\nreleased output:\nprojected reference CHAT + .compare.csv"]
    tiers --> main_view["materialize_main_annotated()\ninternal/benchmark output:\nmain %xsrep/%xsmor + .compare.csv"]
    csv --> released
    csv --> main_view
    released --> safety["exact match -> copy %mor/%gra/%wor\nfull gold coverage -> %mor only\nelse keep gold tiers"]
```

### Key types

```rust
// Intermediate artifacts — produced by build_comparison_artifacts(), consumed by materializer
struct ComparisonArtifacts {
    main_file: ChatFile,        // parsed morphotagged main
    gold_file: ChatFile,        // parsed gold
    bundle: ComparisonBundle,   // alignment + metrics from DP
}

struct ComparisonBundle {
    main_utterances: Vec<UtteranceComparison>,
    gold_utterances: Vec<UtteranceComparison>,
    gold_word_matches: Vec<GoldWordMatch>,
    metrics: CompareMetrics,
}

struct XsrepTierContent {
    items: Vec<CompareTierItem<CompareSurfaceToken>>,
}

struct XsmorTierContent {
    items: Vec<CompareTierItem<ComparePosLabel>>,
}

struct CompareMetricsCsvTable {
    rows: Vec<CompareMetricsCsvRow>,
}

struct CompareMaterializedOutputs {
    chat_output: String,
    metrics_csv: String,
}

struct MainAnnotatedCompareOutputs {
    annotated_main_chat: String,
    metrics_csv: String,
}
```

### How the BA2 `_find_best_segment()` + local DP maps

BA2's `CompareEngine.process()` does everything in one 250-line method:
extract words → conform → find windows → DP align → annotate gold → set timing.

BA3 splits this into layers:

1. **`batchalign_chat_ops::compare`** — pure functions, no ML, no IO:
   - `find_best_segment()` — same local-window idea as BA2
   - `compare(&main, &gold)` → `ComparisonBundle` with main/gold compare views,
     structural word matches, and metrics
   - `project_gold_structurally()` — AST-first gold projection
   - `XsrepTierContent` / `XsmorTierContent` — typed compare-tier models lowered
     once at the `UserDefinedDependentTier` boundary
   - `CompareMetricsCsvTable` / `format_metrics_csv()` — typed metrics rows
     serialized through the Rust `csv` crate

2. **`compare.rs`** — orchestration:
    - `build_comparison_artifacts()` — morphotag main only, parse gold raw, call `compare()`
    - `materialize_released()` — released compare output path
    - `materialize_main_annotated()` — internal benchmark/main output path

3. **`runner/dispatch/compare_pipeline.rs`** — server integration:
    - Resolves gold file from `*.gold.cha` companion
    - Calls the compare orchestrator
    - Writes output files

### How to extend structural gold projection

The gold materializer is no longer a stub. Extend it by working with typed data:

1. Edit `project_gold_structurally()` in
   `batchalign-chat-ops/src/compare.rs`.
2. Use `ComparisonBundle.gold_word_matches` and AST accessors, not `%xsrep` /
   `%xsmor` strings, as the projection source.
3. Keep the current safety rules explicit: exact matches may copy `%mor` /
   `%gra` / `%wor`; full gold-word coverage may project `%mor`; partial `%gra` /
   `%wor` needs chunk-safe mapping before it is allowed.
4. Keep gold raw during artifact construction unless the reference file already
   contains tiers you are intentionally preserving.

### Serialization rule

When a workflow emits structured artifacts, add explicit pre-serialization
types before you add serializer code.

- New semantic strings must get newtypes.
- CHAT tier content should be written from typed models via `WriteChat`.
- CSV outputs should be written from typed row/table models via `csv`.
- Do not drive semantics from `format!`, `join`, `split`, or regex surgery over
  already serialized output.

### Files to read (in order)

1. `crates/batchalign-chat-ops/src/compare.rs` — compare core + structural projection
2. `crates/batchalign-app/src/compare.rs` — orchestration + materializers
3. `crates/batchalign-app/src/runner/dispatch/compare_pipeline.rs` — server dispatch
4. `book/src/migration/ba2-compare-migration.md` — BA2-master compare to BA3 map
5. BA2 reference: `~/batchalign2-master/batchalign/pipelines/analysis/compare.py`
