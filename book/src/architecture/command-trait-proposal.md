# ADR: Command Trait for Full-Lifecycle Command Definition

**Status:** Draft — awaiting validation from Houjun's compare implementation
**Last updated:** 2026-03-21 17:04 EDT

## Problem

Adding a new batchalign3 command requires touching **11 files** across 5 crates,
with no compile-time guarantee that all registration points are consistent:

| # | File | What you add |
|---|------|-------------|
| 1 | `batchalign-types/src/domain.rs` | `ReleasedCommand::Foo` variant |
| 2 | `batchalign-types/src/worker.rs` | `InferTask::Foo` variant (if new task) |
| 3 | `batchalign-app/src/workflow/registry.rs` | `CommandWorkflowDescriptor` entry |
| 4 | `batchalign-app/src/types/options.rs` | `CommandOptions::Foo(FooOptions)` variant + struct |
| 5 | `batchalign-app/src/runner/dispatch/*.rs` | Dispatch match arm |
| 6 | `batchalign-app/src/runner/policy.rs` | `infer_task_for_command()` (derived from registry) |
| 7 | `batchalign-app/src/foo.rs` | Core implementation module |
| 8 | `batchalign-app/src/workflow/foo.rs` | Workflow trait impl |
| 9 | `batchalign-cli/src/args/commands.rs` | `FooArgs` struct + `Commands::Foo` variant |
| 10 | `batchalign-cli/src/args/mod.rs` | `CommandProfile` match arm |
| 11 | `batchalign-cli/src/args/options.rs` | `build_typed_options()` match arm |

If any of these are missed or inconsistent, the error surfaces at runtime (wrong
dispatch, missing capability, silent no-op) rather than at compile time.

### Reference: talkbank-tools

talkbank-tools solves a similar problem with the `AnalysisCommand` trait
(`talkbank-clan/src/framework/command.rs`). Each command implements a 3-phase
lifecycle (process_utterance → end_file → finalize) with typed Config, State,
and Output. Adding a command touches ~5 files. But that trait is synchronous
and single-threaded — batchalign3 commands are async with worker dispatch,
caching, and multiple workflow families.

## Proposed Design

### The Command trait

```rust
use async_trait::async_trait;

/// Full-lifecycle command definition.
///
/// Implementing this trait for a new command should be the ONLY thing
/// needed beyond the implementation module itself. Registration, dispatch,
/// CLI args, and workflow selection are all derived from the trait's
/// associated constants and types.
#[async_trait]
pub trait Command: Send + Sync + 'static {
    // -- Identity (compile-time constants) --

    /// Stable command identity.
    const IDENTITY: ReleasedCommand;

    /// Which ML inference task this command needs from workers.
    /// Commands that don't need inference use a shared task
    /// (e.g., compare reuses `InferTask::Morphosyntax`).
    const INFER_TASK: InferTask;

    /// File extensions this command discovers.
    const EXTENSIONS: &'static [&'static str];

    /// Whether the CLI needs local audio file access
    /// (true for transcribe, benchmark, avqi; false for morphotag, compare).
    const USES_LOCAL_AUDIO: bool;

    // -- Associated types --

    /// Typed options parsed from CLI args and carried through the job.
    type Options: Serialize + DeserializeOwned + Send + Sync;

    /// Typed output produced by one execution.
    type Output: Send;

    // -- Lifecycle methods --

    /// Execute the command for one job submission.
    ///
    /// Receives parsed options, discovered input files, and shared
    /// infrastructure (worker pool, cache, config). Returns typed output.
    async fn execute(
        &self,
        options: &Self::Options,
        files: Vec<InputFile>,
        services: &CommandServices,
    ) -> Result<Self::Output, ServerError>;

    /// Materialize typed output into user-facing files.
    ///
    /// Most commands produce a single CHAT file per input. Compare
    /// produces both annotated CHAT and a metrics CSV sidecar.
    fn materialize(
        &self,
        output: Self::Output,
    ) -> Result<Vec<MaterializedFile>, ServerError>;
}
```

### Supporting types

```rust
/// A discovered input file with metadata.
pub struct InputFile {
    pub path: PathBuf,
    pub filename: FileName,
    /// For compare: the paired gold file, if found.
    pub companion: Option<PathBuf>,
}

/// A materialized output file ready to write.
pub struct MaterializedFile {
    pub filename: String,
    pub content: String,
    pub content_type: ContentType,
}

/// Shared services available to all commands during execution.
pub struct CommandServices {
    pub pool: Arc<WorkerPool>,
    pub cache: Arc<UtteranceCache>,
    pub store: Arc<JobStore>,
    pub config: Arc<ServerConfig>,
    pub engine_versions: Arc<BTreeMap<String, String>>,
    pub progress: Option<ProgressSender>,
}
```

### How current registration points map to the trait

| Current (11 files) | With Command trait |
|---------------------|-------------------|
| `ReleasedCommand::Foo` variant | `const IDENTITY` |
| `CommandWorkflowDescriptor` entry | **Derived** from consts |
| `CommandOptions::Foo(FooOptions)` variant | `type Options` |
| `build_typed_options()` match arm | CLI adapter (still needed, but 1 file) |
| `run_command()` match arm | **Generated** or registry-driven |
| `dispatch_batched_infer()` match arm | **Eliminated** — `execute()` IS the dispatch |
| `infer_task_for_command()` lookup | `const INFER_TASK` |
| `command_requires_infer()` lookup | `INFER_TASK != InferTask::None` |
| `CommandProfile` match arm | **Derived** from consts |
| `InferTask::Foo` variant | Shared across commands (many-to-one, unchanged) |
| Implementation module | `execute()` method body |

**Result:** Adding a command touches **3 files:**
1. The command module (trait impl + options struct + core logic)
2. `ReleasedCommand` enum (add variant)
3. CLI args (add clap struct + `build_typed_options` arm)

The workflow descriptor, dispatch routing, capability queries, and profile
metadata are all derived from the trait's constants.

## How compare maps to this

Compare is currently a `ReferenceProjectionWorkflow` with a swappable
`Materializer`. With the Command trait:

```rust
pub struct CompareCommand;

#[async_trait]
impl Command for CompareCommand {
    const IDENTITY: ReleasedCommand = ReleasedCommand::Compare;
    const INFER_TASK: InferTask = InferTask::Morphosyntax;
    const EXTENSIONS: &'static [&'static str] = &["cha"];
    const USES_LOCAL_AUDIO: bool = false;

    type Options = CompareOptions;
    type Output = CompareMaterializedOutputs;

    async fn execute(
        &self,
        options: &Self::Options,
        files: Vec<InputFile>,
        services: &CommandServices,
    ) -> Result<Self::Output, ServerError> {
        // For each file:
        //   1. Find gold companion (file.companion or *.gold.cha convention)
        //   2. Morphotag the main text via services.pool
        //   3. Parse both, run compare() DP alignment
        //   4. Inject annotations, format metrics
        // Return CompareMaterializedOutputs { annotated_main_chat, metrics_csv }
    }

    fn materialize(
        &self,
        output: Self::Output,
    ) -> Result<Vec<MaterializedFile>, ServerError> {
        Ok(vec![
            MaterializedFile {
                filename: "output.cha".into(),
                content: output.annotated_main_chat,
                content_type: ContentType::Chat,
            },
            MaterializedFile {
                filename: "output.compare.csv".into(),
                content: output.metrics_csv,
                content_type: ContentType::Csv,
            },
        ])
    }
}
```

The existing `CompareWorkflow` and `MainAnnotatedCompareMaterializer` become
implementation details inside `execute()` — the trait provides the public
contract.

## EngineBackend integration

Commands that support engine selection (transcribe, align) can query the
`EngineBackend` trait on their engine types:

```rust
async fn execute(&self, options: &TranscribeOptions, ...) -> Result<...> {
    let backend = options.asr_engine; // AsrEngineName — closed enum
    if backend.is_rust_owned() {
        // Rev.AI path (Rust-owned)
    } else {
        // Worker path (Python-hosted)
    }
}
```

No string matching. The `EngineBackend` trait ensures all engines implement
`wire_name()` for CLI serialization and `is_rust_owned()` for dispatch routing.

## Migration path

The trait is **additive** — it does not require rewriting existing commands.

1. **Phase 0 (now):** This design doc. Houjun validates against compare.
2. **Phase 1:** Define the trait and `CommandServices` in `batchalign-app`.
3. **Phase 2:** Implement `Command` for one simple command (e.g., morphotag)
   alongside the existing dispatch. Both paths work.
4. **Phase 3:** Implement for compare (Houjun's first real use case).
5. **Phase 4:** Migrate remaining commands one at a time.
6. **Phase 5:** Remove legacy dispatch match arms once all commands are migrated.

At each phase, both old and new commands coexist. The runner checks the trait
registry first, falls back to the legacy descriptor registry.

## Open questions for Houjun

1. **Input shape:** Does `Vec<InputFile>` with an optional `companion` field
   work for compare's gold-file pairing? Or does compare need a fundamentally
   different input type (e.g., `Vec<(InputFile, GoldFile)>`)?

2. **Materialization seam:** Is `materialize()` returning `Vec<MaterializedFile>`
   the right abstraction for multi-output commands? Should it receive the
   original input files for filename derivation?

3. **Progress reporting:** Should `CommandServices` include the `ProgressSender`
   for TUI updates, or should progress be a separate concern (e.g., a
   `ProgressReporter` trait that `execute()` receives)?

4. **Incremental processing:** The `--before` flag enables diff-based processing
   (only reprocess changed utterances). How should this fit into the trait?
   Options: (a) field on `InputFile`, (b) separate trait method, (c) handled
   entirely inside `execute()`.

5. **Batch vs per-file:** Some commands (morphotag, utseg) batch all files into
   one worker call for GPU efficiency. Others (transcribe, align) process
   per-file. Should the trait distinguish these, or should `execute()` handle
   both patterns internally?

## What this does NOT change

- The worker protocol (Python IPC) stays the same
- The HTTP API (`/jobs`) stays the same
- The CLI clap arg parsing still needs per-command arg structs
- The `EngineOverrides` / pool keying stays as-is
- The runner's job lifecycle (queue, semaphore, retry) stays as-is
- The memory guard and test safety tiers stay as-is
