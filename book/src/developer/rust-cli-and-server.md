# Rust CLI and Server

**Status:** Current
**Last modified:** 2026-03-21 15:30 EDT

This page covers the Rust control plane that powers `batchalign3`: the CLI
client, the HTTP server, and how to extend them.

The current worker-boundary replacement plan is documented separately in
[Worker Protocol V2](worker-protocol-v2.md). That spec is the source of truth
for replacing the legacy stdio JSON-lines worker contract.

## Crate Map

| Crate | Role |
|-------|------|
| `crates/batchalign-cli` | Clap CLI, dispatch router, daemon lifecycle, output writing |
| `crates/batchalign-app` | Axum HTTP server, job store, worker pool, cache, workflow orchestration |
| `crates/batchalign-chat-ops` | CHAT extraction, injection, validation, ASR post-processing, DP alignment |
| `crates/batchalign-app/src/workflow/` | typed workflow families, bundles, and materializers |
| `pyo3/` | PyO3 bridge (`batchalign_core`) — separate single-crate project, not in the workspace |

## Common Developer Commands

```bash
cargo check --workspace
cargo nextest run --workspace
cargo check --manifest-path pyo3/Cargo.toml    # PyO3 crate (separate)
cargo nextest run --manifest-path pyo3/Cargo.toml
```

## CLI Command Dispatch (Single Source of Truth)

`batchalign_cli::run_command()` in `crates/batchalign-cli/src/lib.rs` is the
**single canonical command router**. The standalone binary (`main.rs`) calls it.
The installed `batchalign3` console command is a tiny Python wrapper
(`batchalign/_cli.py`) that finds and execs the standalone binary — either
packaged in the wheel at `batchalign/_bin/batchalign3`, or from
`target/debug/batchalign3` in a source checkout.

```
main.rs            → batchalign_cli::run_command(cli)
batchalign/_cli.py → os.execv(batchalign/_bin/batchalign3)  [installed]
                   → os.execv(target/debug/batchalign3)      [dev checkout]
```

`main.rs` and `batchalign/_cli.py` are thin wrappers.
No command-specific logic lives in either of them.

The CLI layer now exposes two contributor-facing named seams:

- `ReleasedCommand` in `crates/batchalign-types/src/domain.rs` is the closed
  released command vocabulary for contributor-facing Rust code. Parse external
  strings into this enum as early as possible; keep the old string-backed
  `CommandName` only at wire/storage boundaries.
- `CommandProfile` in `crates/batchalign-cli/src/args/mod.rs` keeps the
  command identity, language, file extensions, and speaker count together as a
  typed profile instead of a positional tuple.
- `DispatchRequest` in `crates/batchalign-cli/src/dispatch/mod.rs` carries the
  typed command profile, I/O settings, and runtime flags into the dispatcher as
  one named boundary object.

The dispatcher also consults
`batchalign_app::released_command_uses_local_audio()` and the shared workflow
registry to decide whether a requested command can stay on an explicit
`--server` path or must fall back to a local daemon because it needs
client-side audio files.

When the CLI is polling or writing file results, `FileErrorDetail` in
`crates/batchalign-cli/src/dispatch/helpers.rs` keeps file-scoped failures as a
named record instead of spreading filename/message pairs through the progress
code.

The command-specific logic now starts in
`crates/batchalign-app/src/workflow/`. That layer owns the typed workflow
families and their intermediate artifacts. `crates/batchalign-app/src/runner/`
should stay focused on job lifecycle, queueing, and policy.

One dependency-graph cleanup already landed here: the standalone binary's OTLP
telemetry stack and update-check helper are now gated behind the
`batchalign-cli` crate's `binary-entry` feature. The PyO3 `cli_entry` path
still shares `run_command()`, but it no longer drags those binary-only
dependencies into the extension build.
The embedded CLI bootstrap path now also lives in `batchalign-cli`
(`run_embedded_cli_from_argv()`), so `pyo3` no longer owns its own `clap`
parsing or Tokio runtime setup.

For day-to-day command work, prefer the workflow layer first:

1. decide the workflow family
2. add the typed bundle/materializer in `crates/batchalign-app/src/workflow/`
3. keep the CLI argument plumbing thin
4. let runner/dispatch handle lifecycle, not semantics

## Adding a New CLI Command

When adding a new processing command (e.g., `batchalign3 foo`), these files
must be updated:

### 1. CLI argument definition

**`crates/batchalign-cli/src/args/mod.rs`** — Add `Commands::Foo(FooArgs)`
variant to the `Commands` enum.

**`crates/batchalign-cli/src/args/commands.rs`** — Define `FooArgs` struct
with clap attributes. Include `CommonOpts` if the command processes files.

### 2. CLI dispatch

**`crates/batchalign-cli/src/lib.rs`** — Add the match arm in
`run_command()`. For processing commands, this typically falls through to
the `cmd =>` wildcard arm that calls `dispatch::dispatch()`. For utility
commands (like `serve`, `jobs`, `models`), add an explicit arm.

### 3. Typed command options

**`crates/batchalign-app/src/types/options.rs`** — Add
`CommandOptions::Foo { ... }` variant to the serde-tagged enum. This is the
wire format between CLI and server.

**`crates/batchalign-cli/src/args/options.rs`** — Add the builder in
`build_typed_options()` that converts `FooArgs` → `CommandOptions::Foo`.

### 4. Server-side task routing and capability gate

**`crates/batchalign-app/src/workflow/registry.rs`** — Register a
`WorkflowDescriptor` for the new command. This drives both
`infer_task_for_command()` and `command_requires_infer()` in
`crates/batchalign-app/src/runner/policy.rs`.

The server's capability gate (`validate_infer_capability_gate()` in
`crates/batchalign-app/src/state.rs`) cross-checks the probe worker's
advertised `infer_tasks` against the workflow registry — commands whose
workflow descriptor requires an infer task must have a matching worker
capability.

On the Python side, you must also add the `InferTask` to `_INFER_TASK_PROBES` in
`batchalign/worker/_handlers.py`. See
[Adding Inference Providers](../developer/adding-engines.md#4-wire-dispatch-and-capability-advertisement)
for details.

### 5. Server-side dispatch shape

Route the command to its orchestrator in the appropriate dispatch module under
`crates/batchalign-app/src/runner/dispatch/`:
- `infer_batched.rs` — `dispatch_batched_infer()` for text-only commands (cross-file batching)
- `fa_pipeline.rs` — `dispatch_fa_infer()` for per-file forced alignment
- `transcribe_pipeline.rs` — `dispatch_transcribe_infer()` for audio-to-CHAT generation
- `benchmark_pipeline.rs` — `dispatch_benchmark_infer()` for transcribe + compare composition
- `compare_pipeline.rs` — `dispatch_reference_projection()` for gold-anchored comparison
- `media_analysis_v2.rs` — `dispatch_media_analysis_v2()` for opensmile/avqi

### 6. Orchestrator module

**`crates/batchalign-app/src/workflow/foo.rs`** — The typed workflow
implementation that owns the command's semantic shape, typed bundles, and
materialization strategy.

**`crates/batchalign-app/src/foo.rs`** — If a compatibility orchestrator is
still needed, keep it thin and delegate into `workflow/foo.rs`. The old
"all logic lives in the orchestrator" model is no longer the preferred one.

For batch text workflows, prefer the named wrappers in
`crates/batchalign-app/src/workflow/text_batch.rs` over raw tuples:

- `TextBatchFileInput` keeps one file name and one owned CHAT payload together.
- `TextBatchFileResults` keeps the per-file outcome shape explicit.
- `TextWorkflowFileError` keeps file-scoped failure details separate from file
  identity instead of returning `String` error messages.

### 7. Worker support

**`batchalign/worker/_model_loading/`** — Register the dynamic batch-infer
handler for `InferTask.FOO` during worker bootstrap if the task needs loaded
runtime state or engine-specific wiring.

**`batchalign/worker/_infer.py`** — Only update this file if the task is a
pure static route that does not need bootstrap-installed runtime wiring.

**`batchalign/inference/foo.py`** — The Python inference module (pure model
invocation, no CHAT awareness).

### 8. CHAT operations (if needed)

**`crates/batchalign-chat-ops/src/foo.rs`** — Payload collection, cache key
computation, result injection functions used by the orchestrator.

## OpenAPI Workflow

```bash
# Generate OpenAPI schema
cargo run -q -p batchalign-cli -- openapi --output openapi.json

# Verify schema is up to date (CI gate)
cargo run -q -p batchalign-cli -- openapi --check --output openapi.json
```

## Relationship to the PyO3 Layer

The CLI/server workspace and the PyO3 extension are separate build targets:

- **Root workspace** (`crates/`): operational control plane (CLI + server)
- **`pyo3/`**: Python extension module (`batchalign_core`)

Both share CHAT operations through `batchalign-chat-ops`. The PyO3 crate
also depends on `batchalign-cli` (for `run_command()`) and `batchalign-app`
(for OpenAPI types), but it now does so with `default-features = false` so the
extension path does not compile the standalone binary's OTLP stack.

See [Building & Development](building.md) for the recommended fast local loop
(`make build-python`, then one `cargo build -p batchalign-cli` if you want the
source-checkout fallback to use the repo binary).
