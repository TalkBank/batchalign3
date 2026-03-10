# Rust CLI and Server

**Status:** Current
**Last updated:** 2026-03-14

This page covers the Rust control plane that powers `batchalign3`: the CLI
client, the HTTP server, and how to extend them.

The current worker-boundary replacement plan is documented separately in
[Worker Protocol V2](worker-protocol-v2.md). That spec is the source of truth
for replacing the legacy stdio JSON-lines worker contract.

## Crate Map

| Crate | Role |
|-------|------|
| `crates/batchalign-cli` | Clap CLI, dispatch router, daemon lifecycle, output writing |
| `crates/batchalign-app` | Axum HTTP server, job store, worker pool, per-command orchestrators, cache |
| `crates/batchalign-chat-ops` | CHAT extraction, injection, validation, ASR post-processing, DP alignment |
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
**single canonical command router**. Both the standalone binary (`main.rs`)
and the PyO3 console_scripts entry point (`pyo3/src/cli_entry.rs`) call it.

```
main.rs          → batchalign_cli::run_command(cli)
cli_entry.rs     → batchalign_cli::run_command(cli)
```

`main.rs` and `cli_entry.rs` are thin wrappers: tracing setup +
`run_command()` call. No command-specific logic lives in either file.

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

**`crates/batchalign-app/src/runner/mod.rs`**:
- `infer_task_for_command()` — Map `"foo"` → `InferTask::Foo`
- `command_requires_infer()` — Whether the command must use server-side
  orchestration (text-only commands: yes; audio commands: depends)

If the command uses the infer path, also add it to `INFER_PATH_COMMANDS` in
`crates/batchalign-app/src/lib.rs`. The server's capability gate cross-checks
this list against the probe worker's advertised `infer_tasks` — commands not in
this list are assumed to not need an infer task and pass through unconditionally.

On the Python side, you must also add the `InferTask` to `_INFER_TASK_PROBES` in
`batchalign/worker/_handlers.py`. See
[Adding Inference Providers](../developer/adding-engines.md#4-wire-dispatch-and-capability-advertisement)
for details.

### 5. Server-side dispatch shape

**`crates/batchalign-app/src/runner/dispatch/infer.rs`** — Route the command
to its orchestrator in the appropriate dispatch shape:
- `dispatch_batched_infer()` for text-only commands (cross-file batching)
- `dispatch_fa_infer()` for per-file audio commands
- `dispatch_transcribe_infer()` for audio-to-CHAT generation

### 6. Orchestrator module

**`crates/batchalign-app/src/foo.rs`** — The per-command orchestrator that
owns the CHAT lifecycle: parse → cache check → typed worker IPC
(`execute_v2` on the live infer surface) → inject → validate → serialize.

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
(for OpenAPI types).
