# Rust Workspace Map

**Status:** Current
**Last modified:** 2026-03-21 15:30 EDT

Batchalign3 currently has three active Rust surfaces plus one repo-local
automation surface:

## 1. Root Cargo Workspace

The main Rust control plane lives at the repository root in [Cargo.toml](../../../Cargo.toml).

Active crates:

| Crate | Responsibility |
| --- | --- |
| `batchalign-chat-ops` | shared CHAT extraction, injection, mapping, validation-adjacent operations |
| `batchalign-app` | server routes, job lifecycle, worker orchestration, OpenAPI, persistence |
| `batchalign-cli` | `batchalign3` binary, argument parsing, dispatch, daemon/log/cache commands |
| `batchalign-types` | shared domain, protocol, and scheduling types |
| `xtask` | repo-local automation and affected-check orchestration |
| `batchalign-app/src/workflow/` | workflow-family implementations for `transcribe`, `align`, `morphotag`, `compare`, `benchmark` |

Typical commands:

```bash
cargo build -p batchalign-cli
make build-rust
cargo check --workspace
cargo nextest run --workspace
cargo nextest run -p batchalign-cli --test cli
cargo nextest run -p batchalign-app --test integration
cargo xtask affected-rust packages
```

## 2. PyO3 Bridge

The Python extension lives under `pyo3/`.

| Crate | Responsibility |
| --- | --- |
| `batchalign-pyo3` | builds the `batchalign_core` extension module used by Python |

Typical commands:

```bash
make build-python
cargo nextest run --manifest-path pyo3/Cargo.toml
```

## 3. Repo Automation

`xtask/` is the home for repo-local developer automation:

- affected-check selection
- local verification presets
- install/build smoke tests
- repository policy checks that need typed access to workspace metadata

Use `cargo xtask ...` when a task understands the repo graph better than a
shell script does. Keep shell wrappers thin.

## Ownership Boundary

- use the root workspace when the change affects CLI behavior, server APIs, job execution, logs, cache handling, or worker orchestration
- use `crates/batchalign-app/src/workflow/` when the change affects command semantics, workflow composition, or output materialization
- use `pyo3/` when the change affects the Python extension surface exposed as `batchalign_core`
- remember that `batchalign-chat-ops` is shared by both surfaces: rebuild
  `batchalign_core` with `make build-python`, and if you plan to run the
  standalone CLI directly after a shared crate change, rebuild the CLI too or
  use `cargo run -p batchalign-cli -- ...`
- use `talkbank-tools` for parser/model/validator behavior that Batchalign consumes as a dependency

## First Files to Read

1. `crates/batchalign-cli/src/lib.rs` — `run_command()`, the single canonical command router
2. `crates/batchalign-cli/src/dispatch/` — CLI dispatch (explicit server vs. auto-daemon)
3. `crates/batchalign-app/src/runner/` — server-side task routing and dispatch shapes
4. `crates/batchalign-app/src/routes/` — Axum HTTP routes
5. `crates/batchalign-app/src/worker/` — worker pool and IPC
6. `crates/batchalign-app/src/workflow/` — workflow-family implementations and typed intermediate artifacts
7. `pyo3/src/lib.rs` — PyO3 module organization and entry points

See also: [Rust CLI and Server](rust-cli-and-server.md) for detailed dispatch
documentation and the checklist for adding new commands.
