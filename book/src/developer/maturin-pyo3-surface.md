# Maturin Build and PyO3 Dependency Surface

**Status:** Current
**Last updated:** 2026-03-20

## Overview

The `batchalign_core` Python extension is built by **maturin** from `pyo3/Cargo.toml`.
This page documents the Rust crate dependencies that maturin must compile, the
rationale for each, and the roadmap for shrinking the build surface further.

## Current Dependency Graph (post-extraction)

```
batchalign-pyo3 (the .so)
  |
  +-- batchalign-types          (newtypes, worker IPC types)
  +-- batchalign-chat-ops       (CHAT parse/inject/align/postprocess)
  +-- batchalign-revai          (Rev.AI transcript projection)
  +-- batchalign-cli            (console_scripts entry point)
  |     +-- batchalign-app      (HTTP server, SQLite, dashboard, worker pool)
  |
  +-- talkbank-model            (CHAT AST, model types)
  +-- talkbank-parser           (tree-sitter parser)
  +-- talkbank-transform        (AST transforms)
```

### Why each dependency exists

| Crate | Used by pyo3 for | Could be removed? |
|-------|-------------------|-------------------|
| `batchalign-types` | Domain newtypes (`DurationMs`, `LanguageCode3`), worker IPC types (`ExecuteRequestV2`, etc.) | No - these are the core shared types |
| `batchalign-chat-ops` | CHAT parsing, morphosyntax injection, FA injection, DP alignment, retokenization, ASR postprocessing | No - this is the main Rust logic |
| `batchalign-revai` | Rev.AI transcript projection to CHAT AST | No - needed by pyfunction wrappers |
| `batchalign-cli` | `run_command()` + `Cli` struct for `uv tool install batchalign3` console_scripts | **Yes** - see roadmap below |
| `batchalign-app` | **Not directly** - pulled transitively by `batchalign-cli` | **Yes** - see roadmap below |
| `talkbank-*` | CHAT AST types, parser, transforms | No - fundamental |

### What was removed (2026-03-20)

Before the `batchalign-types` extraction, pyo3 directly depended on
`batchalign-app` for domain newtypes and worker IPC types. This pulled in
axum, SQLite, moka, tower-http, dashmap, and the entire server stack into every
`uv run maturin develop` cold build.

Now pyo3 imports these types from `batchalign-types` (3 deps: serde, utoipa,
schemars). The `batchalign-app` dependency remains only as a transitive
dependency through `batchalign-cli`.

## Build commands

```bash
# Development rebuild (debug, fast iteration)
uv run maturin develop -m pyo3/Cargo.toml

# Release wheel (for distribution)
uv run maturin build --release -m pyo3/Cargo.toml -i python3.12

# Check compilation without building wheel
cargo check --manifest-path pyo3/Cargo.toml
```

## Roadmap: Removing batchalign-cli from pyo3

The remaining heavy dependency is `batchalign-cli`, which pulls in
`batchalign-app` (the full server). pyo3 needs `batchalign-cli` for exactly
two symbols:

1. **`Cli`** struct (clap derive) - parsed by `cli_entry.rs` for console_scripts
2. **`run_command()`** - the command dispatch function

### Option A: Extract `batchalign-cli-entry` (small, focused)

Create a tiny crate `batchalign-cli-entry` with just the `Cli` struct and
`run_command()`. This crate would depend on `batchalign-app` but pyo3 would
depend on `batchalign-cli-entry` instead of `batchalign-cli`.

**Problem:** `run_command()` calls into the full server (e.g. `transcribe`
starts an axum server). So this crate would still need `batchalign-app`.

### Option B: Feature-gate the server in batchalign-cli

Split `batchalign-cli` into:
- Core CLI parsing (clap, args) - no server dependency
- `server` feature that enables commands requiring the server

pyo3 would use `batchalign-cli` with `default-features = false`, excluding
server commands. The console_scripts entry point would only support a subset
of commands that don't need the server.

**Problem:** Most commands need the server (transcribe, morphotag, align,
etc.). Only `validate`, `check`, and `convert` are server-free.

### Option C: Separate the console_scripts binary (recommended)

Instead of embedding `run_command()` in the pyo3 `.so`, ship a separate
`batchalign3` binary via the wheel's `[project.scripts]` entry. maturin
supports this via `bindings = "bin"` for a separate binary crate.

The pyo3 `.so` would only contain the `batchalign_core` Python module
(CHAT parsing, injection, alignment). The CLI binary would be a separate
build artifact.

**Impact:**
- pyo3 compilation drops `batchalign-cli` and `batchalign-app` entirely
- Cold build time drops to ~1-2 minutes (just chat-ops + types + talkbank-*)
- `uv tool install batchalign3` still gets the CLI binary
- Incremental `uv run maturin develop` is nearly instant

**Complexity:** Moderate. Requires restructuring the wheel packaging to include
both the `.so` and the CLI binary. maturin supports mixed bindings but the
`pyproject.toml` and build pipeline need updates.

### Option D: Process-spawn the CLI (simplest)

Instead of calling `run_command()` from Python, have the console_scripts
entry point spawn `batchalign3` as a subprocess. The binary is installed
alongside the wheel.

**Impact:** Same as Option C for build times. Simpler implementation but
adds process-spawn overhead for CLI invocations via `uv tool install`.

## What NOT to do

- **Do not vendor types.** Never copy-paste type definitions between crates.
  Always use path dependencies. `batchalign-types` is the single source of
  truth for domain newtypes and worker IPC types.

- **Do not add server deps to pyo3.** If pyo3 needs a new type, check if it
  belongs in `batchalign-types` first. Only add deps to pyo3/Cargo.toml if
  they are genuinely needed by the Python extension.

- **Do not add `batchalign-app` back to pyo3.** If you need app types in
  pyo3, move them to `batchalign-types` instead.

## Verification checklist

After any dependency change to `pyo3/Cargo.toml`:

```bash
# 1. pyo3 compiles without batchalign-app as a direct dep
cargo check --manifest-path pyo3/Cargo.toml

# 2. pyo3 compiles without the dashboard
mv frontend/dist frontend/dist.bak
cargo check --manifest-path pyo3/Cargo.toml
mv frontend/dist.bak frontend/dist

# 3. pyo3 tests pass
cargo nextest run --manifest-path pyo3/Cargo.toml

# 4. maturin builds the wheel
uv run maturin develop -m pyo3/Cargo.toml

# 5. Python tests pass
uv run pytest
```
