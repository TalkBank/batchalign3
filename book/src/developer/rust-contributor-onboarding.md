# Rust Contributor Onboarding

**Status:** Current
**Last updated:** 2026-03-15

This page is the shortest path to productive work on the Rust side of Batchalign3.

## Start Here

1. Read the user-facing [CLI reference](../user-guide/cli-reference.md).
2. Read the [Rust workspace map](rust-workspace-map.md).
3. Read the [Rust CLI and Server](rust-cli-and-server.md) for dispatch architecture and command-creation checklist.
4. Read the [migration book](../migration/index.md) if you need historical context from Batchalign2.
5. Run the root workspace tests before changing behavior.

## Current Rust Surfaces

- root workspace:
  - `batchalign-cli`
  - `batchalign-app`
  - `batchalign-chat-ops`
- PyO3 bridge:
  - `pyo3/` building `batchalign_core`

## Setup

```bash
make sync
make build
cargo check --workspace
cargo nextest run --workspace
cargo nextest run --manifest-path pyo3/Cargo.toml
```

If your work touches HK/Cantonese providers, sync those extras into the
repo-managed `.venv` after `make sync`:

```bash
uv sync --group dev --extra hk
```

Rebuild rule of thumb while iterating:

- CLI/server-only changes: `cargo build -p batchalign-cli` or `make build-rust`
- `batchalign-chat-ops` or `pyo3/` changes: `make build-python`
- shared crate changes you want to validate through `./target/debug/batchalign3`:
  rebuild the CLI too or use `cargo run -p batchalign-cli -- ...`

## Where To Work

- CLI flags, logs, cache, daemon behavior: `crates/batchalign-cli`
- server routes, jobs, persistence, OpenAPI: `crates/batchalign-app`
- shared CHAT transformations and mapping logic: `crates/batchalign-chat-ops`
- Python extension boundary: `pyo3/`

## Expectations

- add or update tests before large behavioral changes
- keep public docs in sync with the actual CLI and server surface
- do not introduce maintainer-local filesystem paths into public docs
- treat migration notes as historical context, not as the current API contract

## Useful Commands

```bash
cargo build -p batchalign-cli
make build-python
cargo nextest run -p batchalign-cli --test cli
cargo nextest run -p batchalign-cli --test e2e
cargo nextest run -p batchalign-app --test integration
cargo nextest run --manifest-path pyo3/Cargo.toml
cargo run -q -p batchalign-cli -- openapi --check --output openapi.json
```
