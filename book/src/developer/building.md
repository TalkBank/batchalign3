# Building & Development

**Status:** Current
**Last updated:** 2026-03-20

Development is supported on **Windows, macOS, and Linux**. The instructions below use Unix shell syntax; on Windows, use PowerShell or Git Bash equivalently.

## Prerequisites

- **[uv](https://docs.astral.sh/uv/)** -- Python package manager (all platforms). Used for all dependency management and running commands.
- **Rust (stable)** via [rustup](https://rustup.rs/) (all platforms) -- needed for the Rust CLI and PyO3 extension.
- **Node.js + npm** -- needed for `make build` and `make build-dashboard`, which rebuild the embedded dashboard bundled into the Rust binary.
- **`cargo-nextest`** -- Required for Rust test runs. Install once with `cargo install cargo-nextest --locked`.
- **[maturin](https://www.maturin.rs/)** -- Required only if you modify the Rust `batchalign_core` extension.
- **Python 3.12** for development and current deployment targets. 3.14t/free-threaded experiments are currently paused pending full engine support. See `developer/python-versioning.md`.
- **Platform note:** On macOS, `python` and `python3` may not exist outside a venv. Always use `uv run` to execute Python commands, which handles this automatically on all platforms.

## Development Install

Batchalign3's Rust crates depend on [`talkbank-tools`](https://github.com/talkbank/talkbank-tools) via local path references. Both repos must be cloned as siblings:

```bash
git clone https://github.com/talkbank/talkbank-tools.git
git clone https://github.com/talkbank/batchalign3.git
cd batchalign3
make sync
make build
```

If you do not need the dashboard build during iteration, you can rebuild just
the Rust/PyO3 surfaces with `make build-python` and `make build-rust`.

The expected directory layout:

```
parent/
├── talkbank-tools/    # CHAT grammar, parser, model, transform crates
└── batchalign3/       # This repo (Rust CLI + server + Python ML workers)
```

This creates a `.venv` managed by uv. Never use `pip install` directly.

The base `make sync` environment does **not** include optional HK/Cantonese
engines. If you need to exercise those providers from a source checkout, sync
the matching extras into the repo venv explicitly:

```bash
uv sync --group dev --extra hk
# or a narrower subset:
uv sync --group dev --extra hk-tencent
```

`uv sync` is declarative about extras. A later sync only includes the extras
named on that command, so include the full set you still need rather than
assuming separate `uv sync --extra ...` calls accumulate.

## Running the CLI

`batchalign3` is a Rust binary, not a Python console-script entry point. Use the
compiled binary or `cargo run` during development. Reserve `uv run` for Python
tools such as `pytest`, `mypy`, and `maturin`.

```bash
make build
./target/debug/batchalign3 --help
./target/debug/batchalign3 transcribe input_dir -o output_dir --lang eng
./target/debug/batchalign3 morphotag input_dir -o output_dir
./target/debug/batchalign3 align input_dir -o output_dir

# Or let Cargo rebuild the Rust binary incrementally for you:
cargo run -p batchalign-cli -- transcribe input_dir -o output_dir --lang eng
```

## What to Rebuild After Changes

Use the repo-native build targets so the Rust CLI, the shared `batchalign-chat-ops`
crate, and the `batchalign_core` extension stay in sync:

| What changed | What to rebuild |
| --- | --- |
| Python code only (`batchalign/`) | Nothing; the next worker process picks up the change |
| Rust CLI / server (`crates/batchalign-cli/`, `crates/batchalign-app/`) | `cargo build -p batchalign-cli` or `make build-rust` |
| Shared chat logic (`crates/batchalign-chat-ops/`) or PyO3 bridge (`pyo3/`) | `make build-python`; if you will run `./target/debug/batchalign3` directly after a shared crate change, also rebuild the CLI or use `cargo run -p batchalign-cli -- ...` |
| Cross-cutting or dashboard changes | `make build` (requires Node.js + npm because it rebuilds the embedded dashboard) |

## Rebuilding the Rust Extension

The `batchalign_core` Python package is a PyO3 Rust extension built by maturin.
The repo-native rebuild path is:

```bash
make build-python
```

This runs `uv run maturin develop -m pyo3/Cargo.toml` under the hood.

Run the Rust test suite to verify your changes:

```bash
cargo nextest run --manifest-path pyo3/Cargo.toml
```

## Pre-push Checks

Install the hook once with `make install-hooks`. On every `git push` it runs
the checks below in order and aborts the push on the first failure.

| Gate | Command | What it catches |
| --- | --- | --- |
| **Formatting** | `cargo fmt --all -- --check` | Unformatted Rust code that CI will reject |
| **Clippy** | `cargo clippy --workspace --all-targets -- -D warnings` | Lint warnings promoted to errors; also proves compilation across all targets |
| **Dashboard API drift** | `bash scripts/check_dashboard_api_drift.sh` | REST API schema out of sync with generated TypeScript (CI fails on drift) |

### Why not `cargo check`?

`cargo check --workspace` was removed from the hook because `cargo clippy
--workspace --all-targets` is a strict superset — it invokes the same type
system and borrow checker, then also runs lints. Running both just doubles
compile time for no additional signal.

### Why not mypy?

mypy is run in `make ci-local` and in CI, but is excluded from the pre-push
hook because it is noticeably slower on first run (cold transitive import
resolution) and produces noise from vendored stub packages outside our control.
Run it manually before committing Python changes:

```bash
uv run mypy
# or together with clippy:
make lint
```

### Why not IPC schema drift?

IPC schema drift (`make check-ipc-drift`) is also in `make ci-local` but not
the pre-push hook because it requires a full Rust build of the schema-generation
binary, which is expensive. Run it manually when you change any Rust struct
with `#[derive(schemars::JsonSchema)]` that crosses the Python worker boundary.

### Cost of a CI failure

CI minutes on GitHub Actions are finite and failures block merges. Each full
Rust CI run takes 8–15 minutes. The pre-push hook's goal is to catch the
commonest CI killers (formatting, clippy, API drift) in under a minute on
a warm build cache.

## Type Checking

Run the current mypy gate before every commit:

```bash
uv run mypy
# or together with clippy:
make lint
```

Strictness lives in `mypy.ini`, and CI runs the same repo-native command shape.

Do not commit with mypy errors. Use `# type: ignore[<code>]` only when
necessary, and always include the specific error code.

## Type Annotation Rules

All new and modified code must include type annotations:

- Annotate all function parameters and return types.
- Use modern syntax: `list[str]` not `List[str]`, `str | None` not `Optional[str]`.
- **`Any` and `object` are banned as type annotations.** Use specific types. For ML library types that are expensive to import, use `TYPE_CHECKING` guards with the real type.
- Use `from __future__ import annotations` for forward references where needed.
- Prefer `TYPE_CHECKING` imports for heavy dependencies used only in annotations.

## The CHAT Format Rule

All CHAT parsing and serialization must go through principled AST manipulation via `batchalign_core` Rust functions. This is a hard rule with no exceptions.

**Do not:**
- Use regex or string splitting to extract or modify CHAT content.
- Process CHAT line-by-line in Python.
- Manipulate CHAT header metadata with ad-hoc text code.

**Instead:**
- Use existing `batchalign_core` functions (`parse`, `parse_lenient`, `build_chat`, `add_morphosyntax`, `add_forced_alignment`, `extract_nlp_words`, etc.).
- If the function you need does not exist, add a new Rust function to `batchalign_core` and call it from Python.

CHAT has complex escaping, continuation lines, and encoding rules that ad-hoc text manipulation will get wrong. The Rust AST handles all of this correctly.
