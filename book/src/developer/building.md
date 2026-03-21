# Building & Development

**Status:** Current
**Last modified:** 2026-03-21 07:16 EDT

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
For the fastest contributor loop, `make build-python` is the slim local-dev
profile and `make build-python-full` is the packaged-release surface.

The expected directory layout:

```
parent/
├── talkbank-tools/    # CHAT grammar, parser, model, transform crates
└── batchalign3/       # This repo (Rust CLI + server + Python ML workers)
```

This creates a `.venv` managed by uv. Never use `pip install` directly.

`make sync` provisions the same built-in engine surface as the base package,
including Cantonese/HK providers. There is no separate HK-specific dev extra
path.

## Running the CLI

In a source checkout, `uv run batchalign3` is still the normal way to invoke
the installed console script. After `make build-python`, the Python wrapper
falls back to the repo CLI when the embedded bridge is intentionally omitted,
so the fast extension-only rebuild still leaves you with a runnable
`batchalign3` command. This is the recommended loop while editing command
semantics, workflow families, or most docs.

For the fastest contributor loop, pair `make build-python` with one CLI build
up front:

```bash
cargo build -p batchalign-cli
```

After that, repeated `uv run batchalign3 ...` invocations will use the local
`target/debug/batchalign3` binary through the wrapper fallback. Reserve
`uv run` for Python tools such as `pytest`, `mypy`, and `maturin` when you are
not invoking the CLI.

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
| Shared chat logic (`crates/batchalign-chat-ops/`) or PyO3 bridge (`pyo3/`) | `make build-python`; for the fastest CLI loop in a source checkout, also build the CLI once (`cargo build -p batchalign-cli` or `make build-rust`) so the wrapper can fall back to `target/debug/batchalign3` |
| Workflow-layer changes (`crates/batchalign-app/src/workflow/`) | `make build-rust` and usually `make build-python` if the CLI bridge surface changed |
| Cross-cutting or dashboard changes | `make build` (requires Node.js + npm because it rebuilds the embedded dashboard) |

## Rebuilding the Rust Extension

The `batchalign_core` Python package is a PyO3 Rust extension built by maturin.
The repo-native rebuild path is:

```bash
make build-python
```

This is the **slim local-dev** PyO3 profile. It skips the heavy CLI-entry and
Rev.AI bridge features so local rebuilds do not always drag those crates into
the extension build graph. In a source checkout, the Python wrapper falls back
to the repo CLI when the embedded bridge is absent.

When you need to verify the packaged-release feature set, use:

```bash
make build-python-full
```

That target builds the fuller extension profile used by the packaged install
path, including the embedded CLI bridge and Rev.AI bridge. Use it when you are
changing the packaging contract, not for ordinary workflow development.

## Where Command Logic Should Live

If you are changing command behavior, the first stop should be
`crates/batchalign-app/src/workflow/`.

- Workflow families own command semantics and typed intermediate bundles.
- `crates/batchalign-app/src/runner/` owns job lifecycle and queueing.
- `crates/batchalign-cli/src/dispatch/` should stay thin and focus on
  argument parsing, capability gating, and whether a command runs locally or
  through the server.
- `pyo3/` should stay a thin bridge, not the place where new command logic is
  invented.

Run the Rust test suite to verify your changes:

```bash
cargo nextest run --manifest-path pyo3/Cargo.toml
```

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
