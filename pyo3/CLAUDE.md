# batchalign-core - PyO3 Bridge

## Overview

PyO3 bridge between batchalign (Python) and the talkbank Rust crates. All CHAT
manipulation happens through AST types from `talkbank-model`. The Python pipeline
passes an opaque `ParsedChat` handle between engines -- parse once, mutate in
place, serialize once.

## Layout

Single-crate project (no workspace). `Cargo.toml` and `src/` live directly in `pyo3/`.

## Key Commands

```bash
cargo nextest run --manifest-path pyo3/Cargo.toml
cargo build --manifest-path pyo3/Cargo.toml
cd /path/to/batchalign3 && uv run maturin develop
```

## Rust Coding Standards

These are the workspace-universal Rust coding standards. The canonical copy
lives in the repository root `../CLAUDE.md`.

**Edition and Tooling**
- Rust **2024 edition**.
- `cargo fmt` before committing. Use `cargo fmt` (not standalone `rustfmt`) for workspace-consistent formatting.
- Run `cargo clippy --all-targets -- -D warnings` periodically (dedicated lint passes), not on every change. Fix real issues; do not silence with `#[allow(clippy::...)]` without explicit approval.

**Error Handling**
- **No panics for recoverable conditions.** Use typed errors (`thiserror`); use `miette` for rich diagnostics where appropriate.
- **No silent swallowing.** Every unexpected condition must be handled with explicit error reporting — no `.ok()`, `.unwrap_or_default()`, or silent fallbacks that hide bugs.

**Output and Logging**
- **Library code:** `tracing` macros — never `println!`/`eprintln!`.
- **Test code:** `println!` is acceptable (cargo captures it).

**Lazy Initialization**
- `LazyLock<Regex>` (from `std::sync`) for constant regex patterns. Never call `Regex::new()` inside functions or loops.
- `OnceLock` for per-instance memoization of runtime-determined values.
- Prefer `const` when possible. All lazy init via `std::sync` — no external crate dependencies.

**Type Design**
- **No boolean blindness.** Enums over bools for anything beyond simple on/off. Banned: 2+ bool params, 2+ related bool fields, opposite bool pairs (`foo`/`no_foo`), ambiguous bool returns. OK: `verbose`, `force`, `quiet`, single on/off flags where the name is self-documenting.
- **`BTreeMap` for deterministic JSON** in tests and snapshot tests (not `HashMap`).
- Prefer explicit enums over ambiguous `Option` when there are multiple meaningful states.

**File Size Limits**
- **Recommended:** ≤400 lines. **Hard limit:** ≤800 lines (must be split).

**Refactoring Triggers** — Stop and refactor when you see:
- `x: i32, y: i32` for domain data → use domain structs
- Multiple booleans for state → use enum with variants
- `fn parse() -> Option<T>` where failure reason matters → use `Result<T, ParseError>`
- `match s { "win" => ... }` on raw strings → parse to `enum` at boundary

**Git**
- Conventional Commits: `<type>[scope]: <description>`

## Rules

- **No raw CHAT strings.** Use AST types and serialize via `ChatFile::to_chat_string()`.
- **Parse fragments correctly.** Use `ChatParser` trait, never regex/split raw CHAT.
- **All JSON via serde.** `#[derive(Deserialize)]`/`#[derive(Serialize)]` structs only.
  No hand-rolled JSON parsers.
- **Prefer mutation over re-parse.** Methods mutate `&mut self.inner` directly.
- **Standalone functions are thin wrappers.** Business logic in `_inner()` functions
  operating on `&mut ChatFile`.
- **GIL release.** All pure-Rust methods use `py.detach()` (pyo3 0.28). Callbacks hold
  the GIL only during Python invocation.
- **Callback JSON contracts are frozen.** Changing schemas requires updating both Rust
  serde structs AND Python callback implementations.

## Architecture

```
ParsedChat (#[pyclass])  ->  inner: ChatFile (from talkbank-model)
#[pymethods]             ->  call _inner() on &mut self.inner
#[pyfunction]s           ->  parse -> _inner() -> serialize
```

Modules: `lib.rs` (exports), `cli_entry.rs` (console_scripts entry point),
`forced_alignment.rs`, `extract.rs`, `inject.rs`, `dp_align.rs`,
`utterance_segmentation.rs`, `retokenize.rs`, `mor_parser.rs`
(thin delegation to `talkbank-direct-parser`).

## Callback Protocols

All callbacks: Python callable receives JSON `str`, returns JSON `str`.

- **Morphosyntax**: `{words, terminator, special_forms}` -> `{mor, gra, tokens}`
- **Forced Alignment**: `{words, audio_start_ms, audio_end_ms}` -> `{timings}`
- **Translation**: `{text, speaker}` -> `{translation}`
- **Utterance Segmentation**: `{words, text}` -> `{assignments}`

Batched variants available for morphosyntax and utterance segmentation (preferred).

---
Last Updated: 2026-02-26
