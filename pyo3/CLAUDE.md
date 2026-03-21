# batchalign-core — Rust Worker Runtime

**Status:** Current
**Last modified:** 2026-03-21 14:32 EDT

## Overview

Slim PyO3 bridge providing the Rust worker runtime for batchalign3's Python ML
worker processes. Workers are stateless inference endpoints that load ML models,
receive structured data from the Rust server via stdio JSON-lines IPC, run
inference, and return raw results.

This crate does NOT contain CHAT parsing, AST manipulation, or pipeline
orchestration — all of that lives in the Rust server (`crates/batchalign-app/`)
and `batchalign-chat-ops`.

## Layout

Single-crate project (no workspace). `Cargo.toml` and `src/` live directly in `pyo3/`.

```
pyo3/src/
├── lib.rs                  # Module registration (~95 lines)
├── cli_entry.rs            # PyPI console_scripts entry point
├── worker_protocol.rs      # IPC message dispatch
├── worker_asr_exec.rs      # ASR execution (Whisper, HK providers)
├── worker_fa_exec.rs       # Forced alignment execution
├── worker_media_exec.rs    # Speaker diarization, OpenSMILE, AVQI
├── worker_text_results.rs  # Text task normalization + token alignment
├── worker_artifacts.rs     # Prepared artifact loading from IPC
├── hk_asr_bridge.rs        # HK/Cantonese provider projection + normalization
├── py_json_bridge.rs       # Python→JSON conversion utility
└── revai/                  # Rev.AI native client wrappers (feature-gated)
```

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
- `cargo fmt` before committing.
- Run `cargo clippy --all-targets -- -D warnings` periodically.

**Error Handling**
- **No panics for recoverable conditions.** Use typed errors.
- **No silent swallowing.** No `.ok()`, `.unwrap_or_default()`, or silent fallbacks.

**Output and Logging**
- **Library code:** `tracing` macros — never `println!`/`eprintln!`.
- **Test code:** `println!` is acceptable.

**File Size Limits**
- **Recommended:** ≤400 lines. **Hard limit:** ≤800 lines.

**Git**
- Conventional Commits: `<type>[scope]: <description>`

## Rules

- **All JSON via serde.** `#[derive(Deserialize)]`/`#[derive(Serialize)]` structs only.
- **GIL release.** All pure-Rust methods use `py.detach()` (pyo3 0.28).
- **No CHAT parsing here.** CHAT manipulation is in `batchalign-chat-ops` and
  the Rust server. This crate only bridges Python ML calls.

## Architecture

```
Rust Server (crates/batchalign-app/)
  ├── Parses CHAT, extracts payloads
  ├── Sends IPC request to Python worker (stdio JSON-lines)
  │
  └── Python Worker Process
        ├── worker_protocol.rs: dispatch IPC messages
        ├── worker_*_exec.rs: load prepared artifacts, call ML model
        ├── hk_asr_bridge.rs: project HK provider output
        └── Returns raw results → Rust server injects into CHAT
```
