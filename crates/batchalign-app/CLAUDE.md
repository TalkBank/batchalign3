# batchalign-app — HTTP Server, Job Store, and NLP Orchestration

**Status:** Current
**Last updated:** 2026-03-16

## Overview

Axum-based REST server managing job lifecycle, Python worker dispatch, and server-side
CHAT orchestration (CHAT ownership boundary — server owns parse/cache/inject/serialize,
Python workers provide stateless NLP inference only).

## Module Map

| Module | Purpose |
|--------|---------|
| `lib.rs` | `create_app()`, `AppState`, WebSocket handler, graceful shutdown |
| `cache/` | Tiered utterance cache: moka in-memory hot layer + SQLite cold backend (BLAKE3 keys) |
| `store.rs` | `JobStore` composition, SQLite write-through, conflict detection, memory gating |
| `store/registry.rs` | Owned `JobRegistry` actor for in-memory job projections and mutations |
| `store/counters.rs` | Small `OperationalCounterStore` for health/metrics bookkeeping |
| `runner.rs` | Per-job async task: dispatch routing (infer vs process), parallelism, preflight. Dispatch modules set progress labels at stage transitions; `runner/util.rs` has `set_file_progress()`, `ProgressSender`, and `spawn_progress_forwarder()` for sub-file progress |
| `db.rs` | SQLite persistence (WAL), schema, recovery, TTL pruning |
| `error.rs` | Typed errors → HTTP status codes (404, 409, 500) |
| `morphosyntax.rs` | Server-side morphosyntax orchestrator (parse→clear→collect→cache→infer→inject→serialize) |
| `utseg.rs` | Utterance segmentation orchestrator |
| `translate.rs` | Translation orchestrator (injects `%xtra`) |
| `coref.rs` | Coreference resolution (document-level, sparse, English-only) |
| `fa.rs` | Forced alignment (per-file, multi-group, audio-aware, DP alignment). UTR pre-pass in `dispatch/infer.rs::process_one_fa_file` |
| `media.rs` | Media file resolution with walk cache (60s TTL) |
| `ws.rs` | WebSocket broadcast event types |
| `hostname.rs` | Tailscale IP→hostname resolution |
| `routes/` | HTTP endpoints: health, jobs (CRUD+SSE), media, dashboard, bug reports |
| `types/` | Domain newtypes (`string_id!`/`numeric_id!` macros), API models, parameter structs, worker IPC types, scheduling types |

## Job Registry Concurrency Model

`JobRegistry` no longer exposes a shared `Mutex<HashMap<...>>` boundary.
`JobStore` creates one owned actor task with an `mpsc::UnboundedSender`
mailbox. Callers submit either:

- `Inspect` closures for read-only projections
- `Mutate` closures for in-place transitions

Each request pairs with a `oneshot` reply so callers still `await` a typed
result. Prefer the named store/registry methods for normal job-local work;
`inspect_all()` / `mutate_all()` remain the bulk escape hatches for recovery and
other collection-wide operations.

Route, query, and runner code should think in terms of job transitions and
projections, not in terms of "lock the map and poke fields."

## Key Commands

```bash
cargo nextest run -p batchalign-app
cargo clippy -p batchalign-app -- -D warnings
```

## Route Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/jobs` | Submit job (validates command, checks conflicts) |
| GET | `/jobs`, `/jobs/{id}` | List/get jobs |
| GET | `/jobs/{id}/results[/{filename}]` | Download results |
| GET | `/jobs/{id}/stream` | SSE streaming (real-time progress) |
| POST | `/jobs/{id}/cancel`, `/jobs/{id}/restart` | Lifecycle |
| DELETE | `/jobs/{id}` | Permanent delete |
| GET | `/health` | Version, capabilities, worker state |

## Dispatch Routing (runner.rs)

Three dispatch shapes:
1. **Batched text infer** — morphotag, utseg, translate, coref: pool all utterances from all
   files into one `batch_infer` call for GPU batch efficiency.
2. **Per-file infer** — align (FA), transcribe: files processed concurrently via
   `JoinSet` + `Semaphore(num_workers)`. Each file gets its own worker checkout.
   For align: UTR pre-pass runs before FA grouping with ASR result caching
   (see `dispatch/infer.rs::process_one_fa_file` and `run_utr_pass()`).
   Fallback UTR retries timing recovery after FA failures. For mostly-timed
   files (>50% timed, audio >60s), partial-window ASR runs only on untimed
   regions.
3. **Per-file process** — opensmile, avqi, benchmark: concurrent files via `JoinSet` +
   `Semaphore(num_workers)`, simple worker IPC.

**Post-validation is warn-only** — output is always serialized and written even if
post-validation finds issues. This ensures output CHAT can be inspected for debugging.

## Type System

Domain newtypes are defined in `types/` using `string_id!` and `numeric_id!` macros:
- **`types/macros.rs`** — macro definitions (generates Deref, serde transparent, From, Borrow, etc.)
- **`types/api.rs`** — `JobId`, `CommandName`, `LanguageCode3`, `FileName`, `EngineVersion`, `CorrelationId`, `NumSpeakers`, `UnixTimestamp`, etc.
- **`types/params.rs`** — `CachePolicy`, `WorTierPolicy` enums; `MorphosyntaxParams`, `FaParams`, `AudioContext` structs
- **`pipeline/mod.rs`** — `PipelineServices` (shared infrastructure refs: pool, cache, engine_version)

**Boundary patterns:** Raw `String` from HTTP → `JobId::from()` at handler entry. `&Path` in domain code → `to_string_lossy()` at IPC/JSON. `bool` from CLI → `CachePolicy::from()` at dispatch. See `book/src/architecture/type-driven-design.md`.

## Memory Gate

Polls `sysinfo::available_memory()` with configurable threshold (default 2048 MB, 0=disable).
**Idle worker bypass**: skips memory check when pool has reusable workers for the job's
`(command, lang)` — prevents deadlock where loaded workers hold RAM.

## Middleware Stack

CORS → body limit (100 MB) → panic catching → timeout (5 min) → tracing → compression.
