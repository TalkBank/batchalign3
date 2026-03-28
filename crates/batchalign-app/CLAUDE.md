# batchalign-app — HTTP Server, Job Store, and NLP Orchestration

**Status:** Current
**Last modified:** 2026-03-27 23:16 EDT

## Overview

Axum-based REST server managing job lifecycle, Python worker dispatch, and server-side
CHAT orchestration (CHAT ownership boundary — server owns parse/cache/inject/serialize,
Python workers provide stateless NLP inference only).

## Module Map

| Module | Purpose |
|--------|---------|
| `lib.rs` | `create_app()`, WebSocket handler, graceful shutdown |
| `state.rs` | `AppState`, capability gate (`validate_infer_capability_gate()`) |
| `cache/` | Tiered utterance cache: moka in-memory hot layer + SQLite cold backend (BLAKE3 keys) |
| `store/` | `JobStore` composition, `JobRegistry` actor, `OperationalCounterStore`, SQLite write-through, conflict detection, memory gating |
| `runner/` | Per-job async task: dispatch routing, parallelism, preflight. `runner/policy.rs` has `infer_task_for_command()` and `command_requires_infer()`. `runner/util/` has progress helpers |
| `runner/dispatch/` | Dispatch family implementations: `infer_batched.rs`, `fa_pipeline.rs`, `transcribe_pipeline.rs`, `benchmark_pipeline.rs`, `compare_pipeline.rs`, `media_analysis_v2.rs` |
| `db/` | SQLite persistence (WAL): `schema.rs`, `insert.rs`, `query.rs`, `update.rs`, `recovery.rs` |
| `error.rs` | Typed errors → HTTP status codes (404, 409, 500) |
| `morphosyntax/` | Server-side morphosyntax orchestrator (parse→clear→collect→cache→infer→inject→serialize) |
| `pipeline/` | `PipelineServices`, transcribe pipeline, text infer pipeline, morphosyntax batch |
| `utseg.rs` | Utterance segmentation orchestrator |
| `translate.rs` | Translation orchestrator (injects `%xtra`) |
| `coref.rs` | Coreference resolution (document-level, sparse, English-only) |
| `fa/` | Forced alignment orchestrator (per-file, multi-group, audio-aware, DP alignment, incremental FA) |
| `workflow/` | Workflow-family registry, typed descriptors, traits, and per-command implementations |
| `worker/` | Worker pool, IPC handle, V2 request builders and result types |
| `media.rs` | Media file resolution with walk cache (60s TTL) |
| `ws.rs` | WebSocket broadcast event types |
| `websocket.rs` | WebSocket route and handler |
| `hostname.rs` | Tailscale IP→hostname resolution |
| `routes/` | HTTP endpoints: health, jobs (CRUD+SSE), media, dashboard, bug reports, traces |
| `types/` | API models, parameter structs, worker IPC types, scheduling types, and re-exports of shared domain newtypes from `batchalign-types` |

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

## Dispatch Routing (runner/)

Dispatch shapes (driven by `workflow/registry.rs`):
1. **Batched text infer** (`runner/dispatch/infer_batched.rs`) — morphotag, utseg, translate, coref: pool all utterances from all files, group by language, dispatch language groups with **semaphore-bounded concurrency** (`morphosyntax/batch.rs`, `max_total_workers / max_workers_per_key` concurrent groups), and within each group split into chunks across multiple workers (`morphosyntax/worker.rs`, up to `max_workers_per_key`). Unsupported languages filtered at preflight (`stanza_languages.rs`).
2. **Per-file FA** (`runner/dispatch/fa_pipeline.rs`) — align: files processed concurrently via `JoinSet` + `Semaphore(num_workers)`. UTR pre-pass runs before FA grouping with ASR result caching. Fallback UTR retries timing recovery after FA failures. For mostly-timed files (>50% timed, audio >60s), partial-window ASR runs only on untimed regions.
3. **Per-file transcribe** (`runner/dispatch/transcribe_pipeline.rs`) — transcribe, transcribe_s: per-file audio processing with optional diarization, utseg, and morphosyntax.
4. **Per-file benchmark** (`runner/dispatch/benchmark_pipeline.rs`) — composite transcribe + compare.
5. **Per-file compare** (`runner/dispatch/compare_pipeline.rs`) — gold-anchored projection.
6. **Per-file media analysis** (`runner/dispatch/media_analysis_v2.rs`) — opensmile, avqi: concurrent files via `JoinSet` + `Semaphore(num_workers)`, worker `execute_v2`.

**Post-validation is warn-only** — output is always serialized and written even if
post-validation finds issues. This ensures output CHAT can be inspected for debugging.

## Type System

Domain newtypes are defined in `batchalign-types` using `string_id!` and `numeric_id!`:
- **`../batchalign-types/src/macros.rs`** — macro definitions (generates Deref, serde transparent, From, Borrow, etc.)
- **`../batchalign-types/src/domain/`** — `JobId`, `CommandName`, `ReleasedCommand`, `LanguageCode3`, `LanguageSpec`, `DisplayPath`, `EngineVersion`, `CorrelationId`, `NumSpeakers`, `UnixTimestamp`, `DurationMs`, `MemoryMb`, etc.
- **`../batchalign-types/src/scheduling.rs`** — `AttemptId`, `WorkUnitId`
- **`types/params.rs`** — `CachePolicy`, `WorTierPolicy` enums; `MorphosyntaxParams`, `FaParams`, `AudioContext` structs
- **`pipeline/mod.rs`** — `PipelineServices` (shared infrastructure refs: pool, cache, engine_version)

**Boundary patterns:** Raw `String` from HTTP → `JobId::from()` at handler entry. `&Path` in domain code → `to_string_lossy()` at IPC/JSON. `bool` from CLI → `CachePolicy::from()` at dispatch. See `book/src/architecture/type-driven-design.md`.

## Memory Gate

Polls `sysinfo::available_memory()` with configurable threshold (default 2048 MB, 0=disable).
**Idle worker bypass**: skips memory check when pool has reusable workers for the job's
`(command, lang)` — prevents deadlock where loaded workers hold RAM.

## Middleware Stack

CORS → body limit (`max_body_bytes_mb`, default 100 MB) → panic catching → timeout (5 min) → tracing → compression.

Axum's built-in 2 MB `Json` extractor limit is disabled on job routes so the
outer `RequestBodyLimitLayer` is the sole body-size guard.  See
`book/src/developer/http-body-limits.md` for the full story.
