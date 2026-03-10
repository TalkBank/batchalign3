# Worker Memory Architecture

**Status:** Current
**Last updated:** 2026-03-17

Developer reference for the auto-tuning, memory gate, worker pool, and warmup
internals. For user-facing configuration, see
[Worker Tuning](../user-guide/worker-tuning.md).

## Auto-tuning formula

`compute_job_workers()` in `runner/util/auto_tune.rs` decides per-job file
parallelism:

```
workers = min(num_files, by_cpu, by_memory)
        clamped to [1, max_thread_workers]
```

Where:
- `by_memory = available_mb / (command_base_mb * loading_overhead)`
- `by_cpu = std::thread::available_parallelism()`
- `command_base_mb` comes from `runtime_constants.toml` `[command_base_mb.threaded]`
- `loading_overhead` is 1.5× (from `runtime_constants.toml` `[memory]`)

If `config.max_workers_per_job > 0`, it overrides auto-tuning (still capped).

For single-file jobs, the function short-circuits to 1.

## Memory gate

`memory_gate()` in `store/mod.rs` blocks job dispatch until sufficient RAM is
available. Flow:

1. **Disabled check:** if `critical_mb == 0`, skip entirely
2. **Idle worker bypass:** if the pool has idle workers for `(command, lang)`,
   skip the memory check — those workers are already loaded, no new allocation
   needed. This prevents a deadlock where idle workers consume RAM yet are the
   exact workers the new job needs.
3. **Threshold check:** compare `sysinfo::available_memory()` against
   `memory_gate_mb` (default 2048 MB)
4. **Polling:** if below threshold, poll every 2 seconds with a 60-second
   timeout
5. **Timeout:** return `ServerError::MemoryPressure` with a hint listing
   active workers

The idle worker bypass is critical: without it, a server with warm workers
holding 200 GB of models would fail the 2 GB gate even though the next job
needs no new memory.

## Worker pool

### Structure

```
WorkerPool
├── config: PoolConfig
├── groups: Arc<Mutex<HashMap<WorkerKey, Arc<WorkerGroup>>>>
├── cancel: CancellationToken
└── warmup_status: AtomicU8 (WarmupStatus enum)

WorkerGroup (per (target, lang, engine_overrides) key)
├── idle: std::sync::Mutex<VecDeque<WorkerHandle>>
├── available: Semaphore (permits = idle count)
├── total: AtomicUsize (idle + checked-out)
└── bootstrap: AsyncMutex<()> (serializes spawns per key)
```

### Checkout flow

`checkout()` is the core worker acquisition path:

1. Try `semaphore.try_acquire()` — if a permit exists, pop from idle queue
2. If no permits, try `try_spawn_into_group()` — atomically claim a slot via
   `compare_exchange` on `total`, then spawn under the bootstrap lock
3. If at capacity, `semaphore.acquire().await` — async wait for a worker return
4. Wrap the popped `WorkerHandle` in `CheckedOutWorker` (RAII guard)

`CheckedOutWorker::drop()` returns the worker to the idle queue and releases
a semaphore permit. If the worker was `take()`n (dead), `total` is decremented
instead.

The idle queue uses `std::sync::Mutex` (not tokio), held only for microsecond
push/pop operations — never across `.await`.

### Bootstrap serialization

The `bootstrap` `AsyncMutex` on each `WorkerGroup` prevents a burst of
concurrent requests from launching multiple heavy Python workers for the same
key simultaneously. Only one spawn per key proceeds at a time, smoothing
model-loading spikes without reducing steady-state concurrency.

## runtime_constants.toml

Single source of truth consumed by both Rust (`include_str!` at compile time)
and Python (`read` at import time). Key sections:

- `[cmd2task]` — maps CLI command names to infer task names
- `[worker_caps]` — hard maximums: `max_gpu_workers`, `max_process_workers`,
  `max_thread_workers` (all 8)
- `[memory]` — `default_base_mb` (4000), `loading_overhead` (1.5)
- `[command_base_mb.process]` — per-command budgets for process workers (GIL)
- `[command_base_mb.threaded]` — per-command budgets for thread workers

When a model changes size, update the corresponding `command_base_mb` entry.
No code changes are needed — the TOML value propagates automatically.

## sysinfo macOS limitation

`sysinfo::available_memory()` on macOS returns only free + purgeable pages.
It excludes "inactive" pages (file-backed pages the kernel could reclaim under
pressure). This means the auto-tuner undercounts available memory on macOS,
leading to conservative worker counts. This is intentional — better to
undercount than OOM.

## Warmup internals

### WarmupStatus state machine

```
NotStarted → InProgress → Complete
```

Tracked via `AtomicU8` on the pool. Reported in the `/health` response as
`"warmup_status": "not_started" | "in_progress" | "complete"`.

### Synchronous vs background warmup

Two entry points in `server.rs`:

- **`prepare_workers()`** — probes capabilities, then runs warmup synchronously
  (all commands spawn concurrently within a `JoinSet`, but the function blocks
  until all finish). Used by tests and the `create_app()` path.

- **`prepare_workers_background()`** — probes capabilities, then spawns warmup
  in a `tokio::spawn` background task. The function returns immediately after
  probing. Used by `create_app_with_runtime()` (the server startup path) so
  the HTTP port binds without waiting for models.

### Concurrent warmup

Within `pool.warmup()`, each command spawns as a separate `JoinSet` task. This
means `morphotag`, `align`, and `transcribe` warmup workers load their models
in parallel rather than sequentially. Each task:

1. Resolves the `WorkerTarget` for the command
2. Gets or creates the `WorkerGroup`
3. Claims a slot via atomic `compare_exchange`
4. Acquires the bootstrap lock (serializes per-key, but different keys proceed
   in parallel)
5. Spawns `WorkerHandle::spawn()` with the appropriate config
6. Pushes the handle to the idle queue and releases a semaphore permit

### Probe worker

`detect_capabilities()` spawns a temporary "probe" worker to discover what
commands the Python environment supports. It queries `capabilities()` via the
worker protocol, then shuts down the probe. This runs before warmup so the
server knows which warmup commands to skip.

### No-duplicate guarantee

If a job arrives for `(fa, eng)` while warmup is still spawning that same
worker, the job's `checkout()` call will wait on the semaphore. When the
warmup spawn finishes, it adds a permit and the job acquires it — no duplicate
spawn. This falls out naturally from the semaphore + bootstrap lock design.

## File processing order

Files within a job are currently processed in submission order. Workers are
dispatched via a `JoinSet` with a `Semaphore(num_workers)` limiting concurrency.

A potential future improvement: sort files largest-first before dispatch so
the longest files start processing immediately, reducing straggler effects.
The discovery layer already sorts by file size, but this could be made
explicit in the dispatch path.

See also:
- [Pipeline System](pipeline-system.md) — dispatch shapes and command lifecycle
- [Worker Tuning](../user-guide/worker-tuning.md) — user-facing configuration
