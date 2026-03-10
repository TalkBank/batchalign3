# Worker Tuning

**Status:** Current
**Last updated:** 2026-03-17

This page explains how the server decides how many workers to run, how memory
budgets work, and how to configure warmup and tuning for your hardware.

## How auto-tuning works

When you submit a job, the server decides how many parallel workers to use for
that job's files. The formula is:

1. Look up the command's per-worker memory budget (from `runtime_constants.toml`)
2. Multiply by the loading overhead factor (1.5×) to account for GC and buffers
3. Check available system RAM and CPU core count
4. Pick the minimum of: file count, cores, and `available_ram / budget`
5. Clamp to `[1, max_thread_workers]` (default max: 8)

For a single file, the server always uses 1 worker — no parallelism needed.

If `max_workers_per_job` is set in `server.yaml`, it overrides auto-tuning
(still capped by file count and the hard max).

## Per-command memory profiles

Each command loads different ML models with different memory footprints. These
values come from `runtime_constants.toml` (the single source of truth shared
between Rust and Python):

| Command | Memory per worker (MB) | What drives it |
|---------|----------------------|----------------|
| `morphotag` | 2,000 | Stanza POS/lemma/depparse models (per language) |
| `align` | 4,000 | Whisper or Wave2Vec forced alignment model |
| `transcribe` | 1,500 | Whisper ASR model |
| `utseg` | 2,000 | Stanza constituency parser |
| `translate` | 4,000 | Translation model (Seamless M4T or Google) |
| `coref` | 2,000 | Stanza coreference model |
| `opensmile` | 500 | Lightweight feature extractor |
| `avqi` | 1,500 | Voice quality analysis |
| `compare` | 2,000 | Stanza models (for normalization) |

These are the *thread worker* values (shared-model mode). Process worker values
are higher because each worker loads its own copy of the models.

## Warmup configuration

### The `--warmup` flag

Control which models are pre-loaded at server startup:

```bash
# Presets
batchalign3 serve start --warmup off           # No warmup — workers spawn on first job
batchalign3 serve start --warmup minimal        # Morphotag only
batchalign3 serve start --warmup full           # Morphotag + align + transcribe (default)

# Explicit command list
batchalign3 serve start --warmup align          # Only forced alignment
batchalign3 serve start --warmup morphotag,align  # Both morphotag and align
```

Without `--warmup`, the server uses the `warmup_policy` and `warmup_commands`
from `server.yaml`, defaulting to `full`.

### server.yaml warmup key

```yaml
# List of commands to pre-warm at startup.
# Default: [morphotag, align, transcribe] (the "full" preset)
# Empty list = no warmup.
warmup_commands:
  - morphotag
  - align
```

The `--warmup` CLI flag overrides this config key.

### Background warmup

Warmup runs in the background — the HTTP port binds immediately. While models
are loading:

- The `/health` endpoint reports `"warmup_status": "in_progress"`
- Jobs that need a model still loading will wait for the warmup to finish
  (no duplicate worker spawns — the job reuses the in-progress warmup)
- Once complete, `/health` reports `"warmup_status": "complete"`

All warmup commands load concurrently (not sequentially), so total warmup time
is roughly the time for the slowest model, not the sum of all models.

### On-demand loading

With `--warmup off`, no workers are pre-loaded. Workers spawn lazily on the
first job that needs them. This is ideal for:

- Testing and development
- Users who only run one command type
- Memory-constrained machines where you don't want idle model overhead

## server.yaml reference

Key tuning parameters:

```yaml
# Worker parallelism
max_workers_per_job: 0          # 0 = auto-tune based on RAM and CPU
max_concurrent_jobs: 0          # 0 = auto-tune (roughly 1 slot per 25 GB)

# Memory gate — minimum available RAM (MB) to start a new job
# 0 = disable. Default: 2048
memory_gate_mb: 2048

# Worker lifecycle
worker_idle_timeout_s: 600      # Shut down idle workers after 10 minutes
worker_health_interval_s: 30    # Health check frequency

# Warmup — list of commands to pre-load at startup
# Default: [morphotag, align, transcribe]
# Empty list disables warmup entirely.
warmup_commands:
  - morphotag
  - align
  - transcribe
```

## Scenarios

### 16 GB laptop

```yaml
max_workers_per_job: 1
memory_gate_mb: 2048
warmup_policy: minimal          # Only morphotag — saves ~4 GB
worker_idle_timeout_s: 300      # Free memory faster
```

Or start with no warmup:

```bash
batchalign3 serve start --warmup off
```

### 32 GB desktop

Default settings work well. Auto-tuning will typically run 2-3 parallel workers
for memory-heavy commands and more for lightweight ones.

### 256 GB server (production)

```yaml
max_workers_per_job: 0          # Auto-tune — will pick 4-8 workers
max_concurrent_jobs: 8
warmup_policy: full
worker_idle_timeout_s: 1800     # Keep workers loaded longer
```

### Testing with --warmup off

For quick iteration during development:

```bash
batchalign3 serve start --warmup off --foreground --test-echo
```

Workers start instantly (no ML models loaded). Useful for testing server
infrastructure without waiting for model initialization.

## Troubleshooting

### "Job deferred due to memory pressure"

The memory gate detected insufficient RAM. Possible causes:

1. **Too many concurrent workers.** Reduce `max_workers_per_job` or
   `max_concurrent_jobs` in `server.yaml`.
2. **Other processes using RAM.** Check system memory usage.
3. **Idle workers holding memory.** Workers that haven't been used in a while
   still hold their loaded models. Reduce `worker_idle_timeout_s` to free
   them sooner, or restart the server.

The memory gate has a 60-second timeout. If memory doesn't recover, the job
fails with a `MemoryPressure` error. When idle workers exist for the job's
command, the memory gate is bypassed (those workers are already loaded).

### Only 1 worker running

Auto-tuning decided that only 1 worker fits in available memory. Check:

- `command_base_mb` for your command in `runtime_constants.toml`
- Available system RAM (the server uses `sysinfo::available_memory()`)
- On macOS, `available_memory()` undercounts by excluding inactive pages

Override with `max_workers_per_job` if you know your system can handle more.

### Startup takes too long

Warmup loads ML models from disk (or downloads them on first run). To speed up:

- Use `--warmup minimal` or `--warmup off` if you don't need all commands
- The first run after installation is slowest (model downloads)
- Subsequent starts load from the model cache (~5-20 seconds per model)
- Keep the daemon running (`batchalign3 serve start`) to avoid repeated
  cold starts

See also [Performance](performance.md) and [Server Mode](server-mode.md).
