# Testing

**Status:** Current
**Last updated:** 2026-03-20

## Philosophy

The test suite is split into two tiers by design:

1. **Fast tests** — unit tests, protocol tests, test-echo integration tests.
   No ML models, no GPU, no multi-GB processes. These run in seconds, fully
   parallel, on every edit. This is the inner development loop. It must stay
   fast and safe — a `cargo nextest run` should never crash your machine.

2. **ML tests** — golden snapshots, audio transcription, parity checks,
   profile verification. These spawn real Python workers that load Whisper,
   Stanza, pyannote, etc. Each worker consumes 2–5 GB RAM. They are slow,
   expensive, and dangerous on developer machines.

**ML tests are excluded by default.** You must opt in explicitly, and only
when you have a reason: a change to worker dispatch, a new language, an
inference module edit, a pre-release check. Never as part of routine
edit-compile-test.

This mirrors the Python side, where `uv run pytest` excludes `golden`,
`slow`, and `integration` markers by default.

### Why this matters

On 2026-03-19, two kernel OOM panics in one day were caused by ML test
binaries spawning concurrent Whisper workers during `cargo nextest run`.
Each golden test binary was a separate process that started its own server
with its own worker pool. Running them in parallel exhausted 64 GB of RAM.
See `docs/postmortems/` for incident details.

### Implemented solution: single binary

All ML tests are consolidated into one binary (`ml_golden`). One binary =
one process = one `LazyLock` = one `PreparedWorkers` = one set of loaded
models. Peak memory is ~8-12 GB (one pool) instead of 7x that.

The `LiveServerSession` fixture within the binary is well-designed:
- **One `PreparedWorkers` backend** shared across all 70 tests
- **Fresh HTTP server per session** (new port, new jobs dir, new SQLite)
- **Semaphore-gated sessions** so tests don't collide on control-plane state
- **Warm model cache** across tests — only the first test pays cold-start

### Defense-in-depth layers

These remain as additional safety nets:

| Layer | What | Catches |
|-------|------|---------|
| **nextest default-filter** | `ml_golden` excluded from `cargo nextest run` | Routine dev runs |
| **`ml` nextest profile** | ML tests serialized at max-threads=1 | Explicit opt-in |
| **Global worker cap** | `max_total_workers` (RAM / 4GB, max 32) | Multi-key pool explosion |
| **`WorkerPool::Drop`** | Kills idle workers when pool is dropped | Test cleanup on panic/exit |
| **PID file reaper** | `~/.batchalign3/worker-pids/` scanned on startup | Orphans from crashed servers |
| **Claude Code guard hook** | Blocks `cargo test`/`cargo nextest` if workers detected | AI assistant sessions |

## Quick reference

```bash
# Fast tests only (default — safe, parallel, no models)
cargo nextest run --workspace
make test

# ML tests only (serialized, one at a time)
cargo nextest run --profile ml

# Specific ML test (filter by submodule name)
cargo nextest run --profile ml -E 'binary_id(batchalign-app::ml_golden) & test(golden::)'

# Everything (fast + ML)
cargo nextest run --profile ml

# Python (fast only by default)
uv run pytest

# Python golden/integration
uv run pytest -m golden
uv run pytest -m integration
```

## Nextest configuration

The nextest config lives in `.config/nextest.toml`.

**Default profile:** applies a `default-filter` that excludes all ML test
binaries. `cargo nextest run` runs only fast tests. This is the safe
default.

**ML profile (`--profile ml`):** no default-filter exclusion, so all tests
run. ML binaries are assigned to the `ml` test group with `max-threads = 1`
to prevent concurrent model loading.

**Override the default filter for one run:**
```bash
cargo nextest run --ignore-default-filter -E 'binary_id(batchalign-app::ml_golden)'
```

All ML tests live in one binary (`ml_golden`) with submodules:

| Submodule | What | Models |
|-----------|------|--------|
| `golden` | Text NLP golden snapshots | Stanza |
| `golden_audio` | Audio transcription/alignment | Whisper, Wave2Vec, pyannote |
| `golden_parity` | Batchalign2 output parity | Stanza |
| `live_server_fixture` | Full server with live workers | Mixed |
| `profile_verification` | Worker pool profile grouping | Wave2Vec, Stanza |
| `option_receipt` | Option propagation differential tests | Stanza, Wave2Vec |
| `error_paths` | Graceful failure under live server | Mixed |

## Test categories

| Category | Tool | Command | Models | Runtime | Default |
|----------|------|---------|--------|---------|---------|
| Rust unit tests | cargo | `cargo nextest run --workspace` | None | ~5s | Yes |
| PyO3 unit tests | cargo | `cargo nextest run --manifest-path pyo3/Cargo.toml` | None | ~3s | Yes |
| Python unit tests | pytest | `uv run pytest` | None | ~2s | Yes |
| Worker protocol | cargo | `cargo nextest run --test worker_protocol_matrix` | None (test-echo) | ~5s | Yes |
| Server integration | cargo | `cargo nextest run --test integration` | None (test-echo) | ~5s | Yes |
| JSON compat | cargo | `cargo nextest run --test json_compat` | None | ~1s | Yes |
| ML tests (all) | cargo | `cargo nextest run --profile ml` | Mixed | ~5min | **No** |
| Python golden | pytest | `uv run pytest -m golden` | batchalign_core | ~10s | **No** |
| Python integration | pytest | `uv run pytest -m integration` | Worker | ~5s | **No** |
| HK engines | pytest | `uv run pytest batchalign/tests/hk/` | FunASR+ | ~2min | **No** |

## When to run ML tests

Run ML tests based on what changed, not as a habit:

| What you changed | Run |
|-----------------|-----|
| Rust unit logic (parser, DP, postprocess) | Fast tests only |
| Python inference module | `--profile ml` |
| Worker protocol or IPC types | `worker_protocol_matrix` (fast) + `--profile ml` |
| Worker pool, dispatch, or lifecycle | `--profile ml` |
| FA pipeline or UTR | `--profile ml` |
| Morphosyntax injection or retokenization | `--profile ml` |
| Pre-release or large refactor | Full `--profile ml` |
| Adding a new language | `--profile ml` |

## Python tests

```bash
uv run pytest                                           # Fast only
uv run pytest -m golden -v                              # Golden snapshots
uv run pytest -m integration -v                         # Integration
uv run pytest -m "golden or integration" -v             # Both
uv run pytest batchalign/tests/test_pipeline_api.py -v  # Specific file
```

If you changed `pyo3/` or shared Rust crates that feed `batchalign_core`,
rebuild the extension before running Python tests that import it:

```bash
make build-python   # or: uv run maturin develop
```

### Test doubles

Prefer explicit fake seams over `monkeypatch` when touching production code.
If a test needs to replace runtime behavior, the first question should be
whether the production boundary wants a typed injected dependency instead.

### Worker protocol V2 drift suite

```bash
uv run pytest batchalign/tests/test_worker_protocol_v2_types.py -q
uv run pytest batchalign/tests/test_worker_protocol_v2_artifacts.py -q
uv run pytest batchalign/tests/test_worker_fa_v2.py -q
cargo nextest run -p batchalign-app --test worker_protocol_v2_compat
cargo nextest run -p batchalign-app -E 'test(fa_result_v2)'
cargo nextest run -p batchalign-app --test worker_v2_fa_roundtrip
```

These tests read fixture files under `tests/fixtures/worker_protocol_v2/`
so the Rust and Python schema models stay aligned.

## Rust tests

```bash
# PyO3 extension
cargo nextest run --manifest-path pyo3/Cargo.toml

# Root workspace (fast tests only)
cargo nextest run --workspace

# Focused suites
cargo nextest run -p batchalign-cli --test cli
cargo nextest run -p batchalign-cli --test ci_checks
cargo nextest run -p batchalign-cli --test e2e
cargo nextest run -p batchalign-app --test integration
cargo nextest run -p batchalign-app --test json_compat
```

### Profile verification tests

`ml_golden/profile_verification.rs` exercises the worker profile architecture
under real model inference. Unlike golden tests (which verify output
correctness), these tests verify resource usage:

- **GPU profile sharing**: multi-file align uses a single `SharedGpuWorker`
- **Stanza profile grouping**: morphotag and utseg share one Stanza worker
- **Label regression guard**: all worker keys use `profile:*` prefix

Run with `cargo nextest run --profile ml`.

### ML test skip behavior

Model-gated tests use `require_live_server(InferTask::Xxx, "message")`:

1. Tries to acquire a `LiveServerSession` with a warm worker pool
2. Checks if the required InferTask is available (model installed)
3. Returns `None` (test silently skips) if models are unavailable

Python uses `@pytest.mark.skipif` or `pytest.skip()` for similar gating.

Even under `--profile ml`, tests skip gracefully if models are not
installed. You won't get false failures — just silent skips.

## Worker process safety

ML tests spawn Python worker subprocesses that load multi-GB models.
Several safeguards prevent runaway resource consumption:

**Global worker cap:** The `WorkerPool` enforces a hard ceiling on total
workers across all `(profile, lang, engine)` keys. Default: `available_memory / 4GB`,
capped at 32. Configurable via `max_total_workers` in `server.yaml` or
`PoolConfig`.

**Pool Drop:** `WorkerPool` implements `Drop` to kill all idle workers
synchronously, even when tests exit without calling `pool.shutdown()`.

**PID file reaper:** Each spawned worker writes a PID file to
`~/.batchalign3/worker-pids/{pid}` recording its parent server PID. On
pool startup, stale files (dead workers) are cleaned up and orphans
(live workers whose parent server is dead) are killed via
SIGTERM → 2s wait → SIGKILL.

## Dashboard Playwright tests

```bash
cd frontend
npm run e2e:install
npm run test:e2e
```

If Chromium has not been installed:

```bash
cd frontend
npm run test:e2e:setup
```

## Type checking

```bash
uv run mypy
# or together with clippy:
make lint
```

## CI hygiene

Release-facing CI checks cover:

- CLI/package version sync
- Stale legacy-term detection
- Retired package/path checks
- Command execution path integration coverage

```bash
cargo nextest run -p batchalign-cli --test ci_checks
make ci-local
```

## Coverage

There is a coverage workflow in `.github/workflows/test.yml` (manual
`workflow_dispatch`, not a release gate).

Current snapshot from 2026-03-16:

- Python: 90% across `batchalign/` (678 passed, 3 skipped, 27 deselected)
- Full inference adapter surface covered
- Remaining low-coverage: training, worker bootstrap, test helpers

```bash
# Python coverage (non-integration)
uv run --no-sync pytest -n0 --cov=batchalign --cov-report=term \
  --disable-pytest-warnings -m 'not integration' -q batchalign/tests

# Rust coverage
cargo llvm-cov nextest --manifest-path pyo3/Cargo.toml \
  --lcov --output-path lcov-rust.info
cargo llvm-cov nextest --workspace \
  --lcov --output-path lcov-rust-workspace.info
```
