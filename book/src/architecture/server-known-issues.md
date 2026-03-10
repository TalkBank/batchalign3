# Server Known Issues

**Status:** Current open issues only
**Last verified:** 2026-03-11

This page contains current open operational issues only.

## Open Issues

### 1. First-call deadlock on align (MPS from background thread)

**Symptom:** The first `align` job submitted to the server hangs indefinitely. The process shows 0% CPU, blocked on `pthread_cond_wait`. Subsequent runs after a server restart work fine because HuggingFace model weights are cached on disk.

**Root cause:** Historically, first-use MPS model initialization in background worker threads could deadlock on macOS.

**Current state:** Partially mitigated. The server now warms up key pipelines
(`morphotag`, `align`, `transcribe`) on the main thread during startup. This
reduces first-call deadlock risk significantly, but does not guarantee
elimination if warmup is disabled or a warmup load fails.

**Current mitigations:**
- Keep `warmup: true` in `~/.batchalign3/server.yaml` (default).
- If hangs still occur on a specific machine, retry with CPU-only (`--force-cpu` for CLI workloads) or isolate affected commands.
- If this becomes frequent in production, consider process isolation for the affected command path.

**Diagnosis tool:** If the server appears hung, sample the process:
```bash
sample <pid> -mayDie
```
Look for threads blocked on `lock_PyThread_acquire_lock` / `pthread_cond_wait`.

### 2. No run logs from server jobs

**Symptom:** `~/.batchalign3/logs/run-*.jsonl` files are empty or missing for server-processed jobs.

**Root cause:** Structured run logging (`run_log.py`) is tied to CLI dispatch paths. Server execution goes through `JobStore` worker threads/processes and updates `jobs.db` / API state, but does not emit CLI-style run logs.

**Status:** Open known limitation. Server errors are logged to server stderr (for example launchd log files), and per-job/per-file status is available via API/dashboard. Structured server-side run logs are not currently implemented.
