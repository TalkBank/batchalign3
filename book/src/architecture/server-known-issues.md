# Server Known Issues

**Status:** Current open issues only
**Last modified:** 2026-03-29 09:30 EDT
**Last verified:** 2026-03-29

## Zombie job resurrection on server restart

**Severity:** High — blocks new job submissions via 409 conflict.

**Symptom:** Cancelled or stuck jobs reappear as "running" after every
server restart. They consume workers and block new submissions for the
same files (409 conflict detection).

**Root cause:** SQLite job store persists jobs with `status=running`.
On startup, recovery re-queues any job that was running when the server
stopped. Cancelled jobs that were stuck in synchronous code (e.g., the
injection loop) never transitioned to `cancelled` in SQLite because the
cancellation token is only checked at async yield points.

**Workaround:** Cancel zombie jobs after restart:
```bash
curl -s http://localhost:8001/jobs | python3 -c "
import json,sys
for j in json.load(sys.stdin):
    if j['status'] in ('running','queued'):
        print(j['job_id'])" | while read jid; do
  curl -s -X POST "http://localhost:8001/jobs/$jid/cancel"
done
```
Or delete `~/.batchalign3/jobs.db` before restart (loses all job history).

**Fix needed:**
1. Recovery should not re-queue jobs that were cancelled
2. Consider `max_recovery_age` — don't recover jobs older than N minutes
3. Cooperative cancellation in synchronous loops

**Observed:** 2026-03-29, Net + Bilbo with both Temporal and embedded
backends. Jobs resurrected across 3+ server restarts.

## Tracing output lost in daemon mode (fixed 2026-03-29)

**Fixed in:** commit `faaaa31b`

`serve_cmd.rs` background path used `File::create()` which truncated
the server log on every restart. Changed to `OpenOptions::append()`.

## Ansible deploy kills running jobs

**Severity:** High — production jobs interrupted by routine deploys.

The `batchalign` Ansible role runs `batchalign3 serve stop` and
`pkill -9 batchalign-worker` on ALL targeted machines. It does not
check for active jobs before stopping.

**Workaround:** Check dashboard before deploying. Use `--limit` to
target specific machines. Never deploy to machines with active jobs.

**Fix needed:** The Ansible role should check for active jobs before
stopping (the `server` role does this, but the `batchalign` client
role does not).

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
- Keep `warmup_commands` configured in `~/.batchalign3/server.yaml` (the
  default is `["morphotag", "align", "transcribe"]`).
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

### 3. Large concurrent FA waves can mix worker-protocol collapse with SQLite write contention

**Symptom:** A large `align` job fails many files with `worker_protocol`
errors in a burst. Server logs show repeated lines like:

- `GPU worker: orphaned execute_v2 response ... request_id=fa-v2-request-0`
- `FA processing failed: worker protocol error: GPU worker response channel closed`
- `DB insert_attempt_start failed ... database is locked`

**Observed evidence:** This exact pattern was captured on `brian` on March 20,
2026 under `~/.batchalign3/daemon.log` for job `f0d498b5-ad1`.

**Current understanding:** Two issues can overlap here:

- old FA request IDs were not globally unique enough under shared concurrent GPU
  workers, which could produce orphaned response routing
- attempt persistence is still vulnerable to transient SQLite lock/busy failures
  during bursty per-file startup

**Current state:** Partially mitigated in the rearchitecture branch. The FA
request-correlation fix is already in that branch, and attempt-start
persistence now retries bounded SQLite lock/busy failures instead of failing on
the first transient lock. This still needs soak time before it can be
considered closed.
