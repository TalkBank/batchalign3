# Postmortem: Morphotag Rerun Failures (2026-03-28/29)

**Date:** 2026-03-29
**Severity:** High — multi-day disruption, data loss on Brian's machine
**Author:** Claude Code session

## Summary

A 15,748-file morphotag rerun across all TalkBank data repos was
repeatedly disrupted by deploys killing the CLI process, zombie jobs
blocking new submissions, injection hangs on pathological files, and
a force-push that caused cascading merge conflicts fleet-wide. Brian's
uncommitted working edits were destroyed by a `git reset --hard` on
his machine.

## Timeline

### 2026-03-28 afternoon
- Morphotag rerun started on Bilbo, submitting from Net CLI
- 12 batches (6,000 files) completed successfully
- Batch 13 hung at 0/500 — stuck in language group dispatch

### 2026-03-28 evening
- Multiple deploys to Bilbo and Net killed the CLI process
- CLI uses content mode (submit CHAT text over HTTP, download results)
- Files are written back one at a time as they complete
- But the CLI process dying mid-batch means remaining files in that
  batch are computed but never downloaded

### 2026-03-28 night
- Stale `--no-build` deploys pushed old wheel (no new code running)
- 2.7 GB SQLite cache with 1.2s per query blocked batch processing
- Cache wiped to unblock, but moka hot cache had stale entries too

### 2026-03-29 morning
- Cross-machine morphotag (net CLI → bilbo server) abandoned
- Switched to net-local morphotag (net CLI → net server)
- Injection hang recurred at 150/500 (same 52K-item pattern)
- Direct mode ran accidentally (no --server flag), wasting time
- Zombie jobs from SQLite recovery blocked new submissions (409)
- Force-push data repos from earlier incident caused merge conflicts
  on Brian's machine and study
- `git reset --hard` on Brian's machine destroyed his uncommitted
  working edits (Brian works all day without committing)

## Root Causes

### 1. Deploying kills running jobs
The Ansible batchalign role runs `batchalign3 serve stop` and
`pkill -9 batchalign-worker` unconditionally. No check for active
jobs. No protection for the CLI process submitting to the server.

### 2. Force-push caused cascading conflicts
The earlier corrupted-guard incident was "fixed" by force-pushing
data repos. This diverged every machine's local history from the
remote. Each machine needed manual recovery — and recovering Brian's
machine destroyed his uncommitted work.

### 3. Zombie job resurrection
SQLite-persisted running jobs re-queue on every server restart.
Cancelled jobs don't transition to cancelled in the DB when stuck
in synchronous code (injection loop). The Temporal backend
exacerbates this — the workflow state disagrees with the job store.

### 4. Injection hang on large files
A 52K-utterance window (25 files from CallHome corpus) hung for
hours in the injection phase. Root cause unknown — timing showed
normal files take 8-17ms. The rayon spawn_blocking made the hang
uncancellable. Rayon was reverted.

### 5. --no-build deploys silently stale
The deploy script's `--no-build` flag reuses the last-built wheel
without checking if it matches the current code. Multiple deploys
pushed code from hours ago while new fixes were committed locally.
Fixed with a stale-wheel guard (commit `ee6529b`).

### 6. No tracing output in daemon mode
All `tracing::warn!` output (timing, cache metrics, heartbeat
warnings) was silently discarded because the server log file was
truncated on every daemon restart. Fixed (commit `faaaa31b`).

## Impact

- ~10,700 of 15,748 files processed and pushed to GitHub
- ~4,000 files still need processing (running on Bilbo now)
- Brian's uncommitted working edits on macw@brian destroyed
- Brian's align jobs on Net interrupted multiple times
- Study had merge conflicts that needed manual resolution

## What We Fixed

1. Deploy stale-wheel guard (`ee6529b`)
2. Server log append mode (`faaaa31b`)
3. Proper window timeout (cancellable without rayon)
4. Rayon spawn_blocking reverted (was uncancellable)
5. Text cache off by default (was net-negative)
6. Per-phase timing instrumentation
7. Worker crash stderr capture
8. prek config added to missing data repos

## What Still Needs Fixing

1. **Zombie job resurrection** — recovery should not re-queue
   cancelled jobs. Needs architectural fix in job store + Temporal.
2. **Injection hang root cause** — unknown why 52K-item batches hang
   for hours when timing shows 8-17ms for normal files. Need to
   reproduce with the specific CallHome files.
3. **Ansible deploy should check for active jobs** on client machines.
4. **Window sizing by utterance count, not file count** — prevents
   pathological windows with huge files.
5. **Delete API for Temporal jobs** — cancelled jobs can't be deleted
   because Temporal workflow state disagrees with job store.

## Lessons

1. **Never force-push.** Use `git revert`. Force-push breaks every
   machine that pulled and requires manual recovery that can destroy
   uncommitted work.
2. **Never `git reset --hard` on someone else's machine.** Check
   `git status` first. Brian works all day without committing.
3. **Never deploy during a production run.** Check the dashboard
   for active jobs first.
4. **Always full build when deploying.** `--no-build` is dangerous.
5. **Measure before optimizing.** Rayon was added to solve a
   problem that didn't exist (8-17ms injection), and created a worse
   problem (uncancellable hangs).
