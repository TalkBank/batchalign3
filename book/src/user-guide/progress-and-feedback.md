# Progress and Feedback

**Status:** Current
**Last updated:** 2026-03-17

Batchalign reports real-time progress during processing. This page explains what
to expect for each command, what the progress indicators mean, and when to worry
versus when to wait.

## How Progress Works

Every processing job tracks progress at the **file level**. The server reports
stage transitions and optional sub-file counters (e.g., "Aligning 3/7 groups")
to all connected clients — the CLI, TUI, and React dashboard all consume the
same stream.

There are two progress tiers:

- **Stage labels** — every file shows a stage name ("Reading", "Aligning",
  "Writing") that changes as processing advances.
- **Sub-file counters** — some stages include a current/total counter for
  fine-grained progress within a single file.

## Per-Command Expectations

### align (forced alignment)

Align processes files individually and concurrently. Each file goes through:

1. **Reading** — loading the CHAT file from disk
2. **Resolving audio** — finding and preparing the media file
3. **Recovering utterance timing** (if needed) — re-transcribing to recover
   word timing for untimed utterances. Shows sub-progress for partial-window
   UTR (e.g., "2/5" windows). This step takes roughly as long as the
   recording itself.
4. **Aligning** — forced alignment on utterance groups. Shows sub-progress
   (e.g., "3/7" groups).
5. **Writing** — saving the aligned output

**Timing:** Most of the time is spent in steps 3-4. A 10-minute recording
typically takes 5-15 minutes depending on the engine and number of utterances.

### transcribe

Transcribe processes files individually. Each file goes through:

1. **Resolving audio** → **Transcribing** → **Post-processing** →
   **Building CHAT** → optional **Segmenting** / **Morphosyntax** →
   **Finalizing** → **Writing**

Shows a pipeline stage counter (e.g., "2/5") as each stage completes.

**Timing:** Rev.AI runs roughly in real-time. Whisper may take 2-5x the
audio length.

### morphotag, utseg, translate, coref (batched commands)

These commands batch **all files together** into a single inference call for
GPU efficiency. Progress stages:

1. **Reading** — files are loaded one at a time; each transitions from
   the initial stage to "Reading" during I/O.
2. **Analyzing/Segmenting/Translating** (0/N) — the batch total is published
   before inference starts. During inference, the progress bar shows the batch
   size but individual files don't advance.
3. **Writing** (1/N, 2/N, ...) — as each file's result is written to disk,
   the counter ticks up.

**What "frozen" means:** During step 2, the progress bar won't advance because
all files are processed as a single batch. This is normal — the model is
working on your entire corpus at once. The elapsed timer keeps ticking to
confirm the app is alive.

**Timing:** Depends on corpus size. 50 files typically takes 1-5 minutes for
morphotag, faster for translate and utseg.

## When to Worry vs. When to Wait

**Normal:** Progress frozen during batch processing, or during UTR/transcription
(these are genuinely long-running). The elapsed timer should always be ticking.

**Investigate if:**
- The elapsed timer stops advancing (app may have frozen — try refreshing)
- A file stays in "Reading" for more than 30 seconds (possible I/O issue)
- "Resolving audio" persists for minutes (media file may be missing)

## How to Cancel

- **Desktop app:** Click the red "Cancel" button in the progress view
- **CLI:** Press `Ctrl+C` (graceful shutdown)
- **API:** `POST /jobs/{id}/cancel`

Cancellation is cooperative — the current file finishes its in-progress work
before the job stops.

## Progress Displays

| Client | What you see |
|--------|-------------|
| **Desktop app** | Progress bar + file list with stage labels and sub-counters |
| **CLI** | indicatif progress bar with file count and elapsed time |
| **TUI** | Per-file spinners with stage labels and sub-counters |
| **Dashboard** | Same as desktop app (shared React components) |
