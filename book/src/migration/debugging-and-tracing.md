# Debugging and Tracing Migration

**Status:** Current
**Last updated:** 2026-03-17

## BA2 Baseline: No Principled Debugging Story

Batchalign2 (baseline commit `84ad500b`) had no structured debugging infrastructure:

- **Tiered `-v` console logging** (~100 scattered `L.info()` calls, 39 raw
  `print()` statements) via Python's `logging` module and Rich console formatting
- **Ephemeral console-only output** — no filesystem dumps, no structured traces,
  no per-stage instrumentation, no debug env vars, no timing breakdowns, no metrics
- Debugging meant: run with `-vvv`, read console, manually inspect I/O files

There were no debug artifacts, no offline replay capability, and no way to
reproduce a pipeline failure without re-running the ML models.

## BA3: Three-Tier Debugging Architecture

BA3 introduces a principled three-tier approach:

### Tier 1: Structured Logging (`tracing` crate)

Already shipped. See [Tracing and Debugging](../developer/tracing-and-debugging.md)
for the `-v`/`-vv`/`-vvv` verbosity system, engine boundary tracing, and
per-component instrumentation.

### Tier 2: `--debug-dir` for Reproducible Filesystem Dumps

The `--debug-dir PATH` CLI flag (or `BATCHALIGN_DEBUG_DIR` env var) enables
structured CHAT/JSON artifact dumps at each pipeline stage. This enables:

- **Offline TDD**: load fixture data, call pipeline functions, assert on output
  without running ML models
- **Test fixture generation**: debug artifacts from real pipeline runs become
  regression test inputs
- **Stage decomposition**: inspect intermediate state between UTR, FA grouping,
  and FA alignment

Directory layout for a file `sample.cha`:

```
debug-dir/
  sample_utr_input.cha         # CHAT before UTR injection
  sample_utr_tokens.json       # ASR timing tokens fed to UTR
  sample_utr_output.cha        # CHAT after UTR injection
  sample_utr_result.json       # UTR injection statistics
  sample_fa_input.cha          # CHAT before FA (after UTR)
  sample_fa_grouping.json      # FA group plan (time windows, words)
  sample_fa_group_0.json       # Per-group words + timings
  sample_fa_group_1.json
  sample_fa_output.cha         # Final aligned CHAT
```

### Tier 3: Dashboard Traces (`debug_traces`)

When `--debug-dir` is specified, `debug_traces` is automatically enabled on job
submissions. The server collects `FaTimelineTrace` data for each file and
exposes it via `GET /jobs/{id}/traces` for dashboard visualization.

## Example Workflow: Reproduce a UTR Failure

```bash
# 1. Run alignment with debug artifacts
batchalign3 align input/ output/ --lang eng --debug-dir /tmp/ba3-debug

# 2. Inspect the UTR input and tokens
cat /tmp/ba3-debug/sample_utr_input.cha
jq . /tmp/ba3-debug/sample_utr_tokens.json

# 3. Write a test that loads the fixtures and calls inject_utr_timing directly
# (no ML model needed — the tokens are already captured)
```

## Fine-Grained Cache Overrides

BA3 also introduces `--override-cache-tasks` for per-task cache control:

```bash
# Skip only UTR ASR cache (keep morphosyntax, FA caches)
batchalign3 align input/ output/ --override-cache-tasks utr_asr

# Skip UTR + FA caches
batchalign3 align input/ output/ --override-cache-tasks utr_asr,forced_alignment
```

Valid task names: `morphosyntax`, `utr_asr`, `forced_alignment`,
`utterance_segmentation`, `translation`.

The existing `--override-cache` flag continues to skip all caches.
