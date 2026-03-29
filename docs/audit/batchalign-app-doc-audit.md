# batchalign-app Documentation Audit

**Status:** Current
**Last modified:** 2026-03-28 16:53 EDT

Audit of all Rust source files in `crates/batchalign-app/src/` for documentation
comment coverage and quality. Test files and pure re-export `mod.rs` files are
excluded.

**Methodology:** Automated scan of all 168 `.rs` files, checking for `//!`
module-level doc comments and `///` doc comments on all `pub` items (excluding
`pub(crate)`, `pub use`, and `pub mod` re-exports). Quality reviewed manually
for complex files (dispatch, runner, worker, FA, morphosyntax).

---

## Files Needing Documentation

### temporal_backend.rs
- **Module comment:** PRESENT (good -- explains Temporal alternate backend)
- **Missing pub docs:** `pub async fn run_job_attempt` (L129) -- Temporal activity method, undocumented
- **Notes:** This is a `#[activity]` method on `BatchalignTemporalActivities`. Should document what the activity does (runs a single job attempt via the existing execution engine), its retry semantics, and the `TemporalJobActivityInput`/`TemporalJobActivityOutcome` contract.

### worker/pool/mod.rs
- **Module comment:** PRESENT (excellent -- explains concurrency model, semaphore checkout, RAII guards)
- **Missing pub docs:** `pub enum WarmupStatus` (L90) -- missing top-level `///` doc comment on the enum, though all three variants (`NotStarted`, `InProgress`, `Complete`) are individually documented
- **Notes:** Adding a one-line `/// Lifecycle state of background model warmup.` above the derive block would complete coverage. The comment already exists as a regular `//` comment on L77 but is not a `///` doc comment.

---

## Files with Thin Module Comments (Technically Present, Could Be Richer)

These files pass the coverage check but have one-line module comments on
complex multi-hundred-line files. A newcomer reading these files would benefit
from a brief architectural overview in the module comment.

### runner/dispatch/fa_pipeline.rs (814 lines)
- **Module comment:** `//! Forced alignment dispatch and per-file FA pipeline.`
- **All pub items documented:** Yes
- **Notes:** This is the most complex dispatch pipeline. The module comment does not mention the multi-group FA architecture, UTR pre-pass with ASR result caching, partial-window ASR for mostly-timed files, or the fallback UTR retry after FA failures. All of these are documented in the crate-level CLAUDE.md but not in the source file itself.

### runner/dispatch/transcribe_pipeline.rs (534 lines)
- **Module comment:** `//! Transcription dispatch and per-file transcribe pipeline.`
- **All pub items documented:** Yes
- **Notes:** Does not mention the optional diarization, utseg, and morphosyntax sub-stages, or the difference between `transcribe` and `transcribe_s` commands.

### runner/dispatch/infer_batched.rs (347 lines)
- **Module comment:** `//! Batched text NLP dispatch (morphotag, utseg, translate, coref, compare).`
- **All pub items documented:** Yes
- **Notes:** Does not mention the cross-file pooling, per-language grouping, or semaphore-bounded concurrency model. These are the key architectural decisions in this module.

### morphosyntax/batch.rs (543 lines)
- **Module comment:** `//! Cross-file batch morphosyntax processing and cache helpers.`
- **All pub items documented:** Yes
- **Notes:** Does not describe the parse-clear-collect-cache-infer-inject-serialize pipeline or the two-level parallelism (cross-language + intra-language chunking).

---

## Skipped Files

### Pure Re-export mod.rs Files (4 files)
These contain only `mod` declarations and `pub use` re-exports with no logic:
- `recipe_runner/mod.rs`
- `revai/mod.rs`
- `runner/dispatch/mod.rs`
- `worker/mod.rs`

### Test Files (1 file)
- `runner/tests.rs` -- excluded per audit scope

---

## Well-Documented Files (162 files)

Every file below has both a module-level `//!` comment and `///` doc comments on
all `pub` items. Files are grouped by subsystem for readability.

**Root modules:** `benchmark.rs`, `command_family.rs`, `compare.rs`, `coref.rs`,
`debug_artifacts.rs`, `direct.rs`, `ensure_wav.rs`, `error.rs`, `host_memory.rs`,
`host_policy.rs`, `hostname.rs`, `infer_retry.rs`, `lib.rs`, `media.rs`,
`openapi.rs`, `queue.rs`, `runtime_paths.rs`, `runtime_supervisor.rs`,
`server.rs`, `server_backend.rs`, `stanza_registry.rs`, `state.rs`,
`submission.rs`, `text_batch.rs`, `trace_store.rs`, `translate.rs`, `utseg.rs`,
`websocket.rs`, `ws.rs`

**cache/:** `backend.rs`, `mod.rs`, `sqlite.rs`, `tiered.rs`

**commands/:** `align.rs`, `avqi.rs`, `benchmark.rs`, `catalog.rs`, `compare.rs`,
`coref.rs`, `kernel.rs`, `mod.rs`, `morphotag.rs`, `opensmile.rs`, `spec.rs`,
`transcribe.rs`, `translate.rs`, `utseg.rs`

**db/:** `insert.rs`, `mod.rs`, `query.rs`, `recovery.rs`, `schema.rs`, `update.rs`

**fa/:** `incremental.rs`, `mod.rs`, `transport.rs`

**morphosyntax/:** `mod.rs`, `worker.rs`

**pipeline/:** `mod.rs`, `morphosyntax.rs`, `plan.rs`, `text_infer.rs`, `transcribe.rs`

**recipe_runner/:** `catalog.rs`, `command_spec.rs`, `materialize.rs`, `planner.rs`,
`recipe.rs`, `runtime.rs`, `work_unit.rs`

**revai/:** `asr.rs`, `credentials.rs`, `preflight.rs`, `utr.rs`

**routes/:** `bug_reports.rs`, `dashboard.rs`, `health.rs`, `media_list.rs`,
`mod.rs`, `traces.rs`

**routes/jobs/:** `detail.rs`, `lifecycle.rs`, `mod.rs`, `stream.rs`

**runner/:** `debug_dumper.rs`, `mod.rs`, `policy.rs`

**runner/dispatch/:** `benchmark_pipeline.rs`, `compare_pipeline.rs`,
`fa_pipeline.rs`, `infer_batched.rs`, `media_analysis_v2.rs`, `options.rs`,
`plan.rs`, `transcribe_pipeline.rs`, `utr.rs`

**runner/util/:** `auto_tune.rs`, `batch_progress.rs`, `error_classification.rs`,
`file_status.rs`, `media.rs`, `mod.rs`

**store/:** `counters.rs`, `mod.rs`, `registry.rs`

**store/job/:** `mod.rs`, `types.rs`

**store/queries/:** `db_helpers.rs`, `dispatch.rs`, `execution.rs`,
`file_state.rs`, `lifecycle.rs`, `mod.rs`, `recovery.rs`, `runner.rs`

**transcribe/:** `asr_output.rs`, `infer.rs`, `mod.rs`, `types.rs`

**types/:** `api.rs`, `config.rs`, `domain.rs`, `engines.rs`, `mod.rs`,
`options.rs`, `params.rs`, `request.rs`, `response.rs`, `results.rs`,
`runtime.rs`, `scheduling.rs`, `status.rs`, `traces.rs`, `worker.rs`,
`worker_v2.rs`

**worker/:** `artifacts_v2.rs`, `asr_request_v2.rs`, `asr_result_v2.rs`,
`avqi_request_v2.rs`, `error.rs`, `fa_result_v2.rs`, `memory_guard.rs`,
`opensmile_request_v2.rs`, `provider_credentials.rs`, `python.rs`,
`registry.rs`, `request_builder_v2.rs`, `speaker_request_v2.rs`,
`speaker_result_v2.rs`, `target.rs`, `tcp_handle.rs`, `text_request_v2.rs`,
`text_result_v2.rs`

**worker/handle/:** `config.rs`, `mod.rs`

**worker/pool/:** `checkout.rs`, `execute_v2.rs`, `lifecycle.rs`, `reaper.rs`,
`shared_gpu.rs`, `status.rs`

---

## Summary

| Category | Count |
|----------|-------|
| Files with no module comment | 0 |
| Files with inadequate module comment | 0 |
| Pub items missing doc comments | 2 |
| Files with thin-but-present module comments (could be richer) | 4 |
| Files that are well-documented | 162 |
| Total files audited | 164 |
| Skipped (re-export mod.rs) | 4 |
| Skipped (test files) | 1 |

**Overall assessment:** This crate has exceptional documentation coverage.
162 out of 164 audited files (98.8%) have both module-level comments and
full `pub` item documentation. The crate also enables `#![warn(missing_docs)]`
at the crate root, so the compiler itself enforces documentation on public
items.

Only two concrete gaps exist:
1. `temporal_backend.rs` -- one `pub async fn` missing a doc comment
2. `worker/pool/mod.rs` -- one `pub enum` missing a top-level doc comment (variants are documented)

The four "thin module comment" files are technically compliant but would
benefit a newcomer if their one-line summaries were expanded to mention the
key architectural decisions documented in CLAUDE.md.
