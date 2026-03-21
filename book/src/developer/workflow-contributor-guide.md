# Workflow Contributor Guide

**Status:** Active
**Last modified:** 2026-03-21 07:54 EDT

This is the shortest path for adding a new command, workflow, or engine
without fighting the refactor stream.

If you read code before prose, start at
[`crates/batchalign-app/src/workflow/mod.rs`](/Users/chen/batchalign3-rearch/crates/batchalign-app/src/workflow/mod.rs).
That file is now the map. It points you to:

- [`crates/batchalign-app/src/workflow/traits.rs`](/Users/chen/batchalign3-rearch/crates/batchalign-app/src/workflow/traits.rs)
  for the workflow family traits
- [`crates/batchalign-app/src/workflow/registry.rs`](/Users/chen/batchalign3-rearch/crates/batchalign-app/src/workflow/registry.rs)
  for the released command catalog
- the family-specific modules for the concrete implementations

For batch-oriented text commands, the important typed seams are:

- `TextBatchFileInput` for one named file plus its owned CHAT payload
- `TextBatchFileResults` for one batch's named file outcomes
- `TextWorkflowFileError` for a file-scoped failure that keeps the message
  separate from file identity

## Choose A Family

The registry already assigns released commands to one of these families, so
the first question is usually "which family am I extending?"

- Use `PerFileWorkflow` for a single-file transform.
- Use `CrossFileBatchWorkflow` when work is pooled across files.
- Use `ReferenceProjectionWorkflow` when two artifacts are jointly primary.
- Use `CompositeWorkflow` when you are composing existing workflows.
- Use `Materializer` when the hard part is output shape rather than execution.

## Current Examples

- `transcribe`: [`crates/batchalign-app/src/workflow/transcribe.rs`](/Users/chen/batchalign3-rearch/crates/batchalign-app/src/workflow/transcribe.rs)
- `align`: [`crates/batchalign-app/src/workflow/fa.rs`](/Users/chen/batchalign3-rearch/crates/batchalign-app/src/workflow/fa.rs)
- `morphotag`: [`crates/batchalign-app/src/workflow/morphosyntax.rs`](/Users/chen/batchalign3-rearch/crates/batchalign-app/src/workflow/morphosyntax.rs)
- `compare`: [`crates/batchalign-app/src/workflow/compare.rs`](/Users/chen/batchalign3-rearch/crates/batchalign-app/src/workflow/compare.rs)
- `benchmark`: [`crates/batchalign-app/src/workflow/benchmark.rs`](/Users/chen/batchalign3-rearch/crates/batchalign-app/src/workflow/benchmark.rs)

The first three are the simplest `PerFileWorkflow` examples.
`compare` is the reference-projection example, and `benchmark` is the
composite example that chains workflows while still keeping output materialized
through the workflow layer rather than CLI glue.

`transcribe_s` is the same per-file family as `transcribe`, but surfaced as
the diarized variant in the registry.

## Add A New Command

1. Add or extend the command descriptor in `crates/batchalign-app/src/workflow/registry.rs`.
2. Put the typed request bundle in `crates/batchalign-app/src/workflow/`.
3. Keep the trait implementation thin.
4. Keep the command-specific orchestration in the workflow layer, not in `pyo3`.
5. Keep runner/dispatch code focused on job lifecycle and queueing.

If the command batches text across files, prefer the
`TextBatchFileInput`/`TextBatchFileResults` seam over raw tuples at the
workflow boundary, and keep any file-local error detail in
`TextWorkflowFileError` rather than stringly return values.

## Add A New Engine

1. Keep provider selection at the control-plane boundary.
2. Keep engine-specific transport or worker protocol code in the provider or
   worker layer.
3. Add new typed payloads in a shared crate before widening the workflow API.

## Practical Rule

If a change makes `workflow/*` smaller and more legible, it is probably a real
improvement.
If it pushes orchestration back into `pyo3`, `cli`, or scattered dispatch
tables, it is probably the wrong direction.
