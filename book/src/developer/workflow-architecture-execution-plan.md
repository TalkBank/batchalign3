# Workflow Architecture Execution Plan

**Status:** Historical
**Last modified:** 2026-03-26 14:05 EDT

This page records the execution plan for the now-completed intermediate
workflow-layer refactor. It is preserved for architecture history, but the live
code has since moved fully to command-owned entrypoints and the old
`src/workflow/` tree has been deleted.

The goal is not to rewrite batchalign3 all at once. The goal is to let the new
architecture shape the code that is already being refactored, while recording
the remaining work as a durable resumable program.

The first explicit workflow-trait slice is now in code, centered in
[`crates/batchalign-app/src/workflow/mod.rs`](/Users/chen/batchalign3-rearch/crates/batchalign-app/src/workflow/mod.rs).
That top-level module now points readers at
[`crates/batchalign-app/src/workflow/traits.rs`](/Users/chen/batchalign3-rearch/crates/batchalign-app/src/workflow/traits.rs)
for the workflow family contracts and
[`crates/batchalign-app/src/workflow/registry.rs`](/Users/chen/batchalign3-rearch/crates/batchalign-app/src/workflow/registry.rs)
for the released command catalog.
This page therefore separates what is stable enough for contributors from what
should still be allowed to churn.

## Current Constraints

The plan has to respect the codebase as it exists today:

- `batchalign3` already has real runtime complexity: persistent workers, memory
  gating, file-level concurrency, GPU sharing, and cross-file batching.
- the refactor stream is already busy with trust-boundary cleanup, `pyo3`
  thinning, worker/process hardening, and test architecture work.
- commands do not all share one runtime shape, so the next architecture must
  improve reuse without forcing fake uniformity.

That means the execution strategy must be incremental.

## Stable Now

These are the seams contributors can build against now.

- `crates/batchalign-app/src/workflow/mod.rs`
- `crates/batchalign-app/src/workflow/traits.rs`
- `crates/batchalign-app/src/workflow/registry.rs`
- `PerFileWorkflow`
- `CrossFileBatchWorkflow`
- `ReferenceProjectionWorkflow`
- `CompositeWorkflow`
- `Materializer`

Concrete examples:

- `transcribe`: [`crates/batchalign-app/src/workflow/transcribe.rs`](/Users/chen/batchalign3-rearch/crates/batchalign-app/src/workflow/transcribe.rs)
- `align`: [`crates/batchalign-app/src/workflow/fa.rs`](/Users/chen/batchalign3-rearch/crates/batchalign-app/src/workflow/fa.rs)
- `morphotag`: [`crates/batchalign-app/src/workflow/morphosyntax.rs`](/Users/chen/batchalign3-rearch/crates/batchalign-app/src/workflow/morphosyntax.rs)
- `compare`: [`crates/batchalign-app/src/workflow/compare.rs`](/Users/chen/batchalign3-rearch/crates/batchalign-app/src/workflow/compare.rs)
- `benchmark`: [`crates/batchalign-app/src/workflow/benchmark.rs`](/Users/chen/batchalign3-rearch/crates/batchalign-app/src/workflow/benchmark.rs)

## Immediate Tranche

These changes remain active, but the shape is now narrower and more concrete.
The registry and family traits stay stable while the surrounding orchestration
continues to move.

### 1. Compare becomes the first typed workflow-bundle exemplar

Why now:

- the later `batchalign2-master` compare redesign is the clearest signal that future workflows
  will need multiple input artifacts and multiple output materializations.
- compare already has a natural internal bundle shape, but it still needs to be
  described and surfaced that way in code.

Current state:

- compare bundle construction is explicit in Rust
- the bundle now carries compare views, metrics, and structural word-match
  metadata
- output materialization is separated from bundle construction
- the released output shape now follows `batchalign2-master compare`:
  projected reference CHAT plus CSV metrics
- an internal main-annotated materializer still exists for benchmark-style flows

Immediate goals:

- preserve released compare behavior at the BA2-master surface
- keep alternate internal materializers explicit instead of implicit

Acceptance criteria:

- compare bundle construction is a named internal concept
- projected-reference CHAT output is one explicit materialization step
- CSV metrics output is one explicit materialization step
- the compare CLI/server behavior follows the BA2-master surface

### 2. Keep shrinking `pyo3`

Why now:

- build time, installability, and architectural clarity all depend on this
- `uv tool install batchalign3` will remain fragile until the native boundary is
  much thinner

Immediate goals:

- keep moving shared contracts into small Rust crates
- keep removing application-level dependencies from `pyo3`
- keep Python focused on provider-host responsibilities

Acceptance criteria:

- each refactor wave should reduce or narrow the reason `pyo3` depends on a
  higher-level crate
- new workflow logic should not be added to `pyo3`

### 3. Introduce smarter local verification by dependency fanout

Why now:

- current compile/test cost is slowing the refactor stream materially

Immediate goals:

- affected-package local checks by default
- full gates remain explicit and in CI
- use that machinery to expose architecture fallout quickly

Acceptance criteria:

- fast local entrypoints exist and are documented
- full local gates remain available
- refactor fallout is detected through the dependency-aware paths

## Still Churning

These areas are still expected to move while the contributor seam stays stable.

### 1. Expand compare into multi-materializer support

The current AST-first projection is intentionally conservative. Future work may
add:

- chunk-safe partial `%gra` / `%wor` projection
- richer debugging/alignment bundles
- additional compare materializers for alternate output shapes

### 2. Formalize workflow-family executors beyond the first wrappers

The first typed wrappers are in place, but reusable family harnesses for:

- `opensmile`
- `avqi`
- `utseg`
- `translate`
- `coref`

still need to be extracted from their current dispatcher modules.

### 3. Move command metadata toward typed workflow specs

The CLI and runner tables still need a deeper metadata pass before they stop
being the primary source of command shape.

### 4. Deepen the test-architecture refactor

The test system still contains bootstrap-era assumptions and expensive broad
gates. This work should continue, but it should not block the contributor seam.

## Near-Term Tranche

These should follow once the immediate tranche is stable.

### 1. Formalize workflow-family executors

Target families:

- per-file transform
- cross-file batch transform
- reference projection workflow
- composite workflow

Immediate implementation strategy:

- do not replace all current dispatch modules at once
- instead, extract shared harnesses from the existing family members

Candidate code areas:

- `crates/batchalign-app/src/runner/dispatch/fa_pipeline.rs`
- `crates/batchalign-app/src/runner/dispatch/transcribe_pipeline.rs`
- `crates/batchalign-app/src/runner/dispatch/benchmark_pipeline.rs`
- `crates/batchalign-app/src/runner/dispatch/media_analysis_v2.rs`
- `crates/batchalign-app/src/utseg.rs`
- `crates/batchalign-app/src/translate.rs`
- `crates/batchalign-app/src/coref.rs`

Acceptance criteria:

- at least one per-file harness shared by several commands
- at least one cross-file batch harness shared by several commands
- command-specific files focus on workflow semantics, not repeated runner glue

### 2. Move command metadata toward typed workflow specs

Why:

- command metadata is still duplicated across CLI match arms, option builders,
  and runner tables

Candidate code areas:

- `crates/batchalign-cli/src/args/mod.rs`
- `crates/batchalign-cli/src/lib.rs`
- `crates/batchalign-app/src/runner/mod.rs`

Acceptance criteria:

- new workflow additions should not require touching several unrelated string
  tables
- command metadata becomes easier to reuse from CLI, server, and docs

### 3. Expand compare into multi-materializer support

After the initial bundle/materializer split lands, compare should grow toward:

- main-annotated output
- gold-projected output
- reusable debugging/alignment artifacts

This should happen only after the initial compare-bundle refactor is stable.

## Later Tranche

These items are important, but not part of the current immediate patch stream.

### 1. Full reference-projection family

Generalize beyond compare:

- reviewed transcript projection
- curated-reference alignment workflows
- future cross-document analysis passes

### 2. Typed workflow-planning layer

Introduce explicit planning artifacts before execution:

- request -> plan
- plan -> family executor
- executor -> typed artifact bundle
- materializer -> user-facing outputs

### 3. Deeper test-architecture refactor

The test-system work should eventually mirror the workflow architecture:

- per-family verification
- artifact-invariant tests
- property tests on typed bundles and transformations
- less mock-heavy command-surface testing

### 4. Packaging/install architecture cleanup

This remains release-blocking but is orthogonal enough that it should not be
entangled with the first compare/workflow-family steps.

## Resume Checklist

If work pauses and later resumes, restart from these questions:

1. Is compare still preserving the released output shape while leaving the gold-projection seam visible?
2. Has `pyo3` stayed on the fast extension-only path for day-to-day iteration?
3. Which dispatch family has the most duplicated execution harness now?
4. Which command metadata table should be eliminated next?
5. Which piece of the test architecture is still forcing broad expensive checks
   for local iteration?

## Relationship To Other Docs

- [Hybrid Workflow Architecture](../architecture/hybrid-workflow-architecture.md)
  describes the target design.
- [Architecture Audit](architecture-audit.md) records current active structural
  issues.
- [Performance and Re-Architecture Backlog](performance-and-rearchitecture-backlog.md)
  records longer-horizon opportunities that are not active implementation work.

This page should stay practical. When a tranche becomes implemented and stable,
move the enduring architecture facts into the relevant architecture page and
remove or simplify the completed planning detail here.
