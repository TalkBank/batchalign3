# Adding Inference Providers

**Status:** Current
**Last modified:** 2026-03-24 21:21 EDT

Batchalign3 no longer has a public entry-point plugin system. New engines are
added in-tree as built-in worker capabilities.

This page covers the current extension path.

If you are adding a new command or workflow, decide the workflow family first
and put the command's typed bundle or materializer in
`crates/batchalign-app/src/workflow/`. Engine work should support that
workflow; it should not define the workflow shape on its own.

## Choose the layer first

There are two different things you might be adding:

1. A new **worker-side inference backend** such as a new ASR or FA engine.
2. A new **server command** that needs Rust-side orchestration plus, optionally,
   a new worker inference task.

Most engine work starts in Python and only touches Rust for typed IPC contracts,
command registration, and server orchestration.

## Adding a worker-side inference backend

### 1. Add the inference module

Create a built-in module under `batchalign/inference/` that exposes a pure
inference helper consumed by a typed V2 worker host:

```python
from __future__ import annotations

from batchalign.worker._types_v2 import MyTaskItemV2, MyTaskResultItemV2


def infer_my_task(items: list[MyTaskItemV2]) -> list[MyTaskResultItemV2]:
    results: list[MyTaskResultItemV2] = []
    for item in items:
        results.append(MyTaskResultItemV2(ok=True))
    return results
```

Keep these modules CHAT-free. Python workers should accept structured payloads
and return structured results only.

### 2. Add or reuse the task identifier

If this is a new live infer task, add it in the V2 IPC type definitions:

- `batchalign/worker/_types_v2.py`
- `crates/batchalign-app/src/types/worker_v2.rs`

If you are only adding a new engine behind an existing task such as ASR or FA,
reuse the existing task and add only the new engine selector/state.

### 3. Load model state in the worker

Update `batchalign/worker/_model_loading/` so `load_worker_task()` can
initialize the new engine for the relevant infer task. This is where task-level
engine overrides are resolved and worker state is populated.

For existing command families, you usually update one of:

- `load_asr_engine()`
- `load_fa_engine()`
- `load_translation_engine()`
- `load_stanza_models()` in `worker/_stanza_loading.py`

### 4. Wire dispatch and capability advertisement

Update:

- `batchalign/worker/_execute_v2.py` to route the task or engine
- `batchalign/worker/_text_v2.py` if the task belongs to the shared batched
  text host
- `batchalign/worker/_handlers.py` to advertise `infer_tasks` and
  `engine_versions`

If the new engine is a variant of an existing task, keep the task stable and
report the engine version string through `engine_versions`.

**Capability gate (critical):** The `_capabilities()` function in `_handlers.py`
uses **import probes** to decide which infer tasks to advertise. If you add a new
`InferTask`, you must add it to the `_INFER_TASK_PROBES` dict with the tuple of
Python modules that must be importable:

```python
_INFER_TASK_PROBES: dict[InferTask, tuple[tuple[str, ...], str]] = {
    ...
    InferTask.MY_TASK: (("my_library",), "my-engine-v1"),
}
```

Capabilities are detected lazily from the first real worker spawn — there is no
dedicated probe worker at startup. The capability check uses import probes, not
loaded model state. This means capability advertisement must be based on import
availability, never on `_state.my_model is not None`. If you gate on loaded
model state, your task will not be advertised and the server will silently
exclude the command.

The Rust server cross-checks: commands whose required `InferTask` is not in the
worker's `infer_tasks` list are excluded from the server's advertised
capabilities. See [Capability Detection](../architecture/engine-interface.md#capability-detection)
for the full flow.

### 5. Register dependencies

Add the engine's Python dependencies to the appropriate section in
`pyproject.toml`:

- **Core engines** (expected to work out of the box): add to `dependencies`.
  All standard commands (align, transcribe, translate, morphotag, etc.) have
  their dependencies in `dependencies` so that `uv tool install batchalign3`
  gives users everything.

- **Built-in engines with extra runtime dependencies**: add them to
  `dependencies` if they are part of the supported built-in engine surface.
  Credential-gated or region-specific does not imply a separate install tier.

  Users then install `batchalign3[my-engine]`.

## Adding a new server command

If you are adding a new top-level command (not just a new engine for an
existing command), see the detailed 8-step checklist in
[Rust CLI and Server](rust-cli-and-server.md#adding-a-new-cli-command).

In addition to those Rust-side changes, update these Python-side surfaces:

1. **`batchalign/runtime_constants.toml`** — Add the command-to-task mapping
   (shared by Rust and Python at compile/import time).
2. **`batchalign/runtime.py`** — Add the command to `COMMAND_PROBES` with the
   tuple of Python modules that must be importable for the command to appear in
   `detect_capabilities()`.
3. **`batchalign/worker/_handlers.py`** — Add the `InferTask` to
   `_INFER_TASK_PROBES` so the worker advertises it. This must match the
   same dependencies used in `COMMAND_PROBES`. See
   [step 4 above](#4-wire-dispatch-and-capability-advertisement) for details.
4. **`batchalign/worker/_model_loading/`** — Register the dynamic
   runtime host for the new task if it depends on loaded model state or
   engine-specific wiring. Reserve **`batchalign/worker/_execute_v2.py`** for
   the small task router that dispatches to those prepared hosts.

Remember: command semantics live in the workflow layer, not in the worker
bootstrap layer. The worker layer should only know how to load engines and
execute typed tasks.

## Public extension surfaces that are still supported

These are the stable extension seams that still exist:

- `batchalign.pipeline_api`
  For repo-local direct Python execution outside the released worker protocol.
- `batchalign.providers`
  For legacy typed request/response models still used by compatibility code.
- optional dependency extras in `pyproject.toml`
  For shipping non-core engines without making them mandatory.

There is no supported `batchalign.plugins` discovery API and no
`PluginDescriptor` contract in the public release.

## Test expectations

At minimum, add:

- unit tests for the new inference module
- worker dispatch tests covering `_execute_v2()` or the relevant task host
- bootstrap/handler-registration tests if the task uses dynamic worker runtime
- Rust integration coverage if the new engine changes server orchestration,
  command routing, or capability gating
- doc updates for install syntax, command options, and migration notes if this
  replaces a BA2 or pre-release workflow

Relevant existing coverage lives in:

- `batchalign/tests/`
- `crates/batchalign-app/tests/`
- `crates/batchalign-cli/tests/`

## Rule of thumb

If the change affects CHAT structure, it belongs in Rust.

If the change affects model inference only, it usually belongs in Python plus
the typed worker contract that Rust consumes.
