# Plugin Architecture (Removed)

> **Status: Removed (March 2026).** The plugin system (`batchalign.plugins`,
> `PluginDescriptor`, `InferenceProvider`, `discover_plugins()`) was deleted.
> The only plugin that existed (`batchalign-hk-plugin`) was folded into the
> core repository as built-in engines with optional dependency extras.
>
> The original detailed design history is preserved in maintainer archives; the
> public docs keep only the migration outcome and the current extension pattern.

## Why It Was Removed

1. **Single consumer** — `batchalign-hk-plugin` was the only plugin. The
   discovery machinery existed for one package.
2. **Entry-point fragility** — `importlib.metadata.entry_points()` failed
   silently on broken packages, loaded wrong versions across environments, and
   was difficult to debug.
3. **Enum dispatch is safer** — `AsrEngine` and `FaEngine` enums provide
   compile-time exhaustiveness checking and clear error messages for missing
   engines.
4. **Optional extras solve the same problem** — `pip install
   "batchalign3[hk-tencent]"` is equivalent to installing a separate plugin
   but without the discovery overhead.

## Current Engine Extension Pattern

To add a new inference engine, use the built-in engine pattern documented in
[Adding Inference Providers](adding-engines.md). The pattern is:

1. Create a `(load_*, infer_*)` function pair in `batchalign/inference/`
2. Add an enum variant to `AsrEngine`, `FaEngine`, or the relevant engine enum
3. Wire the loader into `worker/_model_loading/`
4. Register the runtime handler during bootstrap in `worker/_model_loading/`
   and keep `worker/_infer.py` thin
5. Add optional dependencies as an extra in `pyproject.toml`

For a real-world example of this pattern, see the HK engines in
`batchalign/inference/hk/` and the
[HK/Cantonese Engine Architecture](../architecture/hk-cantonese-engines.md).

## Migration Guide for Existing Plugins

If you have an existing `batchalign.plugins` plugin, migrate it to a built-in
engine:

1. Move your `load_*` and `infer_*` functions into `batchalign/inference/`
2. Add an enum variant for your engine in `worker/_types.py`
3. Remove `pyproject.toml` entry points and `PluginDescriptor`
4. Add your dependencies as optional extras in batchalign's `pyproject.toml`
5. Update tests to use `monkeypatch` instead of mock-based plugin patching
