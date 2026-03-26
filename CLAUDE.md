# CLAUDE.md

**Status:** Current
**Last modified:** 2026-03-26 07:32 EDT

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Batchalign is a Rust-primary language sample analysis (LSA) suite from the [TalkBank](https://talkbank.org/) project, with Python used exclusively as a stateless ML model server. It processes conversation audio files and transcripts in [CHAT format](https://talkbank.org/0info/manuals/CHAT.html), providing ASR, forced alignment, morphosyntactic analysis, translation, utterance segmentation, and audio feature extraction.

**Architecture:** Rust owns **all** logic — CHAT parsing, caching, validation, serialization, text normalization, DP alignment, WER computation, ASR post-processing, compound merging, number expansion, retokenization, and result injection. Python workers are stateless ML inference endpoints that load pre-trained neural models, receive structured data (words, audio paths), call ML libraries (Stanza, Whisper, etc.), and return raw model output. No CHAT text, no text processing, and no domain logic exists in Python.

**Locked de-Pythonization target:** keep subprocess workers. Python stays only
for direct ML/library calls and the thinnest worker-side glue needed to host
those calls. Rust owns config, orchestration, payload preparation, cache
policy, post-processing, validation, and all CHAT-aware semantics. Already
landed BA2 compatibility shims such as `batchalign.compat` are out of scope for
this wave: preserve them as migration aids, but do not use them to justify new
non-ML Python logic.

**Supported platforms:** Windows, macOS, and Linux. All code must build and run correctly on all three platforms. Release wheels are built for 5 targets: macOS ARM + Intel, Linux x86 + ARM, Windows x86. Platform-specific code paths (file paths, process spawning, GPU detection) must handle all three OSes.

The CHAT parsing layer is implemented in Rust (`batchalign_core`) for correctness and performance. The Python package is `batchalign3`, available on PyPI.

## Environment

The project uses `uv` exclusively for all Python operations — development, CI, and deployment. **`pip` is banned.** Never use `pip install`, `python -m pip`, or `actions/setup-python` anywhere.

- **Install deps:** `uv sync --group dev` (not `pip install -e ".[dev]"`)
- **Run commands:** `uv run pytest`, `uv run mypy` (not `python -m pytest`)
- **CI:** `astral-sh/setup-uv@v6` with `enable-cache: true` (not `actions/setup-python`)
- **One-off tools:** `uvx pip-audit` (not `pip install pip-audit`)
- **Venv:** managed by `uv sync` — never create manually

## Running in Development

One-time setup after cloning:

```bash
make sync                    # Install Python deps + create venv
make build                   # Build PyO3 extension + CLI (debug) + dashboard
```

Day-to-day (FAST dev loop — use these, not `cargo test`):

```bash
make check                   # Compile check only (~6s incremental)
make test                    # Quick lib tests (~3s after first compile)
make dev-ready               # Rebuild PyO3 extension + CLI (debug)

# Run the CLI:
./target/debug/batchalign3 transcribe input/ output/ --lang eng
cargo run -p batchalign-cli -- transcribe input/ output/ --lang eng

# Release binary for large-scale work:
make build-release
./target/release/batchalign3 transcribe input/ output/ --lang eng
```

**What to rebuild after changes:**

| What changed | What to rebuild |
|-------------|-----------------|
| Python code only (`batchalign/`) | Nothing — picked up automatically by workers |
| Rust CLI (`crates/batchalign-cli/`, `crates/batchalign-app/`, etc.) | `cargo build -p batchalign-cli` or `make build-rust` |
| PyO3 bridge or chat-ops (`pyo3/`, `crates/batchalign-chat-ops/`) | `make build-python` (rebuilds `batchalign_core` native extension) |
| REST API types (`crates/batchalign-app/src/types/`, `crates/batchalign-types/`, `ToSchema` derives) | `bash scripts/generate_dashboard_api_types.sh` (regenerates `openapi.json` + `frontend/src/generated/api.ts`) |
| Worker IPC types (`types/worker_v2.rs`, `morphosyntax/mod.rs`, etc.) | `make generate-ipc-types` (regenerates `ipc-schema/` + `batchalign/generated/`) |
| Everything | `make build` |

**API schema discipline:** When you modify any Rust struct with `#[derive(ToSchema)]` or change route signatures in `routes/`, you **must** run `bash scripts/generate_dashboard_api_types.sh` and commit the regenerated `openapi.json` and `frontend/src/generated/api.ts`. CI will fail on API drift otherwise. Verify with `bash scripts/check_dashboard_api_drift.sh`.

**IPC type discipline:** When you modify any Rust struct with `#[derive(schemars::JsonSchema)]` that crosses the Python worker boundary, you **must** run `make generate-ipc-types` and commit the regenerated `ipc-schema/` and `batchalign/generated/` files. `make ci-local` and `make check-ipc-drift` verify schemas are current. See `book/src/developer/ipc-type-sync.md` for the full workflow and `book/src/developer/ipc-union-migration.md` for the roadmap to full codegen.

## Test Tiers — Read This Before Running Tests

Full testing guide: `book/src/developer/testing.md`

Tests are split into tiers by safety and speed. **NEVER run bare `cargo test`.**
See `docs/memory-safety.md` for the crash history and full rationale.

```bash
# FAST (every edit) — pure Rust, no Python, no ML, no OOM risk
make check                   # Compile check only (~6s)
make test                    # Library tests (~3s incremental)

# EXTENDED (before pushing) — still safe, no ML models
make test-rust               # Library + safe integration tests (~5s)
make lint-rust               # Clippy on key crates only (~10s)

# WORKER TESTS (when touching worker code) — spawns test-echo Python workers
make test-workers            # Serialized, memory-guarded

# SLOW (before releases / on CI)
make lint                    # Full clippy + mypy
make test-python             # pytest suite

# ML GOLDEN TESTS — ONLY on net (256 GB RAM)
make test-ml                 # Loads real Whisper/Stanza models

# DANGEROUS — NEVER RUN THESE:
# cargo test                 # Runs ALL binaries in parallel → OOM crash
# cargo test -p batchalign-app --tests  # Same problem
# cargo nextest run          # Same problem
```

### Diagram Authoring Rules

**Architecture and design documentation MUST include Mermaid diagrams.**
GitHub renders Mermaid natively; all mdBook builds have `mdbook-mermaid` enabled.

#### When to Create a Diagram

Add a diagram when documenting:
- Data flow pipelines (how data transforms through stages)
- Architecture boundaries (what owns what, who calls whom)
- State machines and lifecycles (valid transitions, terminal states)
- Decision trees (option routing, engine selection, fallback paths)
- Type relationships (trait hierarchies, enum variants, ownership)
- Protocols (request/response sequences, IPC message flows)

**If a page describes a pipeline, boundary, or decision flow in prose
without a diagram, the page is incomplete.**

#### Diagram Type Selection

| Situation | Use | Not |
|-----------|-----|-----|
| Data flows through stages | `flowchart TD` or `flowchart LR` | `sequenceDiagram` (no named participants) |
| Request/response between components | `sequenceDiagram` | `flowchart` (hides back-and-forth) |
| Type hierarchies, trait impls | `classDiagram` | `flowchart` (wrong semantics) |
| State transitions, lifecycles | `stateDiagram-v2` | `flowchart` (no state semantics) |
| Decision trees, option routing | `flowchart TD` with diamond nodes | Text lists (hard to follow branches) |

#### The Seven Diagram Rules

These rules exist because a successor who has never met the team will
read these diagrams to understand the system. Every rule directly
addresses a documented failure mode that produces misleading diagrams.

**Rule 1: Name every resource.**
Every node must have a specific name AND its type/role.
Not `"Server"` — use `"Rust Server\n(batchalign-app)"`.
Not `"Cache"` — use `"moka hot cache\n(10k entries)"` or
`"SQLite cold cache\n(cache.db)"`.
A reader must be able to grep the codebase for the node label and find it.

**Rule 2: One concept per diagram.**
Each diagram tells one coherent story. If a page covers multiple
concerns (runtime ownership AND deploy topology AND protocol messages),
use separate diagrams for each. The `server-architecture.md` pattern
(4 focused perspectives on one system) is the model. When in doubt,
split.

**Rule 3: No conveyor belts for interactive flows.**
If two components exchange messages (request/response, IPC, HTTP),
use `sequenceDiagram` to show the actual back-and-forth. A `flowchart`
hides retry loops, error paths, and temporal ordering. Reserve
`flowchart` for genuinely one-directional data pipelines.

**Rule 4: Show real decision points.**
Decision diamonds must use real function names, flag names, and
condition expressions — not `"check condition"`. Example:
`{--before path\nprovided?}` not `{check input?}`. The
`command-flowcharts.md` align diagram is the gold standard.

**Rule 5: Include error and fallback paths.**
Every decision node must show what happens on failure. A diagram
showing only the happy path is misleading. Show retry logic, fallback
engines, cache misses, and error propagation. Mark optional paths
with dashed lines (`-.->`) and error paths with descriptive labels.

**Rule 6: Anchor to source locations.**
Architecture diagram nodes should include the crate, module, or file
path in the label or in prose immediately below. A reader should go
from diagram node to source file in one step. Example:
`"dispatch_fa_infer()\n(workflow/fa.rs)"`.

**Rule 7: Never generate diagrams from source code without verification.**
AI-generated diagrams from source code hallucinate components, invent
connections, and omit critical paths. When creating a diagram of
existing code:
1. Read the actual source files for every entity in the diagram
2. Verify every node corresponds to a real module, function, or type
3. Verify every arrow corresponds to a real call, dependency, or data flow
4. Cross-check against existing diagrams on the same or related pages
5. If you cannot verify a connection, omit it — gaps are better than lies
6. After writing the diagram, list in a comment the source files you
   verified against

**Diagram verification is not optional.** An unverified diagram is worse
than no diagram — it teaches a newcomer a wrong mental model that they
will carry forward and build upon.

#### Formatting Standards

- **Node labels:** `["Name\n(role or path)"]` for multi-line
- **Decision nodes:** `{"condition?\ndetail"}` diamond syntax
- **Edge labels:** `-->|"label"| target` for all non-trivial edges
- **Subgraphs:** Use only for ownership boundaries (e.g., separating
  batchalign3 crates from talkbank-tools crates in cross-repo diagrams)
- **Colors/styles:** Do not use custom colors. Default Mermaid themes
  ensure consistent rendering across GitHub and mdBook
- **Size limit:** Keep diagrams under 30 nodes. If larger, split into
  focused diagrams. The align command flowchart (~35 nodes) is at the
  practical upper limit
- **Angle bracket escaping:** Raw angle brackets in Mermaid labels
  (`Arc<str>`, `Vec<T>`, `&str`) trigger mdBook "unclosed HTML tag"
  warnings. Escape as `&lt;str&gt;` inside labels. Prefer quoted labels
  in diagrams and describe generic type arguments in surrounding prose
  or code spans. Rerun `mdbook build` after changes

#### Placement and Co-evolution

- Place each diagram **inline**, immediately after the prose paragraph
  that introduces the concept it illustrates
- Every diagram must have a prose introduction explaining what it shows
  and why the reader should care
- For complex topics, use the multi-perspective pattern: one overview
  diagram early, then focused detail diagrams in each subsection
- **When a change affects code structure, CLI options, data flow, or
  user-visible behavior**, update **both** the user-facing book pages
  (`book/src/user-guide/`, especially `cli-reference.md`) **and** the
  developer-facing architecture pages (`book/src/architecture/`) in the
  same change. Do not leave the book describing behavior that no longer
  exists or omitting new CLI options

`batchalign3` is a Rust binary (`crates/batchalign-cli`). Use `uv run` only for Python commands (pytest, mypy).

## Build and Test Commands

**ML tests are excluded by default.** `make test` runs only fast library tests. Worker and ML tests are opt-in via `make test-workers` and `make test-ml`. **Never use bare `cargo nextest run` on a developer machine** — it runs test binaries in parallel, which can spawn multiple ML workers and OOM-crash your machine. See `docs/memory-safety.md`.

```bash
# Fast tests only (default — no models, safe, parallel)
uv run pytest
cargo nextest run --workspace
make test                    # runs both

# ML tests (serialized, real models — run only when relevant)
cargo nextest run --profile ml

# Specific ML submodule
cargo nextest run --profile ml -E 'binary_id(batchalign-app::ml_golden) & test(golden::)'

# Python golden/integration
uv run pytest -m golden
uv run pytest -m integration

# Type checking (run before committing)
uv run mypy

# Rust tests (PyO3 crate)
cargo nextest run --manifest-path pyo3/Cargo.toml

# Build wheels for release packaging
uv build --wheel
uv run maturin build --release -i python3.12
```

## Code Quality Requirements

### Testing

**Red/green TDD is mandatory for all new features and bug fixes, no exceptions.**

1. **RED**: Write a failing test that specifies the desired behavior or reproduces the bug
2. **GREEN**: Write the minimum code to make the test pass
3. **REFACTOR**: Clean up while keeping all tests green

**Every bug report or discovered bug MUST start with a failing test.** Do not investigate or fix a bug without first writing a test that reproduces it. The test proves the bug exists, prevents regressions, and documents the expected behavior. Fixing before testing leads to incomplete fixes and missed edge cases.

**No mocks.** `unittest.mock` is banned — zero imports allowed anywhere in the test suite. Test doubles that are alternate implementations of a protocol are allowed. Shared doubles live in `batchalign/tests/doubles.py`.

**Prefer real ML inference over synthetic doubles.** Tests should call actual ML libraries and models, not fabricate synthetic output. Models are downloaded and cached automatically on first run (Stanza, PyCantonese, etc.). Synthetic test doubles hide bugs where the real library behaves differently from expectations. Use the `@pytest.mark.golden` marker for tests that load heavy ML models (Stanza, Whisper). Use synthetic doubles only when real inference is truly impossible (cloud API credentials, proprietary engines). When a feature is motivated by external claims (e.g., "ASR engine X produces per-character output"), write tests that verify the claim against real library behavior before building on it.

**OOM protection is enforced by code, not by convention.** The `conftest.py` autouse fixture `_guard_golden_oom` and `pytest_configure` hook automatically prevent golden/ML tests from running with parallel xdist workers on machines with < 128 GB RAM. This cannot be bypassed — each golden test checks its own safety before loading any models. See `batchalign/tests/conftest.py` for the implementation.

**Tree-sitter fragment parsing.** The tree-sitter parser supports parsing individual CHAT fragments — a single word, a main tier line, a %mor tier — without synthesizing a fake full CHAT document. Create a `TreeSitterParser` handle and call its `_fragment` methods: `parser.parse_word_fragment()`, `parser.parse_main_tier_fragment()`, `parser.parse_mor_tier_fragment()`, `parser.parse_gra_tier_fragment()`, etc. Do not wrap fragments in fake `@UTF8/@Begin/@End` scaffolding when a fragment parser exists. Create the parser once per test or entry point and reuse it.

**CHAT test content gotcha:** When you DO need a full CHAT document (e.g., testing file-level validation or cross-tier alignment), minimal valid CHAT needs `@Languages`, `@Participants`, AND `@ID` lines. Missing `@ID` causes `CHATValidationException: Encountered undeclared tier`.

### Type Annotations

All new and modified code **must** include type annotations:
- Function signatures: annotate all parameters and return types
- Use modern syntax (`list[str]` not `List[str]`, `str | None` not `Optional[str]`)
- **`Any` and `object` are banned as type annotations.** Use specific types. For ML library types that are expensive to import, use `TYPE_CHECKING` guards with the real type. For JSON payloads crossing IPC boundaries, use Pydantic models or `TypedDict`.

### Comments and Docstrings

All new and modified Rust, Python, and TypeScript code **must** carry
contributor-facing documentation comments.

- Every source file needs a header comment or module docstring explaining its
  architectural role.
- Every public/exported type and function needs a doc comment or docstring.
- Non-public helpers still need comments when their control flow, ownership
  model, or invariants are not obvious from the code alone.
- Comments must explain *why this boundary exists*, *what owns the state*, and
  *which invariants callers rely on*. Do not settle for comments that merely
  restate the next line of code.
- When refactoring architecture, update code comments in the same change so new
  contributors can follow the new seam without reading git history.

### Code Organization and Browsability

- **Types and traits are the first layer of documentation.** Favor named
  structs, enums, request/result types, and workflow traits over explaining raw
  primitives in comments.
- **Keep modules small and role-shaped.** Split catch-all files once they
  contain multiple workflows, ownership models, or unrelated helper families.
  A contributor should be able to find inference, workflow, and CLI seams
  quickly by directory layout.
- **Prefer methods when they clarify ownership.** If behavior depends on a
  type's invariants or owned state, keep it in an `impl`. Use free functions
  for adapters, symmetric transforms, and orchestration glue that does not have
  one natural owner.
- **Update touched docs with timestamps.** Any documentation file modified in a
  change must update its `Last modified` field with date and time. **Always run
  `date '+%Y-%m-%d %H:%M %Z'` to get the actual system time** — do not guess,
  hardcode, or use the conversation date.

### Boolean Blindness

- **No boolean blindness.** Enums over bools for anything beyond simple on/off. Banned: 2+ bool params, 2+ related bool fields, opposite bool pairs (`foo`/`no_foo`), ambiguous bool returns. Use `enum.Enum` or `typing.Literal["option1", "option2"]` for multi-way choices. OK as bool: `verbose`, `force`, `quiet`, single on/off flags where the name is self-documenting.

### Newtypes Over Primitives

- **No primitive obsession.** Domain values must have domain types. Function signatures should be self-documenting through type names, not parameter names.
- Use `typing.NewType` (e.g., `TimestampMs = NewType("TimestampMs", int)`) or Pydantic constrained types at module/IPC boundaries. For lightweight internal use, `type` aliases are acceptable when they clarify intent.
- Domain types already defined in `_domain_types.py`: `AudioPath`, `NumSpeakers`, `SpeakerId`, `TimestampMs`. Use these instead of bare `str`/`int`.
- Parse raw strings into typed values at the boundary (CLI args, IPC, file I/O). Interior code should never handle raw strings for typed values.
- New semantic strings introduced for intermediate artifacts or serializers must
  also be typed. Tier labels, token payloads, POS labels, CSV metric keys, and
  similar boundaries do not get to hide inside ad hoc `str` variables.
- **No ad-hoc format parsing.** Use real parsers (JSON: `json`, XML: `xml.etree`, etc.) not regex or string splitting for structured formats. Regex is for flat text only (normalization, search, validation).

### Boundary Hygiene

- **No new tuple-packed domain boundaries.** Do not introduce signatures like
  `(String, String)` or `Vec<(String, Result<String, String>)>` when the
  fields have stable meaning. Name the shape with a struct or newtype.
- **No new panic paths in server/runtime code.** Do not add `unwrap()`,
  `expect()`, or equivalent panic-based control flow in server, worker, CLI
  orchestration, persistence, or long-running background-task code.
- **Use real domain errors.** New domain error types should use `thiserror`
  instead of stringly `Result<_, String>` seams. If an old string error
  boundary must remain temporarily, isolate it at the outermost compatibility
  layer and do not copy it inward.

### Type Checking

Run **mypy** before committing changes (strict mode, configured in `mypy.ini`):
```bash
uv run mypy
```

### CHAT Format Handling — No Text Hacking

**All CHAT parsing and serialization MUST go through principled AST manipulation, never ad-hoc string/regex manipulation of raw CHAT text.** This is a hard rule.

- **Parsing**: Use `batchalign_core` Rust functions. These parse CHAT into a proper AST, manipulate it, and re-serialize correctly.
- **Never** use regex, string splitting, or line-by-line processing to extract or modify CHAT content. If `batchalign_core` doesn't expose the data you need, add a new Rust function.
- **Serializer-owned boundaries only**: if you need new CHAT tiers or other
  structured output, define explicit pre-serialization models first and lower
  them once at the boundary. Do not drive semantics from already serialized CHAT
  lines or user-defined tier text.
- **Parity is semantic, not textual**: when matching legacy behavior, compare the
  meaning and typed output model, not the legacy implementation shell. Do not
  copy `Document`/string architecture when the AST can express the intent.

## Rust Extension (`batchalign_core`)

Standard maturin mixed Python/Rust project.

- `pyo3/` — Single-crate project. `Cargo.toml` and `src/` live directly here.
- `batchalign_core/__init__.py` — re-exports from the native `.so`
- `stubs/batchalign_core/__init__.pyi` — type stubs for static analysis

**Rebuilding after Rust changes:** `uv run maturin develop`

**Running Rust tests:** `cargo nextest run --manifest-path pyo3/Cargo.toml` (nextest is the standard Rust test runner across this repo)

## Architecture

### Worker Architecture

Python workers are stateless ML inference endpoints spawned by the Rust server. Communication uses stdio JSON-lines IPC.

**Protocol:**
1. Rust server spawns: `python -m batchalign.worker --task morphosyntax --lang eng`
2. Worker loads models, prints `{"ready": true, "pid": N, "transport": "stdio"}`
3. Request/response loop over stdin/stdout (JSON-lines)

**Operations:**
- `infer` / `batch_infer` — pure inference for legacy-routed tasks (morphosyntax, utseg, translate, coref)
- `execute_v2` — typed worker-protocol V2 execution (fa, asr, speaker, opensmile, avqi, batched text tasks)
- `health` — worker status
- `capabilities` — available commands and infer tasks

**Worker modules:**
- `_main.py` — model loading and CLI entry point
- `_protocol.py` — stdio JSON-lines serving loop
- `_handlers.py` — health and capabilities handlers
- `_infer.py` — batch_infer dispatch router
- `_types.py` — Pydantic models mirroring Rust `batchalign-types::worker`

### Inference Modules (`batchalign/inference/`)

Each module is a pure inference function: receives structured input, runs ML model, returns structured output. No CHAT parsing, no pipeline orchestration.

| Module | Function | Input → Output |
|--------|----------|----------------|
| `morphosyntax.py` | `batch_infer_morphosyntax()` | words+lang → UD annotations (POS, lemma, depparse) |
| `utseg.py` | `batch_infer_utseg()` | words+lang → constituency parse + boundary assignments |
| `translate.py` | `batch_infer_translate()` | text+lang → translated text |
| `coref.py` | `batch_infer_coref()` | sentences → coreference chains |
| `fa.py` | `batch_infer_fa()` | audio+words → word-level timings |
| `asr.py` | `infer_whisper_prepared_audio()` and provider helpers | prepared waveform or provider media → raw ASR results |
| `speaker.py` | `batch_infer_speaker()` | audio path → speaker diarization segments |
| `opensmile.py` | `extract_features()` | audio path → acoustic features |
| `avqi.py` | `calculate_avqi()` | paired .cs/.sv audio → voice quality index |
| `benchmark.py` | `compute_wer()` | Python convenience wrapper over Rust WER scoring |
| `types.py` | — | Shared type aliases (StanzaNLP, etc.) |

### Dispatch Flow

```
Rust CLI (batchalign3) → Rust Server (crates/)
  ├── Parse CHAT (batchalign-chat-ops)
  ├── Check cache (tiered: moka hot → SQLite cold)
  ├── Call Python worker: batch_infer(task, lang, items)
  │     └── Worker routes to inference module → returns structured results
  ├── Inject results into CHAT AST (batchalign-chat-ops)
  ├── Update cache
  └── Serialize CHAT
```

**Key principle:** Python workers never see CHAT text. They receive extracted words/audio and return structured annotations. The Rust server owns all CHAT parsing, caching, validation, and serialization.

### CLI Command Dispatch (Single Source of Truth)

**`batchalign_cli::run_command()`** in `crates/batchalign-cli/src/lib.rs` is the single canonical command router. The standalone binary (`main.rs`) calls it. The Python console_scripts entry point (`batchalign/_cli.py`) execs the binary directly. **Never duplicate this match block.** If you need to add a new CLI command:

1. Add the `Commands::Foo` variant to `crates/batchalign-cli/src/args/mod.rs`
2. Add the match arm in `run_command()` in `crates/batchalign-cli/src/lib.rs` — this is the only dispatch site
3. If the command needs server-side orchestration, register a `WorkflowDescriptor` in `crates/batchalign-app/src/workflow/registry.rs` (this drives `infer_task_for_command()` and `command_requires_infer()` in `crates/batchalign-app/src/runner/policy.rs`)
4. If it uses the batched infer path, add routing in `crates/batchalign-app/src/runner/dispatch/infer_batched.rs`; for per-file audio paths, see `fa_pipeline.rs` or `transcribe_pipeline.rs` in the same directory
5. Add typed `CommandOptions::Foo` in `crates/batchalign-app/src/types/options.rs` and the builder in `crates/batchalign-cli/src/args/options.rs`

`main.rs` must remain a thin wrapper — tracing setup + `run_command()` call. No command-specific logic.

### Python/Rust Ownership Boundary

**Rust owns all logic** (~5,000+ lines in `batchalign-chat-ops` + server crates):

| Domain | Rust Module | What It Does |
|--------|-------------|--------------|
| CHAT parsing | `parse.rs` | Parse lenient/strict CHAT text to AST |
| CHAT serialization | `serialize.rs` | AST to CHAT text |
| Word extraction | `extract.rs` | Domain-aware extraction (Mor/Wor/Pho/Sin) |
| Result injection | `inject.rs`, `morphosyntax/` | Inject NLP results back into CHAT AST |
| DP alignment | `dp_align.rs` | Hirschberg sequence alignment (WER, retokenization) |
| WER computation | `wer_conform.rs` | Word normalization for WER evaluation |
| Retokenization | `retokenize/` | Character-level DP for Stanza word splits/merges |
| ASR post-processing | `asr_postprocess/` | Compound merging, number expansion, Cantonese normalization, retokenization, disfluency/retrace detection |
| Utterance segmentation | `utseg.rs` | Boundary assignment computation |
| Translation injection | `translate.rs` | %xtra tier injection |
| Coreference injection | `coref.rs` | Sparse %xcoref tier injection |
| Forced alignment | `fa/` | Grouping, extraction, injection, postprocessing, UTR timing recovery |
| Caching | server crates | Tiered cache (moka hot + SQLite cold), BLAKE3-keyed NLP results |

**Python is a pure model server** (~5,900 lines total, all model I/O):

| Python Module | What It Does |
|---------------|--------------|
| `inference/morphosyntax.py` | Load Stanza, call `nlp()`, return raw `to_dict()` |
| `inference/utseg.py` | Load Stanza constituency, return raw parse tree |
| `inference/translate.py` | Call Google Translate / Seamless M4T, return text |
| `inference/coref.py` | Load Stanza coref, return chain structures |
| `inference/fa.py` | Load Whisper/Wave2Vec, return token timestamps |
| `inference/asr.py` | Load Whisper/Rev.AI, return raw ASR tokens |
| `inference/speaker.py` | Load NeMo/Pyannote, return diarization segments |
| `inference/benchmark.py` | Optional Python-facing helper over Rust `batchalign_core.wer_compute()` |
| `inference/hk/` | HK/Cantonese engines: Tencent, Aliyun, FunASR, Cantonese FA |
| `worker/` | stdio JSON-lines IPC, model loading, request routing |

Python contains **zero**: CHAT parsing, text normalization, DP alignment, WER computation, compound merging, number expansion, retokenization, or domain logic. Cantonese text normalization (simplified→HK traditional + domain replacements) is implemented in Rust (`batchalign-chat-ops/src/asr_postprocess/cantonese.rs`) using the `zhconv` crate. Python `_common.py` delegates to `batchalign_core.normalize_cantonese()`.

### Worker State (`_WorkerState`)

Models are loaded directly at worker startup — no pipeline infrastructure:

- `stanza_pipelines` — Stanza NLP pipelines keyed by language
- `stanza_contexts` — Tokenizer realignment contexts
- `utseg_config_builder` — Stanza constituency config factory
- `translate_fn` — Google Translate or Seamless M4T function
- `whisper_fa_model` / `wave2vec_fa_model` — FA models
- `whisper_asr_model` / `rev_api_key` — ASR backend
- `asr_engine` / `fa_engine` — `AsrEngine` / `FaEngine` enum selecting the active engine

### HK/Cantonese Engines

Built-in alternative ASR/FA engines for Hong Kong Cantonese, activated via `--engine-overrides '{"asr": "tencent"}'`. Each engine is a load/infer function pair in `batchalign/inference/hk/`.

| Engine | Task | Module | Dependencies |
|--------|------|--------|-------------|
| `tencent` | ASR | `_tencent_asr.py` | `pip install "batchalign3[hk-tencent]"` + credentials |
| `aliyun` | ASR | `_aliyun_asr.py` | `pip install "batchalign3[hk-aliyun]"` + credentials |
| `funaudio` | ASR | `_funaudio_asr.py` | `pip install "batchalign3[hk-funaudio]"` |
| `wav2vec_canto` | FA | `_cantonese_fa.py` | `pip install "batchalign3[hk-cantonese-fa]"` |

Engine dispatch uses `AsrEngine` and `FaEngine` enums in `worker/_types.py` (no plugin system).

Cantonese text normalization (simplified→HK traditional + domain replacements) is implemented in Rust (`asr_postprocess/cantonese.rs`) using `zhconv` + Aho-Corasick. Python HK engines call `batchalign_core.normalize_cantonese()` — no OpenCC Python dependency needed.

### InferTask Enum

Matches Rust `batchalign-types::worker::InferTask`:

| Python | Wire format | Description |
|--------|-------------|-------------|
| `InferTask.MORPHOSYNTAX` | `"morphosyntax"` | POS tagging, lemmatization, dependency parse |
| `InferTask.UTSEG` | `"utseg"` | Utterance segmentation |
| `InferTask.TRANSLATE` | `"translate"` | Translation |
| `InferTask.COREF` | `"coref"` | Coreference resolution |
| `InferTask.FA` | `"fa"` | Forced alignment |
| `InferTask.ASR` | `"asr"` | Automatic speech recognition |

## Python Runtime Policy

- Python 3.12 is the current supported deployment baseline.
- Active targeting of 3.14t / free-threaded Python is paused.
- Do not add new public-repo workflow assumptions that depend on 3.14t.

## Key Patterns

- Times throughout are in **milliseconds**
- Language codes use 3-letter ISO format (e.g., "eng", "spa")
- Heavy imports (stanza, torch) MUST stay lazy — CLI startup is already ~3s
- `BTreeMap` for deterministic JSON in Rust tests/snapshots (not `HashMap`)
- All Pydantic models at Python/Rust boundary — no ad-hoc dicts

## Gotchas

### Stanza token.id
Stanza `token.id` is ALWAYS a tuple: `(word_id,)` for regular words, `(start, end)` for MWT (multi-word tokens). Never assume it's an int.

### DP Aligner (`batchalign-chat-ops/src/dp_align.rs`)
Hirschberg divide-and-conquer alignment in Rust. Entry points: `align()` for `&[String]`, `align_chars()` for `&[char]`. Uses `MatchMode::Exact` or `MatchMode::CaseInsensitive`. `SMALL_CUTOFF=2048` for full-table fallback. The Python `utils/dp.py` no longer exists — all alignment is Rust-only.

### Retokenize vs Extract: Separator Word Counter Sync
The `extract.rs` module includes tag-marker separators (comma `,`, tag `„`, vocative `‡`) as NLP words in the Mor domain (they have %mor items: `cm|cm`, `end|end`, `beg|beg`). Any code that walks the AST and counts words **must also increment the counter for these separators**. Forgetting to count separators causes the word counter to desync.

### %wor Tier Semantics
- `%wor` is ALWAYS flat — just `word [bullet] word [bullet] ...`, NO groups/annotations
- Words CAN lack timing bullets — means timing unknown, NOT an error
- `%wor` mirrors main tier words 1-1 (same words, same order)
- Stale `%wor` (main tier edited without re-aligning) causes parse failures

## Cross-Repo Path Dependencies

`batchalign3` depends on `talkbank-tools` via local path dependencies:

```
../talkbank-tools/crates/      # talkbank-model, talkbank-parser, talkbank-transform, etc.
./crates/                      # batchalign-chat-ops (path deps to ../../talkbank-tools/crates/)
./pyo3/                        # batchalign-core PyO3 bridge (path deps to ../../talkbank-tools/crates/)
```

After pulling changes in `talkbank-tools`, rebuild: `make build-python`

### Content Walker (from talkbank-model)

`walk_words()` / `walk_words_mut()` centralize recursive traversal of `UtteranceContent` (24 variants) and `BracketedItem` (22 variants). Domain-aware gating: `Some(Mor)` skips retrace groups, `Some(Pho|Sin)` skips PhoGroup/SinGroup, `None` recurses all. Used extensively by `batchalign-chat-ops` (word extraction, FA injection/postprocess).

## Rust Coding Standards

For all Rust code in `crates/` and `pyo3/`.

### Edition and Tooling
- Rust **2024 edition**.
- `cargo fmt` before committing. Use `cargo fmt` (not standalone `rustfmt`).
- Run `cargo clippy --all-targets -- -D warnings` periodically. Fix real issues; do not silence with `#[allow(clippy::...)]` without explicit approval.

### Error Handling
- **No panics for recoverable conditions.** Use typed errors (`thiserror`); use `miette` for rich diagnostics where appropriate.
- **No silent swallowing.** No `.ok()`, `.unwrap_or_default()`, or silent fallbacks that hide bugs.
- **No silent defaults via `unwrap_or` / `unwrap_or_else`.** If a value can fail validation, propagate the error with `?` — never silently substitute a default. Silent fallbacks hide bugs and violate the principle of least surprise. Use `unwrap_or` only when the fallback is explicitly documented and the caller understands the semantics (e.g., a documented default in a config struct).
- **`From<T>` must be infallible.** Never implement `From<&str>` or `From<String>` on a type whose construction can fail. Use `TryFrom` instead. Panicking `From` impls are type holes — they bypass the type system's error tracking. If you need ergonomic construction for known-good compile-time values, provide named factory methods (e.g., `LanguageCode3::eng()`) instead.

### Output and Logging
- **Library crates:** `tracing` macros — never `println!`/`eprintln!`.
- **Test code:** `println!` is acceptable (cargo captures it).

### Lazy Initialization
- `LazyLock<Regex>` (from `std::sync`) for constant regex patterns. Never call `Regex::new()` inside functions or loops.
- `OnceLock` for per-instance memoization. Prefer `const` when possible. All via `std::sync`.

### Type Design
- **No boolean blindness.** Enums over bools for anything beyond simple on/off.
- **`BTreeMap` for deterministic JSON** in tests and snapshot tests (not `HashMap`).
- Prefer explicit enums over ambiguous `Option` when there are multiple meaningful states.

### Newtypes Over Primitives
- **No primitive obsession.** Use `string_id!`/`numeric_id!` macros from `crates/batchalign-types/src/macros.rs` for domain identifiers. Function signatures must be self-documenting through types, not parameter names.
- **String newtypes:** `JobId`, `CommandName`, `LanguageCode3`, `DisplayPath`, `NodeId`, `EngineVersion`, `CorrelationId`. All auto-deref to `&str`.
- **Numeric newtypes:** `NumSpeakers(u32)`, `UnixTimestamp(f64)`, `DurationSeconds(f64)`, `DurationMs(u64)`, `MemoryMb(u64)`, `WorkerPid(u32)`.
- **File paths:** Use `std::path::Path`/`PathBuf`, not `&str`/`String`. Convert to strings only at IPC/JSON boundaries via `to_string_lossy()`.
- **Boundary conversion:** Parse raw strings into newtypes at entry points (HTTP handlers, CLI flags, JSON deserialization) using `TryFrom` or `try_new()`. Interior code never handles raw primitives for typed values. `Deref<Target=str>` enables zero-friction coercion where `&str` is needed.
- **Well-known constants:** For frequently-used validated values, provide named factory methods (e.g., `LanguageCode3::eng()`, `LanguageCode3::spa()`) instead of `From<&str>`. These are infallible by construction and eliminate scattered string literals.
- **New serializer boundaries count too:** if you add a new semantic string in a
  tier model, CSV row model, filename part, or other intermediate output shape,
  wrap it in a named type instead of smuggling it through `String`.
- **No ad-hoc format parsing.** Use real parsers, not regex or string splitting for structured formats.
- **Structured output must have structured models:** introduce explicit
  intermediate types for CHAT tier payloads, CSV rows/tables, JSON payloads,
  etc., then serialize once with the right tool (`WriteChat`, `csv`, `serde`,
  ...). Do not build structured artifacts by concatenating strings.
- **Legacy parity is semantic:** use tests and live oracles to match BA2
  behavior, but do not import BA2's lossy string/document architecture into the
  Rust codebase.
- See `book/src/architecture/type-driven-design.md` for the full pattern catalog and boundary conversion recipes.

### File Size Limits
- **Recommended:** ≤400 lines per file. **Hard limit:** ≤800 lines (must be split).

### Git
Conventional Commits format: `<type>[scope]: <description>`
Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`

## Safety: Local ML Model Execution

**Running batchalign3 with ML models (transcribe, align, morphotag) on a developer machine is dangerous.** Each Whisper model instance consumes 2–15 GB RAM on MPS/CUDA. The memory guard uses per-profile budgets (GPU=6 GB, Stanza=12 GB with overhead) and `DEFAULT_MAX_WORKERS_PER_KEY = 4` to limit spawning, but large corpus runs can still exhaust system memory.

**Rules for local runs:**

- Process **one file at a time** first as a smoke test
- For large corpus runs (>5 files or >1 GB audio), use net (M3 Ultra, 256 GB) instead of a developer machine
- Use `--workers 1` to limit concurrent files per job (wired to `max_workers_per_job` in `ServerConfig`)
- Use `--timeout N` to increase the audio task timeout for very long recordings (default: 1800s = 30 minutes)
- The actual per-key pool ceiling is `max_workers_per_key` in `ServerConfig` (default: 4, configurable in `server.yaml`)

**Known crash incidents:**
- 2026-03-19: 47-file transcription (14 GB audio) with default settings caused kernel OOM on 64 GB machine
- See `docs/postmortems/` for additional incidents

## No-Op CLI Flags Are Banned

**Do not add CLI flags that are parsed but silently ignored.** If a flag exists, it must be wired to actual behavior. If a flag cannot be implemented yet, either:
1. Remove it entirely
2. Emit a clear warning: `"--flag is not yet implemented and has no effect"`

No-op flags are dangerous because users (and AI assistants) rely on them expecting behavior that never happens. The existing BA2 compatibility no-ops (`--adaptive-workers`, `--pool`, `--shared-models`, etc.) in `global_opts.rs` are technical debt that must be resolved — either implement them or remove them.

## Known Issues

### Parser Error Specificity
101 parser integration tests are `#[ignore]` because the parser reports E316 (UnparsableContent) instead of specific error codes. Tracked by `Status: not_implemented` in spec files. Does not affect output correctness — only error reporting granularity.
