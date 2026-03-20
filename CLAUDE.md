# CLAUDE.md

**Status:** Current
**Last updated:** 2026-03-20

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Batchalign is a Rust-primary language sample analysis (LSA) suite from the [TalkBank](https://talkbank.org/) project, with Python used exclusively as a stateless ML model server. It processes conversation audio files and transcripts in CHAT format, providing ASR, forced alignment, morphosyntactic analysis, translation, utterance segmentation, and audio feature extraction.

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

Day-to-day:

```bash
make build                   # Rebuild everything
./target/debug/batchalign3 transcribe input/ output/ --lang eng

# Or let cargo handle incremental rebuilds automatically:
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
| REST API types (`types/api.rs`, `ToSchema` derives) | `bash scripts/generate_dashboard_api_types.sh` (regenerates `openapi.json` + `frontend/src/generated/api.ts`) |
| Worker IPC types (`types/worker_v2.rs`, `morphosyntax/mod.rs`, etc.) | `make generate-ipc-types` (regenerates `ipc-schema/` + `batchalign/generated/`) |
| Everything | `make build` |

**API schema discipline:** When you modify any Rust struct with `#[derive(ToSchema)]` or change route signatures in `routes/`, you **must** run `bash scripts/generate_dashboard_api_types.sh` and commit the regenerated `openapi.json` and `frontend/src/generated/api.ts`. CI will fail on API drift otherwise. Verify with `bash scripts/check_dashboard_api_drift.sh`.

**IPC type discipline:** When you modify any Rust struct with `#[derive(schemars::JsonSchema)]` that crosses the Python worker boundary, you **must** run `make generate-ipc-types` and commit the regenerated `ipc-schema/` and `batchalign/generated/` files. `make ci-local` and `make check-ipc-drift` verify schemas are current. See `book/src/developer/ipc-type-sync.md` for the full workflow and `book/src/developer/ipc-union-migration.md` for the roadmap to full codegen.

**Docs and diagrams discipline:** When a change affects code structure,
CLI options, data flow, or user-visible behavior, update **both** the
user-facing book pages (`book/src/user-guide/`, especially `cli-reference.md`)
**and** the developer-facing architecture pages (`book/src/architecture/`) in
the same change. Include Mermaid diagrams to explain how options, data, and
language information flow through the pipeline. Do not leave the book
describing behavior that no longer exists or omitting new CLI options.

**mdBook/Mermaid gotcha:** Be careful with raw angle brackets in Markdown and
Mermaid labels. mdBook can treat them as HTML and emit warnings or render
incorrectly. Prefer quoted labels in diagrams and describe generic type
arguments in surrounding prose or code spans instead of raw Mermaid labels.

`batchalign3` is a Rust binary (`crates/batchalign-cli`). Use `uv run` only for Python commands (pytest, mypy).

## Build and Test Commands

**ML tests are excluded by default.** `cargo nextest run` and `uv run pytest` both run only fast unit tests. ML/golden tests must be opted into explicitly. See `book/src/developer/testing.md` for the full testing strategy.

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

Development of code must use TDD. Debugging sessions must begin with writing a test that demonstrates the bug.

**No mocks.** `unittest.mock` is banned — zero imports allowed anywhere in the test suite. Test doubles that are alternate implementations of a protocol are allowed. Shared doubles live in `batchalign/tests/doubles.py`.

**CHAT test content gotcha:** Minimal valid CHAT needs `@Languages`, `@Participants`, AND `@ID` lines. Missing `@ID` causes `CHATValidationException: Encountered undeclared tier`.

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

### Boolean Blindness

- **No boolean blindness.** Enums over bools for anything beyond simple on/off. Banned: 2+ bool params, 2+ related bool fields, opposite bool pairs (`foo`/`no_foo`), ambiguous bool returns. Use `enum.Enum` or `typing.Literal["option1", "option2"]` for multi-way choices. OK as bool: `verbose`, `force`, `quiet`, single on/off flags where the name is self-documenting.

### Newtypes Over Primitives

- **No primitive obsession.** Domain values must have domain types. Function signatures should be self-documenting through type names, not parameter names.
- Use `typing.NewType` (e.g., `TimestampMs = NewType("TimestampMs", int)`) or Pydantic constrained types at module/IPC boundaries. For lightweight internal use, `type` aliases are acceptable when they clarify intent.
- Domain types already defined in `_domain_types.py`: `AudioPath`, `NumSpeakers`, `SpeakerId`, `TimestampMs`. Use these instead of bare `str`/`int`.
- Parse raw strings into typed values at the boundary (CLI args, IPC, file I/O). Interior code should never handle raw strings for typed values.
- **No ad-hoc format parsing.** Use real parsers (JSON: `json`, XML: `xml.etree`, etc.) not regex or string splitting for structured formats. Regex is for flat text only (normalization, search, validation).

### Type Checking

Run **mypy** before committing changes (strict mode, configured in `mypy.ini`):
```bash
uv run mypy
```

### CHAT Format Handling — No Text Hacking

**All CHAT parsing and serialization MUST go through principled AST manipulation, never ad-hoc string/regex manipulation of raw CHAT text.** This is a hard rule.

- **Parsing**: Use `batchalign_core` Rust functions. These parse CHAT into a proper AST, manipulate it, and re-serialize correctly.
- **Never** use regex, string splitting, or line-by-line processing to extract or modify CHAT content. If `batchalign_core` doesn't expose the data you need, add a new Rust function.

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

**`batchalign_cli::run_command()`** in `crates/batchalign-cli/src/lib.rs` is the single canonical command router. Both the standalone binary (`main.rs`) and the PyO3 console_scripts entry point (`pyo3/src/cli_entry.rs`) call it. **Never duplicate this match block.** If you need to add a new CLI command:

1. Add the `Commands::Foo` variant to `crates/batchalign-cli/src/args/mod.rs`
2. Add the match arm in `run_command()` in `crates/batchalign-cli/src/lib.rs` — this is the only dispatch site
3. If the command needs server-side orchestration, add `infer_task_for_command()` and `command_requires_infer()` mappings in `crates/batchalign-app/src/runner/mod.rs`
4. If it uses the batched infer path, add routing in `crates/batchalign-app/src/runner/dispatch/infer.rs`
5. Add typed `CommandOptions::Foo` in `crates/batchalign-app/src/types/options.rs` and the builder in `crates/batchalign-cli/src/args/options.rs`

`main.rs` and `cli_entry.rs` must remain thin wrappers — tracing setup + `run_command()` call. No command-specific logic.

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
- **No primitive obsession.** Use `string_id!`/`numeric_id!` macros from `types/macros.rs` for domain identifiers. Function signatures must be self-documenting through types, not parameter names.
- **String newtypes:** `JobId`, `CommandName`, `LanguageCode3`, `FileName`, `NodeId`, `EngineVersion`, `CorrelationId`. All auto-deref to `&str`.
- **Numeric newtypes:** `NumSpeakers(u32)`, `UnixTimestamp(f64)`, `DurationSeconds(f64)`, `DurationMs(u64)`, `MemoryMb(u64)`, `WorkerPid(u32)`.
- **File paths:** Use `std::path::Path`/`PathBuf`, not `&str`/`String`. Convert to strings only at IPC/JSON boundaries via `to_string_lossy()`.
- **Boundary conversion:** Parse raw strings into newtypes at entry points (HTTP handlers, CLI flags, JSON deserialization). Interior code never handles raw primitives for typed values. `Deref<Target=str>` enables zero-friction coercion where `&str` is needed.
- **No ad-hoc format parsing.** Use real parsers, not regex or string splitting for structured formats.
- See `book/src/architecture/type-driven-design.md` for the full pattern catalog and boundary conversion recipes.

### File Size Limits
- **Recommended:** ≤400 lines per file. **Hard limit:** ≤800 lines (must be split).

### Git
Conventional Commits format: `<type>[scope]: <description>`
Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`

## Safety: Local ML Model Execution

**Running batchalign3 with ML models (transcribe, align, morphotag) on a developer machine is dangerous.** Each Whisper model instance consumes 2–5 GB RAM on MPS/CUDA. The auto-tuner (`compute_job_workers()`) and `DEFAULT_MAX_WORKERS_PER_KEY = 8` can spawn multiple concurrent inference requests that exhaust GPU/system memory, causing **unrecoverable kernel-level OOM crashes**.

**Rules for local runs:**

- Process **one file at a time** first as a smoke test
- For large corpus runs (>5 files or >1 GB audio), use net (M3 Ultra, 256 GB) instead of a developer machine
- Use `--workers 1` to limit concurrent files per job (wired to `max_workers_per_job` in `ServerConfig`)
- Use `--timeout N` to increase the audio task timeout for very long recordings (default: 1800s = 30 minutes)
- The actual per-key pool ceiling is `max_workers_per_key` in `ServerConfig` (default: 8, configurable in `server.yaml`)

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
