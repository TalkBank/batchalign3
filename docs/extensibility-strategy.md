# Batchalign3 Extensibility Strategy: Options for Future Development

**Status:** Draft
**Last updated:** 2026-03-17

## Context

Batchalign3's architecture has stabilized around a clear ownership boundary:
**Rust owns all CHAT semantics** (parsing, caching, validation, serialization,
DP alignment, ASR post-processing, result injection) while **Python is a
stateless ML model server** (load model, receive structured input, return raw
output). The main branch is no longer in churn.

The question now is: how should we structure future development so that
contributors can extend batchalign3 — adding new commands, new NLP tasks, new
ML model backends — without stepping on each other's toes?

### What "extending batchalign3" actually means

There are two fundamentally different kinds of extension:

1. **New engine for an existing command** — e.g., a new ASR backend (Azure
   Speech), a new FA model (fine-tuned Wav2Vec for Cantonese), a new
   translation backend (local LLM). The command's CHAT lifecycle doesn't
   change; only the inference provider swaps out. This is primarily a Python
   change.

2. **New command** — e.g., discourse analysis, pragmatic annotation, voice
   onset time measurement, LLM-powered CHAT repair. The full CHAT lifecycle
   must be designed: what gets extracted from the AST, what gets sent to
   inference, what gets injected back, what tiers are affected, what
   validation gates apply. This is primarily a Rust change with a Python
   inference module.

These two patterns have different costs, different contributor profiles, and
different isolation requirements. Any extensibility strategy must address both.

### Why the plugin system was abandoned

The HK/Cantonese engines were originally integrated via a plugin system
(`PluginDescriptor`, `InferenceProvider`, `discover_plugins()`). It was removed
in March 2026 because:

- Each command's dispatch shape is genuinely different (batched text, per-file
  audio, transcription pipeline, media analysis). A single `InferenceProvider`
  interface was either too generic to be useful or too specific to one command
  family.
- Process/thread optimization, GPU scheduling, batch sizing, and memory
  management differ radically between Stanza (GIL-bound, batch-efficient),
  Whisper (GPU-bound, streaming), Rev.AI (HTTP polling), and openSMILE
  (CPU-bound, no ML). Squeezing these into one abstraction created leaky
  boundaries.
- The enum-based dispatch that replaced it is explicit, exhaustive at compile
  time, and trivially testable. Adding a new engine variant is ~50 lines of
  Python across 3 files, not 300 lines of plugin infrastructure.

The question isn't "should we bring plugins back?" — it's "what structure makes
the cost of contribution predictable and the blast radius small?"

---

## The Five Options

Each option differs from the others on at least one critical architectural
dimension. They are not mutually exclusive — the eventual strategy may combine
elements from multiple options.

### Option 1: Guided Recipes (Documentation-First, Zero Code Changes)

**Critical dimension:** No architectural changes. Investment goes entirely into
contributor documentation and PR conventions.

**What it looks like:**

- Write a detailed "Adding a New Engine" guide with a complete worked example
  (e.g., "add Azure Speech as an ASR backend") showing every file touched,
  every test added, every type updated.
- Write a separate "Adding a New Command" guide with a complete worked example
  (e.g., "add a `discourse` command that produces %xdis tiers") showing the
  full Rust + Python round-trip.
- Establish PR conventions: one PR per engine variant, one PR per new command
  phase (types first, then Python inference, then Rust orchestrator, then CLI
  wiring).
- Create a `CONTRIBUTING.md` covering testing requirements (golden tests for
  new commands, unit tests for new engines), review checklist, and branch
  naming conventions.

**Files touched for a new engine (e.g., Azure ASR):**
1. `batchalign/inference/azure_asr.py` — inference module
2. `batchalign/worker/_types.py` — add `AsrEngine.AZURE` variant
3. `batchalign/worker/_model_loading/asr.py` — wire loader
4. `batchalign/worker/_asr_v2.py` — wire V2 runner
5. `batchalign/worker/_handlers.py` — capability probe
6. `pyproject.toml` — optional dependency group

**Files touched for a new command (e.g., discourse analysis):**
1. `crates/batchalign-chat-ops/src/discourse.rs` — CHAT extraction + injection
2. `crates/batchalign-app/src/discourse.rs` — server orchestrator
3. `crates/batchalign-app/src/types/options.rs` — `DiscourseOptions`
4. `crates/batchalign-app/src/runner/mod.rs` — dispatch routing
5. `crates/batchalign-app/src/runner/dispatch/plan.rs` — dispatch plan
6. `crates/batchalign-app/src/runner/dispatch/infer_batched.rs` — wire into batched dispatch
7. `crates/batchalign-cli/src/args/` — CLI args + options builder
8. `crates/batchalign-cli/src/lib.rs` — `run_command()` match arm
9. `batchalign/inference/discourse.py` — Python inference module
10. `batchalign/worker/_types.py` — `InferTask.DISCOURSE`
11. Various test files

**Pros:**
- Zero risk of breaking existing functionality.
- No upfront engineering cost — just writing.
- Forces contributors to understand the architecture (which they need anyway).
- Every extension follows the same explicit wiring pattern, making code review
  predictable.
- People can start contributing immediately.

**Cons:**
- Adding a new command still touches 10+ files across 2 languages — high
  cognitive load for the first contribution.
- No code-level isolation: a PR adding a new command can conflict with another
  PR modifying shared dispatch code.
- No compile-time enforcement that all required wiring steps were completed —
  a contributor might forget to add the capability probe or the dispatch plan,
  and the failure would only surface at runtime.
- Doesn't scale well if we expect many new commands (which is uncertain).

**Best for:** A small team (2-3 people) where the primary extension pattern is
new engines, not new commands, and where contributors are willing to learn the
full architecture.

---

### Option 2: Typed Engine Adapters (Abstract Engine Selection)

**Critical dimension:** Introduce Rust traits that abstract the
model-selection boundary, so adding a new engine variant for an existing command
is a single-file Python change plus a trait impl registration.

**What it looks like:**

Define a trait per task family in `batchalign-app`:

```rust
/// A backend that can produce word-level timings from audio + transcript.
pub trait FaBackend: Send + Sync {
    /// Unique wire name for cache partitioning and capability advertisement.
    fn engine_name(&self) -> &str;

    /// Run inference on one FA group.
    async fn infer_group(
        &self,
        audio_path: &Path,
        words: &[String],
        audio_start_ms: u64,
        audio_end_ms: u64,
    ) -> Result<Vec<Option<WordTiming>>, ServerError>;
}
```

Each engine (Whisper FA, Wave2Vec FA, Cantonese FA) implements this trait. The
FA orchestrator (`fa/mod.rs`) calls `backend.infer_group()` instead of
dispatching through the worker protocol directly. Engine registration happens
at server startup:

```rust
let fa_backend: Arc<dyn FaBackend> = match engine {
    FaEngineType::WhisperFa => Arc::new(WhisperFaBackend::new(pool)),
    FaEngineType::Wave2Vec => Arc::new(Wave2VecBackend::new(pool)),
    FaEngineType::Custom(name) => resolve_custom_fa(name, pool)?,
};
```

On the Python side, adding a new engine still means creating an inference
module, but the Rust side only needs a new `impl FaBackend` — a single file
that adapts the worker V2 protocol to the trait interface.

**Pros:**
- Clean separation: the orchestrator doesn't know about specific engines.
- Adding a new engine for an existing task is a self-contained change (one
  Python module + one Rust adapter file).
- Compile-time enforcement: forgetting to implement a required method is a
  compile error.
- Natural place to put engine-specific configuration, health checks, and
  capability negotiation.

**Cons:**
- Significant upfront refactoring: every existing orchestrator (`fa/mod.rs`,
  `morphosyntax.rs`, `utseg.rs`, etc.) must be refactored to use traits
  instead of direct worker dispatch.
- Traits across the Rust/Python boundary require careful design — the trait
  impl ultimately calls Python workers via IPC, adding an indirection layer.
- Doesn't help with adding new commands (only new engines for existing commands).
- Over-abstraction risk: if most tasks only ever have 1-2 engines, the trait
  layer is ceremony without payoff.
- Async traits (`async fn` in trait) have ergonomic rough edges in Rust even
  with edition 2024 stabilization.

**Best for:** A project expecting many engine variants per command (e.g., 5+
ASR backends, 3+ FA backends) where the engine boundary is the primary
extension point.

---

### Option 3: Command Scaffolding with Code Generation

**Critical dimension:** Reduce the cost of adding a new command from "touch 10+
files by hand" to "run a generator, fill in the blanks."

**What it looks like:**

Create a `scripts/new-command.sh` (or `cargo-generate` template) that, given a
command name and dispatch shape, generates skeleton files:

```bash
./scripts/new-command.sh discourse --shape batched-text
```

This produces:
- `crates/batchalign-chat-ops/src/discourse.rs` — payload types, cache key, injection stubs
- `crates/batchalign-app/src/discourse.rs` — orchestrator skeleton
- `crates/batchalign-app/src/types/options.rs` — `DiscourseOptions` with common fields
- `crates/batchalign-cli/src/args/discourse.rs` — CLI argument struct
- `batchalign/inference/discourse.py` — Python inference module skeleton
- Test skeleton files

The generator also patches the shared dispatch files (`runner/mod.rs`,
`plan.rs`, `options.rs` enum variants, `lib.rs` match arm) with the new command
variant. Scaffolded code compiles and runs (producing a no-op result), and the
contributor fills in the domain logic.

**Pros:**
- Dramatically lowers the activation energy for new commands — the boilerplate
  is handled, contributors focus on domain logic.
- Generated code enforces the correct pattern (cache check → infer → inject →
  validate), preventing structural mistakes.
- The generator itself serves as living documentation of the architecture.
- Could include a test scaffold that verifies the full roundtrip with a dummy
  CHAT file.

**Cons:**
- Generator maintenance burden: every time the dispatch architecture changes,
  the templates must be updated. If the generator gets stale, it produces
  broken code — worse than no generator at all.
- Works well for commands that fit existing dispatch shapes, but novel
  shapes (e.g., a command that reads two CHAT files and produces a comparison)
  require manual escape hatches.
- Patching shared Rust files (adding enum variants, match arms) is fragile —
  code generation into existing files is harder to maintain than generating new
  files.
- Upfront engineering cost to build the generator.

**Best for:** A project expecting many new commands with predictable structure,
where the bottleneck is boilerplate, not design.

---

### Option 4: Separate Engine Packages (Repository Boundary Isolation)

**Critical dimension:** Engine implementations live in separate Python packages
(separate repos, separate release cycles), discovered at runtime via Python
entry points or explicit configuration.

**What it looks like:**

The core `batchalign3` package defines a Python protocol (PEP 544) for each
engine family:

```python
# batchalign/engine_protocol.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class AsrEngineProtocol(Protocol):
    """Protocol that ASR engine packages must implement."""
    engine_name: str

    def load(self, lang: str, device_policy: str, **kwargs: object) -> None: ...
    def infer(self, audio_path: str, num_speakers: int) -> MonologueAsrResponse: ...
```

External packages register via `pyproject.toml` entry points:

```toml
# In a separate package: batchalign3-azure-asr
[project.entry-points."batchalign3.asr_engines"]
azure = "batchalign3_azure_asr:AzureAsrEngine"
```

At worker startup, the worker discovers registered engines and adds them to the
dispatch table. The Rust side learns about available engines via the capability
advertisement (`GET /health`).

**Pros:**
- Maximum isolation: engine contributors don't need to touch the batchalign3
  repo at all.
- Independent release cycles: a new engine version can ship without a
  batchalign3 release.
- Clean dependency isolation: heavy dependencies (Tencent SDK, Aliyun SDK) are
  only pulled in by the packages that need them.
- External contributors (outside the team) can create engines without
  core team review.

**Cons:**
- Runtime discovery is fragile: entry point registration can fail silently, and
  debugging "why isn't my engine showing up?" is harder than "the match arm
  doesn't compile."
- Protocol evolution: if the engine protocol changes, all external packages
  break. Versioning the protocol adds complexity.
- The Rust side still needs to know about engine names for cache key
  partitioning, option validation, and dispatch routing — so the isolation is
  partial.
- Testing burden shifts: the core repo can't test external engines, and
  external engines can't easily test against the full pipeline without
  integration fixtures.
- We had a plugin system. We removed it. Bringing it back in a different
  form risks re-learning the same lessons.
- For a team of 2-3 people working on TalkBank-internal engines, the repo
  boundary is overhead, not isolation.

**Best for:** A project with a large external contributor community adding
engines that the core team shouldn't need to review or maintain — more like an
ecosystem than a team project.

---

### Option 5: Two-Tier Strategy (Recipes for Commands + Protocol for Engines)

**Critical dimension:** Explicitly separate the "new command" and "new engine"
extension patterns and optimize each independently, rather than trying to find
one abstraction that covers both.

**What it looks like:**

**Tier 1 — New engines: Python-side protocol with Rust-side enum registration.**

Define lightweight Python protocols for each engine family (ASR, FA, text NLP).
New engines implement the protocol and register in a single Python registry
file. The Rust side uses a stable `engine_overrides` mechanism (already exists)
to route to arbitrary engine names, and learns capabilities via health check.

The key insight: new engines for existing commands don't need Rust changes to
the orchestrator — only to the engine name enum and the CLI `--engine-overrides`
validation. This is already how HK engines work today; the protocol just makes
the pattern explicit and documented.

Adding a new ASR engine:
1. Create `batchalign/inference/my_engine.py` implementing `AsrEngineProtocol`
2. Add `AsrEngine.MY_ENGINE` variant in `_types.py`
3. Wire loader + runner in `_model_loading/asr.py` and `_asr_v2.py`
4. Add capability probe in `_handlers.py`
5. Add optional deps in `pyproject.toml`

That's 5 files, all Python, all following a mechanical pattern. The Rust side
needs no changes — `engine_overrides` already passes arbitrary engine names
through.

**Tier 2 — New commands: Documented Rust recipes with phased PR process.**

New commands are infrequent, architecturally significant, and require design
review. They don't benefit from abstraction — each one has unique CHAT
semantics. Instead, provide:

- A detailed recipe document per dispatch shape (batched text, per-file audio,
  media analysis, transcription pipeline).
- A phased PR process:
  - **PR 1:** Types + options (Rust). Defines the command's options, cache key
    structure, and wire types. Reviewable in isolation.
  - **PR 2:** Python inference module. Implements the ML inference function.
    Testable against fixtures without the full pipeline.
  - **PR 3:** Rust orchestrator. Implements the CHAT lifecycle (extract → cache
    → infer → inject → validate). The bulk of the design review.
  - **PR 4:** CLI wiring + tests. Adds the command to the CLI, writes golden
    tests, updates docs.
- A checklist PR template that ensures all wiring steps are completed.

**Pros:**
- Honest about the asymmetry: engines and commands are different problems that
  deserve different solutions.
- Low upfront cost: the engine protocol is just documenting what already works
  (the HK pattern). The command recipes require no code generation.
- Phased PRs reduce review burden and conflict risk — two people can work on
  different phases simultaneously.
- The Python-side protocol is lightweight enough that it doesn't reintroduce
  plugin system complexity. No discovery, no entry points, no dynamic loading.
  Just a documented function signature.
- The Rust side stays explicit and exhaustive — no runtime dispatch, no trait
  objects, no dynamic registration.

**Cons:**
- Two different patterns to learn (though they serve different purposes).
- The "protocol" for engines is informal (documented convention, not enforced
  by the type system) unless we add runtime `isinstance` checks.
- New commands still touch many files — the phased PR process reduces cognitive
  load per review but not total engineering effort.
- Doesn't help with truly novel extension patterns (e.g., a command that
  combines audio + text inference in a single pipeline step).

**Best for:** A small team where engines change frequently but commands change
rarely, and where the contributors are comfortable with both
Python and Rust and needs clear guardrails, not abstractions.

---

## Comparison Matrix

| Dimension | Option 1: Recipes | Option 2: Traits | Option 3: Scaffold | Option 4: Packages | Option 5: Two-Tier |
|-----------|:-:|:-:|:-:|:-:|:-:|
| Upfront engineering cost | None | High | Medium | Medium | Low |
| New engine cost (files) | 5-6 | 2-3 | 5-6 | 4-5 (separate repo) | 5 (all Python) |
| New command cost (files) | 10+ | 10+ | 3-4 (generated) | 10+ | 10+ (but phased) |
| Code isolation | None | Engine-level | Command-level | Repository-level | Engine-level |
| Compile-time safety | Existing | Improved | Existing | Reduced | Existing |
| Contributor independence | Low | Medium | Medium | High | Medium |
| Architecture risk | None | Medium | Low | Medium | Low |
| Scales to large community | No | Somewhat | Somewhat | Yes | Somewhat |
| Immediate availability | Yes | No (months) | No (weeks) | No (weeks) | Yes (days) |

## Recommendation Context

For a team of 2 people working on a research infrastructure
project with a finite set of NLP tasks:

- **New commands are rare.** TalkBank's command set (transcribe, align,
  morphotag, utseg, translate, coref, compare, benchmark, opensmile, avqi) is
  already comprehensive. A new command happens maybe 1-2 times per year.

- **New engine variants are more common.** As ML models improve and new
  providers emerge, swapping ASR/FA/translation backends is the dominant
  extension pattern.

- **The team is small.** Isolation between contributors matters less than
  clarity of ownership and predictable review process.

- **Correctness matters more than velocity.** CHAT format handling has strict
  invariants. Any abstraction that makes it easier to violate CHAT semantics
  (e.g., by hiding the orchestration behind a generic pipeline) is a net
  negative.

Given this context, **Option 5 (Two-Tier)** or **Option 1 (Guided Recipes)**
are the lowest-risk starting points. Option 5 is the natural evolution of
Option 1: start with recipes, and if the engine-variant pattern proves to be
the dominant contribution type, formalize it into a lightweight protocol.

Option 2 (Traits) becomes worthwhile only if we find ourselves with 5+ engines
per task family. Option 3 (Scaffolding) becomes worthwhile only if we're adding
3+ new commands per year. Option 4 (Separate Packages) is designed for an
open-source ecosystem with external contributors, which is not our current
situation.

---

## Immediate Action Items (Regardless of Option)

These apply to all five options and can be done now:

1. **Write `CONTRIBUTING.md`** covering branch naming, PR conventions, commit
   message format, and review expectations.

2. **Write "Adding a New Engine" recipe** using Tencent ASR as the worked
   example (it was the most recent engine addition). Show every file, every
   test, every type change.

3. **Write "Adding a New Command" recipe** using `coref` as the worked example
   (it's the simplest CHAT-mutating command — single-document, no audio).

4. **Establish code review boundaries:**

5. **Set up a shared test corpus** for golden-model testing of new engines,
   separate from the existing reference corpus.

6. **Document the phased PR process** for new commands, even if we don't
   formalize it into templates yet.
