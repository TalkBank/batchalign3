---
name: add-command
description: Scaffold a new batchalign3 CLI command end-to-end (Rust CLI + server orchestration + Python worker integration). Use when adding a new analysis/processing command.
disable-model-invocation: true
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Agent
---

# Add a New Batchalign3 Command

**Status:** Reference
**Last updated:** 2026-03-14

Scaffold a new CLI command through all layers. `$ARGUMENTS` should specify the command name and description (e.g., `/add-command sentiment "Sentiment analysis on utterances"`).

## Architecture

```
CLI (batchalign-cli) → Server (batchalign-app) → Worker IPC → Python inference module
```

## Step 1: Determine Command Type

| Type | Example | Python Worker? | Orchestration |
|------|---------|---------------|---------------|
| **ML inference** | morphotag, utseg, translate | Yes — needs inference module | Server extracts words → worker infers → server injects results |
| **Audio processing** | transcribe, align | Yes — needs inference module | Server sends audio path → worker returns segments |
| **File processing** | opensmile, avqi | Yes — uses typed `execute_v2` | Rust prepares audio; worker returns raw analysis payloads |
| **Rust-only** | validate, normalize | No | Pure Rust, no worker needed |

## Step 2: Add CLI Subcommand

**File:** `crates/batchalign-cli/src/cli/args.rs`

```bash
grep -n "enum Commands" crates/batchalign-cli/src/cli/args.rs
```

Add a new variant to the `Commands` enum with clap attributes.

**File:** `crates/batchalign-cli/src/commands/`

Create a new module for the command dispatch logic. Read an existing command for the pattern:

```bash
ls crates/batchalign-cli/src/commands/
```

## Step 3: Add Server Orchestration (if ML inference)

**File:** `crates/batchalign-app/src/`

Create an orchestration module that:
1. Parses CHAT using `batchalign-chat-ops`
2. Extracts relevant data (words, audio paths, etc.)
3. Calls the Python worker via `batch_infer` IPC
4. Injects results back into the CHAT AST
5. Serializes the modified CHAT

Read existing orchestrators for the pattern:

```bash
ls crates/batchalign-app/src/*.rs
```

## Step 4: Add Worker Types

**File:** `batchalign/worker/_types.py`

Add a new `InferTask` variant matching the Rust enum:

```bash
grep -n "class InferTask" batchalign/worker/_types.py
```

Add Pydantic request/response models for the new task's input/output.

**Rust side:** Add matching types to `batchalign-types::worker` (if it exists in the workspace).

## Step 5: Add Inference Module (if ML inference)

**File:** `batchalign/inference/<name>.py`

Create a new inference module following the pattern:

```python
def load_<name>_model(lang: str) -> ModelType:
    """Load the ML model. Called once at worker startup."""
    ...

def batch_infer_<name>(model: ModelType, items: list[InputType]) -> list[OutputType]:
    """Pure inference function. No CHAT, no domain logic."""
    ...
```

Read existing inference modules for the exact pattern:

```bash
ls batchalign/inference/
head -40 batchalign/inference/morphosyntax.py
```

Key rules:
- Heavy imports (torch, stanza) must be lazy
- Return raw model output — no CHAT text processing
- Use Pydantic models for structured I/O
- Type annotations on all functions

## Step 6: Wire Worker Dispatch

**File:** `batchalign/worker/_infer.py`

Add a case to the `batch_infer` dispatch router:

```bash
grep -n "def batch_infer" batchalign/worker/_infer.py
```

**File:** `batchalign/worker/_main.py`

Add model loading for the new command:

```bash
grep -n "def load_models" batchalign/worker/_main.py
```

## Step 7: Add CHAT Operations (if needed)

**File:** `crates/batchalign-chat-ops/src/`

If the command needs to extract data from or inject results into CHAT:

```bash
ls crates/batchalign-chat-ops/src/
```

Follow existing extraction/injection patterns. Use the content walker for AST traversal.

## Step 8: Add Tests

```bash
# Python inference test
cat > batchalign/tests/test_<name>.py

# Rust integration test
# Add to crates/batchalign-app/tests/

# Worker protocol test (manually)
uv run python -m batchalign.worker --task <mapped-task> --lang eng
# Then paste: {"op": "capabilities", "id": "test-1"}
```

## Step 9: Verify

```bash
# Python tests
cd $REPO_ROOT && uv run pytest batchalign/tests/test_<name>.py -v

# Rust compile
cd $REPO_ROOT && cargo check --workspace

# Rust tests
cd $REPO_ROOT && cargo nextest run --workspace

# Type check
cd $REPO_ROOT && uv run mypy batchalign/inference/<name>.py batchalign/worker/
```

## Key Files

| Purpose | Path |
|---------|------|
| CLI args definition | `crates/batchalign-cli/src/cli/args.rs` |
| CLI command dispatch | `crates/batchalign-cli/src/commands/` |
| Server orchestration | `crates/batchalign-app/src/` |
| Worker types (Pydantic) | `batchalign/worker/_types.py` |
| Worker dispatch | `batchalign/worker/_infer.py` |
| Worker model loading | `batchalign/worker/_main.py` |
| Inference modules | `batchalign/inference/` |
| CHAT operations | `crates/batchalign-chat-ops/src/` |
