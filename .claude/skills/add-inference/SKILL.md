---
name: add-inference
description: Add a new Python ML inference module for a new model or task. Use when adding a new ML backend (e.g., a new ASR engine, new FA model, new NLP task).
disable-model-invocation: true
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Agent
---

# Add a New Inference Module

Create a new Python ML inference module. `$ARGUMENTS` should describe the module (e.g., `/add-inference "wav2vec2 ASR for Mandarin"`).

## Architecture

Each inference module is a **pure inference function**: receives structured input, runs ML model, returns structured output. No CHAT parsing, no pipeline orchestration, no domain logic.

```
Worker (_infer.py) → inference/<module>.py → ML library (torch, stanza, etc.) → structured output
```

## Step 1: Understand the Pattern

Read existing inference modules to understand the conventions:

```bash
ls batchalign/inference/
head -60 batchalign/inference/morphosyntax.py
head -60 batchalign/inference/fa.py
```

Every inference module has:
1. **Model loading function** — called once at worker startup
2. **Inference function** — called per-batch, pure computation
3. **Pydantic types** — for structured I/O at IPC boundary

## Step 2: Define Types

**File:** `batchalign/worker/_types.py`

Add Pydantic models for the request and response. These mirror Rust types across the IPC boundary.

```bash
grep -n "class.*BaseModel" batchalign/worker/_types.py | head -20
```

Rules:
- Use domain types from `batchalign/inference/_domain_types.py` (AudioPath, TimestampMs, etc.)
- All fields must have type annotations
- No `Any` or `object` types

If this is a new InferTask variant, add it to the `InferTask` enum:

```bash
grep -n "class InferTask" batchalign/worker/_types.py
```

## Step 3: Create the Inference Module

**File:** `batchalign/inference/<name>.py`

Template:

```python
"""<Name> inference module.

Receives structured input, runs ML model, returns raw output.
No CHAT parsing, no text processing, no domain logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Import heavy ML types here

from batchalign.inference._domain_types import ...


def load_<name>_model(lang: str) -> <ModelType>:
    """Load the ML model for the given language.

    Called once at worker startup. Heavy imports go here.
    """
    import torch  # Lazy import
    ...


def batch_infer_<name>(
    model: <ModelType>,
    items: list[<InputType>],
) -> list[<OutputType>]:
    """Run inference on a batch of items.

    Pure computation — no CHAT text, no side effects.
    """
    ...
```

Key rules:
- **Lazy imports** for heavy libraries (torch, stanza, transformers) — put them inside the function
- **No CHAT text** — receive extracted words/audio, return structured results
- **Type annotations** on all functions and variables
- **No `Any`** — use specific types, `TYPE_CHECKING` guards for expensive imports
- **Pydantic models** at IPC boundaries

## Step 4: Wire into Worker

### Model loading

**File:** `batchalign/worker/_main.py`

Add model loading for the new module:

```bash
grep -n "def load_models" batchalign/worker/_main.py
```

### Dispatch

**File:** `batchalign/worker/_infer.py`

Add a case to route the new InferTask to your inference function:

```bash
grep -n "def batch_infer" batchalign/worker/_infer.py
```

## Step 5: Add to Worker Capabilities

**File:** `batchalign/worker/_handlers.py`

Ensure the new task appears in the capabilities response:

```bash
grep -n "capabilities" batchalign/worker/_handlers.py
```

## Step 6: Add Tests

```python
# batchalign/tests/test_<name>.py
"""Tests for <name> inference module."""

def test_<name>_basic():
    """Test basic inference with minimal input."""
    ...
```

Rules:
- No mocks (`unittest.mock` is banned)
- Use real models or skip if not available (`pytest.mark.skipif`)
- Test with minimal valid input
- Verify output types match Pydantic models

## Step 7: Verify

```bash
# Run the new tests
uv run pytest batchalign/tests/test_<name>.py -v

# Type check
uv run mypy batchalign/inference/<name>.py

# Test worker starts with new infer task
uv run python -m batchalign.worker --task <name> --lang eng
# Should print {"ready": true, ...}

# Full test suite
uv run pytest batchalign --disable-pytest-warnings -x -q
```

## Key Files

| Purpose | Path |
|---------|------|
| Existing inference modules | `batchalign/inference/` |
| Domain type aliases | `batchalign/inference/_domain_types.py` |
| Worker types (Pydantic) | `batchalign/worker/_types.py` |
| Worker dispatch router | `batchalign/worker/_infer.py` |
| Worker model loading | `batchalign/worker/_main.py` |
| Worker capabilities | `batchalign/worker/_handlers.py` |
| HK/Cantonese engines | `batchalign/inference/hk/` |
