---
name: debug-worker
description: Diagnose Python worker IPC issues. Use when a worker fails to start, crashes during inference, returns unexpected output, or the server can't communicate with it.
disable-model-invocation: true
allowed-tools: Bash, Read, Glob, Grep, Agent
---

# Diagnose Worker IPC Issues

Investigate failures in the Rust server ↔ Python worker communication. `$ARGUMENTS` can be an error message, command name, or symptom description.

## Step 1: Identify the Failing Layer

Workers communicate via stdio JSON-lines. Failures occur at three layers:

| Layer | Symptoms | Key Files |
|-------|----------|-----------|
| **Spawn** | "failed to start worker", exit immediately | `crates/batchalign-app/src/worker/` |
| **Protocol** | "invalid JSON", no `ready` message, timeout | `batchalign/worker/_protocol.py` |
| **Inference** | Worker crashes during `batch_infer`, wrong output | `batchalign/inference/*.py` |

## Step 2: Test Worker Directly

Bypass the Rust server to isolate the issue:

```bash
# Start worker manually (should print {"ready": true, ...})
uv run python -m batchalign.worker --task morphosyntax --lang eng

# Send a health check (paste into stdin)
{"op": "health", "id": "test-1"}

# Send a capabilities check
{"op": "capabilities", "id": "test-2"}
```

If the worker doesn't print `{"ready": true, ...}`, the issue is model loading.

## Step 3: Check Model Loading

```bash
# Verbose startup to see import/load errors
uv run python -c "
from batchalign.worker._main import load_models
state = load_models('morphotag', 'eng')
print('Models loaded:', list(vars(state).keys()))
"
```

Common failures:
- **Import errors** — missing dependency (check `uv pip list`)
- **CUDA/MPS issues** — force CPU with `CUDA_VISIBLE_DEVICES=""`
- **Memory** — model doesn't fit; check `sysinfo` memory gate in server

## Step 4: Check IPC Protocol

If the worker starts but the server can't communicate:

```bash
# Check the protocol handler
grep -n "def serve_stdio" batchalign/worker/_protocol.py

# Check the Rust side worker spawn
grep -rn "spawn" crates/batchalign-app/src/worker/
```

Key protocol rules:
- Worker MUST print `{"ready": true, "pid": N, "transport": "stdio"}` on startup
- Each request is one JSON line on stdin
- Each response is one JSON line on stdout
- stderr is for logging only (not parsed)
- `sys.argv` must be set correctly before model imports

## Step 5: Check Inference Path

```bash
# Find the inference module for this command
grep -n "InferTask" batchalign/worker/_infer.py

# Check the specific inference function
cat batchalign/inference/<module>.py
```

Each inference module follows the pattern: receive structured input → call ML model → return structured output. No CHAT parsing, no text processing.

## Step 6: Check Server-Side Orchestration

If the worker returns correct output but the final CHAT is wrong:

```bash
# The orchestration lives in the server crate
ls crates/batchalign-app/src/{morphosyntax,utseg,translate,coref,fa}.rs

# The CHAT manipulation lives in chat-ops
ls crates/batchalign-chat-ops/src/
```

The Rust server does: parse CHAT → extract words → send to worker → inject results → serialize CHAT. If worker output is correct but CHAT output is wrong, the bug is in extraction or injection.

## Step 7: Daemon Issues

If using the auto-daemon (no `--server` flag):

```bash
# Check daemon status
./target/debug/batchalign3 serve status

# Check daemon log
cat ~/.batchalign3/daemon.log

# Stale binary detection
cat ~/.batchalign3/daemon.json  # check build_hash
```

## Key Files

| Purpose | Path |
|---------|------|
| Worker entry point | `batchalign/worker/_main.py` |
| Worker protocol | `batchalign/worker/_protocol.py` |
| Worker handlers | `batchalign/worker/_handlers.py` |
| Inference dispatch | `batchalign/worker/_infer.py` |
| Worker types (Pydantic) | `batchalign/worker/_types.py` |
| Inference modules | `batchalign/inference/*.py` |
| Server worker pool | `crates/batchalign-app/src/worker/` |
| Server orchestrators | `crates/batchalign-app/src/{morphosyntax,utseg,translate,fa}.rs` |
| CHAT extraction/injection | `crates/batchalign-chat-ops/src/` |
| Daemon management | `crates/batchalign-cli/src/daemon.rs` |
