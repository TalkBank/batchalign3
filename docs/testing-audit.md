# Batchalign3 Test Suite Audit

**Status:** Historical
**Last updated:** 2026-03-21 22:32 EDT

> **Note:** This snapshot is frozen as of 2026-03-19. The single source of truth
> for testing guidance is now `book/src/developer/testing.md`. Structural lints
> have moved to `cargo xtask lint-wide-structs` and `cargo xtask lint-ci-hygiene`.

## Summary

**1,795 tests total** across Python (593) and Rust (1,202). The suite has strong
coverage of serialization, type conformance, and individual NLP algorithms, but
critical gaps in integration testing of the actual dispatch and concurrency paths
that users exercise — which is exactly where the Mar 19 production failure occurred.

## Test Inventory by Layer

### Layer 1: Serialization & Type Conformance (≈400 tests)

**What:** JSON roundtrips, Pydantic model validation, Rust↔Python schema
conformance, IPC envelope parsing, newtype construction.

**Verdict:** Overdone. These catch schema drift but never catch runtime bugs.
400 tests proving `InferRequest` serializes correctly don't help when the
worker never receives the request because the stdio pipe is corrupted.

| File | Lang | Count | What |
|------|------|-------|------|
| `test_worker_ipc.py` | Py | 12 | Health/capabilities/infer serialization |
| `test_worker_protocol_v2_types.py` | Py | 5 | V2 type validation |
| `test_worker_protocol_v2_artifacts.py` | Py | 9 | Binary artifact handling |
| `test_ipc_type_conformance.py` | Py | 6 | Rust↔Python schema match |
| `json_compat.rs` | Rs | 19 | JSON snapshot roundtrips |
| `worker_protocol_v2_compat.rs` | Rs | 1 | V2 envelope roundtrip |
| `worker_v2_fa_roundtrip.rs` | Rs | 1 | FA result serialization |
| `types/api.rs` unit tests | Rs | 13 | Newtype construction |
| `types/options.rs` unit tests | Rs | 20 | CommandOptions variants |
| `types/config.rs` unit tests | Rs | 10 | ServerConfig YAML |
| `types/domain.rs` unit tests | Rs | 22 | Domain newtypes |
| `types/worker.rs` unit tests | Rs | 12 | Worker protocol types |
| Plus ~250 more across `worker/*.rs` | Rs | ~250 | Result/request parsing |

### Layer 2: NLP Algorithms (≈550 tests)

**What:** DP alignment, morphosyntax mapping, number expansion, compound
merging, retokenization, UTR, WER normalization, Cantonese processing.

**Verdict:** Good coverage. These are pure functions with well-defined inputs
and outputs. The num2text (64 tests for 12 languages), NLP mapping (144 tests
for UD tags), and retokenization (58 tests) suites are thorough.

| Area | File(s) | Count | What |
|------|---------|-------|------|
| NLP tag mapping | `nlp/mapping/mod.rs` | 144 | UPOS, XPOS, features, deprel |
| Number expansion | `asr_postprocess/num2text.rs` | 64 | 12 languages |
| UTR | `fa/utr.rs`, `fa/utr/two_pass.rs` | 58 | Utterance timing recovery |
| Retokenization | `retokenize/*.rs`, `tokenizer_realign/*.rs` | 58 | Stanza word split/merge |
| Morphosyntax | `morphosyntax/*.rs` | 73 | Stanza output processing |
| Compounds | `asr_postprocess/compounds.rs` | 17 | 3,584-pair compound table |
| Cantonese | `asr_postprocess/cantonese.rs` | 22 | zh-HK normalization |
| Disfluency | `asr_postprocess/cleanup.rs` | 13 | D1 and D1b markers |
| WER | `wer_conform.rs` | 31 | Word normalization |
| Utseg | `utseg.rs`, `utseg_compute.rs` | 24 | Boundary computation |
| Translation | `translate.rs` | 26 | Payload/cache key |
| Coref | `coref.rs` | 31 | Chain injection |

### Layer 3: Python Inference Module Tests (≈280 tests)

**What:** Each inference module (morphosyntax, ASR, FA, speaker, translate,
utseg, coref, opensmile, avqi) tested with faked/monkeypatched model objects.
Also covers worker V2 execute dispatch and HK engine providers.

**Verdict:** Good unit coverage, but tests never exercise the real code path
through the worker protocol. Monkeypatching the model avoids loading times but
means tests never prove the actual model→response→serialization chain works.

| Area | File(s) | Count | Models? |
|------|---------|-------|---------|
| Morphosyntax | `test_morphosyntax_inference.py`, etc. | 70+ | Faked Stanza |
| ASR | `test_asr_inference.py`, `test_asr_model_loading.py` | 45+ | Faked Whisper |
| FA | `test_fa_inference.py`, `test_rust_fa.py` | 20+ | Faked models |
| Speaker | `test_speaker_inference.py` | 15+ | Faked Pyannote |
| Translate | `test_translate_inference.py` | 7 | Faked Google |
| Utseg | `test_utseg_inference.py` | 18 | Faked Stanza |
| Worker V2 | `test_worker_execute_v2.py` | 19 | Monkeypatched hosts |
| V2 matrix | `test_worker_execute_v2_matrix.py` | 8 | Monkeypatched |
| ASR V2 | `test_worker_asr_v2.py` | 9 | Monkeypatched Whisper |
| FA V2 | `test_worker_fa_v2.py` | 8 | Monkeypatched |
| Speaker V2 | `test_worker_speaker_v2.py` | 13 | Monkeypatched |
| HK engines | `hk/test_*.py` (8 files) | 150+ | Faked SDKs |

### Layer 4: Worker Protocol Integration (≈26 tests)

**What:** Spawn a real Python worker subprocess in test-echo mode, communicate
over stdio JSON-lines, verify protocol correctness.

**Verdict:** Covers the happy path but misses the failure mode that caused
the Mar 19 bug. All tests use **sequential** request/response. None test
**concurrent** dispatch through SharedGpuWorker.

| File | Lang | Count | What |
|------|------|-------|------|
| `test_cli_e2e.py` | Py | 9 | Spawn test-echo, sequential protocol |
| `worker_integration.rs` | Rs | 17 | Spawn, pool dispatch, reuse, warmup |

### Layer 5: Server Integration (≈50 tests)

**What:** Start a real HTTP server with test-echo workers, submit jobs via
REST API, poll for completion, verify results.

**Verdict:** Solid coverage of the REST API and job lifecycle. But all tests
use test-echo workers (no ML models), so they verify the control plane but
not the data plane. Job submission, polling, cancellation, restart, deletion
all tested.

| File | Lang | Count | What |
|------|------|-------|------|
| `integration.rs` | Rs | 22 | HTTP API, job lifecycle |
| `e2e.rs` | Rs | 20 | CLI → server → results |
| `error_paths.rs` | Rs | 6 | Error handling |
| `commands.rs` | Rs | 7 | Job commands |
| `daemon_e2e.rs` | Rs | 1 | Daemon lifecycle |
| `option_receipt.rs` | Rs | 5 | Option application |
| `command_matrix.rs` | Rs | 5 | Dispatch verification |
| `profile_verification.rs` | Rs | 3 | Worker profiles |

### Layer 6: Golden Tests with Real Models (≈56 tests)

**What:** Run real NLP pipelines (Stanza, Whisper, Wave2Vec) on test inputs
and compare against golden snapshots. These are the closest thing to
end-to-end tests.

**Verdict:** These are the most valuable tests but they only run when ML
models are installed. They cover **Stanza** profile well (morphotag, utseg,
translate, coref across 8+ languages) and **GPU** profile partially (Whisper
ASR, Wave2Vec FA, speaker diarization). But they don't test through the
**pool dispatch path** — they use the live server fixture which bypasses
SharedGpuWorker.

| File | Lang | Count | What |
|------|------|-------|------|
| `golden.rs` | Rs | 12 | Text NLP pipeline (Stanza) |
| `golden_parity.rs` | Rs | 16 | BA2 parity (8 languages) |
| `golden_audio.rs` | Rs | 23 | Audio pipeline (Whisper, Wave2Vec, speaker) |
| `live_server_fixture.rs` | Rs | 5 | Warm worker reuse, utseg/translate/coref |
| `test_dp_golden.py` | Py | 8 | DP alignment golden |
| `hk/test_integration.py` | Py | 23 | HK engines with real models |

### Layer 7: CLI Parsing & Infrastructure (≈160 tests)

**What:** Clap argument parsing, daemon state management, version checks,
TUI rendering, output path handling, error codes.

**Verdict:** Fine. These are stable and catch regressions in the CLI surface.

| Area | Count | What |
|------|-------|------|
| CLI parsing | 29 | Argument parsing, help text |
| Daemon management | 14 | State file, health checks |
| TUI dashboard | 28 | AppState reducer, rendering |
| Output handling | 13 | Path traversal protection |
| HTTP client | 10 | Health, submit, polling |
| Error codes | 4 | Exit code mapping |
| Compat contracts | 6 | Legacy option aliases |
| Other infra | ~55 | Version, CI checks, struct audits |

---

## Critical Gaps

### Gap 1: GPU Concurrent Dispatch — ZERO TESTS

**Impact:** This is the exact failure mode from Mar 19. `SharedGpuWorker`
multiplexes concurrent requests over a single stdio pipe with hand-rolled
response routing. It has:
- 0 tests sending concurrent requests
- 0 tests verifying response routing by request_id
- 0 tests for reader task failure recovery
- 0 tests proving multiple requests share the same PID/model

**What's needed:**
```
test_gpu_concurrent_dispatch_returns_all_responses
test_gpu_concurrent_dispatch_shares_same_worker_pid
test_gpu_reader_task_failure_fails_all_pending_requests
test_gpu_response_routing_handles_out_of_order_responses
```

### Gap 2: Transcribe End-to-End Through Pool Dispatch — ZERO TESTS

**Impact:** The transcribe command (the most complex pipeline: ASR → speaker
→ FA → post-processing) has golden tests that verify the output format but
none that exercise the actual server dispatch path including worker pool
checkout, SharedGpuWorker routing, retry logic, and timeout handling.

The golden_audio tests use `LiveServerSession` which creates the server
internally — but the path from CLI → daemon → server → pool → worker →
response is never tested as a unit.

### Gap 3: Worker Recovery After Errors — 2 TESTS

**Impact:** `error_paths.rs` has 6 tests but they all verify the *server's*
error response. None verify the *worker's* state after an error — does it
accept the next request correctly, or is it corrupted?

**What's needed:**
```
test_worker_accepts_next_request_after_previous_error
test_worker_accepts_next_request_after_timeout
test_pool_replaces_crashed_worker_transparently
```

### Gap 4: Multi-File Concurrent Processing — 1 PARTIAL TEST

**Impact:** `e2e.rs::e2e_multiple_files` tests that 3 files complete but
doesn't verify they run concurrently (could be sequential and still pass).
The Mar 19 bug manifested specifically when multiple files were dispatched
concurrently to the GPU worker.

### Gap 5: Command × Engine Matrix Integration — NO TESTS

**Impact:** Each command can use multiple engines (e.g., transcribe with
Whisper vs. Rev.AI vs. Tencent). The engine selection, credential injection,
worker routing, and response parsing differ per engine. The
`test_worker_execute_v2_matrix.py` tests verify the dispatch *routing* with
fakes but never test with real engines.

| Command | Engines | Integration tests? |
|---------|---------|-------------------|
| transcribe | whisper, rev, tencent, aliyun, funaudio | golden_audio (whisper only) |
| align (FA) | whisper, wave2vec, wav2vec_canto | golden_audio (wave2vec only) |
| align (UTR) | rev, whisper | 0 |
| morphotag | stanza | golden.rs, golden_parity.rs |
| utseg | stanza | golden.rs, golden_parity.rs |
| translate | google, seamless | golden.rs |
| coref | stanza | golden.rs |
| speaker | pyannote | golden_audio (1 test) |
| opensmile | opensmile | golden_audio (1 test) |
| avqi | avqi | 0 |
| benchmark | whisper | 0 (has golden but no dispatch test) |

### Gap 6: Timeout and Retry Behavior — 0 TESTS

**Impact:** The retry logic (3 attempts, exponential backoff) and timeout
classification (`WorkerTimeout` vs `WorkerCrash` vs `WorkerProtocol`) are
tested only via unit tests in `error_classification.rs`. No integration test
verifies that a timed-out request actually gets retried and succeeds on the
second attempt, or that the timeout values are correct in practice.

### Gap 7: Memory Gate — 0 INTEGRATION TESTS

**Impact:** The memory gate (reject jobs when RAM < threshold) has unit tests
for the gating logic but no integration test verifying that a job submission
actually gets deferred under memory pressure.

---

## Tests of Questionable Value

### Redundant Serialization Tests

Many tests verify the same serialization path from different angles:
- `test_worker_ipc.py` tests Python-side serialization
- `json_compat.rs` tests Rust-side serialization
- `test_ipc_type_conformance.py` tests they match
- `worker_protocol_v2_compat.rs` tests the V2 envelope

Four test files for one concern. Could be consolidated into a single
cross-language conformance suite.

### Retired/Deprecated Path Tests

- `test_retired_runtime_paths.py` (3 tests) — verifies old imports are gone.
  These should be deleted once the migration is complete, not kept permanently.
- `test_compat.py` (30+ tests) — tests the BA2 compatibility shim. Will be
  deleted when BA2 compat is removed.

### Overly Narrow Unit Tests

- `test_typed_handles.py` (9 tests) — tests that dataclass fields exist.
  These test Python's `@dataclass` machinery, not our code.
- `test_worker_bootstrap_runtime.py` (1 test) — single test for a 30-line
  function. The function is already tested indirectly by worker_integration.rs.

---

## Recommendations

### Priority 1: Add GPU concurrent dispatch tests

The SharedGpuWorker is the most complex and most fragile code path, with
zero test coverage. Before any refactoring, add tests that:
1. Send N concurrent execute_v2 requests through the pool
2. Verify all N responses arrive with correct request_ids
3. Verify all requests hit the same worker PID
4. Verify behavior when the reader task dies mid-flight

These tests can use test-echo workers (no ML models needed).

### Priority 2: Add transcribe end-to-end test

One test that runs `batchalign3 transcribe` on a short audio file through
the full CLI → daemon → server → pool → worker → result chain. Use a
10-second audio clip to keep it fast. This would have caught the Mar 19 bug.

### Priority 3: Add timeout/retry integration test

A test that introduces an artificial delay in the test-echo handler to
trigger a timeout, then verifies the retry succeeds. This proves the
retry machinery works end-to-end, not just in isolation.

### Priority 4: Consolidate serialization tests

Merge `test_worker_ipc.py`, `test_ipc_type_conformance.py`,
`worker_protocol_v2_compat.rs`, and `json_compat.rs` into one
cross-language conformance suite.

### Priority 5: Add command × engine matrix

Parameterized integration test that runs each (command, engine) pair on
a test fixture, at least on net where all engines are available.

---

## Test Organization

### Current markers

| Marker | Used in | Meaning |
|--------|---------|---------|
| `@pytest.mark.integration` | Python | Spawns subprocess or needs real models |
| `@pytest.mark.golden` | Python | Compares against .expected files |
| `#[ignore]` | Rust | Skipped by default (usually needs ML models) |
| `require_python!()` | Rust | Skip if Python unavailable |
| `LiveServerSession` | Rust | Skip if real models unavailable |

### Suggested marker additions

| Marker | Meaning |
|--------|---------|
| `@pytest.mark.gpu` | Needs GPU worker (test concurrency) |
| `@pytest.mark.audio` | Needs audio test fixtures |
| `@pytest.mark.engine(name)` | Needs specific engine installed |
| `@pytest.mark.slow` | Takes >30s (ML model loading) |

---

## Running Tests

```bash
# All Python tests (fast, no ML models)
uv run pytest                                     # 593 tests, ~15s

# Python integration tests only
uv run pytest -m integration                      # ~10 tests, ~30s

# Python golden tests (needs --update-golden to regenerate)
uv run pytest -m golden                           # 8 tests

# All Rust tests (fast, test-echo workers)
cargo nextest run --workspace                     # ~1,202 tests

# Rust golden tests with real models (slow, needs Stanza installed)
cargo nextest run --workspace -- golden            # ~51 tests, ~5 min

# Rust worker integration tests only
cargo nextest run -p batchalign-app -- worker_     # ~17 tests

# Full verification (fast tests only, no ML models)
make test                                          # Python + Rust
```
