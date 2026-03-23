# Apple MPS Workarounds

**Status:** Current
**Last updated:** 2026-03-23 18:44 EDT

Apple's Metal Performance Shaders (MPS) backend in PyTorch provides GPU
acceleration on Apple Silicon but has significant limitations that require
explicit workarounds throughout our inference code. This page is the single
reference for all MPS issues we've encountered, the workarounds we've applied,
and the upstream issues to watch for fixes.

Our primary deployment target is `net` — a Mac Studio with an M3 Ultra and
256 GB RAM. Every model loader must work correctly on MPS.

## Hardware Limitations

Metal (Apple's GPU framework) does **not** support:

| Type | Status | PyTorch behavior |
|------|--------|-----------------|
| **bfloat16** | Not in Metal spec | Crashes, wrong results, or `TypeError` depending on operation |
| **float64** | Not in Metal spec | `TypeError: Cannot convert Double to MPS` |
| **int64** | Not in Metal spec | Crashes on some ops (e.g. `abs_out_mps`) |
| **complex128** | Not in Metal spec | Conversion failure |

These are hardware/framework limitations, not PyTorch bugs. No fix is expected.

## Global Workaround

```python
# batchalign/__init__.py
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = str(1)
```

Set at package import time. When an MPS operation isn't implemented, PyTorch
falls back to CPU for that operation instead of crashing. This is a safety net,
not a substitute for explicit dtype control — fallback operations are slow and
can produce unexpected device transfers.

## Per-Module Workarounds

### ASR — Whisper (`inference/asr.py`)

```python
if device.type == "mps":
    asr_dtype = torch.float32   # not bfloat16
```

Whisper ASR uses `bfloat16` on CUDA for speed. On MPS, this crashes with Metal
assertion failures. We force `float32`. A second fallback path also forces
`float32` on MPS for older transformers versions that don't accept `bfloat16`
at all.

The HuggingFace Transformers Whisper pipeline requires
`attn_implementation="eager"` on MPS — the SDPA attention path broke MPS in
transformers v4.40.0.

### Forced Alignment — Whisper FA (`inference/fa.py`)

```python
if device.type == "mps":
    torch_dtype = torch.float32   # not float16
```

Same pattern as ASR. Whisper FA uses `float16` on CUDA, `float32` on MPS/CPU.

### Forced Alignment — Wave2Vec FA (`inference/fa.py`)

```python
model = bundle.get_model()
if device.type == "mps":
    model = model.float()  # Force float32
model = model.to(device)
```

The torchaudio `MMS_FA` bundle's default parameters can include bfloat16 ops
on MPS. Under concurrent load with large audio files (200+ MB video → WAV →
inference), this causes worker crashes that surface as `Broken pipe (os error
32)`. The `.float()` call converts all parameters to float32 before moving to
device.

**Incident:** 2026-03-16, Davida's aphasia-data ACWT job — 6/11 files failed.
See `docs/postmortems/2026-03-16-wave2vec-mps-crash.md`.

### Speaker Diarization (`inference/speaker.py`)

```python
return "cuda" if torch.cuda.is_available() else "mps" if ... else "cpu"
```

Speaker diarization now explicitly falls back to CPU when CUDA is unavailable.
MPS is never used for diarization because:

- **Pyannote on MPS** produces wrong timestamps
  ([pyannote/pyannote-audio#1337](https://github.com/pyannote/pyannote-audio/issues/1337),
  closed as wontfix). Kernel crashes also reported on M4
  ([#1886](https://github.com/pyannote/pyannote-audio/issues/1886)).
- **NeMo** is CUDA-only by design — no MPS support at all.

The device selector (`_device_for_speaker_runtime`) returns `"cuda"` or
`"cpu"`, never `"mps"`.

### Device Policy (`device.py`)

The `BATCHALIGN_FORCE_CPU` environment variable (or `DevicePolicy(force_cpu=True)`)
forces all model loaders onto CPU. This is the escape hatch when MPS causes
problems that dtype coercion alone can't fix.

## Memory Issues on MPS

MPS has well-documented memory management problems:

- **Memory leaks** during inference: usage climbs steadily, eventually OOM
  ([pytorch/pytorch#154329](https://github.com/pytorch/pytorch/issues/154329),
  [#145374](https://github.com/pytorch/pytorch/issues/145374))
- **OOM with memory available**: MPS cache doesn't release when it should
  ([pytorch/pytorch#105839](https://github.com/pytorch/pytorch/issues/105839))
- **`sysinfo::available_memory()`** on macOS undercounts — reports only
  free + purgeable, missing reclaimable file cache. On net (256 GB, heavy I/O),
  this can underreport by tens of GB. No fix exists because macOS doesn't
  expose a `MemAvailable` equivalent like Linux.

**Mitigations:**
- `torch.mps.empty_cache()` — call periodically during long-running inference
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` — disables MPS memory limit (risks
  system instability, not recommended for production)
- Our Rust server's memory gate uses `sysinfo::available_memory()` with a
  configurable threshold (default 2048 MB, `0` to disable). Idle worker bypass
  prevents deadlock when loaded workers hold RAM.

## Upstream Issues to Track

Check these periodically. If an issue is resolved, we may be able to remove
the corresponding workaround.

### bfloat16

| Issue | Status | What to do if fixed |
|-------|--------|-------------------|
| [pytorch/pytorch#141864](https://github.com/pytorch/pytorch/issues/141864) | Closed (won't fix) | N/A — Metal lacks native bfloat16. Would require Apple hardware/firmware change. |
| [pytorch/pytorch#136624](https://github.com/pytorch/pytorch/issues/136624) | Closed | Specific to `torch.arange`; the broader bfloat16 gap remains. |
| [pytorch/pytorch#104191](https://github.com/pytorch/pytorch/issues/104191) | Closed | Specific to `torch.embedding`. |

**Verdict:** bfloat16 on MPS will not be fixed. Our float32 workarounds are permanent.

### Memory

| Issue | Status | What to do if fixed |
|-------|--------|-------------------|
| [pytorch/pytorch#105839](https://github.com/pytorch/pytorch/issues/105839) | Open | MPS OOM with memory available. If fixed, we could remove `empty_cache()` calls. |
| [pytorch/pytorch#154329](https://github.com/pytorch/pytorch/issues/154329) | Open | MPS memory leak during inference. Critical for long-running server. |
| [pytorch/pytorch#145374](https://github.com/pytorch/pytorch/issues/145374) | Open | MPS memory leak in LSTM iterations. |
| [pytorch/pytorch#114096](https://github.com/pytorch/pytorch/issues/114096) | Open | Leak when converting device+type simultaneously via `.to()`. |

### Whisper

| Issue | Status | What to do if fixed |
|-------|--------|-------------------|
| [huggingface/transformers#31408](https://github.com/huggingface/transformers/issues/31408) | Closed | SDPA broke MPS in v4.40.0. Our `attn_implementation="eager"` workaround is for this. Check if later versions fixed SDPA on MPS. |
| [pytorch/pytorch#141774](https://github.com/pytorch/pytorch/issues/141774) | Open | Autocast fails for `scaled_dot_product_attention` on MPS. Related to the SDPA issue above. |
| [pytorch/pytorch#162092](https://github.com/pytorch/pytorch/issues/162092) | Open | Voxtral (Whisper variant) produces gibberish on MPS. |

### Speaker Diarization

| Issue | Status | What to do if fixed |
|-------|--------|-------------------|
| [pyannote/pyannote-audio#1337](https://github.com/pyannote/pyannote-audio/issues/1337) | Closed (wontfix) | Wrong timestamps on MPS. If reversed, we could enable MPS for diarization. |
| [pyannote/pyannote-audio#1886](https://github.com/pyannote/pyannote-audio/issues/1886) | Open | Kernel crash on M4 with MPS. |

### MPS Correctness

| Issue | Status | What to do if fixed |
|-------|--------|-------------------|
| [pytorch/pytorch#134534](https://github.com/pytorch/pytorch/issues/134534) | Open | Model returns wrong tokens on MPS vs CPU. Broad correctness concern. |

## Checklist for New Model Loaders

When adding a new inference module that loads a PyTorch model:

1. **Use `force_cpu_preferred()` as the first check** — respect the operator's
   CPU override.
2. **Force `float32` on MPS** — never use `bfloat16` or `float64`. Use
   `float16` only if the model explicitly supports it on Metal (most don't).
3. **Test the MPS path** — add a parametrized test with
   `(force_cpu, cuda_available, mps_available)` that verifies device selection
   and dtype. See `test_load_wave2vec_fa_forces_float32_on_mps` for the pattern.
4. **Check upstream MPS support** — some libraries (NeMo, older Pyannote)
   don't support MPS at all. Fall back to CPU explicitly.
5. **For HuggingFace models** — check whether the default attention
   implementation works on MPS. Use `attn_implementation="eager"` if SDPA
   is broken.

## Test Coverage

All MPS workarounds are covered by parametrized tests that exercise each
device path without requiring actual hardware:

| Test | File | What it verifies |
|------|------|-----------------|
| `test_load_whisper_fa_selects_device_and_dtype` | `tests/pipelines/fa/test_fa_inference.py` | Whisper FA: float32 on MPS, float16 on CUDA |
| `test_load_wave2vec_fa_forces_float32_on_mps` | `tests/pipelines/fa/test_fa_inference.py` | Wave2Vec FA: `.float()` called on MPS only |
| `test_load_wave2vec_fa_selects_expected_device` | `tests/pipelines/fa/test_fa_inference.py` | Wave2Vec FA: device selection priority |
| `test_load_whisper_asr_uses_mps_and_cantonese_overrides` | `tests/pipelines/asr/test_asr_inference.py` | ASR: float32 on MPS |
| Speaker device selection test | `tests/pipelines/speaker/test_speaker_inference.py` | Speaker: device selection (no dtype check — latent issue) |
