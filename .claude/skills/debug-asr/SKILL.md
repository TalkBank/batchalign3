---
name: debug-asr
description: Debug ASR (speech recognition) and forced alignment issues. Use when transcription is wrong, alignment fails, audio won't process, or Rev.AI/Whisper errors occur.
disable-model-invocation: true
allowed-tools: Bash, Read, Glob, Grep, Agent
---

# Debug ASR and Forced Alignment

Diagnose problems with speech recognition and forced alignment. `$ARGUMENTS` describes the symptom or error.

## Architecture

```
Audio file → Rust server (batchalign-app)
  ├── ASR: audio → Python worker (Whisper/Rev.AI) → raw tokens → Rust post-processing
  │         Post-processing: number expansion → compound merging → Cantonese normalization
  │                          → retokenization → long turn splitting → speaker assignment
  ├── FA:  audio + words → Python worker (Whisper/Wave2Vec) → word timings → Rust injection
  └── Speaker diarization: audio → Python worker (NeMo/Pyannote) → speaker segments
```

## Step 1: Identify the Failing Stage

| Symptom | Stage | Key Files |
|---------|-------|-----------|
| No output at all | Worker spawn or audio loading | `batchalign/worker/_main.py` |
| Wrong words / gibberish | ASR model or wrong language | `batchalign/inference/asr.py` |
| Missing/wrong timestamps | Forced alignment | `batchalign/inference/fa.py` |
| Wrong speaker labels | Diarization | `batchalign/inference/speaker.py` |
| Numbers not expanded | Rust post-processing | `crates/batchalign-chat-ops/src/asr_postprocess/` |
| Compounds not merged | Rust post-processing | `crates/batchalign-chat-ops/src/asr_postprocess/` |
| Cantonese text wrong | Rust normalization | `crates/batchalign-chat-ops/src/asr_postprocess/cantonese.rs` |
| Rev.AI timeout/error | Rev.AI API | `batchalign/inference/asr.py` (RevAiAsrEngine) |
| Whisper crash | GPU/memory | Check CUDA/MPS availability |

## Step 2: Test Worker Directly

Bypass the Rust server to isolate the issue:

```bash
# Start ASR worker
uv run python -m batchalign.worker --task asr --lang eng

# Should print {"ready": true, ...}
# Then paste into stdin:
{"op": "capabilities", "id": "test-1"}
```

If worker doesn't start:
```bash
# Check model loading
uv run python -c "
from batchalign.worker._main import load_models
state = load_models('transcribe', 'eng')
print('Models loaded:', [k for k in vars(state) if getattr(state, k) is not None])
"
```

## Step 3: Check Audio

```bash
# Verify audio file is readable
uv run python -c "
from batchalign.inference.audio import load_audio
audio = load_audio('/path/to/audio.wav')
print(f'Duration: {len(audio[0])/audio[1]:.1f}s, Sample rate: {audio[1]}')
"
```

Common audio issues:
- File doesn't exist or path resolution failed
- Unsupported codec (convert to WAV/MP3 first)
- File too large for available memory
- MPS bfloat16 crash — force `torch.float32` on Apple Silicon

## Step 4: Check ASR Output

```bash
# Run ASR directly
uv run python -c "
from batchalign.inference.asr import load_asr_model, batch_infer_asr
model = load_asr_model('eng', engine='whisper')
result = batch_infer_asr(model, ['/path/to/audio.wav'])
for item in result:
    print(item)
"
```

If ASR output looks wrong:
- Check language code (3-letter ISO: `eng`, `spa`, `zho`, etc.)
- Try a different engine: `whisper` vs `revai`
- Check Whisper model size in config

## Step 5: Check Forced Alignment

```bash
# Run FA directly
uv run python -c "
from batchalign.inference.fa import load_fa_model, batch_infer_fa
model = load_fa_model('eng', engine='whisper')
# FA needs audio + reference words
result = batch_infer_fa(model, [{'audio': '/path/to/audio.wav', 'words': ['hello', 'world']}])
for item in result:
    print(item)
"
```

Common FA issues:
- Words don't match audio (wrong transcript)
- Audio too long for model context
- MPS/CUDA device mismatch

## Step 6: Check Rust Post-Processing

If ASR output is correct but final CHAT output is wrong, the bug is in Rust post-processing:

```bash
# Check number expansion
grep -n "expand_numbers" crates/batchalign-chat-ops/src/asr_postprocess/

# Check compound merging
grep -n "merge_compounds" crates/batchalign-chat-ops/src/asr_postprocess/

# Check retokenization
grep -n "retokenize" crates/batchalign-chat-ops/src/

# Check Cantonese normalization
grep -n "cantonese" crates/batchalign-chat-ops/src/asr_postprocess/
```

The Rust ASR post-processing pipeline:
1. Number expansion (e.g., "42" → "forty two")
2. Compound merging (language-specific)
3. Cantonese normalization (if `zho`/`yue`: simplified→HK traditional + domain replacements)
4. Retokenization (character-level DP alignment with Stanza)
5. Long turn splitting
6. Speaker assignment (from diarization segments)

## Step 7: HK/Cantonese Engines

For Cantonese-specific issues:

```bash
# Check which engine is active
grep -n "AsrEngine\|FaEngine" batchalign/worker/_types.py

# Test Cantonese normalization
uv run python -c "
import batchalign_core
print(batchalign_core.normalize_cantonese('你好世界'))
"
```

HK engines require extra dependencies:
```bash
pip install "batchalign3[hk-tencent]"    # Tencent ASR
pip install "batchalign3[hk-aliyun]"     # Aliyun ASR
pip install "batchalign3[hk-funaudio]"   # FunASR
pip install "batchalign3[hk-cantonese-fa]"  # Cantonese FA
```

## Step 8: Rev.AI Specific

```bash
# Check Rev.AI API key
echo $REVAI_API_KEY | head -c 10

# Rev.AI preflight (parallel pre-submission)
grep -n "preflight" batchalign/worker/_handlers.py

# Check Rev.AI metadata tag
grep -n "batchalign3" batchalign/inference/asr.py
```

Rev.AI uses `batchalign3_{stem}` as the metadata tag for job tracking.

## Key Files

| Purpose | Path |
|---------|------|
| ASR inference | `batchalign/inference/asr.py` |
| FA inference | `batchalign/inference/fa.py` |
| Speaker diarization | `batchalign/inference/speaker.py` |
| Audio loading | `batchalign/inference/audio.py` |
| HK/Cantonese engines | `batchalign/inference/hk/` |
| ASR post-processing (Rust) | `crates/batchalign-chat-ops/src/asr_postprocess/` |
| Cantonese normalization (Rust) | `crates/batchalign-chat-ops/src/asr_postprocess/cantonese.rs` |
| FA injection (Rust) | `crates/batchalign-chat-ops/src/fa/` |
| Retokenization (Rust) | `crates/batchalign-chat-ops/src/retokenize/` |
| Worker entry point | `batchalign/worker/_main.py` |
| Worker protocol | `batchalign/worker/_protocol.py` |
