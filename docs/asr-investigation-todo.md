# ASR Quality Investigation: Evidence-Based Improvement Plan

**Status:** Draft
**Last updated:** 2026-03-18

## Motivation

The German alignment experiment showed a 23-second median timing error —
fundamentally misaligned ASR. The Welsh experiment is the original
non-English regression case. We don't know enough about what's happening
inside the ASR pipeline to diagnose or fix these problems. We need a
systematic, evidence-based investigation.

## What We Don't Know

1. **Which Whisper model size are we using?** `tiny`, `base`, `small`,
   `medium`, `large`, `large-v3`? The size dramatically affects non-English
   quality. `large-v3` is ~10x better than `base` on non-English.

2. ~~**Are we passing the language code to Whisper?**~~ YES — verified.
   `gen_kwargs()` in `types.py` passes `"language": lang` to Whisper's
   `generate_kwargs`. Identical to batchalign2-master. This is NOT the
   cause of poor non-English ASR.

3. **What's the ASR word error rate per language?** We have no baseline WER
   data. We need to measure WER on representative files for each language
   we support before we can evaluate improvements.

4. **Is the timing offset systematic or random?** The German 23s median
   error could be a constant offset (e.g., ASR starts counting from a
   different point than the ground truth) or random drift. Systematic
   offsets are easy to fix; random drift is harder.

5. **What normalization happens between ASR output and the DP alignment?**
   The `asr_postprocess/` pipeline does compound merging, number expansion,
   Cantonese normalization. Are there equivalent normalizations needed for
   German, Welsh, etc.?

6. **Are there language-specific Whisper fine-tunes?** HuggingFace has
   community fine-tuned Whisper models for many languages. Some may be
   substantially better than the base Whisper model on specific languages.

## Investigation Plan

### Phase 1: Baseline WER Measurement

For each language with test files (English, German, Welsh, Hakka, Serbian,
Spanish), measure:
- WER of ASR output against ground truth transcript
- Timing offset distribution (systematic vs random)
- Word-level timing accuracy

**Method:** Use existing `batchalign3 benchmark` command or build a custom
WER tool that also reports timing statistics.

**Files:** Use the multilang experiment files (fusser12, german050814, etc.)
which have ground truth transcripts.

### Phase 2: Configuration Audit

Document the current ASR configuration:
- Whisper model size
- Language detection vs explicit language code
- Beam size, temperature, and other hyperparameters
- Audio preprocessing (sample rate, format conversion)
- Chunking strategy for long files

**Where:** Check `batchalign/inference/asr/` (Python ML server) and
`batchalign-chat-ops/asr_postprocess/` (Rust post-processing).

### Phase 3: Controlled Experiments

For each language, test:
1. **Model size:** Compare `small`, `medium`, `large-v3` on the same file
2. **Language code:** Compare auto-detect vs explicit `--language` parameter
3. **Initial prompt:** Test whether a transcript excerpt as `initial_prompt`
   improves accuracy
4. **Fine-tuned models:** Test language-specific HF models if available
5. **Alternative engines:** Rev.AI, Azure Speech, Deepgram for each language

### Phase 4: Per-Language Post-Processing

Investigate whether language-specific text normalization is needed:
- German: compound word handling, umlaut normalization
- Welsh: mutation handling (soft/nasal/aspirate mutations)
- Hakka: character script normalization (already done for Cantonese)
- Serbian: Latin/Cyrillic script handling

### Phase 5: Infrastructure

- **Per-language WER dashboard:** Automated WER measurement on a standard
  test set, tracked over time
- **Language-specific engine routing:** Route to the best available engine
  per language (e.g., HK engines for Cantonese, Whisper for English)
- **User-visible language support matrix:** Which languages work well,
  which are known-poor, what to expect

## Key Questions for Brian

1. Which languages are highest priority for improvement? (Beyond English)
2. Are there existing WER benchmarks for TalkBank's ASR on specific languages?
3. Is there budget for commercial ASR APIs (Azure, Deepgram) for non-English?
4. Are there collaborators with language-specific ASR expertise we should consult?

## Related

- The 23-second German median error may be a timing offset issue, not a
  recognition quality issue. The ASR might be producing correct words but
  with timestamps that don't match the ground truth coordinate system.
- The fuzzy matching experiment showed no improvement on German because the
  issue is timing, not word matching.
- Cantonese already has language-specific engines (Tencent, Aliyun, FunASR)
  — this pattern could be extended to other languages.
