# HK/Cantonese Engines

**Status:** Current
**Last updated:** 2026-03-15

Batchalign includes alternative ASR and forced alignment engines for Hong Kong
Cantonese. These are built-in modules activated via `--engine-overrides` and
installed through optional dependency extras.

## Available Engines

| Engine | Task | Description |
|--------|------|-------------|
| `tencent` | ASR | Tencent Cloud speech recognition with speaker diarization |
| `aliyun` | ASR | Alibaba Cloud NLS real-time speech recognition (Cantonese only) |
| `funaudio` | ASR | FunASR/SenseVoice local model (no cloud credentials needed) |
| `wav2vec_canto` | FA | Cantonese forced alignment with jyutping preprocessing |

## Installation

Install with the extras for the engine you need:

```bash
# Individual engines
uv tool install "batchalign3[hk-tencent]"
uv tool install "batchalign3[hk-aliyun]"
uv tool install "batchalign3[hk-funaudio]"
uv tool install "batchalign3[hk-cantonese-fa]"

# All HK engines at once
uv tool install "batchalign3[hk]"
```

If you installed `batchalign3` as a library in a virtual environment
instead of as a standalone tool, use `uv pip install` in place of
`uv tool install`.

For a source checkout using the repo-managed `.venv`, sync the extras into that
environment instead:

```bash
# From the batchalign3 repo root
uv sync --group dev --extra hk          # all HK engines
uv sync --group dev --extra hk-tencent  # one engine family only
```

`uv sync` is declarative: a later sync only includes the extras named on that
command. If you need multiple HK engines in the same repo venv, include all of
them together (for example `--extra hk`, or the full set of specific extras)
rather than assuming extras accumulate across separate `uv sync` runs.

The core `batchalign3` package includes the Cantonese text normalization engine
(implemented in Rust) regardless of which extras are installed. Only the
provider SDKs (Tencent SDK, Aliyun NLS, FunASR, pycantonese) are optional.

## Usage

Select an HK engine with `--engine-overrides`:

```bash
# Transcribe with Tencent Cloud ASR
batchalign3 transcribe input/ -o output/ --lang yue \
  --engine-overrides '{"asr": "tencent"}'

# Transcribe with FunASR (local, no credentials)
batchalign3 transcribe input/ -o output/ --lang yue \
  --engine-overrides '{"asr": "funaudio"}'

# Benchmark against a gold CHAT companion in the input directory
batchalign3 benchmark input/ --output output/ --lang yue -n 1 \
  --engine-overrides '{"asr": "tencent"}'

# Force align with Cantonese FA engine
batchalign3 align input/ -o output/ --lang yue \
  --engine-overrides '{"fa": "wav2vec_canto"}'
```

## Credential Configuration

Cloud engines (Tencent, Aliyun) require API credentials in
`~/.batchalign.ini`:

### Tencent Cloud

```ini
[asr]
engine.tencent.id = <secret-id>
engine.tencent.key = <secret-key>
engine.tencent.region = ap-guangzhou
engine.tencent.bucket = <cos-bucket-name>
```

### Aliyun NLS

```ini
[asr]
engine.aliyun.ak_id = <access-key-id>
engine.aliyun.ak_secret = <access-key-secret>
engine.aliyun.ak_appkey = <appkey>
```

Missing or empty credentials raise `ConfigError` with a clear message
indicating which keys are needed.

## Cantonese Text Normalization

All Cantonese ASR output is automatically normalized from simplified/mixed
Chinese to Hong Kong Traditional Chinese. This normalization:

1. **Simplified → HK Traditional** via the `zhconv` Rust engine (same rulesets
   as OpenCC + MediaWiki)
2. **Domain-specific corrections** via a 31-entry replacement table for
   Cantonese character variants (e.g., 系→係, 呀→啊, 中意→鍾意)

Normalization is built into the Rust extension (`batchalign_core`) and runs
automatically during ASR post-processing for `lang=yue`. No additional Python
dependencies (like OpenCC) are required.

## Engine Details

### Tencent Cloud ASR

- Supports speaker diarization with configurable speaker count
- Uploads audio to COS (Tencent Cloud Object Storage), submits ASR job, polls
  for results
- 10-minute safety timeout on ASR polling
- Automatic COS cleanup after transcription
- Per-word timestamps with speaker attribution

### Aliyun NLS ASR

- Cantonese only (`lang=yue` required, other languages rejected at load time)
- WebSocket streaming with real-time sentence callbacks
- Automatic token refresh (23-hour TTL)
- WAV format input required (16 kHz mono)
- Shared result shaping and Cantonese fallback tokenization happen in Rust,
  not in the Python transport adapter

### FunASR/SenseVoice

- Local model — no cloud credentials, no network required
- Automatic model selection: Paraformer (standard) or SenseVoice (multilingual)
- VAD (Voice Activity Detection) built in
- Per-character Cantonese tokenization for timestamp alignment

### Cantonese FA

- Converts Chinese characters to jyutping romanization (via pycantonese)
- Strips tones from jyutping (Wave2Vec MMS expects toneless input)
- Runs Wave2Vec forced alignment on the romanized text
- Maps word-level timings back to original Chinese characters

## See Also

- [HK/Cantonese Engines: Migration and Architecture](../architecture/hk-cantonese-engines.md) — migration rationale, current engine architecture, and normalization details
- [Adding Inference Providers](../developer/adding-engines.md) — how to add new built-in engines
