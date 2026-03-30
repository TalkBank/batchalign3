# Batchalign3 User Guide

**Status:** Current
**Last updated:** 2026-03-29 17:58 EDT

## What is batchalign3?

Batchalign3 is a language sample analysis tool from the
[TalkBank](https://talkbank.org/) project. It processes conversation
transcripts in [CHAT format](https://talkbank.org/0info/manuals/CHAT.html),
providing:

- **Morphosyntactic analysis** (POS tagging, lemmatization, dependency parse)
- **Forced alignment** (word-level timing from audio)
- **Transcription** (speech-to-text from audio)
- **Utterance segmentation** (splitting continuous text into utterances)
- **Translation** (to English)
- **Coreference resolution**
- **Audio feature extraction** (openSMILE, AVQI)

## Installation

Batchalign3 is a Rust+Python hybrid package. Install it with
[uv](https://docs.astral.sh/uv/) (the fast Python package manager).

### From PyPI (when available)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install batchalign3
uv tool install batchalign3

# Verify
batchalign3 --version
```

### From a GitHub release (pre-built wheel)

Download the wheel for your platform from
[GitHub Releases](https://github.com/TalkBank/batchalign3/releases),
then install:

```bash
# macOS Apple Silicon
uv tool install batchalign3-1.0.0-cp312-cp312-macosx_11_0_arm64.whl

# macOS Intel
uv tool install batchalign3-1.0.0-cp312-cp312-macosx_10_12_x86_64.whl

# Linux x86_64
uv tool install batchalign3-1.0.0-cp312-cp312-manylinux_2_17_x86_64.whl

# Windows
uv tool install batchalign3-1.0.0-cp312-cp312-win_amd64.whl
```

### From GitHub source (developers only)

Requires Rust toolchain and the `talkbank-tools` repo cloned alongside:

```bash
uv tool install "batchalign3 @ git+https://github.com/TalkBank/batchalign3"
```

### With optional extras

For Hong Kong Cantonese ASR engines, add the `hk` extra:

```bash
# From PyPI
uv tool install "batchalign3[hk]"

# From wheel
uv tool install "batchalign3[hk] @ batchalign3-1.0.0-cp312-cp312-macosx_11_0_arm64.whl"
```

### Upgrading

```bash
uv tool upgrade batchalign3
```

## Quick Start

No configuration needed. Batchalign3 works out of the box in **direct mode**
— all processing happens locally on your machine.

### Add morphosyntax to a CHAT file

```bash
batchalign3 morphotag transcript.cha -o output/
```

This reads `transcript.cha`, runs Stanza NLP models, and writes the output
with `%mor` and `%gra` tiers to `output/transcript.cha`.

On first run, Stanza models are downloaded automatically (~500 MB for English).
Subsequent runs reuse the cached models.

### Process a directory of files

```bash
batchalign3 morphotag corpus/ -o output/
```

All `.cha` files in `corpus/` (recursive) are processed. Output preserves the
directory structure.

### Modify files in place

```bash
batchalign3 morphotag corpus/ --in-place
```

Overwrites the original files with annotated versions. Use with care.

## Commands

### Text commands (no audio needed)

These work on any machine — laptop, desktop, server. No special hardware
required. Models load in ~4 seconds on first use.

| Command | What it does | Example |
|---------|-------------|---------|
| `morphotag` | Add %mor/%gra tiers (POS, lemma, dependency parse) | `batchalign3 morphotag file.cha -o out/` |
| `utseg` | Split unsegmented text into utterances | `batchalign3 utseg file.cha -o out/` |
| `translate` | Add %xtra tier with English translation | `batchalign3 translate file.cha -o out/` |
| `coref` | Add %xcoref tier with coreference chains | `batchalign3 coref file.cha -o out/` |

### Audio commands (need audio files)

These require the corresponding audio file to be accessible. They load
Whisper/Wave2Vec models (~2-6 GB RAM per concurrent file).

| Command | What it does | Example |
|---------|-------------|---------|
| `align` | Add word-level timing (forced alignment) | `batchalign3 align file.cha -o out/` |
| `transcribe` | Create CHAT transcript from audio | `batchalign3 transcribe audio.wav -o out/` |
| `benchmark` | Compare transcription against gold standard | `batchalign3 benchmark file.cha -o out/` |
| `opensmile` | Extract acoustic features | `batchalign3 opensmile audio.wav -o out/` |
| `avqi` | Calculate voice quality index | `batchalign3 avqi --cs file.cs --sv file.sv -o out/` |

### Useful options

```bash
# Process multiple inputs (files and directories can be mixed)
batchalign3 morphotag file1.cha file2.cha corpus_dir/ -o out/

# Increase verbosity (see worker spawns, timing, cache hits)
batchalign3 morphotag file.cha -o out/ -v

# Even more verbose
batchalign3 morphotag file.cha -o out/ -vv

# Limit concurrent files (prevent OOM on small machines)
batchalign3 align corpus/ -o out/ --workers 1

# Force CPU (disable GPU/MPS acceleration)
batchalign3 align file.cha -o out/ --force-cpu

# Increase timeout for very long recordings (default: 30 minutes)
batchalign3 align long_recording.cha -o out/ --timeout 7200
```

### Align options

```bash
# Include word-level timing tier (%wor)
batchalign3 align file.cha -o out/ --wor

# Use Whisper for forced alignment (default: Wave2Vec)
batchalign3 align file.cha -o out/ --fa-engine whisper

# Enable utterance timing recovery (for untimed files)
batchalign3 align file.cha -o out/ --utr

# Specify media directory explicitly
batchalign3 align file.cha -o out/ --media-dir /path/to/audio/
```

### Transcribe options

```bash
# Use Whisper ASR (default: Rev.AI if configured, else Whisper)
batchalign3 transcribe audio.wav -o out/

# With speaker diarization (who is speaking)
batchalign3 transcribe audio.wav -o out/ --diarize

# With word-level timing
batchalign3 transcribe audio.wav -o out/ --wor
```

## How It Works

### Automatic server detection

Batchalign3 automatically detects if a server is running on your
machine. If it finds one, it routes work through the server — giving you
fleet benefits (warm models, distributed processing, crash recovery)
without any extra flags:

```
$ batchalign3 morphotag corpus/ -o out/
Using local server at http://127.0.0.1:8001 (3 workers available)
```

If no server is running, it falls back to direct mode (in-process):

```
$ batchalign3 morphotag corpus/ -o out/
Running locally (direct mode)...
```

To force direct mode even when a server is available:

```bash
batchalign3 --no-server morphotag file.cha -o out/
```

### Direct mode

When no server is running (or `--no-server` is used), the tool:

1. Starts a Python worker process in the background
2. Loads the appropriate ML model (Stanza for text, Whisper for audio)
3. Parses the CHAT file
4. Extracts words/audio for inference
5. Sends structured data to the worker
6. Injects results back into the CHAT AST
7. Serializes the output

The worker stays alive for ~2 minutes after the last job, so subsequent
runs skip model loading. Worker startup takes ~4 seconds for text models
and ~7 seconds for audio models.

No server, no configuration file, no network access needed (except for
Rev.AI cloud ASR, which requires an API key).

### Server mode

For shared infrastructure (lab servers, clusters), batchalign3 can run
as a persistent server. If a server is running locally, batchalign3
uses it automatically. You can also target a remote server explicitly:

```bash
batchalign3 --server http://server:8001 morphotag file.cha -o out/
```

Server mode keeps models warm permanently, eliminating startup latency.
It supports multiple concurrent jobs, job queuing, crash recovery (via
Temporal), and a web dashboard for monitoring.

**When to use `--server`:**
- The audio files are on a remote machine (not your laptop)
- You want to offload heavy processing from a small machine
- You're at home and the server is on the lab network

**When to use `--no-server` (force direct mode):**
- You're developing/debugging and don't want fleet interference
- You want guaranteed local execution

## Language Support

Batchalign3 uses [Stanza](https://stanfordnlp.github.io/stanza/) for
text analysis. All languages supported by Stanza are supported, including:

English, Spanish, French, German, Chinese, Japanese, Korean, Italian,
Portuguese, Dutch, Russian, Arabic, Hindi, and 60+ others.

Models are downloaded automatically on first use for each language.
Language is detected from the CHAT file's `@Languages` header.

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16+ GB |
| Disk | 2 GB (models) | 5+ GB |
| Python | 3.12 | 3.12 |
| OS | macOS, Linux, Windows | macOS (Apple Silicon) |

Audio commands (align, transcribe) need more RAM:
- ~6 GB per concurrent file for Whisper models
- Use `--workers 1` on machines with < 32 GB RAM

## Configuration

### Rev.AI API key (optional)

For cloud-based ASR (higher quality than local Whisper for English):

```bash
batchalign3 setup
# Follow prompts to enter your Rev.AI API key
```

This writes `~/.batchalign.ini` with your credentials.

### Media mappings (optional)

If your CHAT files reference audio in a different directory (common in
corpus workflows), create `~/.batchalign3/server.yaml`:

```yaml
media_mappings:
  my-corpus-data: /path/to/audio/files
```

Batchalign3 detects the corpus name from the file path and looks up
the audio location automatically.

### For TalkBank team members

See `docs/batchalign3-local-commands.md` for fleet-specific instructions
including NFS media mounts and per-machine recommendations.

## Diagnostics

### Check your setup

```bash
batchalign3 doctor
```

Runs pre-flight checks: Python version, model availability, GPU support,
worker spawn test.

### View logs

```bash
batchalign3 logs              # recent runs
batchalign3 logs --export     # export to file
batchalign3 logs --clear      # clear log history
```

### Cache management

```bash
batchalign3 cache status      # show cache size
batchalign3 cache clear       # clear all caches
```

## Troubleshooting

**"Worker ready timeout"** — Model loading took too long. Try `--timeout 600`
or check available RAM. Each model needs 2-6 GB.

**"No media found"** — Audio file not in the same directory as the CHAT file.
Use `--media-dir /path/to/audio/` or configure `media_mappings` in
`~/.batchalign3/server.yaml`.

**"MPS bfloat16 crash"** — Known Apple Silicon issue. Use `--force-cpu` as a
workaround.

**OOM (out of memory)** — Reduce concurrent files with `--workers 1`. For
large corpus runs, use a machine with more RAM or `--server`.
