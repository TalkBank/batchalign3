# CLI Reference

**Status:** Current
**Last updated:** 2026-03-24 16:17 EDT

This page documents the current public `batchalign3` CLI surface. For anything
you are scripting against, confirm with `batchalign3 <command> --help`.

For detailed input/output patterns and mutation behavior per command, see
[Command I/O Parity](../reference/command-io.md).

## Command shape

```bash
batchalign3 [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS] [PATHS...]
```

Global options go before the command name.

## Global options

| Option | Meaning |
| --- | --- |
| `-v`, `-vv`, `-vvv` | Increase verbosity |
| `--workers N` | Maximum concurrent files per job (default: auto-tune based on RAM and CPU; capped at 8 for GPU commands) |
| `--force-cpu` | Disable MPS/CUDA and force CPU-only models |
| `--server URL` | Explicit remote server URL |
| `--override-cache` | Bypass the utterance analysis cache |
| `--lazy-audio` / `--no-lazy-audio` | Toggle lazy audio loading for ASR/alignment |
| `--tui` / `--no-tui` | Toggle full-screen TUI |
| `--open-dashboard` / `--no-open-dashboard` | Toggle browser auto-open for submitted job pages (macOS only, interactive TTY only) |
| `--engine-overrides JSON` | Select built-in alternative engines with a flat `{string:string}` JSON object; invalid JSON is rejected |

BA2 compatibility flags (`--memlog`, `--mem-guard`, `--adaptive-workers`,
`--pool`, `--shared-models`, etc.) have been removed. If your scripts use them,
remove them.

## Dashboard browser auto-open

On macOS, when you run a processing command interactively (e.g.,
`batchalign3 transcribe corpus/ output/`), the CLI automatically opens the
job's dashboard page in your default browser. This lets you monitor progress
in real time.

The dashboard auto-open is **only** triggered when:

- Running on macOS (no-op on Linux/Windows)
- stderr is connected to an interactive terminal (TTY)
- `--no-open-dashboard` was not passed
- The `BATCHALIGN_NO_BROWSER` environment variable is not set

It will **not** fire in non-interactive contexts: cron jobs, CI pipelines,
SSH sessions without a display, piped output, or scripts. To suppress it
explicitly in interactive sessions, pass `--no-open-dashboard`.

## Common path-processing options

The core processing commands documented below all accept:

| Option | Meaning |
| --- | --- |
| `PATHS...` | Input files or directories |
| `-o`, `--output DIR` | Output directory |
| `--file-list FILE` | Read input paths from a text file |
| `--in-place` | Modify inputs in place |

When exactly two positional paths are provided, the CLI still accepts the
legacy input/output directory form. For new scripts, prefer `-o/--output`.

## Core processing commands

### `align`

```bash
batchalign3 align corpus/ -o aligned/
batchalign3 align file.cha
batchalign3 align transcripts/ -o out/ --media-dir /path/to/audio/
```

Key options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--media-dir PATH` | (alongside .cha) | Directory containing audio files |
| `--utr-engine {rev,whisper}` | `rev` | UTR ASR engine |
| `--utr-strategy {auto,global,two-pass}` | `auto` | Overlap strategy for UTR. `auto` is language-aware: non-English uses `global`, English uses overlap-marker detection |
| `--utr-fuzzy THRESHOLD` | `0.85` | Fuzzy word matching (Jaro-Winkler). Set to `1.0` for exact only |
| `--utr-ca-markers {enabled,disabled}` | `enabled` | Use CA overlap markers (⌈⌉⌊⌋) for windowing |
| `--utr-density-threshold N` | `0.30` | Max overlap fraction before skipping pass-1 exclusion |
| `--utr-tight-buffer MS` | `500` | Pass-2 tight window buffer (milliseconds) |
| `--fa-engine {wav2vec,whisper}` | `wav2vec` | Forced-alignment engine |
| `--utr` / `--no-utr` | enabled | Enable or skip utterance timing recovery |
| `--wor` / `--nowor` | enabled | Toggle `%wor` tier output |
| `--pauses` | off | Group words into pause-separated chunks |
| `--merge-abbrev` | off | Merge abbreviations in output |

**Fuzzy matching is enabled by default** (threshold 0.85). This uses
Jaro-Winkler similarity to tolerate minor ASR substitutions like
"gonna"/"gona" without sacrificing precision. Set `--utr-fuzzy 1.0` to
disable fuzzy matching and require exact word matches.

All UTR parameters were tuned empirically on SBCSAE (English CA),
Jefferson NB (dense CA), TaiwanHakka (Hakka), and APROCSA (English
aphasia, 22 files, 17K utterances). The defaults work well across all
tested corpora. See [Dynamic Programming](../architecture/dynamic-programming.md)
for detailed explanations with diagrams.

### `transcribe`

```bash
batchalign3 transcribe recordings/ -o transcripts/ --lang eng
batchalign3 transcribe interview.wav -o out/
batchalign3 transcribe bilingual.wav -o out/ --lang auto  # auto-detect language
```

Key options:

| Option | Meaning |
| --- | --- |
| `--lang CODE` | 3-letter ISO language code, or `auto` for auto-detection (default: `eng`) |
| `-n`, `--num-speakers N` | Number of speakers (default: `2`) |
| `--asr-engine {rev,whisper,whisperx,whisper-oai}` | ASR engine |
| `--asr-engine-custom NAME` | Explicit custom ASR engine |
| `--diarization {auto,enabled,disabled}` | Dedicated speaker diarization stage (default: `auto`=disabled) |
| `--wor` / `--nowor` | Toggle `%wor` tier output |
| `--batch-size N` | Whisper batch size |
| `--merge-abbrev` | Merge abbreviations in output |

**Speaker labels:** Speaker labels from the ASR engine (e.g., Rev.AI monologue
speakers) are **always** used when present, even without `--diarization`. The
`--diarization` flag only controls whether a dedicated Pyannote/NeMo speaker
model runs as a separate stage. `auto` and `disabled` both skip the dedicated
stage; `enabled` runs it only when ASR output lacks speaker labels.

**Auto-detect language (`--lang auto`):** When `--lang auto` is specified with
a local Whisper engine (`--asr-engine whisper`), the language parameter is
omitted from Whisper's generation kwargs, letting the model auto-detect the
spoken language from the audio. This is useful for bilingual or code-switched
recordings where forcing a single language causes the model to skip or garble
content in the other language. The multilingual `openai/whisper-large-v3`
model is used (language-specific fine-tuned models like `talkbank/CHATWhisper-en`
are bypassed since they are trained for a single language). Rev.AI handles
auto-detection separately through its own API.

Routing note: explicit remote `--server` is ignored for `transcribe` because
the remote server cannot access client-local audio paths.

### `morphotag`

```bash
batchalign3 morphotag corpus/ -o tagged/
batchalign3 morphotag file.cha
```

Key options:

| Option | Meaning |
| --- | --- |
| `--lang CODE` | Language override (3-letter ISO). Overrides `@Languages` header when set |
| `--retokenize` / `--keeptokens` | Retokenize main lines or preserve current tokenization |
| `--skipmultilang` / `--multilang` | Skip or keep multilingual spans |
| `--lexicon FILE` | Manual lexicon override file |
| `--merge-abbrev` | Merge abbreviations in output |

When `--lang` is omitted, the language is read from the CHAT file's
`@Languages` header (first declared language). Per-utterance `[- lang]`
precodes route individual utterances to language-specific Stanza models
regardless of the file-level language. See
[Per-Utterance Language Routing](../reference/per-utterance-language-routing.md).

## Operational commands

### `setup`

Initialize `~/.batchalign.ini`:

```bash
batchalign3 setup
batchalign3 setup --non-interactive --engine whisper
batchalign3 setup --non-interactive --engine rev --rev-key <KEY>
```

Options:

| Option | Meaning |
| --- | --- |
| `--engine {rev,whisper}` | Persist default ASR engine |
| `--rev-key KEY` | Rev.AI key for non-interactive setup |
| `--non-interactive` | Disable prompts |

### `logs`

```bash
batchalign3 logs
batchalign3 logs --last
batchalign3 logs --export
batchalign3 logs --clear
```

Key options:

| Option | Meaning |
| --- | --- |
| `--last` | Show the most recent run log |
| `--raw` | Raw JSONL output with `--last` |
| `--export` | Zip recent logs |
| `--clear` | Delete log files |
| `--follow` | Tail the newest log file |
| `-n`, `--count N` | Number of recent runs to list |

### `serve`

```bash
batchalign3 serve start --foreground
batchalign3 serve status
batchalign3 serve stop
```

`serve start` key options:

| Option | Meaning |
| --- | --- |
| `--port PORT` | Listen port |
| `--host HOST` | Bind address |
| `--config PATH` | Alternate `server.yaml` path |
| `--python PATH` | Worker Python executable |
| `--foreground` | Do not daemonize |
| `--test-echo` | Start test-echo workers |
| `--warmup-policy {off,minimal,full}` | Warmup preset |
| `--worker-idle-timeout-s N` | Idle worker shutdown timeout |

### `jobs`

```bash
batchalign3 jobs --server http://myserver:8000
batchalign3 jobs --server http://myserver:8000 <JOB_ID>
```

`jobs` requires an explicit server URL, either from `--server` or
`BATCHALIGN_SERVER`.

### `cache`

```bash
batchalign3 cache stats
batchalign3 cache clear --yes
batchalign3 cache clear --all --yes
```

Notes:

- `cache stats` shows analysis/media cache status.
- `cache clear` supports `--all` and `-y/--yes`.
- `BATCHALIGN_ANALYSIS_CACHE_DIR` and `BATCHALIGN_MEDIA_CACHE_DIR` relocate the
  underlying caches for isolated runs.
- BA2-compatible flag forms `cache --stats` and `cache --clear` are still accepted.

### `openapi`

```bash
batchalign3 openapi -o openapi.json
batchalign3 openapi --check --output openapi.json
```

`--check` exits non-zero when the target file does not match the generated
schema.

### `models`

```bash
batchalign3 models [ARGS...]
```

Forwards arguments to the Python model training runtime
(`python -m batchalign.models.training.run`). See
[Models Training Runtime ADR](../decisions/models-training-runtime-adr.md).

### `bench`

```bash
batchalign3 bench morphotag input_dir/ output_dir/ --runs 3
```

Positional arguments:

| Argument | Meaning |
| --- | --- |
| `<COMMAND>` | Command to benchmark |
| `<IN_DIR>` | Input directory |
| `<OUT_DIR>` | Output directory |

## Other processing commands

### `translate`

```bash
batchalign3 translate corpus/ -o translated/
```

Adds a `%xtra` tier with English translation to each utterance.

Key options:

| Option | Meaning |
| --- | --- |
| `--lang CODE` | Language override (3-letter ISO). Overrides `@Languages` header |
| `--merge-abbrev` | Merge abbreviations in output |

### `coref`

```bash
batchalign3 coref corpus/ -o coref-output/
```

Adds sparse `%xcoref` coreference annotation tiers.

Key characteristics:

- English-only analysis; non-English files pass through unchanged.
- Document-level processing with cross-sentence context.
- Best treated as a local-oriented workflow rather than an explicit remote-server command.

Key options:

| Option | Meaning |
| --- | --- |
| `--lang CODE` | Language override (3-letter ISO). Overrides `@Languages` header |
| `--merge-abbrev` | Merge abbreviations in output |

### `utseg`

```bash
batchalign3 utseg corpus/ -o segmented/ --lang eng
```

Re-segments utterance boundaries using Stanza constituency parsing (or
punctuation-based fallback for unsupported languages).

Key options:

| Option | Meaning |
| --- | --- |
| `--lang CODE` | 3-letter ISO language code (default: `eng`) |
| `-n`, `--num-speakers N` | Number of speakers (default: `2`) |
| `--merge-abbrev` | Merge abbreviations in output |

### `benchmark`

```bash
batchalign3 benchmark ~/data/input -o ~/data/output --lang eng
```

Transcribes audio files via ASR and evaluates word error rate (WER)
against gold `.cha` transcripts in the same directory. See
[Benchmarks](../reference/benchmarks.md) for details.

Key options:

| Option | Meaning |
| --- | --- |
| `--asr-engine {rev,whisper,whisper-oai}` | ASR engine (default: rev) |
| `--asr-engine-custom NAME` | Explicit custom ASR engine name |
| `--lang CODE` | 3-letter ISO language code (default: `eng`) |
| `-n`, `--num-speakers N` | Number of speakers (default: `2`) |
| `--wor` / `--nowor` | Include or suppress `%wor` in the benchmark hypothesis CHAT |
| `--merge-abbrev` | Merge abbreviations in output |
| `--bank NAME` | Server media bank name |
| `--subdir PATH` | Subdirectory under the bank |

### `opensmile`

```bash
batchalign3 opensmile input_dir/ output_dir/
```

Extracts acoustic features from audio files, producing `.opensmile.csv`
output (not CHAT). Uses positional arguments for input/output directories.

Migration note: BA3 preserves the same feature sets, but writes a row-oriented
CSV (feature names as columns) rather than BA2's transposed feature-as-row
export.

Key options:

| Option | Meaning |
| --- | --- |
| `--feature-set SET` | Feature set: `eGeMAPSv02` (default), `eGeMAPSv01b`, `GeMAPSv01b`, `ComParE_2016` |
| `--lang CODE` | 3-letter ISO language code (default: `eng`) |
| `--bank NAME` | Server media bank name |
| `--subdir PATH` | Subdirectory under the bank |

### `compare`

```bash
batchalign3 compare corpus/ -o compared/
```

Compares transcripts against gold references (`FILE.gold.cha` in the same
directory) to compute WER and inject per-utterance `%xsrep` alignment
tiers. Also writes a `.compare.csv` metrics file alongside each output.

Key options:

| Option | Meaning |
| --- | --- |
| `--lang CODE` | 3-letter ISO language code (default: `eng`) |
| `-n`, `--num-speakers N` | Number of speakers (default: `2`) |
| `--merge-abbrev` | Merge abbreviations in output |

### `avqi`

```bash
batchalign3 avqi input_dir/ output_dir/
```

Calculates Acoustic Voice Quality Index from paired `.cs`/`.sv` audio
files. Uses positional arguments for input/output directories. Produces
`.avqi.txt` output.

BA3 keeps the same AVQI metrics and text output format while moving the audio
preprocessing behind the typed media-analysis worker boundary.

Key options:

| Option | Meaning |
| --- | --- |
| `--lang CODE` | 3-letter ISO language code (default: `eng`) |

### `version`

```bash
batchalign3 version
```

Prints version and build information.

## Exit codes

`batchalign3` uses stable non-zero exit code categories:

| Code | Meaning |
| --- | --- |
| `2` | Usage/input error |
| `3` | Configuration error |
| `4` | Network/connectivity error |
| `5` | Server/job lifecycle error |
| `6` | Local runtime error |

Exit code `1` is reserved for unexpected failures outside the typed categories.
