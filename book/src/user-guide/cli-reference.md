# CLI Reference

**Status:** Current
**Last updated:** 2026-03-17

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
| `--workers N` | Maximum worker processes |
| `--force-cpu` | Disable MPS/CUDA and force CPU-only models |
| `--server URL` | Explicit remote server URL |
| `--override-cache` | Bypass the utterance analysis cache |
| `--lazy-audio` / `--no-lazy-audio` | Toggle lazy audio loading for ASR/alignment |
| `--tui` / `--no-tui` | Toggle full-screen TUI |
| `--open-dashboard` / `--no-open-dashboard` | Toggle browser auto-open for submitted job pages |
| `--engine-overrides JSON` | Select built-in alternative engines with a flat `{string:string}` JSON object; invalid JSON is rejected |

Compatibility no-op flags such as `--use-cache` and `--no-force-cpu` are still
accepted but should not be used in new docs or new scripts.

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
```

Key options:

| Option | Meaning |
| --- | --- |
| `--utr-engine {rev,whisper}` | UTR engine |
| `--utr-engine-custom NAME` | Explicit custom UTR engine |
| `--utr-strategy {auto,global,two-pass}` | Overlap strategy for `+<` utterances during UTR |
| `--fa-engine {wav2vec,whisper}` | Forced-alignment engine |
| `--fa-engine-custom NAME` | Explicit custom FA engine |
| `--utr` / `--no-utr` | Enable or skip utterance timing recovery |
| `--wor` / `--nowor` | Toggle `%wor` tier output |
| `--pauses` | Group words into pause-separated chunks |
| `--merge-abbrev` | Merge abbreviations in output |

### `transcribe`

```bash
batchalign3 transcribe recordings/ -o transcripts/ --lang eng
batchalign3 transcribe interview.wav -o out/
```

Key options:

| Option | Meaning |
| --- | --- |
| `--lang CODE` | 3-letter ISO language code (default: `eng`) |
| `-n`, `--num-speakers N` | Number of speakers (default: `2`) |
| `--asr-engine {rev,whisper,whisperx,whisper-oai}` | ASR engine |
| `--asr-engine-custom NAME` | Explicit custom ASR engine |
| `--diarization {auto,enabled,disabled}` | Speaker diarization mode |
| `--wor` / `--nowor` | Toggle `%wor` tier output |
| `--batch-size N` | Whisper batch size |
| `--merge-abbrev` | Merge abbreviations in output |

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
| `--retokenize` / `--keeptokens` | Retokenize main lines or preserve current tokenization |
| `--skipmultilang` / `--multilang` | Skip or keep multilingual spans |
| `--lexicon FILE` | Manual lexicon override file |
| `--merge-abbrev` | Merge abbreviations in output |

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
| `--merge-abbrev` | Merge abbreviations in output |

### `coref`

```bash
batchalign3 coref corpus/ -o coref-output/
```

Adds sparse `%xcoref` coreference annotation tiers.

Key options:

| Option | Meaning |
| --- | --- |
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
| `--wor` / `--nowor` | Toggle `%wor` tier output |
| `--merge-abbrev` | Merge abbreviations in output |
| `--bank NAME` | Server media bank name |
| `--subdir PATH` | Subdirectory under the bank |

### `opensmile`

```bash
batchalign3 opensmile input_dir/ output_dir/
```

Extracts acoustic features from audio files, producing `.opensmile.csv`
output (not CHAT). Uses positional arguments for input/output directories.

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
| `--merge-abbrev` | Merge abbreviations in output |

### `avqi`

```bash
batchalign3 avqi input_dir/ output_dir/
```

Calculates Acoustic Voice Quality Index from paired `.cs`/`.sv` audio
files. Uses positional arguments for input/output directories. Produces
`.avqi.txt` output.

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
