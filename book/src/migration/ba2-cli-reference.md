# Batchalign2 CLI Reference (Baseline)

**Status:** Reference
**Last updated:** 2026-03-16

This document captures the complete CLI surface of Batchalign2 at the
`84ad500b` baseline (2026-01-09) — the final optimization push before the
BA3 migration. It serves as the permanent regression test baseline and
historical record for BA2→BA3 parity verification.

**Source:** `batchalign/cli/cli.py` using `rich_click` (Click wrapper).

---

## Global Options

These are defined on the top-level `batchalign` group and available to all
commands. BA3 preserves them as hidden no-ops (except `--verbose`, `--workers`,
and `--force-cpu` which are wired).

| Flag | Type | Default | BA3 Status |
|------|------|---------|------------|
| `-v` / `--verbose` | count | `0` | Wired (global verbosity) |
| `--workers` | int | `os.cpu_count()` | Wired (worker count) |
| `--memlog` | flag | off | Hidden no-op |
| `--mem-guard` / `--no-mem-guard` | flag | off | Hidden no-op |
| `--adaptive-workers` / `--no-adaptive-workers` | bool | `True` | Hidden no-op |
| `--pool` / `--no-pool` | bool | `True` | Hidden no-op |
| `--lazy-audio` / `--no-lazy-audio` | bool | `True` | Wired |
| `--adaptive-safety-factor` | float | `1.35` | Hidden no-op |
| `--adaptive-warmup` | int | `2` | Hidden no-op |
| `--force-cpu` / `--no-force-cpu` | bool | `False` | Wired (environment var) |
| `--shared-models` / `--no-shared-models` | bool | `False` | Hidden no-op |

All commands except `avqi`, `setup`, and `version` use the `common_options`
decorator which adds positional `IN_DIR` and `OUT_DIR` arguments (both
`click.Path(exists=True, file_okay=False)`).

---

## Processing Commands

### `align`

Forced alignment: adds word-level timing bullets to existing CHAT transcripts.

| Flag | Type | Default | Help | BA3 Status |
|------|------|---------|------|------------|
| `--whisper` / `--rev` | exclusive pair | `--rev` | UTR engine selection | Hidden compat alias → `--utr-engine` |
| `--wav2vec` / `--whisper_fa` | exclusive pair | `--wav2vec` | FA engine selection | Hidden compat alias → `--fa-engine` |
| `--pauses` | flag | off | Add pauses between words | Wired |
| `--wor` / `--nowor` | bool | `True` | Write %wor tier | Wired (`default_value_t = true`) |
| `--merge-abbrev` / `--no-merge-abbrev` | bool | `False` | Merge abbreviations | Wired |

**Pipeline task:** `"fa"` (forced alignment).

### `transcribe`

Create transcripts from audio files via ASR.

| Flag | Type | Default | Help | BA3 Status |
|------|------|---------|------|------------|
| `--whisper_oai` / `--rev` | exclusive pair | `--rev` | ASR engine (OAI variant) | Hidden compat alias → `--asr-engine` |
| `--whisper` / `--rev` | exclusive pair | `--rev` | ASR engine (HF variant) | Hidden compat alias → `--asr-engine` |
| `--whisperx` / `--rev` | exclusive pair | `--rev` | ASR engine (WhisperX variant) | Hidden compat alias → `--asr-engine` |
| `--diarize` / `--nodiarize` | bool | `False` | Speaker diarization | Hidden compat alias → `--diarization` |
| `--wor` / `--nowor` | bool | `False` | Write %wor tier | **Regression (R1):** parsed but not consumed |
| `--merge-abbrev` / `--no-merge-abbrev` | bool | `False` | Merge abbreviations | Wired |
| `--lang` | str | `"eng"` | Language code | Wired |
| `-n` / `--num_speakers` | int | `2` | Expected speaker count | Wired |

**Pipeline task:** `"asr"` (without diarization) or `"asr"` + speaker pipeline (with diarization).

**Note on `--wor`:** BA2 default was `False` (no %wor). ASR provides word-level
timing, so %wor generation is technically possible. BA3 unconditionally generates
%wor when timing exists — the `--wor`/`--nowor` toggle is silently ignored.

### `morphotag`

Morphosyntactic analysis (POS, lemma, dependency parse).

| Flag | Type | Default | Help | BA3 Status |
|------|------|---------|------|------------|
| `--retokenize` / `--keeptokens` | bool | `False` | Retokenize main line for UD | Wired |
| `--skipmultilang` / `--multilang` | bool | `False` | Skip multilingual files | Wired |
| `--lexicon` | path | `None` | Manual lexicon override | Wired |
| `--override-cache` / `--use-cache` | bool | `False` | Bypass analysis cache | Wired |
| `--merge-abbrev` / `--no-merge-abbrev` | bool | `False` | Merge abbreviations | Wired |

**Pipeline task:** `"morphosyntax"`.

### `translate`

Translation to English.

| Flag | Type | Default | Help | BA3 Status |
|------|------|---------|------|------------|
| `--merge-abbrev` / `--no-merge-abbrev` | bool | `False` | Merge abbreviations | Wired |

**Pipeline task:** `"translate"`.

### `coref` (hidden)

Coreference resolution. Hidden from `--help` in BA2.

| Flag | Type | Default | Help | BA3 Status |
|------|------|---------|------|------------|
| `--merge-abbrev` / `--no-merge-abbrev` | bool | `False` | Merge abbreviations | Wired |

**Pipeline task:** `"coref"`.

### `utseg`

Utterance segmentation.

| Flag | Type | Default | Help | BA3 Status |
|------|------|---------|------|------------|
| `--lang` | str | `"eng"` | Language code | Wired |
| `-n` / `--num_speakers` | int | `2` | Expected speaker count | Wired |
| `--merge-abbrev` / `--no-merge-abbrev` | bool | `False` | Merge abbreviations | Wired |

**Pipeline task:** `"utseg"`.

### `benchmark`

ASR word error rate benchmarking against gold transcripts.

| Flag | Type | Default | Help | BA3 Status |
|------|------|---------|------|------------|
| `--whisper` / `--rev` | exclusive pair | `--rev` | ASR engine (HF variant) | Hidden compat alias → `--asr-engine` |
| `--whisper_oai` / `--rev` | exclusive pair | `--rev` | ASR engine (OAI variant) | Hidden compat alias → `--asr-engine` |
| `--lang` | str | `"eng"` | Language code | Wired |
| `-n` / `--num_speakers` | int | `2` | Expected speaker count | Wired |
| `--wor` / `--nowor` | bool | `False` | Write %wor tier | **Regression (R2):** parsed but not consumed |
| `--merge-abbrev` / `--no-merge-abbrev` | bool | `False` | Merge abbreviations | Wired |

**Pipeline task:** `"asr"` (transcribe) + `"morphosyntax"` (compare).

### `compare`

Transcript comparison against gold-standard references.

| Flag | Type | Default | Help | BA3 Status |
|------|------|---------|------|------------|
| `--lang` | str | `"eng"` | Language code | **Regression (R3):** missing from BA3 |
| `--merge-abbrev` / `--no-merge-abbrev` | bool | `False` | Merge abbreviations | Wired |

**Pipeline task:** `"morphosyntax"` (compare uses morphosyntax to tag both transcripts before WER computation).

**Note on `--lang`:** BA2 passed `--lang` to the compare pipeline for
morphosyntax. BA3 `CompareArgs` has no `--lang` field; `command_meta()`
hardcodes `"eng"`. Non-English compare is broken.

### `opensmile`

OpenSMILE acoustic feature extraction.

| Flag | Type | Default | Help | BA3 Status |
|------|------|---------|------|------------|
| `--feature-set` | choice | `"eGeMAPSv02"` | eGeMAPSv02 / eGeMAPSv01b / GeMAPSv01b / ComParE_2016 | Wired |
| `--lang` | str | `"eng"` | Language code | Wired |

**Note:** Uses its own `input_dir`/`output_dir` positional arguments instead of
`common_options`.

**Pipeline task:** `"opensmile"`.

### `avqi`

Acoustic Voice Quality Index from paired .cs/.sv audio files.

| Flag | Type | Default | Help | BA3 Status |
|------|------|---------|------|------------|
| `--lang` | str | `"eng"` | Language code | Wired |

**Note:** Uses its own `input_dir`/`output_dir` positional arguments instead of
`common_options`.

**Pipeline task:** `"avqi"`.

---

## Admin Commands

### `setup`

Interactive configuration wizard. Creates/updates `~/.batchalign.ini` with
default ASR engine and Rev.AI API key.

No command-specific flags in BA2. BA3 adds `--engine`, `--rev-key`, and
`--non-interactive` for scripted setup.

### `version`

Prints version and credits via `pyfiglet`.

No flags. BA3 equivalent: `batchalign3 version`.

---

## Utility Commands

### `cache`

Cache management. Registered as an external Click subcommand from the
BA2 `cache` CLI module.

BA2 subcommands: `stats`, `clear`, `warm`. BA3 supports `stats` and `clear`
(with `--all` and `--yes` options).

### `bench`

Repeated benchmark execution for performance measurement. Registered as
an external Click subcommand from the BA2 `bench` CLI module.

BA3 equivalent: `batchalign3 bench <command> <in_dir> <out_dir> --runs N`.

### `models`

Model training utilities. Registered via `add_command` from
`batchalign.models.training.run`.

BA2 subcommand: `train`. BA3 adds `prep` (Rust-native training text extraction)
alongside `train` (Python runtime).

---

## batchalignHK Plugin (Archived)

The HK plugin was a separate PyPI package (`batchalign-hk-plugin`) that
registered additional ASR/FA engines via Python entry points. It was folded
into batchalign3 as built-in engines in March 2026; there is no separate HK
install tier now.

### Plugin Discovery

BA2 used `importlib.metadata.entry_points(group="batchalign.inference")` to
discover plugin-provided `InferenceProvider` implementations at startup. Each
provider registered `PluginDescriptor` objects declaring engine name, task type,
and factory function.

### Engines

| Engine | Task | Module | Credentials |
|--------|------|--------|-------------|
| `tencent` | ASR | `batchalign_hk.tencent_asr` | Tencent Cloud API key |
| `aliyun` | ASR | `batchalign_hk.aliyun_asr` | Aliyun NLS API key |
| `funaudio` | ASR | `batchalign_hk.funaudio_asr` | None (local model) |
| `wav2vec_canto` | FA | `batchalign_hk.cantonese_fa` | None (local model) |

### Selection

Engines were selected via `--engine-overrides '{"asr": "tencent"}'` on the
CLI. The JSON payload was parsed into a `BTreeMap<String, String>` and
forwarded to worker dispatch, which matched the engine name against plugin
registrations.

### BA3 Status

All four engines are now built-in modules under `batchalign/inference/hk/`.
Engine dispatch uses `AsrEngine`/`FaEngine` enums in `worker/_types.py`.
The plugin discovery mechanism (`PluginDescriptor`, `InferenceProvider`,
entry points) has been completely removed. See
[Plugin Removal Notes](../developer/plugins.md) for the full migration record.

---

## Pipeline Task Mapping

| Command | Pipeline Task String | Notes |
|---------|---------------------|-------|
| `align` | `"fa"` | Forced alignment |
| `transcribe` | `"asr"` | Without diarization |
| `transcribe` (diarized) | `"asr"` + speaker | With `--diarize` |
| `morphotag` | `"morphosyntax"` | POS + lemma + depparse |
| `translate` | `"translate"` | Google Translate / Seamless M4T |
| `coref` | `"coref"` | English only, document-level |
| `utseg` | `"utseg"` | Constituency parse → boundaries |
| `benchmark` | `"asr"` + `"morphosyntax"` | Transcribe then compare |
| `compare` | `"morphosyntax"` | Tags both sides before WER |
| `opensmile` | `"opensmile"` | Feature extraction |
| `avqi` | `"avqi"` | Voice quality index |

---

## Regression Summary

| ID | Command | Flag | Severity | Description |
|----|---------|------|----------|-------------|
| R1 | `transcribe` | `--wor`/`--nowor` | Critical | Parsed but not consumed; %wor unconditionally generated |
| R2 | `benchmark` | `--wor`/`--nowor` | Medium | Parsed but not consumed; same root cause as R1 |
| R3 | `compare` | `--lang` | Medium | Missing from BA3; hardcoded to `"eng"` |
