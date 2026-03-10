# Benchmarks

Batchalign provides a `benchmark` command to evaluate ASR accuracy against
gold transcripts. It transcribes each audio file, compares the result
against the corresponding gold `.cha` transcript, and reports word error
rate (WER).

## What is WER?

Word Error Rate measures how many words the ASR system got wrong compared
to a human-verified reference transcript. Lower is better — 0% means
perfect, 100% means every word was wrong or missing.

## Input requirements

Place audio files (`.mp3`, `.mp4`, `.wav`) and their corresponding gold
`.cha` transcripts in the same input directory. Each audio file must have
a matching `.cha` file with the same stem (e.g., `interview.wav` and
`interview.cha`).

## Example

```bash
batchalign3 benchmark ~/ba_data/input -o ~/ba_data/output --lang eng
```

## Options

| Option | Meaning |
| --- | --- |
| `--asr-engine {rev,whisper,whisper-oai}` | ASR engine (default: rev) |
| `--asr-engine-custom NAME` | Explicit custom ASR engine name |
| `--lang CODE` | 3-letter ISO language code (default: `eng`) |
| `-n`, `--num-speakers N` | Number of speakers (default: `2`) |
| `--wor` / `--nowor` | Toggle `%wor` tier output |
| `--merge-abbrev` | Merge abbreviations in output |
| `--bank NAME` | Server media bank name (for server-side media resolution) |
| `--subdir PATH` | Subdirectory under the bank |

## Output

The command creates new `.cha` files in the output directory containing the
ASR transcript with evaluation metrics. A companion `.compare.csv` file
records aggregate WER, accuracy, and match/insertion/deletion counts.

Example output interpretation:

```
WER: 12.3%  (accuracy: 87.7%)
Matches: 877  Insertions: 45  Deletions: 33  Substitutions: 45
Total reference words: 1000
```

## See also

- [Command I/O Parity](command-io.md) — section 9 for full benchmark dispatch details
- [CLI Reference](../user-guide/cli-reference.md) — benchmark entry in the CLI docs
