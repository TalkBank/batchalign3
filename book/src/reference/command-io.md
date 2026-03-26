# Batchalign Command I/O Parity: Local CLI vs Server

**Status:** Current
**Last updated:** 2026-03-26 14:05 EDT

This document describes the input/output flow for every batchalign command,
comparing local CLI dispatch with the server-based (`--server`) dispatch.
For implementation details, treat the command-owned entrypoints under
`crates/batchalign-app/src/commands/` plus the owning orchestrator modules
(`compare.rs`, `benchmark.rs`, `transcribe/`, `fa/`, and `morphosyntax/`) as
the source of truth for command semantics. The CLI and runner layers should
stay thin.

For each command: what goes in, where it comes from, what gets written, and
whether files are mutated in place.

---

## Global Path Semantics

Most processing commands use shared `CommonOpts`:

```bash
batchalign3 <command> PATH [PATH ...] [-o OUTPUT_DIR] [--file-list FILE] [--in-place]
```

- Inputs can be files and/or directories.
- `-o/--output` omitted means direct-write behavior for mutating commands.
- `--file-list` is its own input mode: the file's contents become the input path set.
- `--in-place` is available on commands that use `CommonOpts`.

Exceptions:

- `batchalign3 opensmile INPUT_DIR OUTPUT_DIR`
- `batchalign3 avqi INPUT_DIR OUTPUT_DIR`

For legacy readability, the tables below still use `IN_DIR`/`OUT_DIR` shorthand.
Interpret `IN_DIR` as "input path set" in current CLI usage.

When you are adding a new command or changing an existing one, remember the
current architecture split:

- CLI args live in `crates/batchalign-cli`
- released-command identity and top-level orchestration live in
  `crates/batchalign-app/src/commands/`
- shared command-shape metadata lives in
  `crates/batchalign-app/src/command_family.rs`
- reusable text-batch helper types live in
  `crates/batchalign-app/src/text_batch.rs`
- job lifecycle / queueing live in `crates/batchalign-app/src/runner/`
- output materialization belongs with the owning command or orchestrator module

When output resolves to the same path as input, mutating commands overwrite the
original `.cha` file (no automatic backup).

For generation commands such as `transcribe` and `benchmark`, omitting `-o` or
passing `--in-place` still creates new output files next to the source media; it
does not rewrite the media input.

---

## Command Reference

### 1. align

**Purpose:** Add word-level and utterance-level time alignment to existing
CHAT transcripts by running forced alignment against the corresponding audio.

| Aspect | Local CLI | Server (`--server`) |
|--------|-----------|---------------------|
| **Input files** | `.cha` files in `IN_DIR` | `.cha` content sent as text over HTTP |
| **Input media** | Audio referenced by `@Media:` header, found adjacent to `.cha` or via `--lazy-audio` | Audio resolved server-side from `media_roots` / `media_mappings` via `@Media:` header name |
| **Extensions filter** | `["cha"]` | Same |
| **Output** | `.cha` with `%wor` timing line, word `time` fields populated | Same `.cha` returned as text, written to `OUT_DIR` |
| **Mutation** | If `OUT_DIR = IN_DIR`: overwrites original `.cha` in place. Media files untouched. | Same — client writes returned `.cha` over original path |
| **Key options** | `--utr-engine`, `--utr-engine-custom`, `--utr-strategy`, `--fa-engine`, `--fa-engine-custom`, `--pauses`, `--wor/--nowor`, `--override-cache` | All passed through typed command options |

**What changes in the `.cha`:** `%wor` tier added/updated with word-level
timestamps. Utterance-level bullet times (`\x15start_end\x15`) updated.
Existing `%mor`, `%gra` tiers preserved. Media file is read but never modified.

**Non-matching files:** For directory inputs, the current Rust CLI copies
non-`.cha` files and dummy CHAT files from `IN_DIR` to `OUT_DIR` before
submitting matching files, in both single-server content mode and local-daemon
paths mode.

---

### 2. transcribe

**Purpose:** Create a new CHAT transcript from audio files via ASR.

| Aspect | Local CLI / local daemon | Explicit remote `--server` |
|--------|----------------------------|-----------------------------|
| **Input files** | `.mp3`, `.mp4`, `.wav` files in `IN_DIR` | Not used in current CLI path |
| **Extensions filter** | `["mp3", "mp4", "wav"]` | Not used |
| **Output** | New `.cha` files (audio extension replaced: `foo.wav` → `foo.cha`) | Explicit `--server` is ignored; local daemon path still writes the output |
| **Mutation** | **Never mutates input.** Creates new `.cha` files in `OUT_DIR`. Original audio untouched. If `OUT_DIR = IN_DIR`, the new `.cha` appears alongside the audio. | Same effective result, because dispatch falls back to the local daemon |
| **Key options** | `--asr-engine`, `--asr-engine-custom`, `--diarization`, `--wor/--nowor`, `--lang`, `-n`, `--batch-size` | Same effective options after local fallback |

**Current routing note (Rust CLI):** explicit `--server` is ignored for
`transcribe`/`transcribe_s`; these commands run via local daemon paths-mode.
If the local daemon is already running from an older build, the CLI warns about
the build mismatch; restart the daemon before validating transcribe changes or
you may unknowingly exercise stale code.

**What gets created:** A new `.cha` file per audio file. Contains `@Comment`
line with Batchalign version and ASR engine name, `@Languages`, `@Participants`,
`@ID`, and utterance lines with timing. No `%mor`/`%gra` tiers.

**Rev.AI `--lang auto` note:** `--lang auto` is not always equivalent to
explicit `--lang eng`, even when the final transcript is treated as English.
There are two internal paths:

1. **Language ID succeeds before transcript submission** — BA3 resolves the
   request to English up front, and the Rev request path matches explicit
   `--lang eng`.
2. **Language ID fails or returns an unmapped code** — BA3 submits a true Rev
   auto request. Later stages may still resolve the resulting transcript to
   English for segmentation and CHAT headers, but provider-side request options
   differ from explicit English.

This distinction matters because provider punctuation, diarization, and turn
boundaries can differ across those two request paths.

**Note on hidden BA2 aliases:** Hidden compatibility flags such as `--diarize`,
`--whisper`, and `--rev` still parse, but they are migration shims. Public docs
should prefer `--diarization` and `--asr-engine`.

### 3. transcribe_s (transcribe --diarize)

Identical to `transcribe` above, except the pipeline includes speaker
diarization (Pyannote). Output `.cha` files have multiple `@Participants`
and speaker-attributed utterances. Not a separate CLI command — triggered
by `batchalign3 transcribe --diarize`.

---

### 4. morphotag

**Purpose:** Add morphosyntactic analysis (`%mor` and `%gra` tiers) to
existing CHAT transcripts.

| Aspect | Local CLI | Server (`--server`) |
|--------|-----------|---------------------|
| **Input files** | `.cha` files in `IN_DIR` | `.cha` content sent as text |
| **Extensions filter** | `["cha"]` | Same |
| **Output** | `.cha` with `%mor` and `%gra` tiers added/replaced | Same `.cha` returned as text |
| **Mutation** | If `OUT_DIR = IN_DIR`: **overwrites original `.cha` in place**. | Same |
| **Key options** | `--retokenize`, `--skipmultilang`, `--lexicon <CSV>`, `--override-cache`, `--merge-abbrev` | All passed. Lexicon CSV is read on the client and injected into typed command options before submission. |

**What changes in the `.cha`:** `%mor` tier added/replaced with POS tags and
lemmas. `%gra` tier added/replaced with dependency relations. Main tier text
may be retokenized if `--retokenize` is set. Special `%mor` notation
(`@Options: dummy`) is auto-detected and preserved.

**No media involved.** This is a text-only operation.

---

### 5. utseg

**Purpose:** Segment a transcript into utterances using Stanza.

| Aspect | Local CLI | Server (`--server`) |
|--------|-----------|---------------------|
| **Input files** | `.cha` files in `IN_DIR` | `.cha` content sent as text |
| **Extensions filter** | `["cha"]` | Same |
| **Output** | `.cha` with utterance boundaries recomputed | Same |
| **Mutation** | If `OUT_DIR = IN_DIR`: **overwrites original `.cha` in place**. | Same |
| **Key options** | `--lang`, `-n`, `--merge-abbrev` | All passed |

**What changes in the `.cha`:** Utterance boundaries (`*SPK:` lines) are
recomputed. Existing `%mor`/`%gra` tiers may be invalidated (would need
re-running morphotag afterwards).

**No media involved.**

---

### 6. translate

**Purpose:** Add English translations to non-English transcripts.

| Aspect | Local CLI | Server (`--server`) |
|--------|-----------|---------------------|
| **Input files** | `.cha` files in `IN_DIR` | `.cha` content sent as text |
| **Extensions filter** | `["cha"]` | Same |
| **Output** | `.cha` with translation tiers | Same |
| **Mutation** | If `OUT_DIR = IN_DIR`: **overwrites original `.cha` in place**. | Same |
| **Key options** | `--merge-abbrev` | Passed |

**What changes in the `.cha`:** Translation tier added to each utterance.

**No media involved.**

---

### 7. coref

**Purpose:** Add coreference annotations to transcripts.

| Aspect | Local CLI | Server (`--server`) |
|--------|-----------|---------------------|
| **Input files** | `.cha` files in `IN_DIR` | `.cha` content sent as text |
| **Extensions filter** | `["cha"]` | Same |
| **Output** | `.cha` with coreference annotations | Same |
| **Mutation** | If `OUT_DIR = IN_DIR`: **overwrites original `.cha` in place**. | Same |
| **Key options** | `--merge-abbrev` | Passed |

**No media involved.**

---

### 8. compare

**Purpose:** Compare CHAT transcripts against gold-standard references to compute
word error rate (WER) and inject per-utterance comparison annotations.

| Aspect | Local CLI | Server (`--server`) |
|--------|-----------|---------------------|
| **Input files** | `.cha` files in `IN_DIR` | `.cha` content sent as text |
| **Gold files** | `FILE.gold.cha` in same directory as `FILE.cha` | Gold files sent alongside main files, or read from server filesystem in paths mode |
| **Extensions filter** | `["cha"]` | Same |
| **Output** | `.cha` with `%xsrep` / `%xsmor` tiers + `.compare.csv` metrics | Same — client writes both files to `OUT_DIR` |
| **Mutation** | If `OUT_DIR = IN_DIR`: **overwrites original `.cha` in place**. Gold files are never modified. | Same |
| **Key options** | `--lang`, `--merge-abbrev`, `--override-cache` | All passed through typed command options |

**What changes in the `.cha`:** The released output is the projected
gold/reference transcript written at the main file's output path. BA3
morphotags the main transcript, keeps the gold transcript raw during artifact
construction, projects structurally safe `%mor` / `%gra` / `%wor` information
onto the gold AST, and injects `%xsrep` / `%xsmor` on that projected reference
output. `%xsrep` uses `word`, `+word`, and `-word`; `%xsmor` mirrors the same
alignment with POS tags such as `NOUN`, `+ADJ`, and `-?`. Those tiers are now
materialized from typed compare-tier models and lowered once at the final CHAT
serialization boundary.

**Additional output:** A companion `.compare.csv` file is written alongside each
`.cha` output with aggregate metrics (WER, accuracy, match/insertion/deletion
counts, total word counts) plus per-POS rows. The CSV is emitted from a typed
metrics table model via the Rust `csv` crate, not by assembling row strings by
hand.

**Gold file convention:** For each `FILE.cha`, the gold companion is
`FILE.gold.cha` in the same directory. Files ending in `.gold.cha` are
automatically skipped as inputs (they are companions). If no gold file is
found, the file is marked as failed with an error message.

**Pipeline:** pair main + gold → morphosyntax on main only → parse raw gold →
BA2-style per-gold-utterance local-window alignment → `ComparisonBundle`
(main view, gold view, structural word matches, metrics) → materialization. The
command-owned compare layer now models compare as a reference-projection
command rather than "just another per-file mutator." The semantic unit is the
comparison bundle, not a flat text rewrite.

**Output shapes:** compare can materialize more than one view of the same
comparison bundle. The released command now emits the projected reference view.
Benchmark-style flows can still materialize a main-annotated view internally.
The projection path works over the CHAT AST: exact structural matches can copy
`%mor` / `%gra` / `%wor`, while partial matches stay conservative instead of
reconstructing tiers from strings. Compare parity is semantic — the workflow
matches BA2 behavior without copying BA2's string/document shell.

**No media involved.** This is a text-only operation.

---

### 9. benchmark

**Purpose:** Run ASR and evaluate word accuracy against ground truth.

| Aspect | Local CLI | Server (`--server`) |
|--------|-----------|---------------------|
| **Input files** | `.mp3`, `.mp4`, `.wav` files in `IN_DIR` | Media filenames, or server-side bank listings via `--bank` / `--subdir` |
| **Extensions filter** | `["mp3", "mp4", "wav"]` | Same |
| **Output** | New `.cha` files with ASR output + eval metrics | Same |
| **Mutation** | **Never mutates input.** Creates new `.cha` files. | Same |
| **Key options** | `--asr-engine`, `--asr-engine-custom`, `--lang`, `-n`, `--wor/--nowor`, `--bank`, `--subdir` | All passed |

**Same I/O pattern as transcribe** — creates new `.cha` files with audio
extension renamed. Additionally includes evaluation metrics from comparing
ASR output against reference transcripts.

`benchmark` is a composite command: it runs transcribe first and then calls a
main-annotated compare path internally. It deliberately shares compare-side
internals, but it does **not** share compare's released projected-reference
contract. If you are changing benchmark behavior, look at the command-owned
Rust layer first rather than adding logic in CLI dispatch.

---

### 10. opensmile

**Purpose:** Extract acoustic features from audio files.

| Aspect | Local CLI | Server (`--server`) |
|--------|-----------|---------------------|
| **Input files** | `.mp3`, `.mp4`, `.wav` files in `INPUT_DIR` | Media filenames; server resolves audio |
| **Extensions filter** | `["mp3", "mp4", "wav"]` | Same |
| **Output** | `.opensmile.csv` files (NOT `.cha`) | Same — server returns CSV as text with `content_type: "csv"` |
| **Mutation** | **Never mutates input.** Creates new `.opensmile.csv` files in `OUT_DIR`. | Same |
| **Key options** | `--feature-set` (eGeMAPSv02, etc.), `--lang` | All passed |

**Special output:** This is the only command that produces non-CHAT output.
The client handles this by checking `content_type` in the server response
and using the server-provided filename directly.

---

### 11. avqi

**Purpose:** Calculate Acoustic Voice Quality Index from paired `.cs`/`.sv`
audio files.

| Aspect | Local CLI | Server (`--server`) |
|--------|-----------|---------------------|
| **Input files** | Paired `.cs.*` and `.sv.*` audio files in input paths | `--server` is ignored; command runs on local daemon |
| **Output** | `.avqi.txt` with metrics per file pair | Same (written locally by daemon paths-mode) |
| **Mutation** | **Never mutates input.** Creates new `.avqi.txt` files. | Same |

**Current routing note:** `dispatch.rs` forces `avqi` to local daemon mode even
when `--server` is provided, because AVQI depends on local paired-file discovery.

**Current syntax note:** `opensmile` and `avqi` do not use the shared `PATHS` /
`-o` command form. Their CLI syntax is positional:

```bash
batchalign3 opensmile INPUT_DIR OUTPUT_DIR
batchalign3 avqi INPUT_DIR OUTPUT_DIR
```

---

## Summary: Input Sources and Mutation Patterns

### Commands that mutate `.cha` files in place (when `OUT_DIR = IN_DIR`)

| Command | Input | What changes |
|---------|-------|--------------|
| **align** | Existing `.cha` + audio | Adds `%wor` tier, updates bullet times |
| **morphotag** | Existing `.cha` | Adds/replaces `%mor` + `%gra` tiers |
| **utseg** | Existing `.cha` | Recomputes utterance boundaries |
| **translate** | Existing `.cha` | Adds translation tier |
| **coref** | Existing `.cha` | Adds coreference annotations |
| **compare** | Existing `.cha` + gold `.cha` | Writes projected reference `.cha` with `%xsrep` / `%xsmor`, plus `.compare.csv` |

These commands read `.cha`, process the `Document`, and write the result
back. When `OUT_DIR = IN_DIR`, the original file is **overwritten**. The
audio files referenced by `align` are read but never modified.

### Commands that create new files (never mutate input)

| Command | Input | Output created |
|---------|-------|----------------|
| **transcribe** | Audio files (`.mp3`/`.mp4`/`.wav`) | New `.cha` files |
| **benchmark** | Audio files | New `.cha` files with eval metrics |
| **opensmile** | Audio files | New `.opensmile.csv` files |
| **avqi** | Paired `.cs`/`.sv` audio | New `.avqi.txt` files |

These commands never touch the input files. The output always has a
different extension or name than the input.

---

## Server Dispatch: What Crosses the Network

| Direction | CHAT commands (align, morphotag, ...) | Media commands (transcribe, opensmile, ...) |
|-----------|---------------------------------------|---------------------------------------------|
| **Client → Server** | Full `.cha` text (~2KB each) | Only filenames or server-bank selectors (not file content) |
| **Server → Client** | Processed `.cha` text | Processed `.cha` or `.csv` text |
| **Media** | Server resolves from `media_roots` via `@Media:` header | Server resolves from `media_roots` / `media_mappings` |

Audio/video files **never cross the network**. The server must have access
to the media via its configured `media_roots` or `media_mappings` (typically
NFS/SMB mounts).

---

## Local Daemon Dispatch: Paths Mode

When the CLI uses a local daemon (no explicit `--server`), it uses **paths
mode** — only filesystem paths cross HTTP, not file content.

| Direction | What crosses HTTP |
|-----------|-------------------|
| **Client → Server** | `source_paths` (absolute paths to input files) and `output_paths` (absolute paths for output) |
| **Server → Client** | Nothing — daemon writes directly to `output_paths` |

The daemon reads input files and writes output files directly via the
filesystem. No staging directory is created. No content transfer. No result
download. The CLI polls `GET /jobs/{id}` for status only.

**Advantages over content mode:**
- No 100MB body limit or 1000-file cap
- No staging I/O (4 data copies eliminated)
- Transcribe works (daemon shares the filesystem)
- All commands behave identically to local CLI from the user's perspective

**Media resolution:** Same as local CLI — the daemon resolves `@Media:`
headers against its filesystem (same machine, same paths).

---

## Non-Matching File Handling

**Current Rust CLI** (`discover.rs`, `dispatch/single.rs`, `dispatch/paths.rs`):
- Files that don't match the command's extensions are copied from `IN_DIR` to
  `OUT_DIR` for directory inputs.
- Dummy CHAT files (`@Options: dummy`) are copied unchanged and are not
  submitted for processing.
- Matching files are sorted by size descending before submission to reduce
  straggler effects on long runs.

This means current single-server content mode and local-daemon paths mode are
closer than the older Python split: both preserve non-matching files and both
filter dummy CHAT locally.

---

## Parity Status

| Command | I/O parity | Options parity | Daemon paths mode | Notes |
|---------|------------|----------------|-------------------|-------|
| align | Full | Full | Full | Media resolution differs (local path vs server lookup) but equivalent |
| transcribe | Full | Full | Full | Explicit `--server` is currently ignored; routed to local daemon |
| transcribe_s | Full | Full | Full | Triggered by `--diarize`; explicit `--server` ignored |
| morphotag | Full | Full | Full | Lexicon CSV read on client, sent as parsed dict |
| utseg | Full | Full | Full | |
| translate | Full | Full | Full | |
| coref | Full | Full | Full | |
| benchmark | Full | Full | Full | |
| opensmile | Full | Full | Full | Special CSV output handling on both sides |
| compare | Full | Full | Full | Gold file resolved locally or server-side |
| avqi | Full (local) | Full (local) | Full | Routed to local daemon even when `--server` is set |
