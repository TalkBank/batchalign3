# Testing Learnings: Integration Test Development

**Status:** Current
**Last updated:** 2026-03-18

Lessons learned the hard way while building the BA2-parity integration test suite.
These should be distilled into CLAUDE.md, the book, or test infrastructure docs.

## trim_chat_audio.py Does Not Rebase Timing Bullets

**Problem:** `trim_chat_audio.py` trims the audio to a time range but leaves CHAT timing bullets as absolute positions from the original recording. When the trimmed audio starts at 0ms but the bullets say 15010ms, the aligner fails with `invalid audio window start=X end=Y` (start > end, or beyond audio duration).

**Impact:** Non-English align tests (Spanish, French, Cantonese) fail because the corpus-extracted CHAT fixtures have un-rebased timing.

**Fix needed:** Either `trim_chat_audio.py` should subtract the trim offset from all timing bullets, OR test fixtures should be manually rebased. This is a gap in the trim tooling.

**Workaround for tests:** For align tests with corpus-extracted fixtures, strip ALL timing bullets from the CHAT input (the aligner will produce new timing from scratch). This works because align's job is to ADD timing, not preserve existing timing.

## Rev.AI Tests Require Config, Not Just Env Vars

The test helper `require_revai_key()` only checks `REVAI_API_KEY` and `BATCHALIGN_REV_API_KEY` environment variables. But the actual Rev.AI key lives in `~/.batchalign.ini`. Tests should read from the config file (via the server's config resolution) or the env var should be set in the test environment.

## D1/D1b Status (Disfluency/Retrace)

The disfluency replacement and n-gram retrace detection are ALREADY IMPLEMENTED in BA3 (commit `ba3d7c2f`). The D1/D1b parity tests both PASS — this was fixed before these tests were written. The parity audit memory is stale.

## Media Resolution for Align Tests

**Problem:** Align tests submitted stripped CHAT files via paths-mode, but the server couldn't find the audio.

**Root cause:** The batchalign3 server resolves media by looking for the `@Media` basename (from the CHAT header) as a file in the same directory as the input CHAT file. If the CHAT file is named `test_stripped.cha` but `@Media` says `test, audio`, the server looks for `test_stripped.mp3` — not `test.mp3`.

**Rule:** For paths-mode align/transcribe jobs:
1. The CHAT file's `@Media` basename must match the audio filename (without extension)
2. The audio file must be in the same directory as the CHAT file
3. The CHAT filename itself does NOT need to match `@Media` — but it's simpler if it does

**Fix in tests:** Create a subdirectory per test case with the audio and CHAT file both named after the `@Media` basename. The stripped CHAT overwrites the original in this subdirectory.

## @Media Header Must Match Audio Filename

The `@Media` header in CHAT is the canonical media reference. When trimming corpus files with `prepare_corpus_media_fixture.py`, the tool appends `-trimmed` to the media filename. The CHAT's `@Media` header is updated to match. But if you rename the audio file (e.g., from `040707-trimmed.mp3` to `spa_marrero_clip.mp3`), you MUST also update the `@Media` header to `spa_marrero_clip, audio`.

## CHAT Validation of Test Fixtures

We are NOT currently running CHAT validation on our test fixtures before using them. The `validate` command is in `talkbank-cli`, not `batchalign-cli`. We should validate all fixtures as a pre-check.

## BA2 CLI Quirks

- `batchalignjan9 morphotag` does NOT accept `--lang` — it reads from `@Languages` in the CHAT file
- `batchalignjan9 translate` does NOT accept `--lang` — same
- `batchalignjan9 utseg` DOES accept `--lang`
- `batchalignjan9 coref` exists but is broken (Stanza version incompatibility: `Config.__init__() missing 1 required positional argument: 'plateau_epochs'`)

## %gra ROOT Convention

BA2 uses self-referencing ROOT: `4|7|ROOT` (word 4's head is word 7, which IS the root — so root points to itself). BA3 uses UD-standard ROOT: `4|0|ROOT` (root's head is 0, the conventional UD sentinel). This is an intentional, systematic difference in every `%gra` line. The parity comparison normalizes this.

## Stanza Model Availability

Not all Stanza models are downloaded by default. Tests must handle missing models gracefully:
- French constituency model (`utseg`) may not be downloaded
- Cantonese constituency model (`utseg`) may not be downloaded
- Non-English Stanza models download on first use (can be slow)

The test harness treats `JobStatus::Failed` for non-English as a skip, not a failure.

## mm-hmm Tokenization

BA2's Stanza tokenizes "mm-hmm" → `mmhmm` (merged, 1 token). BA3's Stanza tokenizes it → `mm–hmm` (with en-dash, different token count). This causes %mor and %gra differences. Documented as cosmetic, not a bug.

## LiveServerSession Resource Management

- Semaphore(1) — only one session at a time
- Models load once, shared across all sessions via `PreparedWorkers`
- `max_workers_per_key: 2` in tests (low memory)
- Workers persist with 1-hour idle timeout (no re-loading between tests)
- Don't run multiple test binaries concurrently — each spawns its own worker pool

## Translation Differences

BA2 and BA3 produce slightly different `%xtra` (translation) tiers:
- BA2 preserves trailing punctuation in translations: `%xtra:hello world .`
- BA3 may omit trailing punctuation: `%xtra:hello world`
- BA2 translates disfluent text verbatim; BA3 may clean it up
- These are translation model/post-processing differences, not bugs

## Fixture Provenance

Every test fixture must have documented provenance in `PROVENANCE.md`:
- Source corpus and file path
- Extraction method (manual, ffmpeg, `prepare_corpus_media_fixture.py`)
- Language and key features
- Why this fixture was chosen

## Trim Utilities

The workspace has purpose-built tools for creating test fixtures:
- `scripts/analysis/prepare_corpus_media_fixture.py` — copies CHAT + downloads media from net, then trims
- `scripts/analysis/trim_chat_audio.py` — trims CHAT + audio to a range of main-tier utterance lines
- Both use timing bullets from the transcript to determine audio trim range
- Both handle `@Media` header updates automatically
