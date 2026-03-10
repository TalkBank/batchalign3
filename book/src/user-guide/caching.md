# Caching

**Status:** Current
**Last updated:** 2026-03-15

## Why is my second run so fast?

Batchalign automatically caches NLP results so that processing the same
corpus again skips expensive model inference. If you run `morphotag` on a
200-file corpus, edit five files, and re-run, only those five files trigger
fresh inference. Everything else comes from the cache instantly.

This happens automatically — no flags needed.

## What gets cached?

| Analysis | Cached? |
|----------|---------|
| Morphosyntax (`morphotag`) | Yes |
| Utterance segmentation (`utseg`) | Yes |
| Translation (`translate`) | Yes |
| Forced alignment word timings (`align`) | Yes |
| ASR results for timing recovery (`align --utr`) | Yes |
| Speaker diarization | No |
| Coreference (`coref`) | No |
| OpenSMILE features (`opensmile`) | No |
| AVQI scores (`avqi`) | No |

Cached tasks store results per utterance, keyed by content. Uncached tasks
either depend on full-document context (diarization, coreference) or are
fast enough to recompute every time (OpenSMILE, AVQI).

## When does the cache invalidate?

| What changed | What re-runs | What stays cached |
|-------------|-------------|-------------------|
| Edited the transcript text | Morphosyntax, utseg, translation, FA | UTR ASR results |
| Re-recorded or replaced the audio file | FA, UTR ASR | Morphosyntax, utseg, translation |
| Changed the language code | Everything re-runs | Nothing |
| Upgraded batchalign (new model versions) | Stale entries auto-invalidated | Entries from unchanged engines |

Cache keys are content-addressed: they hash the actual input (words, audio
fingerprint, language). Changing any input component produces a different
key, so stale results are never returned. Engine version strings are stored
alongside each entry, so upgrading a model (e.g., a new Stanza release)
automatically invalidates old results without manual intervention.

## How to force fresh results

Use the `--override-cache` global flag:

```bash
batchalign3 --override-cache morphotag corpus/ -o output/
```

This skips all cache lookups, forcing every utterance through fresh
inference. New results are still stored in the cache for future runs.

Use this when you suspect cached results are wrong, or after manually
updating model files outside of a normal batchalign upgrade.

## Where are the caches stored?

| Cache | macOS default | Linux default |
|-------|---------------|---------------|
| Analysis cache DB | `~/Library/Caches/batchalign3/cache.db` | `~/.cache/batchalign3/cache.db` |
| Media conversion cache | `~/Library/Application Support/batchalign3/media_cache/` | `~/.local/share/batchalign3/media_cache/` |

The analysis cache is a single SQLite database file. The media cache stores
converted WAV artifacts for inputs such as `.mp4` and `.m4a`.

For isolated runs or testing, you can relocate them with environment
variables:

```bash
export BATCHALIGN_ANALYSIS_CACHE_DIR=/tmp/ba-analysis-cache
export BATCHALIGN_MEDIA_CACHE_DIR=/tmp/ba-media-cache
```

## How to clear the cache

Use the built-in cache command:

```bash
batchalign3 cache stats          # See cache size and entry count
batchalign3 cache clear --yes    # Clear the cache
```

`cache stats` and `cache clear` operate on both the analysis cache and the
media conversion cache.

Or delete the `cache.db` file and/or the media-cache directory directly.

To selectively refresh without clearing everything, use `--override-cache`
on specific runs instead — old entries for other corpora remain available.
