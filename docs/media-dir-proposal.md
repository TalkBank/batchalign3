# --media-dir CLI flag for align command

**Status:** Implemented
**Last updated:** 2026-03-18

## Problem

The `align` command requires audio files to be collocated with .cha files.
This forces users to copy audio to the same directory as their CHAT files,
which is wasteful (audio files are large) and doesn't match how TalkBank
data is organized (CHAT in one tree, media in another on the server).

## Current Behavior

```bash
# Audio MUST be next to the .cha file
batchalign3 align /path/to/file.cha -o output/
# Looks for /path/to/file.mp3 (or .mp4, .wav)
```

## Proposed Behavior

```bash
# Specify where audio lives separately
batchalign3 align /path/to/file.cha -o output/ --media-dir /media/path/
# Looks for /media/path/file.mp3
```

When `--media-dir` is set, the aligner resolves audio by:
1. Take the .cha filename stem (e.g., `file`)
2. Look for `{media-dir}/{stem}.{mp3,mp4,wav}`

When not set, current behavior (colocation) is preserved.

## Implementation

The server already has `media_root` in its config for media resolution.
The CLI just needs to pass `--media-dir` through to the server's media
resolution path. For local daemon mode, the daemon reads files directly
from the filesystem, so `media_root` controls where it looks.

**Files to modify:**
- `crates/batchalign-cli/src/args/commands.rs` — add `--media-dir` to `AlignArgs`
- `crates/batchalign-app/src/media.rs` — use custom media root when provided
- `crates/batchalign-app/src/types/options.rs` — add `media_dir` to `AlignOptions`

Estimated effort: ~30 minutes.

## Workaround

Currently, symlinks work:
```bash
ln -s /media/path/file.mp3 /chat/path/file.mp3
```

But this is manual and error-prone for large batches.
