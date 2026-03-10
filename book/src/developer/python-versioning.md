# Python Version Support

**Status:** Current  
**Last verified:** 2026-03-05

## Current policy

The current deployment and contributor baseline is Python 3.12.

Active targeting of Python 3.14t / free-threaded Python is paused until the
full engine set required by the public `batchalign3` command surface is
supported.

## Runtime ownership

| Layer | Runtime | Notes |
|---|---|---|
| Rust CLI / server control plane | Native Rust binary | No ML inference in Rust |
| Main worker runtime | Python 3.12 | Current target for active deployment |
| Sidecar/local compatibility runtime | Python 3.12 | Used where command/runtime split requires it |

## Why 3.12 remains the target

The blocker is engine coverage, not the core Rust/PyO3 boundary by itself.

Publicly important examples:

- `transcribe` / `transcribe_s` still depend on stacks that are not ready for
  3.14t deployment
- diarization-related paths also remain incomplete on 3.14t

Promising 3.14t experiments do not change the current release policy.

## Current guidance

- treat Python 3.12 as the supported deployment baseline
- do not plan production around 3.14t yet
- if 3.14t experiments resume later, document that as a fresh current decision
  after end-to-end engine coverage exists

## Related pages

- [Building & Development](building.md)
