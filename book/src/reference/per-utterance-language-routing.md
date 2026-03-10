# Per-Utterance Language Routing

**Status:** Current behavior reference  
**Last verified:** 2026-03-05

## Current behavior

Batchalign supports per-utterance language handling through utterance-level
language directives and file-level language metadata.

Current practical behavior:

- utterance language information is represented in the parsed CHAT structure
- current morphosyntax/runtime paths can use utterance-level language
  information when deciding how to process or skip utterances
- this is the supported current language-routing boundary in the public product

## Why this matters

Per-utterance routing is the current released mechanism for multilingual
handling that is more precise than a single file-wide language and simpler than
full per-word routing.

This is the important distinction for users:

- file-wide language alone is often too coarse for multilingual corpora
- per-word routing is not the current public runtime boundary
- per-utterance routing is the supported middle ground

## Interaction with `skipmultilang`

Current logic may use utterance-level language directives to decide whether an
utterance should be processed normally or skipped under multilingual-safety
rules, depending on command options and workflow.

## Relationship to current migration story

If you are comparing released BA2 to current BA3, the migration-relevant point
is that current BA3 is more explicit about language-aware routing boundaries and
about what is handled at utterance level versus what is not handled at word
level.

The full Jan 9 BA2 -> Feb 9 BA2 -> current BA3 history belongs in the migration
book, not here.

## Current limit

Per-utterance routing should not be read as full per-word bilingual analysis.
That richer behavior is outside the current released public contract.

See:

- [Per-Word Language Routing](per-word-language-routing.md)
- [Multilingual Support](multilingual.md)
- [Language Data Model](language-handling.md)
