# L2 and Language Switching

**Status:** Current behavior reference  
**Last verified:** 2026-03-05

## Current behavior

Batchalign distinguishes between:

- utterance-level language directives
- word-level language markers such as `@s` and `@s:lang`

For current `%mor` behavior, word-level language-marked forms are handled
conservatively.

## Current `%mor` rule

For `@s` or `@s:lang` words:

- the word is recognized as foreign/code-switched
- current `%mor` output uses `L2|xxx`
- current batchalign3 does not preserve a full lexical/morphological analysis
  for that word inside `%mor`

This is a deliberate conservative choice. It is safer than reusing morphology
from the wrong language model and presenting it as valid analysis.

## Utterance-level versus word-level behavior

### Utterance-level

Utterance-level language directives affect utterance handling and routing
boundaries.

### Word-level

Word-level language markers identify foreign/code-switched words, but do not
currently trigger full per-word language-specific morphosyntax routing.

## Current limit

The parsed word-level language information is not currently used to route each
marked word through a separate language-specific NLP pipeline.

So the current public boundary is:

- preserve that the word is foreign/code-switched
- avoid claiming full morphology for it

## Related references

- [Per-Word Language Routing](per-word-language-routing.md)
- [Per-Utterance Language Routing](per-utterance-language-routing.md)
- [Language Data Model](language-handling.md)
