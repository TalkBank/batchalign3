# Retrace Detection

**Status:** Current behavior reference  
**Last verified:** 2026-03-05

This page documents how current batchalign retrace detection works. It does not
preserve branch-era side-by-side implementation archaeology.

## CHAT convention

CHAT marks repeated word sequences with `[/]` for partial retracing:

- single-word retrace: `the [/] the dog .`
- multi-word retrace: `<I want> [/] I want a cookie .`

Angle brackets are only used when multiple words must be grouped.

## Current detection rule

Current retrace detection uses sliding-window repeated-sequence matching over
lexical words, with language-specific safeguards such as higher minimum span for
Chinese.

## Current implementation properties

The current implementation is structured to avoid two common older failure
modes:

- larger repeated spans are preferred over smaller fragmentary matches
- overlap-safe claiming prevents the same region from being marked repeatedly by
  conflicting matches

## Current formatting rule

Retrace formatting is structure-driven:

- single-word retraces serialize without angle brackets
- multi-word retraces serialize as grouped bracketed spans

The important current property is that bracket choice follows the structured
representation rather than ad-hoc string postprocessing.

## Known limits

- current retrace detection targets exact repetition, not richer reformulation
  analysis such as `[//]`
- non-lexical or already-heavily-annotated content may reduce what can be
  recognized as a retrace candidate
- overlap-heavy or highly noisy utterances may still require manual review

## Legacy note

Earlier versions of this page compared older Python and newer Rust
implementations in detail. For public docs, the important point is the current
behavioral contract: retraces are detected structurally and formatted from
structure, not by fragile detokenize-time heuristics.
