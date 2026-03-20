# CLI Options Audit: Completed Assessment

**Status:** Current
**Last updated:** 2026-03-18

## Assessment

Reviewed all CLI option structs against CLAUDE.md boolean blindness rules.

### Already Fixed
- `CaMarkerPolicy` enum (was `bool`)
- `UtrMatchMode` enum (was hardcoded)
- `TwoPassConfig` scoped struct (was flat fields)
- `AlignOptions` has `Default` impl

### Acceptable Booleans (per CLAUDE.md rules)
These are single on/off flags where the name is self-documenting:
- `pauses: bool` — "add pauses" is clear
- `force: bool`, `quiet: bool` — standard CLI flags
- `skip_alignment: bool` — single skip flag
- `roundtrip: bool` — single enable flag

### Opposing Bool Pairs (technically banned, kept for BA2 compat)
These use `--flag`/`--no-flag` patterns which CLAUDE.md bans:
- `--utr` / `--no-utr` → resolved to `Option<UtrEngine>` internally
- `--wor` / `--nowor` → resolved to `WorTierPolicy` enum internally
- `--merge-abbrev` / `--no-merge-abbrev` → resolved to `MergeAbbrevPolicy` internally
- `--diarize` / `--nodiarize` → resolved to mode internally

**Decision:** Keep for BA2 backward compatibility. The internal types are
already correct enums. The CLI surface is the only place bools appear,
and changing flag names would break existing scripts.

### No Further Action Needed
The audit found no actionable issues beyond what was already fixed.
The remaining bool pairs are BA2 compat constraints that resolve to
proper types internally.
