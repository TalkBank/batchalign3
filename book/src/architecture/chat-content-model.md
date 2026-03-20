# CHAT Content Model

**Status:** Current
**Last updated:** 2026-03-18

This chapter explains how batchalign3 represents CHAT main-tier content as a typed AST, how the type hierarchy nests, and what each level contains. Understanding this hierarchy is essential for writing correct content traversals.

## The Type Hierarchy

CHAT main-tier content is a tree with three nesting levels. Each level has its own enum of content types:

```
ChatFile
└── Line::Utterance
    └── MainTier
        └── TierContent
            ├── content: Vec<UtteranceContent>     ← Level 1
            │   ├── Word(Box<Word>)
            │   │   └── content: Vec<WordContent>  ← Level 3
            │   ├── OverlapPoint(OverlapPoint)
            │   ├── Group(Group)
            │   │   └── BracketedContent
            │   │       └── Vec<BracketedItem>     ← Level 2
            │   ├── PhoGroup, SinGroup, Quotation
            │   │   └── (same BracketedContent)
            │   ├── Pause, Event, Separator, ...
            │   └── AnnotatedWord, AnnotatedGroup, ...
            ├── bullet: Option<Bullet>
            ├── linkers: Linkers
            └── terminator: Terminator
```

### Level 1: UtteranceContent (24 variants)

The top-level content items on a main tier. This is what you iterate when you walk `utterance.main.content.content.0`:

| Category | Variants | Notes |
|----------|----------|-------|
| **Words** | `Word`, `AnnotatedWord`, `ReplacedWord` | Lexical tokens; `Annotated<Word>` adds `[* m]`-style scoped annotations |
| **Groups** | `Group`, `AnnotatedGroup`, `PhoGroup`, `SinGroup`, `Quotation` | Bracketed structures containing `BracketedItem`s |
| **CA markers** | `OverlapPoint`, `Separator` | CA overlap markers (⌈⌉⌊⌋), comma/tag separators |
| **Events** | `Event`, `AnnotatedEvent`, `OtherSpokenEvent` | `&=laughs`, `&*SPK:word` |
| **Actions** | `AnnotatedAction` | `0 [= gestures]` |
| **Timing** | `InternalBullet` | Mid-utterance timestamp |
| **Scope markers** | `LongFeatureBegin/End`, `NonvocalBegin/End/Simple`, `UnderlineBegin/End` | Paired scope delimiters |
| **Other** | `Freecode`, `Pause` | `[^ note]`, `(.)` |

**Critical rule:** Every `match` on `UtteranceContent` must explicitly list all 24 variants — no `_ =>` catch-alls. This is enforced by project policy to prevent silent data loss when new variants are added.

### Level 2: BracketedItem (22 variants)

Content inside groups (`<...>`, `‹...›`, `〔...〕`, `"..."`). Accessed via `group.content.content.0`:

```rust
// Group.content is BracketedContent
// BracketedContent.content is BracketedItems (newtype over Vec<BracketedItem>)
// BracketedItems.0 is Vec<BracketedItem>
let items: &[BracketedItem] = &group.content.content.0;
```

`BracketedItem` mirrors `UtteranceContent` closely — same word, event, pause, and marker types — plus `AnnotatedGroup` for nested `<...> [/]` retrace groups. Groups can nest arbitrarily deep.

### Level 3: WordContent (11 variants)

Content inside a single word token. Accessed via `word.content`:

```rust
for wc in &word.content {
    match wc {
        WordContent::Text(text) => { /* plain text segment */ }
        WordContent::Shortening(text) => { /* (lo) omitted sound */ }
        WordContent::OverlapPoint(marker) => { /* ⌈ inside a word */ }
        WordContent::CAElement(ca) => { /* ↑ ↓ prosody markers */ }
        WordContent::CADelimiter(ca) => { /* ° ∆ paired delimiters */ }
        WordContent::StressMarker(_) => { /* ˈ ˌ */ }
        WordContent::Lengthening(_) => { /* : */ }
        WordContent::SyllablePause(_) => { /* ^ */ }
        WordContent::CompoundMarker(_) => { /* + in ice+cream */ }
        WordContent::UnderlineBegin(_) | WordContent::UnderlineEnd(_) => {}
    }
}
```

**Key insight:** Overlap markers can appear at any level — as standalone `UtteranceContent::OverlapPoint` (space-separated: `⌈ word ⌉`), as `BracketedItem::OverlapPoint` (inside groups), or as `WordContent::OverlapPoint` (intra-word: `butt⌈er⌉`). Any traversal looking for overlap markers must check all three levels.

## Annotated Types

The `Annotated<T>` wrapper adds scoped annotations (`[/]`, `[* m]`, `[= explanation]`, etc.) to any annotatable type:

```rust
pub struct Annotated<T> {
    pub inner: T,                         // the wrapped item
    pub scoped_annotations: Vec<ScopedAnnotation>,
    pub span: Span,
}
```

At Level 1: `AnnotatedWord(Box<Annotated<Word>>)`, `AnnotatedGroup(Annotated<Group>)`, `AnnotatedEvent(Annotated<Event>)`, `AnnotatedAction(Annotated<Action>)`.

At Level 2: same variants exist in `BracketedItem`.

When traversing, you typically want the inner item: `annotated.inner` gives you the `Word`, `Group`, etc.

## Replaced Words

`ReplacedWord` represents `word [: replacement]` — a word with a replacement form:

```rust
pub struct ReplacedWord {
    pub word: Word,                    // the surface form
    pub replacement: Replacement,      // the [: ...] replacement
}

pub struct Replacement {
    pub words: Vec<Word>,              // replacement words (may be empty)
}
```

When extracting words for NLP, the convention depends on the tier domain:
- **Wor domain:** use replacement words if non-empty, else the surface form
- **Mor domain:** same, but check `counts_for_tier` for each

## Tier Domains

Different NLP tasks need different views of the same content. The `TierDomain` enum controls which words count for each tier and how groups are traversed:

| Domain | Used by | Skips | Counts separators? |
|--------|---------|-------|--------------------|
| `Mor` | %mor/%gra generation | Retrace groups | Yes (`,` `„` `‡` have mor items) |
| `Wor` | %wor generation, FA | Nothing | No |
| `Pho` | %pho alignment | PhoGroups | No |
| `Sin` | %sin alignment | SinGroups | No |

The content walker (`walk_words`) takes `Option<TierDomain>`:
- `Some(domain)` — domain-aware gating (skip domain-specific groups)
- `None` — recurse everything unconditionally

## The Validation Traversal Pattern

The validation module in `talkbank-model` provides a reference implementation of complete content traversal. See `validation/utterance/overlap.rs` for an example that:
1. Walks `UtteranceContent` (Level 1)
2. Recurses into `BracketedContent` (Level 2)
3. Scans `WordContent` (Level 3) for intra-word overlap markers
4. Handles all annotated variants (`AnnotatedWord`, `AnnotatedGroup`)

Any new traversal should follow this same pattern to ensure no content is missed.

## Common Pitfalls

1. **"Consecutive" means in-order traversal, not adjacent array indices.**
   When CHAT tools speak of "consecutive" or "sequential" items on the main
   tier, this ALWAYS means in **document order via recursive traversal** —
   accounting for groups (`<...>`), retrace groups (`<...> [/]`), quotations
   (`"..."`), and all other bracketed structures. Two items may be
   "consecutive" in the linguistic stream even if they are separated by
   group boundaries in the AST. Never check adjacency in the flat
   `Vec<UtteranceContent>` — always use `walk_words` or equivalent
   in-order traversal.

2. **Missing intra-word content:** Overlap markers, CA elements, and other markers can appear inside `Word.content`. If you only check `UtteranceContent::OverlapPoint`, you miss `WordContent::OverlapPoint` (e.g., `butt⌈er⌉`, `a⌈nd`).

3. **Missing annotated variants:** `UtteranceContent::AnnotatedWord` and `AnnotatedGroup` are easy to forget. They contain the same inner types but wrapped in `Annotated<T>`.

4. **BracketedContent access:** `Group.content` is `BracketedContent`, which has `.content: BracketedItems`, which has `.0: Vec<BracketedItem>`. The double `.content.content.0` is not a typo.

5. **Separator counter sync:** In the Mor domain, tag-marker separators (`,` `„` `‡`) count as NLP words because they have %mor items (`cm|cm`, `end|end`, `beg|beg`). Any code counting words in the Mor domain must also count these separators to stay in sync with %mor tier word counts.
