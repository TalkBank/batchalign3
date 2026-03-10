//! Language-specific MWT (Multi-Word Token) override rules.
//!
//! Stanza's neural tokenizer and MWT models have known quirks for certain
//! languages. batchalign2 (`ud.py:659-698`) worked around these with ad-hoc
//! string manipulation. This module replaces that with a typed data model.
//!
//! ## Design
//!
//! Each override is an [`MwtPatch`] with:
//! - A [`MwtMatch`] condition (how to identify the token)
//! - An [`MwtAction`] (what to do with it)
//!
//! Overrides are stored in per-language static tables, looked up by the
//! [`apply_mwt_patches`] function after the character-alignment merge step.

/// How to match a token for patching.
#[derive(Debug, Clone)]
pub enum MwtMatch {
    /// Exact case-insensitive match on the token text.
    Exact(&'static str),
    /// Token ends with this suffix (case-insensitive).
    EndsWith(&'static str),
    /// Token is a tuple `(text, True)` from Stanza's MWT model with exact text.
    MwtTaggedExact(&'static str),
}

/// What to do with a matched token.
#[derive(Debug, Clone)]
pub enum MwtAction {
    /// Force MWT expansion: emit `(text, True)`.
    ForceMwt,
    /// Suppress MWT expansion: emit `(text, False)`.
    SuppressMwt,
    /// Replace with a plain string (no MWT hint — let model decide).
    PlainText(&'static str),
    /// Split on apostrophe into parts, each with `(part, False)`.
    /// Used for French elision prefixes like "jusqu'à" → ("jusqu'", False), ("à", False).
    /// Currently handled by `is_french_elision` inline logic; retained for future use.
    #[allow(dead_code)]
    SplitElision,
}

/// A single MWT override rule.
#[derive(Debug, Clone)]
pub struct MwtPatch {
    /// Condition for matching this rule.
    pub match_rule: MwtMatch,
    /// Action to take when matched.
    pub action: MwtAction,
}

/// An adjacent-token merge rule (e.g. Italian "le" + "i" → "lei").
#[derive(Debug, Clone)]
pub struct MwtMergeRule {
    /// Text of the preceding token (case-insensitive).
    pub prev_text: &'static str,
    /// Text of the current token (case-insensitive).
    pub curr_text: &'static str,
    /// Merged replacement text.
    pub merged_text: &'static str,
}

/// Result of applying patches to a token list.
///
/// Encodes Stanza's postprocessor MWT hint convention:
/// - `Plain` — no hint, let Stanza's model decide
/// - `Hint(text, true)` — force MWT expansion
/// - `Hint(text, false)` — suppress MWT expansion
#[derive(Debug, Clone, PartialEq)]
pub enum PatchedToken {
    /// Plain string — no MWT hint.
    Plain(String),
    /// MWT hint tuple: `(text, should_expand)`.
    Hint(String, bool),
}

impl PatchedToken {
    /// Extract the text content regardless of variant.
    pub fn text(&self) -> &str {
        match self {
            PatchedToken::Plain(s) | PatchedToken::Hint(s, _) => s,
        }
    }
}

// ─── Per-language override tables ────────────────────────────────────────────

/// French elision prefixes that should be split on apostrophe.
/// e.g. "jusqu'à" → ("jusqu'", False), ("à", False)
const FRENCH_ELISION_PREFIXES: &[&str] = &["jusqu", "puisqu", "quelqu", "aujourd"];

/// French MWT override rules.
///
/// Sources: batchalign2 ud.py lines 671-689
static FRENCH_PATCHES: &[MwtPatch] = &[
    // "aujourd'hui" → plain string (not a contraction)
    MwtPatch {
        match_rule: MwtMatch::Exact("aujourd'hui"),
        action: MwtAction::PlainText("aujourd'hui"),
    },
    // "au" → force MWT expansion (à + le)
    MwtPatch {
        match_rule: MwtMatch::Exact("au"),
        action: MwtAction::ForceMwt,
    },
];

/// Italian MWT override rules.
///
/// Sources: batchalign2 ud.py lines 662-668
static ITALIAN_PATCHES: &[MwtPatch] = &[
    // l' tagged as MWT by Stanza → suppress expansion
    MwtPatch {
        match_rule: MwtMatch::MwtTaggedExact("l'"),
        action: MwtAction::SuppressMwt,
    },
];

/// Italian adjacent-token merge rules.
static ITALIAN_MERGES: &[MwtMergeRule] = &[
    // Stanza incorrectly splits "lei" into "le" + "i"
    MwtMergeRule {
        prev_text: "le",
        curr_text: "i",
        merged_text: "lei",
    },
];

/// Portuguese MWT override rules.
///
/// Sources: batchalign2 ud.py lines 669-670
static PORTUGUESE_PATCHES: &[MwtPatch] = &[
    // d'água → force MWT expansion
    MwtPatch {
        match_rule: MwtMatch::Exact("d'água"),
        action: MwtAction::ForceMwt,
    },
];

/// Dutch MWT override rules.
///
/// Sources: batchalign2 ud.py lines 694-695
static DUTCH_PATCHES: &[MwtPatch] = &[
    // 's possessive → suppress MWT expansion
    MwtPatch {
        match_rule: MwtMatch::EndsWith("'s"),
        action: MwtAction::SuppressMwt,
    },
];

// ─── Patch application ──────────────────────────────────────────────────────

/// Look up the override tables for a given language.
fn patches_for_lang(alpha2: &str) -> (&'static [MwtPatch], &'static [MwtMergeRule]) {
    match alpha2 {
        "fr" => (FRENCH_PATCHES, &[]),
        "it" => (ITALIAN_PATCHES, ITALIAN_MERGES),
        "pt" => (PORTUGUESE_PATCHES, &[]),
        "nl" => (DUTCH_PATCHES, &[]),
        _ => (&[], &[]),
    }
}

/// Check if a token text matches a match rule.
fn rule_matches(text: &str, is_mwt_tagged: bool, rule: &MwtMatch) -> bool {
    let lower = text.to_lowercase();
    match rule {
        MwtMatch::Exact(expected) => lower == *expected,
        MwtMatch::EndsWith(suffix) => lower.ends_with(suffix),
        MwtMatch::MwtTaggedExact(expected) => is_mwt_tagged && lower == *expected,
    }
}

/// Check if a French token should be split on elision.
///
/// Returns true for tokens like "jusqu'à", "puisqu'il" where the prefix
/// before the first apostrophe is a known elision prefix.
fn is_french_elision(text: &str) -> bool {
    if !text.contains('\'') {
        return false;
    }
    if let Some(prefix) = text.split('\'').next() {
        FRENCH_ELISION_PREFIXES.contains(&prefix.to_lowercase().as_str())
    } else {
        false
    }
}

/// Check if a French token has multiple clitics (2+ apostrophes).
///
/// e.g. "d'l'attraper" → should be split into ("d'", False), ("l'", False), ("attraper", False)
fn is_french_multi_clitic(text: &str) -> bool {
    text.chars().filter(|&c| c == '\'').count() >= 2
}

/// Apply language-specific MWT patches to a list of tokens.
///
/// This is called after the character-alignment merge step. Input tokens
/// are `PatchedToken` values (either plain strings or MWT hint tuples).
///
/// The function is pure: no side effects, no string hacking on CHAT text.
/// All decisions are driven by the static override tables above.
pub fn apply_mwt_patches(tokens: Vec<PatchedToken>, alpha2: &str) -> Vec<PatchedToken> {
    let (patches, merges) = patches_for_lang(alpha2);
    if patches.is_empty() && merges.is_empty() && alpha2 != "fr" {
        return tokens;
    }

    let mut result: Vec<PatchedToken> = Vec::with_capacity(tokens.len());

    for tok in &tokens {
        let text = tok.text();
        let is_mwt_tagged = matches!(tok, PatchedToken::Hint(_, true));

        // Check adjacent-token merge rules
        if !merges.is_empty() {
            let text_lower = text.to_lowercase();
            let mut merged = false;
            for rule in merges {
                if text_lower == rule.curr_text
                    && let Some(prev) = result.last()
                    && prev.text().to_lowercase() == rule.prev_text
                {
                    result.pop();
                    result.push(PatchedToken::Plain(rule.merged_text.to_string()));
                    merged = true;
                    break;
                }
            }
            if merged {
                continue;
            }
        }

        // Check single-token patch rules
        let mut patched = false;
        for patch in patches {
            if rule_matches(text, is_mwt_tagged, &patch.match_rule) {
                match &patch.action {
                    MwtAction::ForceMwt => {
                        result.push(PatchedToken::Hint(text.to_string(), true));
                    }
                    MwtAction::SuppressMwt => {
                        result.push(PatchedToken::Hint(text.to_string(), false));
                    }
                    MwtAction::PlainText(replacement) => {
                        result.push(PatchedToken::Plain(replacement.to_string()));
                    }
                    MwtAction::SplitElision => {
                        // Split on first apostrophe
                        if let Some(pos) = text.find('\'') {
                            let (before, after) = text.split_at(pos + 1);
                            result.push(PatchedToken::Hint(before.to_string(), false));
                            if !after.is_empty() {
                                result.push(PatchedToken::Hint(after.to_string(), false));
                            }
                        } else {
                            result.push(tok.clone());
                        }
                    }
                }
                patched = true;
                break;
            }
        }
        if patched {
            continue;
        }

        // French-specific: multi-clitic splitting and elision prefix splitting
        if alpha2 == "fr" {
            if is_french_multi_clitic(text) {
                // Split on each apostrophe: "d'l'attraper" → "d'" + "l'" + "attraper"
                let parts: Vec<&str> = text.split('\'').collect();
                for (i, part) in parts.iter().enumerate() {
                    if i < parts.len() - 1 {
                        result.push(PatchedToken::Hint(format!("{part}'"), false));
                    } else if !part.is_empty() {
                        result.push(PatchedToken::Hint(part.to_string(), false));
                    }
                }
                continue;
            }

            if is_french_elision(text)
                && let Some(pos) = text.find('\'')
            {
                let (before, after) = text.split_at(pos + 1);
                result.push(PatchedToken::Hint(before.to_string(), false));
                if !after.is_empty() {
                    result.push(PatchedToken::Hint(after.to_string(), false));
                }
                continue;
            }
        }

        // No patch applied — pass through unchanged
        result.push(tok.clone());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── French ──────────────────────────────────────────────────────────

    #[test]
    fn test_french_aujourdhui_becomes_plain() {
        let tokens = vec![PatchedToken::Plain("aujourd'hui".into())];
        let result = apply_mwt_patches(tokens, "fr");
        assert_eq!(result, vec![PatchedToken::Plain("aujourd'hui".into())]);
    }

    #[test]
    fn test_french_au_forces_mwt() {
        let tokens = vec![PatchedToken::Plain("au".into())];
        let result = apply_mwt_patches(tokens, "fr");
        assert_eq!(result, vec![PatchedToken::Hint("au".into(), true)]);
    }

    #[test]
    fn test_french_multi_clitic_split() {
        let tokens = vec![PatchedToken::Plain("d'l'attraper".into())];
        let result = apply_mwt_patches(tokens, "fr");
        assert_eq!(
            result,
            vec![
                PatchedToken::Hint("d'".into(), false),
                PatchedToken::Hint("l'".into(), false),
                PatchedToken::Hint("attraper".into(), false),
            ]
        );
    }

    #[test]
    fn test_french_elision_prefix_split() {
        let tokens = vec![PatchedToken::Plain("jusqu'à".into())];
        let result = apply_mwt_patches(tokens, "fr");
        assert_eq!(
            result,
            vec![
                PatchedToken::Hint("jusqu'".into(), false),
                PatchedToken::Hint("à".into(), false),
            ]
        );
    }

    #[test]
    fn test_french_puisqu_elision() {
        let tokens = vec![PatchedToken::Plain("puisqu'il".into())];
        let result = apply_mwt_patches(tokens, "fr");
        assert_eq!(
            result,
            vec![
                PatchedToken::Hint("puisqu'".into(), false),
                PatchedToken::Hint("il".into(), false),
            ]
        );
    }

    #[test]
    fn test_french_regular_word_unchanged() {
        let tokens = vec![PatchedToken::Plain("maison".into())];
        let result = apply_mwt_patches(tokens, "fr");
        assert_eq!(result, vec![PatchedToken::Plain("maison".into())]);
    }

    // ── Italian ─────────────────────────────────────────────────────────

    #[test]
    fn test_italian_l_prime_mwt_suppressed() {
        let tokens = vec![PatchedToken::Hint("l'".into(), true)];
        let result = apply_mwt_patches(tokens, "it");
        assert_eq!(result, vec![PatchedToken::Hint("l'".into(), false)]);
    }

    #[test]
    fn test_italian_l_prime_plain_unchanged() {
        let tokens = vec![PatchedToken::Plain("l'".into())];
        let result = apply_mwt_patches(tokens, "it");
        assert_eq!(result, vec![PatchedToken::Plain("l'".into())]);
    }

    #[test]
    fn test_italian_lei_merge() {
        let tokens = vec![
            PatchedToken::Plain("le".into()),
            PatchedToken::Plain("i".into()),
        ];
        let result = apply_mwt_patches(tokens, "it");
        assert_eq!(result, vec![PatchedToken::Plain("lei".into())]);
    }

    #[test]
    fn test_italian_le_followed_by_other_unchanged() {
        let tokens = vec![
            PatchedToken::Plain("le".into()),
            PatchedToken::Plain("case".into()),
        ];
        let result = apply_mwt_patches(tokens, "it");
        assert_eq!(
            result,
            vec![
                PatchedToken::Plain("le".into()),
                PatchedToken::Plain("case".into()),
            ]
        );
    }

    // ── Portuguese ──────────────────────────────────────────────────────

    #[test]
    fn test_portuguese_dagua_forces_mwt() {
        let tokens = vec![PatchedToken::Plain("d'água".into())];
        let result = apply_mwt_patches(tokens, "pt");
        assert_eq!(result, vec![PatchedToken::Hint("d'água".into(), true)]);
    }

    // ── Dutch ───────────────────────────────────────────────────────────

    #[test]
    fn test_dutch_possessive_s_suppressed() {
        let tokens = vec![PatchedToken::Plain("vader's".into())];
        let result = apply_mwt_patches(tokens, "nl");
        assert_eq!(result, vec![PatchedToken::Hint("vader's".into(), false)]);
    }

    #[test]
    fn test_dutch_regular_word_unchanged() {
        let tokens = vec![PatchedToken::Plain("huis".into())];
        let result = apply_mwt_patches(tokens, "nl");
        assert_eq!(result, vec![PatchedToken::Plain("huis".into())]);
    }

    // ── No patches for unknown languages ────────────────────────────────

    #[test]
    fn test_unknown_language_passthrough() {
        let tokens = vec![PatchedToken::Plain("hello".into())];
        let result = apply_mwt_patches(tokens, "xx");
        assert_eq!(result, vec![PatchedToken::Plain("hello".into())]);
    }

    // ── English unchanged ───────────────────────────────────────────────

    #[test]
    fn test_english_passthrough() {
        let tokens = vec![PatchedToken::Hint("don't".into(), true)];
        let result = apply_mwt_patches(tokens, "en");
        assert_eq!(result, vec![PatchedToken::Hint("don't".into(), true)]);
    }
}
