//! Single-word UD-to-CHAT MOR mapping.
//!
//! Converts individual UD words into CHAT `Mor` structures, including
//! lemma cleaning, POS name mapping, and POS-specific feature dispatch.

use super::features::{adj_features, det_features, noun_features, pron_features, verb_features};
use super::mapping::{MappingContext, MappingError, lang2};
use super::{UdPunctable, UdWord, UniversalPos, lang_ja, sanitize_mor_text};
use smallvec::SmallVec;
use std::collections::HashMap;
use talkbank_model::model::dependent_tier::mor::{Mor, MorFeature, MorStem, MorWord, PosCategory};

/// Maps a raw UD word to a CHAT Mor structure.
///
/// ## Processing Order (matches Python master exactly)
///
/// 1. Clean lemma (strip special chars, handle unknowns — Python `handler()`)
/// 2. Apply Japanese verb form overrides (can change POS + lemma)
/// 3. Map POS category (lowercased UPOS name — Python convention)
/// 4. Compute POS-specific suffixes (dispatch on original UPOS)
/// 5. Sanitize and assemble
///
/// ## Data Integrity
///
/// To prevent "Syntactic Contamination" (reserved CHAT characters like `|` or `#` leaking
/// into the AST), every field is individually validated and sanitized via `sanitize_mor_text`.
pub fn map_ud_word_to_mor(ud: &UdWord, ctx: &MappingContext) -> Result<Mor, MappingError> {
    // Early return for actual punctuation (matches Python handle() dispatch)
    if matches!(ud.lemma.as_str(), "." | "!" | "?" | "," | "$,") {
        return Ok(map_actual_punct(ud));
    }

    // 1. Parse UD features
    let feats = parse_feats(ud.feats.as_deref());

    // 2. Clean lemma (Python handler() cleanup)
    let (mut cleaned_lemma, _is_unknown) = clean_lemma(&ud.lemma, &ud.text);

    // 3. Determine effective POS name (lowercased UPOS, possibly overridden by Japanese rules)
    let mut effective_pos = upos_to_name(&ud.upos).to_string();
    if lang2(&ctx.lang) == "ja"
        && let Some(ovr) = lang_ja::japanese_verbform(&effective_pos, &cleaned_lemma, &ud.text)
    {
        effective_pos = ovr.pos.to_string();
        cleaned_lemma = ovr.lemma.to_string();
        cleaned_lemma = cleaned_lemma.replace(',', "cm");
    }

    // 4. Japanese PUNCT → cm; Japanese comma lemma → cm
    if lang2(&ctx.lang) == "ja" {
        if matches!(ud.upos, UdPunctable::Value(UniversalPos::Punct)) {
            effective_pos = "cm".to_string();
        }
        if ud.lemma == "、" || ud.lemma == "," {
            effective_pos = "cm".to_string();
        }
    }

    // 5. Compute POS-specific features (dispatch on ORIGINAL UPOS)
    let features = compute_features(&ud.upos, &feats, &effective_pos, ud, ctx);

    // 6. Sanitize the cleaned lemma
    let sanitized_lemma = sanitize_mor_text(&cleaned_lemma);

    // Reject empty stems at construction time. An empty stem serializes as
    // "pos|" (bare pipe) which the strict parser rejects as E342, and the
    // AST-level validator reports as E711.
    if sanitized_lemma.is_empty() {
        return Err(MappingError::EmptyStem {
            word: ud.text.clone(),
            lemma: ud.lemma.clone(),
            upos: format!("{:?}", &ud.upos),
        });
    }

    // 7. Build MorWord
    let mor_word = MorWord::new(
        PosCategory::new(&effective_pos),
        MorStem::new(sanitized_lemma),
    )
    .with_features(features);

    Ok(Mor::new(mor_word))
}

/// Early return for actual punctuation marks (comma → cm|cm).
fn map_actual_punct(ud: &UdWord) -> Mor {
    let (pos_name, stem) = if ud.lemma == "," || ud.lemma == "$," {
        ("cm", "cm")
    } else {
        ("punct", ud.lemma.as_str())
    };

    let mor_word = MorWord::new(PosCategory::new(pos_name), MorStem::new(stem));
    Mor::new(mor_word)
}

/// Parse UD features string (e.g. "Number=Plur|Person=3") into a Map.
pub(super) fn parse_feats(feats: Option<&str>) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if let Some(f) = feats {
        for pair in f.split('|') {
            let mut parts = pair.split('=');
            if let (Some(key), Some(value)) = (parts.next(), parts.next()) {
                // UD multi-value features use commas (e.g. PronType=Int,Rel).
                // We preserve the comma as-is to respect UD conventions and
                // avoid lossy concatenation ("Int,Rel" stays "Int,Rel", not
                // "IntRel").  The tree-sitter grammar and %mor parser both
                // accept commas in suffix values.
                map.insert(key.to_string(), value.to_string());
            }
        }
    }
    map
}

// ─── Lemma Cleaning ──────────────────────────────────────────────────────────
//
// Matches Python `handler()` from `ud.py` — the generic lemma cleanup that runs
// for ALL parts of speech before POS-specific suffix handlers.

/// Cleans a UD lemma for use as a CHAT %mor stem.
///
/// Returns `(cleaned_lemma, is_unknown)`. `is_unknown` is true when the lemma
/// starts with `0` (CHAT omission marker), matching Python's `unknown` flag.
pub(super) fn clean_lemma(lemma: &str, text: &str) -> (String, bool) {
    let mut target = lemma.to_string();
    let mut unknown = false;

    // Handle Japanese quotes
    if target.trim() == "\u{300D}" || target.trim() == "\u{300C}" {
        target = text.to_string();
    }
    if target == "\"" {
        target = text.to_string();
    }
    if target.is_empty() {
        target = text.to_string();
    }
    target = target.replace(['\u{300D}', '\u{300C}'], "");

    // Unknown flag: lemma starts with 0
    if target.starts_with('0') && target.len() > 1 {
        // Python: target = word.text[1:]
        if text.len() > 1 {
            target = text[1..].to_string();
        }
        unknown = true;
    }

    // <SOS> token → fall back to surface form
    if target.contains("<SOS>") {
        target = text.to_string();
    }

    // Strip various characters
    target = target.replace(['$', '.'], "");

    // Strip leading/trailing dashes
    if target.starts_with('-') && target.len() > 1 {
        target = target[1..].to_string();
    }
    if target.ends_with('-') && target.len() > 1 {
        target = target[..target.len() - 1].to_string();
    }

    // Replace double dashes
    target = target.replace("--", "-");
    target = target.replace("--", "-");

    // Strip multi-char patterns
    target = target.replace("<unk>", "");
    target = target.replace("<SOS>", "");
    target = target.replace("/100", "");
    target = target.replace("/r", "");
    // Strip single characters
    target = target.replace([',', '\'', '~', '(', ')'], "");

    // If pipe in lemma, take everything before it
    if target.contains('|') {
        target = target.split('|').next().unwrap_or("").trim().to_string();
    }

    // Clean out alternate spellings
    target = target.replace(['_', '+'], "");

    // "door zogen" special case
    if target == "door zogen" {
        target = text.to_string();
    }

    // Fix dash: ASCII hyphen → en-dash (U+2013)
    target = target.replace('-', "\u{2013}");

    // Smart quote check: left double quotation mark (U+201C)
    if target.contains('\u{201C}') {
        target = text.to_string();
    }

    // Strip @\w trailing pattern (Python: re.sub(r'@\w$', '', target))
    let chars: Vec<char> = target.chars().collect();
    if chars.len() >= 2
        && chars[chars.len() - 2] == '@'
        && (chars[chars.len() - 1].is_alphanumeric() || chars[chars.len() - 1] == '_')
    {
        target = chars[..chars.len() - 2].iter().collect::<String>();
    }

    target = target.trim().to_string();

    // Final safeguard: after all stripping, never return an empty lemma.
    //
    // Example: Stanza's GUM MWT model can treat a possessive like "Claus'" as
    // MWT, splitting it into [Claus (PROPN), ' (PUNCT)].  The apostrophe token
    // has lemma="'", which gets stripped to "" by line 423 above.  An empty
    // lemma produces MorStem::new("") → "punct|" (bare pipe, no stem) → E342.
    //
    // This is a Stanza model quirk: possessive apostrophes are not true MWT
    // contractions and should not be split.  Python master avoids this by
    // returning (text, False) tuples from tokenizer_processor, preventing MWT
    // re-expansion of merged CHAT words.  Our _tokenizer_realign.py uses plain
    // strings instead, so the MWT model may re-expand possessives.  We accept
    // this as a known divergence (documented in docs/mwt-handling.md) and guard
    // defensively here so we never emit an invalid stem.
    if target.is_empty() && !text.is_empty() {
        target = text.to_string();
    }
    if target.is_empty() {
        target = "x".to_string();
    }

    (target, unknown)
}

// ─── POS Name Mapping ────────────────────────────────────────────────────────
//
// Python convention: POS category = UPOS lowercased. No subcategories from xpos.

/// Maps a UD UPOS tag to the Python-convention POS category name.
///
/// Python does `word.upos.lower()` to produce the POS category:
/// NOUN → "noun", VERB → "verb", PRON → "pron", ADP → "adp", etc.
fn upos_to_name(upos: &UdPunctable<UniversalPos>) -> &'static str {
    match upos {
        UdPunctable::Value(UniversalPos::Noun) => "noun",
        UdPunctable::Value(UniversalPos::Propn) => "propn",
        UdPunctable::Value(UniversalPos::Verb) => "verb",
        UdPunctable::Value(UniversalPos::Aux) => "aux",
        UdPunctable::Value(UniversalPos::Adj) => "adj",
        UdPunctable::Value(UniversalPos::Adv) => "adv",
        UdPunctable::Value(UniversalPos::Det) => "det",
        UdPunctable::Value(UniversalPos::Adp) => "adp",
        UdPunctable::Value(UniversalPos::Num) => "num",
        UdPunctable::Value(UniversalPos::Part) => "part",
        UdPunctable::Value(UniversalPos::Intj) => "intj",
        UdPunctable::Value(UniversalPos::Pron) => "pron",
        UdPunctable::Value(UniversalPos::Cconj) => "cconj",
        UdPunctable::Value(UniversalPos::Sconj) => "sconj",
        UdPunctable::Value(UniversalPos::Sym) => "x",
        UdPunctable::Value(UniversalPos::Punct) => "punct",
        UdPunctable::Value(UniversalPos::X) => "x",
        UdPunctable::Punct(_) => "punct",
    }
}

// ─── Suffix Dispatch ─────────────────────────────────────────────────────────

/// Pushes a non-empty feature part.
pub(super) fn push_feature(features: &mut SmallVec<[MorFeature; 4]>, value: &str) {
    if !value.is_empty() {
        features.push(MorFeature::flat(value));
    }
}

/// Pushes a UD feature value as a feature if present.
pub(super) fn push_feat(
    features: &mut SmallVec<[MorFeature; 4]>,
    feats: &HashMap<String, String>,
    key: &str,
) {
    if let Some(val) = feats.get(key) {
        push_feature(features, val);
    }
}

/// Dispatches to the appropriate POS-specific feature handler.
///
/// Dispatch is on the ORIGINAL UPOS (like Python's `HANDLERS` dict),
/// not the effective POS (which may have been changed by Japanese overrides).
fn compute_features(
    original_upos: &UdPunctable<UniversalPos>,
    feats: &HashMap<String, String>,
    effective_pos: &str,
    ud: &UdWord,
    ctx: &MappingContext,
) -> SmallVec<[MorFeature; 4]> {
    match original_upos {
        UdPunctable::Value(UniversalPos::Verb | UniversalPos::Aux) => {
            verb_features(feats, effective_pos, ud, ctx)
        }
        UdPunctable::Value(UniversalPos::Pron) => pron_features(feats, ud, ctx),
        UdPunctable::Value(UniversalPos::Det) => det_features(feats, ctx),
        UdPunctable::Value(UniversalPos::Adj) => adj_features(feats),
        UdPunctable::Value(UniversalPos::Noun | UniversalPos::Propn) => {
            noun_features(feats, ud, ctx)
        }
        // SYM goes through PUNCT handler in Python, which uses no features
        UdPunctable::Value(UniversalPos::Sym | UniversalPos::Punct) => SmallVec::new(),
        // All other POS types: no features (matches Python generic handler)
        _ => SmallVec::new(),
    }
}

/// Returns true if the token text represents a known clitic for the given language.
pub(super) fn is_clitic(text: &str, ctx: &MappingContext) -> bool {
    match lang2(&ctx.lang) {
        "en" => text == "n't" || text == "'s" || text == "'ve" || text == "'ll",
        "fr" => text.ends_with('\'') || text == "-ce" || text == "-être" || text == "-là",
        "it" => text.ends_with('\''),
        _ => false,
    }
}
