//! POS-specific feature handlers for UD-to-CHAT morphosyntax mapping.
//!
//! Each function produces the suffix sequence for a specific POS category,
//! matching the Python master's `handler__VERB`, `handler__PRON`, etc.

use super::mor_word::{push_feat, push_feature};
use super::{UdWord, lang_en, lang_fr};
use smallvec::SmallVec;
use std::collections::HashMap;
use talkbank_model::model::dependent_tier::mor::MorFeature;

use super::mapping::{MappingContext, lang2};

// ─── VERB/AUX Suffixes (Python handler__VERB) ───────────────────────────────
//
// Format: -VerbForm-Aspect-Mood-Tense-Polarity-Polite-HebBinyan-HebExistential-NumberPerson-irr

pub(super) fn verb_features(
    feats: &HashMap<String, String>,
    effective_pos: &str,
    ud: &UdWord,
    ctx: &MappingContext,
) -> SmallVec<[MorFeature; 4]> {
    // Python: if "sconj" in res → skip verb features
    if effective_pos.contains("sconj") {
        return SmallVec::new();
    }

    // Python: elif word.text == "ろ" → skip verb features
    if ud.text == "\u{308D}" {
        // ろ
        return SmallVec::new();
    }

    // Python: elif "verb" not in res and "aux" not in res
    if !effective_pos.contains("verb") && !effective_pos.contains("aux") {
        // Special case: "たり" gets "-Inf-S"
        if ud.text == "\u{305F}\u{308A}" {
            // たり
            let mut s = SmallVec::new();
            push_feature(&mut s, "Inf");
            push_feature(&mut s, "S");
            return s;
        }
        return SmallVec::new();
    }

    let mut suffixes = SmallVec::new();

    // VerbForm (always present, default "Inf")
    let verb_form = feats
        .get("VerbForm")
        .cloned()
        .unwrap_or_else(|| "Inf".to_string());
    push_feature(&mut suffixes, &verb_form);

    // Aspect, Mood, Tense, Polarity, Polite
    push_feat(&mut suffixes, feats, "Aspect");
    push_feat(&mut suffixes, feats, "Mood");
    push_feat(&mut suffixes, feats, "Tense");
    push_feat(&mut suffixes, feats, "Polarity");
    push_feat(&mut suffixes, feats, "Polite");

    // Hebrew-specific (lowercase)
    if let Some(v) = feats.get("HebBinyan") {
        push_feature(&mut suffixes, &v.to_lowercase());
    }
    if let Some(v) = feats.get("HebExistential") {
        push_feature(&mut suffixes, &v.to_lowercase());
    }

    // Number + Person (e.g. "S3", "P1")
    // Python: person defaults to "", number defaults to "Sing"
    let person_raw = feats.get("Person").map(|s| s.as_str()).unwrap_or("");
    let number_raw = feats.get("Number").map(|s| s.as_str()).unwrap_or("Sing");
    let number_char = number_raw.chars().next().unwrap_or('S');
    let person_str = if person_raw == "0" { "4" } else { person_raw };
    let num_person = format!("{}{}", number_char, person_str);
    push_feature(&mut suffixes, &num_person);

    // English irregular past tense
    if lang2(&ctx.lang) == "en"
        && let Some(tense) = feats.get("Tense")
        && tense == "Past"
        && lang_en::is_irregular(&ud.lemma, &ud.text)
    {
        push_feature(&mut suffixes, "irr");
    }

    suffixes
}

// ─── PRON Suffixes (Python handler__PRON) ────────────────────────────────────
//
// Format: -PronType-Case-Reflex-NumberPerson

pub(super) fn pron_features(
    feats: &HashMap<String, String>,
    ud: &UdWord,
    ctx: &MappingContext,
) -> SmallVec<[MorFeature; 4]> {
    let mut parts = Vec::new();

    // 1. PronType (default "Int" — matches Python)
    let pron_type = feats.get("PronType").map(|s| s.as_str()).unwrap_or("Int");
    parts.push(pron_type.to_string());

    // 2. Case — French uses word-level case lookup
    let case = if lang2(&ctx.lang) == "fr" {
        lang_fr::french_pronoun_case(&ud.text).to_string()
    } else {
        feats.get("Case").cloned().unwrap_or_default()
    };
    if !case.is_empty() {
        parts.push(case);
    }

    // 3. Reflex (Yes → "reflx")
    if let Some(reflex) = feats.get("Reflex")
        && reflex == "Yes"
    {
        parts.push("reflx".to_string());
    }

    // 4. Number + Person (e.g., "S3", "P1")
    // Special case: "that", "who" get no number string
    if ud.text != "that" && ud.text != "who" {
        let person_raw = feats.get("Person").map(|s| s.as_str()).unwrap_or("1");
        let person_str = if person_raw == "0" { "4" } else { person_raw };
        let number = feats
            .get("Number")
            .map(|n| if n.starts_with('P') { "P" } else { "S" })
            .unwrap_or("S");
        parts.push(format!("{}{}", number, person_str));
    }

    // Filter empty parts and join
    let non_empty: Vec<&str> = parts
        .iter()
        .map(|s| s.as_str())
        .filter(|s| !s.is_empty())
        .collect();
    if non_empty.is_empty() {
        SmallVec::new()
    } else {
        let mut suffixes = SmallVec::new();
        for part in non_empty {
            push_feature(&mut suffixes, part);
        }
        suffixes
    }
}

// ─── DET Suffixes (Python handler__DET) ──────────────────────────────────────
//
// Format: -Gender-Definite-PronType-Number-Psor

pub(super) fn det_features(
    feats: &HashMap<String, String>,
    ctx: &MappingContext,
) -> SmallVec<[MorFeature; 4]> {
    let mut suffixes = SmallVec::new();

    let number = feats.get("Number").map(|s| s.as_str()).unwrap_or("");

    // Gender (with French default and common-value clearing)
    let gender_default = if lang2(&ctx.lang) == "fr" {
        if number == "Plur" { "" } else { "Masc" }
    } else {
        ""
    };
    let gender = feats
        .get("Gender")
        .cloned()
        .unwrap_or_else(|| gender_default.to_string());
    // Clear common defaults that mean "unspecified"
    if !gender.is_empty() && gender != "Com,Neut" && gender != "Com" {
        push_feature(&mut suffixes, &gender);
    }

    // Definite (always present, default "Def")
    let definite = feats.get("Definite").map(|s| s.as_str()).unwrap_or("Def");
    push_feature(&mut suffixes, definite);

    // PronType
    push_feat(&mut suffixes, feats, "PronType");

    // Number
    push_feature(&mut suffixes, number);

    // Psor: Number[psor] + Person[psor]
    let np = feats
        .get("Number[psor]")
        .and_then(|s| s.chars().next())
        .map(|c| c.to_string())
        .unwrap_or_default();
    let pp = feats.get("Person[psor]").map(|s| s.as_str()).unwrap_or("");
    let psor = format!("{}{}", np, pp);
    push_feature(&mut suffixes, &psor);

    suffixes
}

// ─── ADJ Suffixes (Python handler__ADJ) ──────────────────────────────────────
//
// Format: -Degree-Case-NumberPerson

pub(super) fn adj_features(feats: &HashMap<String, String>) -> SmallVec<[MorFeature; 4]> {
    let mut suffixes = SmallVec::new();

    // Degree (default "Pos", but "Pos" is cleared to empty)
    let degree = feats.get("Degree").map(|s| s.as_str()).unwrap_or("Pos");
    if degree != "Pos" {
        push_feature(&mut suffixes, degree);
    }

    // Case
    if let Some(case) = feats.get("Case") {
        push_feature(&mut suffixes, case);
    }

    // Number + Person
    let number = feats
        .get("Number")
        .and_then(|s| s.chars().next())
        .unwrap_or('S');
    let person_raw = feats.get("Person").map(|s| s.as_str()).unwrap_or("1");
    let person_str = if person_raw == "0" { "4" } else { person_raw };
    push_feature(&mut suffixes, &format!("{}{}", number, person_str));

    suffixes
}

// ─── NOUN/PROPN Suffixes (Python handler__NOUN) ─────────────────────────────
//
// Format: -Gender-Number-Case-PronType-Ger-Apm

pub(super) fn noun_features(
    feats: &HashMap<String, String>,
    ud: &UdWord,
    ctx: &MappingContext,
) -> SmallVec<[MorFeature; 4]> {
    let mut suffixes = SmallVec::new();

    // Gender (default "Com,Neut", cleared for common values)
    let gender = feats
        .get("Gender")
        .cloned()
        .unwrap_or_else(|| "Com,Neut".to_string());
    if gender != "Com,Neut" && gender != "Com" {
        push_feature(&mut suffixes, &gender);
    }

    // Number (default "Sing", cleared for "Sing")
    let number = feats.get("Number").map(|s| s.as_str()).unwrap_or("Sing");
    if number != "Sing" {
        push_feature(&mut suffixes, number);
    }

    // Case (deprel "obj" without Case → "Acc")
    let case = feats.get("Case").cloned().unwrap_or_else(|| {
        if ud.deprel == "obj" {
            "Acc".to_string()
        } else {
            String::new()
        }
    });
    push_feature(&mut suffixes, &case);

    // PronType
    push_feat(&mut suffixes, feats, "PronType");

    // Gerund (English -ing words tagged as NOUN)
    if lang2(&ctx.lang) == "en" && ud.text.ends_with("ing") {
        push_feature(&mut suffixes, "Ger");
    }

    // French APM (auditory plural marking)
    if lang2(&ctx.lang) == "fr" && number == "Plur" && lang_fr::is_apm_noun(&ud.text) {
        push_feature(&mut suffixes, "Apm");
    }

    suffixes
}
