//! Merge consecutive single-letter words that form known abbreviations.
//!
//! ASR engines typically emit abbreviations as individual letters (e.g.,
//! `*CHI: F B I .`). This transform collapses such letter sequences back
//! into a single word when the concatenation matches a known abbreviation
//! from `data/abbrev.json`.
//!
//! This is the AST-based replacement for BA2's `tostring(merge_abbrev=True)`,
//! which operated on serialized CHAT strings.

use std::collections::HashSet;
use std::sync::LazyLock;

use talkbank_model::model::{ChatFile, Line, UtteranceContent, Word};

/// Known abbreviations (original case from `abbrev.json`).
///
/// Entries are stored in their canonical case (mostly uppercase). Matching
/// is case-insensitive: we uppercase the concatenated letters before lookup.
static ABBREV: LazyLock<HashSet<String>> = LazyLock::new(|| {
    let data: Vec<String> = serde_json::from_str(include_str!("../data/abbrev.json"))
        .expect("embedded abbrev.json is valid");
    // Store uppercased for case-insensitive matching
    data.into_iter().map(|s| s.to_uppercase()).collect()
});

/// Merge consecutive single-letter words matching known abbreviations.
///
/// Walks all utterances in the file and replaces runs of single-letter
/// `Word` items whose uppercase concatenation appears in [`ABBREV`] with
/// a single `Word` containing the merged abbreviation.
///
/// Non-word content items (events, pauses, groups, separators, etc.) act
/// as sequence breakers — they prevent merging across boundaries.
pub fn merge_abbreviations(chat_file: &mut ChatFile) {
    for line in chat_file.lines.iter_mut() {
        if let Line::Utterance(utt) = line {
            merge_in_content_items(&mut utt.main.content.content.0);
        }
    }
}

/// Merge abbreviations within a flat content-item list.
///
/// Scans for maximal runs of consecutive single-letter `UtteranceContent::Word`
/// items and checks every sub-sequence (longest first) against `ABBREV`.
fn merge_in_content_items(items: &mut Vec<UtteranceContent>) {
    if items.len() < 2 {
        return;
    }

    let mut result: Vec<UtteranceContent> = Vec::with_capacity(items.len());
    let mut i = 0;

    while i < items.len() {
        // Find a maximal run of single-letter words starting at i
        let run_start = i;
        let mut letters: Vec<String> = Vec::new();

        while i < items.len() {
            if let Some(letter) = single_letter_word(&items[i]) {
                letters.push(letter);
                i += 1;
            } else {
                break;
            }
        }

        if letters.len() < 2 {
            // No run or single letter — emit as-is
            if !letters.is_empty() {
                // Single letter word, not part of a mergeable run
                result.push(items[run_start].clone());
            } else {
                // Non-word item
                result.push(items[i].clone());
                i += 1;
            }
            continue;
        }

        // We have a run of >= 2 single-letter words.
        // Try to match longest-first sub-sequences greedily.
        let mut j = 0;
        while j < letters.len() {
            let mut matched = false;
            // Try longest match first
            let max_len = letters.len() - j;
            for len in (2..=max_len).rev() {
                let candidate: String = letters[j..j + len]
                    .iter()
                    .map(|s| s.to_uppercase())
                    .collect::<String>();
                if ABBREV.contains(&candidate) {
                    // Build the merged word using the original case from the letters
                    let merged_text: String = letters[j..j + len].concat();
                    result.push(UtteranceContent::Word(Box::new(Word::simple(merged_text))));
                    j += len;
                    matched = true;
                    break;
                }
            }
            if !matched {
                // Single letter didn't match any abbreviation prefix — emit as-is
                result.push(items[run_start + j].clone());
                j += 1;
            }
        }
    }

    *items = result;
}

/// If this content item is a bare `Word` with exactly one Unicode letter,
/// return that letter. Returns `None` for annotated words, replaced words,
/// and non-word content.
fn single_letter_word(item: &UtteranceContent) -> Option<String> {
    match item {
        UtteranceContent::Word(w) => {
            let text = w.cleaned_text();
            let mut chars = text.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) if c.is_alphabetic() => Some(c.to_string()),
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_lenient;
    use crate::serialize::to_chat_string;

    /// Helper: parse CHAT, apply merge, serialize, return output.
    fn merge_and_serialize(chat: &str) -> String {
        let (mut file, _) = parse_lenient(chat);
        merge_abbreviations(&mut file);
        to_chat_string(&file)
    }

    /// Extract just the main-tier content words from the output.
    fn main_tier_words(chat_output: &str) -> Vec<String> {
        chat_output
            .lines()
            .filter(|l| l.starts_with('*'))
            .flat_map(|l| {
                let after_colon = l.split_once(':').map(|x| x.1).unwrap_or("");
                after_colon
                    .split_whitespace()
                    .filter(|w| !w.starts_with('\u{15}') && *w != "." && *w != "?" && *w != "!")
                    .map(String::from)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn minimal_chat(utterance: &str) -> String {
        format!(
            "@UTF8\n\
             @Begin\n\
             @Languages:\teng\n\
             @Participants:\tCHI Target_Child\n\
             @ID:\teng|test|CHI|||||Target_Child|||\n\
             *CHI:\t{utterance}\n\
             @End\n"
        )
    }

    #[test]
    fn merge_fbi() {
        let chat = minimal_chat("the F B I is here .");
        let out = merge_and_serialize(&chat);
        let words = main_tier_words(&out);
        assert!(
            words.contains(&"FBI".to_string()),
            "Expected 'FBI' in words: {words:?}"
        );
        assert!(
            !words.contains(&"F".to_string()),
            "Should not contain lone 'F': {words:?}"
        );
    }

    #[test]
    fn merge_cia() {
        let chat = minimal_chat("the C I A agent .");
        let out = merge_and_serialize(&chat);
        let words = main_tier_words(&out);
        assert!(
            words.contains(&"CIA".to_string()),
            "Expected 'CIA' in words: {words:?}"
        );
    }

    #[test]
    fn no_merge_unknown() {
        let chat = minimal_chat("X Y Z Q W .");
        let out = merge_and_serialize(&chat);
        let words = main_tier_words(&out);
        // XYZ is in the list, so it should merge. But Q W is not, so stays separate.
        assert!(
            words.contains(&"XYZ".to_string()),
            "Expected 'XYZ' in words: {words:?}"
        );
        assert!(
            words.contains(&"Q".to_string()),
            "Expected lone 'Q' in words: {words:?}"
        );
        assert!(
            words.contains(&"W".to_string()),
            "Expected lone 'W' in words: {words:?}"
        );
    }

    #[test]
    fn merge_preserves_surrounding_words() {
        let chat = minimal_chat("I saw the F B I today .");
        let out = merge_and_serialize(&chat);
        let words = main_tier_words(&out);
        assert_eq!(
            words,
            vec!["I", "saw", "the", "FBI", "today"],
            "Surrounding words should be preserved"
        );
    }

    #[test]
    fn no_single_letter_merge() {
        // A single letter that happens to be an abbreviation should not be touched
        // (abbreviation merging requires at least 2 letters)
        let chat = minimal_chat("I like A .");
        let out = merge_and_serialize(&chat);
        let words = main_tier_words(&out);
        assert_eq!(words, vec!["I", "like", "A"]);
    }

    #[test]
    fn case_insensitive_matching() {
        let chat = minimal_chat("the f b i is here .");
        let out = merge_and_serialize(&chat);
        let words = main_tier_words(&out);
        assert!(
            words.contains(&"fbi".to_string()),
            "Expected 'fbi' (preserving original case) in words: {words:?}"
        );
    }

    #[test]
    fn no_content_no_crash() {
        let chat = minimal_chat(".");
        let _out = merge_and_serialize(&chat);
        // Should not crash on empty content
    }

    #[test]
    fn greedy_longest_match() {
        // NATO is 4 letters. N+A should not consume letters needed for NATO.
        // NA is also in the abbrev list. Test that longest match wins.
        let chat = minimal_chat("the N A T O treaty .");
        let out = merge_and_serialize(&chat);
        let words = main_tier_words(&out);
        assert!(
            words.contains(&"NATO".to_string()),
            "Expected 'NATO' (longest match) in words: {words:?}"
        );
    }

    #[test]
    fn adjacent_abbreviations() {
        let chat = minimal_chat("F B I C I A .");
        let out = merge_and_serialize(&chat);
        let words = main_tier_words(&out);
        assert!(
            words.contains(&"FBI".to_string()),
            "Expected 'FBI' in words: {words:?}"
        );
        assert!(
            words.contains(&"CIA".to_string()),
            "Expected 'CIA' in words: {words:?}"
        );
    }
}
