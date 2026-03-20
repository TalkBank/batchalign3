//! Post-ASR cleanup: disfluency marking and n-gram retrace detection.
//!
//! Ports batchalign2's `DisfluencyReplacementEngine` and `NgramRetraceEngine`
//! from `batchalign/pipelines/cleanup/`. These ran as pipeline stages after ASR
//! in BA2 and are now integrated into the Rust ASR post-processing pipeline.
//!
//! **Disfluency replacement:** Marks filled pauses ("um" → "&-um") and applies
//! orthographic replacements ("'cause" → "because") from per-language wordlists.
//!
//! **N-gram retrace detection:** Detects repeated n-grams within each utterance
//! and wraps them in CHAT retrace notation (`<word word> [/] word word`).

use super::{AsrNormalizedText, AsrWord, SpeakerIndex, Utterance, WordKind, ENDING_PUNCT, MOR_PUNCT};
use std::collections::HashMap;
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Disfluency wordlists
// ---------------------------------------------------------------------------

/// A word replacement entry: original → main_line_form.
///
/// For filled pauses: original="um", main_line="&-um".
/// For replacements: original="'cause", main_line="(be)cause".
#[derive(Debug, Clone)]
struct Replacement {
    /// What appears on the main line in CHAT (e.g. "&-um", "(be)cause").
    main_line: &'static str,
}

/// Filled pauses for English. From BA2's `support/filled_pauses.eng`.
///
/// Format: (original, main_line).
/// In CHAT, filled pauses appear as `&-{text}` on the main line.
const FILLED_PAUSES_ENG: &[(&str, &str)] = &[
    ("um", "&-um"),
    ("ur", "&-ur"),
    ("uh", "&-uh"),
];

/// Orthographic replacements for English. From BA2's `support/replacements.eng`.
///
/// These normalize informal spellings to standard CHAT forms.
const REPLACEMENTS_ENG: &[(&str, &str)] = &[
    ("mm-hmm", "mhm"),
    ("mm-hum", "mhm"),
    ("'em", "(th)em"),
    ("cuz", "(be)cause"),
    ("'cause", "(be)cause"),
];

/// Compiled filled-pause lookup (language → word → replacement).
static FILLED_PAUSE_MAP: LazyLock<HashMap<&'static str, HashMap<&'static str, Replacement>>> =
    LazyLock::new(|| {
        let mut map = HashMap::new();
        let mut eng = HashMap::new();
        for &(orig, main) in FILLED_PAUSES_ENG {
            eng.insert(orig, Replacement { main_line: main });
        }
        map.insert("eng", eng);
        map
    });

/// Compiled replacement lookup (language → word → replacement).
static REPLACEMENT_MAP: LazyLock<HashMap<&'static str, HashMap<&'static str, Replacement>>> =
    LazyLock::new(|| {
        let mut map = HashMap::new();
        let mut eng = HashMap::new();
        for &(orig, main) in REPLACEMENTS_ENG {
            eng.insert(orig, Replacement { main_line: main });
        }
        map.insert("eng", eng);
        map
    });

// ---------------------------------------------------------------------------
// Disfluency replacement
// ---------------------------------------------------------------------------

/// Apply filled-pause marking and orthographic replacements to utterances.
///
/// Matches BA2's `DisfluencyReplacementEngine.process()` which runs
/// `_mark_utterance(ut, "filled_pauses", TokenType.FP, lang)` then
/// `_mark_utterance(ut, "replacements", TokenType.REGULAR, lang)`.
///
/// For filled pauses, the word text is replaced with the `&-` prefixed form
/// (e.g. "um" → "&-um"). For replacements, the word text is replaced with
/// the main-line form (e.g. "'cause" → "(be)cause").
pub fn apply_disfluency_replacements(utterances: &mut [Utterance], lang: &str) {
    let fp_map = FILLED_PAUSE_MAP.get(lang);
    let repl_map = REPLACEMENT_MAP.get(lang);

    // No wordlists for this language — nothing to do.
    if fp_map.is_none() && repl_map.is_none() {
        return;
    }

    for utt in utterances.iter_mut() {
        for word in utt.words.iter_mut() {
            let lower = word.text.to_lowercase();

            // Filled pauses first (higher priority, matches BA2 order).
            if let Some(fp) = fp_map.and_then(|m| m.get(lower.as_str())) {
                word.text = AsrNormalizedText::new(fp.main_line);
                continue;
            }

            // Then orthographic replacements.
            if let Some(repl) = repl_map.and_then(|m| m.get(lower.as_str())) {
                word.text = AsrNormalizedText::new(repl.main_line);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// N-gram retrace detection
// ---------------------------------------------------------------------------

/// Detect repeated n-grams within each utterance and mark them as retraces.
///
/// Matches BA2's `NgramRetraceEngine.process()`. The algorithm scans for
/// repeated n-grams of increasing length (1..len for most languages, 2..len
/// for Chinese/Cantonese). When a repeated n-gram is found, all occurrences
/// except the last are marked with `WordKind::Retrace` and have punctuation
/// stripped from their text.
///
/// Example: words `["I", "am", "I", "am", "going", "."]`
///   → words `["I"(Retrace), "am"(Retrace), "I", "am", "going", "."]`
///
/// The `build_chat` module reads `WordKind::Retrace` to construct proper
/// `<...> [/]` bracketed annotation groups in the CHAT AST, rather than
/// encoding notation into the word text (which would be string hacking).
pub fn apply_retrace_detection(utterances: &mut [Utterance], lang: &str) {
    let min_ngram = if lang == "yue" || lang == "zho" { 2 } else { 1 };

    for utt in utterances.iter_mut() {
        let content_len = content_word_count(&utt.words);
        if content_len < 2 {
            continue;
        }

        // Build index of content-word positions (skipping punct/terminators).
        let content_indices: Vec<usize> = utt
            .words
            .iter()
            .enumerate()
            .filter(|(_, w)| !is_punct_or_terminator(w.text.as_str()))
            .map(|(idx, _)| idx)
            .collect();

        // Track which content positions are retraces.
        let mut is_retrace = vec![false; content_indices.len()];

        // Scan for n-gram retraces (BA2 algorithm).
        for n in min_ngram..content_indices.len() {
            let mut begin = 0;
            while begin + n < content_indices.len() {
                let gram: Vec<AsrNormalizedText> = (0..n)
                    .map(|i| utt.words[content_indices[begin + i]].text.clone())
                    .collect();
                let mut root = begin;

                while root + 2 * n <= content_indices.len() {
                    let next_matches = (0..n).all(|i| {
                        utt.words[content_indices[root + n + i]].text == gram[i]
                    });
                    if next_matches {
                        for i in 0..n {
                            is_retrace[begin + i] = true;
                        }
                        root += n;
                    } else {
                        break;
                    }
                }
                begin += 1;
            }
        }

        // Apply: set kind=Retrace and strip punctuation on marked words.
        for (content_pos, &orig_idx) in content_indices.iter().enumerate() {
            if is_retrace[content_pos] {
                utt.words[orig_idx].kind = WordKind::Retrace;
                utt.words[orig_idx].text = AsrNormalizedText::new(strip_punct(utt.words[orig_idx].text.as_str()));
            }
        }
    }
}

/// Count non-punctuation words in an utterance.
fn content_word_count(words: &[AsrWord]) -> usize {
    words.iter().filter(|w| !is_punct_or_terminator(w.text.as_str())).count()
}

/// Check if a word is punctuation or a sentence terminator.
fn is_punct_or_terminator(text: &str) -> bool {
    ENDING_PUNCT.contains(&text) || MOR_PUNCT.contains(&text)
}

/// Strip CHAT punctuation from a word (for retrace content).
fn strip_punct(text: &str) -> String {
    let mut result = text.to_string();
    for &p in ENDING_PUNCT.iter().chain(MOR_PUNCT.iter()) {
        result = result.replace(p, "");
    }
    result.trim().to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_word(text: &str) -> AsrWord {
        AsrWord::new(text, Some(0), Some(100))
    }

    fn make_utt(speaker: usize, words: &[&str]) -> Utterance {
        Utterance {
            speaker: SpeakerIndex(speaker),
            words: words.iter().map(|w| make_word(w)).collect(),
            lang: None,
        }
    }

    // -- Disfluency tests --

    #[test]
    fn filled_pause_um_becomes_filler_marker() {
        let mut utts = vec![make_utt(0, &["I", "um", "went", "."])];
        apply_disfluency_replacements(&mut utts, "eng");
        assert_eq!(utts[0].words[1].text, "&-um");
    }

    #[test]
    fn filled_pause_uh_becomes_filler_marker() {
        let mut utts = vec![make_utt(0, &["uh", "hello", "."])];
        apply_disfluency_replacements(&mut utts, "eng");
        assert_eq!(utts[0].words[0].text, "&-uh");
    }

    #[test]
    fn replacement_cause_becomes_because() {
        let mut utts = vec![make_utt(0, &["I", "'cause", "you", "know", "."])];
        apply_disfluency_replacements(&mut utts, "eng");
        assert_eq!(utts[0].words[1].text, "(be)cause");
    }

    #[test]
    fn replacement_mmhmm_becomes_mhm() {
        let mut utts = vec![make_utt(0, &["mm-hmm", "."])];
        apply_disfluency_replacements(&mut utts, "eng");
        assert_eq!(utts[0].words[0].text, "mhm");
    }

    #[test]
    fn case_insensitive_matching() {
        let mut utts = vec![make_utt(0, &["Um", "UM", "."])];
        apply_disfluency_replacements(&mut utts, "eng");
        assert_eq!(utts[0].words[0].text, "&-um");
        assert_eq!(utts[0].words[1].text, "&-um");
    }

    #[test]
    fn no_wordlist_for_language_is_noop() {
        let mut utts = vec![make_utt(0, &["um", "hello", "."])];
        let original = utts[0].words[0].text.clone();
        apply_disfluency_replacements(&mut utts, "fra");
        assert_eq!(utts[0].words[0].text, original);
    }

    // -- Retrace tests --

    #[test]
    fn simple_word_retrace() {
        let mut utts = vec![make_utt(0, &["I", "I", "went", "."])];
        apply_retrace_detection(&mut utts, "eng");
        let texts: Vec<&str> = utts[0].words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec!["I", "I", "went", "."]);
        assert_eq!(utts[0].words[0].kind, WordKind::Retrace);
        assert_eq!(utts[0].words[1].kind, WordKind::Regular);
        assert_eq!(utts[0].words[2].kind, WordKind::Regular);
    }

    #[test]
    fn bigram_retrace() {
        let mut utts = vec![make_utt(0, &["I", "am", "I", "am", "going", "."])];
        apply_retrace_detection(&mut utts, "eng");
        let texts: Vec<&str> = utts[0].words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec!["I", "am", "I", "am", "going", "."]);
        assert_eq!(utts[0].words[0].kind, WordKind::Retrace);
        assert_eq!(utts[0].words[1].kind, WordKind::Retrace);
        assert_eq!(utts[0].words[2].kind, WordKind::Regular);
        assert_eq!(utts[0].words[3].kind, WordKind::Regular);
    }

    #[test]
    fn triple_retrace() {
        let mut utts = vec![make_utt(0, &["go", "go", "go", "home", "."])];
        apply_retrace_detection(&mut utts, "eng");
        let texts: Vec<&str> = utts[0].words.iter().map(|w| w.text.as_str()).collect();
        // BA2's algorithm: unigram "go" repeats, so first two marked as retrace.
        assert_eq!(texts, vec!["go", "go", "go", "home", "."]);
        assert_eq!(utts[0].words[0].kind, WordKind::Retrace);
        assert_eq!(utts[0].words[1].kind, WordKind::Retrace);
        assert_eq!(utts[0].words[2].kind, WordKind::Regular);
    }

    #[test]
    fn no_retrace_when_no_repeats() {
        let mut utts = vec![make_utt(0, &["I", "went", "home", "."])];
        apply_retrace_detection(&mut utts, "eng");
        let texts: Vec<&str> = utts[0].words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec!["I", "went", "home", "."]);
        assert!(utts[0].words.iter().all(|w| w.kind == WordKind::Regular));
    }

    #[test]
    fn chinese_skips_unigram_retraces() {
        // In Chinese/Cantonese, BA2 starts at n=2 to avoid single-char retraces.
        let mut utts = vec![make_utt(0, &["我", "我", "去", "."])];
        apply_retrace_detection(&mut utts, "yue");
        // No retrace because min_ngram=2 for Cantonese.
        assert!(utts[0].words.iter().all(|w| w.kind == WordKind::Regular));
    }

    #[test]
    fn chinese_detects_bigram_retrace() {
        let mut utts = vec![make_utt(0, &["我", "去", "我", "去", "了", "."])];
        apply_retrace_detection(&mut utts, "zho");
        assert_eq!(utts[0].words[0].kind, WordKind::Retrace);
        assert_eq!(utts[0].words[1].kind, WordKind::Retrace);
        assert_eq!(utts[0].words[2].kind, WordKind::Regular);
        assert_eq!(utts[0].words[3].kind, WordKind::Regular);
    }

    #[test]
    fn disfluency_and_retrace_compose() {
        // "um um I went" → "&-um &-um I went" (disfluency) →
        // first "&-um" marked Retrace, second "&-um" stays Regular.
        let mut utts = vec![make_utt(0, &["um", "um", "I", "went", "."])];
        apply_disfluency_replacements(&mut utts, "eng");
        apply_retrace_detection(&mut utts, "eng");
        let texts: Vec<&str> = utts[0].words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec!["&-um", "&-um", "I", "went", "."]);
        assert_eq!(utts[0].words[0].kind, WordKind::Retrace);
        assert_eq!(utts[0].words[1].kind, WordKind::Regular);
    }
}
