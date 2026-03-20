//! Transcript comparison: main vs gold-standard reference.
//!
//! Extracts words from both transcripts, normalizes them via
//! [`wer_conform::conform_words`], runs DP alignment, and produces
//! per-utterance comparison annotations with accuracy metrics.
//!
//! This is the Rust implementation of Python's `CompareEngine` +
//! `CompareAnalysisEngine` from batchalign2.

use talkbank_model::Span;
use talkbank_model::alignment::helpers::TierDomain;
use talkbank_model::model::{DependentTier, Line, NonEmptyString, UserDefinedDependentTier};

use crate::dp_align::{self, AlignResult, MatchMode};
use crate::extract::{self, ExtractedUtterance};
use crate::wer_conform;

/// Status of a compared token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareStatus {
    /// Word matches between main and gold.
    Match,
    /// Word present in main but not in gold (insertion).
    ExtraMain,
    /// Word present in gold but not in main (deletion).
    ExtraGold,
}

/// A single token in the comparison output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompareToken {
    /// The word text.
    pub text: String,
    /// Match status.
    pub status: CompareStatus,
}

/// Per-utterance comparison result.
#[derive(Debug, Clone)]
pub struct UtteranceComparison {
    /// Zero-based utterance index in the main file.
    pub utterance_index: usize,
    /// Speaker code.
    pub speaker: String,
    /// Comparison tokens (matches, insertions, deletions).
    pub tokens: Vec<CompareToken>,
}

/// Aggregate comparison metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct CompareMetrics {
    /// Word Error Rate: (insertions + deletions) / total_gold_words.
    pub wer: f64,
    /// 1.0 - wer (clamped to [0, 1]).
    pub accuracy: f64,
    /// Number of matching words.
    pub matches: usize,
    /// Words in main but not in gold.
    pub insertions: usize,
    /// Words in gold but not in main.
    pub deletions: usize,
    /// Total words in the gold transcript (matches + deletions).
    pub total_gold_words: usize,
    /// Total words in the main transcript (matches + insertions).
    pub total_main_words: usize,
}

/// Full comparison result.
#[derive(Debug, Clone)]
pub struct CompareResult {
    /// Per-utterance comparison annotations.
    pub utterances: Vec<UtteranceComparison>,
    /// Aggregate metrics.
    pub metrics: CompareMetrics,
}

/// Punctuation and fillers to exclude from comparison (matching BA2 behavior).
fn is_punct_or_filler(word: &str) -> bool {
    static PUNCT: &[&str] = &[
        ".", "?", "!", ",", "‡", "„", "+/.", "+//.", "+//?", "+...", "++.", "+\".", "+\"?",
    ];
    static FILLERS: &[&str] = &["um", "uhm", "em", "mhm", "uhhm", "eh", "uh", "hm"];

    let w = word.trim();
    PUNCT.contains(&w) || FILLERS.contains(&w.to_lowercase().as_str())
}

/// Apply conform_words per word, returning expanded tokens and an index
/// mapping back to the original word list.
///
/// `mapping[j]` = index into the original `words` list that `conformed[j]`
/// originated from.
fn conform_with_mapping(words: &[String]) -> (Vec<String>, Vec<usize>) {
    let mut conformed = Vec::new();
    let mut mapping = Vec::new();
    for (idx, word) in words.iter().enumerate() {
        let expanded = wer_conform::conform_words(std::slice::from_ref(word));
        for token in expanded {
            conformed.push(token);
            mapping.push(idx);
        }
    }
    (conformed, mapping)
}

/// Compare a main transcript against a gold-standard reference.
///
/// Both inputs are parsed CHAT files. Words are extracted from the Mor
/// domain (excluding punctuation and fillers), normalized via
/// `conform_words`, then aligned with the Hirschberg DP aligner.
///
/// Returns per-utterance comparison annotations and aggregate metrics.
pub fn compare(main_file: &crate::ChatFile, gold_file: &crate::ChatFile) -> CompareResult {
    // 1. Extract words from both files
    let main_utts = extract::extract_words(main_file, TierDomain::Mor);
    let gold_utts = extract::extract_words(gold_file, TierDomain::Mor);

    // 2. Flatten words, filtering punctuation and fillers
    let (main_words, main_info) = flatten_words(&main_utts);
    let (gold_words, _gold_info) = flatten_words(&gold_utts);

    // 3. Apply conform with index mapping
    let (conformed_main, main_map) = conform_with_mapping(&main_words);
    let (conformed_gold, gold_map) = conform_with_mapping(&gold_words);

    // 4. DP alignment (case-insensitive to match BA2's match_fn behavior)
    let alignment = dp_align::align(&conformed_main, &conformed_gold, MatchMode::CaseInsensitive);

    // 5. Redistribute alignment results per main utterance
    let num_main_utts = main_utts.len();
    let mut utt_tokens: Vec<Vec<(f64, CompareToken)>> = vec![Vec::new(); num_main_utts];

    let mut current_main_utt: usize = 0;
    let mut last_main_position: f64 = -1.0;
    let mut main_cursor: usize = 0;
    let mut gold_cursor: usize = 0;

    for item in &alignment {
        match item {
            AlignResult::Match { key, .. } => {
                let orig_main_idx = main_map[main_cursor];
                let (utt_idx, word_position) = main_info[orig_main_idx];
                current_main_utt = utt_idx;
                last_main_position = word_position as f64;

                utt_tokens[utt_idx].push((
                    word_position as f64,
                    CompareToken {
                        text: key.clone(),
                        status: CompareStatus::Match,
                    },
                ));
                main_cursor += 1;
                gold_cursor += 1;
            }
            AlignResult::ExtraPayload { key, .. } => {
                // Word in main but not in gold
                let orig_main_idx = main_map[main_cursor];
                let (utt_idx, word_position) = main_info[orig_main_idx];
                current_main_utt = utt_idx;
                last_main_position = word_position as f64;

                utt_tokens[utt_idx].push((
                    word_position as f64,
                    CompareToken {
                        text: key.clone(),
                        status: CompareStatus::ExtraMain,
                    },
                ));
                main_cursor += 1;
            }
            AlignResult::ExtraReference { key, .. } => {
                // Word in gold but not in main — position after last main word
                let pos = last_main_position + 0.5;
                let target_utt = if current_main_utt < num_main_utts {
                    current_main_utt
                } else {
                    num_main_utts.saturating_sub(1)
                };
                utt_tokens[target_utt].push((
                    pos,
                    CompareToken {
                        text: key.clone(),
                        status: CompareStatus::ExtraGold,
                    },
                ));
                gold_cursor += 1;
            }
        }
    }

    // Sort each utterance's tokens by position (stable sort preserves order
    // within the same position).
    for tokens in &mut utt_tokens {
        tokens.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    // 6. Build per-utterance comparison results
    let mut utterances = Vec::with_capacity(num_main_utts);
    for (utt_idx, utt) in main_utts.iter().enumerate() {
        let tokens: Vec<CompareToken> = utt_tokens[utt_idx]
            .iter()
            .map(|(_, tok)| tok.clone())
            .collect();
        utterances.push(UtteranceComparison {
            utterance_index: utt_idx,
            speaker: utt.speaker.as_str().to_string(),
            tokens,
        });
    }

    // 7. Compute metrics
    let mut matches = 0usize;
    let mut extra_main = 0usize;
    let mut extra_gold = 0usize;

    for utt in &utterances {
        for tok in &utt.tokens {
            match tok.status {
                CompareStatus::Match => matches += 1,
                CompareStatus::ExtraMain => extra_main += 1,
                CompareStatus::ExtraGold => extra_gold += 1,
            }
        }
    }

    let total_gold = matches + extra_gold;
    let total_main = matches + extra_main;
    let wer = if total_gold > 0 {
        (extra_main + extra_gold) as f64 / total_gold as f64
    } else {
        0.0
    };
    let accuracy = (1.0 - wer).clamp(0.0, 1.0);

    let metrics = CompareMetrics {
        wer,
        accuracy,
        matches,
        insertions: extra_main,
        deletions: extra_gold,
        total_gold_words: total_gold,
        total_main_words: total_main,
    };

    // Suppress unused variable warnings
    let _ = (gold_cursor, gold_map);

    CompareResult {
        utterances,
        metrics,
    }
}

/// Flatten extracted utterances into a word list and info vector.
///
/// Returns:
/// - `words`: cleaned text for each non-punct/non-filler word
/// - `info`: `(utterance_index, word_position)` for each word
fn flatten_words(utts: &[ExtractedUtterance]) -> (Vec<String>, Vec<(usize, usize)>) {
    let mut words = Vec::new();
    let mut info = Vec::new();

    for utt in utts {
        let mut word_pos = 0usize;
        for extracted in &utt.words {
            let text = extracted.text.as_str();
            if is_punct_or_filler(text) {
                word_pos += 1;
                continue;
            }
            words.push(text.to_string());
            info.push((utt.utterance_index.0, word_pos));
            word_pos += 1;
        }
    }

    (words, info)
}

/// Format comparison results as a %xsrep tier string.
///
/// Each token is annotated with its status:
/// - `word` for matches
/// - `[+ main]word` for insertions (in main but not gold)
/// - `[- gold]word` for deletions (in gold but not main)
pub fn format_xsrep(comparison: &UtteranceComparison) -> String {
    let mut parts = Vec::with_capacity(comparison.tokens.len());
    for tok in &comparison.tokens {
        match tok.status {
            CompareStatus::Match => parts.push(tok.text.clone()),
            CompareStatus::ExtraMain => parts.push(format!("[+ main]{}", tok.text)),
            CompareStatus::ExtraGold => parts.push(format!("[- gold]{}", tok.text)),
        }
    }
    parts.join(" ")
}

/// Format comparison metrics as CSV rows with header.
pub fn format_metrics_csv(metrics: &CompareMetrics) -> String {
    format!(
        "metric,value\nwer,{:.4}\naccuracy,{:.4}\nmatches,{}\ninsertions,{}\ndeletions,{}\ntotal_gold_words,{}\ntotal_main_words,{}",
        metrics.wer,
        metrics.accuracy,
        metrics.matches,
        metrics.insertions,
        metrics.deletions,
        metrics.total_gold_words,
        metrics.total_main_words,
    )
}

/// Inject comparison results into a CHAT file as `%xsrep` dependent tiers.
///
/// For each [`UtteranceComparison`], finds the corresponding utterance in the
/// file (by `utterance_index`) and adds a `%xsrep` user-defined tier containing
/// the formatted comparison annotations.
///
/// Uses `replace_or_add_tier` to ensure idempotent injection.
pub fn inject_comparison(chat_file: &mut crate::ChatFile, result: &CompareResult) {
    // Build a map from utterance_index to the formatted xsrep string
    let mut utt_line_indices: Vec<usize> = Vec::new();
    for (line_idx, line) in chat_file.lines.iter().enumerate() {
        if matches!(line, Line::Utterance(_)) {
            utt_line_indices.push(line_idx);
        }
    }

    for utt_comparison in &result.utterances {
        if utt_comparison.tokens.is_empty() {
            continue;
        }

        let utt_idx = utt_comparison.utterance_index;
        if utt_idx >= utt_line_indices.len() {
            tracing::warn!(
                utt_idx,
                num_utterances = utt_line_indices.len(),
                "Compare utterance_index out of range"
            );
            continue;
        }

        let line_idx = utt_line_indices[utt_idx];
        let xsrep_text = format_xsrep(utt_comparison);

        if let Some(Line::Utterance(utt)) = chat_file.lines.get_mut(line_idx) {
            let Some(label) = NonEmptyString::new("xsrep") else {
                continue;
            };
            let Some(content) = NonEmptyString::new(&xsrep_text) else {
                continue;
            };

            let new_tier = DependentTier::UserDefined(UserDefinedDependentTier {
                label,
                content,
                span: Span::DUMMY,
            });
            crate::inject::replace_or_add_tier(&mut utt.dependent_tiers, new_tier);
        }
    }
}

/// Remove existing `%xsrep` tiers from all utterances.
pub fn clear_comparison(chat_file: &mut crate::ChatFile) {
    for line in chat_file.lines.iter_mut() {
        if let Line::Utterance(utt) = line {
            utt.dependent_tiers.retain(|tier| {
                !matches!(
                    tier,
                    DependentTier::UserDefined(ud) if ud.label.as_str() == "xsrep"
                )
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_lenient;

    /// Build a minimal CHAT file with given utterance lines.
    fn make_chat(utterances: &[(&str, &str)]) -> String {
        let mut lines = vec![
            "@UTF8".to_string(),
            "@Begin".to_string(),
            "@Languages:\teng".to_string(),
            "@Participants:\tCHI Target_Child, MOT Mother".to_string(),
            "@ID:\teng|test|CHI|3;|female|||Target_Child|||".to_string(),
            "@ID:\teng|test|MOT||female|||Mother|||".to_string(),
        ];
        for (speaker, text) in utterances {
            lines.push(format!("*{speaker}:\t{text}"));
        }
        lines.push("@End".to_string());
        lines.join("\n")
    }

    #[test]
    fn identical_transcripts() {
        let chat = make_chat(&[("CHI", "hello world ."), ("MOT", "good morning .")]);
        let (main_file, _) = parse_lenient(&chat);
        let (gold_file, _) = parse_lenient(&chat);

        let result = compare(&main_file, &gold_file);
        assert_eq!(result.metrics.wer, 0.0);
        assert_eq!(result.metrics.accuracy, 1.0);
        assert_eq!(result.metrics.matches, 4);
        assert_eq!(result.metrics.insertions, 0);
        assert_eq!(result.metrics.deletions, 0);
        assert_eq!(result.metrics.total_gold_words, 4);
        assert_eq!(result.metrics.total_main_words, 4);
    }

    #[test]
    fn single_substitution() {
        let main = make_chat(&[("CHI", "hello earth .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (main_file, _) = parse_lenient(&main);
        let (gold_file, _) = parse_lenient(&gold);

        let result = compare(&main_file, &gold_file);
        // "earth" in main, "world" in gold => 1 insertion + 1 deletion
        assert!(result.metrics.wer > 0.0);
        assert_eq!(result.metrics.matches, 1); // "hello" matches
        assert_eq!(result.metrics.insertions, 1); // "earth"
        assert_eq!(result.metrics.deletions, 1); // "world"
    }

    #[test]
    fn extra_word_in_main() {
        let main = make_chat(&[("CHI", "hello big world .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (main_file, _) = parse_lenient(&main);
        let (gold_file, _) = parse_lenient(&gold);

        let result = compare(&main_file, &gold_file);
        assert_eq!(result.metrics.matches, 2); // "hello", "world"
        assert_eq!(result.metrics.insertions, 1); // "big"
        assert_eq!(result.metrics.deletions, 0);
        assert_eq!(result.metrics.total_gold_words, 2);
        assert_eq!(result.metrics.total_main_words, 3);
    }

    #[test]
    fn missing_word_in_main() {
        let main = make_chat(&[("CHI", "hello .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (main_file, _) = parse_lenient(&main);
        let (gold_file, _) = parse_lenient(&gold);

        let result = compare(&main_file, &gold_file);
        assert_eq!(result.metrics.matches, 1); // "hello"
        assert_eq!(result.metrics.insertions, 0);
        assert_eq!(result.metrics.deletions, 1); // "world"
        assert_eq!(result.metrics.total_gold_words, 2);
        assert_eq!(result.metrics.total_main_words, 1);
    }

    #[test]
    fn empty_main() {
        // Main has an utterance but no content words (just terminator)
        let main = make_chat(&[("CHI", ".")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (main_file, _) = parse_lenient(&main);
        let (gold_file, _) = parse_lenient(&gold);

        let result = compare(&main_file, &gold_file);
        assert_eq!(result.metrics.matches, 0);
        assert_eq!(result.metrics.deletions, 2);
        assert_eq!(result.metrics.wer, 1.0);
    }

    #[test]
    fn empty_gold() {
        let main = make_chat(&[("CHI", "hello world .")]);
        let gold = make_chat(&[("CHI", ".")]);
        let (main_file, _) = parse_lenient(&main);
        let (gold_file, _) = parse_lenient(&gold);

        let result = compare(&main_file, &gold_file);
        assert_eq!(result.metrics.matches, 0);
        assert_eq!(result.metrics.insertions, 2);
        assert_eq!(result.metrics.total_gold_words, 0);
        assert_eq!(result.metrics.wer, 0.0); // no gold words => wer=0
    }

    #[test]
    fn case_insensitive_matching() {
        let main = make_chat(&[("CHI", "Hello World .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (main_file, _) = parse_lenient(&main);
        let (gold_file, _) = parse_lenient(&gold);

        let result = compare(&main_file, &gold_file);
        assert_eq!(result.metrics.wer, 0.0);
        assert_eq!(result.metrics.matches, 2);
    }

    #[test]
    fn conform_normalizes_contractions() {
        // "he's" should be expanded to "he is" by conform_words
        let main = make_chat(&[("CHI", "he's going .")]);
        let gold = make_chat(&[("CHI", "he is going .")]);
        let (main_file, _) = parse_lenient(&main);
        let (gold_file, _) = parse_lenient(&gold);

        let result = compare(&main_file, &gold_file);
        // After conform: main = ["he", "is", "going"], gold = ["he", "is", "going"]
        assert_eq!(result.metrics.wer, 0.0);
    }

    #[test]
    fn multiple_utterances() {
        let main = make_chat(&[("CHI", "hello ."), ("MOT", "goodbye .")]);
        let gold = make_chat(&[("CHI", "hello ."), ("MOT", "goodbye .")]);
        let (main_file, _) = parse_lenient(&main);
        let (gold_file, _) = parse_lenient(&gold);

        let result = compare(&main_file, &gold_file);
        assert_eq!(result.metrics.wer, 0.0);
        assert_eq!(result.metrics.matches, 2);
        assert_eq!(result.utterances.len(), 2);
    }

    #[test]
    fn format_xsrep_output() {
        let utt = UtteranceComparison {
            utterance_index: 0,
            speaker: "CHI".to_string(),
            tokens: vec![
                CompareToken {
                    text: "hello".to_string(),
                    status: CompareStatus::Match,
                },
                CompareToken {
                    text: "big".to_string(),
                    status: CompareStatus::ExtraMain,
                },
                CompareToken {
                    text: "world".to_string(),
                    status: CompareStatus::Match,
                },
                CompareToken {
                    text: "today".to_string(),
                    status: CompareStatus::ExtraGold,
                },
            ],
        };
        let xsrep = format_xsrep(&utt);
        assert_eq!(xsrep, "hello [+ main]big world [- gold]today");
    }

    #[test]
    fn format_metrics_csv_output() {
        let metrics = CompareMetrics {
            wer: 0.25,
            accuracy: 0.75,
            matches: 3,
            insertions: 1,
            deletions: 0,
            total_gold_words: 3,
            total_main_words: 4,
        };
        let csv = format_metrics_csv(&metrics);
        assert!(csv.contains("wer,0.2500"));
        assert!(csv.contains("accuracy,0.7500"));
        assert!(csv.contains("matches,3"));
        assert!(csv.contains("insertions,1"));
        assert!(csv.contains("deletions,0"));
    }

    #[test]
    fn wer_computation_is_correct() {
        // 2 matches, 1 insertion, 1 deletion out of 3 gold words
        let metrics = CompareMetrics {
            wer: 0.0,      // will be computed
            accuracy: 0.0, // will be computed
            matches: 2,
            insertions: 1,
            deletions: 1,
            total_gold_words: 3, // matches + deletions
            total_main_words: 3, // matches + insertions
        };
        // WER = (ins + del) / total_gold = 2/3 ≈ 0.6667
        let expected_wer = 2.0 / 3.0;
        let _ = metrics;

        // Test via actual compare
        let main = make_chat(&[("CHI", "hello big world .")]);
        let gold = make_chat(&[("CHI", "hello world today .")]);
        let (main_file, _) = parse_lenient(&main);
        let (gold_file, _) = parse_lenient(&gold);

        let result = compare(&main_file, &gold_file);
        // main: hello, big, world
        // gold: hello, world, today
        // align: hello=match, big=extra_main, world=match, today=extra_gold
        assert_eq!(result.metrics.matches, 2);
        assert_eq!(result.metrics.insertions, 1);
        assert_eq!(result.metrics.deletions, 1);
        assert!((result.metrics.wer - expected_wer).abs() < 0.001);
    }

    #[test]
    fn is_punct_or_filler_works() {
        assert!(is_punct_or_filler("."));
        assert!(is_punct_or_filler("?"));
        assert!(is_punct_or_filler("!"));
        assert!(is_punct_or_filler("+/."));
        assert!(is_punct_or_filler("um"));
        assert!(is_punct_or_filler("uh"));
        assert!(!is_punct_or_filler("hello"));
        assert!(!is_punct_or_filler("world"));
    }

    #[test]
    fn conform_with_mapping_tracks_indices() {
        let words: Vec<String> = vec!["he's".to_string(), "going".to_string()];
        let (conformed, mapping) = conform_with_mapping(&words);
        // "he's" -> ["he", "is"], "going" -> ["going"]
        assert_eq!(conformed, vec!["he", "is", "going"]);
        assert_eq!(mapping, vec![0, 0, 1]);
    }

    #[test]
    fn inject_comparison_adds_xsrep_tiers() {
        let main = make_chat(&[("CHI", "hello big world .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (mut main_file, _) = parse_lenient(&main);
        let (gold_file, _) = parse_lenient(&gold);

        let result = compare(&main_file, &gold_file);
        inject_comparison(&mut main_file, &result);

        // Find the utterance and check it has an %xsrep tier
        let serialized = crate::serialize::to_chat_string(&main_file);
        assert!(
            serialized.contains("%xsrep:"),
            "Output should contain %xsrep tier"
        );
        assert!(
            serialized.contains("[+ main]big"),
            "Should mark 'big' as extra_main"
        );
    }

    #[test]
    fn clear_comparison_removes_xsrep() {
        let main = make_chat(&[("CHI", "hello world .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (mut main_file, _) = parse_lenient(&main);
        let (gold_file, _) = parse_lenient(&gold);

        let result = compare(&main_file, &gold_file);
        inject_comparison(&mut main_file, &result);

        // Verify xsrep was added
        let serialized = crate::serialize::to_chat_string(&main_file);
        assert!(serialized.contains("%xsrep:"));

        // Clear and verify removal
        clear_comparison(&mut main_file);
        let serialized = crate::serialize::to_chat_string(&main_file);
        assert!(!serialized.contains("%xsrep:"));
    }

    #[test]
    fn inject_comparison_idempotent() {
        let main = make_chat(&[("CHI", "hello big world .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (mut main_file, _) = parse_lenient(&main);
        let (gold_file, _) = parse_lenient(&gold);

        let result = compare(&main_file, &gold_file);
        inject_comparison(&mut main_file, &result);
        let first = crate::serialize::to_chat_string(&main_file);

        // Inject again — should produce the same output (replace, not duplicate)
        inject_comparison(&mut main_file, &result);
        let second = crate::serialize::to_chat_string(&main_file);
        assert_eq!(first, second);
    }

    #[test]
    fn format_metrics_csv_has_header() {
        let metrics = CompareMetrics {
            wer: 0.25,
            accuracy: 0.75,
            matches: 3,
            insertions: 1,
            deletions: 0,
            total_gold_words: 3,
            total_main_words: 4,
        };
        let csv = format_metrics_csv(&metrics);
        assert!(csv.starts_with("metric,value\n"));
        assert!(csv.contains("wer,0.2500"));
    }
}
