//! ASR post-processing: compound merging, number expansion, retokenization,
//! disfluency marking, and retrace detection.
//!
//! This module ports the Python ASR post-processing pipeline to Rust. After
//! the Python worker returns raw ASR tokens (via `batch_infer` with task
//! `"asr"`), the Rust server applies these transformations before utterance
//! segmentation and CHAT assembly.
//!
//! # Pipeline stages
//!
//! 1. **Compound merging** — merge adjacent words that form known compounds
//! 2. **Multi-word splitting** — split tokens containing spaces, interpolate timestamps
//! 3. **Number expansion** — convert digit strings to word form
//! 4. **Cantonese normalization** — simplified→HK traditional + domain replacements (lang=yue only)
//! 5. **Long turn splitting** — chunk monologues >300 words
//! 6. **Retokenization** — split into utterances by punctuation
//! 7. **Disfluency replacement** — mark filled pauses ("um" → "&-um") and orthographic
//!    replacements ("'cause" → "(be)cause") from per-language wordlists
//! 8. **N-gram retrace detection** — detect repeated n-grams and wrap in `<...> [/]`

mod asr_types;
pub mod cantonese;
mod cleanup;
mod compounds;
pub mod lang_detect;
mod num2chinese;
mod num2text;

pub use asr_types::{AsrNormalizedText, AsrRawText, AsrTimestampSecs, ChatWordText, SpeakerIndex};

use serde::{Deserialize, Serialize};

pub use compounds::merge_compounds;
pub use num2text::expand_number;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// What role a word plays in the CHAT output.
///
/// The `build_chat` module reads this to decide how to represent the word
/// in the AST. Regular words become `UtteranceContent::Word`; retrace words
/// get wrapped in `<...> [/]` bracketed groups; filled pauses are already
/// encoded in the text as `&-um` etc. and parse normally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WordKind {
    /// Normal content word (or filled pause already in `&-um` form).
    #[default]
    Regular,
    /// This word is part of a retrace group — a repeated n-gram that
    /// should be wrapped in `<...> [/]` annotation in the CHAT output.
    Retrace,
}

/// A single token from ASR output, with timing information.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct AsrWord {
    /// The word text (normalized through the ASR pipeline).
    pub text: AsrNormalizedText,
    /// Start time in milliseconds (None if unknown).
    pub start_ms: Option<i64>,
    /// End time in milliseconds (None if unknown).
    pub end_ms: Option<i64>,
    /// What kind of word this is (regular, retrace, etc.).
    #[serde(default)]
    pub kind: WordKind,
}

impl AsrWord {
    /// Create a regular (non-retrace) word with timing.
    pub fn new(text: impl Into<String>, start_ms: Option<i64>, end_ms: Option<i64>) -> Self {
        Self {
            text: AsrNormalizedText::new(text),
            start_ms,
            end_ms,
            kind: WordKind::default(),
        }
    }
}

/// A speaker-attributed utterance after retokenization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Utterance {
    /// Speaker index (0-based).
    pub speaker: SpeakerIndex,
    /// Words in the utterance (last word is a terminator like ".").
    pub words: Vec<AsrWord>,
    /// Detected language for this utterance (ISO 639-3), if different from
    /// the primary language. Used for `[- lang]` code-switching precodes in CHAT.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
}

/// Raw monologue from ASR output (before post-processing).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AsrMonologue {
    /// Speaker index (0-based).
    pub speaker: SpeakerIndex,
    /// Raw ASR elements.
    pub elements: Vec<AsrElement>,
}

/// What kind of raw ASR element this is.
///
/// Currently only `Text` and `Punctuation` are emitted by providers.
/// Defaults to `Text` when not specified (e.g. omitted from JSON).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AsrElementKind {
    /// A word token.
    #[default]
    Text,
    /// A punctuation token (period, question mark, etc.).
    Punctuation,
}

/// A single element from raw ASR output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AsrElement {
    /// Token text (raw from the ASR provider).
    pub value: AsrRawText,
    /// Start time in seconds.
    #[serde(default)]
    pub ts: AsrTimestampSecs,
    /// End time in seconds.
    #[serde(default)]
    pub end_ts: AsrTimestampSecs,
    /// Element kind: text or punctuation.
    #[serde(default)]
    pub kind: AsrElementKind,
}

/// Raw ASR output structure (matches Rev.AI format).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AsrOutput {
    /// Speaker monologues.
    pub monologues: Vec<AsrMonologue>,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// CHAT-legal sentence terminators.
const ENDING_PUNCT: &[&str] = &[
    ".", "?", "!", "+...", "+/.", "+//.", "+/?", "+!?", "+\"/.", "+\".", "+//?", "+..?", "+.",
    "...", "(.)",
];

/// CHAT morphological punctuation markers.
const MOR_PUNCT: &[&str] = &["‡", "„", ","];

/// RTL punctuation that needs ASCII normalization.
const RTL_PUNCT: &[(&str, &str)] = &[("؟", "?"), ("۔", "."), ("،", ","), ("؛", ";")];

/// Maximum words per turn before splitting.
const MAX_TURN_LEN: usize = 300;

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Run the full ASR post-processing pipeline on raw ASR output.
///
/// Applies compound merging, timing conversion, multi-word splitting,
/// number expansion, long turn splitting, punctuation-based retokenization,
/// disfluency replacement, and n-gram retrace detection. Returns
/// speaker-attributed utterances ready for CHAT assembly via `build_chat()`.
pub fn process_raw_asr(output: &AsrOutput, lang: &str) -> Vec<Utterance> {
    let mut all_utterances = Vec::new();

    for monologue in &output.monologues {
        let speaker = monologue.speaker;

        // Stage 1: compound merging
        let merged = merge_compounds(&monologue.elements);

        // Stage 2: extract words with ms timings, filter pauses
        let mut words = extract_timed_words(&merged);

        // Stage 3: split multi-word tokens with timestamp interpolation
        words = split_multiword_tokens(words, lang);

        // Stage 4: number expansion
        words = expand_numbers_in_words(words, lang);

        // Stage 4b: Cantonese normalization (simplified→HK traditional + domain replacements)
        if lang == "yue" {
            words = normalize_cantonese_words(words);
        }

        // Stage 5: long turn splitting
        let chunks = split_long_turns(words);

        // Stage 6: retokenize (split into utterances by punctuation)
        for chunk in chunks {
            let utts = retokenize(speaker, chunk);
            all_utterances.extend(utts);
        }
    }

    // Stage 7: disfluency replacement (filled pauses + orthographic normalizations).
    // Matches BA2's DisfluencyReplacementEngine which ran after ASR on all utterances.
    cleanup::apply_disfluency_replacements(&mut all_utterances, lang);

    // Stage 8: n-gram retrace detection.
    // Matches BA2's NgramRetraceEngine which ran after disfluency on all utterances.
    cleanup::apply_retrace_detection(&mut all_utterances, lang);

    all_utterances
}

/// Extract timed words from ASR elements, converting seconds to milliseconds.
///
/// Filters out pause markers (like `<pause>`) and blank values.
fn extract_timed_words(elements: &[AsrElement]) -> Vec<AsrWord> {
    let mut words = Vec::new();
    for elem in elements {
        let value = elem.value.as_str().trim();
        if value.is_empty() {
            continue;
        }
        // Filter pause markers like <pause>, <inaudible>, etc.
        if value.starts_with('<') && value.ends_with('>') {
            continue;
        }
        let (start_ms, end_ms) = normalized_timing_range(elem.ts.as_f64(), elem.end_ts.as_f64());
        words.push(AsrWord::new(value, start_ms, end_ms));
    }
    words
}

/// Split tokens containing spaces into multiple words with interpolated timestamps.
///
/// Also handles hyphen-prefixed words by joining them with the previous word.
fn split_multiword_tokens(words: Vec<AsrWord>, lang: &str) -> Vec<AsrWord> {
    let mut result: Vec<AsrWord> = Vec::new();

    for word in words {
        // Join hyphen-prefixed words with previous
        if word.text.starts_with('-') && !result.is_empty() {
            let prev = result.last_mut().unwrap();
            prev.text.push_str(word.text.as_str());
            prev.end_ms = word.end_ms;
            continue;
        }

        result.extend(split_chunk_word(word, lang));
    }

    result
}

fn normalized_timing_range(start_s: f64, end_s: f64) -> (Option<i64>, Option<i64>) {
    if !start_s.is_finite() || !end_s.is_finite() {
        return (None, None);
    }

    let start_ms = (start_s * 1000.0).round() as i64;
    let end_ms = (end_s * 1000.0).round() as i64;
    if end_ms <= start_ms {
        (None, None)
    } else {
        (Some(start_ms), Some(end_ms))
    }
}

fn split_chunk_word(word: AsrWord, lang: &str) -> Vec<AsrWord> {
    let mut parts: Vec<(String, bool)> = Vec::new();
    let mut current = String::new();

    let flush_current = |parts: &mut Vec<(String, bool)>, current: &mut String| {
        if !current.is_empty() {
            parts.push((std::mem::take(current), false));
        }
    };

    for ch in word.text.as_str().chars() {
        if ch.is_whitespace() {
            flush_current(&mut parts, &mut current);
            continue;
        }

        if let Some(separator) = normalized_split_separator(ch) {
            flush_current(&mut parts, &mut current);
            if let Some(text) = separator {
                parts.push((text.to_string(), true));
            }
            continue;
        }

        current.push(ch);
    }
    flush_current(&mut parts, &mut current);

    let mut expanded_parts: Vec<(String, bool)> = Vec::new();
    for (text, is_separator) in parts {
        if is_separator {
            expanded_parts.push((text, true));
            continue;
        }
        expanded_parts.extend(expand_language_part(text, lang));
    }
    let parts = expanded_parts;

    if parts.len() == 1 && !parts[0].1 && parts[0].0 == word.text.as_str() {
        return vec![word];
    }

    let total_text_chars: usize = parts
        .iter()
        .filter(|(_, is_separator)| !*is_separator)
        .map(|(text, _)| text.chars().count())
        .sum();

    let mut consumed_chars = 0usize;
    let total_span = match (word.start_ms, word.end_ms) {
        (Some(start), Some(end)) if end > start && total_text_chars > 0 => {
            Some((start, end - start))
        }
        _ => None,
    };

    parts
        .into_iter()
        .map(|(text, is_separator)| {
            if is_separator {
                return AsrWord::new(text, None, None);
            }

            let timings = total_span.map(|(start, span)| {
                let part_chars = text.chars().count();
                let part_start = start + (span * consumed_chars as i64 / total_text_chars as i64);
                consumed_chars += part_chars;
                let part_end = start + (span * consumed_chars as i64 / total_text_chars as i64);
                (Some(part_start), Some(part_end))
            });

            let (start_ms, end_ms) = timings.unwrap_or((None, None));
            AsrWord::new(text, start_ms, end_ms)
        })
        .collect()
}

fn expand_language_part(text: String, lang: &str) -> Vec<(String, bool)> {
    if lang != "yue" || !should_split_cantonese_chars(&text) {
        return vec![(text, false)];
    }

    let tokens = cantonese::cantonese_char_tokens(&text);
    if tokens.len() <= 1 {
        return vec![(text, false)];
    }
    tokens.into_iter().map(|token| (token, false)).collect()
}

fn should_split_cantonese_chars(text: &str) -> bool {
    let mut has_cjk = false;
    for ch in text.chars() {
        if ch.is_ascii_alphabetic() || ch.is_ascii_digit() {
            return false;
        }
        if is_cjk_ideograph(ch) {
            has_cjk = true;
        }
    }
    has_cjk
}

fn is_cjk_ideograph(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0x2F800..=0x2FA1F
    )
}

fn normalized_split_separator(ch: char) -> Option<Option<&'static str>> {
    match ch {
        '.' => Some(Some(".")),
        '?' | '？' | '؟' => Some(Some("?")),
        '!' | '！' => Some(Some("!")),
        ',' | '，' | '、' | '،' => Some(Some(",")),
        '¿' | '¡' => Some(None),
        '。' => Some(Some(".")),
        _ => None,
    }
}

/// Normalize Cantonese text in all words (simplified→HK traditional + domain replacements).
fn normalize_cantonese_words(words: Vec<AsrWord>) -> Vec<AsrWord> {
    words
        .into_iter()
        .map(|w| AsrWord {
            text: w.text.map(cantonese::normalize_cantonese),
            ..w
        })
        .collect()
}

/// Expand digit strings to word form in all words.
fn expand_numbers_in_words(words: Vec<AsrWord>, lang: &str) -> Vec<AsrWord> {
    words
        .into_iter()
        .map(|w| AsrWord {
            text: w.text.map(|t| expand_number(t, lang)),
            ..w
        })
        .collect()
}

/// Split a word list into chunks of at most [`MAX_TURN_LEN`].
fn split_long_turns(words: Vec<AsrWord>) -> Vec<Vec<AsrWord>> {
    if words.len() <= MAX_TURN_LEN {
        return vec![words];
    }
    words.chunks(MAX_TURN_LEN).map(|c| c.to_vec()).collect()
}

/// Check if a word is or ends with a sentence-ending punctuation mark.
fn is_ending_punct(word: &str) -> bool {
    if ENDING_PUNCT.contains(&word) {
        return true;
    }
    // Check RTL punctuation
    for (rtl, _) in RTL_PUNCT {
        if word == *rtl {
            return true;
        }
    }
    false
}

/// Check if a word ends with ending punctuation (last char).
fn ends_with_ending_punct(word: &str) -> bool {
    match word.chars().last() {
        Some(c) => {
            let mut buf = [0u8; 4];
            is_ending_punct(c.encode_utf8(&mut buf))
        }
        None => false,
    }
}

/// Normalize RTL punctuation to ASCII equivalent.
fn normalize_punct(word: &str) -> String {
    for (rtl, ascii) in RTL_PUNCT {
        if word == *rtl {
            return ascii.to_string();
        }
    }
    word.to_string()
}

/// Split a word list into utterances based on punctuation boundaries.
///
/// This is the simple punctuation-based retokenizer (no BERT model).
fn retokenize(speaker: SpeakerIndex, words: Vec<AsrWord>) -> Vec<Utterance> {
    let mut utterances = Vec::new();
    let mut buf: Vec<AsrWord> = Vec::new();

    for word in words {
        // Normalize Japanese period and remove inverted punctuation
        let word = AsrWord {
            text: word
                .text
                .map(|t| t.replace('。', ".").replace(['¿', '¡'], " ")),
            ..word
        };

        buf.push(word);

        let last_text = buf.last().unwrap().text.as_str();

        if is_ending_punct(last_text) {
            // Whole word is ending punct — flush utterance
            let punct = normalize_punct(last_text);
            buf.last_mut().unwrap().text = AsrNormalizedText::new(punct);
            utterances.push(Utterance {
                speaker,
                words: std::mem::take(&mut buf),
                lang: None,
            });
        } else if ends_with_ending_punct(last_text) {
            // Last character is ending punct — split the word
            let text = buf.pop().unwrap();
            let s = text.text.as_str();
            let last_char_boundary = s.char_indices().next_back().map(|(i, _)| i).unwrap_or(0);
            let word_part = &s[..last_char_boundary];
            let punct_part = &s[last_char_boundary..];

            if !word_part.is_empty() {
                buf.push(AsrWord::new(word_part, text.start_ms, text.end_ms));
            }
            buf.push(AsrWord::new(normalize_punct(punct_part), None, None));
            utterances.push(Utterance {
                speaker,
                words: std::mem::take(&mut buf),
                lang: None,
            });
        }
    }

    // Flush remaining words
    if !buf.is_empty() {
        // Remove trailing MOR_PUNCT
        while buf
            .last()
            .is_some_and(|w| MOR_PUNCT.contains(&w.text.as_str()))
        {
            buf.pop();
        }
        if !buf.is_empty() {
            // Append terminator
            buf.push(AsrWord::new(".", None, None));
            utterances.push(Utterance {
                speaker,
                words: buf,
                lang: None,
            });
        }
    }

    utterances
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn elem(value: &str, ts: f64, end_ts: f64) -> AsrElement {
        AsrElement {
            value: AsrRawText::new(value),
            ts: AsrTimestampSecs(ts),
            end_ts: AsrTimestampSecs(end_ts),
            kind: AsrElementKind::Text,
        }
    }

    #[test]
    fn test_extract_timed_words_filters_pauses() {
        let elems = vec![
            elem("hello", 0.0, 0.5),
            elem("<pause>", 0.5, 1.0),
            elem("world", 1.0, 1.5),
        ];
        let words = extract_timed_words(&elems);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "world");
    }

    #[test]
    fn test_extract_timed_words_converts_to_ms() {
        let elems = vec![elem("hello", 1.234, 2.567)];
        let words = extract_timed_words(&elems);
        assert_eq!(words[0].start_ms, Some(1234));
        assert_eq!(words[0].end_ms, Some(2567));
    }

    #[test]
    fn test_extract_timed_words_treats_zero_duration_as_untimed() {
        let elems = vec![elem("hello", 0.0, 0.0)];
        let words = extract_timed_words(&elems);
        assert_eq!(words[0].start_ms, None);
        assert_eq!(words[0].end_ms, None);
    }

    #[test]
    fn test_split_multiword_tokens() {
        let words = vec![AsrWord::new("hello world", Some(0), Some(1000))];
        let result = split_multiword_tokens(words, "eng");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "hello");
        assert_eq!(result[0].start_ms, Some(0));
        assert_eq!(result[0].end_ms, Some(500));
        assert_eq!(result[1].text, "world");
        assert_eq!(result[1].start_ms, Some(500));
        assert_eq!(result[1].end_ms, Some(1000));
    }

    #[test]
    fn test_split_multiword_tokens_splits_embedded_sentence_punctuation() {
        let words = vec![AsrWord::new("hello?world!", None, None)];
        let result = split_multiword_tokens(words, "eng");
        let texts: Vec<&str> = result.iter().map(|word| word.text.as_str()).collect();
        assert_eq!(texts, vec!["hello", "?", "world", "!"]);
        assert!(
            result
                .iter()
                .all(|word| word.start_ms.is_none() && word.end_ms.is_none())
        );
    }

    #[test]
    fn test_hyphen_joining() {
        let words = vec![
            AsrWord::new("hello", Some(0), Some(500)),
            AsrWord::new("-world", Some(500), Some(1000)),
        ];
        let result = split_multiword_tokens(words, "eng");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "hello-world");
        assert_eq!(result[0].start_ms, Some(0));
        assert_eq!(result[0].end_ms, Some(1000));
    }

    #[test]
    fn test_split_long_turns() {
        let words: Vec<AsrWord> = (0..650)
            .map(|i| AsrWord::new(format!("word{i}"), Some(i as i64), Some(i as i64 + 1)))
            .collect();
        let chunks = split_long_turns(words);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 300);
        assert_eq!(chunks[1].len(), 300);
        assert_eq!(chunks[2].len(), 50);
    }

    #[test]
    fn test_retokenize_simple() {
        let words = vec![
            AsrWord::new("hello", Some(0), Some(500)),
            AsrWord::new("world", Some(500), Some(1000)),
            AsrWord::new(".", None, None),
        ];
        let utts = retokenize(SpeakerIndex(0), words);
        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].speaker, SpeakerIndex(0));
        assert_eq!(utts[0].words.len(), 3);
        assert_eq!(utts[0].words[2].text, ".");
    }

    #[test]
    fn test_retokenize_splits_on_period() {
        let words = vec![
            AsrWord::new("hello", Some(0), Some(500)),
            AsrWord::new(".", Some(500), Some(600)),
            AsrWord::new("world", Some(600), Some(1000)),
        ];
        let utts = retokenize(SpeakerIndex(0), words);
        assert_eq!(utts.len(), 2);
        assert_eq!(utts[0].words.len(), 2); // hello .
        assert_eq!(utts[0].words[0].text, "hello");
        assert_eq!(utts[0].words[1].text, ".");
        assert_eq!(utts[1].words.len(), 2); // world .
        assert_eq!(utts[1].words[0].text, "world");
        assert_eq!(utts[1].words[1].text, "."); // auto-appended
    }

    #[test]
    fn test_retokenize_trailing_no_terminator() {
        let words = vec![
            AsrWord::new("hello", Some(0), Some(500)),
            AsrWord::new("world", Some(500), Some(1000)),
        ];
        let utts = retokenize(SpeakerIndex(0), words);
        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].words.last().unwrap().text, "."); // auto-appended
    }

    #[test]
    fn test_retokenize_rtl_punct() {
        let words = vec![
            AsrWord::new("hello", Some(0), Some(500)),
            AsrWord::new("؟", None, None),
        ];
        let utts = retokenize(SpeakerIndex(0), words);
        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].words[1].text, "?"); // normalized
    }

    /// Golden test: matches Python `_process_raw_asr` output for simple input.
    #[test]
    fn test_process_raw_asr_golden_simple() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    elem("hello", 0.0, 0.5),
                    elem("world", 0.5, 1.0),
                    AsrElement {
                        value: AsrRawText::new("."),
                        ts: AsrTimestampSecs(1.0),
                        end_ts: AsrTimestampSecs(1.1),
                        kind: AsrElementKind::Punctuation,
                    },
                    elem("how", 1.5, 2.0),
                    elem("are", 2.0, 2.3),
                    elem("you", 2.3, 2.5),
                ],
            }],
        };
        let utts = process_raw_asr(&output, "eng");
        assert_eq!(utts.len(), 2);

        // First utterance: hello world .
        assert_eq!(utts[0].words[0].text, "hello");
        assert_eq!(utts[0].words[0].start_ms, Some(0));
        assert_eq!(utts[0].words[1].text, "world");
        assert_eq!(utts[0].words[1].start_ms, Some(500));
        assert_eq!(utts[0].words[2].text, ".");

        // Second utterance: how are you .
        assert_eq!(utts[1].words[0].text, "how");
        assert_eq!(utts[1].words[0].start_ms, Some(1500));
        assert_eq!(utts[1].words.last().unwrap().text, ".");
    }

    /// Golden test: compound merging in pipeline.
    #[test]
    fn test_process_raw_asr_golden_compound() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    elem("the", 0.0, 0.3),
                    elem("air", 0.3, 0.6),
                    elem("plane", 0.6, 0.9),
                    AsrElement {
                        value: AsrRawText::new("."),
                        ts: AsrTimestampSecs(0.9),
                        end_ts: AsrTimestampSecs(1.0),
                        kind: AsrElementKind::Punctuation,
                    },
                ],
            }],
        };
        let utts = process_raw_asr(&output, "eng");
        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].words[0].text, "the");
        assert_eq!(utts[0].words[1].text, "airplane");
    }

    /// Golden test: number expansion in pipeline.
    #[test]
    fn test_process_raw_asr_golden_number() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    elem("I", 0.0, 0.3),
                    elem("have", 0.3, 0.6),
                    elem("5", 0.6, 0.9),
                    elem("cats", 0.9, 1.2),
                    AsrElement {
                        value: AsrRawText::new("."),
                        ts: AsrTimestampSecs(1.2),
                        end_ts: AsrTimestampSecs(1.3),
                        kind: AsrElementKind::Punctuation,
                    },
                ],
            }],
        };
        let utts = process_raw_asr(&output, "eng");
        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].words[2].text, "five");
    }

    /// Golden test: Cantonese normalization in pipeline.
    #[test]
    fn test_process_raw_asr_golden_cantonese() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    elem("你", 0.0, 0.3),
                    elem("真系", 0.3, 0.6),
                    elem("好", 0.6, 0.9),
                    elem("吵", 0.9, 1.2),
                    elem("呀", 1.2, 1.5),
                ],
            }],
        };
        let utts = process_raw_asr(&output, "yue");
        assert_eq!(utts.len(), 1);
        let tokens: Vec<&str> = utts[0]
            .words
            .iter()
            .map(|word| word.text.as_str())
            .collect();
        assert_eq!(tokens, vec!["你", "真", "係", "好", "嘈", "啊", "."]);
    }

    /// Cantonese normalization should NOT activate for non-yue languages.
    #[test]
    fn test_process_raw_asr_no_cantonese_for_eng() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![elem("系", 0.0, 0.5)],
            }],
        };
        let utts = process_raw_asr(&output, "eng");
        assert_eq!(utts[0].words[0].text, "系"); // NOT normalized
    }

    #[test]
    fn test_process_raw_asr_handles_single_chunk_cantonese_whisper_output() {
        let parser = talkbank_parser::TreeSitterParser::new().unwrap();
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![elem(
                    "這麼搞笑?我還清了啊!我還覺得奇怪為什麼在一個三次頭的電話打工呢?",
                    0.0,
                    0.0,
                )],
            }],
        };

        let utts = process_raw_asr(&output, "yue");
        assert_eq!(utts.len(), 3);
        assert_eq!(utts[0].words.last().unwrap().text, "?");
        assert_eq!(utts[1].words.last().unwrap().text, "!");
        assert_eq!(utts[2].words.last().unwrap().text, "?");
        assert_eq!(
            utts[0]
                .words
                .iter()
                .map(|word| word.text.as_str())
                .collect::<Vec<_>>(),
            vec!["這", "麼", "搞", "笑", "?"]
        );
        assert!(
            utts.iter()
                .flat_map(|utt| utt.words.iter())
                .filter(|word| !matches!(word.text.as_str(), "." | "!" | "?"))
                .count()
                > 10
        );
        assert!(
            utts.iter()
                .flat_map(|utt| utt.words.iter())
                .all(|word| !(word.start_ms == Some(0) && word.end_ms == Some(0)))
        );

        let desc = crate::build_chat::transcript_from_asr_utterances(
            &utts,
            &["PAR".to_string()],
            &["yue".to_string()],
            Some("05b_clip"),
            true,
        );
        let chat = crate::build_chat::build_chat(&desc).expect("build chat");
        let serialized = crate::serialize::to_chat_string(&chat);
        let (_parsed, errors) = crate::parse::parse_lenient(&parser, &serialized);
        assert!(
            errors.is_empty(),
            "generated CHAT should reparse cleanly: {errors:?}"
        );
    }

    #[test]
    fn test_process_raw_asr_keeps_ascii_words_intact_for_yue() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![elem("hello", 0.0, 0.5)],
            }],
        };

        let utts = process_raw_asr(&output, "yue");
        assert_eq!(utts.len(), 1);
        assert_eq!(
            utts[0]
                .words
                .iter()
                .map(|word| word.text.as_str())
                .collect::<Vec<_>>(),
            vec!["hello", "."]
        );
    }
}
