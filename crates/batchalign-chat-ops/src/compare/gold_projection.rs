//! Gold-projected comparison: per-gold-utterance windowed alignment.
//!
//! Replicates the BA2 `CompareEngine` algorithm:
//! 1. Extract and conform words from both main and gold transcripts.
//! 2. For each gold utterance, find the best matching window in the
//!    remaining main tokens using bag-of-words overlap scoring.
//! 3. Run DP alignment within that window.
//! 4. Record per-word match mappings for timing/morphology projection.
//!
//! The projection step copies inline timing bullets and `%mor`/`%gra`
//! dependent tiers from matched main words/utterances to gold, then
//! injects `%xsrep` comparison annotations on gold utterances.

use std::collections::HashMap;

use talkbank_model::Span;
use talkbank_model::alignment::helpers::TierDomain;
use talkbank_model::model::{
    Bullet, DependentTier, Line, NonEmptyString, UserDefinedDependentTier,
};

use crate::dp_align::{self, AlignResult, MatchMode};
use crate::extract::{self, ExtractedUtterance};
use crate::inject::replace_or_add_tier;

use super::{
    CompareMetrics, CompareStatus, CompareToken, UtteranceComparison, compute_metrics,
    conform_with_mapping, format_xsrep, is_punct_or_filler,
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A matched word pair linking a main form to a gold form.
#[derive(Debug, Clone)]
pub struct WordMatch {
    /// Index of the main utterance.
    pub main_utt_idx: usize,
    /// Word position within the main utterance (Mor-domain, excluding punct/fillers).
    pub main_word_pos: usize,
    /// Index of the gold utterance.
    pub gold_utt_idx: usize,
    /// Word position within the gold utterance (Mor-domain, excluding punct/fillers).
    pub gold_word_pos: usize,
}

/// Per-gold-utterance alignment result.
#[derive(Debug, Clone)]
pub struct GoldUtteranceAlignment {
    /// Gold utterance index (zero-based).
    pub gold_utt_idx: usize,
    /// Speaker code.
    pub speaker: String,
    /// Comparison tokens for this gold utterance.
    pub tokens: Vec<CompareToken>,
    /// Word-level match mappings for projection (only for `Match` tokens).
    pub word_matches: Vec<WordMatch>,
}

/// Full gold-projection comparison bundle.
///
/// Contains per-gold-utterance alignments, word-level match mappings, and
/// aggregate metrics with per-POS breakdown.
#[derive(Debug, Clone)]
pub struct GoldProjectionBundle {
    /// Per-gold-utterance alignment results.
    pub utterances: Vec<GoldUtteranceAlignment>,
    /// Aggregate metrics (with per-POS breakdown).
    pub metrics: CompareMetrics,
}

// ---------------------------------------------------------------------------
// Windowed alignment: bag-of-words segment search
// ---------------------------------------------------------------------------

/// Find the best contiguous window in `main_tokens` that matches `gold_tokens`
/// by multiset overlap.
///
/// Considers windows of length `gold_len ± 2`. Among equally scoring windows,
/// prefers closer length to gold, then the latest position. Returns `(start, end)`.
///
/// This is a direct port of BA2's `_find_best_segment`.
fn find_best_segment(gold_tokens: &[String], main_tokens: &[String]) -> (usize, usize) {
    if gold_tokens.is_empty() || main_tokens.is_empty() {
        return (0, 0);
    }

    let gold_len = gold_tokens.len();
    let main_len = main_tokens.len();

    // Build gold token counts
    let mut gold_counts: HashMap<&str, usize> = HashMap::new();
    for t in gold_tokens {
        *gold_counts.entry(t.as_str()).or_default() += 1;
    }

    let min_window = gold_len.saturating_sub(2).max(1);
    let max_window = main_len.min(gold_len + 2);

    let mut best = (0usize, main_len.min(gold_len));
    let mut best_score: f64 = -1.0;
    let mut best_len_delta: Option<usize> = None;

    for span in min_window..=max_window {
        // Initialize window counts for first window position
        let mut window_counts: HashMap<&str, usize> = HashMap::new();
        for t in &main_tokens[..span] {
            *window_counts.entry(t.as_str()).or_default() += 1;
        }

        let mut overlap: usize = window_counts
            .iter()
            .map(|(&k, &v)| v.min(*gold_counts.get(k).unwrap_or(&0)))
            .sum();

        for start in 0..=(main_len - span) {
            if start > 0 {
                // Slide window: remove left token, add right token
                let left = main_tokens[start - 1].as_str();
                let right = main_tokens[start + span - 1].as_str();

                // Remove left from overlap
                let wc_left = *window_counts.get(left).unwrap_or(&0);
                let gc_left = *gold_counts.get(left).unwrap_or(&0);
                overlap -= wc_left.min(gc_left);
                *window_counts.entry(left).or_default() -= 1;
                let wc_left_new = *window_counts.get(left).unwrap_or(&0);
                overlap += wc_left_new.min(gc_left);

                // Add right to overlap
                let wc_right = *window_counts.get(right).unwrap_or(&0);
                let gc_right = *gold_counts.get(right).unwrap_or(&0);
                overlap -= wc_right.min(gc_right);
                *window_counts.entry(right).or_default() += 1;
                let wc_right_new = *window_counts.get(right).unwrap_or(&0);
                overlap += wc_right_new.min(gc_right);
            }

            let score = overlap as f64 / gold_len as f64;
            let len_delta = span.abs_diff(gold_len);
            let end = start + span;

            if score > best_score {
                best = (start, end);
                best_score = score;
                best_len_delta = Some(len_delta);
            } else if (score - best_score).abs() < f64::EPSILON {
                match best_len_delta {
                    None => {
                        best = (start, end);
                        best_len_delta = Some(len_delta);
                    }
                    Some(prev_delta) if len_delta < prev_delta => {
                        best = (start, end);
                        best_len_delta = Some(len_delta);
                    }
                    Some(prev_delta) if len_delta == prev_delta && end > best.1 => {
                        best = (start, end);
                    }
                    _ => {}
                }
            }
        }
    }

    best
}

// ---------------------------------------------------------------------------
// Word info extraction (includes utterance/word indices)
// ---------------------------------------------------------------------------

/// Info tuple for a flattened word: `(utterance_index, word_position_in_utterance)`.
type WordInfo = (usize, usize);

/// Flatten extracted utterances into conformed tokens with index mappings.
///
/// Returns:
/// - `words`: original word texts (excluding punct/fillers)
/// - `info`: `(utterance_index, word_position)` per word
/// - `conformed`: conformed (normalized) token list
/// - `conform_map`: `conform_map[j]` → index into `words` that `conformed[j]` came from
fn flatten_and_conform(
    utts: &[ExtractedUtterance],
) -> (Vec<String>, Vec<WordInfo>, Vec<String>, Vec<usize>) {
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

    let (conformed, conform_map) = conform_with_mapping(&words);
    (words, info, conformed, conform_map)
}

// ---------------------------------------------------------------------------
// POS extraction from %mor tier
// ---------------------------------------------------------------------------

/// Extract the POS tag for a specific word position within an utterance.
///
/// Walks the `%mor` tier items. The items align 1-to-1 with Mor-domain words
/// (including tag-marker separators), so `mor_domain_word_index` is the
/// zero-based index into the flattened Mor-domain word list for this utterance.
///
/// Returns the uppercased POS string, or `"?"` if unavailable.
fn extract_pos(chat_file: &crate::ChatFile, utt_idx: usize, mor_domain_word_index: usize) -> String {
    let mut utterance_cursor = 0usize;
    for line in &chat_file.lines {
        if let Line::Utterance(utt) = line {
            if utterance_cursor == utt_idx {
                for tier in &utt.dependent_tiers {
                    if let DependentTier::Mor(mor_tier) = tier {
                        if let Some(mor_item) = mor_tier.items.get(mor_domain_word_index) {
                            return mor_item.main.pos.as_ref().to_uppercase();
                        }
                    }
                }
                return "?".to_string();
            }
            utterance_cursor += 1;
        }
    }
    "?".to_string()
}

// ---------------------------------------------------------------------------
// Gold-projected comparison
// ---------------------------------------------------------------------------

/// Compare a main transcript against a gold reference using per-gold-utterance
/// windowed alignment.
///
/// This replicates the BA2 `CompareEngine` algorithm:
/// 1. Extract and conform all words from both transcripts.
/// 2. Partition conformed gold tokens by utterance.
/// 3. For each gold utterance (in order), find the best bag-of-words window
///    in the remaining main tokens, then DP-align within that window.
/// 4. Record per-word match mappings and compute per-POS metrics.
///
/// Both inputs must be parsed CHAT files. The main file should already have
/// `%mor`/`%gra` tiers (from a prior morphosyntax pass) so that POS tags
/// can be extracted for per-POS metrics.
pub fn compare_gold_projection(
    main_file: &crate::ChatFile,
    gold_file: &crate::ChatFile,
) -> GoldProjectionBundle {
    // 1. Extract words from both files
    let main_utts = extract::extract_words(main_file, TierDomain::Mor);
    let gold_utts = extract::extract_words(gold_file, TierDomain::Mor);

    // 2. Flatten and conform
    let (_main_words, main_info, conformed_main, main_map) = flatten_and_conform(&main_utts);
    let (_gold_words, gold_info, conformed_gold, gold_map) = flatten_and_conform(&gold_utts);

    // 3. Partition conformed gold tokens by utterance
    let num_gold_utts = gold_utts.len();
    let mut gold_utt_tokens: Vec<Vec<String>> = vec![Vec::new(); num_gold_utts];
    let mut gold_utt_maps: Vec<Vec<usize>> = vec![Vec::new(); num_gold_utts];

    for (j, token) in conformed_gold.iter().enumerate() {
        let orig_idx = gold_map[j];
        let (utt_idx, _) = gold_info[orig_idx];
        gold_utt_tokens[utt_idx].push(token.clone());
        gold_utt_maps[utt_idx].push(orig_idx);
    }

    // 4. Per-gold-utterance windowed alignment
    let mut alignments: Vec<GoldUtteranceAlignment> = Vec::with_capacity(num_gold_utts);
    let mut search_start: usize = 0;

    for gold_utt_idx in 0..num_gold_utts {
        let g_tokens = &gold_utt_tokens[gold_utt_idx];
        let g_maps = &gold_utt_maps[gold_utt_idx];

        let speaker = gold_utts[gold_utt_idx].speaker.as_str().to_string();

        if g_tokens.is_empty() {
            alignments.push(GoldUtteranceAlignment {
                gold_utt_idx,
                speaker,
                tokens: Vec::new(),
                word_matches: Vec::new(),
            });
            continue;
        }

        // Find best window in remaining main tokens
        let remaining_main = &conformed_main[search_start..];
        let (win_start, win_end) = find_best_segment(g_tokens, remaining_main);

        let abs_start = search_start + win_start;
        let abs_end = search_start + win_end;

        // DP-align within the window
        let window_main = &conformed_main[abs_start..abs_end];
        let alignment = dp_align::align(window_main, g_tokens, MatchMode::CaseInsensitive);

        let mut tokens: Vec<CompareToken> = Vec::new();
        let mut word_matches: Vec<WordMatch> = Vec::new();
        let mut local_main_cursor: usize = 0;
        let mut local_gold_cursor: usize = 0;

        for item in &alignment {
            match item {
                AlignResult::Match { key, .. } => {
                    let global_main_idx = abs_start + local_main_cursor;
                    let orig_main_idx = main_map[global_main_idx];
                    let (main_utt_idx, main_word_pos) = main_info[orig_main_idx];

                    let orig_gold_idx = g_maps[local_gold_cursor];
                    let (_, gold_word_pos) = gold_info[orig_gold_idx];

                    // POS from the main file's %mor tier
                    let pos = extract_pos(main_file, main_utt_idx, main_word_pos);

                    tokens.push(CompareToken {
                        text: key.clone(),
                        pos,
                        status: CompareStatus::Match,
                    });

                    word_matches.push(WordMatch {
                        main_utt_idx,
                        main_word_pos,
                        gold_utt_idx,
                        gold_word_pos,
                    });

                    local_main_cursor += 1;
                    local_gold_cursor += 1;
                }
                AlignResult::ExtraReference { key, .. } => {
                    // In gold but not main (deletion)
                    let orig_gold_idx = g_maps[local_gold_cursor];
                    let (_, gold_word_pos) = gold_info[orig_gold_idx];

                    let pos = extract_pos(gold_file, gold_utt_idx, gold_word_pos);

                    tokens.push(CompareToken {
                        text: key.clone(),
                        pos,
                        status: CompareStatus::ExtraGold,
                    });
                    local_gold_cursor += 1;
                }
                AlignResult::ExtraPayload { key, .. } => {
                    // In main but not gold (insertion)
                    let global_main_idx = abs_start + local_main_cursor;
                    let orig_main_idx = main_map[global_main_idx];
                    let (main_utt_idx, main_word_pos) = main_info[orig_main_idx];

                    let pos = extract_pos(main_file, main_utt_idx, main_word_pos);

                    tokens.push(CompareToken {
                        text: key.clone(),
                        pos,
                        status: CompareStatus::ExtraMain,
                    });
                    local_main_cursor += 1;
                }
            }
        }

        alignments.push(GoldUtteranceAlignment {
            gold_utt_idx,
            speaker,
            tokens,
            word_matches,
        });

        search_start = abs_end;
    }

    // 5. Compute metrics with per-POS breakdown
    let utt_comparisons: Vec<UtteranceComparison> = alignments
        .iter()
        .map(|a| UtteranceComparison {
            utterance_index: a.gold_utt_idx,
            speaker: a.speaker.clone(),
            tokens: a.tokens.clone(),
        })
        .collect();
    let metrics = compute_metrics(&utt_comparisons);

    GoldProjectionBundle {
        utterances: alignments,
        metrics,
    }
}

// ---------------------------------------------------------------------------
// Timing projection: copy inline bullets from main to gold
// ---------------------------------------------------------------------------

/// Project word-level timing (inline bullets) from main to gold for matched words.
///
/// For each [`WordMatch`], copies the `inline_bullet` from the corresponding
/// main word to the gold word. Also sets utterance-level timing on gold
/// utterances from the earliest/latest timed words.
pub fn project_timing_to_gold(
    main_file: &crate::ChatFile,
    gold_file: &mut crate::ChatFile,
    bundle: &GoldProjectionBundle,
) {
    use talkbank_model::alignment::helpers::walk_words;
    use talkbank_model::alignment::helpers::walk_words_mut;
    use talkbank_model::alignment::helpers::WordItem;
    use talkbank_model::alignment::helpers::WordItemMut;

    // 1. Collect timing from main words: (utt_idx, word_pos) → Option<Bullet>
    let mut main_timings: HashMap<(usize, usize), Option<Bullet>> = HashMap::new();
    let mut main_utt_cursor = 0usize;
    for line in &main_file.lines {
        if let Line::Utterance(utt) = line {
            let mut word_pos = 0usize;
            walk_words(
                &utt.main.content.content,
                Some(TierDomain::Mor),
                &mut |leaf| match leaf {
                    WordItem::Word(word) => {
                        main_timings
                            .insert((main_utt_cursor, word_pos), word.inline_bullet.clone());
                        word_pos += 1;
                    }
                    WordItem::ReplacedWord(replaced) => {
                        // Use the first replacement word's timing if available
                        let bullet = if !replaced.replacement.words.is_empty() {
                            replaced.replacement.words[0].inline_bullet.clone()
                        } else {
                            replaced.word.inline_bullet.clone()
                        };
                        main_timings.insert((main_utt_cursor, word_pos), bullet);
                        word_pos += 1;
                    }
                    WordItem::Separator(_) => {
                        word_pos += 1;
                    }
                },
            );
            main_utt_cursor += 1;
        }
    }

    // 2. Build match lookup: (gold_utt_idx, gold_word_pos) → (main_utt_idx, main_word_pos)
    let mut match_map: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    for utt_align in &bundle.utterances {
        for wm in &utt_align.word_matches {
            match_map.insert(
                (wm.gold_utt_idx, wm.gold_word_pos),
                (wm.main_utt_idx, wm.main_word_pos),
            );
        }
    }

    // 3. Walk gold utterances and project timing
    let mut gold_utt_cursor = 0usize;
    for line in gold_file.lines.iter_mut() {
        if let Line::Utterance(utt) = line {
            let mut word_pos = 0usize;
            let mut first_timing: Option<u64> = None;
            let mut last_timing: Option<u64> = None;

            walk_words_mut(
                &mut utt.main.content.content,
                Some(TierDomain::Mor),
                &mut |leaf| match leaf {
                    WordItemMut::Word(word) => {
                        if let Some(&(main_utt, main_pos)) =
                            match_map.get(&(gold_utt_cursor, word_pos))
                        {
                            if let Some(Some(bullet)) = main_timings.get(&(main_utt, main_pos)) {
                                word.inline_bullet = Some(bullet.clone());
                                if first_timing.is_none() {
                                    first_timing = Some(bullet.timing.start_ms);
                                }
                                last_timing = Some(bullet.timing.end_ms);
                            }
                        }
                        word_pos += 1;
                    }
                    WordItemMut::ReplacedWord(replaced) => {
                        if let Some(&(main_utt, main_pos)) =
                            match_map.get(&(gold_utt_cursor, word_pos))
                        {
                            if let Some(Some(bullet)) = main_timings.get(&(main_utt, main_pos)) {
                                if !replaced.replacement.words.is_empty() {
                                    replaced.replacement.words[0].inline_bullet =
                                        Some(bullet.clone());
                                } else {
                                    replaced.word.inline_bullet = Some(bullet.clone());
                                }
                                if first_timing.is_none() {
                                    first_timing = Some(bullet.timing.start_ms);
                                }
                                last_timing = Some(bullet.timing.end_ms);
                            }
                        }
                        word_pos += 1;
                    }
                    WordItemMut::Separator(_) => {
                        word_pos += 1;
                    }
                },
            );

            // Set utterance-level timing from timed words
            if let (Some(start), Some(end)) = (first_timing, last_timing) {
                utt.main.content.bullet = Some(Bullet::new(start, end));
            }

            gold_utt_cursor += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// %mor/%gra projection: copy dependent tiers from main to gold
// ---------------------------------------------------------------------------

/// Project `%mor` and `%gra` tiers from main to gold for matched utterances.
///
/// For each gold utterance that has matched words, builds a new `%mor` tier
/// from the main's morphological items. Matched words get the main's `Mor`
/// item; unmatched gold words (deletions) keep a placeholder `?|?`.
pub fn project_mor_to_gold(
    main_file: &crate::ChatFile,
    gold_file: &mut crate::ChatFile,
    bundle: &GoldProjectionBundle,
) {
    use talkbank_model::model::dependent_tier::mor::{
        Mor, MorStem, MorTier, MorWord, PosCategory,
    };

    // 1. Collect %mor items from main: (utt_idx) → Vec<Mor>
    let mut main_mor_items: HashMap<usize, Vec<Mor>> = HashMap::new();
    let mut main_utt_cursor = 0usize;
    for line in &main_file.lines {
        if let Line::Utterance(utt) = line {
            for tier in &utt.dependent_tiers {
                if let DependentTier::Mor(mor_tier) = tier {
                    main_mor_items.insert(main_utt_cursor, mor_tier.items.to_vec());
                }
            }
            main_utt_cursor += 1;
        }
    }

    // 2. Build match lookups
    //    (gold_utt, gold_word_pos) → (main_utt, main_word_pos)
    let mut match_map: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    for utt_align in &bundle.utterances {
        for wm in &utt_align.word_matches {
            match_map.insert(
                (wm.gold_utt_idx, wm.gold_word_pos),
                (wm.main_utt_idx, wm.main_word_pos),
            );
        }
    }

    // 3. Walk gold utterances and build projected %mor tiers
    let mut gold_utt_cursor = 0usize;
    for line in gold_file.lines.iter_mut() {
        if let Line::Utterance(utt) = line {
            // Count Mor-domain words in this gold utterance
            let mut extracted_words = Vec::new();
            crate::extract::collect_utterance_content(
                &utt.main.content.content,
                TierDomain::Mor,
                &mut extracted_words,
            );
            let gold_word_count = extracted_words.len();

            if gold_word_count > 0 {
                // Build new %mor items
                let mut new_mor_items: Vec<Mor> = Vec::with_capacity(gold_word_count);
                let placeholder = Mor::new(MorWord::new(
                    PosCategory::from("?"),
                    MorStem::from("?"),
                ));

                for word_pos in 0..gold_word_count {
                    if let Some(&(main_utt, main_pos)) =
                        match_map.get(&(gold_utt_cursor, word_pos))
                    {
                        // Use main's Mor item for matched words
                        if let Some(items) = main_mor_items.get(&main_utt) {
                            if let Some(item) = items.get(main_pos) {
                                new_mor_items.push(item.clone());
                                continue;
                            }
                        }
                        new_mor_items.push(placeholder.clone());
                    } else {
                        new_mor_items.push(placeholder.clone());
                    }
                }

                let new_tier = DependentTier::Mor(MorTier::new_mor(new_mor_items));
                replace_or_add_tier(&mut utt.dependent_tiers, new_tier);
            }

            gold_utt_cursor += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// %xsrep injection on gold utterances
// ---------------------------------------------------------------------------

/// Inject `%xsrep` comparison annotations into gold utterances.
///
/// For each gold utterance with comparison tokens, adds a `%xsrep` user-defined
/// dependent tier containing the formatted comparison string.
pub fn inject_gold_comparison(
    gold_file: &mut crate::ChatFile,
    bundle: &GoldProjectionBundle,
) {
    let mut utt_line_indices: Vec<usize> = Vec::new();
    for (line_idx, line) in gold_file.lines.iter().enumerate() {
        if matches!(line, Line::Utterance(_)) {
            utt_line_indices.push(line_idx);
        }
    }

    for utt_align in &bundle.utterances {
        if utt_align.tokens.is_empty() {
            continue;
        }

        let utt_idx = utt_align.gold_utt_idx;
        if utt_idx >= utt_line_indices.len() {
            tracing::warn!(
                utt_idx,
                num_utterances = utt_line_indices.len(),
                "Gold compare utterance_index out of range"
            );
            continue;
        }

        let line_idx = utt_line_indices[utt_idx];

        // Format as UtteranceComparison for reuse of format_xsrep
        let utt_comparison = UtteranceComparison {
            utterance_index: utt_idx,
            speaker: utt_align.speaker.clone(),
            tokens: utt_align.tokens.clone(),
        };
        let xsrep_text = format_xsrep(&utt_comparison);

        if let Some(Line::Utterance(utt)) = gold_file.lines.get_mut(line_idx) {
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
            replace_or_add_tier(&mut utt.dependent_tiers, new_tier);
        }
    }
}

// ---------------------------------------------------------------------------
// Full gold projection: timing + mor + comparison annotations
// ---------------------------------------------------------------------------

/// Apply the full gold projection from a comparison bundle.
///
/// This is the main entry point for the gold-projection materializer:
/// 1. Projects word-level timing from main to gold.
/// 2. Projects `%mor` tiers from main to gold.
/// 3. Injects `%xsrep` comparison annotations on gold utterances.
pub fn apply_gold_projection(
    main_file: &crate::ChatFile,
    gold_file: &mut crate::ChatFile,
    bundle: &GoldProjectionBundle,
) {
    project_timing_to_gold(main_file, gold_file, bundle);
    project_mor_to_gold(main_file, gold_file, bundle);
    inject_gold_comparison(gold_file, bundle);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_lenient;

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
    fn find_best_segment_exact_match() {
        let gold = vec!["hello".into(), "world".into()];
        let main = vec!["hello".into(), "world".into()];
        let (start, end) = find_best_segment(&gold, &main);
        assert_eq!(start, 0);
        assert_eq!(end, 2);
    }

    #[test]
    fn find_best_segment_offset() {
        let gold = vec!["world".into()];
        let main = vec!["hello".into(), "big".into(), "world".into()];
        let (start, end) = find_best_segment(&gold, &main);
        assert_eq!(start, 2);
        assert_eq!(end, 3);
    }

    #[test]
    fn find_best_segment_empty() {
        let gold: Vec<String> = vec![];
        let main = vec!["hello".into()];
        assert_eq!(find_best_segment(&gold, &main), (0, 0));

        let gold = vec!["hello".into()];
        let main: Vec<String> = vec![];
        assert_eq!(find_best_segment(&gold, &main), (0, 0));
    }

    #[test]
    fn gold_projection_identical() {
        let chat = make_chat(&[("CHI", "hello world .")]);
        let (main_file, _) = parse_lenient(&chat);
        let (gold_file, _) = parse_lenient(&chat);

        let bundle = compare_gold_projection(&main_file, &gold_file);
        assert_eq!(bundle.metrics.wer, 0.0);
        assert_eq!(bundle.metrics.matches, 2);
        assert_eq!(bundle.utterances.len(), 1);
        assert_eq!(bundle.utterances[0].word_matches.len(), 2);
    }

    #[test]
    fn gold_projection_with_deletion() {
        let main = make_chat(&[("CHI", "hello .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (main_file, _) = parse_lenient(&main);
        let (gold_file, _) = parse_lenient(&gold);

        let bundle = compare_gold_projection(&main_file, &gold_file);
        assert_eq!(bundle.metrics.matches, 1);
        assert_eq!(bundle.metrics.deletions, 1);
        assert_eq!(bundle.utterances[0].tokens.len(), 2);
    }

    #[test]
    fn gold_projection_with_insertion() {
        let main = make_chat(&[("CHI", "hello big world .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (main_file, _) = parse_lenient(&main);
        let (gold_file, _) = parse_lenient(&gold);

        let bundle = compare_gold_projection(&main_file, &gold_file);
        assert_eq!(bundle.metrics.matches, 2);
        assert_eq!(bundle.metrics.insertions, 1);
    }

    #[test]
    fn gold_projection_multi_utterance() {
        let main = make_chat(&[("CHI", "hello ."), ("MOT", "goodbye .")]);
        let gold = make_chat(&[("CHI", "hello ."), ("MOT", "goodbye .")]);
        let (main_file, _) = parse_lenient(&main);
        let (gold_file, _) = parse_lenient(&gold);

        let bundle = compare_gold_projection(&main_file, &gold_file);
        assert_eq!(bundle.metrics.wer, 0.0);
        assert_eq!(bundle.metrics.matches, 2);
        assert_eq!(bundle.utterances.len(), 2);
        assert_eq!(bundle.utterances[0].word_matches.len(), 1);
        assert_eq!(bundle.utterances[1].word_matches.len(), 1);
    }

    #[test]
    fn inject_gold_comparison_adds_xsrep() {
        let main = make_chat(&[("CHI", "hello big world .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (main_file, _) = parse_lenient(&main);
        let (mut gold_file, _) = parse_lenient(&gold);

        let bundle = compare_gold_projection(&main_file, &gold_file);
        inject_gold_comparison(&mut gold_file, &bundle);

        let serialized = crate::serialize::to_chat_string(&gold_file);
        assert!(serialized.contains("%xsrep:"), "Should contain %xsrep tier");
    }

    #[test]
    fn full_gold_projection() {
        let main = make_chat(&[("CHI", "hello big world .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (main_file, _) = parse_lenient(&main);
        let (mut gold_file, _) = parse_lenient(&gold);

        let bundle = compare_gold_projection(&main_file, &gold_file);
        apply_gold_projection(&main_file, &mut gold_file, &bundle);

        let serialized = crate::serialize::to_chat_string(&gold_file);
        assert!(serialized.contains("%xsrep:"));
    }

    #[test]
    fn per_pos_metrics_populated_when_mor_present() {
        // Without %mor, POS defaults to "?" for all tokens
        let main = make_chat(&[("CHI", "hello world .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (main_file, _) = parse_lenient(&main);
        let (gold_file, _) = parse_lenient(&gold);

        let bundle = compare_gold_projection(&main_file, &gold_file);
        // With no %mor tiers, all POS should be "?"
        assert!(bundle.metrics.per_pos.contains_key("?"));
    }
}
