//! Two-pass UTR strategy for `+<` overlap-aware timing recovery.
//!
//! When overlapping speech is transcribed as separate utterances with `+<`
//! linkers, the backchannel words end up at the wrong position in the global
//! DP reference sequence. Pass 1 excludes `+<` utterances so the main-speaker
//! words align correctly, then pass 2 recovers backchannel timing from the
//! previous utterance's audio window.
//!
//! ## FA grouping stability
//!
//! Two-pass UTR can change anchor points for `estimate_untimed_boundaries`,
//! which alters FA group boundaries. Fewer, wider groups cause worse FA
//! alignment and lower final coverage — observed on German and Welsh files.
//!
//! When [`GroupingContext`] is provided, the best-of-both comparison uses FA
//! group counts as the primary signal: if two-pass creates fewer groups than
//! global, it falls back to global UTR before FA runs. Timed utterance count
//! remains the tiebreaker when grouping context is unavailable or group counts
//! are equal.

use talkbank_model::model::{Bullet, ChatFile, Line};

use crate::dp_align::{self, MatchMode};
use crate::fa::grouping::group_utterances;

use super::{AsrTimingToken, UtrResult, UtrStrategy, collect_utr_utterance_info, run_global_utr};

/// Parameters needed to compare FA grouping outcomes between strategies.
///
/// When provided, the best-of-both fallback in [`TwoPassOverlapUtr`] uses
/// `group_utterances()` to compare how many FA groups each strategy would
/// produce. Fewer groups means wider FA windows and worse alignment — the
/// specific failure mode observed on non-English files.
#[derive(Debug, Clone, Copy)]
pub struct GroupingContext {
    /// Total audio duration in milliseconds (needed for untimed boundary
    /// estimation inside `group_utterances`).
    pub total_audio_ms: u64,
    /// Maximum FA group duration in milliseconds.
    pub max_group_ms: u64,
}

/// Two-pass overlap-aware UTR strategy.
///
/// **Pass 1:** Runs the global alignment with `+<` utterances excluded from
/// the flattened word sequence. Main-speaker words align correctly without
/// backchannel interference.
///
/// **Pass 2:** For each `+<` utterance, finds the previous utterance's bullet
/// (set in pass 1), filters ASR tokens to that time window, and runs a small
/// DP alignment to recover the backchannel's timing.
///
/// When no `+<` utterances exist, pass 2 is a no-op and the result is
/// identical to [`super::GlobalUtr`].
///
/// ## FA grouping stability
///
/// When [`grouping_context`](Self::grouping_context) is set, the best-of-both
/// comparison checks FA group counts: if two-pass creates fewer groups than
/// global, it falls back to global to avoid the wider-window regression.
pub struct TwoPassOverlapUtr {
    /// Optional: total audio duration and max group size for grouping comparison.
    /// When set, the best-of-both fallback compares FA grouping outcomes.
    pub grouping_context: Option<GroupingContext>,
}

impl Default for TwoPassOverlapUtr {
    fn default() -> Self {
        Self::new()
    }
}

impl TwoPassOverlapUtr {
    /// Create a `TwoPassOverlapUtr` without grouping context (timed-count only).
    pub fn new() -> Self {
        Self {
            grouping_context: None,
        }
    }

    /// Create a `TwoPassOverlapUtr` with grouping context for FA stability.
    pub fn with_grouping_context(total_audio_ms: u64, max_group_ms: u64) -> Self {
        Self {
            grouping_context: Some(GroupingContext {
                total_audio_ms,
                max_group_ms,
            }),
        }
    }
}

impl UtrStrategy for TwoPassOverlapUtr {
    fn inject(&self, chat_file: &mut ChatFile, asr_tokens: &[AsrTimingToken]) -> UtrResult {
        // Run two-pass on a clone so we can compare against global.
        let mut two_pass_file = chat_file.clone();
        let two_pass_result = run_two_pass_inner(&mut two_pass_file, asr_tokens);

        // Run global on a separate clone for comparison.
        let mut global_file = chat_file.clone();
        let global_result = run_global_utr(&mut global_file, asr_tokens, false);

        let prefer_two_pass = if let Some(ctx) = &self.grouping_context {
            // Primary signal: FA group count. Fewer groups means wider FA
            // windows, which causes worse alignment on non-English files.
            let two_pass_groups =
                group_utterances(&two_pass_file, ctx.max_group_ms, Some(ctx.total_audio_ms)).len();
            let global_groups =
                group_utterances(&global_file, ctx.max_group_ms, Some(ctx.total_audio_ms)).len();

            if two_pass_groups != global_groups {
                // Prefer whichever creates more groups (more precise FA windows).
                two_pass_groups >= global_groups
            } else {
                // Equal groups — fall back to timed utterance count.
                let two_pass_timed = count_timed_utterances(&two_pass_file);
                let global_timed = count_timed_utterances(&global_file);
                // When equal, prefer two-pass (better backchannel placement).
                two_pass_timed >= global_timed
            }
        } else {
            // No grouping context — use timed utterance count only.
            let two_pass_timed = count_timed_utterances(&two_pass_file);
            let global_timed = count_timed_utterances(&global_file);
            two_pass_timed >= global_timed
        };

        if prefer_two_pass {
            *chat_file = two_pass_file;
            two_pass_result
        } else {
            tracing::info!(
                "Two-pass UTR created fewer FA groups than global — falling back to global"
            );
            *chat_file = global_file;
            global_result
        }
    }
}

/// Core two-pass implementation: pass 1 excludes `+<`, pass 2 recovers them.
fn run_two_pass_inner(chat_file: &mut ChatFile, asr_tokens: &[AsrTimingToken]) -> UtrResult {
    // Pass 1: global alignment excluding +< utterances.
    let mut result = run_global_utr(chat_file, asr_tokens, true);

    if asr_tokens.is_empty() {
        return result;
    }

    // Pass 2: recover timing for +< utterances from predecessor windows.
    let utt_infos = collect_utr_utterance_info(chat_file);

    // Collect current bullets (after pass 1) for window lookup.
    let utt_bullets: Vec<Option<(u64, u64)>> = chat_file
        .lines
        .iter()
        .filter_map(|line| {
            if let Line::Utterance(utt) = line {
                Some(
                    utt.main
                        .content
                        .bullet
                        .as_ref()
                        .map(|b| (b.timing.start_ms, b.timing.end_ms)),
                )
            } else {
                None
            }
        })
        .collect();

    // Track which +< utterances we successfully time in pass 2.
    let mut pass2_bullets: Vec<(usize, u64, u64)> = Vec::new();

    for (utt_idx, info) in utt_infos.iter().enumerate() {
        if !info.has_lazy_overlap || info.has_bullet || info.words.is_empty() {
            continue;
        }

        // Adaptive window: try narrow first, widen on failure.
        if let Some((start_ms, end_ms)) =
            recover_with_adaptive_window(&info.words, asr_tokens, utt_idx, &utt_bullets)
        {
            pass2_bullets.push((utt_idx, start_ms, end_ms));
            // Adjust counts: this was counted as unmatched in pass 1.
            result.unmatched -= 1;
            result.injected += 1;
        }
    }

    // Apply pass 2 bullets to the ChatFile.
    if !pass2_bullets.is_empty() {
        let mut utt_idx = 0;
        let mut bullet_iter = pass2_bullets.iter().peekable();
        for line in &mut chat_file.lines {
            if let Line::Utterance(utt) = line {
                if let Some(&&(target_idx, start_ms, end_ms)) = bullet_iter.peek()
                    && utt_idx == target_idx
                {
                    utt.main.content.bullet = Some(Bullet::new(start_ms, end_ms));
                    bullet_iter.next();
                }
                utt_idx += 1;
            }
        }
    }

    result
}

/// Count utterances that have a bullet (timed) in the chat file.
fn count_timed_utterances(chat_file: &ChatFile) -> usize {
    chat_file
        .lines
        .iter()
        .filter(|line| {
            if let Line::Utterance(utt) = line {
                utt.main.content.bullet.is_some()
            } else {
                false
            }
        })
        .count()
}

/// Recover backchannel timing with an adaptive window strategy.
///
/// Instead of a fixed buffer, tries increasingly wider windows around the
/// predecessor utterance until a match is found or all attempts are exhausted.
///
/// **Strategy:**
/// 1. **Narrow (±500ms):** Tight window around predecessor. Works well when
///    ASR timing is accurate (typical for English).
/// 2. **Predecessor duration:** Window equals the predecessor's full duration
///    added as buffer on each side. Catches backchannels whose ASR timing
///    drifts by up to the utterance length.
/// 3. **Double predecessor duration:** For poor ASR (non-English) where timing
///    can be significantly offset.
///
/// Stops at the first match to prefer the tightest plausible window.
fn recover_with_adaptive_window(
    words: &[String],
    asr_tokens: &[AsrTimingToken],
    utt_idx: usize,
    utt_bullets: &[Option<(u64, u64)>],
) -> Option<(u64, u64)> {
    // Find predecessor bullet
    let (pred_start, pred_end) = find_predecessor_bullet(utt_idx, utt_bullets)?;
    let pred_duration = pred_end.saturating_sub(pred_start);

    // Try increasingly wider buffers
    let buffers = [
        500,                           // tight: ±500ms
        pred_duration.max(2000),       // medium: ±predecessor duration (min 2s)
        (pred_duration * 2).max(5000), // wide: ±2x predecessor duration (min 5s)
    ];

    for buffer_ms in buffers {
        let window_start = pred_start.saturating_sub(buffer_ms);
        let window_end = pred_end + buffer_ms;

        if let Some(timing) = recover_overlap_timing(words, asr_tokens, window_start, window_end) {
            return Some(timing);
        }
    }

    None
}

/// Find the nearest preceding utterance's bullet range (no buffer applied).
fn find_predecessor_bullet(
    utt_idx: usize,
    utt_bullets: &[Option<(u64, u64)>],
) -> Option<(u64, u64)> {
    for prev_idx in (0..utt_idx).rev() {
        if let Some(bullet) = utt_bullets[prev_idx] {
            return Some(bullet);
        }
    }
    None
}

/// Recover timing for a small set of overlap words from ASR tokens within a
/// constrained time window.
///
/// Filters `asr_tokens` to those overlapping `[window_start_ms, window_end_ms]`,
/// then runs a Hirschberg DP alignment of the `+<` utterance's words against
/// the windowed tokens. Returns the matched time span if any words matched.
///
/// This is cheap — typically 1–3 backchannel words against 5–20 windowed tokens.
pub fn recover_overlap_timing(
    words: &[String],
    asr_tokens: &[AsrTimingToken],
    window_start_ms: u64,
    window_end_ms: u64,
) -> Option<(u64, u64)> {
    // Filter ASR tokens to those overlapping the window.
    let windowed: Vec<(usize, &AsrTimingToken)> = asr_tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| t.start_ms < window_end_ms && t.end_ms > window_start_ms)
        .collect();

    if windowed.is_empty() {
        return None;
    }

    let windowed_texts: Vec<String> = windowed.iter().map(|(_, t)| t.text.clone()).collect();

    let alignment = dp_align::align(words, &windowed_texts, MatchMode::CaseInsensitive);

    let mut min_start: Option<u64> = None;
    let mut max_end: Option<u64> = None;

    for result_item in &alignment {
        if let dp_align::AlignResult::Match { reference_idx, .. } = result_item {
            let token = windowed[*reference_idx].1;
            match min_start {
                Some(s) if token.start_ms < s => min_start = Some(token.start_ms),
                None => min_start = Some(token.start_ms),
                _ => {}
            }
            match max_end {
                Some(e) if token.end_ms > e => max_end = Some(token.end_ms),
                None => max_end = Some(token.end_ms),
                _ => {}
            }
        }
    }

    match (min_start, max_end) {
        (Some(start), Some(end)) => Some((start, end)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_asr_tokens(words_with_times: &[(&str, u64, u64)]) -> Vec<AsrTimingToken> {
        words_with_times
            .iter()
            .map(|(text, start, end)| AsrTimingToken {
                text: text.to_string(),
                start_ms: *start,
                end_ms: *end,
            })
            .collect()
    }

    #[test]
    fn test_recover_overlap_timing_finds_mhm_in_window() {
        let words = vec!["mhm".to_string()];
        let tokens = make_asr_tokens(&[("mhm", 1800, 2200)]);
        let result = recover_overlap_timing(&words, &tokens, 0, 3000);
        assert_eq!(result, Some((1800, 2200)));
    }

    #[test]
    fn test_recover_overlap_timing_no_match_outside_window() {
        let words = vec!["mhm".to_string()];
        let tokens = make_asr_tokens(&[("mhm", 5000, 5500)]);
        let result = recover_overlap_timing(&words, &tokens, 0, 3000);
        assert_eq!(result, None);
    }

    #[test]
    fn test_recover_overlap_timing_multi_word() {
        let words = vec!["oh".to_string(), "okay".to_string()];
        let tokens = make_asr_tokens(&[("oh", 1500, 1700), ("okay", 1800, 2200)]);
        let result = recover_overlap_timing(&words, &tokens, 0, 3000);
        assert_eq!(result, Some((1500, 2200)));
    }

    #[test]
    fn test_recover_overlap_timing_empty_window() {
        let words = vec!["mhm".to_string()];
        let tokens = make_asr_tokens(&[("mhm", 1800, 2200)]);
        // Window that doesn't overlap any tokens
        let result = recover_overlap_timing(&words, &tokens, 5000, 6000);
        assert_eq!(result, None);
    }

    #[test]
    fn test_find_predecessor_bullet_immediate() {
        let bullets = vec![
            Some((1000, 3000)),
            None, // +< utterance at index 1
        ];
        let bullet = find_predecessor_bullet(1, &bullets);
        assert_eq!(bullet, Some((1000, 3000)));
    }

    #[test]
    fn test_find_predecessor_bullet_skips_untimed() {
        let bullets = vec![
            Some((1000, 3000)),
            None, // untimed (not +<)
            None, // +< utterance at index 2
        ];
        let bullet = find_predecessor_bullet(2, &bullets);
        assert_eq!(bullet, Some((1000, 3000)));
    }

    #[test]
    fn test_find_predecessor_bullet_none_at_start() {
        let bullets = vec![
            None, // +< utterance at index 0 — no predecessor
        ];
        let bullet = find_predecessor_bullet(0, &bullets);
        assert_eq!(bullet, None);
    }

    #[test]
    fn test_adaptive_window_finds_with_narrow() {
        let words = vec!["mhm".to_string()];
        let tokens = make_asr_tokens(&[("mhm", 1800, 2200)]);
        let bullets = vec![
            Some((1000, 3000)), // predecessor
            None,               // +< utterance
        ];
        // "mhm" at 1800 is within narrow window (1000-500=500, 3000+500=3500)
        let result = recover_with_adaptive_window(&words, &tokens, 1, &bullets);
        assert_eq!(result, Some((1800, 2200)));
    }

    #[test]
    fn test_adaptive_window_widens_to_find_match() {
        let words = vec!["mhm".to_string()];
        // "mhm" is 5 seconds after predecessor ends — too far for narrow (±500ms)
        // but within medium (predecessor duration = 2000ms, so ±2000ms → window 0..7000)
        let tokens = make_asr_tokens(&[("mhm", 6500, 6800)]);
        let bullets = vec![
            Some((1000, 3000)), // predecessor: 2s duration
            None,               // +< utterance
        ];
        let result = recover_with_adaptive_window(&words, &tokens, 1, &bullets);
        assert_eq!(result, Some((6500, 6800)));
    }

    #[test]
    fn test_adaptive_window_no_predecessor() {
        let words = vec!["mhm".to_string()];
        let tokens = make_asr_tokens(&[("mhm", 1800, 2200)]);
        let bullets = vec![None]; // no predecessor
        let result = recover_with_adaptive_window(&words, &tokens, 0, &bullets);
        assert_eq!(result, None);
    }

    /// When pass 2 leaves more unmatched than global would, the best-of-both
    /// fallback should use global results instead.
    #[test]
    fn test_best_of_both_falls_back_to_global() {
        use talkbank_direct_parser::DirectParser;

        let chat_text = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant, INV Investigator
@ID:\teng|test|PAR|||||Participant|||
@ID:\teng|test|INV|||||Investigator|||
@Media:\ttest, audio
*PAR:\tI went to the store yesterday .
*INV:\t+< ja .
*PAR:\tand bought some groceries .
@End
";
        let parser = DirectParser::new().unwrap();
        let mut chat = parser.parse_chat_file(chat_text).unwrap();

        // ASR tokens: PAR's words + "ja" appears far from predecessor window.
        // The global DP can match "ja" because it sees the full stream.
        // Pass 2 windowed recovery can't find "ja" near the predecessor.
        let tokens = vec![
            AsrTimingToken {
                text: "I".into(),
                start_ms: 100,
                end_ms: 300,
            },
            AsrTimingToken {
                text: "went".into(),
                start_ms: 400,
                end_ms: 800,
            },
            AsrTimingToken {
                text: "to".into(),
                start_ms: 900,
                end_ms: 1100,
            },
            AsrTimingToken {
                text: "the".into(),
                start_ms: 1200,
                end_ms: 1400,
            },
            AsrTimingToken {
                text: "store".into(),
                start_ms: 1500,
                end_ms: 2000,
            },
            AsrTimingToken {
                text: "yesterday".into(),
                start_ms: 2300,
                end_ms: 3000,
            },
            // "ja" appears 50 seconds later (simulating poor ASR timing for non-English)
            AsrTimingToken {
                text: "ja".into(),
                start_ms: 50000,
                end_ms: 50500,
            },
            AsrTimingToken {
                text: "and".into(),
                start_ms: 5000,
                end_ms: 5300,
            },
            AsrTimingToken {
                text: "bought".into(),
                start_ms: 5400,
                end_ms: 5800,
            },
            AsrTimingToken {
                text: "some".into(),
                start_ms: 5900,
                end_ms: 6200,
            },
            AsrTimingToken {
                text: "groceries".into(),
                start_ms: 6300,
                end_ms: 7000,
            },
        ];

        let result = TwoPassOverlapUtr::new().inject(&mut chat, &tokens);

        // The fallback should have kicked in: global can match "ja" at 50000ms,
        // while pass 2 can't find "ja" in the predecessor window (100-3000ms ± buffer).
        // With best-of-both, all 3 utterances should be timed.
        println!(
            "injected={} skipped={} unmatched={}",
            result.injected, result.skipped, result.unmatched
        );
        assert_eq!(
            result.unmatched, 0,
            "best-of-both should fall back to global and time all utterances"
        );
        assert_eq!(result.injected, 3);
    }

    /// When grouping context is provided and two-pass creates fewer FA groups
    /// than global (the wider-window regression), the fallback should use
    /// global results even if two-pass timed more utterances.
    #[test]
    fn test_grouping_fallback_prefers_more_groups() {
        use talkbank_direct_parser::DirectParser;

        // Construct a scenario where two-pass changes bullet placement enough
        // to merge FA groups. We use a file with a +< backchannel between two
        // main-speaker utterances. The ASR tokens are positioned so two-pass
        // recovers the +< at a different time window than global would.
        let chat_text = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant, INV Investigator
@ID:\teng|test|PAR|||||Participant|||
@ID:\teng|test|INV|||||Investigator|||
@Media:\ttest, audio
*PAR:\tI went to the store yesterday .
*INV:\t+< ja .
*PAR:\tand bought some groceries .
@End
";
        let parser = DirectParser::new().unwrap();
        let mut chat = parser.parse_chat_file(chat_text).unwrap();

        // ASR tokens: "ja" far from predecessor → two-pass can't find it,
        // global can. This means two-pass leaves "ja" untimed, which changes
        // estimated boundaries and could create fewer groups.
        let tokens = vec![
            AsrTimingToken {
                text: "I".into(),
                start_ms: 100,
                end_ms: 300,
            },
            AsrTimingToken {
                text: "went".into(),
                start_ms: 400,
                end_ms: 800,
            },
            AsrTimingToken {
                text: "to".into(),
                start_ms: 900,
                end_ms: 1100,
            },
            AsrTimingToken {
                text: "the".into(),
                start_ms: 1200,
                end_ms: 1400,
            },
            AsrTimingToken {
                text: "store".into(),
                start_ms: 1500,
                end_ms: 2000,
            },
            AsrTimingToken {
                text: "yesterday".into(),
                start_ms: 2300,
                end_ms: 3000,
            },
            // "ja" at 50s — too far for two-pass windowed recovery
            AsrTimingToken {
                text: "ja".into(),
                start_ms: 50000,
                end_ms: 50500,
            },
            AsrTimingToken {
                text: "and".into(),
                start_ms: 5000,
                end_ms: 5300,
            },
            AsrTimingToken {
                text: "bought".into(),
                start_ms: 5400,
                end_ms: 5800,
            },
            AsrTimingToken {
                text: "some".into(),
                start_ms: 5900,
                end_ms: 6200,
            },
            AsrTimingToken {
                text: "groceries".into(),
                start_ms: 6300,
                end_ms: 7000,
            },
        ];

        // With grouping context: total_audio_ms covers the full range,
        // max_group_ms is small enough to create multiple groups when
        // all utterances are timed.
        let strategy = TwoPassOverlapUtr::with_grouping_context(60000, 15000);
        let result = strategy.inject(&mut chat, &tokens);

        // Global should be preferred because it can time "ja" (creating
        // tighter groups), while two-pass leaves it untimed (wider groups).
        assert_eq!(
            result.unmatched, 0,
            "grouping fallback should choose global which times all utterances"
        );
        assert_eq!(result.injected, 3);
    }

    /// When two-pass creates equal or more FA groups than global, two-pass
    /// should be preferred (better backchannel placement).
    #[test]
    fn test_grouping_keeps_two_pass_when_groups_equal_or_better() {
        use talkbank_direct_parser::DirectParser;

        let chat_text = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant, INV Investigator
@ID:\teng|test|PAR|||||Participant|||
@ID:\teng|test|INV|||||Investigator|||
@Media:\ttest, audio
*PAR:\tI went to the store yesterday .
*INV:\t+< mhm .
*PAR:\tand bought some groceries .
@End
";
        let parser = DirectParser::new().unwrap();
        let mut chat = parser.parse_chat_file(chat_text).unwrap();

        // "mhm" close to predecessor → two-pass windowed recovery succeeds.
        // Both strategies should create the same groups.
        let tokens = vec![
            AsrTimingToken {
                text: "I".into(),
                start_ms: 100,
                end_ms: 300,
            },
            AsrTimingToken {
                text: "went".into(),
                start_ms: 400,
                end_ms: 800,
            },
            AsrTimingToken {
                text: "to".into(),
                start_ms: 900,
                end_ms: 1100,
            },
            AsrTimingToken {
                text: "the".into(),
                start_ms: 1200,
                end_ms: 1400,
            },
            AsrTimingToken {
                text: "store".into(),
                start_ms: 1500,
                end_ms: 2000,
            },
            AsrTimingToken {
                text: "yesterday".into(),
                start_ms: 2300,
                end_ms: 3000,
            },
            AsrTimingToken {
                text: "mhm".into(),
                start_ms: 1800,
                end_ms: 2200,
            },
            AsrTimingToken {
                text: "and".into(),
                start_ms: 5000,
                end_ms: 5300,
            },
            AsrTimingToken {
                text: "bought".into(),
                start_ms: 5400,
                end_ms: 5800,
            },
            AsrTimingToken {
                text: "some".into(),
                start_ms: 5900,
                end_ms: 6200,
            },
            AsrTimingToken {
                text: "groceries".into(),
                start_ms: 6300,
                end_ms: 7000,
            },
        ];

        let strategy = TwoPassOverlapUtr::with_grouping_context(60000, 15000);
        let result = strategy.inject(&mut chat, &tokens);

        // Two-pass should be kept (groups equal, better backchannel placement).
        assert_eq!(result.injected, 3, "two-pass should time all 3 utterances");
        assert_eq!(result.unmatched, 0);
    }
}
