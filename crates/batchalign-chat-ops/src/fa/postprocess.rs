//! Post-processing: fix end times, bound by utterance, update bullets.

use talkbank_model::alignment::helpers::{
    WordItem, WordItemMut, walk_words, walk_words_mut,
};
use talkbank_model::model::{Bullet, Utterance, UtteranceContent, Word};

use super::get_word_timing;
use super::{FaTimingMode, TimeSpan};

/// Post-process timings: set word end times, bound by utterance, update bullets.
///
/// 1. For `Continuous` mode: set each word's end time to the next word's start time
/// 2. Bound all word times within utterance bullet range
/// 3. Drop invalid timings (start >= end)
/// 4. Update utterance bullet from word timings
pub fn postprocess_utterance_timings(utterance: &mut Utterance, timing_mode: FaTimingMode) {
    let mut word_timings: Vec<Option<TimeSpan>> = Vec::new();
    collect_word_timings(&utterance.main.content.content, &mut word_timings);

    if word_timings.is_empty() {
        return;
    }

    // For Continuous mode: set each word's end_ms to next word's start_ms.
    // Uses a backward pass (O(w)) instead of per-word forward scan (O(w²)).
    if timing_mode == FaTimingMode::Continuous {
        let n = word_timings.len();

        // Last timed word: extend to utterance bullet end or +500ms
        // (must happen first — the backward pass below would leave it unchanged
        // since there's no next_start, but we need to set its end explicitly)
        for i in (0..n).rev() {
            if let Some(span) = word_timings[i] {
                if span.start_ms == span.end_ms {
                    let fallback_end = if let Some(ref bullet) = utterance.main.content.bullet {
                        let utt_end = bullet.timing.end_ms;
                        if utt_end > span.start_ms {
                            utt_end
                        } else {
                            span.start_ms + 500
                        }
                    } else {
                        span.start_ms + 500
                    };
                    word_timings[i] = Some(TimeSpan::new(span.start_ms, fallback_end));
                }
                break;
            }
        }

        // Backward pass: track the next timed word's start_ms and propagate it
        // as the current word's end_ms.
        let mut next_start: Option<u64> = None;
        for i in (0..n).rev() {
            if let Some(span) = word_timings[i] {
                if let Some(ns) = next_start {
                    word_timings[i] = Some(TimeSpan::new(span.start_ms, ns));
                }
                next_start = Some(span.start_ms);
            }
        }
    }

    // Bound by utterance bullet range
    if let Some(ref bullet) = utterance.main.content.bullet {
        let utt_start = bullet.timing.start_ms;
        let utt_end = bullet.timing.end_ms;

        for timing in &mut word_timings {
            if let Some(span) = timing {
                let clamped_start = span.start_ms.max(utt_start);
                let clamped_end = span.end_ms.min(utt_end);
                if clamped_start >= clamped_end {
                    tracing::warn!(
                        "word timing dropped: clamped to utterance boundary made start >= end"
                    );
                    *timing = None;
                } else {
                    *span = TimeSpan::new(clamped_start, clamped_end);
                }
            }
        }
    }

    // Write timings back to the AST
    let mut idx = 0;
    set_word_timings(&mut utterance.main.content.content, &word_timings, &mut idx);
}

/// Collect current word timings in document order.
///
/// Visits ALL words (no alignability filter). For replaced words, only the
/// original (spoken) word's timing is collected.
pub(super) fn collect_word_timings(content: &[UtteranceContent], out: &mut Vec<Option<TimeSpan>>) {
    // domain=None: recurse into all groups unconditionally
    walk_words(content, None, &mut |leaf| match leaf {
        WordItem::Word(word) => {
            out.push(get_word_timing(word));
        }
        WordItem::ReplacedWord(replaced) => {
            out.push(get_word_timing(&replaced.word));
        }
        WordItem::Separator(_) => {}
    });
}

/// Write timings back into word AST nodes.
///
/// Visits ALL words in document order (same order as `collect_word_timings`).
/// For replaced words, sets timing on the original (spoken) word only.
fn set_word_timings(
    content: &mut [UtteranceContent],
    timings: &[Option<TimeSpan>],
    idx: &mut usize,
) {
    // domain=None: recurse into all groups unconditionally
    walk_words_mut(content, None, &mut |leaf| match leaf {
        WordItemMut::Word(word) => {
            set_word_timing(word, timings, idx);
        }
        WordItemMut::ReplacedWord(replaced) => {
            set_word_timing(&mut replaced.word, timings, idx);
        }
        WordItemMut::Separator(_) => {}
    });
}

fn set_word_timing(word: &mut Word, timings: &[Option<TimeSpan>], idx: &mut usize) {
    if *idx < timings.len() {
        match timings[*idx] {
            Some(span) => {
                word.inline_bullet = Some(Bullet::new(span.start_ms, span.end_ms));
            }
            None => {
                word.inline_bullet = None;
            }
        }
    }
    *idx += 1;
}
