//! Timestamp injection into the CHAT AST.

use talkbank_model::alignment::helpers::{
    TierDomain, WordItemMut, walk_words_mut, counts_for_tier,
};
use talkbank_model::model::{Bullet, Utterance, Word};

use super::WordTiming;

/// Read cursor into a flat array of word timings for an FA group.
///
/// Advances by one for each Wor-alignable word encountered.
pub struct TimingCursor<'a> {
    timings: &'a [Option<WordTiming>],
    pos: usize,
}

#[allow(dead_code)]
impl<'a> TimingCursor<'a> {
    /// Create a new cursor at position 0.
    pub fn new(timings: &'a [Option<WordTiming>]) -> Self {
        Self { timings, pos: 0 }
    }

    /// Create a new cursor starting at the given offset.
    pub fn with_offset(timings: &'a [Option<WordTiming>], offset: usize) -> Self {
        Self {
            timings,
            pos: offset,
        }
    }

    /// Advance the position and return the timing at the previous position.
    ///
    /// Always advances by one, even past the end — this matches the FA injection
    /// invariant that every alignable word must advance the cursor.
    pub fn take(&mut self) -> Option<&WordTiming> {
        let slot = self.timings.get(self.pos);
        self.pos += 1;
        slot.and_then(|o| o.as_ref())
    }

    /// Current read position.
    pub fn position(&self) -> usize {
        self.pos
    }
}

/// Inject word-level timings into the AST for a specific utterance.
///
/// `timings` is indexed by the flat word position within the group.
/// Only words that are Wor-alignable get timing (matching the extraction order).
///
/// * `utterance` - The utterance whose words will receive inline timing bullets.
/// * `timings` - Flat array of optional timings for the entire FA group. Each
///   element corresponds to one Wor-alignable word across all utterances in the
///   group.
/// * `timing_offset` - Current read position into `timings`. Advanced by one for
///   each Wor-alignable word encountered in this utterance. The caller should
///   initialize this to 0 for the first utterance in a group and pass the same
///   mutable reference through consecutive utterances.
pub fn inject_timings_for_utterance(
    utterance: &mut Utterance,
    timings: &[Option<WordTiming>],
    timing_offset: &mut usize,
) {
    let mut cursor = TimingCursor::with_offset(timings, *timing_offset);
    // domain=None: recurse into all groups unconditionally (FA needs all words)
    walk_words_mut(
        &mut utterance.main.content.content,
        None,
        &mut |leaf| match leaf {
            WordItemMut::Word(word) => {
                inject_timing_on_word(word, &mut cursor);
            }
            WordItemMut::ReplacedWord(replaced) => {
                if !replaced.replacement.words.is_empty() {
                    for word in &mut replaced.replacement.words {
                        inject_timing_on_word(word, &mut cursor);
                    }
                } else {
                    inject_timing_on_word(&mut replaced.word, &mut cursor);
                }
            }
            WordItemMut::Separator(_) => {}
        },
    );
    *timing_offset = cursor.position();
}

fn inject_timing_on_word(word: &mut Word, cursor: &mut TimingCursor<'_>) {
    if !counts_for_tier(word, TierDomain::Wor) {
        return;
    }
    if let Some(t) = cursor.take() {
        word.inline_bullet = Some(Bullet::new(t.start_ms, t.end_ms));
    }
}
