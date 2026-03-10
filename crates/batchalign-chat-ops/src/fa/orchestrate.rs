//! Full-file orchestration: apply FA results, enforce monotonicity, enforce E704.

use std::collections::HashMap;

use talkbank_model::alignment::align_main_to_wor;
use talkbank_model::model::{
    BracketedItem, ChatFile, DependentTier, Line, Utterance, UtteranceContent,
};

use super::FaTimingMode;
use super::injection::inject_timings_for_utterance;
use super::postprocess::postprocess_utterance_timings;
use super::{
    FaGroup, WordTiming, add_wor_tier, count_alignable_main_words, get_utterance_mut,
    update_utterance_bullet,
};

/// Apply FA results to a ChatFile: inject timings, postprocess, optionally
/// generate %wor, enforce monotonicity, enforce E704 same-speaker non-overlap.
///
/// `groups` and `responses` must be parallel: `responses[i]` is the aligned
/// timings for `groups[i]`.
///
/// When `write_wor` is `true`, a `%wor` tier is generated for each utterance.
/// When `false`, existing `%wor` tiers are left untouched and no new ones are added.
pub fn apply_fa_results(
    chat_file: &mut ChatFile,
    groups: &[FaGroup],
    responses: &[Vec<Option<WordTiming>>],
    timing_mode: FaTimingMode,
    write_wor: bool,
) {
    // 1. Distribute timings from each group's response to utterances
    for (group, timings) in groups.iter().zip(responses.iter()) {
        let mut timing_offset: usize = 0;

        for &utt_idx in &group.utterance_indices {
            let utt = match get_utterance_mut(chat_file, utt_idx) {
                Some(u) => u,
                None => continue,
            };
            inject_timings_for_utterance(utt, timings, &mut timing_offset);
        }
    }

    // 2. Postprocess all grouped utterances
    let all_utt_indices: Vec<crate::indices::UtteranceIdx> = groups
        .iter()
        .flat_map(|g| g.utterance_indices.iter().copied())
        .collect();

    for &utt_idx in &all_utt_indices {
        if let Some(utt) = get_utterance_mut(chat_file, utt_idx) {
            postprocess_utterance_timings(utt, timing_mode);
            update_utterance_bullet(utt);
            if write_wor {
                add_wor_tier(utt);
            }
        }
    }

    // NOTE: E362 (monotonicity) and E704 (same-speaker overlap) enforcement
    // was removed here. These passes aggressively stripped timing from
    // utterances that had "imperfect but usable" timings, causing severe
    // regressions vs batchalign 0.8.x (up to 60% timing loss on real data).
    // The CHAT validator in talkbank-tools flags these violations after the
    // fact — the FA pipeline should not silently destroy timing data.
}

/// Refresh a CHAT file that already carries reusable `%wor` timing.
///
/// This is the cheap rerun path for `align`. Instead of sending audio back
/// through the FA worker, the function:
///
/// 1. aligns each `%wor` tier back to the main tier,
/// 2. rehydrates main-tier `inline_bullet` timing from `%wor`,
/// 3. removes any parsed `InternalBullet` tokens left over from roundtripped
///    serialization,
/// 4. recomputes utterance bullets, and
/// 5. optionally regenerates `%wor`.
///
/// Callers should only use this after [`super::has_reusable_wor_timing`]
/// succeeds.
pub fn refresh_existing_alignment(chat_file: &mut ChatFile, write_wor: bool) {
    for line in &mut chat_file.lines {
        let Line::Utterance(utterance) = line else {
            continue;
        };
        if count_alignable_main_words(utterance) == 0 {
            continue;
        }

        let refreshed = refresh_existing_alignment_for_utterance(utterance, write_wor);
        assert!(
            refreshed,
            "refresh_existing_alignment requires reusable %wor timing"
        );
    }
}

/// Return `true` when one utterance has reusable `%wor` timing.
///
/// This is the per-utterance form of the cheap rerun check. It is useful for
/// selective reuse in incremental align workflows where only some utterances
/// remain trustworthy after manual edits.
pub fn has_reusable_wor_timing_for_utterance(utterance: &Utterance) -> bool {
    collect_wor_backed_timings(utterance).is_some()
}

/// Refresh one utterance from its existing `%wor` timing.
///
/// Returns `true` when the utterance had a clean reusable `%wor` mapping and
/// was refreshed successfully. Returns `false` when `%wor` was missing,
/// mismatched, or partially untimed.
pub fn refresh_existing_alignment_for_utterance(
    utterance: &mut Utterance,
    write_wor: bool,
) -> bool {
    let Some(timings) = collect_wor_backed_timings(utterance) else {
        return false;
    };

    strip_internal_bullet_tokens(&mut utterance.main.content.content.0);
    let mut offset = 0usize;
    inject_timings_for_utterance(utterance, &timings, &mut offset);
    update_utterance_bullet(utterance);
    if write_wor {
        add_wor_tier(utterance);
    }
    true
}

/// Refresh timing for utterances with reusable `%wor`, leaving stale ones
/// untouched for FA worker processing.
///
/// This is the per-utterance counterpart to [`refresh_existing_alignment()`].
/// Unlike that function (which asserts all utterances are reusable), this one
/// only refreshes utterances in the provided set, skipping stale ones that
/// will go through FA workers.
pub fn refresh_reusable_utterances(
    chat_file: &mut ChatFile,
    reusable_indices: &std::collections::HashSet<usize>,
    write_wor: bool,
) {
    let mut utt_idx = 0usize;
    for line in &mut chat_file.lines {
        let Line::Utterance(utterance) = line else {
            continue;
        };
        if reusable_indices.contains(&utt_idx) {
            let refreshed = refresh_existing_alignment_for_utterance(utterance, write_wor);
            debug_assert!(
                refreshed,
                "utterance {utt_idx} was in reusable set but refresh failed"
            );
        }
        utt_idx += 1;
    }
}

/// Enforce E362: strip timing from utterances whose start time is before
/// the previous utterance's start time (non-monotonic ordering).
pub fn enforce_monotonicity(chat_file: &mut ChatFile) {
    let mut last_start_ms: u64 = 0;
    for line in chat_file.lines.iter_mut() {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };
        match utt.main.content.bullet.as_ref().map(|b| b.timing.start_ms) {
            Some(s) if s < last_start_ms => {
                tracing::warn!(
                    start_ms = s,
                    last_start_ms,
                    "stripping non-monotonic utterance timing to enforce E362"
                );
                strip_utterance_timing(utt);
            }
            Some(s) => last_start_ms = s,
            None => {}
        }
    }
}

/// Enforce E704: strip timing from the EARLIER utterance when consecutive
/// same-speaker utterances overlap by more than 500ms tolerance.
pub fn strip_e704_same_speaker_overlaps(chat_file: &mut ChatFile) {
    const E704_TOLERANCE_MS: u64 = 500;

    let utt_info: Vec<(usize, String, u64, u64)> = chat_file
        .lines
        .iter()
        .enumerate()
        .filter_map(|(i, line)| {
            if let Line::Utterance(u) = line {
                let bullet = u.main.content.bullet.as_ref()?;
                let speaker = u.main.speaker.as_str().to_string();
                Some((i, speaker, bullet.timing.start_ms, bullet.timing.end_ms))
            } else {
                None
            }
        })
        .collect();

    let mut to_strip: Vec<usize> = Vec::new();
    let mut last_by_speaker: HashMap<String, (usize, u64)> = HashMap::new();

    for &(line_idx, ref speaker, start_ms, end_ms) in &utt_info {
        if let Some(&(prev_idx, prev_end)) = last_by_speaker.get(speaker.as_str())
            && prev_end > start_ms + E704_TOLERANCE_MS
        {
            to_strip.push(prev_idx);
        }
        last_by_speaker.insert(speaker.clone(), (line_idx, end_ms));
    }

    for idx in to_strip {
        if let Line::Utterance(utt) = &mut chat_file.lines[idx] {
            strip_utterance_timing(utt);
        }
    }
}

// ---------------------------------------------------------------------------
// Timing stripping helpers
// ---------------------------------------------------------------------------

/// Strip all timing information from utterance content items.
///
/// Removes `InternalBullet` items and clears `inline_bullet` from all words.
pub fn strip_timing_from_content(items: &mut Vec<UtteranceContent>) {
    items.retain(|item| !matches!(item, UtteranceContent::InternalBullet(_)));

    for item in items.iter_mut() {
        match item {
            UtteranceContent::Word(w) => {
                w.inline_bullet = None;
            }
            UtteranceContent::AnnotatedWord(aw) => {
                aw.inner.inline_bullet = None;
            }
            UtteranceContent::ReplacedWord(rw) => {
                rw.word.inline_bullet = None;
            }
            UtteranceContent::Group(g) => {
                strip_timing_from_bracketed(&mut g.content.content.0);
            }
            UtteranceContent::AnnotatedGroup(ag) => {
                strip_timing_from_bracketed(&mut ag.inner.content.content.0);
            }
            _ => {}
        }
    }
}

/// Remove parsed internal bullet tokens while preserving `Word.inline_bullet`.
///
/// This is used by the cheap rerun path after `%wor` timing is copied back to
/// main-tier words. Without this cleanup the serializer would emit both the
/// old parsed bullet tokens and the refreshed word-level bullets.
fn strip_internal_bullet_tokens(items: &mut Vec<UtteranceContent>) {
    items.retain(|item| !matches!(item, UtteranceContent::InternalBullet(_)));

    for item in items.iter_mut() {
        match item {
            UtteranceContent::Group(group) => {
                strip_internal_bullet_tokens_bracketed(&mut group.content.content.0);
            }
            UtteranceContent::AnnotatedGroup(group) => {
                strip_internal_bullet_tokens_bracketed(&mut group.inner.content.content.0);
            }
            _ => {}
        }
    }
}

fn strip_internal_bullet_tokens_bracketed(items: &mut Vec<BracketedItem>) {
    items.retain(|item| !matches!(item, BracketedItem::InternalBullet(_)));

    for item in items.iter_mut() {
        match item {
            BracketedItem::AnnotatedGroup(group) => {
                strip_internal_bullet_tokens_bracketed(&mut group.inner.content.content.0);
            }
            BracketedItem::PhoGroup(group) => {
                strip_internal_bullet_tokens_bracketed(&mut group.content.content.0);
            }
            BracketedItem::SinGroup(group) => {
                strip_internal_bullet_tokens_bracketed(&mut group.content.content.0);
            }
            BracketedItem::Quotation(group) => {
                strip_internal_bullet_tokens_bracketed(&mut group.content.content.0);
            }
            _ => {}
        }
    }
}

fn strip_timing_from_bracketed(items: &mut Vec<BracketedItem>) {
    items.retain(|item| !matches!(item, BracketedItem::InternalBullet(_)));

    for item in items.iter_mut() {
        match item {
            BracketedItem::Word(w) => {
                w.inline_bullet = None;
            }
            BracketedItem::AnnotatedWord(aw) => {
                aw.inner.inline_bullet = None;
            }
            BracketedItem::AnnotatedGroup(ag) => {
                strip_timing_from_bracketed(&mut ag.inner.content.content.0);
            }
            _ => {}
        }
    }
}

/// Collect a flat timing vector for main-tier Wor-alignable words by aligning
/// the existing `%wor` tier back onto the main tier.
fn collect_wor_backed_timings(utterance: &Utterance) -> Option<Vec<Option<WordTiming>>> {
    let wor = utterance.wor_tier()?.clone();
    let alignment = align_main_to_wor(&utterance.main, &wor);
    if !alignment.is_error_free() {
        return None;
    }

    let main_word_count = count_alignable_main_words(utterance);
    let wor_words: Vec<_> = wor.words().collect();
    let mut timings = vec![None; main_word_count];

    for pair in &alignment.pairs {
        let (Some(main_idx), Some(wor_idx)) = (pair.source_index, pair.target_index) else {
            return None;
        };
        if main_idx >= main_word_count || wor_idx >= wor_words.len() {
            return None;
        }
        let bullet = wor_words[wor_idx].inline_bullet.as_ref()?;
        timings[main_idx] = Some(WordTiming::new(
            bullet.timing.start_ms,
            bullet.timing.end_ms,
        ));
    }

    Some(timings)
}

/// Strip timing and %wor from a single utterance.
pub(super) fn strip_utterance_timing(utt: &mut Utterance) {
    utt.main.content.bullet = None;
    strip_timing_from_content(&mut utt.main.content.content.0);
    // Remove %wor tiers.
    utt.dependent_tiers
        .retain(|t| !matches!(t, DependentTier::Wor(_)));
}
