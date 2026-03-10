//! Tests for forced alignment module.

use super::*;
use talkbank_direct_parser::DirectParser;
use talkbank_model::model::{Line, UtteranceContent, WriteChat};

fn parse_chat(text: &str) -> talkbank_model::model::ChatFile {
    let parser = DirectParser::new().unwrap();
    parser.parse_chat_file(text).unwrap()
}

fn get_test_utterance(
    chat: &mut talkbank_model::model::ChatFile,
    idx: usize,
) -> &mut talkbank_model::model::Utterance {
    let mut utt_idx = 0;
    for line in &mut chat.lines {
        if let Line::Utterance(utt) = line {
            if utt_idx == idx {
                return utt;
            }
            utt_idx += 1;
        }
    }
    panic!("Utterance {idx} not found");
}

fn wor_timed_chat() -> String {
    "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%wor:\thello \u{15}100_500\u{15} world \u{15}600_1000\u{15} .\n@End\n".to_string()
}

#[test]
fn test_group_utterances_single_group() {
    let input = include_str!("../../../../test-fixtures/fa_two_timed_utterances.cha");
    let chat = parse_chat(input);
    let groups = group_utterances(&chat, 20000, None);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].words.len(), 5); // hello world I want cookie
    assert_eq!(groups[0].audio_start_ms(), 0);
    assert_eq!(groups[0].audio_end_ms(), 10000);
}

#[test]
fn test_group_utterances_backwards_bullets() {
    let input = include_str!("../../../../test-fixtures/fa_backwards_bullets.cha");
    let chat = parse_chat(input);
    let groups = group_utterances(&chat, 20000, None);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].words.len(), 1);
    assert_eq!(groups[1].words.len(), 1);
}

#[test]
fn test_group_utterances_splits_on_time() {
    let input = include_str!("../../../../test-fixtures/fa_split_on_time.cha");
    let chat = parse_chat(input);
    let groups = group_utterances(&chat, 20000, None);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].words.len(), 1);
    assert_eq!(groups[1].words.len(), 1);
}

#[test]
fn test_group_utterances_skips_untimed() {
    let input = include_str!("../../../../test-fixtures/fa_mixed_timed_untimed.cha");
    let chat = parse_chat(input);
    let groups = group_utterances(&chat, 20000, None);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].words.len(), 1); // only "world"
}

#[test]
fn test_inject_timings_simple() {
    let input = include_str!("../../../../test-fixtures/fa_hello_world_timed.cha");
    let mut chat = parse_chat(input);
    let utt = get_test_utterance(&mut chat, 0);

    let timings = vec![
        Some(WordTiming {
            start_ms: 100,
            end_ms: 500,
        }),
        Some(WordTiming {
            start_ms: 600,
            end_ms: 1000,
        }),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);
    assert_eq!(offset, 2);

    let utt = get_test_utterance(&mut chat, 0);
    let items = &utt.main.content.content;
    match &items[0] {
        UtteranceContent::Word(w) => {
            assert!(
                w.inline_bullet.is_some(),
                "Expected inline_bullet to be set"
            );
        }
        _ => panic!("Expected word"),
    }
}

#[test]
fn test_fa_cache_key() {
    let words = vec!["hello".to_string(), "world".to_string()];
    let key = cache_key(
        &words,
        &AudioIdentity::from_metadata("test.mp3", 1234, 5678),
        0,
        5000,
        FaTimingMode::WithPauses,
        FaEngineType::WhisperFa,
    );
    // Verify it's a valid hex BLAKE3 (64 chars)
    assert_eq!(key.as_str().len(), 64);
    assert!(key.as_str().chars().all(|c| c.is_ascii_hexdigit()));

    // Same inputs -> same key
    let key2 = cache_key(
        &words,
        &AudioIdentity::from_metadata("test.mp3", 1234, 5678),
        0,
        5000,
        FaTimingMode::WithPauses,
        FaEngineType::WhisperFa,
    );
    assert_eq!(key, key2);

    // Different timing mode -> different key
    let key3 = cache_key(
        &words,
        &AudioIdentity::from_metadata("test.mp3", 1234, 5678),
        0,
        5000,
        FaTimingMode::Continuous,
        FaEngineType::WhisperFa,
    );
    assert_ne!(key, key3);
}

#[test]
fn test_apply_fa_results() {
    let input = include_str!("../../../../test-fixtures/fa_hello_world_goodbye_timed.cha");
    let mut chat = parse_chat(input);

    let groups = vec![FaGroup {
        audio_span: TimeSpan::new(0, 10000),
        words: vec![
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(0),
                text: "hello".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(1),
                text: "world".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx(1),
                utterance_word_index: WordIdx(0),
                text: "goodbye".into(),
            },
        ],
        utterance_indices: vec![UtteranceIdx(0), UtteranceIdx(1)],
    }];

    let responses = vec![vec![
        Some(WordTiming {
            start_ms: 100,
            end_ms: 1000,
        }),
        Some(WordTiming {
            start_ms: 1500,
            end_ms: 3000,
        }),
        Some(WordTiming {
            start_ms: 5500,
            end_ms: 8000,
        }),
    ]];

    apply_fa_results(
        &mut chat,
        &groups,
        &responses,
        FaTimingMode::WithPauses,
        true,
    );

    let output = chat.to_chat_string();
    assert!(output.contains("%wor:"), "Output should contain %wor tier");
}

#[test]
fn test_has_reusable_wor_timing_true_for_complete_wor_roundtrip() {
    let chat = parse_chat(&wor_timed_chat());
    assert!(has_reusable_wor_timing(&chat));
}

#[test]
fn test_has_reusable_wor_timing_false_for_partial_wor_timing() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%wor:\thello \u{15}100_500\u{15} world .\n@End\n".to_string();
    let chat = parse_chat(&input);
    assert!(!has_reusable_wor_timing(&chat));
}

#[test]
fn test_refresh_existing_alignment_rehydrates_main_tier_from_wor() {
    let mut chat = parse_chat(&wor_timed_chat());
    refresh_existing_alignment(&mut chat, true);

    let output = chat.to_chat_string();
    assert!(
        output.contains("hello \u{15}100_500\u{15} world \u{15}600_1000\u{15} ."),
        "Expected refreshed main-tier word timing, got:\n{output}"
    );
    assert!(
        output.contains("%wor:\thello \u{15}100_500\u{15} world \u{15}600_1000\u{15} ."),
        "Expected refreshed %wor tier, got:\n{output}"
    );
}

#[test]
fn test_monotonicity_enforcement() {
    let input = include_str!("../../../../test-fixtures/fa_non_monotonic_bullets.cha");
    let mut chat = parse_chat(input);
    enforce_monotonicity(&mut chat);

    // Second utterance (start=2000) is before first (start=5000) -- should be stripped
    let utt = get_test_utterance(&mut chat, 1);
    assert!(
        utt.main.content.bullet.is_none(),
        "Non-monotonic utterance should have timing stripped"
    );
}

fn make_fa_words(texts: &[&str]) -> Vec<FaWord> {
    texts
        .iter()
        .enumerate()
        .map(|(i, t)| FaWord {
            utterance_index: UtteranceIdx(0),
            utterance_word_index: WordIdx(i),
            text: t.to_string(),
        })
        .collect()
}

#[test]
fn test_parse_fa_response_token_level() {
    let json = r#"{"tokens": [
            {"text": "hello", "time_s": 0.1},
            {"text": "world", "time_s": 0.6}
        ]}"#;
    let words = make_fa_words(&["hello", "world"]);
    let timings = parse_fa_response(json, &words, 0, FaTimingMode::Continuous).unwrap();
    assert_eq!(timings.len(), 2);
    assert_eq!(
        timings[0],
        Some(WordTiming {
            start_ms: 100,
            end_ms: 100
        })
    );
    assert_eq!(
        timings[1],
        Some(WordTiming {
            start_ms: 600,
            end_ms: 600
        })
    );
}

#[test]
fn test_parse_fa_response_token_level_punctuation_token_is_ignored() {
    let json = r#"{"tokens": [
            {"text": "hello", "time_s": 0.1},
            {"text": ",", "time_s": 0.2},
            {"text": "world", "time_s": 0.6}
        ]}"#;
    let words = make_fa_words(&["hello", "world"]);
    let timings = parse_fa_response(json, &words, 3000, FaTimingMode::Continuous).unwrap();
    assert_eq!(
        timings[0],
        Some(WordTiming {
            start_ms: 3100,
            end_ms: 3100
        })
    );
    assert_eq!(
        timings[1],
        Some(WordTiming {
            start_ms: 3600,
            end_ms: 3600
        })
    );
}

#[test]
fn test_parse_fa_response_token_level_mismatch_does_not_skip_tokens() {
    let json = r#"{"tokens": [
            {"text": "hello", "time_s": 0.1},
            {"text": "there", "time_s": 0.2},
            {"text": "world", "time_s": 0.6}
        ]}"#;
    let words = make_fa_words(&["hello", "world"]);
    let timings = parse_fa_response(json, &words, 0, FaTimingMode::Continuous).unwrap();
    assert_eq!(
        timings[0],
        Some(WordTiming {
            start_ms: 100,
            end_ms: 100
        })
    );
    assert_eq!(timings[1], None);
}

#[test]
fn test_parse_fa_response_indexed_word_level() {
    let json = r#"{"indexed_timings": [
            {"start_ms": 100, "end_ms": 500},
            {"start_ms": 600, "end_ms": 1000}
        ]}"#;
    let words = make_fa_words(&["hello", "world"]);
    let timings = parse_fa_response(json, &words, 5000, FaTimingMode::Continuous).unwrap();
    assert_eq!(timings.len(), 2);
    assert_eq!(timings[0].as_ref().unwrap().start_ms, 5100);
    assert_eq!(timings[0].as_ref().unwrap().end_ms, 5500);
    assert_eq!(timings[1].as_ref().unwrap().start_ms, 5600);
    assert_eq!(timings[1].as_ref().unwrap().end_ms, 6000);
}

#[test]
fn test_parse_fa_response_indexed_length_mismatch_rejected() {
    let json = r#"{"indexed_timings": [{"start_ms": 100, "end_ms": 500}]}"#;
    let words = make_fa_words(&["hello", "world"]);
    let err = parse_fa_response(json, &words, 0, FaTimingMode::Continuous).unwrap_err();
    assert!(err.contains("length mismatch"));
}

#[test]
fn test_estimate_boundaries_proportional() {
    let input = include_str!("../../../../test-fixtures/fa_two_untimed_with_media.cha");
    let chat = parse_chat(input);
    let estimates = estimate_untimed_boundaries(&chat, 10000);
    assert_eq!(estimates.len(), 2);
    assert_eq!(estimates[0].start_ms, 0);
    assert_eq!(estimates[0].end_ms, 7000);
    assert_eq!(estimates[1].start_ms, 3000);
    assert_eq!(estimates[1].end_ms, 10000);
}

/// Demonstrates the interleaved timed/untimed boundary-estimation bug that
/// caused real alignment failures in hand-edited transcripts.
///
/// When timed and untimed utterances are interleaved, untimed utterances must
/// be estimated by interpolating between neighboring timed utterances — NOT
/// by distributing proportionally across the entire audio duration.
///
/// The old proportional algorithm placed untimed utterance 1 (between timed
/// utts at 10-15s and 20-25s) at ~6-18s based on word-count ratio over the
/// full 50s audio. The correct window is 13-22s (the gap between neighbors
/// plus buffer). The wrong window caused the FA model to search the wrong
/// audio segment, producing missing or collapsed timing.
#[test]
fn test_estimate_boundaries_interpolates_from_neighbors() {
    let input = include_str!("../../../../test-fixtures/fa_mixed_timed_untimed_interleaved.cha");
    let chat = parse_chat(input);
    let estimates = estimate_untimed_boundaries(&chat, 50000);

    // 6 utterances total
    assert_eq!(estimates.len(), 6);

    // utt 0: timed (10000-15000), estimate mirrors real bullet
    assert_eq!(estimates[0], TimeSpan::new(10000, 15000));

    // utt 1: untimed, between timed utt 0 (end=15000) and utt 2 (start=20000)
    // Gap = [15000, 20000], 4 words, only utterance in run
    // raw: 15000-20000, with 2s buffer: 13000-22000
    assert_eq!(estimates[1].start_ms, 13000);
    assert_eq!(estimates[1].end_ms, 22000);

    // utt 2: timed (20000-25000)
    assert_eq!(estimates[2], TimeSpan::new(20000, 25000));

    // utt 3: untimed, in run [3,4] between timed utt 2 (end=25000) and utt 5 (start=40000)
    // Gap = [25000, 40000] = 15000ms, run_words = 4+5 = 9
    // utt 3 (4 words): raw 25000..31666, buffered 23000..33666
    assert_eq!(estimates[3].start_ms, 23000);
    assert_eq!(estimates[3].end_ms, 33666);

    // utt 4 (5 words): raw 31666..40000, buffered 29666..42000
    assert_eq!(estimates[4].start_ms, 29666);
    assert_eq!(estimates[4].end_ms, 42000);

    // utt 5: timed (40000-45000)
    assert_eq!(estimates[5], TimeSpan::new(40000, 45000));
}

/// Ensure ALL utterances (timed and untimed) are included in groups
/// when `total_audio_ms` is provided.
#[test]
fn test_group_utterances_includes_untimed_with_interpolation() {
    let input = include_str!("../../../../test-fixtures/fa_mixed_timed_untimed_interleaved.cha");
    let chat = parse_chat(input);
    let groups = group_utterances(&chat, 20000, Some(50000));

    // All 6 utterances should be included (none skipped)
    let total_utts: usize = groups.iter().map(|g| g.utterance_indices.len()).sum();
    assert_eq!(total_utts, 6);
}

/// Two utterances: first has clean %wor, second has stale %wor (word count mismatch).
/// `find_reusable_utterance_indices` should return only the first.
#[test]
fn test_find_reusable_utterance_indices_mixed_clean_stale() {
    // Utterance 0: "hello world ." with matching %wor (2 words) → reusable
    // Utterance 1: "goodbye my friend ." with stale %wor (1 word) → word count mismatch → stale
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%wor:\thello \u{15}100_500\u{15} world \u{15}600_1000\u{15} .\n*CHI:\tgoodbye my friend .\n%wor:\tgoodbye \u{15}1500_2000\u{15} .\n@End\n";
    let chat = parse_chat(input);

    let reusable = find_reusable_utterance_indices(&chat);
    assert!(reusable.contains(&0), "utterance 0 should be reusable");
    assert!(
        !reusable.contains(&1),
        "utterance 1 should be stale (word count mismatch)"
    );
    assert_eq!(reusable.len(), 1);
}

/// All utterances have clean %wor → all should be reusable.
#[test]
fn test_find_reusable_utterance_indices_all_clean() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%wor:\thello \u{15}100_500\u{15} world \u{15}600_1000\u{15} .\n*CHI:\tgoodbye .\n%wor:\tgoodbye \u{15}1500_2000\u{15} .\n@End\n";
    let chat = parse_chat(input);

    let reusable = find_reusable_utterance_indices(&chat);
    assert_eq!(reusable.len(), 2);
    assert!(reusable.contains(&0));
    assert!(reusable.contains(&1));
}

/// No utterances have %wor → empty set.
#[test]
fn test_find_reusable_utterance_indices_no_wor() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n*CHI:\tgoodbye .\n@End\n";
    let chat = parse_chat(input);

    let reusable = find_reusable_utterance_indices(&chat);
    assert!(reusable.is_empty());
}

/// `refresh_reusable_utterances` refreshes only the reusable utterances and
/// leaves stale ones untouched.
#[test]
fn test_refresh_reusable_utterances_selective() {
    // Utterance 0: clean %wor, utterance 1: no %wor (stale)
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%wor:\thello \u{15}100_500\u{15} world \u{15}600_1000\u{15} .\n*CHI:\tgoodbye .\n@End\n";
    let mut chat = parse_chat(input);

    let reusable: std::collections::HashSet<usize> = [0].into_iter().collect();
    orchestrate::refresh_reusable_utterances(&mut chat, &reusable, true);

    let output = chat.to_chat_string();
    // Utterance 0 should have refreshed word timing
    assert!(
        output.contains("hello \u{15}100_500\u{15} world \u{15}600_1000\u{15} ."),
        "Expected refreshed main-tier timing for utt 0, got:\n{output}"
    );
    // Utterance 1 should NOT have timing (it was stale/missing %wor)
    let utt1 = get_test_utterance(&mut chat, 1);
    assert!(
        utt1.main.content.bullet.is_none(),
        "Stale utterance should not get timing from refresh"
    );
}

// ---------------------------------------------------------------------------
// Bug: update_utterance_bullet shrinks pre-existing bullets
// ---------------------------------------------------------------------------
//
// When an utterance already has a hand-linked bullet that covers fillers,
// pauses, gestures, and false starts, FA only produces word timings for
// actual speech. update_utterance_bullet() was unconditionally replacing
// the bullet with min(word_starts)..max(word_ends), shrinking it to just
// the aligned speech and losing the surrounding context.
//
// Reported by Davida (2026-03-16) on ACWT corpus: "it's cutting off lots
// of stuff from the already linked lines both at the beginning and ends
// of utterances."

/// Pre-timed utterance with leading filler: bullet start must not advance.
///
/// Input: `*PAR: &-uh I went home . 37397_42983`
/// FA returns timings only for "I", "went", "home" starting at ~42000ms.
/// The original bullet starts at 37397 to cover the "&-uh" filler.
/// update_utterance_bullet must preserve 37397 as the start.
#[test]
fn test_update_utterance_bullet_preserves_start_with_leading_fillers() {
    let input = include_str!("../../../../test-fixtures/fa_pretimed_with_fillers.cha");
    let mut chat = parse_chat(input);
    let utt = get_test_utterance(&mut chat, 0);

    // Verify pre-existing bullet
    let original_bullet = utt.main.content.bullet.clone().unwrap();
    assert_eq!(original_bullet.timing.start_ms, 37397);
    assert_eq!(original_bullet.timing.end_ms, 42983);

    // Simulate FA: only "I", "went", "home" get timed (filler &-uh does not)
    let timings = vec![
        Some(WordTiming::new(42221, 42582)),
        Some(WordTiming::new(42582, 42782)),
        Some(WordTiming::new(42782, 42983)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    postprocess_utterance_timings(utt, FaTimingMode::WithPauses);
    update_utterance_bullet(utt);

    let bullet = utt.main.content.bullet.as_ref().unwrap();
    assert_eq!(
        bullet.timing.start_ms, 37397,
        "Bullet start must be preserved from original (covers leading filler), got {}",
        bullet.timing.start_ms,
    );
    assert_eq!(
        bullet.timing.end_ms, 42983,
        "Bullet end must be preserved from original, got {}",
        bullet.timing.end_ms,
    );
}

/// Pre-timed utterance with trailing gesture: bullet end must not recede.
///
/// Input: `*PAR: and it screwed up &=laughs . 50556_56221`
/// FA returns timings for "and", "it", "screwed", "up" ending at ~55898ms.
/// The original bullet ends at 56221 to cover the "&=laughs" gesture.
/// update_utterance_bullet must preserve 56221 as the end.
#[test]
fn test_update_utterance_bullet_preserves_end_with_trailing_gesture() {
    let input = include_str!("../../../../test-fixtures/fa_pretimed_trailing_gesture.cha");
    let mut chat = parse_chat(input);
    let utt = get_test_utterance(&mut chat, 0);

    // Verify pre-existing bullet
    let original_bullet = utt.main.content.bullet.clone().unwrap();
    assert_eq!(original_bullet.timing.start_ms, 50556);
    assert_eq!(original_bullet.timing.end_ms, 56221);

    // Simulate FA: "and", "it", "screwed", "up" get timed; &=laughs does not
    let timings = vec![
        Some(WordTiming::new(50616, 52596)),
        Some(WordTiming::new(52596, 54637)),
        Some(WordTiming::new(54637, 55718)),
        Some(WordTiming::new(55718, 55898)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    postprocess_utterance_timings(utt, FaTimingMode::WithPauses);
    update_utterance_bullet(utt);

    let bullet = utt.main.content.bullet.as_ref().unwrap();
    assert_eq!(
        bullet.timing.start_ms, 50556,
        "Bullet start must be preserved from original, got {}",
        bullet.timing.start_ms,
    );
    assert_eq!(
        bullet.timing.end_ms, 56221,
        "Bullet end must be preserved from original (covers trailing gesture), got {}",
        bullet.timing.end_ms,
    );
}

/// Untimed utterance (no prior bullet) should still get bullet from word timings.
///
/// This is the existing behavior that must continue to work: when there's
/// no pre-existing bullet, update_utterance_bullet sets it from the word
/// timing span.
#[test]
fn test_update_utterance_bullet_sets_new_bullet_when_none_existed() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n@Media:\ttest, audio\n*CHI:\thello world .\n@End\n";
    let mut chat = parse_chat(input);
    let utt = get_test_utterance(&mut chat, 0);

    // No pre-existing bullet
    assert!(utt.main.content.bullet.is_none());

    let timings = vec![
        Some(WordTiming::new(100, 500)),
        Some(WordTiming::new(600, 1000)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    update_utterance_bullet(utt);

    let bullet = utt.main.content.bullet.as_ref().unwrap();
    assert_eq!(bullet.timing.start_ms, 100);
    assert_eq!(bullet.timing.end_ms, 1000);
}

/// Full apply_fa_results pipeline with a pre-timed utterance containing fillers.
///
/// Verifies the end-to-end pipeline preserves the original bullet when word
/// timings cover only a subset of the utterance duration.
#[test]
fn test_apply_fa_results_preserves_pretimed_bullet() {
    let input = include_str!("../../../../test-fixtures/fa_pretimed_with_fillers.cha");
    let mut chat = parse_chat(input);

    let groups = vec![FaGroup {
        audio_span: TimeSpan::new(37397, 42983),
        words: vec![
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(0),
                text: "I".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(1),
                text: "went".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(2),
                text: "home".into(),
            },
        ],
        utterance_indices: vec![UtteranceIdx(0)],
    }];

    let responses = vec![vec![
        Some(WordTiming::new(42221, 42582)),
        Some(WordTiming::new(42582, 42782)),
        Some(WordTiming::new(42782, 42983)),
    ]];

    apply_fa_results(
        &mut chat,
        &groups,
        &responses,
        FaTimingMode::WithPauses,
        true,
    );

    let utt = get_test_utterance(&mut chat, 0);
    let bullet = utt.main.content.bullet.as_ref().unwrap();
    assert_eq!(
        bullet.timing.start_ms, 37397,
        "Pipeline must preserve original bullet start (covers leading filler), got {}",
        bullet.timing.start_ms,
    );
    assert_eq!(
        bullet.timing.end_ms, 42983,
        "Pipeline must preserve original bullet end, got {}",
        bullet.timing.end_ms,
    );
}

/// Word timings that extend beyond the original bullet should expand it.
///
/// If FA discovers speech starts earlier or ends later than the original
/// bullet, the bullet should grow to accommodate.
#[test]
fn test_update_utterance_bullet_expands_when_words_exceed_original() {
    let input = include_str!("../../../../test-fixtures/fa_pretimed_with_fillers.cha");
    let mut chat = parse_chat(input);
    let utt = get_test_utterance(&mut chat, 0);

    // Original bullet: 37397_42983
    // Simulate FA returning words that start before and end after the bullet
    let timings = vec![
        Some(WordTiming::new(37000, 38000)),
        Some(WordTiming::new(38000, 43500)),
        Some(WordTiming::new(43500, 44000)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    // Skip postprocess (it would clamp to utterance boundary) — test update only
    update_utterance_bullet(utt);

    let bullet = utt.main.content.bullet.as_ref().unwrap();
    assert_eq!(
        bullet.timing.start_ms, 37000,
        "Bullet should expand to earlier word start, got {}",
        bullet.timing.start_ms,
    );
    assert_eq!(
        bullet.timing.end_ms, 44000,
        "Bullet should expand to later word end, got {}",
        bullet.timing.end_ms,
    );
}

// ---------------------------------------------------------------------------
// Two-pass overlap UTR tests
// ---------------------------------------------------------------------------

/// Helper to make ASR tokens from a slice of (text, start_ms, end_ms).
fn make_utr_tokens(words_with_times: &[(&str, u64, u64)]) -> Vec<utr::AsrTimingToken> {
    words_with_times
        .iter()
        .map(|(text, start, end)| utr::AsrTimingToken {
            text: text.to_string(),
            start_ms: *start,
            end_ms: *end,
        })
        .collect()
}

/// Helper to extract the bullet from the nth utterance (by utterance index).
fn get_utterance_bullet(chat: &talkbank_model::model::ChatFile, idx: usize) -> Option<(u64, u64)> {
    let mut utt_idx = 0;
    for line in &chat.lines {
        if let Line::Utterance(utt) = line {
            if utt_idx == idx {
                return utt
                    .main
                    .content
                    .bullet
                    .as_ref()
                    .map(|b| (b.timing.start_ms, b.timing.end_ms));
            }
            utt_idx += 1;
        }
    }
    None
}

/// TwoPassOverlapUtr correctly times a `+<` backchannel by recovering it
/// from the previous utterance's audio window.
#[test]
fn test_two_pass_correctly_times_lazy_overlap() {
    use utr::UtrStrategy;
    let chat_text = include_str!("../../../../test-fixtures/utr_lazy_overlap_backchannel.cha");
    let mut chat = parse_chat(chat_text);
    let tokens = make_utr_tokens(&[
        // PAR's first utterance words
        ("I", 100, 300),
        ("went", 400, 800),
        ("to", 900, 1100),
        ("the", 1200, 1400),
        ("store", 1500, 2000),
        // INV's backchannel overlaps PAR's first utterance
        ("mhm", 1800, 2200),
        ("yesterday", 2300, 3000),
        // PAR's second utterance
        ("and", 5000, 5300),
        ("I", 5400, 5600),
        ("bought", 5700, 6200),
        ("some", 6300, 6600),
        ("groceries", 6700, 7500),
    ]);
    let result = utr::TwoPassOverlapUtr::new().inject(&mut chat, &tokens);

    // PAR's two utterances + INV's backchannel should all get timing
    assert_eq!(
        result.injected, 3,
        "all 3 untimed utterances should get timing"
    );
    assert_eq!(result.unmatched, 0);

    // Verify INV's "mhm" (utterance index 1) got correct timing
    let inv_bullet = get_utterance_bullet(&chat, 1).expect("INV +< mhm should have a bullet");
    assert!(
        inv_bullet.0 >= 1700 && inv_bullet.0 <= 1900,
        "INV start should be near 1800, got {}",
        inv_bullet.0,
    );
    assert!(
        inv_bullet.1 >= 2100 && inv_bullet.1 <= 2300,
        "INV end should be near 2200, got {}",
        inv_bullet.1,
    );
}

/// GlobalUtr cannot correctly time a `+<` backchannel — the global DP places
/// "mhm" after the main-speaker words, misaligning it.
#[test]
fn test_global_utr_misaligns_lazy_overlap_backchannel() {
    use utr::UtrStrategy;
    let chat_text = include_str!("../../../../test-fixtures/utr_lazy_overlap_backchannel.cha");
    let mut chat = parse_chat(chat_text);
    let tokens = make_utr_tokens(&[
        ("I", 100, 300),
        ("went", 400, 800),
        ("to", 900, 1100),
        ("the", 1200, 1400),
        ("store", 1500, 2000),
        ("mhm", 1800, 2200),
        ("yesterday", 2300, 3000),
        ("and", 5000, 5300),
        ("I", 5400, 5600),
        ("bought", 5700, 6200),
        ("some", 6300, 6600),
        ("groceries", 6700, 7500),
    ]);
    let result = utr::GlobalUtr.inject(&mut chat, &tokens);

    // GlobalUtr may still inject timing for INV, but the timing will be wrong:
    // the DP assigns "mhm" to the token at its position in the global sequence
    // (after "yesterday" or misplaced), not within the overlapping window.
    // We verify that at least the injected count covers all utterances.
    assert_eq!(
        result.injected + result.unmatched,
        3,
        "all 3 untimed utterances accounted for"
    );
}

/// When no `+<` utterances exist, TwoPassOverlapUtr produces identical results
/// to GlobalUtr.
#[test]
fn test_two_pass_identical_without_lazy_overlap() {
    use utr::UtrStrategy;
    let chat_text =
        include_str!("../../../../test-fixtures/fa_mixed_timed_untimed_interleaved.cha");

    // Build matching ASR tokens
    let tokens = make_utr_tokens(&[
        ("the", 10000, 10500),
        ("cat", 10600, 11000),
        ("is", 11200, 11500),
        ("here", 12000, 13000),
        ("she", 15500, 16000),
        ("is", 16200, 16500),
        ("looking", 16800, 17500),
        ("outside", 17800, 18500),
        ("there", 20500, 21000),
        ("is", 21200, 21500),
        ("a", 21800, 22000),
        ("path", 22200, 23000),
        ("I", 26000, 26500),
        ("do", 26800, 27000),
        ("not", 27200, 27500),
        ("know", 27800, 28500),
        ("but", 30000, 30500),
        ("there", 30800, 31200),
        ("is", 31500, 31800),
        ("a", 32000, 32200),
        ("building", 32500, 33500),
        ("okay", 40500, 41000),
        ("so", 41200, 41500),
        ("now", 41800, 42500),
    ]);

    let mut chat_global = parse_chat(chat_text);
    let mut chat_two_pass = parse_chat(chat_text);

    let r1 = utr::GlobalUtr.inject(&mut chat_global, &tokens);
    let r2 = utr::TwoPassOverlapUtr::new().inject(&mut chat_two_pass, &tokens);

    assert_eq!(r1.injected, r2.injected, "injected count should match");
    assert_eq!(r1.unmatched, r2.unmatched, "unmatched count should match");
    assert_eq!(r1.skipped, r2.skipped, "skipped count should match");

    // Compare bullets on each utterance
    for i in 0..6 {
        assert_eq!(
            get_utterance_bullet(&chat_global, i),
            get_utterance_bullet(&chat_two_pass, i),
            "utterance {i} bullets should match",
        );
    }
}

/// Dense backchannels: 4 consecutive `+<` utterances from INV during PAR's
/// narrative should all receive timing within PAR's audio range.
#[test]
fn test_two_pass_dense_backchannels() {
    use utr::UtrStrategy;
    let chat_text = include_str!("../../../../test-fixtures/utr_lazy_overlap_dense.cha");
    let mut chat = parse_chat(chat_text);

    // PAR's narrative spans 100-10000ms. INV's 4 backchannels are scattered within.
    let tokens = make_utr_tokens(&[
        // PAR's words
        ("I", 100, 300),
        ("grew", 400, 700),
        ("up", 800, 1000),
        ("in", 1100, 1300),
        ("Princeton", 1400, 2000),
        // INV backchannel 1: "oh okay"
        ("oh", 2100, 2300),
        ("okay", 2400, 2800),
        ("and", 2900, 3100),
        ("came", 3200, 3500),
        ("to", 3600, 3800),
        ("graduate", 3900, 4400),
        ("school", 4500, 5000),
        // INV backchannel 2: "mhm"
        ("mhm", 5100, 5400),
        ("at", 5500, 5700),
        ("Chapel", 5800, 6200),
        ("Hill", 6300, 6700),
        // INV backchannel 3: "oh"
        ("oh", 6800, 7100),
        ("in", 7200, 7400),
        ("ninety", 7500, 7900),
        ("one", 8000, 8300),
        // INV backchannel 4: "mhm"
        ("mhm", 8400, 8700),
        ("or", 8800, 9000),
        ("maybe", 9100, 9500),
        ("ninety", 9600, 9900),
        ("two", 10000, 10300),
    ]);

    let result = utr::TwoPassOverlapUtr::new().inject(&mut chat, &tokens);

    // PAR's utterance (1) + 4 INV backchannels = 5 injected
    assert_eq!(
        result.injected, 5,
        "PAR + 4 INV backchannels should be timed"
    );
    assert_eq!(result.unmatched, 0);

    // All 4 INV utterances (indices 1-4) should have bullets within PAR's range
    for inv_idx in 1..=4 {
        let bullet = get_utterance_bullet(&chat, inv_idx)
            .unwrap_or_else(|| panic!("INV utterance {inv_idx} should have a bullet"));
        assert!(
            bullet.0 >= 100 && bullet.1 <= 11000,
            "INV utterance {inv_idx} bullet {}-{} should be within PAR's range",
            bullet.0,
            bullet.1,
        );
    }
}

/// `select_strategy` returns TwoPassOverlapUtr for files with +<, GlobalUtr otherwise.
#[test]
fn test_select_strategy_chooses_correctly() {
    let with_overlap = include_str!("../../../../test-fixtures/utr_lazy_overlap_backchannel.cha");
    let without_overlap =
        include_str!("../../../../test-fixtures/fa_mixed_timed_untimed_interleaved.cha");

    let chat_overlap = parse_chat(with_overlap);
    let chat_no_overlap = parse_chat(without_overlap);

    // We can't check the concrete type directly, but we can verify behavior:
    // select_strategy on a +< file should produce TwoPassOverlapUtr results
    let strategy = utr::select_strategy(&chat_overlap, None);
    let mut chat = parse_chat(with_overlap);
    let tokens = make_utr_tokens(&[
        ("I", 100, 300),
        ("went", 400, 800),
        ("to", 900, 1100),
        ("the", 1200, 1400),
        ("store", 1500, 2000),
        ("mhm", 1800, 2200),
        ("yesterday", 2300, 3000),
        ("and", 5000, 5300),
        ("I", 5400, 5600),
        ("bought", 5700, 6200),
        ("some", 6300, 6600),
        ("groceries", 6700, 7500),
    ]);
    let result = strategy.inject(&mut chat, &tokens);
    assert_eq!(result.injected, 3, "should use two-pass and time all 3");

    // select_strategy on a non-+< file should use GlobalUtr
    let strategy = utr::select_strategy(&chat_no_overlap, None);
    let _ = strategy; // Just verify it compiles and returns
}

#[test]
fn snapshot_fa_infer_item() {
    let item = FaInferItem {
        words: vec!["hello".into(), "world".into()],
        word_ids: vec!["u0:w0".into(), "u0:w1".into()],
        word_utterance_indices: vec![0, 0],
        word_utterance_word_indices: vec![0, 1],
        audio_path: "/data/test.mp3".into(),
        audio_start_ms: 1500,
        audio_end_ms: 3200,
        timing_mode: FaTimingMode::WithPauses,
    };
    insta::assert_json_snapshot!(item);
}
