//! Tests for the `batchalign_core` PyO3 bridge.
//!
//! All test code from the original `lib.rs` lives here. Tests don't count
//! toward file size limits so this file can be monolithic.

use batchalign_chat_ops::fa::strip_e704_same_speaker_overlaps;
use batchalign_chat_ops::morphosyntax::{
    MorphosyntaxBatchItem, MultilingualPolicy, TokenizationMode,
    collect_payloads as collect_morphosyntax_payloads,
    inject_results as inject_morphosyntax_results,
};
use batchalign_chat_ops::utseg::UtsegResponse;
use talkbank_model::WriteChat;

use crate::test_helpers as t;
use crate::tier_ops::validate_user_tier_label;
use crate::{AsrWordJson, FaTimingsResponse, ParsedChat, TierEntryJson};

use crate::build::build_chat_inner;
use crate::parse::{
    errors_to_json, parse_lenient_with_warnings, parse_strict_pure, strip_timing_on_chat_file,
};

#[test]
fn test_strip_timing_removes_bullets_and_wor() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thello world . \u{0015}100_500\u{0015}
%wor:\thello \u{0015}100_300\u{0015} world \u{0015}300_500\u{0015}
@End
";
    let result = t::strip_timing(input).unwrap();
    // Bullets should be gone
    assert!(
        !result.contains('\u{0015}'),
        "timing bullets should be stripped"
    );
    // %wor tier should be gone
    assert!(!result.contains("%wor:"), "%wor tier should be stripped");
    // Content preserved
    assert!(result.contains("hello"));
    assert!(result.contains("world"));
    // Structure preserved
    assert!(result.contains("@UTF8"));
    assert!(result.contains("*CHI:"));
    assert!(result.contains("@End"));
}

#[test]
fn test_strip_timing_preserves_form_types() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tnice fiu@s:hun . \u{0015}100_500\u{0015}
@End
";
    let result = t::strip_timing(input).unwrap();
    // @s:hun should be preserved (not corrupted to @shun)
    assert!(
        result.contains("@s:hun"),
        "form type @s:hun must be preserved"
    );
    assert!(
        !result.contains('\u{0015}'),
        "timing bullets should be stripped"
    );
}

#[test]
fn test_parse_and_serialize_minimal() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thello .
@End
";
    let result = t::parse_and_serialize(input).unwrap();
    assert!(result.contains("@UTF8"));
    assert!(result.contains("*CHI:"));
    assert!(result.contains("hello"));
    assert!(result.contains("@End"));
}

#[test]
fn test_parse_error_returns_err() {
    // Empty string has no @UTF8/@Begin -- cannot produce a valid CHAT file
    let result = t::parse_and_serialize("");
    assert!(result.is_err());
}

#[test]
fn test_extract_nlp_words_mor() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thello world .
@End
";
    let result = t::extract_nlp_words(input, "mor").unwrap();
    assert!(result.contains("hello"));
    assert!(result.contains("world"));
    assert!(result.contains("\"speaker\":\"CHI\""));
    assert!(result.contains("\"word_id\":\"u0:w0\""));
    assert!(result.contains("\"utterance_word_index\":0"));
}

#[test]
fn test_extract_nlp_words_retrace_skipped_for_mor() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\t<I want> [/] I need cookie .
@End
";
    let result = t::extract_nlp_words(input, "mor").unwrap();
    // In mor domain, retraced words "I want" (inside <> [/]) should be skipped
    // Only "I", "need", "cookie" should appear
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
    let words: Vec<&str> = parsed[0]["words"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w["text"].as_str().unwrap())
        .collect();
    assert_eq!(words, vec!["I", "need", "cookie"]);
}

#[test]
fn test_extract_nlp_words_retrace_kept_for_wor() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\t<I want> [/] I need cookie .
@End
";
    let result = t::extract_nlp_words(input, "wor").unwrap();
    // In wor domain, retraced words ARE included (they were spoken)
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
    let words: Vec<&str> = parsed[0]["words"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w["text"].as_str().unwrap())
        .collect();
    assert_eq!(words, vec!["I", "want", "I", "need", "cookie"]);
}

#[test]
fn test_extract_nlp_words_word_ids_stable() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thello world .
*CHI:\tgood job .
@End
";
    let result = t::extract_nlp_words(input, "wor").unwrap();
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
    let ids_u0: Vec<&str> = parsed[0]["words"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w["word_id"].as_str().unwrap())
        .collect();
    let ids_u1: Vec<&str> = parsed[1]["words"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w["word_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids_u0, vec!["u0:w0", "u0:w1"]);
    assert_eq!(ids_u1, vec!["u1:w0", "u1:w1"]);
}

#[test]
fn test_extract_invalid_domain() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thello .
@End
";
    let result = t::extract_nlp_words(input, "invalid");
    assert!(result.is_err());
}

#[test]
fn test_extract_special_form_c() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tgumma@c is yummy .
@End
";
    let result = t::extract_nlp_words(input, "mor").unwrap();
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
    let words = parsed[0]["words"].as_array().unwrap();
    // First word "gumma" should have form_type "c"
    assert_eq!(words[0]["text"].as_str().unwrap(), "gumma");
    assert_eq!(words[0]["form_type"].as_str().unwrap(), "c");
    assert!(!words[0]["lang_marker"].as_bool().unwrap());
    // Other words should have null form_type
    assert_eq!(words[1]["text"].as_str().unwrap(), "is");
    assert!(words[1]["form_type"].is_null());
}

#[test]
fn test_extract_special_form_s() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thola@s is hi .
@End
";
    let result = t::extract_nlp_words(input, "mor").unwrap();
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
    let words = parsed[0]["words"].as_array().unwrap();
    assert_eq!(words[0]["text"].as_str().unwrap(), "hola");
    // @s words have lang_marker=true and no form_type
    assert!(words[0]["form_type"].is_null());
    assert!(words[0]["lang_marker"].as_bool().unwrap());
}

#[test]
fn test_extract_metadata_basic() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
@Media:\ttest_audio, audio
*CHI:\thello world .
@End
";
    let result = t::extract_metadata(input).unwrap();
    let val: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(val["langs"], serde_json::json!(["eng"]));
    assert_eq!(val["media_name"], "test_audio");
    assert_eq!(val["media_type"], "audio");
}

#[test]
fn test_extract_metadata_multi_lang() {
    let input = "\
@UTF8
@Begin
@Languages:\teng, spa
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thello world .
@End
";
    let result = t::extract_metadata(input).unwrap();
    let val: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(val["langs"], serde_json::json!(["eng", "spa"]));
    assert!(val["media_name"].is_null());
    assert!(val["media_type"].is_null());
}

#[test]
fn test_extract_metadata_no_languages_defaults_eng() {
    // Minimal CHAT with no @Languages header
    let input = "\
@UTF8
@Begin
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thello .
@End
";
    let result = t::extract_metadata(input).unwrap();
    let val: serde_json::Value = serde_json::from_str(&result).unwrap();
    // Empty langs array -- caller decides default
    assert_eq!(val["langs"], serde_json::json!([]));
}

#[test]
fn test_extract_metadata_video() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
@Media:\tmy_video, video
*CHI:\thello .
@End
";
    let result = t::extract_metadata(input).unwrap();
    let val: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(val["media_type"], "video");
}

#[test]
fn test_reassign_speakers_removes_old_id() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR0 Participant Participant
@ID:\teng|corpus|PAR0||male|||Participant|||
*PAR0:\thello . \u{0015}100_500\u{0015}
*PAR0:\tworld . \u{0015}1000_2000\u{0015}
@End
";
    let segments = r#"[
        {"start_ms": 0, "end_ms": 600, "speaker": "SPEAKER_0"},
        {"start_ms": 800, "end_ms": 2100, "speaker": "SPEAKER_1"}
    ]"#;
    let result = t::reassign_speakers(input, segments, "eng").unwrap();
    // Old PAR0 should not appear
    assert!(
        !result.contains("PAR0"),
        "old PAR0 should be removed: {}",
        result
    );
    // New PA0 and PA1 should appear
    assert!(result.contains("PA0"), "PA0 should be present: {}", result);
    assert!(result.contains("PA1"), "PA1 should be present: {}", result);
    // Both @ID lines should exist
    let id_count = result.matches("@ID:").count();
    assert_eq!(id_count, 2, "should have exactly 2 @ID lines: {}", result);
}

#[test]
fn test_add_utterance_timing() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tthe dog is big .
@End
";
    let asr_words = r#"[
        {"word": "the", "start_ms": 100, "end_ms": 200},
        {"word": "dog", "start_ms": 250, "end_ms": 400},
        {"word": "is", "start_ms": 450, "end_ms": 500},
        {"word": "big", "start_ms": 550, "end_ms": 700}
    ]"#;
    let result = t::add_utterance_timing(input, asr_words).unwrap();
    // Should have bullet markers
    assert!(
        result.contains("\u{0015}"),
        "should have bullet markers: {}",
        result
    );
    // Should have a %wor tier
    assert!(
        result.contains("%wor:"),
        "should have %%wor tier: {}",
        result
    );
    // Utterance should have overall timing
    assert!(
        result.contains("100_700"),
        "should have utterance timing: {}",
        result
    );
}

#[test]
fn test_add_utterance_timing_with_word_ids() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tthe dog is big .
@End
";
    let asr_words = r#"[
        {"word": "big", "start_ms": 550, "end_ms": 700, "word_id": "u0:w3"},
        {"word": "is", "start_ms": 450, "end_ms": 500, "word_id": "u0:w2"},
        {"word": "dog", "start_ms": 250, "end_ms": 400, "word_id": "u0:w1"},
        {"word": "the", "start_ms": 100, "end_ms": 200, "word_id": "u0:w0"}
    ]"#;
    let result = t::add_utterance_timing(input, asr_words).unwrap();
    assert!(result.contains("%wor:"));
    assert!(result.contains("the \u{0015}100_200\u{0015}"));
    assert!(result.contains("dog \u{0015}250_400\u{0015}"));
    assert!(result.contains("is \u{0015}450_500\u{0015}"));
    assert!(result.contains("big \u{0015}550_700\u{0015}"));
}

#[test]
fn test_add_utterance_timing_with_word_ids_repeated_words() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tthe the dog .
@End
";
    let asr_words = r#"[
        {"word": "the", "start_ms": 300, "end_ms": 400, "word_id": "u0:w1"},
        {"word": "dog", "start_ms": 500, "end_ms": 700, "word_id": "u0:w2"},
        {"word": "the", "start_ms": 100, "end_ms": 200, "word_id": "u0:w0"}
    ]"#;
    let result = t::add_utterance_timing(input, asr_words).unwrap();
    assert!(result.contains("the \u{0015}100_200\u{0015} the \u{0015}300_400\u{0015}"));
    assert!(result.contains("dog \u{0015}500_700\u{0015}"));
}

#[test]
fn test_add_utterance_timing_with_mixed_word_ids() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tthe dog is big .
@End
";
    let asr_words = r#"[
        {"word": "big", "start_ms": 550, "end_ms": 700, "word_id": "u0:w3"},
        {"word": "the", "start_ms": 100, "end_ms": 200, "word_id": "u0:w0"},
        {"word": "dog", "start_ms": 250, "end_ms": 400},
        {"word": "is", "start_ms": 450, "end_ms": 500}
    ]"#;
    let result = t::add_utterance_timing(input, asr_words).unwrap();
    assert!(result.contains("the \u{0015}100_200\u{0015}"));
    assert!(result.contains("dog \u{0015}250_400\u{0015}"));
    assert!(result.contains("is \u{0015}450_500\u{0015}"));
    assert!(result.contains("big \u{0015}550_700\u{0015}"));
}

#[test]
fn test_add_utterance_timing_invalid_word_id_falls_back_to_monotonic() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tthe dog is big .
@End
";
    let asr_words = r#"[
        {"word": "the", "start_ms": 100, "end_ms": 200, "word_id": "bad-id"},
        {"word": "dog", "start_ms": 250, "end_ms": 400},
        {"word": "is", "start_ms": 450, "end_ms": 500},
        {"word": "big", "start_ms": 550, "end_ms": 700}
    ]"#;
    let result = t::add_utterance_timing(input, asr_words).unwrap();
    assert!(result.contains("the \u{0015}100_200\u{0015}"));
    assert!(result.contains("dog \u{0015}250_400\u{0015}"));
    assert!(result.contains("is \u{0015}450_500\u{0015}"));
    assert!(result.contains("big \u{0015}550_700\u{0015}"));
}

#[test]
fn test_add_utterance_timing_no_window_monotonic_fallback() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child, MOT Mother
@ID:\teng|test|CHI||female|||Target_Child|||
@ID:\teng|test|MOT|||||Mother|||
*CHI:\talpha beta .
*MOT:\tgamma delta .
@End
";
    // No utterance bullets => no window constraints. Fallback should be
    // deterministic monotonic matching (no global DP remapping).
    let asr_words = r#"[
        {"word": "gamma", "start_ms": 1200, "end_ms": 1300},
        {"word": "delta", "start_ms": 1400, "end_ms": 1500},
        {"word": "alpha", "start_ms": 100, "end_ms": 200},
        {"word": "beta", "start_ms": 300, "end_ms": 400}
    ]"#;
    let result = t::add_utterance_timing(input, asr_words).unwrap();
    assert!(result.contains("alpha \u{0015}100_200\u{0015}"));
    assert!(result.contains("beta \u{0015}300_400\u{0015}"));
    assert!(!result.contains("gamma \u{0015}1200_1300\u{0015}"));
    assert!(!result.contains("delta \u{0015}1400_1500\u{0015}"));
}

#[test]
fn test_add_utterance_timing_window_constrained_fallback() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child, MOT Mother
@ID:\teng|test|CHI||female|||Target_Child|||
@ID:\teng|test|MOT|||||Mother|||
*CHI:\talpha beta . \u{0015}0_1000\u{0015}
*MOT:\tgamma delta . \u{0015}1000_2000\u{0015}
@End
";
    // Deliberately out-of-order across utterance windows: global DP can only
    // match one contiguous block, while window-constrained fallback should
    // recover all words.
    let asr_words = r#"[
        {"word": "gamma", "start_ms": 1200, "end_ms": 1300},
        {"word": "delta", "start_ms": 1400, "end_ms": 1500},
        {"word": "alpha", "start_ms": 100, "end_ms": 200},
        {"word": "beta", "start_ms": 300, "end_ms": 400}
    ]"#;
    let result = t::add_utterance_timing(input, asr_words).unwrap();
    assert!(result.contains("alpha \u{0015}100_200\u{0015}"));
    assert!(result.contains("beta \u{0015}300_400\u{0015}"));
    assert!(result.contains("gamma \u{0015}1200_1300\u{0015}"));
    assert!(result.contains("delta \u{0015}1400_1500\u{0015}"));
}

#[test]
fn test_add_utterance_timing_ambiguous_window_skips_global_dp() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\talpha . \u{0015}0_1000\u{0015}
*CHI:\tbeta . \u{0015}900_1900\u{0015}
@End
";
    // ASR timing overlaps both utterance windows (with tolerance), so no
    // unique window assignment exists. With window metadata present we
    // should not run global fallback.
    let asr_words = r#"[
        {"word": "beta", "start_ms": 950, "end_ms": 980}
    ]"#;
    let result = t::add_utterance_timing(input, asr_words).unwrap();
    assert!(
        !result.contains("beta \u{0015}950_980\u{0015}"),
        "ambiguous window should remain unassigned without global fallback: {}",
        result
    );
}

#[test]
fn test_e704_strips_overlapping_same_speaker() {
    // Two FAT utterances whose bullets overlap by >500ms.
    // The EARLIER one should have its timing stripped.
    let b = "\u{0015}";
    let input = format!(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tFAT Father
@ID:\teng|test|FAT|||||Father|||
*FAT:\tit's probably the +... {b}1000_2200{b}
%wor:\tit's {b}1000_1400{b} probably {b}1400_1800{b} the {b}1800_2200{b} +...
*FAT:\tit's called the great cat chase . {b}1500_3000{b}
%wor:\tit's {b}1500_1900{b} called {b}1900_2300{b} the {b}2300_2600{b} great {b}2600_2800{b} cat {b}2800_2900{b} chase {b}2900_3000{b} .
@End
"
    );
    let mut chat_file = parse_strict_pure(&input).unwrap();
    strip_e704_same_speaker_overlaps(&mut chat_file);
    let result = chat_file.to_chat_string();

    // First utterance should have timing stripped (it's the earlier one
    // whose end 2200 overlaps with next start 1500 by 700ms > 500ms).
    assert!(
        !result.contains("1000_2200"),
        "first utterance bullet should be stripped: {}",
        result
    );
    // Its %wor tier should be removed.
    assert!(
        result.contains("it's probably the"),
        "first utterance text preserved: {}",
        result
    );

    // Second utterance should keep its timing.
    assert!(
        result.contains("1500_3000"),
        "second utterance bullet should be preserved: {}",
        result
    );
    assert!(
        result.contains("%wor:"),
        "second utterance %wor preserved: {}",
        result
    );
}

#[test]
fn test_e704_no_strip_within_tolerance() {
    // Two FAT utterances that overlap by exactly 400ms (within 500ms tolerance).
    let b = "\u{0015}";
    let input = format!(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tFAT Father
@ID:\teng|test|FAT|||||Father|||
*FAT:\thello . {b}1000_1800{b}
%wor:\thello {b}1000_1800{b} .
*FAT:\tworld . {b}1400_2500{b}
%wor:\tworld {b}1400_2500{b} .
@End
"
    );
    let mut chat_file = parse_strict_pure(&input).unwrap();
    strip_e704_same_speaker_overlaps(&mut chat_file);
    let result = chat_file.to_chat_string();

    // Both bullets preserved -- overlap of 400ms is within 500ms tolerance.
    assert!(
        result.contains("1000_1800"),
        "first bullet preserved within tolerance: {}",
        result
    );
    assert!(
        result.contains("1400_2500"),
        "second bullet preserved within tolerance: {}",
        result
    );
}

#[test]
fn test_e704_cross_speaker_overlap_allowed() {
    // FAT and CHI overlapping is fine -- E704 only applies to same-speaker.
    let b = "\u{0015}";
    let input = format!(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tFAT Father, CHI Target_Child
@ID:\teng|test|FAT|||||Father|||
@ID:\teng|test|CHI||female|||Target_Child|||
*FAT:\thello . {b}1000_3000{b}
%wor:\thello {b}1000_3000{b} .
*CHI:\thi . {b}1500_2500{b}
%wor:\thi {b}1500_2500{b} .
@End
"
    );
    let mut chat_file = parse_strict_pure(&input).unwrap();
    strip_e704_same_speaker_overlaps(&mut chat_file);
    let result = chat_file.to_chat_string();

    // Both bullets preserved -- different speakers can overlap.
    assert!(
        result.contains("1000_3000"),
        "FAT bullet preserved (cross-speaker): {}",
        result
    );
    assert!(
        result.contains("1500_2500"),
        "CHI bullet preserved (cross-speaker): {}",
        result
    );
}

#[test]
fn test_parsed_chat_parse_and_serialize() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thello .
@End
";
    let handle = t::make_handle(input);
    let result = handle.inner.to_chat_string();
    assert!(result.contains("@UTF8"));
    assert!(result.contains("*CHI:"));
    assert!(result.contains("hello"));
    assert!(result.contains("@End"));
}

#[test]
fn test_parsed_chat_add_comment() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thello .
@End
";
    let mut handle = t::make_handle(input);
    t::add_comment(&mut handle, "This is a test comment");
    let result = handle.inner.to_chat_string();
    assert!(
        result.contains("@Comment:\tThis is a test comment"),
        "should contain comment: {}",
        result
    );
    // Comment should appear before the utterance
    let comment_pos = result.find("@Comment:").unwrap();
    let utt_pos = result.find("*CHI:").unwrap();
    assert!(
        comment_pos < utt_pos,
        "comment should appear before utterance"
    );
}

#[test]
fn test_parsed_chat_strip_timing() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thello world . \u{0015}100_500\u{0015}
%wor:\thello \u{0015}100_300\u{0015} world \u{0015}300_500\u{0015}
@End
";
    let mut handle = t::make_handle_lenient(input);
    strip_timing_on_chat_file(&mut handle.inner);
    let result = handle.inner.to_chat_string();
    assert!(
        !result.contains('\u{0015}'),
        "timing bullets should be stripped, got:\n{result}"
    );
    assert!(!result.contains("%wor:"), "%wor tier should be stripped");
}

#[test]
fn test_parsed_chat_build() {
    let json = r#"{
        "langs": ["eng"],
        "participants": [{"id": "PAR0", "name": "Participant", "role": "Participant"}],
        "utterances": [
            {
                "speaker": "PAR0",
                "words": [
                    {"text": "hello", "start_ms": 100, "end_ms": 500},
                    {"text": "."}
                ]
            }
        ]
    }"#;
    let handle = ParsedChat {
        inner: build_chat_inner(talkbank_model::model::Provenance::new(json.to_string())).unwrap(),
        warnings: vec![],
    };
    let result = handle.inner.to_chat_string();
    assert!(result.contains("hello"));
    assert!(result.contains("PAR0"));
}

#[test]
fn test_serde_fa_timings_response() {
    let json = r#"{"timings": [[100, 500], null, [600, 1000]]}"#;
    let resp: FaTimingsResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.timings.len(), 3);
    assert_eq!(resp.timings[0], Some([100, 500]));
    assert_eq!(resp.timings[1], None);
    assert_eq!(resp.timings[2], Some([600, 1000]));
}

#[test]
fn test_serde_segmentation_response() {
    let json = r#"{"assignments": [0, 0, 1, 1, 2]}"#;
    let resp: UtsegResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.assignments, vec![0, 0, 1, 1, 2]);
}

#[test]
fn test_serde_translation_response() {
    #[derive(serde::Deserialize)]
    struct TranslationResponse {
        translation: String,
    }
    let json = r#"{"translation": "Hello world"}"#;
    let resp: TranslationResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.translation, "Hello world");
}

#[test]
fn test_serde_asr_word_json() {
    let json = r#"[{"word": "hello", "start_ms": 100, "end_ms": 500}]"#;
    let words: Vec<AsrWordJson> = serde_json::from_str(json).unwrap();
    assert_eq!(words.len(), 1);
    assert_eq!(words[0].word, "hello");
    assert_eq!(words[0].start_ms, 100);
    assert_eq!(words[0].end_ms, 500);
    assert!(words[0].word_id.is_none());

    let json_with_id = r#"[{"word": "hello", "start_ms": 100, "end_ms": 500, "word_id": "u0:w0"}]"#;
    let with_id: Vec<AsrWordJson> = serde_json::from_str(json_with_id).unwrap();
    assert_eq!(with_id[0].word_id.as_deref(), Some("u0:w0"));
}

#[test]
fn test_serde_tier_entry_json() {
    let json = r#"[{"utterance_index": 0, "label": "xcoref", "content": "(1, -, 1)"}]"#;
    let entries: Vec<TierEntryJson> = serde_json::from_str(json).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].utterance_index, 0);
    assert_eq!(entries[0].label, "xcoref");
    assert_eq!(entries[0].content, "(1, -, 1)");
}

#[test]
fn test_validate_user_tier_label_accepts_x_prefix() {
    assert!(validate_user_tier_label("xcoref").is_ok());
    assert!(validate_user_tier_label("xfoo").is_ok());
    assert!(validate_user_tier_label("xtest").is_ok());
}

#[test]
fn test_validate_user_tier_label_rejects_standard() {
    assert!(validate_user_tier_label("mor").is_err());
    assert!(validate_user_tier_label("gra").is_err());
    assert!(validate_user_tier_label("wor").is_err());
    assert!(validate_user_tier_label("com").is_err());
}

#[test]
fn test_validate_user_tier_label_rejects_no_x_prefix() {
    assert!(validate_user_tier_label("coref").is_err());
    assert!(validate_user_tier_label("mytier").is_err());
}

#[test]
fn test_build_text_level_utterance() {
    let transcript = r#"{
        "langs": ["eng"],
        "participants": [{"id": "CHI", "name": "Child", "role": "Target_Child"}],
        "utterances": [
            {"speaker": "CHI", "text": "I want cookies .", "start_ms": 1000, "end_ms": 3000}
        ]
    }"#;
    let result = t::build_chat(transcript).unwrap();
    assert!(
        result.contains("*CHI:"),
        "should have speaker CHI: {}",
        result
    );
    assert!(
        result.contains("I want cookies"),
        "should contain text: {}",
        result
    );
    assert!(
        result.contains("\u{0015}1000_3000\u{0015}"),
        "should have timing bullet: {}",
        result
    );
    assert!(result.contains("@End"), "should have @End: {}", result);
}

#[test]
fn test_build_text_level_utterance_no_timing() {
    let transcript = r#"{
        "langs": ["eng"],
        "participants": [{"id": "CHI", "name": "Child", "role": "Target_Child"}],
        "utterances": [
            {"speaker": "CHI", "text": "hello world ."}
        ]
    }"#;
    let result = t::build_chat(transcript).unwrap();
    assert!(result.contains("*CHI:"), "should have speaker: {}", result);
    assert!(
        result.contains("hello world"),
        "should contain text: {}",
        result
    );
    assert!(
        !result.contains("\u{0015}"),
        "should NOT have timing bullet: {}",
        result
    );
}

#[test]
fn test_build_mixed_text_and_word_utterances() {
    let transcript = r#"{
        "langs": ["eng"],
        "participants": [
            {"id": "CHI", "name": "Child", "role": "Target_Child"},
            {"id": "MOT", "name": "Mother", "role": "Mother"}
        ],
        "utterances": [
            {"speaker": "CHI", "text": "hello .", "start_ms": 0, "end_ms": 500},
            {"speaker": "MOT", "words": [
                {"text": "hi", "start_ms": 600, "end_ms": 900},
                {"text": ".", "start_ms": null, "end_ms": null}
            ]}
        ]
    }"#;
    let result = t::build_chat(transcript).unwrap();
    assert!(result.contains("*CHI:"), "should have CHI: {}", result);
    assert!(result.contains("*MOT:"), "should have MOT: {}", result);
    assert!(result.contains("hello"), "should contain hello: {}", result);
    assert!(result.contains("hi"), "should contain hi: {}", result);
}

#[test]
fn test_build_text_level_question_terminator() {
    let transcript = r#"{
        "langs": ["eng"],
        "participants": [{"id": "CHI", "name": "Child", "role": "Target_Child"}],
        "utterances": [
            {"speaker": "CHI", "text": "is it good ?", "start_ms": 0, "end_ms": 1000}
        ]
    }"#;
    let result = t::build_chat(transcript).unwrap();
    assert!(
        result.contains("?"),
        "should have question mark: {}",
        result
    );
}

#[test]
fn test_batch_morphosyntax_payload_serialization() {
    let item = MorphosyntaxBatchItem {
        words: vec!["hello".to_string(), "world".to_string()],
        terminator: ".".to_string(),
        special_forms: vec![(None, None), (None, None)],
        lang: talkbank_model::model::LanguageCode::new("eng"),
    };
    let json = serde_json::to_string(&[&item]).unwrap();
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0]["words"].as_array().unwrap().len(), 2);
    assert_eq!(parsed[0]["terminator"].as_str().unwrap(), ".");
    assert_eq!(parsed[0]["lang"].as_str().unwrap(), "eng");
}

#[test]
fn test_batch_morphosyntax_payload_with_special_forms() {
    use talkbank_model::model::FormType;
    use talkbank_model::validation::LanguageResolution;
    let item = MorphosyntaxBatchItem {
        words: vec!["gumma".to_string(), "biberon".to_string()],
        terminator: ".".to_string(),
        special_forms: vec![
            (Some(FormType::C), None),
            (
                None,
                Some(LanguageResolution::Single(
                    talkbank_model::model::LanguageCode::new("spa"),
                )),
            ),
        ],
        lang: talkbank_model::model::LanguageCode::new("eng"),
    };
    let json = serde_json::to_string(&[&item]).unwrap();
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
    assert_eq!(
        parsed[0]["words"].as_array().unwrap()[0].as_str().unwrap(),
        "gumma"
    );
    assert_eq!(
        parsed[0]["words"].as_array().unwrap()[1].as_str().unwrap(),
        "biberon"
    );
    // special_forms: [[form_type, resolved_lang], [form_type, resolved_lang]]
    let sf = &parsed[0]["special_forms"];
    assert!(sf[0][0].is_string()); // First word has FormType::C
    assert!(sf[0][1].is_null()); // No language marker
    assert!(sf[1][0].is_null()); // Second word has no form type
    assert!(sf[1][1].is_object()); // Has resolved language (Single)
}

// -----------------------------------------------------------------------
// Tests: collect_morphosyntax_payloads (Phase 1)
// -----------------------------------------------------------------------

fn parse_chat(text: &str) -> talkbank_model::model::ChatFile {
    parse_strict_pure(text).unwrap()
}

fn lang(code: &str) -> talkbank_model::model::LanguageCode {
    talkbank_model::model::LanguageCode::new(code)
}

#[test]
fn test_collect_simple_words() {
    let chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tI eat cookies .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, total) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    assert_eq!(total, 1);
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].2.words, vec!["I", "eat", "cookies"]);
    assert_eq!(items[0].2.terminator, ".");
    assert_eq!(items[0].2.lang.as_str(), "eng");
}

#[test]
fn test_collect_retrace_excluded() {
    let chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\t<I want> [/] I need cookie .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, _) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    assert_eq!(items[0].2.words, vec!["I", "need", "cookie"]);
}

#[test]
fn test_collect_multi_utterance() {
    let chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child, MOT Mother
@ID:\teng|test|CHI||female|||Target_Child|||
@ID:\teng|test|MOT|||||Mother|||
*CHI:\tI eat cookies .
*MOT:\tgood job .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, total) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    assert_eq!(total, 2);
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].2.words, vec!["I", "eat", "cookies"]);
    assert_eq!(items[1].2.words, vec!["good", "job"]);
}

#[test]
fn test_collect_question_terminator() {
    let chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\twhat is that ?
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, _) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    assert_eq!(items[0].2.terminator, "?");
}

#[test]
fn test_collect_skipmultilang_filters() {
    let chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child, MOT Mother
@ID:\teng|test|CHI||female|||Target_Child|||
@ID:\teng|test|MOT|||||Mother|||
*CHI:\tI eat cookies .
*MOT:\t[- spa] hola amigo .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, total) = collect_morphosyntax_payloads(
        &chat,
        &primary,
        &declared,
        MultilingualPolicy::SkipNonPrimary,
    );
    assert_eq!(total, 2);
    assert_eq!(
        items.len(),
        1,
        "skipmultilang should exclude [- spa] utterance"
    );
    assert_eq!(items[0].2.words, vec!["I", "eat", "cookies"]);
}

#[test]
fn test_collect_skipmultilang_false_keeps_all() {
    let chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child, MOT Mother
@ID:\teng|test|CHI||female|||Target_Child|||
@ID:\teng|test|MOT|||||Mother|||
*CHI:\tI eat cookies .
*MOT:\t[- spa] hola amigo .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, _) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    assert_eq!(items.len(), 2);
}

#[test]
fn test_collect_special_form_c() {
    let chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tgumma@c is yummy .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, _) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    let sf = &items[0].2.special_forms;
    assert_eq!(sf[0].0, Some(talkbank_model::model::FormType::C));
    assert!(sf[0].1.is_none(), "@c should have no language resolution");
    assert!(sf[1].0.is_none(), "'is' should have no form type");
    assert!(sf[2].0.is_none(), "'yummy' should have no form type");
}

#[test]
fn test_collect_special_form_s_resolves_language() {
    let chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thola@s is hi .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, _) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    let sf = &items[0].2.special_forms;
    assert!(
        sf[0].0.is_none(),
        "@s is a language marker, not a form type"
    );
    assert!(sf[0].1.is_some(), "@s should have resolved language");
}

#[test]
fn test_collect_empty_chat() {
    let chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, total) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    assert_eq!(total, 0);
    assert!(items.is_empty());
}

#[test]
fn test_collect_utterance_lang_code() {
    let chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng, spa
@Participants:\tCHI Target_Child, MOT Mother
@ID:\teng|test|CHI||female|||Target_Child|||
@ID:\teng|test|MOT|||||Mother|||
*CHI:\thello .
*MOT:\t[- spa] hola .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng"), lang("spa")];
    let (items, _) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    assert_eq!(items[0].2.lang.as_str(), "eng");
    assert_eq!(items[1].2.lang.as_str(), "spa");
}

// -----------------------------------------------------------------------
// Tests: inject_morphosyntax_results (Phase 3)
// -----------------------------------------------------------------------

/// Build a minimal UdResponse from word data for testing injection.
///
/// Each tuple: (text, lemma, upos_str, head, deprel).
/// `upos_str` is deserialized from JSON to get proper `UdPunctable<UniversalPos>`.
fn make_ud_response(words: &[(&str, &str, &str, usize, &str)]) -> crate::nlp::UdResponse {
    let words_json: Vec<serde_json::Value> = words
        .iter()
        .enumerate()
        .map(|(i, word)| {
            let (text, lemma, upos, head, deprel) = *word;
            serde_json::json!({
                "id": i + 1,
                "text": text,
                "lemma": lemma,
                "upos": upos,
                "xpos": null,
                "feats": null,
                "head": head,
                "deprel": deprel,
                "deps": null,
                "misc": null
            })
        })
        .collect();

    serde_json::from_value(serde_json::json!({
        "sentences": [{
            "words": words_json
        }]
    }))
    .unwrap()
}

fn empty_ud_response() -> crate::nlp::UdResponse {
    crate::nlp::UdResponse { sentences: vec![] }
}

#[test]
fn test_inject_simple_morphology() {
    let mut chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tI eat cookies .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, _) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);

    let responses = vec![make_ud_response(&[
        ("I", "I", "PRON", 2, "nsubj"),
        ("eat", "eat", "VERB", 0, "root"),
        ("cookies", "cookie", "NOUN", 2, "obj"),
    ])];

    inject_morphosyntax_results(
        &mut chat,
        items,
        responses,
        &lang("eng"),
        TokenizationMode::Preserve,
        &std::collections::BTreeMap::new(),
    )
    .unwrap();
    let output = chat.to_chat_string();
    assert!(output.contains("%mor:"), "should have %mor tier");
    assert!(output.contains("%gra:"), "should have %gra tier");
    assert!(output.contains("pron|I"), "PRON->pron");
    assert!(output.contains("verb|eat"), "VERB->verb");
    assert!(output.contains("noun|cookie"), "NOUN->noun with lemma");
}

#[test]
fn test_inject_terminator_in_mor() {
    let mut chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tI eat cookies .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, _) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    let responses = vec![make_ud_response(&[
        ("I", "I", "PRON", 2, "nsubj"),
        ("eat", "eat", "VERB", 0, "root"),
        ("cookies", "cookie", "NOUN", 2, "obj"),
    ])];
    inject_morphosyntax_results(
        &mut chat,
        items,
        responses,
        &lang("eng"),
        TokenizationMode::Preserve,
        &std::collections::BTreeMap::new(),
    )
    .unwrap();
    let output = chat.to_chat_string();
    let mor_line: &str = output.lines().find(|l| l.starts_with("%mor:")).unwrap();
    assert!(
        mor_line.trim_end().ends_with('.'),
        "mor should end with period terminator"
    );
}

#[test]
fn test_inject_question_terminator() {
    let mut chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\twhat is that ?
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, _) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    let responses = vec![make_ud_response(&[
        ("what", "what", "PRON", 3, "nsubj"),
        ("is", "be", "AUX", 3, "cop"),
        ("that", "that", "PRON", 0, "root"),
    ])];
    inject_morphosyntax_results(
        &mut chat,
        items,
        responses,
        &lang("eng"),
        TokenizationMode::Preserve,
        &std::collections::BTreeMap::new(),
    )
    .unwrap();
    let output = chat.to_chat_string();
    let mor_line: &str = output.lines().find(|l| l.starts_with("%mor:")).unwrap();
    assert!(
        mor_line.trim_end().ends_with('?'),
        "mor should end with question mark"
    );
}

#[test]
fn test_inject_empty_response_no_tiers() {
    let mut chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tI eat cookies .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, _) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    let responses = vec![empty_ud_response()];
    inject_morphosyntax_results(
        &mut chat,
        items,
        responses,
        &lang("eng"),
        TokenizationMode::Preserve,
        &std::collections::BTreeMap::new(),
    )
    .unwrap();
    let output = chat.to_chat_string();
    assert!(
        !output.contains("%mor:"),
        "empty response should not add tiers"
    );
}

#[test]
fn test_inject_multi_utterance() {
    let mut chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child, MOT Mother
@ID:\teng|test|CHI||female|||Target_Child|||
@ID:\teng|test|MOT|||||Mother|||
*CHI:\tI eat cookies .
*MOT:\tgood job .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, _) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    let responses = vec![
        make_ud_response(&[
            ("I", "I", "PRON", 2, "nsubj"),
            ("eat", "eat", "VERB", 0, "root"),
            ("cookies", "cookie", "NOUN", 2, "obj"),
        ]),
        make_ud_response(&[
            ("good", "good", "ADJ", 2, "amod"),
            ("job", "job", "NOUN", 0, "root"),
        ]),
    ];
    inject_morphosyntax_results(
        &mut chat,
        items,
        responses,
        &lang("eng"),
        TokenizationMode::Preserve,
        &std::collections::BTreeMap::new(),
    )
    .unwrap();
    let output = chat.to_chat_string();
    let mor_count = output.lines().filter(|l| l.starts_with("%mor:")).count();
    assert_eq!(mor_count, 2, "both utterances should have %mor");
}

#[test]
fn test_inject_special_form_c_overrides_pos() {
    let mut chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tgumma@c is yummy .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, _) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    let responses = vec![make_ud_response(&[
        ("gumma", "gumma", "X", 2, "flat"),
        ("is", "be", "AUX", 0, "root"),
        ("yummy", "yummy", "ADJ", 2, "xcomp"),
    ])];
    inject_morphosyntax_results(
        &mut chat,
        items,
        responses,
        &lang("eng"),
        TokenizationMode::Preserve,
        &std::collections::BTreeMap::new(),
    )
    .unwrap();
    let output = chat.to_chat_string();
    assert!(
        output.contains("c|gumma"),
        "@c word should get c| POS tag, got: {}",
        output
    );
}

#[test]
fn test_inject_special_form_s_produces_l2_xxx() {
    let mut chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thola@s is hi .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, _) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    let responses = vec![make_ud_response(&[
        ("hola", "hola", "X", 0, "root"),
        ("is", "be", "AUX", 1, "cop"),
        ("hi", "hi", "INTJ", 1, "discourse"),
    ])];
    inject_morphosyntax_results(
        &mut chat,
        items,
        responses,
        &lang("eng"),
        TokenizationMode::Preserve,
        &std::collections::BTreeMap::new(),
    )
    .unwrap();
    let output = chat.to_chat_string();
    assert!(
        output.contains("L2|xxx"),
        "@s word should get L2|xxx, got: {}",
        output
    );
}

#[test]
fn test_inject_preserves_structure() {
    let mut chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tI eat cookies .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, _) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    let responses = vec![empty_ud_response()];
    inject_morphosyntax_results(
        &mut chat,
        items,
        responses,
        &lang("eng"),
        TokenizationMode::Preserve,
        &std::collections::BTreeMap::new(),
    )
    .unwrap();
    let output = chat.to_chat_string();
    assert!(output.contains("@UTF8"));
    assert!(output.contains("@Begin"));
    assert!(output.contains("@Languages:"));
    assert!(output.contains("*CHI:"));
    assert!(output.contains("I eat cookies"));
    assert!(output.contains("@End"));
}

#[test]
fn test_inject_retrace_preserved() {
    let mut chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\t<I want> [/] I need cookie .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, _) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    let responses = vec![make_ud_response(&[
        ("I", "I", "PRON", 2, "nsubj"),
        ("need", "need", "VERB", 0, "root"),
        ("cookie", "cookie", "NOUN", 2, "obj"),
    ])];
    inject_morphosyntax_results(
        &mut chat,
        items,
        responses,
        &lang("eng"),
        TokenizationMode::Preserve,
        &std::collections::BTreeMap::new(),
    )
    .unwrap();
    let output = chat.to_chat_string();
    assert!(output.contains("[/]"), "retrace group should be preserved");
    assert!(output.contains("%mor:"), "should have morphology");
}

#[test]
fn test_inject_retokenize_splits_contraction() {
    let mut chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tI don't know .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, _) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    let responses = vec![make_ud_response(&[
        ("I", "I", "PRON", 4, "nsubj"),
        ("do", "do", "AUX", 4, "aux"),
        ("n't", "not", "PART", 4, "advmod"),
        ("know", "know", "VERB", 0, "root"),
    ])];
    inject_morphosyntax_results(
        &mut chat,
        items,
        responses,
        &lang("eng"),
        TokenizationMode::StanzaRetokenize,
        &std::collections::BTreeMap::new(),
    )
    .unwrap();
    let output = chat.to_chat_string();
    assert!(output.contains("%mor:"), "should have morphology");
    // After retokenization, the main tier should have the split tokens
    assert!(
        output.contains("do") && output.contains("n't"),
        "retokenize should split contraction, got: {}",
        output
    );
}

#[test]
fn test_inject_no_retokenize_keeps_original() {
    // When retokenize=false, the response must have the same number of
    // items as CHAT words.  "don't" is one CHAT word, so we provide one
    // MOR item for it (as the pretokenized pipeline would).
    let mut chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tI don't know .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, _) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    let responses = vec![make_ud_response(&[
        ("I", "I", "PRON", 3, "nsubj"),
        ("don't", "do", "AUX", 3, "aux"),
        ("know", "know", "VERB", 0, "root"),
    ])];
    inject_morphosyntax_results(
        &mut chat,
        items,
        responses,
        &lang("eng"),
        TokenizationMode::Preserve,
        &std::collections::BTreeMap::new(),
    )
    .unwrap();
    let output = chat.to_chat_string();
    assert!(
        output.contains("don't"),
        "without retokenize, original should be preserved"
    );
}

#[test]
fn test_inject_output_round_trips() {
    let mut chat = parse_chat(
        "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tI eat cookies .
@End
",
    );
    let primary = lang("eng");
    let declared = vec![lang("eng")];
    let (items, _) =
        collect_morphosyntax_payloads(&chat, &primary, &declared, MultilingualPolicy::ProcessAll);
    let responses = vec![make_ud_response(&[
        ("I", "I", "PRON", 2, "nsubj"),
        ("eat", "eat", "VERB", 0, "root"),
        ("cookies", "cookie", "NOUN", 2, "obj"),
    ])];
    inject_morphosyntax_results(
        &mut chat,
        items,
        responses,
        &lang("eng"),
        TokenizationMode::Preserve,
        &std::collections::BTreeMap::new(),
    )
    .unwrap();
    let output = chat.to_chat_string();
    // Re-parse should succeed
    let reparsed = t::parse_and_serialize(&output);
    assert!(
        reparsed.is_ok(),
        "output should re-parse: {:?}",
        reparsed.err()
    );
}

// -----------------------------------------------------------------------
// Existing tests below
// -----------------------------------------------------------------------

/// Regression: quotation terminators (+"/. and +".） must parse successfully.
/// These are standard CHAT utterance terminators used in quoted speech.
/// Previously the direct parser's mor_tier.rs was missing them.
#[test]
fn test_quotation_terminators_parse() {
    // +"/. (quoted_new_line) -- quote continues next line
    let input = "\
@UTF8
@Begin
@Languages:\tita
@Participants:\tMOT Mother
@ID:\tita|test|MOT||female|||Mother|||
*MOT:\tsi , dice +\"/.
%mor:\tintj|si cm|cm verb|dire-Fin-Ind-Pres-S3 +\"/.
@End
";
    let result = t::parse_and_serialize(input);
    assert!(
        result.is_ok(),
        "quotation terminator +\"/. should parse: {:?}",
        result.err()
    );
    let output = result.unwrap();
    assert!(output.contains("+\"/."), "terminator should round-trip");

    // +". (quoted_period_simple) -- quote ends with period
    let input2 = "\
@UTF8
@Begin
@Languages:\tita
@Participants:\tMOT Mother
@ID:\tita|test|MOT||female|||Mother|||
*MOT:\tquello disse +\".
%mor:\tdet|quello-Masc-Def-Dem-Sing verb|dire-Fin-Ind-Past-S3 +\".
@End
";
    let result2 = t::parse_and_serialize(input2);
    assert!(
        result2.is_ok(),
        "quotation terminator +\". should parse: {:?}",
        result2.err()
    );
}

#[test]
fn test_validate_alignments_clean_chat() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thello world .
%mor:\tn|hello n|world .
%gra:\t1|2|MOD 2|0|ROOT 3|2|PUNCT
@End
";
    let errors = t::validate_alignments(input);
    assert!(
        errors.is_empty(),
        "Expected no errors for valid CHAT, got: {:?}",
        errors
    );
}

#[test]
fn test_validate_alignments_mismatched_mor_gra() {
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thello world .
%mor:\tn|hello n|world .
%gra:\t1|0|ROOT
@End
";
    let errors = t::validate_alignments(input);
    assert!(!errors.is_empty(), "Expected errors for mor/gra mismatch");
}

#[test]
fn test_validate_alignments_terminator_value_mismatch_e716() {
    // Main tier has "?" but %mor has "." -- must produce E716
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thuh ?
%mor:\tco|huh .
@End
";
    let errors = t::validate_alignments(input);
    assert!(
        !errors.is_empty(),
        "Expected E716 for terminator value mismatch"
    );
    let has_e716 = errors.iter().any(|e| e.contains("E716"));
    assert!(has_e716, "Expected E716 error code, got: {:?}", errors);
}

#[test]
fn test_validate_alignments_matching_terminators_ok() {
    // Both have "?" -- no error
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thuh ?
%mor:\tco|huh ?
@End
";
    let errors = t::validate_alignments(input);
    assert!(
        errors.is_empty(),
        "Expected no errors when terminators match, got: {:?}",
        errors
    );
}

// -----------------------------------------------------------------------
// Tests: errors_to_json
// -----------------------------------------------------------------------

#[test]
fn test_errors_to_json_empty() {
    let json = errors_to_json(&[]);
    assert_eq!(json, "[]");
}

#[test]
fn test_errors_to_json_roundtrip() {
    let span = talkbank_model::Span { start: 10, end: 20 };
    let mut location = talkbank_model::SourceLocation::new(span);
    location.line = Some(5);
    location.column = Some(3);

    let error = talkbank_model::ParseError::new(
        talkbank_model::ErrorCode::TestError,
        talkbank_model::Severity::Error,
        location,
        talkbank_model::ErrorContext::new("test input", span, "bad"),
        "Something went wrong".to_string(),
    )
    .with_suggestion("Fix it");

    let json = errors_to_json(&[error]);
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0]["code"].as_str().unwrap(), "E002");
    assert_eq!(parsed[0]["severity"].as_str().unwrap(), "error");
    assert_eq!(parsed[0]["line"].as_u64().unwrap(), 5);
    assert_eq!(parsed[0]["column"].as_u64().unwrap(), 3);
    assert_eq!(
        parsed[0]["message"].as_str().unwrap(),
        "Something went wrong"
    );
    assert_eq!(parsed[0]["suggestion"].as_str().unwrap(), "Fix it");
}

#[test]
fn test_parse_warnings_captured_on_lenient_parse() {
    // A CHAT file with errors that lenient parsing recovers from
    // (e.g., missing @End) should capture warnings in the handle.
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|test|CHI|||||Child|||\n*CHI:\thello .\n";
    // Missing @End -- lenient parse should succeed but with a warning
    let (cf, warnings) = parse_lenient_with_warnings(input).unwrap();
    assert!(!cf.lines.is_empty(), "Lenient parse should produce output");
    // The handle wrapper should have the same warnings
    let handle = ParsedChat {
        inner: cf,
        warnings: warnings.clone(),
    };
    let json = handle.py_parse_warnings();
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
    // Should have at least one warning about missing @End
    assert_eq!(parsed.len(), warnings.len());
}

#[test]
fn test_validate_structured_returns_json() {
    // CHAT with mismatched mor alignment to produce validation errors
    let input = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Child
@ID:\teng|test|CHI|||||Child|||
*CHI:\thello world .
%mor:\tn|hello .
@End
";
    let handle = t::make_handle_lenient(input);
    // Call the inner validate_alignments directly (no Python GIL needed)
    let errors = handle.inner.validate_alignments();
    let json = errors_to_json(&errors);
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
    // Should have at least one alignment mismatch error
    assert!(
        !parsed.is_empty(),
        "Expected validation errors for mismatched mor, got empty"
    );
    assert!(parsed[0]["code"].is_string());
    assert!(parsed[0]["message"].is_string());
}
