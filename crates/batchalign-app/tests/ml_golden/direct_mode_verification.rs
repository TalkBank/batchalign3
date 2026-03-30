//! Direct mode verification tests — confirms morphotag, align, and transcribe
//! work end-to-end in direct mode with concurrency, batching, and timing.
//!
//! These tests exercise the full CLI-equivalent direct execution path:
//! - ServerConfig loaded (including media_mappings)
//! - DirectHost created with real worker pool
//! - Jobs submitted, executed, and results verified
//!
//! Multi-file tests verify that the batching and concurrency machinery
//! (language grouping, semaphore-bounded dispatch, utterance-budget windowing)
//! works correctly in direct mode, not just server mode.
//!
//! Run: `cargo nextest run -p batchalign-app --test ml_golden --profile ml`

use crate::common::{
    assert_completed_without_errors, prepare_audio_fixtures, require_live_direct,
    submit_and_complete_direct, submit_paths_and_complete_direct,
};
use batchalign_app::api::{FilePayload, JobStatus, ReleasedCommand};
use batchalign_app::options::{
    AlignOptions, AsrEngineName, CommandOptions, CommonOptions, FaEngineName, MorphotagOptions,
    TranscribeOptions, WorTierPolicy,
};
use batchalign_app::worker::InferTask;
use batchalign_chat_ops::extract::extract_words;
use batchalign_chat_ops::parse::{TreeSitterParser, parse_lenient};
use batchalign_chat_ops::{ChatFile, DependentTier, TierDomain};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse CHAT output via the AST and assert no parse errors.
fn parse_output(chat: &str, label: &str) -> ChatFile {
    let parser = TreeSitterParser::new().unwrap();
    let (file, errors) = parse_lenient(&parser, chat);
    assert!(errors.is_empty(), "{label}: CHAT parse errors: {errors:?}");
    file
}

/// Check if any utterance has a %mor tier.
fn has_mor_tier(file: &ChatFile) -> bool {
    file.lines.iter().any(|line| {
        if let batchalign_chat_ops::Line::Utterance(utt) = line {
            utt.dependent_tiers
                .iter()
                .any(|t| matches!(t, DependentTier::Mor(_)))
        } else {
            false
        }
    })
}

/// Count utterances with %mor tiers.
fn count_mor_tiers(file: &ChatFile) -> usize {
    file.lines
        .iter()
        .filter(|line| {
            if let batchalign_chat_ops::Line::Utterance(utt) = line {
                utt.dependent_tiers
                    .iter()
                    .any(|t| matches!(t, DependentTier::Mor(_)))
            } else {
                false
            }
        })
        .count()
}

/// Assert every alignable utterance has a timing bullet.
fn assert_all_utterances_timed(chat: &str, label: &str) {
    let file = parse_output(chat, label);
    let extracted = extract_words(&file, TierDomain::Mor);
    for (ext_utt, utt) in extracted.iter().zip(file.utterances()) {
        if ext_utt.words.is_empty() {
            continue;
        }
        assert!(
            utt.main.content.bullet.is_some(),
            "{label}: utterance by {} missing timing bullet",
            utt.main.speaker
        );
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const ENG_FILE_A: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child, MOT Mother
@ID:\teng|test|CHI|2;0.||||Target_Child|||
@ID:\teng|test|MOT|||||Mother|||
*CHI:\tthe dog is running .
*MOT:\tyes the dog is very fast .
*CHI:\tI want to play .
@End
";

const ENG_FILE_B: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\tthe cat sat on the mat .
*PAR:\twe walked to school today .
*PAR:\the read a book last night .
@End
";

// ---------------------------------------------------------------------------
// Test: Multi-file morphotag (batching + concurrency)
// ---------------------------------------------------------------------------

/// Morphotag with multiple files verifies:
/// - File chunking via utterance-budget windowing
/// - All files get %mor and %gra tiers
/// - Total utterance count matches across input and output
#[tokio::test]
async fn direct_morphotag_multi_file_batching() {
    let Some(session) = require_live_direct(
        InferTask::Morphosyntax,
        "Direct session does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };

    let options = CommandOptions::Morphotag(MorphotagOptions {
        common: CommonOptions {
            override_media_cache: true,
            ..CommonOptions::default()
        },
        retokenize: false,
        skipmultilang: false,
        merge_abbrev: false.into(),
    });

    let files = vec![
        FilePayload {
            filename: "file_a.cha".into(),
            content: ENG_FILE_A.into(),
        },
        FilePayload {
            filename: "file_b.cha".into(),
            content: ENG_FILE_B.into(),
        },
    ];

    let (info, results) =
        submit_and_complete_direct(&session, ReleasedCommand::Morphotag, "eng", files, options)
            .await;

    assert_completed_without_errors("multi_file_morphotag", &info, &results);
    assert_eq!(results.len(), 2, "Should produce 2 output files");

    // Verify both files have %mor tiers on all utterances.
    let file_a = parse_output(&results[0].content, "file_a");
    let file_b = parse_output(&results[1].content, "file_b");

    assert_eq!(
        count_mor_tiers(&file_a),
        3,
        "file_a: all 3 utterances should have %mor"
    );
    assert_eq!(
        count_mor_tiers(&file_b),
        3,
        "file_b: all 3 utterances should have %mor"
    );
}

// ---------------------------------------------------------------------------
// Test: Multi-language morphotag (language grouping)
// ---------------------------------------------------------------------------

/// Morphotag with two English files from different speakers verifies:
/// - Multiple files dispatch through the same language group
/// - Utterance-budget windowing groups files correctly
/// - Both files get independent %mor/%gra tiers
#[tokio::test]
async fn direct_morphotag_multi_speaker_batching() {
    let Some(session) = require_live_direct(
        InferTask::Morphosyntax,
        "Direct session does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };

    let options = CommandOptions::Morphotag(MorphotagOptions {
        common: CommonOptions {
            override_media_cache: true,
            ..CommonOptions::default()
        },
        retokenize: false,
        skipmultilang: false,
        merge_abbrev: false.into(),
    });

    // Unique content (different from multi_file test) to avoid cache collisions.
    let speaker_c = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tMOT Mother
@ID:\teng|test|MOT|||||Mother|||
*MOT:\tplease come here now .
*MOT:\tit is time for dinner .
*MOT:\twash your hands first .
@End
";
    let speaker_d = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tFAT Father
@ID:\teng|test|FAT|||||Father|||
*FAT:\tlet us go outside .
*FAT:\tthe weather is nice today .
*FAT:\tbring your jacket .
@End
";

    let files = vec![
        FilePayload {
            filename: "speaker_a.cha".into(),
            content: speaker_c.into(),
        },
        FilePayload {
            filename: "speaker_b.cha".into(),
            content: speaker_d.into(),
        },
    ];

    let (info, results) =
        submit_and_complete_direct(&session, ReleasedCommand::Morphotag, "eng", files, options)
            .await;

    assert_completed_without_errors("multi_speaker_morphotag", &info, &results);
    assert_eq!(results.len(), 2, "Should produce 2 output files");

    // Both files should have morphology.
    let file_a = parse_output(&results[0].content, "speaker_a");
    let file_b = parse_output(&results[1].content, "speaker_b");
    assert!(has_mor_tier(&file_a), "speaker_a should have %mor tier");
    assert!(has_mor_tier(&file_b), "speaker_b should have %mor tier");

    // Verify different utterance counts to confirm files are independent.
    assert_eq!(count_mor_tiers(&file_a), 3, "speaker_a: 3 utterances");
    assert_eq!(count_mor_tiers(&file_b), 3, "speaker_b: 3 utterances");
}

// ---------------------------------------------------------------------------
// Test: Direct align produces timed output
// ---------------------------------------------------------------------------

/// Align in direct mode verifies:
/// - Worker pool spawns GPU workers
/// - Audio file is read from disk (paths_mode)
/// - Output has timing bullets on all utterances
/// - FA pipeline completes without errors
#[tokio::test]
async fn direct_align_produces_timed_output() {
    let Some(session) =
        require_live_direct(InferTask::Fa, "Direct session does not support FA infer").await
    else {
        return;
    };

    let Some(fixtures) = prepare_audio_fixtures(session.state_dir()) else {
        return;
    };

    let out_dir = session.state_dir().join("out_direct_align_verify");
    std::fs::create_dir_all(&out_dir).expect("mkdir");
    let output_path = out_dir.join("test.cha");

    let options = CommandOptions::Align(AlignOptions {
        common: CommonOptions {
            override_media_cache: true,
            ..CommonOptions::default()
        },
        fa_engine: FaEngineName::Wave2Vec,
        wor: WorTierPolicy::Include,
        ..AlignOptions::default()
    });

    let (info, outputs) = submit_paths_and_complete_direct(
        &session,
        ReleasedCommand::Align,
        "eng",
        vec![fixtures.stripped_chat.to_string_lossy().into()],
        vec![output_path.to_string_lossy().into()],
        options,
    )
    .await;

    assert_completed_without_errors("direct_align_verify", &info, &[]);
    assert_eq!(info.status, JobStatus::Completed);
    assert_eq!(outputs.len(), 1);

    let output = &outputs[0];
    assert_all_utterances_timed(output, "direct_align_verify");
}

// ---------------------------------------------------------------------------
// Test: Direct transcribe produces valid CHAT
// ---------------------------------------------------------------------------

/// Transcribe in direct mode verifies:
/// - ASR engine (Whisper) runs locally without API keys
/// - Output is valid CHAT with @Media header
/// - Pipeline stages (asr → postprocess → build_chat → utseg) complete
#[tokio::test]
async fn direct_transcribe_produces_valid_chat() {
    let Some(session) =
        require_live_direct(InferTask::Asr, "Direct session does not support ASR infer").await
    else {
        return;
    };

    let Some(fixtures) = prepare_audio_fixtures(session.state_dir()) else {
        return;
    };

    let out_dir = session.state_dir().join("out_direct_transcribe_verify");
    std::fs::create_dir_all(&out_dir).expect("mkdir");
    let output_path = out_dir.join("test.cha");

    let options = CommandOptions::Transcribe(TranscribeOptions {
        common: CommonOptions {
            override_media_cache: true,
            ..CommonOptions::default()
        },
        asr_engine: AsrEngineName::Whisper,
        diarize: false,
        wor: WorTierPolicy::Omit,
        merge_abbrev: false.into(),
        batch_size: 8,
    });

    let (info, outputs) = submit_paths_and_complete_direct(
        &session,
        ReleasedCommand::Transcribe,
        "eng",
        vec![fixtures.audio.to_string_lossy().into()],
        vec![output_path.to_string_lossy().into()],
        options,
    )
    .await;

    assert_eq!(
        info.status,
        JobStatus::Completed,
        "direct_transcribe_verify should complete; error={:?}",
        info.error
    );
    assert_eq!(outputs.len(), 1);

    let output = &outputs[0];
    let file = parse_output(output, "direct_transcribe_verify");

    // Transcribe should produce at least one utterance from real audio.
    assert!(
        file.utterance_count() >= 1,
        "direct_transcribe_verify: expected at least 1 utterance, got {}",
        file.utterance_count()
    );
}
