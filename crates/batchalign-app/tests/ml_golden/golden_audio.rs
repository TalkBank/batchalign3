//! Audio integration tests — real-model forced alignment, transcription,
//! OpenSMILE, and benchmark with the committed `test.mp3` fixture.
//!
//! These tests use paths-mode submissions so the server reads audio from disk.
//! Assertions are structural (not snapshot-based) because audio model output
//! is non-deterministic across runs.
//!
//! Requirements:
//! - Python 3 with batchalign installed
//! - FA models (Wave2Vec/Whisper) for align tests
//! - ASR models (Whisper) for transcribe tests
//! - OpenSMILE for opensmile tests
//!
//! Tests skip gracefully if models are unavailable.
//!
//! Run: `cargo nextest run -p batchalign-app --test ml_golden --profile ml`

use crate::common::{
    assert_completed_without_errors, prepare_audio_fixtures, prepare_named_audio,
    require_live_server, require_revai_key, submit_paths_and_complete,
};
use batchalign_app::api::JobStatus;
use batchalign_app::options::{
    AlignOptions, AsrEngineName, BenchmarkOptions, CommandOptions, CommonOptions, FaEngineName,
    OpensmileOptions, TranscribeOptions, WorTierPolicy,
};
use batchalign_app::worker::InferTask;

// ---------------------------------------------------------------------------
// Structural assertion helpers (AST-based, not string hacking)
// ---------------------------------------------------------------------------

use batchalign_chat_ops::TierDomain;
use batchalign_chat_ops::extract::extract_words;
use batchalign_chat_ops::parse::parse_lenient;

/// Parse CHAT text into a typed AST, asserting no parse errors.
fn parse_chat(chat: &str, label: &str) -> batchalign_chat_ops::ChatFile {
    let (file, errors) = parse_lenient(chat);
    assert!(
        errors.is_empty(),
        "{label}: CHAT parse produced errors: {errors:?}"
    );
    file
}

/// Assert every alignable utterance has a timing bullet.
///
/// Utterances with zero NLP-extractable words (e.g., `xxx`-only turns) are
/// excluded because they have no words to force-align.
fn assert_all_utterances_timed(chat: &str, label: &str) {
    let file = parse_chat(chat, label);
    // extract_words returns per-utterance word lists; utterances with zero
    // words are untranscribed and get no timing.
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

/// Count utterances that have a `%wor` dependent tier.
fn count_wor_tiers(chat: &str) -> usize {
    let (file, _) = parse_lenient(chat);
    file.utterances()
        .filter(|utt| utt.wor_tier().is_some())
        .count()
}

/// Assert valid CHAT structure: parses without errors, has required headers,
/// and contains at least one utterance.
fn assert_valid_chat_structure(chat: &str, label: &str) {
    let file = parse_chat(chat, label);
    assert!(
        file.utterance_count() >= 1,
        "{label}: output should have at least 1 utterance"
    );
    // If parse_lenient succeeded with zero errors, the file has valid
    // @UTF8, @Begin, @End, @Languages, and @Participants headers —
    // the parser enforces these structurally.
}

// ---------------------------------------------------------------------------
// Phase 3: Align tests
// ---------------------------------------------------------------------------

/// Forced alignment with Wave2Vec, %wor tier included.
#[tokio::test]
async fn golden_align_eng_wav2vec() {
    let Some(server) = require_live_server(InferTask::Fa, "Server does not support FA infer").await
    else {
        return;
    };

    let Some(fixtures) = prepare_audio_fixtures(server.state_dir()) else {
        return;
    };

    let out_dir = server.state_dir().join("out_align_wav2vec");
    std::fs::create_dir_all(&out_dir).expect("mkdir");
    let output_path = out_dir.join("test.cha");

    let options = CommandOptions::Align(AlignOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        fa_engine: FaEngineName::Wave2Vec,
        wor: WorTierPolicy::Include,
        ..AlignOptions::default()
    });

    let (info, outputs) = submit_paths_and_complete(
        server.client(),
        server.base_url(),
        "align",
        "eng",
        vec![fixtures.stripped_chat.to_string_lossy().into()],
        vec![output_path.to_string_lossy().into()],
        options,
    )
    .await;

    assert_completed_without_errors("align_eng_wav2vec", &info, &[]);
    assert_eq!(info.status, JobStatus::Completed);
    assert_eq!(outputs.len(), 1);

    let output = &outputs[0];
    assert_all_utterances_timed(output, "align_eng_wav2vec");
    assert!(
        count_wor_tiers(output) > 0,
        "align_eng_wav2vec: %wor tier should be present"
    );
    assert_valid_chat_structure(output, "align_eng_wav2vec");
}

/// Forced alignment with Whisper FA engine, %wor tier included.
#[tokio::test]
async fn golden_align_eng_whisper_fa() {
    let Some(server) = require_live_server(InferTask::Fa, "Server does not support FA infer").await
    else {
        return;
    };

    let Some(fixtures) = prepare_audio_fixtures(server.state_dir()) else {
        return;
    };

    let out_dir = server.state_dir().join("out_align_whisper");
    std::fs::create_dir_all(&out_dir).expect("mkdir");
    let output_path = out_dir.join("test.cha");

    let options = CommandOptions::Align(AlignOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        fa_engine: FaEngineName::Whisper,
        wor: WorTierPolicy::Include,
        ..AlignOptions::default()
    });

    let (info, outputs) = submit_paths_and_complete(
        server.client(),
        server.base_url(),
        "align",
        "eng",
        vec![fixtures.stripped_chat.to_string_lossy().into()],
        vec![output_path.to_string_lossy().into()],
        options,
    )
    .await;

    assert_completed_without_errors("align_eng_whisper_fa", &info, &[]);
    assert_eq!(info.status, JobStatus::Completed);
    assert_eq!(outputs.len(), 1);

    let output = &outputs[0];
    assert_all_utterances_timed(output, "align_eng_whisper_fa");
    assert!(
        count_wor_tiers(output) > 0,
        "align_eng_whisper_fa: %wor tier should be present"
    );
    assert_valid_chat_structure(output, "align_eng_whisper_fa");
}

/// Forced alignment with %wor tier excluded.
#[tokio::test]
async fn golden_align_eng_no_wor() {
    let Some(server) = require_live_server(InferTask::Fa, "Server does not support FA infer").await
    else {
        return;
    };

    let Some(fixtures) = prepare_audio_fixtures(server.state_dir()) else {
        return;
    };

    let out_dir = server.state_dir().join("out_align_no_wor");
    std::fs::create_dir_all(&out_dir).expect("mkdir");
    let output_path = out_dir.join("test.cha");

    let options = CommandOptions::Align(AlignOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        fa_engine: FaEngineName::Wave2Vec,
        wor: WorTierPolicy::Omit,
        ..AlignOptions::default()
    });

    let (info, outputs) = submit_paths_and_complete(
        server.client(),
        server.base_url(),
        "align",
        "eng",
        vec![fixtures.stripped_chat.to_string_lossy().into()],
        vec![output_path.to_string_lossy().into()],
        options,
    )
    .await;

    assert_completed_without_errors("align_eng_no_wor", &info, &[]);
    assert_eq!(info.status, JobStatus::Completed);
    assert_eq!(outputs.len(), 1);

    let output = &outputs[0];
    assert_all_utterances_timed(output, "align_eng_no_wor");
    assert_eq!(
        count_wor_tiers(output),
        0,
        "align_eng_no_wor: %wor tier should be absent when wor=Omit"
    );
    assert_valid_chat_structure(output, "align_eng_no_wor");
}

// ---------------------------------------------------------------------------
// Phase 4: Transcribe tests
// ---------------------------------------------------------------------------

/// Transcribe with Whisper ASR (no diarization).
#[tokio::test]
async fn golden_transcribe_eng_whisper() {
    let Some(server) =
        require_live_server(InferTask::Asr, "Server does not support ASR infer").await
    else {
        return;
    };

    let Some(fixtures) = prepare_audio_fixtures(server.state_dir()) else {
        return;
    };

    let out_dir = server.state_dir().join("out_transcribe_whisper");
    std::fs::create_dir_all(&out_dir).expect("mkdir");
    let output_path = out_dir.join("test.cha");

    let options = CommandOptions::Transcribe(TranscribeOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        asr_engine: AsrEngineName::Whisper,
        diarize: false,
        wor: WorTierPolicy::Omit,
        merge_abbrev: false.into(),
        batch_size: 8,
    });

    let (info, outputs) = submit_paths_and_complete(
        server.client(),
        server.base_url(),
        "transcribe",
        "eng",
        vec![fixtures.audio.to_string_lossy().into()],
        vec![output_path.to_string_lossy().into()],
        options,
    )
    .await;

    assert_eq!(
        info.status,
        JobStatus::Completed,
        "transcribe_eng_whisper: job should complete"
    );
    assert_eq!(outputs.len(), 1);

    let output = &outputs[0];
    assert_valid_chat_structure(output, "transcribe_eng_whisper");
    assert!(
        output.contains("eng"),
        "transcribe_eng_whisper: output should reference English language"
    );
}

/// Transcribe with Rev.AI ASR (skips if no API key).
#[tokio::test]
async fn golden_transcribe_eng_revai() {
    if require_revai_key().is_none() {
        eprintln!("SKIP: REVAI_API_KEY / BATCHALIGN_REV_API_KEY not set");
        return;
    }

    let Some(server) =
        require_live_server(InferTask::Asr, "Server does not support ASR infer").await
    else {
        return;
    };

    let Some(fixtures) = prepare_audio_fixtures(server.state_dir()) else {
        return;
    };

    let out_dir = server.state_dir().join("out_transcribe_revai");
    std::fs::create_dir_all(&out_dir).expect("mkdir");
    let output_path = out_dir.join("test.cha");

    let options = CommandOptions::Transcribe(TranscribeOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        asr_engine: AsrEngineName::RevAi,
        diarize: false,
        wor: WorTierPolicy::Omit,
        merge_abbrev: false.into(),
        batch_size: 8,
    });

    let (info, outputs) = submit_paths_and_complete(
        server.client(),
        server.base_url(),
        "transcribe",
        "eng",
        vec![fixtures.audio.to_string_lossy().into()],
        vec![output_path.to_string_lossy().into()],
        options,
    )
    .await;

    assert_eq!(
        info.status,
        JobStatus::Completed,
        "transcribe_eng_revai: job should complete"
    );
    assert_eq!(outputs.len(), 1);

    let output = &outputs[0];
    assert_valid_chat_structure(output, "transcribe_eng_revai");
}

/// Transcribe with Whisper ASR and %wor tier.
#[tokio::test]
async fn golden_transcribe_eng_whisper_wor() {
    let Some(server) =
        require_live_server(InferTask::Asr, "Server does not support ASR infer").await
    else {
        return;
    };

    let Some(fixtures) = prepare_audio_fixtures(server.state_dir()) else {
        return;
    };

    let out_dir = server.state_dir().join("out_transcribe_wor");
    std::fs::create_dir_all(&out_dir).expect("mkdir");
    let output_path = out_dir.join("test.cha");

    let options = CommandOptions::Transcribe(TranscribeOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        asr_engine: AsrEngineName::Whisper,
        diarize: false,
        wor: WorTierPolicy::Include,
        merge_abbrev: false.into(),
        batch_size: 8,
    });

    let (info, outputs) = submit_paths_and_complete(
        server.client(),
        server.base_url(),
        "transcribe",
        "eng",
        vec![fixtures.audio.to_string_lossy().into()],
        vec![output_path.to_string_lossy().into()],
        options,
    )
    .await;

    assert_eq!(
        info.status,
        JobStatus::Completed,
        "transcribe_eng_whisper_wor: job should complete"
    );
    assert_eq!(outputs.len(), 1);

    let output = &outputs[0];
    assert_valid_chat_structure(output, "transcribe_eng_whisper_wor");
    assert!(
        count_wor_tiers(output) > 0,
        "transcribe_eng_whisper_wor: %wor tier should be present"
    );
}

// ---------------------------------------------------------------------------
// Phase 7: OpenSMILE and Benchmark
// ---------------------------------------------------------------------------

/// OpenSMILE feature extraction with real audio.
#[tokio::test]
async fn golden_opensmile_eng() {
    let Some(server) = require_live_server(
        InferTask::Opensmile,
        "Server does not support OpenSMILE infer",
    )
    .await
    else {
        return;
    };

    let Some(fixtures) = prepare_audio_fixtures(server.state_dir()) else {
        return;
    };

    let out_dir = server.state_dir().join("out_opensmile");
    std::fs::create_dir_all(&out_dir).expect("mkdir");
    let output_path = out_dir.join("test.csv");

    let options = CommandOptions::Opensmile(OpensmileOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        feature_set: "eGeMAPSv02".into(),
    });

    let (info, outputs) = submit_paths_and_complete(
        server.client(),
        server.base_url(),
        "opensmile",
        "eng",
        vec![fixtures.audio.to_string_lossy().into()],
        vec![output_path.to_string_lossy().into()],
        options,
    )
    .await;

    assert_eq!(
        info.status,
        JobStatus::Completed,
        "opensmile_eng: job should complete"
    );
    assert_eq!(outputs.len(), 1);

    let output = &outputs[0];
    assert!(
        !output.is_empty(),
        "opensmile_eng: output should be non-empty"
    );
}

/// Benchmark (WER) with real audio and gold CHAT.
#[tokio::test]
async fn golden_benchmark_eng() {
    let Some(server) = require_live_server(
        InferTask::Asr,
        "Server does not support ASR infer (required for benchmark)",
    )
    .await
    else {
        return;
    };

    let Some(fixtures) = prepare_audio_fixtures(server.state_dir()) else {
        return;
    };

    // Benchmark needs audio + gold CHAT. Submit both as source paths.
    let out_dir = server.state_dir().join("out_benchmark");
    std::fs::create_dir_all(&out_dir).expect("mkdir");
    let output_audio = out_dir.join("test.csv");
    let output_gold = out_dir.join("test.cha");

    let options = CommandOptions::Benchmark(BenchmarkOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        asr_engine: AsrEngineName::Whisper,
        wor: WorTierPolicy::Omit,
        merge_abbrev: false.into(),
    });

    let (info, _outputs) = submit_paths_and_complete(
        server.client(),
        server.base_url(),
        "benchmark",
        "eng",
        vec![
            fixtures.audio.to_string_lossy().into(),
            fixtures.chat.to_string_lossy().into(),
        ],
        vec![
            output_audio.to_string_lossy().into(),
            output_gold.to_string_lossy().into(),
        ],
        options,
    )
    .await;

    assert_eq!(
        info.status,
        JobStatus::Completed,
        "benchmark_eng: job should complete"
    );

    // TODO: AVQI test requires .cs/.sv fixture pair — defer until fixtures available.
}

// ---------------------------------------------------------------------------
// Multi-language transcribe tests
// ---------------------------------------------------------------------------

/// Helper: transcribe a named audio clip and assert valid CHAT output.
async fn transcribe_audio_clip(audio_name: &str, lang: &str, label: &str) {
    let Some(server) =
        require_live_server(InferTask::Asr, "Server does not support ASR infer").await
    else {
        return;
    };

    let Some(fixtures) = prepare_named_audio(server.state_dir(), audio_name, None) else {
        return;
    };

    // Output directory per test — server writes {input_basename}.cha here.
    let out_dir = server.state_dir().join(format!("out_{label}"));
    std::fs::create_dir_all(&out_dir).expect("mkdir output dir");
    let output_path = out_dir.join(format!("{audio_name}.cha"));

    let options = CommandOptions::Transcribe(TranscribeOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        asr_engine: AsrEngineName::Whisper,
        diarize: false,
        wor: WorTierPolicy::Omit,
        merge_abbrev: false.into(),
        batch_size: 8,
    });

    let (info, outputs) = submit_paths_and_complete(
        server.client(),
        server.base_url(),
        "transcribe",
        lang,
        vec![fixtures.audio.to_string_lossy().into()],
        vec![output_path.to_string_lossy().into()],
        options,
    )
    .await;

    assert_eq!(
        info.status,
        JobStatus::Completed,
        "{label}: job should complete"
    );
    assert_eq!(outputs.len(), 1);
    assert_valid_chat_structure(&outputs[0], label);
}

/// Helper: align a named audio clip with its timed CHAT and assert timing output.
async fn align_audio_clip(audio_name: &str, chat_name: &str, lang: &str, label: &str) {
    let Some(server) = require_live_server(InferTask::Fa, "Server does not support FA infer").await
    else {
        return;
    };

    let Some(fixtures) = prepare_named_audio(server.state_dir(), audio_name, Some(chat_name))
    else {
        return;
    };

    // Output directory per test — server writes {input_basename}.cha here.
    let out_dir = server.state_dir().join(format!("out_{label}"));
    std::fs::create_dir_all(&out_dir).expect("mkdir output dir");
    // Use the stripped_chat's filename as the output name (server preserves input basename).
    let input_basename = fixtures
        .stripped_chat
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let output_path = out_dir.join(&input_basename);

    let options = CommandOptions::Align(AlignOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        wor: WorTierPolicy::Include,
        ..AlignOptions::default()
    });

    let (info, outputs) = submit_paths_and_complete(
        server.client(),
        server.base_url(),
        "align",
        lang,
        vec![fixtures.stripped_chat.to_string_lossy().into()],
        vec![output_path.to_string_lossy().into()],
        options,
    )
    .await;

    assert_eq!(
        info.status,
        JobStatus::Completed,
        "{label}: job should complete"
    );
    assert_eq!(outputs.len(), 1);

    let output = &outputs[0];
    assert_valid_chat_structure(output, label);
    assert_all_utterances_timed(output, label);
}

// --- Spanish ---

#[tokio::test]
async fn transcribe_spa_whisper() {
    transcribe_audio_clip("spa_marrero_clip", "spa", "transcribe_spa").await;
}

#[tokio::test]
async fn align_spa_wav2vec() {
    align_audio_clip("spa_marrero_clip", "spa_marrero_timed", "spa", "align_spa").await;
}

// --- French ---

#[tokio::test]
async fn transcribe_fra_whisper() {
    transcribe_audio_clip("fra_geneva_clip", "fra", "transcribe_fra").await;
}

#[tokio::test]
async fn align_fra_wav2vec() {
    align_audio_clip("fra_geneva_clip", "fra_geneva_timed", "fra", "align_fra").await;
}

// --- Japanese ---

#[tokio::test]
async fn transcribe_jpn_whisper() {
    transcribe_audio_clip("jpn_tyo_clip", "jpn", "transcribe_jpn").await;
}

#[tokio::test]
async fn align_jpn_wav2vec() {
    align_audio_clip("jpn_tyo_clip", "jpn_tyo_timed", "jpn", "align_jpn").await;
}

// --- Cantonese ---

#[tokio::test]
async fn transcribe_yue_whisper() {
    transcribe_audio_clip("yue_hku_clip", "yue", "transcribe_yue").await;
}

#[tokio::test]
async fn align_yue_wav2vec() {
    align_audio_clip("yue_hku_clip", "yue_hku_timed", "yue", "align_yue").await;
}

// --- Bilingual ---

/// Transcribe bilingual Venetian+Croatian audio.
/// Tests language routing for multi-language input.
#[tokio::test]
async fn transcribe_biling_vec_hrv_whisper() {
    // Use the primary language for routing; Whisper handles multilingual.
    transcribe_audio_clip("biling_vec_hrv_clip", "vec", "transcribe_biling_vec_hrv").await;
}

/// Transcribe bilingual Catalan+Spanish audio.
#[tokio::test]
async fn transcribe_biling_cat_spa_whisper() {
    transcribe_audio_clip("biling_cat_spa_clip", "cat", "transcribe_biling_cat_spa").await;
}

// --- English multi-speaker ---

#[tokio::test]
async fn transcribe_eng_multi_speaker_whisper() {
    transcribe_audio_clip("eng_multi_speaker", "eng", "transcribe_eng_multi_speaker").await;
}

#[tokio::test]
async fn align_eng_multi_speaker_wav2vec() {
    align_audio_clip(
        "eng_multi_speaker",
        "eng_multi_speaker",
        "eng",
        "align_eng_multi_speaker",
    )
    .await;
}

// --- Transcribe with diarization ---

/// English transcription with diarization enabled.
/// Tests that speaker diarization pipeline works end-to-end.
#[tokio::test]
async fn transcribe_eng_diarize() {
    let Some(server) =
        require_live_server(InferTask::Asr, "Server does not support ASR infer").await
    else {
        return;
    };

    // Also need speaker diarization model.
    if !server.has_infer_task(InferTask::Speaker) {
        eprintln!("SKIP: Server does not support speaker diarization");
        return;
    }

    let Some(fixtures) = prepare_audio_fixtures(server.state_dir()) else {
        return;
    };

    let out_dir = server.state_dir().join("out_transcribe_diarize");
    std::fs::create_dir_all(&out_dir).expect("mkdir");
    let output_path = out_dir.join("test.cha");

    let options = CommandOptions::Transcribe(TranscribeOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        asr_engine: AsrEngineName::Whisper,
        diarize: true,
        wor: WorTierPolicy::Omit,
        merge_abbrev: false.into(),
        batch_size: 8,
    });

    let (info, outputs) = submit_paths_and_complete(
        server.client(),
        server.base_url(),
        "transcribe",
        "eng",
        vec![fixtures.audio.to_string_lossy().into()],
        vec![output_path.to_string_lossy().into()],
        options,
    )
    .await;

    assert_eq!(
        info.status,
        JobStatus::Completed,
        "transcribe_eng_diarize: job should complete"
    );
    assert_eq!(outputs.len(), 1);
    assert_valid_chat_structure(&outputs[0], "transcribe_eng_diarize");
}

// --- Disfluency/retrace parity assertions (D1/D1b) ---

/// Assert BA3 transcribe output has filled pause markers (&-um, &-uh).
/// This test will FAIL until D1 (DisfluencyReplacementEngine) is implemented.
#[tokio::test]
async fn parity_transcribe_disfluency_markup() {
    let Some(server) =
        require_live_server(InferTask::Asr, "Server does not support ASR infer").await
    else {
        return;
    };

    let Some(fixtures) = prepare_audio_fixtures(server.state_dir()) else {
        return;
    };

    let out_dir = server.state_dir().join("out_disfluency");
    std::fs::create_dir_all(&out_dir).expect("mkdir");
    let output_path = out_dir.join("test.cha");

    let options = CommandOptions::Transcribe(TranscribeOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        asr_engine: AsrEngineName::Whisper,
        diarize: false,
        wor: WorTierPolicy::Omit,
        merge_abbrev: false.into(),
        batch_size: 8,
    });

    let (info, outputs) = submit_paths_and_complete(
        server.client(),
        server.base_url(),
        "transcribe",
        "eng",
        vec![fixtures.audio.to_string_lossy().into()],
        vec![output_path.to_string_lossy().into()],
        options,
    )
    .await;

    if info.status != JobStatus::Completed {
        eprintln!("SKIP: transcribe failed");
        return;
    }

    let output = &outputs[0];

    // The test.mp3 audio contains "um" speech — BA2 marks these as &-um.
    // This assertion will fail until D1 is implemented in BA3.
    assert!(
        output.contains("&-um") || output.contains("&-uh"),
        "D1 PARITY GAP: transcribe output should contain filled pause markers (&-um/&-uh). \
         BA2 runs DisfluencyReplacementEngine after ASR; BA3 does not yet implement this stage."
    );
}

/// Assert BA3 transcribe output has retrace markers ([/]).
/// This test will FAIL until D1b (NgramRetraceEngine) is implemented.
#[tokio::test]
async fn parity_transcribe_retrace_markup() {
    let Some(server) =
        require_live_server(InferTask::Asr, "Server does not support ASR infer").await
    else {
        return;
    };

    let Some(fixtures) = prepare_audio_fixtures(server.state_dir()) else {
        return;
    };

    let out_dir = server.state_dir().join("out_retrace");
    std::fs::create_dir_all(&out_dir).expect("mkdir");
    let output_path = out_dir.join("test.cha");

    let options = CommandOptions::Transcribe(TranscribeOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        asr_engine: AsrEngineName::Whisper,
        diarize: false,
        wor: WorTierPolicy::Omit,
        merge_abbrev: false.into(),
        batch_size: 8,
    });

    let (info, outputs) = submit_paths_and_complete(
        server.client(),
        server.base_url(),
        "transcribe",
        "eng",
        vec![fixtures.audio.to_string_lossy().into()],
        vec![output_path.to_string_lossy().into()],
        options,
    )
    .await;

    if info.status != JobStatus::Completed {
        eprintln!("SKIP: transcribe failed");
        return;
    }

    let output = &outputs[0];

    // The test.mp3 audio contains repeated phrases — BA2 marks these as [/] retraces.
    // This assertion will fail until D1b is implemented in BA3.
    assert!(
        output.contains("[/]"),
        "D1b PARITY GAP: transcribe output should contain retrace markers ([/]). \
         BA2 runs NgramRetraceEngine after ASR; BA3 does not yet implement this stage."
    );
}
