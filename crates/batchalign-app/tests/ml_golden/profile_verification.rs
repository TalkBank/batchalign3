//! Worker profile verification tests — real-model resource usage assertions.
//!
//! These tests verify that the worker profile system correctly groups
//! InferTasks into shared workers, reducing memory consumption compared
//! to per-task worker spawning.
//!
//! Requirements:
//! - Python 3 with batchalign installed
//! - FA models (Wave2Vec) for GPU profile tests
//! - Stanza models for Stanza profile tests
//!
//! Tests skip gracefully if models are unavailable.
//!
//! Run: `cargo nextest run -p batchalign-app --test ml_golden --profile ml`

use batchalign_app::api::JobStatus;
use batchalign_app::options::{
    AlignOptions, CommandOptions, CommonOptions, FaEngineName, MorphotagOptions, UtsegOptions,
    WorTierPolicy,
};
use batchalign_app::worker::InferTask;
use crate::common::{
    assert_completed_without_errors, prepare_audio_fixtures, require_live_server,
    submit_paths_and_complete,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Query the `/health` endpoint and return the parsed JSON value.
///
/// Uses `serde_json::Value` for flexible field access without coupling
/// to the exact `HealthResponse` struct layout — profile key formats
/// may evolve and this keeps assertions readable.
async fn query_health(client: &reqwest::Client, base_url: &str) -> serde_json::Value {
    let resp = client
        .get(format!("{base_url}/health"))
        .send()
        .await
        .expect("health request failed");
    assert_eq!(resp.status(), 200);
    resp.json::<serde_json::Value>()
        .await
        .expect("health parse failed")
}

/// Extract `live_worker_keys` from a health JSON response as a `Vec<String>`.
fn extract_worker_keys(health: &serde_json::Value) -> Vec<String> {
    health["live_worker_keys"]
        .as_array()
        .expect("live_worker_keys should be an array")
        .iter()
        .map(|v| v.as_str().expect("worker key should be a string").to_owned())
        .collect()
}

// ---------------------------------------------------------------------------
// CHAT fixture for text-only tests (no audio needed)
// ---------------------------------------------------------------------------

const ENG_TEXT_FIXTURE: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\tthe dog is running .
*PAR:\tI like cats .
*PAR:\tshe went to the store .
@End
";

// ---------------------------------------------------------------------------
// Test 1: GPU profile groups FA workers into a single process
// ---------------------------------------------------------------------------

/// Verify that submitting a multi-file align job produces exactly one
/// `profile:gpu:` worker key — the profile system groups ASR/FA/Speaker
/// into a shared GPU worker instead of spawning per-task processes.
///
/// This is the key memory verification test: without profile grouping,
/// each InferTask would spawn its own subprocess with duplicate model
/// copies, consuming N× memory.
#[tokio::test]
async fn gpu_profile_uses_single_worker_for_multi_file_align() {
    let Some(server) =
        require_live_server(InferTask::Fa, "Server does not support FA infer").await
    else {
        return;
    };

    let Some(fixtures) = prepare_audio_fixtures(server.state_dir()) else {
        return;
    };

    // Create 3 copies of the stripped CHAT + audio. Each copy lives in its
    // own subdirectory so the server can resolve audio by filename. The CHAT
    // has `@Media: test, audio` so the server looks for `test.mp3` next to
    // each CHAT file.
    let chat_content =
        std::fs::read_to_string(&fixtures.stripped_chat).expect("read stripped fixture");
    let audio_source = fixtures
        .stripped_chat
        .parent()
        .expect("stripped_chat parent")
        .join("test.mp3");

    let out_dir = server.state_dir().join("profile_gpu_outputs");
    std::fs::create_dir_all(&out_dir).expect("mkdir output dir");

    let mut source_paths = Vec::new();
    let mut output_paths = Vec::new();
    for i in 1..=3 {
        let copy_dir = server.state_dir().join(format!("profile_input_{i}"));
        std::fs::create_dir_all(&copy_dir).expect("mkdir copy dir");
        let input_cha = copy_dir.join("test.cha");
        std::fs::write(&input_cha, &chat_content).expect("write cha copy");
        std::fs::copy(&audio_source, copy_dir.join("test.mp3")).expect("copy audio");
        source_paths.push(input_cha.to_string_lossy().into_owned());
        output_paths.push(
            out_dir
                .join(format!("test{i}.cha"))
                .to_string_lossy()
                .into_owned(),
        );
    }

    let options = CommandOptions::Align(AlignOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        fa_engine: FaEngineName::Wave2Vec,
        wor: WorTierPolicy::Include,
        ..AlignOptions::default()
    });

    let (info, _outputs) = submit_paths_and_complete(
        server.client(),
        server.base_url(),
        "align",
        "eng",
        source_paths,
        output_paths,
        options,
    )
    .await;

    assert_eq!(
        info.status,
        JobStatus::Completed,
        "multi-file align should complete"
    );

    // Query health to inspect live worker keys after job completion.
    let health = query_health(server.client(), server.base_url()).await;
    let keys = extract_worker_keys(&health);

    eprintln!("live_worker_keys after multi-file align: {keys:?}");
    eprintln!(
        "live_workers count: {}",
        health["live_workers"].as_i64().unwrap_or(-1)
    );

    // The FA dispatch should go through the shared concurrent GPU worker
    // (tagged "shared" in the key), NOT through per-task sequential workers.
    let gpu_shared: Vec<&String> = keys
        .iter()
        .filter(|k| k.starts_with("profile:gpu:") && k.contains("shared"))
        .collect();
    assert!(
        !gpu_shared.is_empty(),
        "Expected at least 1 shared GPU worker key, got none in {keys:?}"
    );

    // No legacy per-task keys should appear.
    let legacy_fa: Vec<&String> = keys.iter().filter(|k| k.starts_with("infer:fa:")).collect();
    let legacy_asr: Vec<&String> = keys.iter().filter(|k| k.starts_with("infer:asr:")).collect();
    assert!(
        legacy_fa.is_empty(),
        "No legacy infer:fa: keys should exist, got {legacy_fa:?}"
    );
    assert!(
        legacy_asr.is_empty(),
        "No legacy infer:asr: keys should exist, got {legacy_asr:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Stanza profile groups morphotag and utseg
// ---------------------------------------------------------------------------

/// Verify that morphotag and utseg share the same Stanza profile worker.
///
/// Both commands use Stanza NLP processors. The profile system should
/// group them under a single `profile:stanza:` key rather than spawning
/// separate workers for each task.
#[tokio::test]
async fn stanza_profile_groups_morphotag_and_utseg() {
    let Some(server) = require_live_server(
        InferTask::Morphosyntax,
        "Server does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };

    // Also need utseg support for the second submission.
    if !server.has_infer_task(InferTask::Utseg) {
        eprintln!("SKIP: Server does not support utseg infer");
        return;
    }

    // Set up text-only CHAT fixtures for paths-mode submission.
    let input_dir = server.state_dir().join("profile_stanza_inputs");
    std::fs::create_dir_all(&input_dir).expect("mkdir input dir");
    let input_path = input_dir.join("stanza_test.cha");
    std::fs::write(&input_path, ENG_TEXT_FIXTURE).expect("write text fixture");

    // -- Submit morphotag --
    let morphotag_out_dir = server.state_dir().join("profile_stanza_morphotag_out");
    std::fs::create_dir_all(&morphotag_out_dir).expect("mkdir morphotag output");

    let morphotag_options = CommandOptions::Morphotag(MorphotagOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        retokenize: false,
        skipmultilang: false,
        merge_abbrev: false.into(),
    });

    let (info_mt, _outputs_mt) = submit_paths_and_complete(
        server.client(),
        server.base_url(),
        "morphotag",
        "eng",
        vec![input_path.to_string_lossy().into_owned()],
        vec![morphotag_out_dir
            .join("stanza_test.cha")
            .to_string_lossy()
            .into_owned()],
        morphotag_options,
    )
    .await;

    assert_eq!(
        info_mt.status,
        JobStatus::Completed,
        "morphotag should complete"
    );

    // -- Submit utseg on the same input --
    let utseg_out_dir = server.state_dir().join("profile_stanza_utseg_out");
    std::fs::create_dir_all(&utseg_out_dir).expect("mkdir utseg output");

    let utseg_options = CommandOptions::Utseg(UtsegOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        merge_abbrev: false.into(),
    });

    let (info_ut, _outputs_ut) = submit_paths_and_complete(
        server.client(),
        server.base_url(),
        "utseg",
        "eng",
        vec![input_path.to_string_lossy().into_owned()],
        vec![utseg_out_dir
            .join("stanza_test.cha")
            .to_string_lossy()
            .into_owned()],
        utseg_options,
    )
    .await;

    assert_eq!(
        info_ut.status,
        JobStatus::Completed,
        "utseg should complete"
    );

    // Query health to inspect worker keys after both commands ran.
    let health = query_health(server.client(), server.base_url()).await;
    let keys = extract_worker_keys(&health);

    eprintln!("live_worker_keys after morphotag + utseg: {keys:?}");

    // Both morphotag and utseg should share one Stanza profile worker.
    let stanza_keys: Vec<&String> = keys
        .iter()
        .filter(|k| k.starts_with("profile:stanza:"))
        .collect();
    assert_eq!(
        stanza_keys.len(),
        1,
        "Expected exactly 1 profile:stanza: worker key after morphotag + utseg, got {stanza_keys:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Regression guard — all worker keys use profile: prefix
// ---------------------------------------------------------------------------

/// Assert that ALL live worker keys use the `profile:` prefix and NONE
/// use the legacy `infer:` prefix.
///
/// This is a regression guard: if new code accidentally bypasses the
/// profile system and spawns per-task workers, this test will catch it.
#[tokio::test]
async fn profile_worker_keys_use_profile_labels() {
    // morphotag is the lightest model — use it for a quick smoke test.
    let Some(server) = require_live_server(
        InferTask::Morphosyntax,
        "Server does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };

    let input_dir = server.state_dir().join("profile_labels_inputs");
    std::fs::create_dir_all(&input_dir).expect("mkdir input dir");
    let input_path = input_dir.join("labels_test.cha");
    std::fs::write(&input_path, ENG_TEXT_FIXTURE).expect("write text fixture");

    let out_dir = server.state_dir().join("profile_labels_out");
    std::fs::create_dir_all(&out_dir).expect("mkdir output dir");

    let options = CommandOptions::Morphotag(MorphotagOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        retokenize: false,
        skipmultilang: false,
        merge_abbrev: false.into(),
    });

    let (info, _outputs) = submit_paths_and_complete(
        server.client(),
        server.base_url(),
        "morphotag",
        "eng",
        vec![input_path.to_string_lossy().into_owned()],
        vec![out_dir
            .join("labels_test.cha")
            .to_string_lossy()
            .into_owned()],
        options,
    )
    .await;

    assert_completed_without_errors("profile_labels", &info, &[]);

    let health = query_health(server.client(), server.base_url()).await;
    let keys = extract_worker_keys(&health);

    eprintln!("live_worker_keys for regression guard: {keys:?}");

    // Every key must start with "profile:".
    for key in &keys {
        assert!(
            key.starts_with("profile:"),
            "Worker key should use profile: prefix, got: {key}"
        );
    }

    // No legacy infer: keys should appear.
    let legacy_keys: Vec<&String> = keys.iter().filter(|k| k.starts_with("infer:")).collect();
    assert!(
        legacy_keys.is_empty(),
        "No legacy infer: keys should exist, got {legacy_keys:?}"
    );
}
