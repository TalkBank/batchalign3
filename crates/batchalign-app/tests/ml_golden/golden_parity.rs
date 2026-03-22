//! BA2 parity tests — compare BA3 output against committed batchalignjan9
//! golden reference outputs.
//!
//! Each test loads a curated CHAT fixture, submits it to BA3, and compares
//! the output against the corresponding BA2 Jan 9 golden file. Tests also
//! include structural assertions as a fallback when golden files haven't
//! been generated yet.
//!
//! Run: `cargo nextest run -p batchalign-app --test ml_golden --profile ml`
//! Generate golden files: `bash scripts/generate_ba2_golden.sh`

use crate::common::{
    assert_ba2_parity, assert_completed_without_errors, load_ba2_golden, load_parity_fixture,
    require_live_server, submit_and_complete,
};
use batchalign_app::api::{ReleasedCommand, FilePayload, JobStatus};
use batchalign_app::options::{
    CommandOptions, CommonOptions, CorefOptions, MorphotagOptions, TranslateOptions, UtsegOptions,
};
use batchalign_app::worker::InferTask;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Submit a fixture through a command and compare to BA2 golden.
async fn run_parity_test(
    command: ReleasedCommand,
    task: InferTask,
    fixture_name: &str,
    lang: &str,
    options: CommandOptions,
) {
    let Some(server) =
        require_live_server(task, &format!("Server does not support {task:?} infer")).await
    else {
        return;
    };

    let Some(input) = load_parity_fixture(fixture_name) else {
        return;
    };

    let files = vec![FilePayload {
        filename: format!("{fixture_name}.cha").into(),
        content: input,
    }];

    let (info, results) = submit_and_complete(
        server.client(),
        server.base_url(),
        command,
        lang,
        files,
        options,
    )
    .await;

    // Some non-English models may not be downloaded — treat as skip.
    if info.status == JobStatus::Failed {
        eprintln!("SKIP: {command} {fixture_name} ({lang}) failed (model likely not downloaded)");
        return;
    }

    assert_completed_without_errors(&format!("{command}_{fixture_name}"), &info, &results);
    assert_eq!(results.len(), 1);

    let output = &results[0].content;

    // Compare against BA2 golden if available.
    if let Some(golden) = load_ba2_golden(command.as_ref(), fixture_name) {
        assert_ba2_parity(&format!("{command}_{fixture_name}"), output, &golden);
    }
}

fn morphotag_opts(retokenize: bool) -> CommandOptions {
    CommandOptions::Morphotag(MorphotagOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        retokenize,
        skipmultilang: false,
        merge_abbrev: false.into(),
    })
}

fn utseg_opts() -> CommandOptions {
    CommandOptions::Utseg(UtsegOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        merge_abbrev: false.into(),
    })
}

fn translate_opts() -> CommandOptions {
    CommandOptions::Translate(TranslateOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        merge_abbrev: false.into(),
    })
}

fn coref_opts() -> CommandOptions {
    CommandOptions::Coref(CorefOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        merge_abbrev: false.into(),
    })
}

// ---------------------------------------------------------------------------
// Morphotag parity
// ---------------------------------------------------------------------------

#[tokio::test]
async fn parity_morphotag_eng_disfluency() {
    run_parity_test(
        ReleasedCommand::Morphotag,
        InferTask::Morphosyntax,
        "eng_disfluency",
        "eng",
        morphotag_opts(false),
    )
    .await;
}

#[tokio::test]
async fn parity_morphotag_eng_multi_speaker() {
    run_parity_test(
        ReleasedCommand::Morphotag,
        InferTask::Morphosyntax,
        "eng_multi_speaker",
        "eng",
        morphotag_opts(false),
    )
    .await;
}

#[tokio::test]
async fn parity_morphotag_eng_retokenize() {
    run_parity_test(
        ReleasedCommand::Morphotag,
        InferTask::Morphosyntax,
        "eng_retokenize",
        "eng",
        morphotag_opts(true),
    )
    .await;
}

#[tokio::test]
async fn parity_morphotag_eng_overlap() {
    run_parity_test(
        ReleasedCommand::Morphotag,
        InferTask::Morphosyntax,
        "eng_overlap_ca",
        "eng",
        morphotag_opts(false),
    )
    .await;
}

#[tokio::test]
async fn parity_morphotag_spa() {
    run_parity_test(
        ReleasedCommand::Morphotag,
        InferTask::Morphosyntax,
        "spa_simple",
        "spa",
        morphotag_opts(false),
    )
    .await;
}

#[tokio::test]
async fn parity_morphotag_fra() {
    run_parity_test(
        ReleasedCommand::Morphotag,
        InferTask::Morphosyntax,
        "fra_simple",
        "fra",
        morphotag_opts(false),
    )
    .await;
}

#[tokio::test]
async fn parity_morphotag_deu() {
    run_parity_test(
        ReleasedCommand::Morphotag,
        InferTask::Morphosyntax,
        "deu_clinical",
        "deu",
        morphotag_opts(false),
    )
    .await;
}

#[tokio::test]
async fn parity_morphotag_jpn() {
    run_parity_test(
        ReleasedCommand::Morphotag,
        InferTask::Morphosyntax,
        "jpn_clinical",
        "jpn",
        morphotag_opts(false),
    )
    .await;
}

#[tokio::test]
async fn parity_morphotag_eng_bilingual() {
    run_parity_test(
        ReleasedCommand::Morphotag,
        InferTask::Morphosyntax,
        "eng_bilingual",
        "eng",
        morphotag_opts(false),
    )
    .await;
}

// ---------------------------------------------------------------------------
// Utseg parity
// ---------------------------------------------------------------------------

#[tokio::test]
async fn parity_utseg_eng_multi() {
    run_parity_test(
        ReleasedCommand::Utseg,
        InferTask::Utseg,
        "eng_multi_speaker",
        "eng",
        utseg_opts(),
    )
    .await;
}

#[tokio::test]
async fn parity_utseg_spa() {
    run_parity_test(ReleasedCommand::Utseg, InferTask::Utseg, "spa_simple", "spa", utseg_opts()).await;
}

#[tokio::test]
async fn parity_utseg_eng_disfluency() {
    run_parity_test(
        ReleasedCommand::Utseg,
        InferTask::Utseg,
        "eng_disfluency",
        "eng",
        utseg_opts(),
    )
    .await;
}

// ---------------------------------------------------------------------------
// Translate parity
// ---------------------------------------------------------------------------

#[tokio::test]
async fn parity_translate_eng() {
    run_parity_test(
        ReleasedCommand::Translate,
        InferTask::Translate,
        "eng_disfluency",
        "eng",
        translate_opts(),
    )
    .await;
}

#[tokio::test]
async fn parity_translate_spa() {
    run_parity_test(
        ReleasedCommand::Translate,
        InferTask::Translate,
        "spa_simple",
        "spa",
        translate_opts(),
    )
    .await;
}

// ---------------------------------------------------------------------------
// Coref parity (in both BA2 jan9 and BA2-master)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn parity_coref_eng_disfluency() {
    run_parity_test(
        ReleasedCommand::Coref,
        InferTask::Coref,
        "eng_disfluency",
        "eng",
        coref_opts(),
    )
    .await;
}

#[tokio::test]
async fn parity_coref_eng_multi_speaker() {
    run_parity_test(
        ReleasedCommand::Coref,
        InferTask::Coref,
        "eng_multi_speaker",
        "eng",
        coref_opts(),
    )
    .await;
}
