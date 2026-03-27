//! Focused checks for the real-model live-server fixture.
//!
//! These tests are intentionally server-specific. They verify the fixture and
//! HTTP-facing control-plane invariants that direct execution does not have:
//! - prepared workers stay warm across isolated server sessions
//! - each acquired session gets a fresh runtime layout with no prior jobs

use crate::common::{
    LiveServerSession, assert_completed_without_errors, require_live_server, submit_and_complete,
};
use batchalign_app::api::{FilePayload, ReleasedCommand};
use batchalign_app::options::{
    CommandOptions, CommonOptions, CorefOptions, MorphotagOptions, TranslateOptions, UtsegOptions,
};
use batchalign_app::worker::InferTask;

/// Minimal English CHAT sample for morphotag fixture checks.
const ENG_SIMPLE: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\thello world .
@End
";

/// Multi-utterance sample for utseg live-fixture checks.
const ENG_MULTI_UTT: &str = "\
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

/// Multi-sentence sample for coref live-fixture checks.
const ENG_COREF: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tthe dog ran .
*CHI:\tit was fast .
*CHI:\tthe cat slept .
@End
";

/// The fixture should reuse the same warmed worker process across isolated sessions.
#[tokio::test]
async fn live_fixture_reuses_warmed_workers_across_sessions() {
    let Some(first) = LiveServerSession::acquire().await else {
        return;
    };
    if !first.has_infer_task(InferTask::Morphosyntax) {
        eprintln!("SKIP: live fixture does not support morphosyntax infer");
        return;
    }

    let first_health = first.health().await;
    assert!(
        first_health
            .loaded_pipelines
            .iter()
            .any(|pipeline| pipeline.contains("infer:morphosyntax:eng")),
        "expected the live fixture to pre-warm an English morphosyntax worker"
    );
    let first_pipelines = first_health.loaded_pipelines.clone();
    first.close().await;

    let Some(second) = LiveServerSession::acquire().await else {
        return;
    };
    if !second.has_infer_task(InferTask::Morphosyntax) {
        eprintln!("SKIP: live fixture does not support morphosyntax infer");
        return;
    }

    let second_health = second.health().await;
    let reused: Vec<&String> = second_health
        .loaded_pipelines
        .iter()
        .filter(|pipeline| first_pipelines.contains(*pipeline))
        .collect();
    assert!(
        !reused.is_empty(),
        "expected at least one warmed worker process to survive across isolated sessions"
    );
    second.close().await;
}

/// The fixture should provide a fresh runtime layout and empty job store each time.
#[tokio::test]
async fn live_fixture_isolates_runtime_state_between_sessions() {
    let Some(first) = LiveServerSession::acquire().await else {
        return;
    };
    if !first.has_infer_task(InferTask::Morphosyntax) {
        eprintln!("SKIP: live fixture does not support morphosyntax infer");
        return;
    }

    let first_state_dir = first.state_dir().to_path_buf();
    assert!(
        first_state_dir.join("jobs").exists(),
        "fixture session should own an explicit jobs directory"
    );

    let files = vec![FilePayload {
        filename: "fixture-isolation.cha".into(),
        content: ENG_SIMPLE.into(),
    }];
    let options = CommandOptions::Morphotag(MorphotagOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        retokenize: false,
        skipmultilang: false,
        merge_abbrev: false.into(),
    });
    let (job, results) = submit_and_complete(
        first.client(),
        first.base_url(),
        ReleasedCommand::Morphotag,
        "eng",
        files,
        options,
    )
    .await;
    assert_completed_without_errors("live_fixture_isolation", &job, &results);

    let first_jobs = first.list_jobs().await;
    assert_eq!(
        first_jobs.len(),
        1,
        "first session should see its submitted job"
    );
    first.close().await;

    let Some(second) = LiveServerSession::acquire().await else {
        return;
    };
    if !second.has_infer_task(InferTask::Morphosyntax) {
        eprintln!("SKIP: live fixture does not support morphosyntax infer");
        return;
    }

    assert_ne!(
        second.state_dir(),
        first_state_dir.as_path(),
        "each session should receive a fresh runtime-owned state directory"
    );
    let second_jobs = second.list_jobs().await;
    assert!(
        second_jobs.is_empty(),
        "fresh fixture session should start with an empty job listing"
    );
    let second_health = second.health().await;
    assert_eq!(
        second_health.active_jobs, 0,
        "fresh fixture session should not inherit active jobs"
    );
    second.close().await;
}

/// The fixture should run a second infer-only command family when that backend is available.
#[tokio::test]
async fn live_fixture_runs_utseg_job_when_available() {
    let Some(server) = require_live_server(
        InferTask::Utseg,
        "live fixture does not support utseg infer",
    )
    .await
    else {
        return;
    };

    let files = vec![FilePayload {
        filename: "fixture-utseg.cha".into(),
        content: ENG_MULTI_UTT.into(),
    }];
    let options = CommandOptions::Utseg(UtsegOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        merge_abbrev: false.into(),
    });

    let (info, results) = submit_and_complete(
        server.client(),
        server.base_url(),
        ReleasedCommand::Utseg,
        "eng",
        files,
        options,
    )
    .await;

    assert_completed_without_errors("live_fixture_utseg", &info, &results);
    assert_eq!(results.len(), 1);
    assert!(
        !results[0].content.is_empty(),
        "utseg output should not be empty"
    );
}

/// The fixture should run translate jobs when that backend is available.
#[tokio::test]
async fn live_fixture_runs_translate_job_when_available() {
    let Some(server) = require_live_server(
        InferTask::Translate,
        "live fixture does not support translate infer",
    )
    .await
    else {
        return;
    };

    let files = vec![FilePayload {
        filename: "fixture-translate.cha".into(),
        content: ENG_SIMPLE.into(),
    }];
    let options = CommandOptions::Translate(TranslateOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        merge_abbrev: false.into(),
    });

    let (info, results) = submit_and_complete(
        server.client(),
        server.base_url(),
        ReleasedCommand::Translate,
        "eng",
        files,
        options,
    )
    .await;

    assert_completed_without_errors("live_fixture_translate", &info, &results);
    assert_eq!(results.len(), 1);
    // Simple tier-presence smoke-check — %xtra is a user-defined extension tier
    // with no typed AST accessor. contains() is the pragmatic choice here.
    assert!(
        results[0].content.contains("%xtra:"),
        "translate output should contain %xtra tier"
    );
}

/// The fixture should run coref jobs when that backend is available.
#[tokio::test]
async fn live_fixture_runs_coref_job_when_available() {
    let Some(server) = require_live_server(
        InferTask::Coref,
        "live fixture does not support coref infer",
    )
    .await
    else {
        return;
    };

    let files = vec![FilePayload {
        filename: "fixture-coref.cha".into(),
        content: ENG_COREF.into(),
    }];
    let options = CommandOptions::Coref(CorefOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        merge_abbrev: false.into(),
    });

    let (info, results) = submit_and_complete(
        server.client(),
        server.base_url(),
        ReleasedCommand::Coref,
        "eng",
        files,
        options,
    )
    .await;

    assert_completed_without_errors("live_fixture_coref", &info, &results);
    assert_eq!(results.len(), 1);
    // Simple structural smoke-check — verifying the server produced CHAT
    // output with recognizable structure. Not semantic CHAT parsing.
    assert!(
        results[0].content.contains("@Begin") && results[0].content.contains("*CHI:"),
        "coref output should remain valid CHAT with CHI speaker"
    );
}
