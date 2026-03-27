//! Error path tests — verify graceful failures for bad inputs.
//!
//! Most tests in this module now use direct local execution with real workers so
//! model-hitting failures are exercised without the server control plane. The
//! malformed-request coverage stays on the HTTP path because it is explicitly a
//! request-validation test.
//!
//! Run: `cargo nextest run -p batchalign-app --test ml_golden --profile ml`

use crate::common::{require_live_direct, require_live_server, submit_and_complete_direct};
use batchalign_app::api::{
    FilePayload, JobStatus, JobSubmission, LanguageCode3, LanguageSpec, NumSpeakers,
    ReleasedCommand,
};
use batchalign_app::options::{AlignOptions, CommandOptions, CommonOptions, MorphotagOptions};
use batchalign_app::worker::InferTask;

// ---------------------------------------------------------------------------
// Error path tests
// ---------------------------------------------------------------------------

/// Align with missing audio should fail gracefully (not crash the server).
#[tokio::test]
async fn error_align_missing_audio() {
    let Some(session) =
        require_live_direct(InferTask::Fa, "Direct session does not support FA infer").await
    else {
        return;
    };

    // Create a CHAT file referencing @Media that does not exist on disk.
    let chat_with_media = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
@Media:\tnonexistent_audio, audio
*PAR:\thello world .
@End
";

    let input_path = session.state_dir().join("missing_audio.cha");
    let output_path = session.state_dir().join("missing_audio_out.cha");
    std::fs::write(&input_path, chat_with_media).expect("write input");

    let options = CommandOptions::Align(AlignOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        ..AlignOptions::default()
    });

    let submission = JobSubmission {
        command: ReleasedCommand::Align,
        lang: LanguageSpec::Resolved(LanguageCode3::eng()),
        num_speakers: NumSpeakers(1),
        files: vec![],
        media_files: vec![],
        media_mapping: String::new(),
        media_subdir: String::new(),
        source_dir: String::new(),
        options,
        paths_mode: true,
        source_paths: vec![input_path.to_string_lossy().into()],
        output_paths: vec![output_path.to_string_lossy().into()],
        display_names: vec![],
        debug_traces: false,
        before_paths: vec![],
    };

    let (final_info, _detail) = session.run_submission(submission).await;
    assert_eq!(
        final_info.status,
        JobStatus::Failed,
        "Align with missing audio should fail gracefully"
    );
}

/// Morphotag on an empty CHAT file should complete or fail gracefully.
#[tokio::test]
async fn error_morphotag_empty_file() {
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
            override_cache: true,
            ..CommonOptions::default()
        },
        retokenize: false,
        skipmultilang: false,
        merge_abbrev: false.into(),
    });

    // Minimal valid CHAT with no utterances.
    let empty_chat = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
@End
";

    let files = vec![FilePayload {
        filename: "empty.cha".into(),
        content: empty_chat.into(),
    }];

    let (info, results) =
        submit_and_complete_direct(&session, ReleasedCommand::Morphotag, "eng", files, options)
            .await;

    // Should complete (passthrough with no utterances to process) or fail gracefully.
    assert!(
        matches!(info.status, JobStatus::Completed | JobStatus::Failed),
        "Empty file should complete or fail gracefully, not crash"
    );

    // If completed, the output should still be valid CHAT.
    // Simple contains() smoke-check — not semantic CHAT parsing, just verifying
    // the server produced a structurally complete file with an @End marker.
    if info.status == JobStatus::Completed {
        assert!(!results.is_empty());
        let output = &results[0].content;
        assert!(
            output.contains("@End"),
            "Output should be valid CHAT (contains @End)"
        );
    }
}

/// Submitting a job with an invalid command name should be rejected.
///
/// This remains an HTTP-server test because the behavior under test is request
/// deserialization and validation at the `/jobs` boundary, not command execution.
#[tokio::test]
async fn error_invalid_command_name() {
    let Some(server) = require_live_server(
        InferTask::Morphosyntax,
        "Server does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };

    // Build a raw JSON submission with an invalid command in the options tag.
    // The internally-tagged CommandOptions enum should reject unknown commands.
    let body = serde_json::json!({
        "command": "nonexistent_command",
        "lang": "eng",
        "num_speakers": 1,
        "files": [{
            "filename": "test.cha",
            "content": "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello .\n@End\n"
        }],
        "media_files": [],
        "media_mapping": "",
        "media_subdir": "",
        "source_dir": "",
        "options": {
            "command": "nonexistent_command",
            "override_cache": false
        },
        "paths_mode": false,
        "source_paths": [],
        "output_paths": [],
        "display_names": [],
        "debug_traces": false,
        "before_paths": []
    });

    let resp = server
        .client()
        .post(format!("{}/jobs", server.base_url()))
        .json(&body)
        .send()
        .await
        .expect("POST /jobs");

    // Should be rejected with a 4xx status (400 or 422).
    let status = resp.status().as_u16();
    assert!(
        (400..500).contains(&status),
        "Invalid command should be rejected with 4xx, got {status}"
    );
}

// ---------------------------------------------------------------------------
// Edge case tests — unusual but valid CHAT
// ---------------------------------------------------------------------------

/// Morphotag on `xxx` (unintelligible) should pass through without crash.
#[tokio::test]
async fn edge_morphotag_xxx_utterance() {
    let Some(session) = require_live_direct(
        InferTask::Morphosyntax,
        "Direct session does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };

    let chat = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\txxx .
*PAR:\thello world .
@End
";

    let (info, results) = submit_and_complete_direct(
        &session,
        ReleasedCommand::Morphotag,
        "eng",
        vec![FilePayload {
            filename: "xxx_test.cha".into(),
            content: chat.into(),
        }],
        CommandOptions::Morphotag(MorphotagOptions {
            common: CommonOptions {
                override_cache: true,
                ..CommonOptions::default()
            },
            retokenize: false,
            skipmultilang: false,
            merge_abbrev: false.into(),
        }),
    )
    .await;

    assert!(
        matches!(info.status, JobStatus::Completed | JobStatus::Failed),
        "xxx utterance should not crash the server"
    );

    if info.status == JobStatus::Completed {
        let output = &results[0].content;
        // Simple contains() smoke-check — not semantic CHAT parsing, just
        // verifying the server produced a structurally complete file.
        assert!(output.contains("@End"), "Output should be valid CHAT");
    }
}

/// Morphotag on `www` (untranscribed speech) should pass through without crash.
#[tokio::test]
async fn edge_morphotag_www_utterance() {
    let Some(session) = require_live_direct(
        InferTask::Morphosyntax,
        "Direct session does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };

    let chat = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\twww .
*PAR:\thello world .
@End
";

    let (info, results) = submit_and_complete_direct(
        &session,
        ReleasedCommand::Morphotag,
        "eng",
        vec![FilePayload {
            filename: "www_test.cha".into(),
            content: chat.into(),
        }],
        CommandOptions::Morphotag(MorphotagOptions {
            common: CommonOptions {
                override_cache: true,
                ..CommonOptions::default()
            },
            retokenize: false,
            skipmultilang: false,
            merge_abbrev: false.into(),
        }),
    )
    .await;

    assert!(
        matches!(info.status, JobStatus::Completed | JobStatus::Failed),
        "www utterance should not crash the server"
    );

    if info.status == JobStatus::Completed {
        let output = &results[0].content;
        // Simple contains() smoke-check — not semantic CHAT parsing, just
        // verifying the server produced a structurally complete file.
        assert!(output.contains("@End"), "Output should be valid CHAT");
    }
}

/// Malformed CHAT (no @Begin) should fail gracefully.
#[tokio::test]
async fn error_morphotag_invalid_chat() {
    let Some(session) = require_live_direct(
        InferTask::Morphosyntax,
        "Direct session does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };

    let bad_chat = "\
@UTF8
@Languages:\teng
@Participants:\tPAR Participant
*PAR:\thello .
@End
";

    let (info, _results) = submit_and_complete_direct(
        &session,
        ReleasedCommand::Morphotag,
        "eng",
        vec![FilePayload {
            filename: "invalid.cha".into(),
            content: bad_chat.into(),
        }],
        CommandOptions::Morphotag(MorphotagOptions {
            common: CommonOptions {
                override_cache: true,
                ..CommonOptions::default()
            },
            retokenize: false,
            skipmultilang: false,
            merge_abbrev: false.into(),
        }),
    )
    .await;

    // Should complete or fail — never crash.
    assert!(
        matches!(info.status, JobStatus::Completed | JobStatus::Failed),
        "Invalid CHAT should not crash the server"
    );
}
