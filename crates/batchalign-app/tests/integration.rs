//! Integration tests for batchalign-server.
//!
//! These tests spin up a real HTTP server with test-echo workers
//! (no ML models), submit jobs, and verify responses.
//!
//! Requirements: Python 3 with batchalign installed.
//! Tests skip gracefully if unavailable.

mod common;

use batchalign_app::api::{
    FilePayload, FileResult, HealthResponse, HealthStatus, JobInfo, JobListItem, JobResultResponse,
    JobStatus, JobSubmission, LanguageCode3, LanguageSpec, MemoryMb, NumSpeakers, ReleasedCommand,
};
use batchalign_app::config::ServerConfig;
use batchalign_app::create_app;
use batchalign_app::options::{CommandOptions, CommonOptions, TranscribeOptions};
use batchalign_app::worker::pool::PoolConfig;
use common::resolve_python;
use tokio::sync::{Semaphore, SemaphorePermit};

/// Serialize real-server integration tests so concurrent test workers do not
/// fight over Python subprocesses, sockets, and filesystem-backed state.
///
/// These tests intentionally exercise the full HTTP + worker stack. Running
/// many of them at once under Rust's default test parallelism can create
/// resource-pressure flakes that are irrelevant to the behavior under test.
static SERVER_TEST_SLOTS: Semaphore = Semaphore::const_new(1);

fn has_test_plugin(python_path: &str) -> bool {
    let check = r#"
import importlib.metadata as m
try:
    ep = next((ep for ep in m.entry_points(group="batchalign.plugins") if ep.name == "cantotag_test"), None)
    if ep is None:
        print(0)
    else:
        ep.load()
        print(1)
except Exception:
    print(0)
"#;
    let output = std::process::Command::new(python_path)
        .args(["-c", check])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let s = String::from_utf8_lossy(&out.stdout);
            s.trim() == "1"
        }
        _ => false,
    }
}

macro_rules! require_python {
    () => {
        match resolve_python() {
            Some(path) => path,
            None => {
                eprintln!("SKIP: Python 3 with batchalign not available");
                return;
            }
        }
    };
}

/// Start a test server on a random port and return the base URL.
async fn start_test_server(
    python_path: &str,
) -> (
    String,
    tempfile::TempDir,
    std::sync::Arc<batchalign_app::AppState>,
    SemaphorePermit<'static>,
) {
    let config = ServerConfig {
        host: "127.0.0.1".into(),
        port: 0, // Will pick a real port below
        job_ttl_days: 7,
        warmup_commands: vec![],
        memory_gate_mb: MemoryMb(0), // Disable memory gate in tests
        ..Default::default()
    };
    start_test_server_with_config(python_path, config).await
}

/// Start a test server with an explicit server config.
async fn start_test_server_with_config(
    python_path: &str,
    config: ServerConfig,
) -> (
    String,
    tempfile::TempDir,
    std::sync::Arc<batchalign_app::AppState>,
    SemaphorePermit<'static>,
) {
    let permit = SERVER_TEST_SLOTS
        .acquire()
        .await
        .expect("integration test semaphore should stay open");
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let jobs_dir = tmp.path().join("jobs");
    std::fs::create_dir_all(&jobs_dir).expect("mkdir jobs");

    let pool_config = PoolConfig {
        python_path: python_path.into(),
        test_echo: true,
        health_check_interval_s: 600, // Effectively disable for tests
        idle_timeout_s: 600,
        ready_timeout_s: 30,
        max_workers_per_key: 8,
        verbose: 0,
        engine_overrides: String::new(),
        runtime: Default::default(),
        ..Default::default()
    };

    let db_dir = tmp.path().join("db");
    std::fs::create_dir_all(&db_dir).expect("mkdir db");

    let (router, state) = create_app(
        config,
        pool_config,
        Some(jobs_dir.to_string_lossy().into()),
        Some(db_dir),
        Some("test-build-hash".into()),
    )
    .await
    .expect("create_app");

    // Bind to port 0 to get a random available port
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let port = listener.local_addr().expect("local_addr").port();

    let base_url = format!("http://127.0.0.1:{port}");

    tokio::spawn(async move {
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .ok();
    });

    // Brief pause to let the server start accepting connections
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    (base_url, tmp, state, permit)
}

fn test_submission(files: Vec<FilePayload>) -> JobSubmission {
    JobSubmission {
        command: ReleasedCommand::Transcribe,
        lang: LanguageSpec::Resolved(LanguageCode3::eng()),
        num_speakers: NumSpeakers(1),
        files,
        media_files: vec![],
        media_mapping: String::new(),
        media_subdir: String::new(),
        source_dir: String::new(),
        options: CommandOptions::Transcribe(TranscribeOptions {
            common: CommonOptions::default(),
            asr_engine: batchalign_app::options::AsrEngineName::RevAi,
            diarize: false,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 8,
        }),
        paths_mode: false,
        source_paths: vec![],
        output_paths: vec![],
        display_names: vec![],
        debug_traces: false,
        before_paths: vec![],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn health_endpoint() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_test_server(&python).await;

    let resp = reqwest::get(format!("{base_url}/health"))
        .await
        .expect("GET /health");
    assert_eq!(resp.status(), 200);

    let health: HealthResponse = resp.json().await.expect("parse health");
    assert_eq!(health.status, HealthStatus::Ok);
    assert!(!health.version.is_empty());
    assert_eq!(health.cache_backend, "sqlite");
}

#[tokio::test]
async fn submit_and_get_job() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_test_server(&python).await;

    let client = reqwest::Client::new();

    let submission = test_submission(vec![FilePayload {
        filename: "test.cha".into(),
        content: "@UTF8\n@Begin\n*CHI:\thello .\n@End\n".into(),
    }]);

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    assert_eq!(resp.status(), 200);

    let info: JobInfo = resp.json().await.expect("parse job info");
    assert_eq!(info.command, ReleasedCommand::Transcribe);
    assert_eq!(info.lang, LanguageSpec::Resolved(LanguageCode3::eng()));
    assert_eq!(info.total_files, 1);
    let job_id = info.job_id.clone();

    // Poll until the job finishes (test-echo is fast)
    let info = poll_job_done(&client, &base_url, &job_id).await;
    assert!(
        matches!(info.status, JobStatus::Completed),
        "Expected Completed, got {:?}",
        info.status
    );
    assert_eq!(info.completed_files, 1);
}

#[tokio::test]
async fn submit_job_echoes_request_id_header() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_test_server(&python).await;

    let client = reqwest::Client::new();
    let submission = test_submission(vec![FilePayload {
        filename: "hdr.cha".into(),
        content: "@UTF8\n@Begin\n*CHI:\thello .\n@End\n".into(),
    }]);

    let resp = client
        .post(format!("{base_url}/jobs"))
        .header("x-request-id", "external-trace-123")
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    assert_eq!(resp.status(), 200);

    let req_id = resp
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(req_id, "external-trace-123");
}

#[tokio::test]
async fn submit_job_sets_request_id_header_when_missing() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_test_server(&python).await;

    let client = reqwest::Client::new();
    let submission = test_submission(vec![FilePayload {
        filename: "fallback.cha".into(),
        content: "@UTF8\n@Begin\n*CHI:\thello .\n@End\n".into(),
    }]);

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    assert_eq!(resp.status(), 200);

    let req_id = resp
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(!req_id.is_empty());

    let info: JobInfo = resp.json().await.expect("parse job info");
    assert_eq!(req_id, info.job_id.to_string());
}

// Plugin command test removed: with ReleasedCommand (closed enum), unknown
// commands are rejected at JSON deserialization. See unknown_command_returns_422.

#[tokio::test]
async fn submit_and_get_results() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_test_server(&python).await;

    let client = reqwest::Client::new();

    let content = "@UTF8\n@Begin\n*CHI:\thello .\n@End\n";
    let submission = test_submission(vec![FilePayload {
        filename: "result_test.cha".into(),
        content: content.into(),
    }]);

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    let info: JobInfo = resp.json().await.expect("parse");
    let job_id = info.job_id;

    // Wait for completion
    poll_job_done(&client, &base_url, &job_id).await;

    // Get results
    let resp = client
        .get(format!("{base_url}/jobs/{job_id}/results"))
        .send()
        .await
        .expect("GET /jobs/{id}/results");
    assert_eq!(resp.status(), 200);

    let results: JobResultResponse = resp.json().await.expect("parse results");
    assert_eq!(results.job_id, job_id);
    assert!(matches!(results.status, JobStatus::Completed));
    assert!(!results.files.is_empty());

    // Test-echo worker echoes input — the result should contain content
    let first = &results.files[0];
    assert!(first.error.is_none());
    assert!(!first.content.is_empty());
}

#[tokio::test]
async fn get_single_result() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_test_server(&python).await;

    let client = reqwest::Client::new();

    let submission = test_submission(vec![FilePayload {
        filename: "single.cha".into(),
        content: "@UTF8\n@Begin\n*CHI:\tworld .\n@End\n".into(),
    }]);

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    let info: JobInfo = resp.json().await.expect("parse");
    let job_id = info.job_id;

    poll_job_done(&client, &base_url, &job_id).await;

    // Get single file result
    let resp = client
        .get(format!("{base_url}/jobs/{job_id}/results/single.cha"))
        .send()
        .await
        .expect("GET single result");
    assert_eq!(resp.status(), 200);

    let result: FileResult = resp.json().await.expect("parse file result");
    assert!(result.error.is_none());
    assert!(!result.content.is_empty());
}

#[tokio::test]
async fn list_jobs() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_test_server(&python).await;

    let client = reqwest::Client::new();

    // Submit two jobs
    for name in &["list_a.cha", "list_b.cha"] {
        let sub = test_submission(vec![FilePayload {
            filename: (*name).into(),
            content: "@UTF8\n@Begin\n*CHI:\thi .\n@End\n".into(),
        }]);
        client
            .post(format!("{base_url}/jobs"))
            .json(&sub)
            .send()
            .await
            .expect("POST /jobs");
    }

    let resp = client
        .get(format!("{base_url}/jobs"))
        .send()
        .await
        .expect("GET /jobs");
    assert_eq!(resp.status(), 200);

    let jobs: Vec<JobListItem> = resp.json().await.expect("parse job list");
    assert!(jobs.len() >= 2);
}

#[tokio::test]
async fn cancel_job() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_test_server(&python).await;

    let client = reqwest::Client::new();

    let submission = test_submission(vec![FilePayload {
        filename: "cancel.cha".into(),
        content: "@UTF8\n@Begin\n*CHI:\tcancel .\n@End\n".into(),
    }]);

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    let info: JobInfo = resp.json().await.expect("parse");
    let job_id = info.job_id;

    // Cancel immediately
    let resp = client
        .post(format!("{base_url}/jobs/{job_id}/cancel"))
        .send()
        .await
        .expect("POST cancel");
    assert_eq!(resp.status(), 200);

    // Verify status is cancelled (or possibly already completed if echo was faster)
    let resp = client
        .get(format!("{base_url}/jobs/{job_id}"))
        .send()
        .await
        .expect("GET job");
    let info: JobInfo = resp.json().await.expect("parse");
    assert!(matches!(
        info.status,
        JobStatus::Cancelled | JobStatus::Completed
    ));
}

#[tokio::test]
async fn delete_completed_job() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_test_server(&python).await;

    let client = reqwest::Client::new();

    let submission = test_submission(vec![FilePayload {
        filename: "delete.cha".into(),
        content: "@UTF8\n@Begin\n*CHI:\tdelete .\n@End\n".into(),
    }]);

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    let info: JobInfo = resp.json().await.expect("parse");
    let job_id = info.job_id;

    // Wait for completion
    poll_job_done(&client, &base_url, &job_id).await;

    // Delete
    let resp = client
        .delete(format!("{base_url}/jobs/{job_id}"))
        .send()
        .await
        .expect("DELETE");
    assert_eq!(resp.status(), 200);

    // Verify 404
    let resp = client
        .get(format!("{base_url}/jobs/{job_id}"))
        .send()
        .await
        .expect("GET deleted");
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn unknown_command_returns_422() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_test_server(&python).await;

    let client = reqwest::Client::new();

    // Send raw JSON with an invalid command string — ReleasedCommand is a
    // closed enum, so axum's JSON extractor rejects unknown variants at
    // deserialization time (HTTP 422).
    let raw = serde_json::json!({
        "command": "nonexistent_command",
        "lang": "eng",
        "num_speakers": 1,
        "files": [{"filename": "bad.cha", "content": "content"}],
        "media_files": [],
        "media_mapping": "",
        "media_subdir": "",
        "source_dir": "",
        "options": null,
    });

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&raw)
        .send()
        .await
        .expect("POST /jobs");
    assert_eq!(resp.status(), 422);
}

#[tokio::test]
async fn no_files_returns_400() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_test_server(&python).await;

    let client = reqwest::Client::new();
    let submission = test_submission(vec![]);

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn delete_running_job_returns_409() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_test_server(&python).await;

    let client = reqwest::Client::new();

    // Submit multiple files to increase the chance it's still running
    let files: Vec<FilePayload> = (0..5)
        .map(|i| FilePayload {
            filename: format!("file_{i}.cha").into(),
            content: "@UTF8\n@Begin\n*CHI:\thello .\n@End\n".into(),
        })
        .collect();
    let submission = test_submission(files);

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    assert_eq!(resp.status(), 200, "POST /jobs should succeed");
    let body = resp.text().await.expect("read body");
    let info: JobInfo =
        serde_json::from_str(&body).unwrap_or_else(|e| panic!("parse POST body: {e}\n{body}"));
    let job_id = info.job_id;

    // Try to delete immediately — might be running
    let resp = client
        .delete(format!("{base_url}/jobs/{job_id}"))
        .send()
        .await
        .expect("DELETE");

    let status = resp.status().as_u16();
    // It's either 409 (running) or 200 (already done, test-echo is fast)
    assert!(
        status == 409 || status == 200,
        "Expected 409 or 200, got {status}"
    );
}

#[tokio::test]
async fn job_not_found_returns_404() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_test_server(&python).await;

    let resp = reqwest::get(format!("{base_url}/jobs/nonexistent"))
        .await
        .expect("GET /jobs/nonexistent");
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn results_before_completion_returns_409() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_test_server(&python).await;

    let client = reqwest::Client::new();

    // Submit many files so it won't finish instantly
    let files: Vec<FilePayload> = (0..10)
        .map(|i| FilePayload {
            filename: format!("slow_{i}.cha").into(),
            content: "@UTF8\n@Begin\n*CHI:\thello .\n@End\n".into(),
        })
        .collect();
    let submission = test_submission(files);

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    assert_eq!(resp.status(), 200, "POST /jobs should succeed");
    let body = resp.text().await.expect("read body");
    let info: JobInfo =
        serde_json::from_str(&body).unwrap_or_else(|e| panic!("parse POST body: {e}\n{body}"));
    let job_id = info.job_id;

    // Immediately request results — should be 409 or 200 (race)
    let resp = client
        .get(format!("{base_url}/jobs/{job_id}/results"))
        .send()
        .await
        .expect("GET results");

    let status = resp.status().as_u16();
    // Accept either 409 (still running) or 200 (already done — test-echo is fast)
    assert!(
        status == 409 || status == 200,
        "Expected 409 or 200, got {status}"
    );
}

#[tokio::test]
async fn restart_failed_job() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_test_server(&python).await;

    let client = reqwest::Client::new();

    let submission = test_submission(vec![FilePayload {
        filename: "restart.cha".into(),
        content: "@UTF8\n@Begin\n*CHI:\trestart .\n@End\n".into(),
    }]);

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    let info: JobInfo = resp.json().await.expect("parse");
    let job_id = info.job_id;

    // Wait for completion
    poll_job_done(&client, &base_url, &job_id).await;

    // Cancel it first (so it's in a restartable state)
    // Actually, completed jobs can't be restarted — only cancelled/failed.
    // So let's submit and cancel quickly.
    let submission2 = test_submission(vec![FilePayload {
        filename: "restart2.cha".into(),
        content: "@UTF8\n@Begin\n*CHI:\trestart2 .\n@End\n".into(),
    }]);

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission2)
        .send()
        .await
        .expect("POST /jobs");
    let info: JobInfo = resp.json().await.expect("parse");
    let job_id2 = info.job_id;

    // Cancel it
    client
        .post(format!("{base_url}/jobs/{job_id2}/cancel"))
        .send()
        .await
        .expect("cancel");

    // Brief pause to let cancel propagate
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Check status - if it's cancelled, try restart
    let resp = client
        .get(format!("{base_url}/jobs/{job_id2}"))
        .send()
        .await
        .expect("GET job");
    let info: JobInfo = resp.json().await.expect("parse");

    if info.status == JobStatus::Cancelled {
        let resp = client
            .post(format!("{base_url}/jobs/{job_id2}/restart"))
            .send()
            .await
            .expect("restart");
        assert_eq!(resp.status(), 200);

        let restarted: JobInfo = resp.json().await.expect("parse restart");
        assert_eq!(restarted.status, JobStatus::Queued);

        // Wait for re-completion
        let final_info = poll_job_done(&client, &base_url, &job_id2).await;
        assert_eq!(final_info.status, JobStatus::Completed);
    }
    // If it completed before we could cancel, that's fine — test-echo is fast
}

#[tokio::test]
async fn restart_completed_job_returns_409() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_test_server(&python).await;

    let client = reqwest::Client::new();

    let submission = test_submission(vec![FilePayload {
        filename: "no_restart.cha".into(),
        content: "@UTF8\n@Begin\n*CHI:\tno_restart .\n@End\n".into(),
    }]);

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    let info: JobInfo = resp.json().await.expect("parse");
    let job_id = info.job_id;

    poll_job_done(&client, &base_url, &job_id).await;

    // Try to restart a completed job — should be 409
    let resp = client
        .post(format!("{base_url}/jobs/{job_id}/restart"))
        .send()
        .await
        .expect("restart");
    assert_eq!(resp.status(), 409);
}

#[tokio::test]
async fn paths_mode_job() {
    let python = require_python!();
    let (base_url, tmp, _state, _permit) = start_test_server(&python).await;

    let client = reqwest::Client::new();

    // Create input and output files for paths mode
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output");
    std::fs::create_dir_all(&input_dir).expect("mkdir input");
    std::fs::create_dir_all(&output_dir).expect("mkdir output");

    let input_path = input_dir.join("paths_test.cha");
    let output_path = output_dir.join("paths_test.cha");
    std::fs::write(&input_path, "@UTF8\n@Begin\n*CHI:\tpaths .\n@End\n").expect("write input");

    let submission = JobSubmission {
        command: ReleasedCommand::Transcribe,
        lang: LanguageSpec::Resolved(LanguageCode3::eng()),
        num_speakers: NumSpeakers(1),
        files: vec![],
        media_files: vec![],
        media_mapping: String::new(),
        media_subdir: String::new(),
        source_dir: String::new(),
        options: CommandOptions::Transcribe(TranscribeOptions {
            common: CommonOptions::default(),
            asr_engine: batchalign_app::options::AsrEngineName::RevAi,
            diarize: false,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 8,
        }),
        paths_mode: true,
        source_paths: vec![input_path.to_string_lossy().into()],
        output_paths: vec![output_path.to_string_lossy().into()],
        display_names: vec![],
        debug_traces: false,
        before_paths: vec![],
    };

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    assert_eq!(resp.status(), 200);

    let info: JobInfo = resp.json().await.expect("parse");
    assert_eq!(info.total_files, 1);
    let job_id = info.job_id;

    // Wait for completion
    let final_info = poll_job_done(&client, &base_url, &job_id).await;
    assert_eq!(final_info.status, JobStatus::Completed);
}

#[tokio::test]
async fn multiple_files_in_one_job() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_test_server(&python).await;

    let client = reqwest::Client::new();

    let files: Vec<FilePayload> = (0..3)
        .map(|i| FilePayload {
            filename: format!("multi_{i}.cha").into(),
            content: format!("@UTF8\n@Begin\n*CHI:\tfile{i} .\n@End\n"),
        })
        .collect();

    let submission = test_submission(files);

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    assert_eq!(resp.status(), 200);

    let info: JobInfo = resp.json().await.expect("parse");
    assert_eq!(info.total_files, 3);
    let job_id = info.job_id;

    let final_info = poll_job_done(&client, &base_url, &job_id).await;
    assert_eq!(final_info.status, JobStatus::Completed);
    assert_eq!(final_info.completed_files, 3);
}

#[tokio::test]
async fn multi_file_job_uses_parallel_workers() {
    let python = require_python!();
    let config = ServerConfig {
        host: "127.0.0.1".into(),
        port: 0,
        warmup_commands: vec![],
        max_workers_per_job: 3, // Force 3 parallel workers
        memory_gate_mb: MemoryMb(0),
        ..Default::default()
    };
    let (base_url, _tmp, _state, _permit) = start_test_server_with_config(&python, config).await;

    let client = reqwest::Client::new();

    // Submit 5 files in one job
    let files: Vec<FilePayload> = (0..5)
        .map(|i| FilePayload {
            filename: format!("parallel_{i}.cha").into(),
            content: format!("@UTF8\n@Begin\n*CHI:\tparallel{i} .\n@End\n"),
        })
        .collect();

    let submission = test_submission(files);

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    assert_eq!(resp.status(), 200);

    let info: JobInfo = resp.json().await.expect("parse");
    assert_eq!(info.total_files, 5);
    let job_id = info.job_id;

    // Wait for completion
    let final_info = poll_job_done(&client, &base_url, &job_id).await;
    assert_eq!(final_info.status, JobStatus::Completed);
    assert_eq!(final_info.completed_files, 5);

    // Verify num_workers was set (should be min(3, 5) = 3)
    assert!(
        final_info.num_workers.is_some(),
        "Expected num_workers to be set"
    );
    let nw = final_info.num_workers.unwrap();
    assert!(
        (1..=3).contains(&nw),
        "Expected num_workers in [1, 3], got {nw}"
    );

    // Verify all results are accessible
    let resp = client
        .get(format!("{base_url}/jobs/{job_id}/results"))
        .send()
        .await
        .expect("GET results");
    assert_eq!(resp.status(), 200);

    let results: JobResultResponse = resp.json().await.expect("parse results");
    assert_eq!(results.files.len(), 5);
    for file in &results.files {
        assert!(file.error.is_none(), "unexpected error: {:?}", file.error);
    }
}

// ---------------------------------------------------------------------------
// Capability gate: real worker (not test-echo)
// ---------------------------------------------------------------------------

/// Verify that create_app succeeds with a real Python worker whose import
/// probes (commands) are broader than its loaded infer tasks. Before the fix,
/// this would crash with "worker capability gate failed".
#[tokio::test]
async fn server_starts_with_real_worker_capability_gate() {
    let python_path = require_python!();

    let config = ServerConfig {
        host: "127.0.0.1".into(),
        port: 0,
        job_ttl_days: 7,
        warmup_commands: vec![],
        memory_gate_mb: MemoryMb(0),
        ..Default::default()
    };

    let tmp = tempfile::TempDir::new().expect("tempdir");
    let jobs_dir = tmp.path().join("jobs");
    std::fs::create_dir_all(&jobs_dir).expect("mkdir jobs");
    let db_dir = tmp.path().join("db");
    std::fs::create_dir_all(&db_dir).expect("mkdir db");

    // Real worker, NOT test-echo. This exercises the capability gate.
    let pool_config = PoolConfig {
        python_path: python_path.clone(),
        test_echo: false,
        health_check_interval_s: 600,
        idle_timeout_s: 600,
        ready_timeout_s: 60,
        max_workers_per_key: 8,
        verbose: 0,
        engine_overrides: String::new(),
        runtime: Default::default(),
        ..Default::default()
    };

    let result = create_app(
        config,
        pool_config,
        Some(jobs_dir.to_string_lossy().into()),
        Some(db_dir),
        Some("test-build-hash".into()),
    )
    .await;

    match result {
        Ok((router, state)) => {
            // Server started — verify capabilities were filtered, not rejected.
            assert!(
                !state.capabilities().is_empty(),
                "should have at least one capability"
            );
            eprintln!(
                "Server started OK. Capabilities: {:?}, Infer tasks: {:?}",
                state.capabilities(),
                state.infer_tasks()
            );

            // Quick health check
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind");
            let port = listener.local_addr().expect("local_addr").port();
            tokio::spawn(async move {
                axum::serve(
                    listener,
                    router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
                )
                .await
                .ok();
            });
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

            let client = reqwest::Client::new();
            let resp = client
                .get(format!("http://127.0.0.1:{port}/health"))
                .send()
                .await
                .expect("health request");
            assert!(resp.status().is_success());
            let health: HealthResponse = resp.json().await.expect("parse health");
            assert_eq!(health.status, HealthStatus::Ok);
        }
        Err(e) => {
            panic!(
                "create_app should succeed with real worker after capability gate fix, \
                 but failed: {e}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: poll until job reaches terminal state
// ---------------------------------------------------------------------------

async fn poll_job_done(client: &reqwest::Client, base_url: &str, job_id: &str) -> JobInfo {
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(60);
    let mut poll_count = 0u32;

    loop {
        let resp = client
            .get(format!("{base_url}/jobs/{job_id}"))
            .send()
            .await
            .expect("GET job");
        let status_code = resp.status();
        let body = resp.text().await.expect("read body");
        let info: JobInfo = serde_json::from_str(&body)
            .unwrap_or_else(|e| panic!("parse job failed (HTTP {status_code}): {e}\nbody: {body}"));

        poll_count += 1;
        if poll_count <= 3 || poll_count.is_multiple_of(50) {
            eprintln!(
                "  poll #{poll_count}: job={job_id} status={:?} completed={}/{}",
                info.status, info.completed_files, info.total_files
            );
        }

        if matches!(
            info.status,
            JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled
        ) {
            return info;
        }

        assert!(
            tokio::time::Instant::now() < deadline,
            "Job {job_id} did not finish within 60s (status: {:?})",
            info.status
        );

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }
}

// ---------------------------------------------------------------------------
// Concurrent jobs
// ---------------------------------------------------------------------------

/// Submit 3 jobs simultaneously, all should complete successfully.
#[tokio::test]
async fn concurrent_jobs_complete() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_test_server(&python).await;

    let client = reqwest::Client::new();

    // Submit 3 jobs concurrently
    let mut job_ids = Vec::new();
    for i in 0..3 {
        let sub = test_submission(vec![FilePayload {
            filename: format!("concurrent_{i}.cha").into(),
            content: format!("@UTF8\n@Begin\n*CHI:\tconcurrent{i} .\n@End\n"),
        }]);

        let resp = client
            .post(format!("{base_url}/jobs"))
            .json(&sub)
            .send()
            .await
            .expect("POST /jobs");
        assert_eq!(resp.status(), 200);
        let info: JobInfo = resp.json().await.expect("parse");
        job_ids.push(info.job_id.clone());
    }

    // Poll all 3 concurrently
    let (r1, r2, r3) = tokio::join!(
        poll_job_done(&client, &base_url, &job_ids[0]),
        poll_job_done(&client, &base_url, &job_ids[1]),
        poll_job_done(&client, &base_url, &job_ids[2]),
    );

    assert_eq!(r1.status, JobStatus::Completed, "Job 0 should complete");
    assert_eq!(r2.status, JobStatus::Completed, "Job 1 should complete");
    assert_eq!(r3.status, JobStatus::Completed, "Job 2 should complete");

    // Verify results for each job
    for (i, job_id) in job_ids.iter().enumerate() {
        let resp = client
            .get(format!("{base_url}/jobs/{job_id}/results"))
            .send()
            .await
            .expect("GET results");
        assert_eq!(resp.status(), 200);
        let results: JobResultResponse = resp.json().await.expect("parse results");
        assert_eq!(results.files.len(), 1, "Job {i} should have 1 result");
        assert!(
            results.files[0].error.is_none(),
            "Job {i} should have no error"
        );
    }
}
