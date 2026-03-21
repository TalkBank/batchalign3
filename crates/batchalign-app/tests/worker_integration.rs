//! Integration tests for the infer-era Python worker.
//!
//! These tests spawn real Python workers in `--test-echo` mode
//! (no ML models) and verify the Rust side can communicate over the infer-only
//! worker protocol.
//!
//! Requirements: Python 3 with batchalign installed.
//! Skip gracefully if unavailable.

mod common;

use batchalign_app::api::{LanguageCode3, NumSpeakers, WorkerLanguage};
use batchalign_app::worker::error::WorkerError;
use batchalign_app::worker::handle::{WorkerConfig, WorkerHandle};
use batchalign_app::worker::pool::{PoolConfig, WorkerPool};
use batchalign_app::worker::{BatchInferRequest, InferRequest, InferTask, WorkerProfile};
use common::resolve_python;
use serde_json::{Value, json};
use std::collections::BTreeMap;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

macro_rules! require_python {
    () => {{
        // Memory guard: skip test entirely if insufficient RAM to safely spawn workers.
        // This prevents kernel OOM panics that have crashed Franklin's machine repeatedly.
        let available_mb = batchalign_app::worker::memory_guard::available_memory_mb();
        if available_mb < 4096 {
            eprintln!(
                "SKIP: insufficient memory ({available_mb} MB available, 4096 MB required). \
                 Worker tests need at least 4 GB free RAM."
            );
            return;
        }
        match resolve_python() {
            Some(path) => path,
            None => {
                eprintln!("SKIP: Python 3 with batchalign not available");
                return;
            }
        }
    }};
}

fn infer_request(payload: Value) -> InferRequest {
    InferRequest {
        task: InferTask::Morphosyntax,
        lang: LanguageCode3::eng(),
        payload,
    }
}

fn batch_request(task: InferTask, items: Vec<Value>) -> BatchInferRequest {
    BatchInferRequest {
        task,
        lang: LanguageCode3::eng(),
        items,
        mwt: BTreeMap::new(),
    }
}

#[tokio::test]
async fn spawn_test_echo_worker() {
    let python = require_python!();
    let config = WorkerConfig {
        python_path: python,
        test_echo: true,
        profile: WorkerProfile::Stanza,
        lang: WorkerLanguage::from(LanguageCode3::eng()),
        ready_timeout_s: 30,
        ..Default::default()
    };

    let handle = WorkerHandle::spawn(config).await.expect("spawn failed");
    assert!(*handle.pid() > 0, "should have a valid pid");
    assert_eq!(handle.profile_label(), "profile:stanza");
    assert_eq!(handle.lang(), "eng");
    assert_eq!(handle.transport(), "stdio");

    handle.shutdown().await.expect("shutdown failed");
}

#[tokio::test]
async fn health_check_works() {
    let python = require_python!();
    let config = WorkerConfig {
        python_path: python,
        test_echo: true,
        profile: WorkerProfile::Stanza,
        lang: WorkerLanguage::from(LanguageCode3::eng()),
        ready_timeout_s: 30,
        ..Default::default()
    };

    let mut handle = WorkerHandle::spawn(config).await.expect("spawn failed");
    let health = handle.health_check().await.expect("health check failed");
    assert_eq!(health.status, batchalign_app::worker::WorkerHealthStatus::Ok);
    assert_eq!(health.command, "profile:stanza");
    assert_eq!(health.lang, WorkerLanguage::from(LanguageCode3::eng()));
    assert!(*health.pid > 0);

    handle.shutdown().await.expect("shutdown failed");
}

#[tokio::test]
async fn capabilities_test_echo() {
    let python = require_python!();
    let config = WorkerConfig {
        python_path: python,
        test_echo: true,
        profile: WorkerProfile::Stanza,
        lang: WorkerLanguage::from(LanguageCode3::eng()),
        ready_timeout_s: 30,
        ..Default::default()
    };

    let mut handle = WorkerHandle::spawn(config).await.expect("spawn failed");
    let caps = handle.capabilities().await.expect("capabilities failed");
    assert!(caps.commands.iter().any(|c| c == "test-echo"));
    assert!(caps.commands.iter().any(|c| c == "morphotag"));
    assert!(caps.infer_tasks.is_empty());
    assert!(caps.engine_versions.is_empty());
    assert!(!caps.free_threaded);

    handle.shutdown().await.expect("shutdown failed");
}

#[tokio::test]
async fn infer_echo_returns_payload() {
    let python = require_python!();
    let config = WorkerConfig {
        python_path: python,
        test_echo: true,
        profile: WorkerProfile::Stanza,
        lang: WorkerLanguage::from(LanguageCode3::eng()),
        ready_timeout_s: 30,
        ..Default::default()
    };

    let mut handle = WorkerHandle::spawn(config).await.expect("spawn failed");
    let payload = json!({"words": ["hello", "world"], "lang": "eng"});
    let response = handle
        .infer(&infer_request(payload.clone()))
        .await
        .expect("infer failed");
    assert_eq!(response.result, Some(payload));
    assert!(response.error.is_none());

    handle.shutdown().await.expect("shutdown failed");
}

#[tokio::test]
async fn batch_infer_echo_returns_items() {
    let python = require_python!();
    let config = WorkerConfig {
        python_path: python,
        test_echo: true,
        profile: WorkerProfile::Stanza,
        lang: WorkerLanguage::from(LanguageCode3::eng()),
        ready_timeout_s: 30,
        ..Default::default()
    };

    let mut handle = WorkerHandle::spawn(config).await.expect("spawn failed");
    let items = vec![
        json!({"words": ["hello"], "lang": "eng"}),
        json!({"words": ["world"], "lang": "eng"}),
    ];
    let response = handle
        .batch_infer(&batch_request(InferTask::Morphosyntax, items.clone()))
        .await
        .expect("batch infer failed");
    assert_eq!(response.results.len(), 2);
    assert_eq!(response.results[0].result, Some(items[0].clone()));
    assert_eq!(response.results[1].result, Some(items[1].clone()));

    handle.shutdown().await.expect("shutdown failed");
}

#[tokio::test]
async fn pool_dispatch_batch_infer_spawns_and_processes() {
    let python = require_python!();
    let pool = WorkerPool::new(PoolConfig {
        python_path: python,
        health_check_interval_s: 60,
        idle_timeout_s: 300,
        ready_timeout_s: 30,
        test_echo: true,
        max_workers_per_key: 8,
        verbose: 0,
        engine_overrides: String::new(),
        runtime: Default::default(),
        ..Default::default()
    });

    let item = json!({"words": ["hello", "pool"], "lang": "eng"});
    let response = pool
        .dispatch_batch_infer(
            &LanguageCode3::eng(),
            &batch_request(InferTask::Morphosyntax, vec![item.clone()]),
        )
        .await
        .expect("dispatch failed");
    assert_eq!(response.results[0].result, Some(item));

    assert_eq!(pool.worker_count(), 1);
    let summary = pool.worker_summary();
    assert_eq!(summary.len(), 1);
    assert!(summary[0].starts_with("profile:stanza:eng:pid="));
    assert!(summary[0].contains(":transport=stdio"));

    pool.shutdown().await;
    assert_eq!(pool.worker_count(), 0);
}

#[tokio::test]
async fn pool_reuses_existing_worker() {
    let python = require_python!();
    let pool = WorkerPool::new(PoolConfig {
        python_path: python,
        health_check_interval_s: 60,
        idle_timeout_s: 300,
        ready_timeout_s: 30,
        test_echo: true,
        max_workers_per_key: 8,
        verbose: 0,
        engine_overrides: String::new(),
        runtime: Default::default(),
        ..Default::default()
    });

    for i in 0..3 {
        let item = json!({"request": i});
        let response = pool
            .dispatch_batch_infer(
                &LanguageCode3::eng(),
                &batch_request(InferTask::Morphosyntax, vec![item.clone()]),
            )
            .await
            .expect("dispatch failed");
        assert_eq!(response.results[0].result, Some(item));
    }

    assert_eq!(pool.worker_count(), 1);
    pool.shutdown().await;
}

#[tokio::test]
async fn pool_multiple_task_groups() {
    let python = require_python!();
    let pool = WorkerPool::new(PoolConfig {
        python_path: python,
        health_check_interval_s: 60,
        idle_timeout_s: 300,
        ready_timeout_s: 30,
        test_echo: true,
        max_workers_per_key: 8,
        verbose: 0,
        engine_overrides: String::new(),
        runtime: Default::default(),
        ..Default::default()
    });

    let morph_item = json!({"task": "morph"});
    let fa_item = json!({"task": "fa"});
    let r1 = pool
        .dispatch_batch_infer(
            &LanguageCode3::eng(),
            &batch_request(InferTask::Morphosyntax, vec![morph_item.clone()]),
        )
        .await
        .expect("dispatch 1 failed");
    let r2 = pool
        .dispatch_batch_infer(
            &LanguageCode3::eng(),
            &batch_request(InferTask::Fa, vec![fa_item.clone()]),
        )
        .await
        .expect("dispatch 2 failed");
    assert_eq!(r1.results[0].result, Some(morph_item));
    assert_eq!(r2.results[0].result, Some(fa_item));
    assert_eq!(pool.worker_count(), 2);

    pool.shutdown().await;
}

#[tokio::test]
async fn pool_warmup_uses_infer_targets() {
    let python = require_python!();
    let pool = WorkerPool::new(PoolConfig {
        python_path: python,
        health_check_interval_s: 60,
        idle_timeout_s: 300,
        ready_timeout_s: 30,
        test_echo: true,
        max_workers_per_key: 8,
        verbose: 0,
        engine_overrides: String::new(),
        runtime: Default::default(),
        ..Default::default()
    });

    pool.warmup(&[
        batchalign_app::server::WarmupTarget { command: "morphotag".into(), lang: WorkerLanguage::from(LanguageCode3::eng()) },
        batchalign_app::server::WarmupTarget { command: "align".into(), lang: WorkerLanguage::from(LanguageCode3::eng()) },
    ])
    .await;

    let summary = pool.worker_summary();
    // morphotag → Stanza profile (sequential group), align → GPU profile (SharedGpuWorker)
    assert_eq!(pool.worker_count(), 2);
    assert!(
        summary
            .iter()
            .any(|entry| entry.starts_with("profile:stanza:eng:")),
        "expected a Stanza profile worker in summary: {summary:?}"
    );
    assert!(
        summary
            .iter()
            .any(|entry| entry.starts_with("profile:gpu:eng:")),
        "expected a GPU profile worker in summary: {summary:?}"
    );

    pool.shutdown().await;
}

#[tokio::test]
async fn pool_pre_scale_respects_max_workers_per_key() {
    let python = require_python!();
    let pool = WorkerPool::new(PoolConfig {
        python_path: python,
        health_check_interval_s: 60,
        idle_timeout_s: 300,
        ready_timeout_s: 30,
        test_echo: true,
        max_workers_per_key: 2,
        verbose: 0,
        engine_overrides: String::new(),
        runtime: Default::default(),
        ..Default::default()
    });

    pool.pre_scale(&"morphotag".into(), WorkerLanguage::from(LanguageCode3::eng()), 4)
        .await;
    let count = pool.worker_count();
    assert!(
        count <= 2,
        "Expected at most 2 workers (max_workers_per_key=2), got {count}"
    );

    pool.shutdown().await;
}

#[cfg(unix)]
#[tokio::test]
async fn pool_serializes_worker_bootstrap_per_key() {
    let python = require_python!();
    let dir = tempfile::TempDir::new().expect("tempdir");
    let wrapped_python = dir.path().join("wrapped-python");
    std::fs::write(
        &wrapped_python,
        format!(
            "#!/bin/sh\nsleep 0.5\nexec \"{}\" \"$@\"\n",
            python.replace('"', "\\\"")
        ),
    )
    .expect("write wrapped python");
    let mut perms = std::fs::metadata(&wrapped_python)
        .expect("metadata")
        .permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&wrapped_python, perms).expect("chmod wrapped python");

    let pool = WorkerPool::new(PoolConfig {
        python_path: wrapped_python.to_string_lossy().into_owned(),
        health_check_interval_s: 60,
        idle_timeout_s: 300,
        ready_timeout_s: 30,
        test_echo: true,
        max_workers_per_key: 3,
        verbose: 0,
        engine_overrides: String::new(),
        runtime: Default::default(),
        ..Default::default()
    });

    let item1 = json!({"request": 1});
    let item2 = json!({"request": 2});
    let item3 = json!({"request": 3});
    let lang = LanguageCode3::eng();
    let request1 = batch_request(InferTask::Morphosyntax, vec![item1.clone()]);
    let request2 = batch_request(InferTask::Morphosyntax, vec![item2.clone()]);
    let request3 = batch_request(InferTask::Morphosyntax, vec![item3.clone()]);
    let started = tokio::time::Instant::now();
    let (r1, r2, r3) = tokio::join!(
        pool.dispatch_batch_infer(&lang, &request1),
        pool.dispatch_batch_infer(&lang, &request2),
        pool.dispatch_batch_infer(&lang, &request3),
    );
    let elapsed = started.elapsed();

    assert_eq!(
        r1.expect("dispatch 1 failed").results[0].result,
        Some(item1)
    );
    assert_eq!(
        r2.expect("dispatch 2 failed").results[0].result,
        Some(item2)
    );
    assert_eq!(
        r3.expect("dispatch 3 failed").results[0].result,
        Some(item3)
    );
    assert_eq!(pool.worker_count(), 3);
    assert!(
        elapsed >= std::time::Duration::from_millis(1100),
        "expected serialized bootstrap to take at least 1.1s, got {:?}",
        elapsed
    );

    pool.shutdown().await;
}

#[tokio::test]
async fn spawn_failure_bad_python_path() {
    let config = WorkerConfig {
        python_path: "/nonexistent/python3".to_string(),
        test_echo: true,
        profile: WorkerProfile::Stanza,
        lang: WorkerLanguage::from(LanguageCode3::eng()),
        num_speakers: NumSpeakers(1),
        ready_timeout_s: 5,
        ..Default::default()
    };

    let err = match WorkerHandle::spawn(config).await {
        Err(e) => e,
        Ok(_) => panic!("expected spawn to fail with bad python path"),
    };
    assert!(
        matches!(err, WorkerError::SpawnFailed(_)),
        "expected SpawnFailed, got: {err}"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn spawn_tolerates_non_json_stdout_preamble_before_ready() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let fake_python = dir.path().join("fake-python");
    std::fs::write(
        &fake_python,
        "#!/bin/sh\nprintf 'Downloading: \"https://example.invalid/model.pt\" to /tmp/model.pt\\n'\nprintf '{\"ready\":true,\"pid\":1234,\"transport\":\"stdio\"}\\n'\nsleep 30\n",
    )
    .expect("write fake python");
    let mut perms = std::fs::metadata(&fake_python)
        .expect("metadata")
        .permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&fake_python, perms).expect("chmod fake python");

    let config = WorkerConfig {
        python_path: fake_python.to_string_lossy().into_owned(),
        test_echo: true,
        profile: WorkerProfile::Stanza,
        lang: WorkerLanguage::from(LanguageCode3::eng()),
        num_speakers: NumSpeakers(1),
        ready_timeout_s: 5,
        ..Default::default()
    };

    let handle = WorkerHandle::spawn(config).await.expect("spawn failed");
    assert!(*handle.pid() > 0, "should have a valid pid");
    assert_eq!(handle.transport(), "stdio");
}

#[cfg(unix)]
#[tokio::test]
async fn spawn_failure_includes_worker_startup_stderr() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let fake_python = dir.path().join("fake-python");
    std::fs::write(
        &fake_python,
        "#!/bin/sh\nprintf 'synthetic worker startup failure\\n' >&2\nexit 23\n",
    )
    .expect("write fake python");
    let mut perms = std::fs::metadata(&fake_python)
        .expect("metadata")
        .permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&fake_python, perms).expect("chmod fake python");

    let config = WorkerConfig {
        python_path: fake_python.to_string_lossy().into_owned(),
        test_echo: true,
        profile: WorkerProfile::Stanza,
        lang: WorkerLanguage::from(LanguageCode3::eng()),
        num_speakers: NumSpeakers(1),
        ready_timeout_s: 5,
        ..Default::default()
    };

    let err = match WorkerHandle::spawn(config).await {
        Err(e) => e,
        Ok(_) => panic!("expected spawn to fail with synthetic stderr"),
    };
    match err {
        WorkerError::ReadyParseFailed(message) => {
            assert!(
                message.contains("worker closed stdout without emitting ready signal"),
                "missing ready failure detail: {message}"
            );
            assert!(
                message.contains("synthetic worker startup failure"),
                "missing worker stderr detail: {message}"
            );
        }
        other => panic!("expected ReadyParseFailed, got: {other}"),
    }
}

#[cfg(unix)]
#[tokio::test]
async fn health_check_tolerates_non_protocol_stdout_between_requests() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let fake_python = dir.path().join("fake-python");
    std::fs::write(
        &fake_python,
        "#!/bin/sh\nprintf '{\"ready\":true,\"pid\":1234,\"transport\":\"stdio\"}\\n'\nIFS= read -r req || exit 1\nprintf 'torch: loading checkpoint shards\\n'\nprintf '{\"op\":\"health\",\"response\":{\"status\":\"ok\",\"command\":\"profile:stanza\",\"lang\":\"eng\",\"pid\":1234,\"uptime_s\":0}}\\n'\nIFS= read -r req || exit 0\nprintf '{\"op\":\"shutdown\"}\\n'\n",
    )
    .expect("write fake python");
    let mut perms = std::fs::metadata(&fake_python)
        .expect("metadata")
        .permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&fake_python, perms).expect("chmod fake python");

    let config = WorkerConfig {
        python_path: fake_python.to_string_lossy().into_owned(),
        test_echo: true,
        profile: WorkerProfile::Stanza,
        lang: WorkerLanguage::from(LanguageCode3::eng()),
        num_speakers: NumSpeakers(1),
        ready_timeout_s: 5,
        ..Default::default()
    };

    let mut handle = WorkerHandle::spawn(config).await.expect("spawn failed");
    let health = handle.health_check().await.expect("health check failed");
    assert_eq!(health.status, batchalign_app::worker::WorkerHealthStatus::Ok);
    assert_eq!(health.command, "profile:stanza");
    assert_eq!(health.lang, WorkerLanguage::from(LanguageCode3::eng()));

    handle.shutdown().await.expect("shutdown failed");
}

/// Two different InferTasks within the same profile share one worker.
#[tokio::test]
async fn profile_groups_related_tasks_into_single_worker() {
    let python = require_python!();
    let pool = WorkerPool::new(PoolConfig {
        python_path: python,
        health_check_interval_s: 60,
        idle_timeout_s: 300,
        ready_timeout_s: 30,
        test_echo: true,
        max_workers_per_key: 8,
        verbose: 0,
        engine_overrides: String::new(),
        runtime: Default::default(),
        ..Default::default()
    });

    // Dispatch morphosyntax and utseg — both Stanza profile.
    let morph_item = json!({"task": "morph"});
    let utseg_item = json!({"task": "utseg"});
    pool.dispatch_batch_infer(
        &LanguageCode3::eng(),
        &batch_request(InferTask::Morphosyntax, vec![morph_item]),
    )
    .await
    .expect("morphosyntax dispatch failed");
    pool.dispatch_batch_infer(
        &LanguageCode3::eng(),
        &batch_request(InferTask::Utseg, vec![utseg_item]),
    )
    .await
    .expect("utseg dispatch failed");

    // Both should use the same Stanza worker — only 1 worker total.
    assert_eq!(
        pool.worker_count(),
        1,
        "morphosyntax and utseg should share a single Stanza profile worker"
    );

    pool.shutdown().await;
}

/// Three different profiles produce exactly three workers.
#[tokio::test]
async fn each_profile_gets_its_own_worker() {
    let python = require_python!();
    let pool = WorkerPool::new(PoolConfig {
        python_path: python,
        health_check_interval_s: 60,
        idle_timeout_s: 300,
        ready_timeout_s: 30,
        test_echo: true,
        max_workers_per_key: 8,
        verbose: 0,
        engine_overrides: String::new(),
        runtime: Default::default(),
        ..Default::default()
    });

    // Dispatch one task from each profile via batch_infer.
    pool.dispatch_batch_infer(
        &LanguageCode3::eng(),
        &batch_request(InferTask::Morphosyntax, vec![json!({"p": "stanza"})]),
    )
    .await
    .expect("stanza dispatch failed");
    pool.dispatch_batch_infer(
        &LanguageCode3::eng(),
        &batch_request(InferTask::Translate, vec![json!({"p": "io"})]),
    )
    .await
    .expect("io dispatch failed");
    pool.dispatch_batch_infer(
        &LanguageCode3::eng(),
        &batch_request(InferTask::Fa, vec![json!({"p": "gpu"})]),
    )
    .await
    .expect("gpu dispatch failed");

    // Three different profiles -> three workers.
    assert_eq!(
        pool.worker_count(),
        3,
        "expected one worker per profile (Stanza, IO, GPU)"
    );

    pool.shutdown().await;
}
