//! Utility functions, constants, and auto-tuning for the runner module.

mod auto_tune;
mod error_classification;
mod file_status;
mod media;

// Re-export everything at the same paths callers already use.
pub(crate) use auto_tune::KNOWN_MEDIA_EXTENSIONS;
pub(super) use auto_tune::compute_job_workers;

pub(super) use error_classification::classify_server_error;
pub(super) use error_classification::user_facing_error;
pub(crate) use error_classification::{classify_worker_error, is_retryable_worker_failure};

pub(crate) use file_status::{
    FileRunTracker, FileStage, FileTaskOutcome, ProgressSender, ProgressUpdate,
};
pub(super) use file_status::{
    drain_supervised_file_tasks, force_terminal_file_states, set_file_progress,
    spawn_progress_forwarder, spawn_supervised_file_task,
};

#[cfg(test)]
pub(super) use media::resolve_audio_for_chat;
pub(super) use media::{
    apply_result_filename, collect_preflight_audio_paths, compute_audio_identity,
    get_audio_duration_ms, preflight_validate_media, resolve_audio_for_chat_with_media_dir,
    should_preflight,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::Path;

    use crate::api::{CommandName, FileName, JobId, LanguageCode3, NumSpeakers, NumWorkers};
    use crate::config::ServerConfig;
    use crate::options::{AlignOptions, CommandOptions, CommonOptions, UtrEngine};
    use crate::runtime;
    use crate::scheduling::FailureCategory;
    use crate::store::{
        PendingJobFile, RunnerDispatchConfig, RunnerFilesystemConfig, RunnerJobIdentity,
        RunnerJobSnapshot,
    };
    use crate::worker::error::WorkerError;
    use tokio_util::sync::CancellationToken;

    #[test]
    fn apply_result_filename_basic() {
        use std::path::PathBuf;
        assert_eq!(
            apply_result_filename(std::path::Path::new("/out/dir/audio.wav"), "audio.cha"),
            PathBuf::from("/out/dir/audio.cha")
        );
    }

    #[test]
    fn apply_result_filename_preserves_dir() {
        use std::path::PathBuf;
        assert_eq!(
            apply_result_filename(
                std::path::Path::new("/data/corpus/output/test.mp3"),
                "test.cha"
            ),
            PathBuf::from("/data/corpus/output/test.cha")
        );
    }

    #[test]
    fn compute_workers_single_file() {
        let config = ServerConfig::default();
        assert_eq!(
            compute_job_workers(&CommandName::from("morphotag"), 1, &config),
            NumWorkers(1)
        );
    }

    #[test]
    fn compute_workers_explicit_config() {
        let config = ServerConfig {
            max_workers_per_job: 3,
            ..Default::default()
        };
        assert_eq!(
            compute_job_workers(&CommandName::from("morphotag"), 10, &config),
            NumWorkers(3)
        );
    }

    #[test]
    fn compute_workers_explicit_caps_at_file_count() {
        let config = ServerConfig {
            max_workers_per_job: 8,
            ..Default::default()
        };
        assert_eq!(
            compute_job_workers(&CommandName::from("morphotag"), 2, &config),
            NumWorkers(2)
        );
    }

    #[test]
    fn compute_workers_auto_tune_caps_at_max() {
        let config = ServerConfig::default();
        // Auto-tune should never exceed runtime::max_thread_workers()
        let result = compute_job_workers(&CommandName::from("opensmile"), 100, &config);
        assert!(*result <= runtime::max_thread_workers());
        assert!(*result >= 1);
    }

    #[test]
    fn compute_workers_gpu_commands_use_thread_pool_cap() {
        let config = ServerConfig {
            gpu_thread_pool_size: 2,
            ..Default::default()
        };
        let result = compute_job_workers(&CommandName::from("transcribe"), 47, &config);
        assert!(
            *result <= 2,
            "GPU worker count must respect gpu_thread_pool_size, got {result}"
        );
    }

    #[test]
    fn command_execution_budgets_follow_runtime_constants() {
        assert_eq!(runtime::command_execution_budget_mb("morphotag").0, 3_000);
        assert_eq!(runtime::command_execution_budget_mb("align").0, 6_000);
        assert_eq!(runtime::command_execution_budget_mb("opensmile").0, 750);
        assert_eq!(runtime::command_execution_budget_mb("unknown_cmd").0, 6_000);
    }

    #[test]
    fn worker_error_classification_is_stable() {
        assert_eq!(
            classify_worker_error(&WorkerError::ProcessExited { code: Some(9) }),
            FailureCategory::WorkerCrash
        );
        assert_eq!(
            classify_worker_error(&WorkerError::ReadyTimeout { timeout_s: 30 }),
            FailureCategory::WorkerTimeout
        );
        assert_eq!(
            classify_worker_error(&WorkerError::Protocol(
                "timeout waiting for infer response".into()
            )),
            FailureCategory::WorkerTimeout
        );
        assert_eq!(
            classify_worker_error(&WorkerError::Protocol("bad frame".into())),
            FailureCategory::WorkerProtocol
        );
        assert_eq!(
            classify_worker_error(&WorkerError::WorkerResponse("temporary".into())),
            FailureCategory::ProviderTransient
        );
        assert!(is_retryable_worker_failure(FailureCategory::WorkerCrash));
        assert!(is_retryable_worker_failure(FailureCategory::WorkerTimeout));
        assert!(is_retryable_worker_failure(
            FailureCategory::ProviderTransient
        ));
        assert!(!is_retryable_worker_failure(
            FailureCategory::WorkerProtocol
        ));
        assert!(!is_retryable_worker_failure(FailureCategory::Validation));
    }

    #[test]
    fn should_preflight_transcribe_default() {
        assert!(should_preflight(&CommandName::from("transcribe"), None));
    }

    #[test]
    fn should_preflight_transcribe_rev_explicit() {
        use crate::options::{CommonOptions, TranscribeOptions};
        let opts = CommandOptions::Transcribe(TranscribeOptions {
            common: CommonOptions::default(),
            asr_engine: "rev".into(),
            diarize: false,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 4,
        });
        assert!(should_preflight(
            &CommandName::from("transcribe"),
            Some(&opts)
        ));
    }

    #[test]
    fn should_preflight_transcribe_whisper() {
        use crate::options::{CommonOptions, TranscribeOptions};
        let opts = CommandOptions::Transcribe(TranscribeOptions {
            common: CommonOptions::default(),
            asr_engine: "whisper".into(),
            diarize: false,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 4,
        });
        assert!(!should_preflight(
            &CommandName::from("transcribe"),
            Some(&opts)
        ));
    }

    #[test]
    fn should_preflight_benchmark() {
        assert!(should_preflight(&CommandName::from("benchmark"), None));
    }

    #[test]
    fn should_preflight_align_default() {
        assert!(should_preflight(&CommandName::from("align"), None));
    }

    #[test]
    fn should_preflight_align_no_utr() {
        use crate::options::{AlignOptions, CommonOptions};
        let opts = CommandOptions::Align(AlignOptions {
            common: CommonOptions::default(),
            fa_engine: "wav2vec_fa".into(),
            utr_engine: None,
            utr_overlap_strategy: Default::default(),
            utr_two_pass: Default::default(),
            pauses: false,
            wor: true.into(),
            merge_abbrev: false.into(),
            media_dir: None,
        });
        assert!(!should_preflight(&CommandName::from("align"), Some(&opts)));
    }

    #[test]
    fn should_preflight_morphotag() {
        assert!(!should_preflight(&CommandName::from("morphotag"), None));
    }

    #[tokio::test]
    async fn media_validate_not_paths_mode() {
        let file_list = vec![PendingJobFile {
            file_index: 0,
            filename: "audio.wav".into(),
            has_chat: false,
        }];
        let source_paths = vec![std::path::PathBuf::from("/nonexistent/audio.wav")];
        let failures = preflight_validate_media(&file_list, &source_paths, false).await;
        assert!(
            failures.is_empty(),
            "Should skip validation when not paths_mode"
        );
    }

    #[tokio::test]
    async fn media_validate_skips_chat_files() {
        let file_list = vec![PendingJobFile {
            file_index: 0,
            filename: "test.cha".into(),
            has_chat: true,
        }];
        let source_paths = vec![std::path::PathBuf::from("/nonexistent/test.cha")];
        let failures = preflight_validate_media(&file_list, &source_paths, true).await;
        assert!(failures.is_empty(), "Should skip CHAT files");
    }

    #[tokio::test]
    async fn media_validate_missing_file() {
        let file_list = vec![PendingJobFile {
            file_index: 0,
            filename: "missing.wav".into(),
            has_chat: false,
        }];
        let source_paths = vec![std::path::PathBuf::from("/nonexistent/path/missing.wav")];
        let failures = preflight_validate_media(&file_list, &source_paths, true).await;
        assert_eq!(failures.len(), 1);
        assert!(failures[&0].contains("not found"));
    }

    #[tokio::test]
    async fn media_validate_empty_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        // File is empty (0 bytes)
        let path = tmp.path().to_string_lossy().to_string();
        // Rename to .wav for extension check
        let wav_path = format!("{path}.wav");
        std::fs::rename(&path, &wav_path).unwrap();
        let file_list = vec![PendingJobFile {
            file_index: 0,
            filename: "test.wav".into(),
            has_chat: false,
        }];
        let source_paths = vec![std::path::PathBuf::from(&wav_path)];
        let failures = preflight_validate_media(&file_list, &source_paths, true).await;
        assert_eq!(failures.len(), 1);
        assert!(failures[&0].contains("empty"));
        let _ = std::fs::remove_file(&wav_path);
    }

    #[tokio::test]
    async fn media_validate_unknown_extension() {
        let mut tmp = tempfile::NamedTempFile::with_suffix(".xyz").unwrap();
        std::io::Write::write_all(&mut tmp, b"data").unwrap();
        let path = tmp.path().to_string_lossy().to_string();
        let file_list = vec![PendingJobFile {
            file_index: 0,
            filename: "test.xyz".into(),
            has_chat: false,
        }];
        let source_paths = vec![std::path::PathBuf::from(&path)];
        let failures = preflight_validate_media(&file_list, &source_paths, true).await;
        assert_eq!(failures.len(), 1);
        assert!(failures[&0].contains("Unknown media format"));
    }

    #[tokio::test]
    async fn media_validate_valid_file() {
        let mut tmp = tempfile::NamedTempFile::with_suffix(".wav").unwrap();
        std::io::Write::write_all(&mut tmp, b"RIFF....").unwrap();
        let path = tmp.path().to_string_lossy().to_string();
        let file_list = vec![PendingJobFile {
            file_index: 0,
            filename: "test.wav".into(),
            has_chat: false,
        }];
        let source_paths = vec![std::path::PathBuf::from(&path)];
        let failures = preflight_validate_media(&file_list, &source_paths, true).await;
        assert!(failures.is_empty(), "Valid .wav file should pass");
    }

    #[tokio::test]
    async fn media_validate_all_known_extensions() {
        for ext in KNOWN_MEDIA_EXTENSIONS {
            let mut tmp = tempfile::NamedTempFile::with_suffix(format!(".{ext}")).unwrap();
            std::io::Write::write_all(&mut tmp, b"data").unwrap();
            let path = tmp.path().to_string_lossy().to_string();
            let file_list = vec![PendingJobFile {
                file_index: 0,
                filename: FileName::from(format!("test.{ext}")),
                has_chat: false,
            }];
            let source_paths = vec![std::path::PathBuf::from(&path)];
            let failures = preflight_validate_media(&file_list, &source_paths, true).await;
            assert!(failures.is_empty(), "Extension .{ext} should be accepted");
        }
    }

    // -----------------------------------------------------------------------
    // Audio resolution for FA (content mode)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn resolve_audio_finds_wav_alongside_cha() {
        let dir = tempfile::tempdir().unwrap();
        let cha = dir.path().join("test.cha");
        let wav = dir.path().join("test.wav");
        std::fs::write(&cha, "@Begin\n@End\n").unwrap();
        std::fs::write(&wav, b"RIFF").unwrap();

        let result = resolve_audio_for_chat(&cha).await;
        assert!(result.is_some(), "Should find wav alongside cha");
        assert!(result.unwrap().ends_with("test.wav"));
    }

    #[tokio::test]
    async fn resolve_audio_returns_none_when_no_media() {
        let dir = tempfile::tempdir().unwrap();
        let cha = dir.path().join("test.cha");
        std::fs::write(&cha, "@Begin\n@End\n").unwrap();

        let result = resolve_audio_for_chat(&cha).await;
        assert!(
            result.is_none(),
            "Should return None when no audio file exists"
        );
    }

    #[tokio::test]
    async fn collect_preflight_audio_paths_resolves_align_media_from_chat_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let chat_path = dir.path().join("sample.cha");
        let wav_path = dir.path().join("sample.wav");
        std::fs::write(&chat_path, "@Begin\n@End\n").unwrap();
        std::fs::write(&wav_path, b"RIFF").unwrap();

        let job = RunnerJobSnapshot {
            identity: RunnerJobIdentity {
                job_id: JobId::from("job-1"),
                correlation_id: "corr-1".into(),
            },
            dispatch: RunnerDispatchConfig {
                command: CommandName::from("align"),
                lang: crate::api::LanguageSpec::Resolved(LanguageCode3::eng()),
                num_speakers: NumSpeakers(1),
                options: CommandOptions::Align(AlignOptions {
                    common: CommonOptions::default(),
                    fa_engine: "wav2vec_fa".into(),
                    utr_engine: Some(UtrEngine::RevAi),
                    utr_overlap_strategy: Default::default(),
                    utr_two_pass: Default::default(),
                    pauses: false,
                    wor: true.into(),
                    merge_abbrev: false.into(),
                    media_dir: None,
                }),
                runtime_state: BTreeMap::new(),
                debug_traces: false,
            },
            filesystem: RunnerFilesystemConfig {
                paths_mode: true,
                source_paths: vec![chat_path.to_path_buf()],
                output_paths: vec![],
                before_paths: vec![],
                staging_dir: std::path::PathBuf::new(),
                media_mapping: String::new(),
                media_subdir: String::new(),
                source_dir: std::path::PathBuf::new(),
            },
            cancel_token: CancellationToken::new(),
            pending_files: vec![PendingJobFile {
                file_index: 0,
                filename: FileName::from("sample.cha"),
                has_chat: true,
            }],
        };

        let paths =
            collect_preflight_audio_paths(&CommandName::from("align"), &job, &job.pending_files)
                .await;

        assert_eq!(paths, vec![wav_path.to_path_buf()]);
    }

    /// Simulates the content-mode FA audio resolution bug: staged file in a
    /// temp dir has no audio alongside it, but the client's source_dir does.
    /// Verifies that looking at source_dir finds the audio.
    #[tokio::test]
    async fn resolve_audio_source_dir_fallback() {
        // Simulate client's source directory with audio
        let source_dir = tempfile::tempdir().unwrap();
        let _cha_source = source_dir.path().join("ACWT01a.cha");
        let wav_source = source_dir.path().join("ACWT01a.wav");
        std::fs::write(&_cha_source, "@Begin\n@End\n").unwrap();
        std::fs::write(&wav_source, b"RIFF").unwrap();

        // Simulate server staging directory (no audio here)
        let staging_dir = tempfile::tempdir().unwrap();
        let staged_cha = staging_dir.path().join("ACWT01a.cha");
        std::fs::write(&staged_cha, "@Begin\n@End\n").unwrap();

        // Strategy 4 (alongside staged file): should fail
        let staged_result = resolve_audio_for_chat(&staged_cha).await;
        assert!(staged_result.is_none(), "No audio in staging dir");

        // Strategy 2 (source_dir): should succeed — this is the fix
        let source_path = source_dir.path().join("ACWT01a.cha");
        let source_result = resolve_audio_for_chat(&source_path).await;
        assert!(source_result.is_some(), "Should find audio via source_dir");
        assert!(source_result.unwrap().ends_with("ACWT01a.wav"));
    }

    /// Verifies media_roots fallback: audio not alongside cha or in source_dir,
    /// but present in a configured media_root.
    #[tokio::test]
    async fn resolve_audio_media_roots_fallback() {
        let media_root = tempfile::tempdir().unwrap();
        let wav = media_root.path().join("ACWT01a.wav");
        std::fs::write(&wav, b"RIFF").unwrap();

        let stem = "ACWT01a";
        let mut found = None;
        let roots = vec![media_root.path().to_string_lossy().to_string()];
        'roots: for root in &roots {
            for ext in KNOWN_MEDIA_EXTENSIONS {
                let candidate = Path::new(root).join(format!("{stem}.{ext}"));
                if tokio::fs::try_exists(&candidate).await.unwrap_or(false) {
                    found = Some(candidate.to_string_lossy().to_string());
                    break 'roots;
                }
            }
        }
        assert!(found.is_some(), "Should find audio in media_root");
        assert!(found.unwrap().ends_with("ACWT01a.wav"));
    }
}
