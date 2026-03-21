use super::*;

    use std::collections::{BTreeMap, HashMap, HashSet};
    use std::path::PathBuf;
    use std::sync::Arc;

    use tokio::sync::broadcast;
    use tokio_util::sync::CancellationToken;

    use crate::api::{
        CommandName, FileName, JobId, JobStatus, LanguageCode3, LanguageSpec, NumSpeakers,
        UnixTimestamp,
    };
    use crate::db::JobDB;
    use crate::options::{CommandOptions, CommonOptions, OpensmileOptions};
    use crate::scheduling::{AttemptOutcome, FailureCategory, WorkUnitKind};
    use crate::store::{
        FileStatus, Job, JobDispatchConfig, JobExecutionState, JobFilesystemConfig, JobIdentity,
        JobLeaseState, JobRuntimeControl, JobScheduleState, JobSourceContext, JobStore,
        PendingJobFile,
    };
    use crate::worker::InferTask;
    use crate::ws::BROADCAST_CAPACITY;

    use super::{
        command_requires_infer, infer_task_for_command, record_preflight_media_failures,
        result_filename_for_command,
    };

    /// Build a minimal paths-mode media job for prevalidation tests.
    fn make_media_job(job_id: &str, source_path: &str) -> Job {
        let filename = "missing.wav";
        let mut file_statuses = HashMap::new();
        file_statuses.insert(
            filename.to_string(),
            FileStatus::new(FileName::from(filename)),
        );

        Job {
            identity: JobIdentity {
                job_id: JobId::from(job_id),
                correlation_id: format!("test-{job_id}").into(),
            },
            dispatch: JobDispatchConfig {
                command: "opensmile".into(),
                lang: LanguageSpec::Resolved(LanguageCode3::eng()),
                num_speakers: NumSpeakers(1),
                options: CommandOptions::Opensmile(OpensmileOptions {
                    common: CommonOptions::default(),
                    feature_set: "eGeMAPSv02".into(),
                }),
                runtime_state: BTreeMap::new(),
                debug_traces: false,
            },
            source: JobSourceContext {
                submitted_by: "127.0.0.1".into(),
                submitted_by_name: "localhost".into(),
                source_dir: PathBuf::new(),
            },
            filesystem: JobFilesystemConfig {
                filenames: vec![FileName::from(filename)],
                has_chat: vec![false],
                staging_dir: PathBuf::new(),
                paths_mode: true,
                source_paths: vec![PathBuf::from(source_path)],
                output_paths: vec![PathBuf::new()],
                before_paths: Vec::new(),
                media_mapping: String::new(),
                media_subdir: String::new(),
            },
            execution: JobExecutionState {
                status: JobStatus::Queued,
                file_statuses,
                results: Vec::new(),
                error: None,
                completed_files: 0,
            },
            schedule: JobScheduleState {
                submitted_at: UnixTimestamp(1.0),
                completed_at: None,
                next_eligible_at: None,
                num_workers: None,
                lease: JobLeaseState {
                    leased_by_node: None,
                    expires_at: None,
                    heartbeat_at: None,
                },
            },
            runtime: JobRuntimeControl {
                cancel_token: CancellationToken::new(),
                runner_active: false,
            },
        }
    }

    #[test]
    fn infer_task_mapping_is_stable() {
        assert_eq!(
            infer_task_for_command(&CommandName::from("morphotag")),
            Some(InferTask::Morphosyntax)
        );
        assert_eq!(
            infer_task_for_command(&CommandName::from("utseg")),
            Some(InferTask::Utseg)
        );
        assert_eq!(
            infer_task_for_command(&CommandName::from("translate")),
            Some(InferTask::Translate)
        );
        assert_eq!(
            infer_task_for_command(&CommandName::from("coref")),
            Some(InferTask::Coref)
        );
        assert_eq!(
            infer_task_for_command(&CommandName::from("align")),
            Some(InferTask::Fa)
        );
        assert_eq!(
            infer_task_for_command(&CommandName::from("transcribe")),
            Some(InferTask::Asr)
        );
        assert_eq!(
            infer_task_for_command(&CommandName::from("compare")),
            Some(InferTask::Morphosyntax)
        );
        assert_eq!(
            infer_task_for_command(&CommandName::from("opensmile")),
            Some(InferTask::Opensmile)
        );
        assert_eq!(
            infer_task_for_command(&CommandName::from("avqi")),
            Some(InferTask::Avqi)
        );
        assert_eq!(
            infer_task_for_command(&CommandName::from("benchmark")),
            Some(InferTask::Asr)
        );
    }

    #[test]
    fn text_commands_always_require_infer() {
        for command in [
            "morphotag",
            "utseg",
            "translate",
            "coref",
            "opensmile",
            "avqi",
        ] {
            let cmd = CommandName::from(command);
            assert!(command_requires_infer(&cmd));
            assert!(command_requires_infer(&cmd));
        }
    }

    #[test]
    fn align_always_requires_infer() {
        let cmd = CommandName::from("align");
        assert!(command_requires_infer(&cmd));
        assert!(command_requires_infer(&cmd));
    }

    #[test]
    fn non_infer_commands_do_not_require_infer() {
        assert!(!command_requires_infer(
            &CommandName::from("transcribe"),
        ));
        assert!(!command_requires_infer(
            &CommandName::from("benchmark"),
        ));
    }

    #[test]
    fn transcribe_result_filename_preserves_relative_path() {
        assert_eq!(
            result_filename_for_command(&CommandName::from("transcribe"), "sub/nested.wav"),
            "sub/nested.cha"
        );
        assert_eq!(
            result_filename_for_command(&CommandName::from("transcribe_s"), "nested.mp3"),
            "nested.cha"
        );
    }

    #[test]
    fn non_transcribe_result_filename_is_unchanged() {
        assert_eq!(
            result_filename_for_command(&CommandName::from("morphotag"), "sub/nested.cha"),
            "sub/nested.cha"
        );
    }

    /// Preflight media validation should still leave a durable setup attempt so
    /// the file's failure appears in the attempt log instead of only in file
    /// status.
    #[tokio::test]
    async fn preflight_media_failure_records_setup_attempt() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let missing_path = tempdir.path().join("missing.wav");
        let db = Arc::new(JobDB::open(Some(tempdir.path())).await.expect("open db"));
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::new(crate::config::ServerConfig::default(), Some(db.clone()), tx);
        store
            .submit(make_media_job(
                "job-media-preflight",
                &missing_path.display().to_string(),
            ))
            .await
            .expect("submit job");

        let file_list = vec![PendingJobFile {
            file_index: 0,
            filename: FileName::from("missing.wav"),
            has_chat: false,
        }];
        let failures = HashMap::from([(0usize, String::from("Media file not found"))]);

        let failed_indices = record_preflight_media_failures(
            &store,
            &JobId::from("job-media-preflight"),
            &file_list,
            &failures,
        )
        .await;

        assert_eq!(failed_indices, HashSet::from([0usize]));

        let attempts = db
            .load_attempts_for_job("job-media-preflight")
            .await
            .expect("load attempts");
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].work_unit_kind, WorkUnitKind::FileSetup);
        assert_eq!(attempts[0].outcome, AttemptOutcome::Failed);
        assert_eq!(
            attempts[0].failure_category,
            Some(FailureCategory::Validation)
        );

        let detail = store
            .get_job_detail(&JobId::from("job-media-preflight"))
            .await
            .expect("job detail");
        assert_eq!(detail.file_statuses.len(), 1);
        assert_eq!(
            detail.file_statuses[0].status,
            crate::api::FileStatusKind::Error
        );
    }
