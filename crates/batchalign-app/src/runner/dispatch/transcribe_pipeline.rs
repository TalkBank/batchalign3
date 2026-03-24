//! Transcription dispatch and per-file transcribe pipeline.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::api::{ContentType, EngineVersion, NumWorkers, RevAiJobId, UnixTimestamp};
use crate::cache::UtteranceCache;
use crate::pipeline::PipelineServices;
use crate::scheduling::{FailureCategory, RetryPolicy, WorkUnitKind};
use crate::worker::pool::WorkerPool;
use tracing::warn;

use crate::store::{JobStore, RunnerJobSnapshot, unix_now};
use crate::transcribe::TranscribeOptions;

use super::super::util::{
    FileRunTracker, FileStage, FileTaskOutcome, apply_result_filename, classify_server_error,
    drain_supervised_file_tasks, is_retryable_worker_failure, spawn_progress_forwarder,
    spawn_supervised_file_task, user_facing_error,
};
use super::TranscribeDispatchPlan;
use super::infer_batched::apply_merge_abbrev;

/// Shared runtime dependencies for top-level transcribe dispatch.
///
/// The runner always passes this bundle together after it chooses the
/// transcribe command family, so keeping it typed here makes the dispatch seam
/// explicit and keeps the function signature narrow.
pub(in crate::runner) struct TranscribeDispatchRuntime {
    /// Worker pool used for ASR and optional speaker diarization requests.
    pub pool: Arc<WorkerPool>,
    /// Shared utterance cache used by post-ASR server-side stages.
    pub cache: Arc<UtteranceCache>,
    /// Current engine version string for cache partitioning.
    pub engine_version: EngineVersion,
    /// Optional preflight Rev.AI job ids keyed by original audio path.
    pub rev_job_ids: Arc<HashMap<PathBuf, RevAiJobId>>,
    /// Maximum number of file tasks to run concurrently for this job.
    pub num_workers: NumWorkers,
}

/// Dispatch transcribe via the server-side infer path.
///
/// Like FA, transcribe is per-file (each file has its own audio). Files are
/// processed concurrently up to `num_workers` at a time, bounded by a semaphore.
pub(in crate::runner) async fn dispatch_transcribe_infer(
    job: &RunnerJobSnapshot,
    store: &Arc<JobStore>,
    runtime: TranscribeDispatchRuntime,
    plan: TranscribeDispatchPlan,
) {
    let TranscribeDispatchPlan {
        base_options,
        should_merge_abbrev,
    } = plan;
    let job_id = &job.identity.job_id;

    // Process files concurrently, bounded by available workers.
    let file_sem = Arc::new(tokio::sync::Semaphore::new(runtime.num_workers.0.max(1)));
    let mut tasks = Vec::new();

    for file in &job.pending_files {
        // Check cancellation before spawning
        if job.cancel_token.is_cancelled() {
            break;
        }

        let Ok(permit) = file_sem.clone().acquire_owned().await else { tracing::warn!("file semaphore closed during shutdown"); break; };
        let store = store.clone();
        let pool = runtime.pool.clone();
        let cache = runtime.cache.clone();
        let job = job.clone();
        let engine_version = runtime.engine_version.clone();
        let mut opts = base_options.clone();
        let file = file.clone();
        let rev_job_ids = runtime.rev_job_ids.clone();
        let filename = file.filename.clone();

        tasks.push(spawn_supervised_file_task(
            filename,
            "transcribe file task",
            async move {
                let _permit = permit;
                let services = PipelineServices::new(&pool, &cache, &engine_version);
                process_one_transcribe_file(
                    &job,
                    &store,
                    services,
                    &file,
                    &mut opts,
                    rev_job_ids.as_ref(),
                    should_merge_abbrev,
                )
                .await
            },
        ));
    }

    let abnormal_exits = drain_supervised_file_tasks(store, job_id, &job.cancel_token, tasks).await;
    if abnormal_exits > 0 {
        warn!(
            job_id = %job_id,
            abnormal_exits,
            "Supervised transcribe file tasks exited abnormally"
        );
    }
}

fn transcribe_media_name_for_chat(
    original_audio_path: &Path,
    converted_audio_path: &Path,
) -> Option<String> {
    original_audio_path
        .file_stem()
        .or_else(|| original_audio_path.file_name())
        .or_else(|| converted_audio_path.file_stem())
        .or_else(|| converted_audio_path.file_name())
        .map(|s| s.to_string_lossy().into_owned())
}

/// Process a single audio file through the transcribe pipeline.
async fn process_one_transcribe_file(
    job: &RunnerJobSnapshot,
    store: &Arc<JobStore>,
    services: PipelineServices<'_>,
    file: &crate::store::PendingJobFile,
    opts: &mut TranscribeOptions,
    rev_job_ids: &HashMap<PathBuf, RevAiJobId>,
    should_merge_abbrev: bool,
) -> FileTaskOutcome {
    let job_id = &job.identity.job_id;
    let correlation_id = &*job.identity.correlation_id;
    let file_index = file.file_index;
    let filename = file.filename.as_ref();
    let lifecycle = FileRunTracker::new(store, job_id, filename);
    let started_at = unix_now();

    lifecycle
        .begin_first_attempt(
            WorkUnitKind::FileInfer,
            started_at,
            FileStage::ResolvingAudio,
        )
        .await;

    // Resolve audio path
    let original_audio_path: PathBuf =
        if job.filesystem.paths_mode && file_index < job.filesystem.source_paths.len() {
            job.filesystem.source_paths[file_index].clone()
        } else {
            job.filesystem.staging_dir.join("input").join(filename)
        };
    let audio_path = original_audio_path.clone();
    let paths_mode = job.filesystem.paths_mode;
    let output_paths = job.filesystem.output_paths.clone();
    let staging_dir = job.filesystem.staging_dir.clone();

    opts.rev_job_id = rev_job_ids.get(&original_audio_path).cloned();

    // Convert non-WAV media (e.g. mp4) to WAV via ffmpeg if needed.
    let audio_path = match crate::ensure_wav::ensure_wav(&audio_path, None).await {
        Ok(p) => p,
        Err(e) => {
            let err_msg = format!("Media conversion failed for {filename}: {e}");
            lifecycle
                .fail(&err_msg, FailureCategory::Validation, unix_now())
                .await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };

    // Preserve the source media basename for CHAT even when ffmpeg conversion
    // routes inference through a cached WAV artifact.
    opts.media_name = transcribe_media_name_for_chat(&original_audio_path, &audio_path);

    let audio_path_str = audio_path.to_string_lossy();
    tracing::info!(
        job_id = %job_id,
        correlation_id = %correlation_id,
        filename = %filename,
        audio_path = %audio_path_str,
        "Starting transcribe for file"
    );

    let retry_policy = RetryPolicy::default();

    for attempt_number in 1..=retry_policy.max_attempts {
        if attempt_number > 1 {
            lifecycle
                .restart_attempt(WorkUnitKind::FileInfer, unix_now(), FileStage::Transcribing)
                .await;
        } else {
            lifecycle.stage(FileStage::Transcribing).await;
        }

        // Create a progress forwarder for the transcribe pipeline stages.
        let progress_tx =
            spawn_progress_forwarder(store.clone(), job_id.clone(), filename.to_string());

        let debug_dir = job
            .dispatch
            .options
            .common()
            .debug_dir
            .as_deref()
            .map(Path::new);
        match crate::transcribe::process_transcribe(
            &audio_path,
            services,
            opts,
            Some(&progress_tx),
            debug_dir,
        )
        .await
        {
            Ok(output_text) => {
                lifecycle.stage(FileStage::Writing).await;
                let finished_at = unix_now();

                // Optionally merge abbreviations before writing
                let output_text = if should_merge_abbrev {
                    apply_merge_abbrev(&output_text)
                } else {
                    output_text
                };

                // Determine output filename (.cha extension)
                let output_filename = Path::new(filename)
                    .with_extension("cha")
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned();

                // Write output
                let write_path = if paths_mode && file_index < output_paths.len() {
                    apply_result_filename(&output_paths[file_index], &output_filename)
                } else {
                    staging_dir.join("output").join(&output_filename)
                };

                if let Some(parent) = write_path.parent() {
                    let _ = tokio::fs::create_dir_all(parent).await;
                }

                if let Err(e) = tokio::fs::write(&write_path, &output_text).await {
                    warn!(
                        job_id = %job_id,
                        correlation_id = %correlation_id,
                        filename = %filename,
                        error = %e,
                        "Failed to write transcribe output"
                    );
                }

                lifecycle
                    .complete_with_result(
                        output_filename.clone().into(),
                        ContentType::Chat,
                        finished_at,
                    )
                    .await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
            Err(e) => {
                let finished_at = unix_now();
                let category = classify_server_error(&e);
                let raw_msg = format!("Transcribe failed: {e}");
                warn!(
                    job_id = %job_id,
                    filename,
                    category = %category,
                    raw_error = %raw_msg,
                    "Transcribe error (raw)"
                );
                let err_msg = user_facing_error(category, "Transcription", filename, &raw_msg);
                let has_retry_budget = attempt_number < retry_policy.max_attempts;

                if matches!(&e, crate::error::ServerError::Worker(_))
                    && is_retryable_worker_failure(category)
                    && has_retry_budget
                {
                    let backoff_ms = retry_policy.backoff_for_retry(attempt_number);
                    let retry_at = UnixTimestamp(finished_at.0 + (backoff_ms.0 as f64 / 1000.0));
                    lifecycle
                        .retry(
                            retry_at,
                            category,
                            &format!("{err_msg}; retrying in {backoff_ms} ms"),
                            finished_at,
                        )
                        .await;
                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms.0)).await;
                    continue;
                }

                lifecycle.fail(&err_msg, category, finished_at).await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
        }
    }

    FileTaskOutcome::MissingTerminalState
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::path::Path;
    use std::sync::Arc;

    use tokio::sync::broadcast;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::options::{AsrEngineName, FaEngineName};
    use crate::api::{LanguageCode3, LanguageSpec};
    use crate::api::{
        EngineVersion, DisplayPath, FileStatusKind, JobId, JobStatus, NumSpeakers, ReleasedCommand,
        UnixTimestamp,
    };
    use crate::cache::UtteranceCache;
    use crate::db::JobDB;
    use crate::options::{CommandOptions, CommonOptions, TranscribeOptions as TranscribeCommand};
    use crate::store::{
        FileStatus, Job, JobDispatchConfig, JobExecutionState, JobFilesystemConfig, JobIdentity,
        JobLeaseState, JobRuntimeControl, JobScheduleState, JobSourceContext,
    };
    use crate::transcribe::AsrBackend;
    use crate::ws::BROADCAST_CAPACITY;

    /// Build a minimal transcribe job whose source path points at a missing
    /// media file so setup fails before any model work runs.
    fn make_transcribe_job(job_id: &str, source_path: &Path, output_path: &Path) -> Job {
        let filename = "missing.mp4";
        let mut file_statuses = HashMap::new();
        file_statuses.insert(
            filename.to_string(),
            FileStatus::new(DisplayPath::from(filename)),
        );

        Job {
            identity: JobIdentity {
                job_id: JobId::from(job_id),
                correlation_id: format!("test-{job_id}").into(),
            },
            dispatch: JobDispatchConfig {
                command: ReleasedCommand::Transcribe,
                lang: LanguageSpec::Resolved(LanguageCode3::eng()),
                num_speakers: NumSpeakers(1),
                options: CommandOptions::Transcribe(TranscribeCommand {
                    common: CommonOptions::default(),
                    asr_engine: AsrEngineName::RevAi,
                    diarize: false,
                    wor: false.into(),
                    merge_abbrev: false.into(),
                    batch_size: 8,
                }),
                runtime_state: BTreeMap::new(),
                debug_traces: false,
            },
            source: JobSourceContext {
                submitted_by: "127.0.0.1".into(),
                submitted_by_name: "localhost".into(),
                source_dir: std::path::PathBuf::new(),
            },
            filesystem: JobFilesystemConfig {
                filenames: vec![DisplayPath::from(filename)],
                has_chat: vec![false],
                staging_dir: std::path::PathBuf::new(),
                paths_mode: true,
                source_paths: vec![source_path.to_path_buf()],
                output_paths: vec![output_path.to_path_buf()],
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

    /// Media-conversion failure should still record a failed attempt because
    /// the first attempt now begins before audio normalization.
    #[tokio::test]
    async fn missing_media_records_failed_attempt() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let source_path = tempdir.path().join("missing.mp4");
        let output_path = tempdir.path().join("out");
        let cache_dir = tempdir.path().join("cache");
        let db = Arc::new(JobDB::open(Some(tempdir.path())).await.expect("open db"));
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = Arc::new(JobStore::new(
            crate::config::ServerConfig::default(),
            Some(db.clone()),
            tx,
        ));
        store
            .submit(make_transcribe_job(
                "job-transcribe",
                &source_path,
                &output_path,
            ))
            .await
            .expect("submit job");

        let snapshot = store
            .runner_snapshot(&JobId::from("job-transcribe"))
            .await
            .expect("runner snapshot");
        let file = snapshot
            .pending_files
            .first()
            .cloned()
            .expect("pending file");
        let pool = WorkerPool::new(crate::worker::pool::PoolConfig::default());
        let cache = UtteranceCache::sqlite(Some(cache_dir))
            .await
            .expect("open cache");
        let engine_version = EngineVersion::from("test-asr");
        let services = PipelineServices::new(&pool, &cache, &engine_version);
        let mut opts = crate::transcribe::TranscribeOptions {
            backend: AsrBackend::Worker(crate::transcribe::AsrWorkerMode::LocalWhisperV2),
            diarize: false,
            speaker_backend: None,
            lang: LanguageSpec::Resolved(LanguageCode3::eng()),
            num_speakers: 1,
            with_utseg: true,
            with_morphosyntax: false,
            override_cache: false,
            write_wor: false,
            media_name: None,
            rev_job_id: None,
        };

        process_one_transcribe_file(
            &snapshot,
            &store,
            services,
            &file,
            &mut opts,
            &HashMap::new(),
            false,
        )
        .await;

        let attempts = db
            .load_attempts_for_job("job-transcribe")
            .await
            .expect("load attempts");
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].work_unit_kind, WorkUnitKind::FileInfer);
        assert_eq!(
            attempts[0].outcome,
            crate::scheduling::AttemptOutcome::Failed
        );
        assert_eq!(
            attempts[0].failure_category,
            Some(FailureCategory::Validation)
        );

        let detail = store
            .get_job_detail(&JobId::from("job-transcribe"))
            .await
            .expect("job detail");
        assert_eq!(detail.file_statuses.len(), 1);
        assert_eq!(detail.file_statuses[0].status, FileStatusKind::Error);
    }

    #[tokio::test]
    async fn transcribe_media_name_uses_original_basename_after_mp4_conversion() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().join("cache");
        let original_audio_path = dir.path().join("interview.mp4");
        let ffmpeg_out = tokio::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-t",
                "0.1",
                original_audio_path.to_string_lossy().as_ref(),
            ])
            .output()
            .await;
        if ffmpeg_out.is_err() || !ffmpeg_out.expect("ffmpeg output").status.success() {
            eprintln!("skipping: could not generate test mp4");
            return;
        }

        let converted_audio_path = crate::ensure_wav::ensure_wav(&original_audio_path, Some(&cache_dir))
            .await
            .expect("convert mp4 to cached wav");

        assert_ne!(
            converted_audio_path.file_stem(),
            original_audio_path.file_stem(),
            "test requires cached wav basename to differ from original media basename"
        );
        assert_eq!(
            transcribe_media_name_for_chat(&original_audio_path, &converted_audio_path).as_deref(),
            Some("interview"),
            "CHAT @Media should preserve the original media basename after conversion"
        );
    }
}
