//! File status tracking, progress reporting, and retry state management.

use std::future::Future;
use std::sync::Arc;

use crate::api::{
    ContentType, DisplayPath, FileProgressStage, JobId, ReleasedCommand, UnixTimestamp,
};
use crate::scheduling::{AttemptOutcome, FailureCategory, RetryDisposition, WorkUnitKind};
use crate::store::{
    AttemptFinishRecord, CompletedFileOutput, FileFailureRecord, FileProgressRecord,
    FileRetryRecord, JobStore, unix_now,
};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Mark a file as actively processing and persist the start timestamp.
pub(in crate::runner) async fn mark_file_processing(
    store: &JobStore,
    job_id: &JobId,
    filename: &str,
    started_at: UnixTimestamp,
) {
    store
        .mark_file_processing(job_id, filename, started_at)
        .await;
}

/// Mark a file as successfully completed and attach one result entry.
pub(in crate::runner) async fn mark_file_done(
    store: &JobStore,
    job_id: &JobId,
    filename: &str,
    result_filename: DisplayPath,
    content_type: ContentType,
    finished_at: UnixTimestamp,
) {
    store
        .mark_file_done(
            job_id,
            filename,
            finished_at,
            Some(CompletedFileOutput {
                filename: result_filename,
                content_type,
            }),
        )
        .await;
}

/// Mark a file as done without recording a downloadable result artifact.
pub(in crate::runner) async fn mark_file_done_without_result(
    store: &JobStore,
    job_id: &JobId,
    filename: &str,
    finished_at: UnixTimestamp,
) {
    store
        .mark_file_done(job_id, filename, finished_at, None)
        .await;
}

/// Set a file to error status.
pub(in crate::runner) async fn set_file_error(
    store: &JobStore,
    job_id: &JobId,
    filename: &str,
    error: &str,
    category: FailureCategory,
    finished_at: UnixTimestamp,
) {
    store
        .mark_file_error(
            job_id,
            filename,
            &FileFailureRecord {
                message: error.to_string(),
                category,
                finished_at,
            },
        )
        .await;
}

/// Increment the control-plane counter for started work-unit attempts.
pub(in crate::runner) async fn start_file_attempt(
    store: &JobStore,
    job_id: &JobId,
    filename: &str,
    work_unit_kind: WorkUnitKind,
    started_at: UnixTimestamp,
) {
    store
        .start_file_attempt(job_id, filename, work_unit_kind, started_at)
        .await;
}

/// Finalize a successful file attempt after the output has been persisted.
pub(in crate::runner) async fn finish_file_attempt_success(
    store: &JobStore,
    job_id: &JobId,
    filename: &str,
    finished_at: UnixTimestamp,
) {
    store
        .db_finish_attempt_for_file(
            job_id,
            AttemptFinishRecord {
                filename,
                outcome: AttemptOutcome::Succeeded,
                failure_category: None,
                disposition: RetryDisposition::Succeed,
                finished_at,
            },
        )
        .await;
}

/// Mark a file as waiting for a retry after a transient attempt failure.
pub(in crate::runner) async fn set_file_retry_pending(
    store: &JobStore,
    job_id: &JobId,
    filename: &str,
    retry_at: UnixTimestamp,
    category: FailureCategory,
    message: &str,
    finished_at: UnixTimestamp,
) {
    store
        .mark_file_retry_pending(
            job_id,
            filename,
            &FileRetryRecord {
                message: message.to_string(),
                category,
                finished_at,
                retry_at,
            },
        )
        .await;
}

/// Clear transient retry state before a new attempt starts or succeeds.
pub(in crate::runner) async fn clear_retry_state(store: &JobStore, job_id: &JobId, filename: &str) {
    store.clear_file_retry_state(job_id, filename).await;
}

/// Update ephemeral progress fields on a file and broadcast the update.
///
/// Progress fields are never persisted to SQLite — they are purely for
/// live display in the CLI/TUI/React dashboard.
pub(in crate::runner) async fn set_file_progress(
    store: &JobStore,
    job_id: &JobId,
    filename: &str,
    stage: FileStage,
    current: Option<i64>,
    total: Option<i64>,
) {
    store
        .set_file_progress(
            job_id,
            filename,
            &FileProgressRecord {
                stage: stage.api_stage(),
                current,
                total,
            },
        )
        .await;
}

/// Runner-side helper for one file's lifecycle and attempt bookkeeping.
///
/// Dispatch code should prefer this helper over hand-sequencing raw store
/// mutations. That keeps the per-file state machine explicit:
///
/// - begin the first processing attempt
/// - move between human-readable stages
/// - restart the attempt after retryable failures
/// - finish as success, retry, or terminal error
pub(crate) struct FileRunTracker<'a> {
    store: &'a JobStore,
    job_id: &'a JobId,
    filename: &'a str,
}

/// Explicit completion contract for one supervised file task.
///
/// A task returns `TerminalStateRecorded` only after it has already written the
/// final file status that the runner should trust. Any early return, panic, or
/// cancellation that skips that write path must surface as
/// `MissingTerminalState` so the supervisor can record a concrete failure.
pub(crate) enum FileTaskOutcome {
    /// The task itself recorded success or terminal failure for the file.
    TerminalStateRecorded,
    /// The task exited without recording a terminal file state.
    MissingTerminalState,
}

/// Canonical runner-owned stage labels for file lifecycles.
///
/// This keeps the control-plane stage vocabulary typed all the way through the
/// runner and store layers before the API derives an operator-facing label.
/// The enum intentionally covers both top-level runner stages and the
/// lower-level FA/transcribe pipeline progress labels that feed the same file
/// status channel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FileStage {
    /// Generic processing for worker-owned media commands.
    Processing,
    /// Initial file read/setup work.
    Reading,
    /// Media discovery or normalization.
    ResolvingAudio,
    /// Utterance-level timing recovery before alignment.
    RecoveringUtteranceTiming,
    /// Fallback timing recovery after an FA failure.
    RecoveringTimingFallback,
    /// Main forced-alignment stage.
    Aligning,
    /// Main transcription stage.
    Transcribing,
    /// Main benchmark stage.
    Benchmarking,
    /// Cache partition / cache lookup stage inside FA.
    CheckingCache,
    /// Apply aligned timings or inferred annotations back into the document.
    ApplyingResults,
    /// ASR output post-processing before CHAT construction.
    PostProcessing,
    /// Build a CHAT document from intermediate utterance state.
    BuildingChat,
    /// Utterance segmentation within the transcribe pipeline.
    SegmentingUtterances,
    /// Morphosyntax enrichment within the transcribe pipeline.
    AnalyzingMorphosyntax,
    /// Final pipeline serialization/finalization step.
    Finalizing,
    /// Final serialization/write stage.
    Writing,
    /// Batched morphosyntax analysis.
    Analyzing,
    /// Batched utterance segmentation.
    Segmenting,
    /// Batched translation.
    Translating,
    /// Batched coreference resolution.
    ResolvingCoreference,
    /// Batched transcript/reference comparison.
    Comparing,
}

impl FileStage {
    /// Resolve the initial batch-infer stage for a top-level command.
    pub(crate) fn for_batch_command(command: ReleasedCommand) -> Self {
        match command {
            ReleasedCommand::Morphotag => Self::Analyzing,
            ReleasedCommand::Utseg => Self::Segmenting,
            ReleasedCommand::Translate => Self::Translating,
            ReleasedCommand::Coref => Self::ResolvingCoreference,
            ReleasedCommand::Compare => Self::Comparing,
            _ => Self::Processing,
        }
    }

    /// Convert the runner-local stage vocabulary to the stable API enum.
    pub(crate) const fn api_stage(self) -> FileProgressStage {
        match self {
            Self::Processing => FileProgressStage::Processing,
            Self::Reading => FileProgressStage::Reading,
            Self::ResolvingAudio => FileProgressStage::ResolvingAudio,
            Self::RecoveringUtteranceTiming => FileProgressStage::RecoveringUtteranceTiming,
            Self::RecoveringTimingFallback => FileProgressStage::RecoveringTimingFallback,
            Self::Aligning => FileProgressStage::Aligning,
            Self::Transcribing => FileProgressStage::Transcribing,
            Self::Benchmarking => FileProgressStage::Benchmarking,
            Self::CheckingCache => FileProgressStage::CheckingCache,
            Self::ApplyingResults => FileProgressStage::ApplyingResults,
            Self::PostProcessing => FileProgressStage::PostProcessing,
            Self::BuildingChat => FileProgressStage::BuildingChat,
            Self::SegmentingUtterances => FileProgressStage::SegmentingUtterances,
            Self::AnalyzingMorphosyntax => FileProgressStage::AnalyzingMorphosyntax,
            Self::Finalizing => FileProgressStage::Finalizing,
            Self::Writing => FileProgressStage::Writing,
            Self::Analyzing => FileProgressStage::Analyzing,
            Self::Segmenting => FileProgressStage::Segmenting,
            Self::Translating => FileProgressStage::Translating,
            Self::ResolvingCoreference => FileProgressStage::ResolvingCoreference,
            Self::Comparing => FileProgressStage::Comparing,
        }
    }
}

impl<'a> FileRunTracker<'a> {
    /// Bind the helper to one `(job_id, filename)` pair.
    pub(crate) fn new(store: &'a JobStore, job_id: &'a JobId, filename: &'a str) -> Self {
        Self {
            store,
            job_id,
            filename,
        }
    }

    /// Mark the file as processing, open the first durable attempt, and set the
    /// initial stage label shown to operators.
    pub(crate) async fn begin_first_attempt(
        &self,
        work_unit_kind: WorkUnitKind,
        started_at: UnixTimestamp,
        stage: FileStage,
    ) {
        mark_file_processing(self.store, self.job_id, self.filename, started_at).await;
        clear_retry_state(self.store, self.job_id, self.filename).await;
        start_file_attempt(
            self.store,
            self.job_id,
            self.filename,
            work_unit_kind,
            started_at,
        )
        .await;
        self.stage(stage).await;
    }

    /// Open a durable setup attempt that fails before the file ever enters the
    /// normal processing pipeline.
    ///
    /// This is used for preflight rejection paths such as missing or
    /// incompatible media where we still want attempt history but should not
    /// advertise the file as actively processing.
    pub(crate) async fn record_setup_failure(
        &self,
        started_at: UnixTimestamp,
        error: &str,
        category: FailureCategory,
        finished_at: UnixTimestamp,
    ) {
        clear_retry_state(self.store, self.job_id, self.filename).await;
        start_file_attempt(
            self.store,
            self.job_id,
            self.filename,
            WorkUnitKind::FileSetup,
            started_at,
        )
        .await;
        self.fail(error, category, finished_at).await;
    }

    /// Clear retry-only state, open the next attempt, and publish the stage
    /// label for the new run.
    pub(crate) async fn restart_attempt(
        &self,
        work_unit_kind: WorkUnitKind,
        started_at: UnixTimestamp,
        stage: FileStage,
    ) {
        clear_retry_state(self.store, self.job_id, self.filename).await;
        start_file_attempt(
            self.store,
            self.job_id,
            self.filename,
            work_unit_kind,
            started_at,
        )
        .await;
        self.stage(stage).await;
    }

    /// Update the current human-readable progress stage.
    pub(crate) async fn stage(&self, stage: FileStage) {
        set_file_progress(self.store, self.job_id, self.filename, stage, None, None).await;
    }

    /// Record a retryable failure and publish the retry deadline.
    pub(crate) async fn retry(
        &self,
        retry_at: UnixTimestamp,
        category: FailureCategory,
        message: &str,
        finished_at: UnixTimestamp,
    ) {
        set_file_retry_pending(
            self.store,
            self.job_id,
            self.filename,
            retry_at,
            category,
            message,
            finished_at,
        )
        .await;
    }

    /// Record a terminal file failure.
    pub(crate) async fn fail(
        &self,
        error: &str,
        category: FailureCategory,
        finished_at: UnixTimestamp,
    ) {
        set_file_error(
            self.store,
            self.job_id,
            self.filename,
            error,
            category,
            finished_at,
        )
        .await;
    }

    /// Mark the file as done with a downloadable result and close the active
    /// attempt as successful.
    pub(crate) async fn complete_with_result(
        &self,
        result_filename: DisplayPath,
        content_type: ContentType,
        finished_at: UnixTimestamp,
    ) {
        mark_file_done(
            self.store,
            self.job_id,
            self.filename,
            result_filename,
            content_type,
            finished_at,
        )
        .await;
        finish_file_attempt_success(self.store, self.job_id, self.filename, finished_at).await;
    }

    /// Mark the file as done without a downloadable artifact and close the
    /// active attempt as successful.
    pub(crate) async fn complete_without_result(&self, finished_at: UnixTimestamp) {
        mark_file_done_without_result(self.store, self.job_id, self.filename, finished_at).await;
        finish_file_attempt_success(self.store, self.job_id, self.filename, finished_at).await;
    }
}

/// A progress update from an orchestrator to the dispatch layer.
pub(crate) struct ProgressUpdate {
    /// Typed lifecycle/progress label.
    pub label: FileStage,
    /// Current progress counter (optional).
    pub current: Option<i64>,
    /// Total items for progress (optional).
    pub total: Option<i64>,
}

impl ProgressUpdate {
    /// Construct a typed progress update for the shared file-status channel.
    pub(crate) fn new(label: FileStage, current: Option<i64>, total: Option<i64>) -> Self {
        Self {
            label,
            current,
            total,
        }
    }
}

/// Sender half for progress updates. Orchestrators hold this.
pub(crate) type ProgressSender = tokio::sync::mpsc::UnboundedSender<ProgressUpdate>;

/// Handle to one spawned file task whose terminal file-state transition is
/// supervised by the runner rather than inferred later from a job-wide sweep.
pub(in crate::runner) struct SpawnedFileTask {
    /// Human-readable task role for diagnostics (`"align file"`, etc.).
    pub role: &'static str,
    /// Logical filename owned by this task.
    pub filename: DisplayPath,
    /// Join handle for the spawned task.
    pub handle: JoinHandle<FileTaskOutcome>,
}

/// Spawn one supervised file task.
///
/// The inner future still owns the real command logic. The supervision layer is
/// responsible only for one invariant: once the task stops running, the runner
/// must know whether the corresponding file already reached a terminal state.
pub(in crate::runner) fn spawn_supervised_file_task<F>(
    filename: DisplayPath,
    role: &'static str,
    future: F,
) -> SpawnedFileTask
where
    F: Future<Output = FileTaskOutcome> + Send + 'static,
{
    let handle = tokio::spawn(future);

    SpawnedFileTask {
        role,
        filename,
        handle,
    }
}

/// Drain a batch of supervised file tasks and convert abnormal exits into
/// explicit file failures immediately.
///
/// This keeps panics and early returns from being discovered only by the
/// runner's coarse "force unfinished files to terminal state" fallback.
pub(in crate::runner) async fn drain_supervised_file_tasks(
    store: &JobStore,
    job_id: &JobId,
    cancel_token: &CancellationToken,
    tasks: Vec<SpawnedFileTask>,
) -> usize {
    let mut abnormal_exits = 0usize;

    for task in tasks {
        match task.handle.await {
            Ok(FileTaskOutcome::TerminalStateRecorded) => {}
            Ok(FileTaskOutcome::MissingTerminalState) => {
                abnormal_exits += 1;
                record_abnormal_file_task_exit(
                    store,
                    job_id,
                    task.filename.as_ref(),
                    task.role,
                    cancel_token.is_cancelled(),
                    None,
                )
                .await;
            }
            Err(join_error) => {
                abnormal_exits += 1;
                record_abnormal_file_task_exit(
                    store,
                    job_id,
                    task.filename.as_ref(),
                    task.role,
                    cancel_token.is_cancelled(),
                    Some(join_error.to_string()),
                )
                .await;
            }
        }
    }

    abnormal_exits
}

/// Create a progress channel and spawn a forwarder task that routes updates
/// to the store for a specific `(job_id, filename)`.
///
/// Returns the sender half. The forwarder runs until the sender is dropped.
pub(in crate::runner) fn spawn_progress_forwarder(
    store: Arc<JobStore>,
    job_id: JobId,
    filename: String,
) -> ProgressSender {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ProgressUpdate>();
    tokio::spawn(async move {
        while let Some(update) = rx.recv().await {
            set_file_progress(
                &store,
                &job_id,
                &filename,
                update.label,
                update.current,
                update.total,
            )
            .await;
        }
    });
    tx
}

/// Record a non-standard file-task exit as an explicit terminal failure.
///
/// This path is only for supervision failures: task panic, task cancellation,
/// or a task returning without ever marking its file done/error.
async fn record_abnormal_file_task_exit(
    store: &JobStore,
    job_id: &JobId,
    filename: &str,
    role: &str,
    job_cancelled: bool,
    join_error: Option<String>,
) {
    let finished_at = unix_now();
    let (message, category, outcome) = if job_cancelled {
        (
            format!("{role} stopped after job cancellation before recording a terminal file state"),
            FailureCategory::Cancelled,
            AttemptOutcome::Cancelled,
        )
    } else if let Some(join_error) = join_error {
        (
            format!("{role} panicked before recording a terminal file state: {join_error}"),
            FailureCategory::System,
            AttemptOutcome::Failed,
        )
    } else {
        (
            format!("{role} exited without recording a terminal file state"),
            FailureCategory::System,
            AttemptOutcome::Failed,
        )
    };

    store
        .db_finish_attempt_for_file(
            job_id,
            AttemptFinishRecord {
                filename,
                outcome,
                failure_category: Some(category),
                disposition: RetryDisposition::TerminalFailure,
                finished_at,
            },
        )
        .await;

    set_file_error(store, job_id, filename, &message, category, finished_at).await;
}

/// Fallback cleanup for any files that still failed to reach a terminal state.
///
/// The supervised file-task boundary should normally make this path a no-op.
/// It remains as a last-resort guard against leaked tasks or other control-
/// plane bugs.
pub(in crate::runner) async fn force_terminal_file_states(
    store: &JobStore,
    job_id: &JobId,
) -> usize {
    let unfinished: Vec<DisplayPath> = store.unfinished_files(job_id).await;

    if unfinished.is_empty() {
        return 0;
    }

    let now = unix_now();
    for filename in &unfinished {
        let last_status = store
            .file_status_label(job_id, filename)
            .await
            .unwrap_or_default();
        let msg = format!("File did not reach terminal status (last status: {last_status})");
        set_file_error(store, job_id, filename, &msg, FailureCategory::System, now).await;
    }

    store
        .bump_counter(|c| c.forced_terminal_errors += unfinished.len() as i64)
        .await;
    unfinished.len()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use tokio::sync::broadcast;
    use tokio_util::sync::CancellationToken;

    use crate::api::{FileProgressStage, FileStatusKind, JobId, JobStatus, NumSpeakers};
    use crate::db::JobDB;
    use crate::options::{CommandOptions, CommonOptions, MorphotagOptions};
    use crate::scheduling::{AttemptOutcome, RetryDisposition, WorkUnitKind};
    use crate::store::unix_now;
    use crate::store::{
        FileStatus, Job, JobDispatchConfig, JobExecutionState, JobFilesystemConfig, JobIdentity,
        JobLeaseState, JobRuntimeControl, JobScheduleState, JobSourceContext,
    };
    use crate::ws::BROADCAST_CAPACITY;

    use super::*;
    use crate::api::{LanguageCode3, LanguageSpec};

    fn test_config() -> crate::config::ServerConfig {
        crate::config::ServerConfig {
            max_concurrent_jobs: 2,
            ..Default::default()
        }
    }

    fn make_job(id: &str) -> Job {
        let mut file_statuses = HashMap::new();
        file_statuses.insert(
            "a.cha".to_string(),
            FileStatus::new(DisplayPath::from("a.cha")),
        );

        Job {
            identity: JobIdentity {
                job_id: id.into(),
                correlation_id: format!("test-{id}").into(),
            },
            dispatch: JobDispatchConfig {
                command: ReleasedCommand::Morphotag,
                lang: LanguageSpec::Resolved(LanguageCode3::eng()),
                num_speakers: NumSpeakers(1),
                options: CommandOptions::Morphotag(MorphotagOptions {
                    common: CommonOptions::default(),
                    retokenize: false,
                    skipmultilang: false,
                    merge_abbrev: false.into(),
                }),
                runtime_state: std::collections::BTreeMap::new(),
                debug_traces: false,
            },
            source: JobSourceContext {
                submitted_by: "127.0.0.1".into(),
                submitted_by_name: String::new(),
                source_dir: std::path::PathBuf::new(),
            },
            filesystem: JobFilesystemConfig {
                filenames: vec![DisplayPath::from("a.cha")],
                has_chat: vec![true],
                staging_dir: std::path::PathBuf::new(),
                paths_mode: false,
                source_paths: Vec::new(),
                output_paths: Vec::new(),
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
                submitted_at: unix_now(),
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

    #[tokio::test]
    async fn supervised_task_marks_non_terminal_exit_as_error() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = Arc::new(JobStore::new(test_config(), None, tx));
        let job_id = JobId::from("job-1");
        store.submit(make_job("job-1")).await.unwrap();

        mark_file_processing(&store, &job_id, "a.cha", unix_now()).await;

        let tasks = vec![spawn_supervised_file_task(
            DisplayPath::from("a.cha"),
            "test file task",
            async { FileTaskOutcome::MissingTerminalState },
        )];

        let abnormal =
            drain_supervised_file_tasks(&store, &job_id, &CancellationToken::new(), tasks).await;
        assert_eq!(abnormal, 1);

        let detail = store.get_job_detail(&job_id).await.unwrap();
        let file = detail
            .file_statuses
            .into_iter()
            .find(|entry| entry.filename == "a.cha")
            .unwrap();
        assert_eq!(file.status, FileStatusKind::Error);
        assert!(
            file.error
                .as_deref()
                .is_some_and(|msg| msg.contains("exited without recording a terminal file state"))
        );
    }

    #[tokio::test]
    async fn supervised_task_marks_panic_as_error() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = Arc::new(JobStore::new(test_config(), None, tx));
        let job_id = JobId::from("job-2");
        store.submit(make_job("job-2")).await.unwrap();

        mark_file_processing(&store, &job_id, "a.cha", unix_now()).await;

        let tasks = vec![spawn_supervised_file_task(
            DisplayPath::from("a.cha"),
            "panic file task",
            async {
                panic!("boom");
            },
        )];

        let abnormal =
            drain_supervised_file_tasks(&store, &job_id, &CancellationToken::new(), tasks).await;
        assert_eq!(abnormal, 1);

        let detail = store.get_job_detail(&job_id).await.unwrap();
        let file = detail
            .file_statuses
            .into_iter()
            .find(|entry| entry.filename == "a.cha")
            .unwrap();
        assert_eq!(file.status, FileStatusKind::Error);
        assert!(
            file.error
                .as_deref()
                .is_some_and(|msg| msg.contains("panicked before recording a terminal file state"))
        );
    }

    #[tokio::test]
    async fn file_run_tracker_retries_then_completes_cleanly() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let db = Arc::new(JobDB::open(Some(tempdir.path())).await.expect("open db"));
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = Arc::new(JobStore::new(test_config(), Some(db.clone()), tx));
        let job_id = JobId::from("job-tracker");
        store.submit(make_job("job-tracker")).await.unwrap();

        let lifecycle = FileRunTracker::new(&store, &job_id, "a.cha");
        let started_at = unix_now();
        lifecycle
            .begin_first_attempt(WorkUnitKind::FileProcess, started_at, FileStage::Reading)
            .await;

        let retry_finished_at = unix_now();
        let retry_at = crate::store::unix_now();
        lifecycle
            .retry(
                retry_at,
                FailureCategory::ProviderTransient,
                "temporary failure",
                retry_finished_at,
            )
            .await;

        let restarted_at = unix_now();
        lifecycle
            .restart_attempt(
                WorkUnitKind::FileProcess,
                restarted_at,
                FileStage::Processing,
            )
            .await;

        let finished_at = unix_now();
        lifecycle
            .complete_with_result(DisplayPath::from("a.ana"), ContentType::Chat, finished_at)
            .await;

        let detail = store.get_job_detail(&job_id).await.expect("job detail");
        let file = detail
            .file_statuses
            .into_iter()
            .find(|entry| entry.filename == "a.cha")
            .expect("tracked file");
        assert_eq!(file.status, FileStatusKind::Done);
        assert!(file.next_eligible_at.is_none());
        assert!(file.error.is_none());

        let attempts = db
            .load_attempts_for_job("job-tracker")
            .await
            .expect("load attempts");
        assert_eq!(attempts.len(), 2);
        assert_eq!(attempts[0].outcome, AttemptOutcome::RetryableFailure);
        assert_eq!(attempts[0].disposition, RetryDisposition::Retry);
        assert_eq!(attempts[1].outcome, AttemptOutcome::Succeeded);
        assert_eq!(attempts[1].disposition, RetryDisposition::Succeed);
    }

    #[tokio::test]
    async fn file_run_tracker_records_setup_failure() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let db = Arc::new(JobDB::open(Some(tempdir.path())).await.expect("open db"));
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = Arc::new(JobStore::new(test_config(), Some(db.clone()), tx));
        let job_id = JobId::from("job-setup-failure");
        store.submit(make_job("job-setup-failure")).await.unwrap();

        let lifecycle = FileRunTracker::new(&store, &job_id, "a.cha");
        let started_at = unix_now();
        let finished_at = unix_now();
        lifecycle
            .record_setup_failure(
                started_at,
                "media preflight failed",
                FailureCategory::Validation,
                finished_at,
            )
            .await;

        let detail = store.get_job_detail(&job_id).await.expect("job detail");
        let file = detail
            .file_statuses
            .into_iter()
            .find(|entry| entry.filename == "a.cha")
            .expect("tracked file");
        assert_eq!(file.status, FileStatusKind::Error);
        assert_eq!(file.error.as_deref(), Some("media preflight failed"));

        let attempts = db
            .load_attempts_for_job("job-setup-failure")
            .await
            .expect("load attempts");
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].work_unit_kind, WorkUnitKind::FileSetup);
        assert_eq!(attempts[0].outcome, AttemptOutcome::Failed);
        assert_eq!(
            attempts[0].failure_category,
            Some(FailureCategory::Validation)
        );
    }

    #[test]
    fn file_stage_for_batch_command_is_stable() {
        assert_eq!(
            FileStage::for_batch_command(ReleasedCommand::Morphotag),
            FileStage::Analyzing
        );
        assert_eq!(
            FileStage::for_batch_command(ReleasedCommand::Utseg),
            FileStage::Segmenting
        );
        assert_eq!(
            FileStage::for_batch_command(ReleasedCommand::Translate),
            FileStage::Translating
        );
        assert_eq!(
            FileStage::for_batch_command(ReleasedCommand::Coref),
            FileStage::ResolvingCoreference
        );
        assert_eq!(
            FileStage::for_batch_command(ReleasedCommand::Compare),
            FileStage::Comparing
        );
        assert_eq!(
            FileStage::for_batch_command(ReleasedCommand::Align),
            FileStage::Processing
        );
        assert_eq!(FileStage::Writing.api_stage().label(), "Writing");
        assert_eq!(
            FileStage::CheckingCache.api_stage().label(),
            "Checking cache"
        );
        assert_eq!(
            FileStage::PostProcessing.api_stage().label(),
            "Post-processing"
        );
        assert_eq!(FileStage::Aligning.api_stage(), FileProgressStage::Aligning);
        assert_eq!(
            FileStage::AnalyzingMorphosyntax.api_stage(),
            FileProgressStage::AnalyzingMorphosyntax
        );
        assert_eq!(
            FileStage::AnalyzingMorphosyntax.api_stage().label(),
            "Analyzing morphosyntax"
        );
    }
}
