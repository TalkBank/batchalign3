//! In-memory job model and lifecycle methods.
//!
//! Split into submodules:
//! - [`types`] — all struct definitions (Job, JobIdentity, etc.)
//! - This file — `impl Job` methods, conflict detection

mod types;

pub use types::*;

use std::collections::HashMap;

use tokio_util::sync::CancellationToken;

use crate::api::{
    ContentType, DisplayPath, DurationSeconds, FileProgressStage, FileStatusEntry, FileStatusKind,
    JobId, JobInfo, JobListItem, JobStatus, NodeId, ReleasedCommand, UnixTimestamp,
};
use crate::scheduling::LeaseRecord;
use crate::store::{FileResultEntry, ts_iso};

impl Job {
    /// Return the stable job identifier.
    pub fn job_id(&self) -> &JobId {
        &self.identity.job_id
    }

    /// Return the total number of logical files in the job.
    pub fn total_files(&self) -> usize {
        self.filesystem.filenames.len()
    }

    /// Clear the current queue lease metadata.
    pub(crate) fn clear_lease(&mut self) {
        self.schedule.lease.leased_by_node = None;
        self.schedule.lease.expires_at = None;
        self.schedule.lease.heartbeat_at = None;
    }

    /// Return whether a live queue lease currently blocks local dispatch.
    pub(crate) fn lease_blocks_local_dispatch(&self, now: UnixTimestamp) -> bool {
        self.schedule.lease.leased_by_node.is_some()
            && self
                .schedule
                .lease
                .expires_at
                .is_some_and(|timestamp| timestamp.0 > now.0)
    }

    /// Return the earliest time when the job should be reconsidered for dispatch.
    pub(crate) fn next_local_dispatch_wake_at(&self, now: UnixTimestamp) -> Option<UnixTimestamp> {
        let mut wake_at = self
            .schedule
            .next_eligible_at
            .filter(|timestamp| timestamp.0 > now.0);
        if self.lease_blocks_local_dispatch(now) {
            wake_at = match (wake_at, self.schedule.lease.expires_at) {
                (Some(next_eligible_at), Some(lease_expires_at)) => {
                    if next_eligible_at.0 < lease_expires_at.0 {
                        Some(next_eligible_at)
                    } else {
                        Some(lease_expires_at)
                    }
                }
                (None, Some(lease_expires_at)) => Some(lease_expires_at),
                (some, None) => some,
            };
        }
        wake_at
    }

    /// Return whether the job can be claimed by the local queue dispatcher now.
    pub(crate) fn ready_for_local_dispatch(&self, now: UnixTimestamp) -> bool {
        self.execution.status == JobStatus::Queued
            && !self.runtime.runner_active
            && !self.lease_blocks_local_dispatch(now)
            && self
                .schedule
                .next_eligible_at
                .is_none_or(|timestamp| timestamp <= now)
    }

    /// Claim the job for local dispatch and return the resulting lease record.
    pub(crate) fn claim_for_local_dispatch(
        &mut self,
        node_id: &NodeId,
        now: UnixTimestamp,
        lease_ttl_s: f64,
    ) -> Option<LeaseRecord> {
        if !self.ready_for_local_dispatch(now) {
            return None;
        }

        self.runtime.runner_active = true;
        self.schedule.lease.leased_by_node = Some(node_id.clone());
        self.schedule.lease.heartbeat_at = Some(now);
        self.schedule.lease.expires_at = Some(UnixTimestamp(now.0 + lease_ttl_s));
        self.active_lease()
    }

    /// Release any local dispatch claim and clear the job's live lease.
    pub(crate) fn release_local_dispatch_claim(&mut self) {
        self.runtime.runner_active = false;
        self.clear_lease();
    }

    /// Renew the local dispatch lease when the current node still owns it.
    pub(crate) fn renew_local_dispatch_lease(
        &mut self,
        node_id: &NodeId,
        now: UnixTimestamp,
        lease_ttl_s: f64,
    ) -> Option<LeaseRecord> {
        if self.runtime.runner_active
            && self.schedule.lease.leased_by_node.as_deref() == Some(node_id)
            && !self.execution.status.is_terminal()
        {
            self.schedule.lease.heartbeat_at = Some(now);
            self.schedule.lease.expires_at = Some(UnixTimestamp(now.0 + lease_ttl_s));
            self.active_lease()
        } else {
            None
        }
    }

    /// Mark one file as actively processing.
    ///
    /// Entering processing clears any stale retry/error metadata because a new
    /// attempt should present as "currently running", not "running but still
    /// errored from the last attempt".
    pub(crate) fn mark_file_processing(
        &mut self,
        filename: &str,
        started_at: UnixTimestamp,
    ) -> bool {
        let Some(file_status) = self.execution.file_statuses.get_mut(filename) else {
            return false;
        };
        file_status.status = FileStatusKind::Processing;
        file_status.error = None;
        file_status.error_category = None;
        file_status.started_at = Some(started_at);
        file_status.finished_at = None;
        file_status.next_eligible_at = None;
        file_status.progress_current = None;
        file_status.progress_total = None;
        file_status.progress_stage = None;
        true
    }

    /// Mark one file as complete and optionally attach a result record.
    ///
    /// This also clears any retry-era error metadata so the completed file
    /// snapshot matches the persisted database row and the operator-facing API.
    pub(crate) fn mark_file_done(
        &mut self,
        filename: &str,
        finished_at: UnixTimestamp,
        result: Option<CompletedFileOutput>,
    ) -> bool {
        let Some(file_status) = self.execution.file_statuses.get_mut(filename) else {
            return false;
        };
        file_status.status = FileStatusKind::Done;
        file_status.error = None;
        file_status.error_category = None;
        file_status.finished_at = Some(finished_at);
        file_status.next_eligible_at = None;
        file_status.progress_current = None;
        file_status.progress_total = None;
        file_status.progress_stage = None;
        if let Some(result) = result {
            self.execution.results.push(FileResultEntry {
                filename: result.filename,
                content_type: result.content_type,
                error: None,
            });
        }
        self.execution.completed_files += 1;
        true
    }

    /// Mark one file as terminally failed and attach an error result.
    pub(crate) fn mark_file_error(&mut self, filename: &str, failure: &FileFailureRecord) -> bool {
        let Some(file_status) = self.execution.file_statuses.get_mut(filename) else {
            return false;
        };
        file_status.status = FileStatusKind::Error;
        file_status.error = Some(failure.message.clone());
        file_status.error_category = Some(failure.category);
        file_status.finished_at = Some(failure.finished_at);
        file_status.next_eligible_at = None;
        file_status.progress_current = None;
        file_status.progress_total = None;
        file_status.progress_stage = None;
        self.execution.results.push(FileResultEntry {
            filename: DisplayPath::from(filename),
            content_type: ContentType::Chat,
            error: Some(failure.message.clone()),
        });
        self.execution.completed_files += 1;
        true
    }

    /// Record the start of a new file attempt.
    pub(crate) fn start_file_attempt(&mut self, filename: &str, started_at: UnixTimestamp) -> bool {
        let Some(file_status) = self.execution.file_statuses.get_mut(filename) else {
            return false;
        };
        file_status.started_at = Some(started_at);
        file_status.finished_at = None;
        file_status.next_eligible_at = None;
        file_status.progress_stage = None;
        true
    }

    /// Mark one file as waiting for a retry after a transient failure.
    pub(crate) fn mark_file_retry_pending(
        &mut self,
        filename: &str,
        retry: &FileRetryRecord,
    ) -> bool {
        let Some(file_status) = self.execution.file_statuses.get_mut(filename) else {
            return false;
        };
        file_status.status = FileStatusKind::Processing;
        file_status.error = Some(retry.message.clone());
        file_status.error_category = Some(retry.category);
        file_status.finished_at = Some(retry.finished_at);
        file_status.next_eligible_at = Some(retry.retry_at);
        file_status.progress_current = None;
        file_status.progress_total = None;
        file_status.progress_stage = Some(FileProgressStage::RetryScheduled);
        true
    }

    /// Clear transient retry state before a new attempt starts or succeeds.
    ///
    /// Retry scheduling temporarily stores the last retryable error on the
    /// file so operators can see why the retry was queued. Once a new attempt
    /// starts, that stale retry error must disappear from the live file state
    /// or the dashboard/API will report a successful retry as still errored.
    pub(crate) fn clear_file_retry_state(&mut self, filename: &str) -> bool {
        let Some(file_status) = self.execution.file_statuses.get_mut(filename) else {
            return false;
        };
        file_status.error = None;
        file_status.error_category = None;
        file_status.finished_at = None;
        file_status.next_eligible_at = None;
        file_status.progress_stage = None;
        true
    }

    /// Apply an ephemeral progress update to one file.
    pub(crate) fn set_file_progress(
        &mut self,
        filename: &str,
        progress: &FileProgressRecord,
    ) -> bool {
        let Some(file_status) = self.execution.file_statuses.get_mut(filename) else {
            return false;
        };
        file_status.progress_stage = Some(progress.stage);
        file_status.progress_current = progress.current;
        file_status.progress_total = progress.total;
        true
    }

    /// Return the filenames of files that have not yet reached a terminal state.
    pub(crate) fn unfinished_files(&self) -> Vec<DisplayPath> {
        self.execution
            .file_statuses
            .values()
            .filter(|file_status| !file_status.status.is_terminal())
            .map(|file_status| file_status.filename.clone())
            .collect()
    }

    /// Return the current lifecycle label for one file.
    pub(crate) fn file_status_label(&self, filename: &str) -> Option<String> {
        self.execution
            .file_statuses
            .get(filename)
            .map(|file_status| file_status.status.to_string())
    }

    /// Return whether the job's cancellation token has been triggered.
    pub(crate) fn is_cancelled(&self) -> bool {
        self.runtime.cancel_token.is_cancelled()
    }

    /// Return whether every terminal file currently recorded is an error.
    pub(crate) fn all_terminal_files_failed(&self) -> bool {
        let terminal: Vec<FileStatusKind> = self
            .execution
            .file_statuses
            .values()
            .filter(|file_status| file_status.status.is_terminal())
            .map(|file_status| file_status.status)
            .collect();
        !terminal.is_empty()
            && terminal
                .iter()
                .all(|status| *status == FileStatusKind::Error)
    }

    /// Request cancellation and, when still active, transition to cancelled.
    pub(crate) fn request_cancellation(
        &mut self,
        completed_at: UnixTimestamp,
    ) -> Option<UnixTimestamp> {
        self.runtime.cancel_token.cancel();
        if self.execution.status.can_cancel() {
            self.execution.status = JobStatus::Cancelled;
            self.schedule.completed_at = Some(completed_at);
            self.schedule.next_eligible_at = None;
            Some(completed_at)
        } else {
            None
        }
    }

    /// Re-queue the job after memory pressure prevented dispatch.
    pub(crate) fn requeue_after_memory_gate(&mut self, retry_at: UnixTimestamp) {
        self.execution.status = JobStatus::Queued;
        self.schedule.completed_at = None;
        self.schedule.next_eligible_at = Some(retry_at);
    }

    /// Mark the job as actively running.
    pub(crate) fn mark_running(&mut self) {
        self.execution.status = JobStatus::Running;
        self.schedule.next_eligible_at = None;
    }

    /// Record the per-job worker count chosen for this run.
    pub(crate) fn record_worker_count(&mut self, num_workers: usize) {
        self.schedule.num_workers = Some(num_workers as i64);
    }

    /// Fail the job immediately with a job-level error message.
    pub(crate) fn fail(&mut self, error: &str, completed_at: UnixTimestamp) {
        self.execution.status = JobStatus::Failed;
        self.execution.error = Some(error.to_string());
        self.schedule.completed_at = Some(completed_at);
    }

    /// Force the job into a cancelled shutdown state and clear runtime claims.
    pub(crate) fn cancel_for_shutdown(&mut self, completed_at: UnixTimestamp) -> bool {
        self.runtime.cancel_token.cancel();
        if self.execution.status.can_cancel() {
            self.execution.status = JobStatus::Cancelled;
            self.schedule.completed_at = Some(completed_at);
            self.schedule.next_eligible_at = None;
            self.clear_lease();
            self.runtime.runner_active = false;
            true
        } else {
            false
        }
    }

    /// Finalize the job after all file tasks have stopped mutating its state.
    pub(crate) fn finalize(&mut self, final_status: JobStatus, completed_at: UnixTimestamp) {
        self.execution.status = final_status;
        self.schedule.completed_at = Some(completed_at);
        self.schedule.next_eligible_at = None;
        self.execution.completed_files = self
            .execution
            .file_statuses
            .values()
            .filter(|file_status| file_status.status.is_terminal())
            .count() as i64;
    }

    /// Reset the job so unfinished files may run again from queued state.
    pub(crate) fn prepare_for_restart(&mut self) {
        for file_status in self.execution.file_statuses.values_mut() {
            if file_status.status != FileStatusKind::Done {
                file_status.status = FileStatusKind::Queued;
                file_status.error = None;
                file_status.error_category = None;
                file_status.started_at = None;
                file_status.finished_at = None;
                file_status.next_eligible_at = None;
                file_status.current_attempt_id = None;
                file_status.progress_current = None;
                file_status.progress_total = None;
                file_status.progress_stage = None;
            }
        }

        self.execution.status = JobStatus::Queued;
        self.execution.error = None;
        self.schedule.completed_at = None;
        self.schedule.next_eligible_at = None;
        self.clear_lease();
        self.runtime.cancel_token = CancellationToken::new();
        self.runtime.runner_active = false;
        self.execution.completed_files = self
            .execution
            .file_statuses
            .values()
            .filter(|file_status| file_status.status == FileStatusKind::Done)
            .count() as i64;
        self.execution
            .results
            .retain(|result| result.error.is_none());
    }

    /// Reconcile a persisted interrupted/running job during startup recovery.
    pub(crate) fn reconcile_recovered_runtime_state(&mut self) -> RecoveryDisposition {
        let has_resumable = self
            .execution
            .file_statuses
            .values()
            .any(|file_status| file_status.status.is_resumable());

        if has_resumable {
            for file_status in self.execution.file_statuses.values_mut() {
                if file_status.status.is_resumable() {
                    file_status.status = FileStatusKind::Queued;
                    file_status.started_at = None;
                    file_status.finished_at = None;
                    file_status.next_eligible_at = None;
                    file_status.current_attempt_id = None;
                    file_status.progress_current = None;
                    file_status.progress_total = None;
                    file_status.progress_stage = None;
                }
            }
            self.execution.status = JobStatus::Queued;
            self.schedule.completed_at = None;
            self.schedule.next_eligible_at = None;
            self.clear_lease();
            RecoveryDisposition::Requeued
        } else {
            let all_errored = self
                .execution
                .file_statuses
                .values()
                .all(|file_status| file_status.status == FileStatusKind::Error);
            self.execution.status = if all_errored {
                JobStatus::Failed
            } else {
                JobStatus::Completed
            };
            self.execution.completed_files = self.total_files() as i64;
            self.schedule.next_eligible_at = None;
            self.clear_lease();
            if all_errored {
                RecoveryDisposition::Failed
            } else {
                RecoveryDisposition::Completed
            }
        }
    }

    fn active_lease(&self) -> Option<LeaseRecord> {
        Some(LeaseRecord {
            leased_by_node: self.schedule.lease.leased_by_node.clone()?,
            heartbeat_at: self.schedule.lease.heartbeat_at?,
            expires_at: self.schedule.lease.expires_at?,
        })
    }

    /// Convert to the API `JobInfo` response.
    pub fn to_info(&self) -> JobInfo {
        let file_statuses: Vec<FileStatusEntry> = self
            .execution
            .file_statuses
            .values()
            .map(|fs| fs.to_entry())
            .collect();
        let duration_s = self
            .schedule
            .completed_at
            .map(|c| DurationSeconds(c.0 - self.schedule.submitted_at.0));

        JobInfo {
            job_id: self.identity.job_id.clone(),
            status: self.execution.status,
            command: self.dispatch.command,
            options: self.dispatch.options.clone(),
            lang: self.dispatch.lang.clone(),
            source_dir: self.source.source_dir.as_str().to_owned(),
            total_files: self.total_files() as i64,
            completed_files: self.execution.completed_files,
            current_file: None,
            error: self.execution.error.clone(),
            file_statuses,
            submitted_at: Some(ts_iso(self.schedule.submitted_at)),
            submitted_by: if self.source.submitted_by.is_empty() {
                None
            } else {
                Some(self.source.submitted_by.clone())
            },
            submitted_by_name: if self.source.submitted_by_name.is_empty() {
                None
            } else {
                Some(self.source.submitted_by_name.clone())
            },
            completed_at: self.schedule.completed_at.map(ts_iso),
            duration_s,
            next_eligible_at: self.schedule.next_eligible_at,
            num_workers: self.schedule.num_workers,
            active_lease: self.active_lease(),
            batch_progress: None,
            control_plane: None,
        }
    }

    /// Convert to the API `JobListItem` summary.
    pub fn to_list_item(&self) -> JobListItem {
        let error_files = self
            .execution
            .file_statuses
            .values()
            .filter(|fs| fs.status == FileStatusKind::Error)
            .count() as i64;
        let duration_s = self
            .schedule
            .completed_at
            .map(|c| DurationSeconds(c.0 - self.schedule.submitted_at.0));

        JobListItem {
            job_id: self.identity.job_id.clone(),
            status: self.execution.status,
            command: self.dispatch.command,
            lang: self.dispatch.lang.clone(),
            source_dir: self.source.source_dir.as_str().to_owned(),
            total_files: self.total_files() as i64,
            completed_files: self.execution.completed_files,
            error_files,
            submitted_at: Some(ts_iso(self.schedule.submitted_at)),
            submitted_by: if self.source.submitted_by.is_empty() {
                None
            } else {
                Some(self.source.submitted_by.clone())
            },
            submitted_by_name: if self.source.submitted_by_name.is_empty() {
                None
            } else {
                Some(self.source.submitted_by_name.clone())
            },
            completed_at: self.schedule.completed_at.map(ts_iso),
            duration_s,
            next_eligible_at: self.schedule.next_eligible_at,
            num_workers: self.schedule.num_workers,
            active_lease: self.active_lease(),
            control_plane: None,
        }
    }

    /// Return the files that have not yet reached a terminal state.
    pub fn pending_files(&self) -> Vec<PendingJobFile> {
        self.filesystem
            .filenames
            .iter()
            .enumerate()
            .zip(self.filesystem.has_chat.iter().copied())
            .filter_map(|((file_index, filename), has_chat)| {
                let already_done = self
                    .execution
                    .file_statuses
                    .get(&**filename)
                    .map(|status| status.status.is_terminal())
                    .unwrap_or(false);
                if already_done {
                    None
                } else {
                    Some(PendingJobFile {
                        file_index,
                        filename: filename.clone(),
                        has_chat,
                    })
                }
            })
            .collect()
    }

    /// Create the immutable runner-facing snapshot for this job.
    pub fn to_runner_snapshot(&self) -> RunnerJobSnapshot {
        RunnerJobSnapshot {
            identity: RunnerJobIdentity {
                job_id: self.identity.job_id.clone(),
                correlation_id: self.identity.correlation_id.clone(),
            },
            dispatch: RunnerDispatchConfig {
                command: self.dispatch.command,
                lang: self.dispatch.lang.clone(),
                num_speakers: self.dispatch.num_speakers,
                options: self.dispatch.options.clone(),
                runtime_state: self.dispatch.runtime_state.clone(),
                debug_traces: self.dispatch.debug_traces,
            },
            filesystem: RunnerFilesystemConfig {
                paths_mode: self.filesystem.paths_mode,
                source_paths: self.filesystem.source_paths.clone(),
                output_paths: self.filesystem.output_paths.clone(),
                before_paths: self.filesystem.before_paths.clone(),
                staging_dir: self.filesystem.staging_dir.clone(),
                media_mapping: self.filesystem.media_mapping.clone(),
                media_subdir: self.filesystem.media_subdir.clone(),
                source_dir: self.source.source_dir.clone(),
            },
            cancel_token: self.runtime.cancel_token.clone(),
            pending_files: self.pending_files(),
        }
    }
}

// ---------------------------------------------------------------------------
// Conflict detection
// ---------------------------------------------------------------------------

/// Describes one file-level collision between an incoming job submission and an
/// existing active job.
///
/// Conflict detection is keyed on `(submitted_by, filename)`.  If the same
/// client tries to submit a file that is already being processed, one
/// `ConflictEntry` is produced per overlapping filename.  The entries are
/// returned in the 409 Conflict response so the client knows which files
/// collided and with which jobs.
#[derive(Debug)]
pub struct ConflictEntry {
    /// Basename of the conflicting file.
    pub filename: DisplayPath,
    /// Job ID of the existing active job that owns this file.
    pub job_id: JobId,
    /// Command of the existing active job.
    pub command: ReleasedCommand,
    /// Status of the existing active job.
    pub status: JobStatus,
}

pub(crate) fn find_conflicts(jobs: &HashMap<JobId, Job>, incoming: &Job) -> Vec<ConflictEntry> {
    let incoming_keys: std::collections::HashSet<(String, String)> = incoming
        .filesystem
        .filenames
        .iter()
        .map(|fn_| {
            let path = if incoming.source.source_dir.is_empty() {
                String::from(fn_.clone())
            } else {
                format!("{}/{fn_}", incoming.source.source_dir)
            };
            (incoming.source.submitted_by.clone(), path)
        })
        .collect();

    let mut conflicts = Vec::new();
    for active in jobs.values() {
        if !active.execution.status.is_active() {
            continue;
        }
        for fn_ in &active.filesystem.filenames {
            let path = if active.source.source_dir.is_empty() {
                String::from(fn_.clone())
            } else {
                format!("{}/{fn_}", active.source.source_dir)
            };
            let key = (active.source.submitted_by.clone(), path);
            if incoming_keys.contains(&key) {
                conflicts.push(ConflictEntry {
                    filename: fn_.clone(),
                    job_id: active.identity.job_id.clone(),
                    command: active.dispatch.command,
                    status: active.execution.status,
                });
            }
        }
    }
    conflicts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{CorrelationId, LanguageSpec, NumSpeakers};
    use crate::options::CommandOptions;
    use crate::store::FileStatus;
    use std::collections::BTreeMap;

    /// Build a small queued job for projection and conflict tests.
    fn sample_job(job_id: &str, filenames: &[&str]) -> Job {
        let file_statuses = filenames
            .iter()
            .map(|filename| {
                let name = DisplayPath::from(*filename);
                (String::from(name.clone()), FileStatus::new(name))
            })
            .collect();
        let has_chat = filenames.iter().map(|_| true).collect();

        Job {
            identity: JobIdentity {
                job_id: JobId::from(job_id),
                correlation_id: CorrelationId::from(format!("corr-{job_id}")),
            },
            dispatch: JobDispatchConfig {
                command: ReleasedCommand::Morphotag,
                lang: LanguageSpec::Resolved(crate::api::LanguageCode3::eng()),
                num_speakers: NumSpeakers(1),
                options: CommandOptions::Morphotag(crate::options::MorphotagOptions {
                    common: crate::options::CommonOptions::default(),
                    retokenize: false,
                    skipmultilang: false,
                    merge_abbrev: false.into(),
                }),
                runtime_state: BTreeMap::new(),
                debug_traces: false,
            },
            source: JobSourceContext {
                submitted_by: "127.0.0.1".into(),
                submitted_by_name: "localhost".into(),
                source_dir: "/corpus".into(),
            },
            filesystem: JobFilesystemConfig {
                filenames: filenames
                    .iter()
                    .map(|filename| DisplayPath::from(*filename))
                    .collect(),
                has_chat,
                staging_dir: "/tmp/job".into(),
                paths_mode: false,
                source_paths: Vec::new(),
                output_paths: Vec::new(),
                before_paths: Vec::new(),
                media_mapping: Default::default(),
                media_subdir: Default::default(),
                source_dir: Default::default(),
            },
            execution: JobExecutionState {
                status: JobStatus::Queued,
                file_statuses,
                results: Vec::new(),
                error: None,
                completed_files: 0,
            },
            schedule: JobScheduleState {
                submitted_at: UnixTimestamp(100.0),
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

    /// Pending files exclude terminal file states.
    #[test]
    fn pending_files_skip_terminal_entries() {
        let mut job = sample_job("job-1", &["a.cha", "b.cha"]);
        job.execution.file_statuses.get_mut("a.cha").unwrap().status = FileStatusKind::Done;

        let pending = job.pending_files();

        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].filename.as_ref(), "b.cha");
    }

    /// Conflict detection keys on submitter and source-scoped filename.
    #[test]
    fn find_conflicts_uses_submitter_and_source_scope() {
        let mut active = sample_job("active", &["a.cha"]);
        active.execution.status = JobStatus::Running;

        let incoming = sample_job("incoming", &["a.cha"]);
        let jobs = HashMap::from([(active.identity.job_id.clone(), active)]);

        let conflicts = find_conflicts(&jobs, &incoming);

        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].filename, "a.cha");
        assert_eq!(conflicts[0].job_id.as_ref(), "active");
    }

    /// Restart preparation keeps successful work and resets unfinished state.
    #[test]
    fn prepare_for_restart_resets_unfinished_state() {
        let mut job = sample_job("job-1", &["a.cha", "b.cha"]);
        job.execution.status = JobStatus::Failed;
        job.execution.error = Some("failed".into());
        job.execution.file_statuses.get_mut("a.cha").unwrap().status = FileStatusKind::Done;
        let retry_file = job.execution.file_statuses.get_mut("b.cha").unwrap();
        retry_file.status = FileStatusKind::Error;
        retry_file.error = Some("boom".into());
        retry_file.started_at = Some(UnixTimestamp(10.0));
        retry_file.finished_at = Some(UnixTimestamp(12.0));
        retry_file.progress_stage = Some(FileProgressStage::Aligning);
        job.execution.results.push(FileResultEntry {
            filename: DisplayPath::from("a.cha"),
            content_type: ContentType::Chat,
            error: None,
        });
        job.execution.results.push(FileResultEntry {
            filename: DisplayPath::from("b.cha"),
            content_type: ContentType::Chat,
            error: Some("boom".into()),
        });
        job.schedule.completed_at = Some(UnixTimestamp(20.0));
        job.schedule.next_eligible_at = Some(UnixTimestamp(25.0));
        job.schedule.lease.leased_by_node = Some(NodeId::from("node-1"));
        job.schedule.lease.expires_at = Some(UnixTimestamp(30.0));
        job.schedule.lease.heartbeat_at = Some(UnixTimestamp(28.0));
        job.runtime.runner_active = true;

        job.prepare_for_restart();

        assert_eq!(job.execution.status, JobStatus::Queued);
        assert_eq!(job.execution.error, None);
        assert_eq!(job.execution.completed_files, 1);
        assert_eq!(job.execution.results.len(), 1);
        assert_eq!(
            job.execution.file_statuses["a.cha"].status,
            FileStatusKind::Done
        );
        assert_eq!(
            job.execution.file_statuses["b.cha"].status,
            FileStatusKind::Queued
        );
        assert!(job.schedule.completed_at.is_none());
        assert!(job.schedule.next_eligible_at.is_none());
        assert!(job.schedule.lease.leased_by_node.is_none());
        assert!(!job.runtime.runner_active);
    }

    /// Recovery re-queues interrupted jobs when resumable file work remains.
    #[test]
    fn reconcile_recovered_runtime_state_requeues_resumable_files() {
        let mut job = sample_job("job-1", &["a.cha", "b.cha"]);
        job.execution.status = JobStatus::Running;
        job.execution.file_statuses.get_mut("a.cha").unwrap().status = FileStatusKind::Done;
        let resumable = job.execution.file_statuses.get_mut("b.cha").unwrap();
        resumable.status = FileStatusKind::Interrupted;
        resumable.started_at = Some(UnixTimestamp(10.0));
        resumable.finished_at = Some(UnixTimestamp(11.0));
        job.schedule.completed_at = Some(UnixTimestamp(20.0));
        job.schedule.next_eligible_at = Some(UnixTimestamp(21.0));
        job.schedule.lease.leased_by_node = Some(NodeId::from("node-1"));
        job.schedule.lease.expires_at = Some(UnixTimestamp(30.0));
        job.schedule.lease.heartbeat_at = Some(UnixTimestamp(28.0));

        let disposition = job.reconcile_recovered_runtime_state();

        assert_eq!(disposition, RecoveryDisposition::Requeued);
        assert_eq!(job.execution.status, JobStatus::Queued);
        assert_eq!(
            job.execution.file_statuses["b.cha"].status,
            FileStatusKind::Queued
        );
        assert!(job.execution.file_statuses["b.cha"].started_at.is_none());
        assert!(job.schedule.completed_at.is_none());
        assert!(job.schedule.lease.leased_by_node.is_none());
    }

    /// Recovery promotes all-terminal interrupted jobs to a lease-free final state.
    #[test]
    fn reconcile_recovered_runtime_state_promotes_terminal_jobs() {
        let mut job = sample_job("job-1", &["a.cha", "b.cha"]);
        job.execution.status = JobStatus::Interrupted;
        job.execution.file_statuses.get_mut("a.cha").unwrap().status = FileStatusKind::Done;
        let failed = job.execution.file_statuses.get_mut("b.cha").unwrap();
        failed.status = FileStatusKind::Error;
        failed.error = Some("boom".into());
        job.schedule.completed_at = Some(UnixTimestamp(20.0));
        job.schedule.next_eligible_at = Some(UnixTimestamp(21.0));
        job.schedule.lease.leased_by_node = Some(NodeId::from("node-1"));
        job.schedule.lease.expires_at = Some(UnixTimestamp(30.0));
        job.schedule.lease.heartbeat_at = Some(UnixTimestamp(28.0));

        let disposition = job.reconcile_recovered_runtime_state();

        assert_eq!(disposition, RecoveryDisposition::Completed);
        assert_eq!(job.execution.status, JobStatus::Completed);
        assert_eq!(job.execution.completed_files, 2);
        assert!(job.schedule.next_eligible_at.is_none());
        assert!(job.schedule.lease.leased_by_node.is_none());
        assert!(job.schedule.lease.expires_at.is_none());
    }

    /// Local queue claims and renewals stay on the job boundary.
    #[test]
    fn local_dispatch_claim_and_renew_roundtrip() {
        let mut job = sample_job("job-1", &["a.cha"]);
        let node_id = NodeId::from("node-a");
        let claimed = job
            .claim_for_local_dispatch(&node_id, UnixTimestamp(10.0), 30.0)
            .expect("claim");

        assert_eq!(claimed.leased_by_node, node_id);
        assert_eq!(claimed.heartbeat_at, UnixTimestamp(10.0));
        assert_eq!(claimed.expires_at, UnixTimestamp(40.0));
        assert!(job.runtime.runner_active);

        let renewed = job
            .renew_local_dispatch_lease(&node_id, UnixTimestamp(20.0), 30.0)
            .expect("renew");
        assert_eq!(renewed.heartbeat_at, UnixTimestamp(20.0));
        assert_eq!(renewed.expires_at, UnixTimestamp(50.0));

        job.release_local_dispatch_claim();
        assert!(!job.runtime.runner_active);
        assert!(job.schedule.lease.leased_by_node.is_none());
    }

    /// Jobs with live leases or deferrals do not report ready for dispatch.
    #[test]
    fn ready_for_local_dispatch_respects_leases_and_deferrals() {
        let mut job = sample_job("job-1", &["a.cha"]);
        let now = UnixTimestamp(10.0);
        assert!(job.ready_for_local_dispatch(now));

        job.schedule.next_eligible_at = Some(UnixTimestamp(20.0));
        assert!(!job.ready_for_local_dispatch(now));
        assert_eq!(
            job.next_local_dispatch_wake_at(now),
            Some(UnixTimestamp(20.0))
        );

        job.schedule.next_eligible_at = None;
        job.schedule.lease.leased_by_node = Some(NodeId::from("node-a"));
        job.schedule.lease.expires_at = Some(UnixTimestamp(30.0));
        assert!(!job.ready_for_local_dispatch(now));
        assert_eq!(
            job.next_local_dispatch_wake_at(now),
            Some(UnixTimestamp(30.0))
        );
    }

    /// File completion mutates file state and appends a success result.
    #[test]
    fn mark_file_done_updates_file_state() {
        let mut job = sample_job("job-1", &["a.cha"]);
        job.execution.file_statuses.get_mut("a.cha").unwrap().error = Some("stale".into());
        job.execution
            .file_statuses
            .get_mut("a.cha")
            .unwrap()
            .error_category = Some(crate::scheduling::FailureCategory::WorkerTimeout);

        assert!(job.mark_file_done(
            "a.cha",
            UnixTimestamp(12.0),
            Some(CompletedFileOutput {
                filename: DisplayPath::from("a.cha"),
                content_type: ContentType::Chat,
            })
        ));

        assert_eq!(
            job.execution.file_statuses["a.cha"].status,
            FileStatusKind::Done
        );
        assert_eq!(
            job.execution.file_statuses["a.cha"].finished_at,
            Some(UnixTimestamp(12.0))
        );
        assert!(job.execution.file_statuses["a.cha"].error.is_none());
        assert!(
            job.execution.file_statuses["a.cha"]
                .error_category
                .is_none()
        );
        assert_eq!(job.execution.completed_files, 1);
        assert_eq!(job.execution.results.len(), 1);
    }

    /// Retry scheduling stays on the job boundary.
    #[test]
    fn mark_file_retry_pending_sets_retry_metadata() {
        let mut job = sample_job("job-1", &["a.cha"]);

        assert!(job.mark_file_retry_pending(
            "a.cha",
            &FileRetryRecord {
                message: "retry".into(),
                category: crate::scheduling::FailureCategory::WorkerTimeout,
                finished_at: UnixTimestamp(11.0),
                retry_at: UnixTimestamp(20.0),
            }
        ));

        let file_status = &job.execution.file_statuses["a.cha"];
        assert_eq!(file_status.status, FileStatusKind::Processing);
        assert_eq!(file_status.next_eligible_at, Some(UnixTimestamp(20.0)));
        assert_eq!(
            file_status.progress_stage,
            Some(FileProgressStage::RetryScheduled)
        );
    }

    /// Clearing retry state also clears stale retry errors before a new attempt.
    #[test]
    fn clear_file_retry_state_clears_retry_error_metadata() {
        let mut job = sample_job("job-1", &["a.cha"]);
        assert!(job.mark_file_retry_pending(
            "a.cha",
            &FileRetryRecord {
                message: "retry".into(),
                category: crate::scheduling::FailureCategory::WorkerTimeout,
                finished_at: UnixTimestamp(11.0),
                retry_at: UnixTimestamp(20.0),
            }
        ));

        assert!(job.clear_file_retry_state("a.cha"));
        let file_status = &job.execution.file_statuses["a.cha"];
        assert!(file_status.error.is_none());
        assert!(file_status.error_category.is_none());
        assert!(file_status.finished_at.is_none());
        assert!(file_status.next_eligible_at.is_none());
    }
}
