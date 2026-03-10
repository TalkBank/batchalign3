//! Job lifecycle mutations: submit, restart, cancel, cancel_all.

use crate::api::{JobId, JobInfo, JobStatus};
use crate::db::NewJobRecord;
use tracing::{info, warn};

use super::super::job::Job;
use super::super::{JobStore, PersistedJobUpdate, unix_now};
use crate::error::ServerError;

impl JobStore {
    /// Register a job. Returns an error if conflicts are detected.
    pub async fn submit(&self, job: Job) -> Result<(), ServerError> {
        let persist = NewJobRecord {
            job_id: String::from(job.identity.job_id.clone()),
            correlation_id: job.identity.correlation_id.clone(),
            command: String::from(job.dispatch.command.clone()),
            lang: String::from(job.dispatch.lang.clone()),
            num_speakers: job.dispatch.num_speakers.0,
            status: job.execution.status.to_string(),
            staging_dir: job.filesystem.staging_dir.clone(),
            filenames: job
                .filesystem
                .filenames
                .iter()
                .cloned()
                .map(String::from)
                .collect(),
            has_chat: job.filesystem.has_chat.clone(),
            options: job.dispatch.options.clone(),
            media_mapping: job.filesystem.media_mapping.clone(),
            media_subdir: job.filesystem.media_subdir.clone(),
            source_dir: job.source.source_dir.clone(),
            submitted_by: job.source.submitted_by.clone(),
            submitted_by_name: job.source.submitted_by_name.clone(),
            submitted_at: job.schedule.submitted_at.0,
            paths_mode: job.filesystem.paths_mode,
            source_paths: job.filesystem.source_paths.clone(),
            output_paths: job.filesystem.output_paths.clone(),
        };
        let job_id = job.identity.job_id.clone();
        let correlation_id = job.identity.correlation_id.clone();
        let command = job.dispatch.command.clone();
        let total_files = job.total_files();
        self.registry.insert_checked(job).await?;

        // Persist to DB
        if let Some(db) = &self.db
            && let Err(e) = db.insert_job(&persist).await
        {
            warn!(job_id = %job_id, error = %e, "Failed to persist job to DB");
        }

        info!(
            job_id = %job_id,
            correlation_id = %correlation_id,
            command = %command,
            total_files = total_files,
            "Job queued"
        );

        Ok(())
    }

    /// Restart a cancelled or failed job — reset file statuses and re-queue.
    pub async fn restart(&self, job_id: &JobId) -> Result<JobInfo, ServerError> {
        let info = self
            .registry
            .restart_job(job_id)
            .await
            .ok_or_else(|| ServerError::JobNotFound(job_id.clone()))??;
        self.notify_job_item(info.job_update);

        self.db_update_job(
            job_id,
            PersistedJobUpdate {
                status: JobStatus::Queued,
                error: None,
                completed_at: None,
                num_workers: None,
                next_eligible_at: None,
            },
        )
        .await;
        if let Some(db) = &self.db
            && let Err(e) = db.update_job_lease(job_id, None, None, None).await
        {
            warn!(job_id = %job_id, error = %e, "DB update_job_lease failed on restart");
        }

        Ok(info.info)
    }

    /// Cancel all active (running or queued) jobs.
    ///
    /// Returns the number of jobs cancelled.
    pub async fn cancel_all(&self) -> usize {
        self.registry.cancel_all_active(unix_now()).await
    }
}
