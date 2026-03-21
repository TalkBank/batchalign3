//! Read/load operations on the `jobs`, `file_statuses`, and `attempts` tables.

use crate::api::{JobId, NodeId, UnixTimestamp};
use crate::options::CommandOptions;
use crate::scheduling::{AttemptId, AttemptRecord, FailureCategory, WorkUnitId};
use crate::worker::WorkerPid;
use sqlx::Row;

use crate::error::ServerError;

use super::{AttemptRow, FileStatusRow, JobDB, JobRow};

impl JobDB {
    /// Load all jobs with their file_statuses for startup recovery.
    pub async fn load_all_jobs(&self) -> Result<Vec<JobRow>, ServerError> {
        let rows = sqlx::query(
            "SELECT job_id, command, lang, num_speakers, status, error,
                    staging_dir, filenames, has_chat, options,
                    engine_overrides, media_mapping, media_subdir,
                    source_dir, submitted_by,
                    COALESCE(submitted_by_name, '') as submitted_by_name,
                    submitted_at, completed_at, num_workers, next_eligible_at,
                    leased_by_node, lease_expires_at, lease_heartbeat_at,
                    COALESCE(paths_mode, 0) as paths_mode,
                    COALESCE(source_paths, '[]') as source_paths,
                    COALESCE(output_paths, '[]') as output_paths,
                    COALESCE(correlation_id, '') as correlation_id
             FROM jobs
             ORDER BY submitted_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut jobs: Vec<JobRow> = Vec::new();
        for row in &rows {
            let filenames_json: String = row.try_get("filenames")?;
            let has_chat_json: String = row.try_get("has_chat")?;
            let options_json: String = row.try_get("options")?;
            let source_paths_json: String = row.try_get("source_paths")?;
            let output_paths_json: String = row.try_get("output_paths")?;
            let paths_mode_int: i32 = row.try_get("paths_mode")?;

            let job_id: String = row.try_get("job_id")?;
            let command: String = row.try_get("command")?;
            let options: CommandOptions = deserialize_job_field(&job_id, "options", &options_json)?;
            let filenames = deserialize_job_field(&job_id, "filenames", &filenames_json)?;
            let has_chat = deserialize_job_field(&job_id, "has_chat", &has_chat_json)?;
            let source_paths = deserialize_job_field(&job_id, "source_paths", &source_paths_json)?;
            let output_paths = deserialize_job_field(&job_id, "output_paths", &output_paths_json)?;

            let job = JobRow {
                job_id,
                correlation_id: row.try_get("correlation_id")?,
                command,
                lang: row.try_get("lang")?,
                num_speakers: row.try_get("num_speakers")?,
                status: row.try_get("status")?,
                error: row.try_get("error")?,
                staging_dir: row.try_get("staging_dir")?,
                filenames,
                has_chat,
                options,
                media_mapping: row.try_get("media_mapping")?,
                media_subdir: row.try_get("media_subdir")?,
                source_dir: row.try_get("source_dir")?,
                submitted_by: row.try_get("submitted_by")?,
                submitted_by_name: row.try_get("submitted_by_name")?,
                submitted_at: row.try_get("submitted_at")?,
                completed_at: row.try_get("completed_at")?,
                num_workers: row.try_get("num_workers")?,
                next_eligible_at: row.try_get("next_eligible_at")?,
                leased_by_node: row.try_get("leased_by_node")?,
                lease_expires_at: row.try_get("lease_expires_at")?,
                lease_heartbeat_at: row.try_get("lease_heartbeat_at")?,
                paths_mode: paths_mode_int != 0,
                source_paths,
                output_paths,
                file_statuses: Vec::new(),
            };
            jobs.push(job);
        }

        // Load file statuses for each job (N+1 pattern preserved)
        for job in &mut jobs {
            let fs_rows = sqlx::query(
                "SELECT filename, status, error, error_category,
                        COALESCE(bug_report_id, '') as bug_report_id,
                        content_type, started_at, finished_at, next_eligible_at
                 FROM file_statuses
                 WHERE job_id = ?",
            )
            .bind(&job.job_id)
            .fetch_all(&self.pool)
            .await?;

            for fs_row in &fs_rows {
                let bug_report_raw: String = fs_row.try_get("bug_report_id")?;
                let bug_report_id = if bug_report_raw.is_empty() {
                    None
                } else {
                    Some(bug_report_raw)
                };
                job.file_statuses.push(FileStatusRow {
                    filename: fs_row.try_get("filename")?,
                    status: fs_row.try_get("status")?,
                    error: fs_row.try_get("error")?,
                    error_category: fs_row.try_get("error_category")?,
                    bug_report_id,
                    content_type: fs_row.try_get("content_type")?,
                    started_at: fs_row.try_get("started_at")?,
                    finished_at: fs_row.try_get("finished_at")?,
                    next_eligible_at: fs_row.try_get("next_eligible_at")?,
                });
            }
        }

        Ok(jobs)
    }

    /// Load persisted attempts for one job, ordered by start time.
    pub async fn load_attempts_for_job(
        &self,
        job_id: &str,
    ) -> Result<Vec<AttemptRecord>, ServerError> {
        let rows = sqlx::query(
            "SELECT attempt_id, job_id, work_unit_id, work_unit_kind,
                    attempt_number, started_at, finished_at, outcome,
                    failure_category, disposition, worker_node_id, worker_pid
             FROM attempts
             WHERE job_id = ?
             ORDER BY started_at ASC, attempt_number ASC",
        )
        .bind(job_id)
        .fetch_all(&self.pool)
        .await?;

        let mut attempts = Vec::with_capacity(rows.len());
        for row in &rows {
            let attempt_row = AttemptRow {
                attempt_id: row.try_get("attempt_id")?,
                job_id: row.try_get("job_id")?,
                work_unit_id: row.try_get("work_unit_id")?,
                work_unit_kind: row.try_get("work_unit_kind")?,
                attempt_number: row.try_get("attempt_number")?,
                started_at: row.try_get("started_at")?,
                finished_at: row.try_get("finished_at")?,
                outcome: row.try_get("outcome")?,
                failure_category: row.try_get("failure_category")?,
                disposition: row.try_get("disposition")?,
                worker_node_id: row.try_get("worker_node_id")?,
                worker_pid: row.try_get("worker_pid")?,
            };
            attempts.push(attempt_row.try_into()?);
        }

        Ok(attempts)
    }
}

impl TryFrom<AttemptRow> for AttemptRecord {
    type Error = ServerError;

    fn try_from(row: AttemptRow) -> Result<Self, Self::Error> {
        let work_unit_kind = row.work_unit_kind.parse().map_err(|raw: String| {
            ServerError::Validation(format!(
                "invalid persisted work_unit_kind '{}': {raw}",
                row.work_unit_kind
            ))
        })?;
        let outcome = row.outcome.parse().map_err(|raw: String| {
            ServerError::Validation(format!(
                "invalid persisted attempt outcome '{}': {raw}",
                row.outcome
            ))
        })?;
        let disposition = row.disposition.parse().map_err(|raw: String| {
            ServerError::Validation(format!(
                "invalid persisted retry disposition '{}': {raw}",
                row.disposition
            ))
        })?;
        let failure_category = row
            .failure_category
            .as_deref()
            .map(str::parse::<FailureCategory>)
            .transpose()
            .map_err(|raw| {
                ServerError::Validation(format!(
                    "invalid persisted failure category for attempt '{}': {raw}",
                    row.attempt_id
                ))
            })?;

        Ok(AttemptRecord {
            attempt_id: AttemptId(row.attempt_id),
            job_id: JobId(row.job_id),
            work_unit_id: WorkUnitId(row.work_unit_id),
            work_unit_kind,
            attempt_number: row.attempt_number as u32,
            started_at: UnixTimestamp(row.started_at),
            finished_at: row.finished_at.map(UnixTimestamp),
            outcome,
            failure_category,
            disposition,
            worker_node_id: row.worker_node_id.map(NodeId),
            worker_pid: row.worker_pid.map(|pid| WorkerPid(pid as u32)),
        })
    }
}

fn deserialize_job_field<T: serde::de::DeserializeOwned>(
    job_id: &str,
    field_name: &str,
    raw_json: &str,
) -> Result<T, ServerError> {
    serde_json::from_str(raw_json).map_err(|error| {
        ServerError::Persistence(format!(
            "failed to deserialize jobs.{field_name} for job {job_id}: {error}; raw={raw_json}"
        ))
    })
}
