//! Insert operations on the `jobs` and `file_statuses` tables.

use crate::options::CommandOptions;
use crate::scheduling::{AttemptOutcome, RetryDisposition, WorkUnitKind};

use crate::error::ServerError;

use super::JobDB;

/// Typed payload for inserting a new job row and its initial file-status rows.
///
/// This replaces the old long primitive-heavy `insert_job(...)` parameter list.
/// The record is constructed at the store boundary and passed as a single value
/// into the persistence layer.
pub struct NewJobRecord {
    /// Job identifier.
    pub job_id: String,
    /// Correlation identifier for tracing.
    pub correlation_id: String,
    /// Command name.
    pub command: String,
    /// Language code.
    pub lang: String,
    /// Speaker count.
    pub num_speakers: u32,
    /// Initial persisted status.
    pub status: String,
    /// Staging directory path.
    pub staging_dir: String,
    /// Ordered filenames.
    pub filenames: Vec<String>,
    /// Parallel CHAT/media markers.
    pub has_chat: Vec<bool>,
    /// Typed command options.
    pub options: CommandOptions,
    /// Media mapping key.
    pub media_mapping: String,
    /// Media subdirectory.
    pub media_subdir: String,
    /// User-facing source directory.
    pub source_dir: String,
    /// Submitting client address.
    pub submitted_by: String,
    /// Human-readable submitter name.
    pub submitted_by_name: String,
    /// Submission timestamp.
    pub submitted_at: f64,
    /// Whether the job uses direct filesystem paths.
    pub paths_mode: bool,
    /// Absolute source paths for paths mode.
    pub source_paths: Vec<String>,
    /// Absolute output paths for paths mode.
    pub output_paths: Vec<String>,
}

impl JobDB {
    /// Insert a new job row and one `file_statuses` row per filename, wrapped
    /// in a single SQLite transaction.
    ///
    /// # Errors
    ///
    /// Returns `ServerError::Database` if the transaction fails (e.g. duplicate
    /// `job_id` or disk-full).  On error the transaction is rolled back.
    pub async fn insert_job(&self, job: &NewJobRecord) -> Result<(), ServerError> {
        let filenames_json = serde_json::to_string(&job.filenames).unwrap_or_default();
        let has_chat_json = serde_json::to_string(&job.has_chat).unwrap_or_default();
        let options_json = serde_json::to_string(&job.options).unwrap_or_default();
        // engine_overrides stored as empty — overrides are inside CommandOptions.common.engine_overrides
        let engine_json = "{}";
        let source_paths_json = serde_json::to_string(&job.source_paths).unwrap_or_default();
        let output_paths_json = serde_json::to_string(&job.output_paths).unwrap_or_default();
        let paths_mode_int = job.paths_mode as i32;

        let mut tx = self.pool.begin().await?;

        sqlx::query(
            "INSERT INTO jobs (
                job_id, command, lang, num_speakers, status,
                staging_dir, filenames, has_chat,
                options, engine_overrides,
                media_mapping, media_subdir, source_dir,
                submitted_by, submitted_by_name, submitted_at,
                paths_mode, source_paths, output_paths, correlation_id,
                leased_by_node, lease_expires_at, lease_heartbeat_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)",
        )
        .bind(&job.job_id)
        .bind(&job.command)
        .bind(&job.lang)
        .bind(job.num_speakers)
        .bind(&job.status)
        .bind(&job.staging_dir)
        .bind(&filenames_json)
        .bind(&has_chat_json)
        .bind(&options_json)
        .bind(engine_json)
        .bind(&job.media_mapping)
        .bind(&job.media_subdir)
        .bind(&job.source_dir)
        .bind(&job.submitted_by)
        .bind(&job.submitted_by_name)
        .bind(job.submitted_at)
        .bind(paths_mode_int)
        .bind(&source_paths_json)
        .bind(&output_paths_json)
        .bind(&job.correlation_id)
        .execute(&mut *tx)
        .await?;

        for filename in &job.filenames {
            sqlx::query(
                "INSERT INTO file_statuses (job_id, filename, status) VALUES (?, ?, 'queued')",
            )
            .bind(&job.job_id)
            .bind(filename)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    /// Create and persist a new attempt record for a work unit.
    ///
    /// The attempt number is allocated transactionally as `MAX(attempt_number)+1`
    /// for `(job_id, work_unit_id)`, so repeated attempts on the same file keep
    /// a stable monotonic sequence even as fleet support evolves.
    pub async fn insert_attempt_start(
        &self,
        job_id: &str,
        work_unit_id: &str,
        work_unit_kind: WorkUnitKind,
        started_at: f64,
        worker_node_id: Option<&str>,
        worker_pid: Option<u32>,
    ) -> Result<(String, u32), ServerError> {
        let mut tx = self.pool.begin().await?;

        let attempt_number: i64 = sqlx::query_scalar(
            "SELECT COALESCE(MAX(attempt_number), 0) + 1
             FROM attempts
             WHERE job_id = ? AND work_unit_id = ?",
        )
        .bind(job_id)
        .bind(work_unit_id)
        .fetch_one(&mut *tx)
        .await?;

        let attempt_id = format!("{job_id}:{work_unit_id}:{attempt_number}");
        let worker_pid_i64 = worker_pid.map(i64::from);

        sqlx::query(
            "INSERT INTO attempts (
                attempt_id, job_id, work_unit_id, work_unit_kind,
                attempt_number, started_at, outcome, disposition,
                worker_node_id, worker_pid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&attempt_id)
        .bind(job_id)
        .bind(work_unit_id)
        .bind(work_unit_kind.to_string())
        .bind(attempt_number)
        .bind(started_at)
        .bind(AttemptOutcome::Deferred.to_string())
        .bind(RetryDisposition::Defer.to_string())
        .bind(worker_node_id)
        .bind(worker_pid_i64)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;

        Ok((attempt_id, attempt_number as u32))
    }
}
