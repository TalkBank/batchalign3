//! Job lifecycle management endpoints: cancel, delete, restart.

use std::sync::Arc;

use crate::api::{JobId, JobInfo, JobStatus};
use axum::Json;
use axum::extract::{Path, State};

use crate::AppState;
use crate::error::ServerError;

/// Request cancellation of a running or queued job.
///
/// Fires the job's `CancellationToken`, which the runner checks between files.
/// Already-terminal jobs (completed, failed, cancelled) are no-ops. Interrupted
/// jobs can still be cancelled -- they are not considered terminal here because
/// they may be holding worker resources.
#[utoipa::path(
    post,
    path = "/jobs/{job_id}/cancel",
    tag = "jobs",
    params(
        ("job_id" = String, Path, description = "Job identifier")
    ),
    responses(
        (status = 200, description = "Cancel request accepted", body = crate::openapi::StatusMessageResponse),
        (status = 404, description = "Job not found", body = crate::openapi::ErrorResponse)
    )
)]
pub(crate) async fn cancel_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let job_id = JobId::from(job_id);
    let status = state
        .control
        .store
        .job_status(&job_id)
        .await
        .ok_or_else(|| ServerError::JobNotFound(job_id.clone()))?;

    // Intentionally excludes Interrupted — interrupted jobs can still be cancelled.
    if matches!(
        status,
        JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled
    ) {
        return Ok(Json(serde_json::json!({
            "status": status.to_string(),
            "message": "Job already finished."
        })));
    }

    state.control.store.cancel(&job_id).await?;
    Ok(Json(serde_json::json!({
        "status": "cancelled",
        "message": format!("Job {job_id} cancelled.")
    })))
}

/// Permanently remove a terminal job and its associated state.
///
/// Returns 409 if the job is still running -- the caller must cancel it first.
/// Deleting a job removes it from the in-memory store and SQLite, and broadcasts
/// a `JobDeleted` event to connected WebSocket/SSE clients.
#[utoipa::path(
    delete,
    path = "/jobs/{job_id}",
    tag = "jobs",
    params(
        ("job_id" = String, Path, description = "Job identifier")
    ),
    responses(
        (status = 200, description = "Job deleted", body = crate::openapi::StatusMessageResponse),
        (status = 404, description = "Job not found", body = crate::openapi::ErrorResponse),
        (status = 409, description = "Job still running", body = crate::openapi::ErrorResponse)
    )
)]
pub(crate) async fn delete_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let job_id = JobId::from(job_id);
    let is_running = state
        .control
        .store
        .is_running(&job_id)
        .await
        .ok_or_else(|| ServerError::JobNotFound(job_id.clone()))?;

    if is_running {
        return Err(ServerError::JobConflict {
            message: format!("Job {job_id} is running — cancel it first."),
            conflicts: Vec::new(),
        });
    }

    state.control.store.delete(&job_id).await?;
    Ok(Json(serde_json::json!({
        "status": "deleted",
        "message": format!("Job {job_id} deleted.")
    })))
}

/// Reset a failed or interrupted job back to `Queued` and re-run it.
///
/// Only jobs in a terminal non-cancelled state can be restarted. The store
/// resets per-file statuses and clears errors, then a fresh runner task is
/// spawned. This is the primary recovery path after transient worker crashes
/// or OOM kills.
#[utoipa::path(
    post,
    path = "/jobs/{job_id}/restart",
    tag = "jobs",
    params(
        ("job_id" = String, Path, description = "Job identifier")
    ),
    responses(
        (status = 200, description = "Job restarted", body = JobInfo),
        (status = 404, description = "Job not found", body = crate::openapi::ErrorResponse),
        (status = 409, description = "Job not restartable", body = crate::openapi::ErrorResponse)
    )
)]
pub(crate) async fn restart_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Result<Json<JobInfo>, ServerError> {
    let job_id = JobId::from(job_id);
    let info = state.control.store.restart(&job_id).await?;

    state.control.queue.notify();

    Ok(Json(info))
}
