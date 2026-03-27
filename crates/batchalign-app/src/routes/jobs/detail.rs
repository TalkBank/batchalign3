//! Job detail and result retrieval endpoints.

use std::sync::Arc;

use crate::api::{
    ContentType, DisplayPath, FileResult, FileStatusKind, JobId, JobInfo, JobResultResponse,
    JobStatus,
};
use axum::Json;
use axum::extract::{Path, State};

use crate::AppState;
use crate::error::ServerError;

/// Return full detail for a single job, including per-file statuses.
///
/// This is the primary polling endpoint for the CLI: the client calls it
/// repeatedly until `status` reaches a terminal value.
#[utoipa::path(
    get,
    path = "/jobs/{job_id}",
    tag = "jobs",
    params(
        ("job_id" = String, Path, description = "Job identifier")
    ),
    responses(
        (status = 200, description = "Job detail", body = JobInfo),
        (status = 404, description = "Job not found", body = crate::openapi::ErrorResponse)
    )
)]
pub(crate) async fn get_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Result<Json<JobInfo>, ServerError> {
    let job_id = JobId::from(job_id);
    state
        .control
        .backend
        .get_job(&job_id)
        .await
        .map(Json)
        .ok_or_else(|| ServerError::JobNotFound(job_id))
}

/// Retrieve all output files for a terminal job.
///
/// Blocks until the job has completed, failed, or been interrupted -- returns
/// 409 for jobs that are still running or queued. In content mode the response
/// includes the full CHAT text of each output file; in paths mode only filenames
/// are returned (the files were written directly to disk by the runner).
/// Cancelled jobs are intentionally excluded because they may have incomplete
/// results.
#[utoipa::path(
    get,
    path = "/jobs/{job_id}/results",
    tag = "jobs",
    params(
        ("job_id" = String, Path, description = "Job identifier")
    ),
    responses(
        (status = 200, description = "Results for terminal jobs", body = JobResultResponse),
        (status = 404, description = "Job not found", body = crate::openapi::ErrorResponse)
    )
)]
pub(crate) async fn get_results(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Result<Json<JobResultResponse>, ServerError> {
    let job_id = JobId::from(job_id);
    let detail = state
        .control
        .backend
        .get_job_detail(&job_id)
        .await
        .ok_or_else(|| ServerError::JobNotFound(job_id.clone()))?;

    // Intentionally excludes Cancelled — cancelled jobs may have incomplete results.
    if !matches!(
        detail.status,
        JobStatus::Completed | JobStatus::Failed | JobStatus::Interrupted
    ) {
        return Err(ServerError::JobNotTerminal(JobId::from(format!(
            "Job {job_id} is still {}",
            detail.status
        ))));
    }

    if detail.paths_mode {
        let files: Vec<FileResult> = detail
            .results
            .iter()
            .map(|r| FileResult {
                filename: r.filename.clone(),
                content: String::new(),
                content_type: r.content_type,
                error: r.error.clone(),
            })
            .collect();
        return Ok(Json(JobResultResponse {
            job_id: job_id.clone(),
            status: detail.status,
            files,
        }));
    }

    // Content mode — read from staging dir
    let output_dir = detail.staging_dir.join("output");
    let mut files = Vec::new();
    for r in &detail.results {
        let content = if r.error.is_none() {
            let path = output_dir.join(&*r.filename);
            read_result_content(&path).await?
        } else {
            String::new()
        };
        files.push(FileResult {
            filename: r.filename.clone(),
            content,
            content_type: r.content_type,
            error: r.error.clone(),
        });
    }

    Ok(Json(JobResultResponse {
        job_id,
        status: detail.status,
        files,
    }))
}

/// Retrieve the output for a single file within a job.
///
/// Waits for the individual file to reach a terminal status, not the whole job,
/// so clients can fetch results incrementally as files finish. Supports
/// stem-based matching (e.g., requesting `sample.wav` will find `sample.cha`)
/// to handle the common case where the output extension differs from the input.
#[utoipa::path(
    get,
    path = "/jobs/{job_id}/results/{filename}",
    tag = "jobs",
    params(
        ("job_id" = String, Path, description = "Job identifier"),
        ("filename" = String, Path, description = "Requested output filename")
    ),
    responses(
        (status = 200, description = "One file result", body = FileResult),
        (status = 404, description = "File not found", body = crate::openapi::ErrorResponse)
    )
)]
pub(crate) async fn get_single_result(
    State(state): State<Arc<AppState>>,
    Path((job_id, filename)): Path<(String, String)>,
) -> Result<Json<FileResult>, ServerError> {
    let job_id = JobId::from(job_id);
    let detail = state
        .control
        .backend
        .get_job_detail(&job_id)
        .await
        .ok_or_else(|| ServerError::JobNotFound(job_id.clone()))?;

    // Find file status
    let fs = detail
        .file_statuses
        .iter()
        .find(|fs| *fs.filename == *filename)
        .ok_or_else(|| {
            ServerError::FileNotFound(format!("File {filename} not found in job {job_id}"))
        })?;

    if !fs.status.is_terminal() {
        return Err(ServerError::FileNotReady(format!(
            "File {filename} is still {}",
            fs.status
        )));
    }

    if fs.status == FileStatusKind::Error {
        return Ok(Json(FileResult {
            filename: DisplayPath::from(filename.as_str()),
            content: String::new(),
            content_type: ContentType::Chat,
            error: fs.error.clone().or(Some("Unknown error".into())),
        }));
    }

    // Find result entry (handles filename renaming e.g. .wav -> .cha)
    let result_entry = detail
        .results
        .iter()
        .find(|r| *r.filename == *filename)
        .or_else(|| {
            let stem = std::path::Path::new(&filename)
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy();
            detail.results.iter().find(|r| {
                r.error.is_none()
                    && std::path::Path::new(r.filename.as_ref())
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        == stem
            })
        })
        .ok_or_else(|| {
            ServerError::FileNotFound(format!("Result for {filename} not found in job {job_id}"))
        })?;

    let content_type = result_entry.content_type;
    let out_filename = result_entry.filename.clone();

    let content = if !detail.paths_mode {
        let path = detail.staging_dir.join("output").join(&*out_filename);
        read_result_content(&path).await?
    } else {
        String::new()
    };

    Ok(Json(FileResult {
        filename: out_filename,
        content,
        content_type,
        error: None,
    }))
}

async fn read_result_content(path: &std::path::Path) -> Result<String, ServerError> {
    tokio::fs::read_to_string(path).await.map_err(|error| {
        ServerError::Io(std::io::Error::new(
            error.kind(),
            format!("failed to read result file {}: {error}", path.display()),
        ))
    })
}
