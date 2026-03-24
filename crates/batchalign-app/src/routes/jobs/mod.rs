//! Job lifecycle endpoints.
//!
//! Covers the full lifecycle of a processing job: submission, polling, result
//! retrieval, cancellation, deletion, restart, and real-time SSE streaming.
//! All handlers share an `Arc<AppState>` and coordinate through the in-memory
//! `JobStore` (backed by SQLite for crash recovery).

pub(crate) mod detail;
pub(crate) mod lifecycle;
pub(crate) mod stream;

pub(crate) use detail::{get_job, get_results, get_single_result};
pub(crate) use lifecycle::{cancel_job, delete_job, restart_job};
pub(crate) use stream::stream_job;

// Re-export utoipa-generated path structs so that the `OpenApi` derive in
// `openapi.rs` can resolve them at `crate::routes::jobs::__path_*`.
#[allow(unused_imports)]
pub(crate) use detail::{__path_get_job, __path_get_results, __path_get_single_result};
#[allow(unused_imports)]
pub(crate) use lifecycle::{__path_cancel_job, __path_delete_job, __path_restart_job};
#[allow(unused_imports)]
pub(crate) use stream::__path_stream_job;

use std::collections::{BTreeSet, HashMap};
use std::net::SocketAddr;
use std::sync::Arc;

use crate::api::{ReleasedCommand, CorrelationId, DisplayPath, JobInfo, JobStatus, JobSubmission};
use axum::extract::State;
use axum::extract::connect_info::ConnectInfo;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use tracing::info;

use crate::AppState;
use crate::error::ServerError;
use crate::hostname::resolve_hostname;
use crate::store::{
    FileStatus, Job, JobDispatchConfig, JobExecutionState, JobFilesystemConfig, JobIdentity,
    JobLeaseState, JobRuntimeControl, JobScheduleState, JobSourceContext, unix_now,
};
use tokio_util::sync::CancellationToken;

/// Build the jobs router with all job lifecycle endpoints.
///
/// Registers routes for submission, listing, detail, results retrieval,
/// per-file results, cancellation, deletion, restart, and SSE streaming.
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/jobs", post(submit_job))
        .route("/jobs", get(list_jobs))
        .route("/jobs/{job_id}", get(get_job))
        .route("/jobs/{job_id}/results", get(get_results))
        .route("/jobs/{job_id}/results/{filename}", get(get_single_result))
        .route("/jobs/{job_id}/cancel", post(cancel_job))
        .route("/jobs/{job_id}", delete(delete_job))
        .route("/jobs/{job_id}/restart", post(restart_job))
        .route("/jobs/{job_id}/stream", get(stream_job))
}

/// Maximum length for a sanitized correlation ID.
///
/// Client-supplied `X-Request-Id` values are truncated to this length after
/// stripping non-safe characters, so that log lines and database rows stay
/// bounded even if a client sends a very long header.
const CORRELATION_ID_MAX_LEN: usize = 128;

fn command_supported(command: ReleasedCommand, capabilities: &[String]) -> bool {
    capabilities.iter().any(|c| c.as_str() == command.as_ref())
}

fn sanitize_correlation_id(raw: &str) -> Option<CorrelationId> {
    let mut out = String::new();
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | ':') {
            out.push(ch);
            if out.len() >= CORRELATION_ID_MAX_LEN {
                break;
            }
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(CorrelationId::from(out))
    }
}

fn correlation_id_from_headers(headers: &HeaderMap, fallback: &str) -> CorrelationId {
    headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .and_then(sanitize_correlation_id)
        .unwrap_or_else(|| CorrelationId::from(fallback.to_string()))
}

fn supported_command_list(capabilities: &[String]) -> Vec<String> {
    let mut set: BTreeSet<String> = BTreeSet::new();
    for c in capabilities {
        if c != "test-echo" {
            set.insert(c.clone());
        }
    }
    set.into_iter().collect()
}

fn path_mode_filename_from_source(source_path: &str) -> Result<DisplayPath, ServerError> {
    let file_name = std::path::Path::new(source_path)
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            ServerError::Validation(format!(
                "paths_mode source path has no filename component: {source_path}"
            ))
        })?;
    Ok(DisplayPath::from(file_name.to_string_lossy().to_string()))
}

async fn ensure_dir(path: &std::path::Path, context: &str) -> Result<(), ServerError> {
    tokio::fs::create_dir_all(path).await.map_err(|error| {
        ServerError::Io(std::io::Error::new(
            error.kind(),
            format!("{context} {}: {error}", path.display()),
        ))
    })
}

async fn write_staged_file(path: &std::path::Path, content: &str) -> Result<(), ServerError> {
    tokio::fs::write(path, content).await.map_err(|error| {
        ServerError::Io(std::io::Error::new(
            error.kind(),
            format!("staging input file {}: {error}", path.display()),
        ))
    })
}

/// Accept a new processing job and begin execution.
///
/// Validates the command against built-in tasks and worker-advertised capabilities,
/// stages input files (content mode) or records source paths (paths mode), detects
/// `(submitted_by, filename)` conflicts with active jobs, and spawns a background
/// runner task that acquires workers and dispatches files. The response echoes the
/// job back as `JobInfo` and includes the correlation ID in `X-Request-Id`.
#[utoipa::path(
    post,
    path = "/jobs",
    tag = "jobs",
    request_body = JobSubmission,
    responses(
        (status = 200, description = "Submitted job", body = JobInfo),
        (status = 400, description = "Validation error", body = crate::openapi::ErrorResponse)
    )
)]
pub(crate) async fn submit_job(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(submission): Json<JobSubmission>,
) -> Result<impl IntoResponse, ServerError> {
    // Validate command
    if !command_supported(submission.command, &state.workers.capabilities) {
        let supported = supported_command_list(&state.workers.capabilities);
        return Err(ServerError::UnknownCommand(format!(
            "Unknown command: {}. Valid commands: {:?}",
            submission.command, supported
        )));
    }

    // Validate paths_mode
    if let Err(e) = submission.validate() {
        return Err(ServerError::Validation(e.to_string()));
    }

    let job_id = uuid::Uuid::new_v4().to_string()[..12].to_string();
    let correlation_id = correlation_id_from_headers(&headers, &job_id);

    let (filenames, has_chat, staging_dir, paths_mode, source_paths, output_paths) =
        if submission.paths_mode {
            // Paths mode: derive filenames from source_paths or display_names
            let filenames: Vec<DisplayPath> = if !submission.display_names.is_empty() {
                submission
                    .display_names
                    .iter()
                    .map(|s| DisplayPath::from(s.as_str()))
                    .collect()
            } else {
                submission
                    .source_paths
                    .iter()
                    .map(|sp| path_mode_filename_from_source(sp))
                    .collect::<Result<Vec<_>, _>>()?
            };
            let has_chat: Vec<bool> = submission
                .source_paths
                .iter()
                .map(|sp| sp.to_lowercase().ends_with(".cha"))
                .collect();

            let staging_dir = format!("{}/{job_id}", state.environment.paths.jobs_dir);
            ensure_dir(
                std::path::Path::new(&staging_dir),
                "creating paths-mode staging dir",
            )
            .await?;

            (
                filenames,
                has_chat,
                staging_dir,
                true,
                submission.source_paths.clone(),
                submission.output_paths.clone(),
            )
        } else {
            // Content mode
            if submission.files.is_empty() && submission.media_files.is_empty() {
                return Err(ServerError::Validation(
                    "Must provide at least one file or media_file.".into(),
                ));
            }

            let staging_dir = format!("{}/{job_id}", state.environment.paths.jobs_dir);
            let input_dir = format!("{staging_dir}/input");
            ensure_dir(
                std::path::Path::new(&input_dir),
                "creating content-mode input dir",
            )
            .await?;
            ensure_dir(
                std::path::Path::new(&format!("{staging_dir}/output")),
                "creating content-mode output dir",
            )
            .await?;

            let mut filenames: Vec<DisplayPath> = Vec::new();
            let mut has_chat = Vec::new();

            for fp in &submission.files {
                filenames.push(fp.filename.clone());
                has_chat.push(true);
                let dest = format!("{input_dir}/{}", fp.filename);
                if let Some(parent) = std::path::Path::new(&dest).parent() {
                    ensure_dir(parent, "creating staged input parent dir").await?;
                }
                write_staged_file(std::path::Path::new(&dest), &fp.content).await?;
            }
            for media_name in &submission.media_files {
                filenames.push(DisplayPath::from(media_name.as_str()));
                has_chat.push(false);
            }

            (
                filenames,
                has_chat,
                staging_dir,
                false,
                Vec::new(),
                Vec::new(),
            )
        };

    let mut file_statuses = HashMap::new();
    for f in &filenames {
        file_statuses.insert(f.to_string(), FileStatus::new(f.clone()));
    }

    info!(
        job_id = %job_id,
        correlation_id = %correlation_id,
        command = %submission.command,
        total_files = filenames.len(),
        paths_mode = paths_mode,
        submitted_by = %addr.ip(),
        "Job submission accepted"
    );

    let job = Job {
        identity: JobIdentity {
            job_id: job_id.clone().into(),
            correlation_id: correlation_id.clone(),
        },
        dispatch: JobDispatchConfig {
            command: submission.command.clone(),
            lang: submission.lang.clone(),
            num_speakers: submission.num_speakers,
            options: submission.options.clone(),
            runtime_state: std::collections::BTreeMap::new(),
            debug_traces: submission.debug_traces,
        },
        source: JobSourceContext {
            submitted_by: addr.ip().to_string(),
            submitted_by_name: resolve_hostname(&addr.ip()),
            source_dir: submission.source_dir.clone().into(),
        },
        filesystem: JobFilesystemConfig {
            filenames: filenames.clone(),
            has_chat,
            staging_dir: staging_dir.into(),
            paths_mode,
            source_paths: source_paths.into_iter().map(Into::into).collect(),
            output_paths: output_paths.into_iter().map(Into::into).collect(),
            before_paths: if submission.paths_mode {
                submission
                    .before_paths
                    .iter()
                    .map(|p| p.clone().into())
                    .collect()
            } else {
                Vec::new()
            },
            media_mapping: submission.media_mapping.clone(),
            media_subdir: submission.media_subdir.clone(),
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
    };

    let info = job.to_info();
    state.control.store.submit(job).await?;

    state.control.queue.notify();

    let mut response_headers = HeaderMap::new();
    if let Ok(request_id) = HeaderValue::from_str(&correlation_id) {
        response_headers.insert("x-request-id", request_id);
    }

    Ok((StatusCode::OK, response_headers, Json(info)))
}

/// Return a summary of every job the server knows about (active, completed,
/// failed, or cancelled).
///
/// Used by the dashboard and CLI `jobs` command. The response is intentionally
/// compact -- per-file detail is omitted and available via `GET /jobs/{id}`.
#[utoipa::path(
    get,
    path = "/jobs",
    tag = "jobs",
    responses(
        (status = 200, description = "List all jobs", body = [crate::api::JobListItem])
    )
)]
pub(crate) async fn list_jobs(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<crate::api::JobListItem>> {
    Json(state.control.store.list_all().await)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advertised_command_is_supported() {
        assert!(command_supported(
            ReleasedCommand::Morphotag,
            &["morphotag".to_string()]
        ));
    }

    #[test]
    fn command_not_in_capabilities_is_rejected() {
        assert!(!command_supported(
            ReleasedCommand::Morphotag,
            &["align".to_string()]
        ));
    }

    #[test]
    fn supported_list_uses_advertised_capabilities_only() {
        let caps = vec![
            "cantotag".to_string(),
            "morphotag".to_string(),
            "test-echo".to_string(),
        ];
        let supported = supported_command_list(&caps);
        assert!(supported.iter().any(|c| c == "morphotag"));
        assert!(supported.iter().any(|c| c == "cantotag"));
        assert!(!supported.iter().any(|c| c == "test-echo"));
    }

    #[test]
    fn correlation_id_uses_x_request_id_when_valid() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-request-id",
            axum::http::HeaderValue::from_static("req-123_abc"),
        );
        let cid = correlation_id_from_headers(&headers, "fallback");
        assert_eq!(cid, "req-123_abc");
    }

    #[test]
    fn correlation_id_falls_back_when_missing() {
        let headers = HeaderMap::new();
        let cid = correlation_id_from_headers(&headers, "job123");
        assert_eq!(cid, "job123");
    }

    #[test]
    fn correlation_id_sanitizes_invalid_chars() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-request-id",
            axum::http::HeaderValue::from_static("abc$%/def"),
        );
        let cid = correlation_id_from_headers(&headers, "job123");
        assert_eq!(cid, "abcdef");
    }
}
