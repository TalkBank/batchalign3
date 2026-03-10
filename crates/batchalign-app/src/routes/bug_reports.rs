//! Bug report endpoints — `GET /bug-reports` and `GET /bug-reports/{id}`.
//!
//! Reads JSON files from the bug-reports directory (`~/.batchalign3/bug-reports/`
//! by default, configurable via `AppState.environment.paths.bug_reports_dir`).

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::routing::get;
use axum::{Json, Router};
use tracing::warn;

use crate::AppState;
use crate::error::ServerError;

/// Query parameters for `GET /bug-reports`.
///
/// `limit` caps the number of reports returned (newest first). The default
/// of 50 keeps the response lightweight for the dashboard while still
/// surfacing recent issues.
#[derive(serde::Deserialize)]
pub struct ListParams {
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    50
}

/// Build the bug-reports router (`GET /bug-reports`, `GET /bug-reports/{id}`).
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/bug-reports", get(list_bug_reports))
        .route("/bug-reports/{id}", get(get_bug_report))
}

/// Return the most recent bug reports, sorted newest-first.
///
/// Bug reports are JSON files written to disk by the validation layer when
/// it detects semantic errors (e.g., alignment mismatches, monotonicity
/// violations). This endpoint scans the reports directory by mtime so the
/// dashboard can surface recent failures without requiring database queries.
/// Missing or unreadable files are silently skipped.
#[utoipa::path(
    get,
    path = "/bug-reports",
    tag = "bug-reports",
    params(
        ("limit" = usize, Query, description = "Maximum number of reports to return")
    ),
    responses(
        (status = 200, description = "Bug report documents (newest first)")
    )
)]
pub(crate) async fn list_bug_reports(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListParams>,
) -> Json<Vec<serde_json::Value>> {
    let dir = &state.environment.paths.bug_reports_dir;

    let entries = match tokio::fs::read_dir(dir).await {
        Ok(entries) => entries,
        Err(_) => return Json(Vec::new()),
    };

    // Collect .json files with metadata for sorting
    let mut files: Vec<(std::time::SystemTime, std::path::PathBuf)> = Vec::new();
    let mut entries = entries;
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "json") {
            let mtime = entry
                .metadata()
                .await
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(std::time::UNIX_EPOCH);
            files.push((mtime, path));
        }
    }

    // Sort by mtime descending (newest first)
    files.sort_by(|a, b| b.0.cmp(&a.0));
    files.truncate(params.limit);

    let mut reports = Vec::new();
    for (_mtime, path) in &files {
        match tokio::fs::read_to_string(path).await {
            Ok(content) => match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(value) => reports.push(value),
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "Invalid bug report JSON");
                }
            },
            Err(e) => {
                warn!(path = %path.display(), error = %e, "Failed to read bug report");
            }
        }
    }

    Json(reports)
}

/// Return a single bug report by its ID (filename stem).
///
/// The ID corresponds to the JSON filename on disk without the `.json`
/// extension. Returns the raw JSON document as-is so the dashboard can
/// render full diagnostic detail (stack traces, file paths, expected vs.
/// actual values).
#[utoipa::path(
    get,
    path = "/bug-reports/{id}",
    tag = "bug-reports",
    params(
        ("id" = String, Path, description = "Bug report ID")
    ),
    responses(
        (status = 200, description = "Bug report JSON document"),
        (status = 404, description = "Bug report not found", body = crate::openapi::ErrorResponse)
    )
)]
pub(crate) async fn get_bug_report(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let path =
        std::path::Path::new(&state.environment.paths.bug_reports_dir).join(format!("{id}.json"));

    let content = tokio::fs::read_to_string(&path)
        .await
        .map_err(|_| ServerError::FileNotFound(format!("Bug report {id} not found")))?;

    let value: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| ServerError::Validation(format!("Invalid bug report JSON: {e}")))?;

    Ok(Json(value))
}
