//! SSE streaming endpoint for real-time job progress.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use crate::api::{JobInfo, JobStatus};
use axum::extract::{Path, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use tokio_stream::StreamExt;

use crate::AppState;
use crate::error::ServerError;
use crate::ws::WsEvent;

/// SSE stream for real-time per-file progress of a single job.
///
/// Sends:
/// - `snapshot` event with current file statuses on connect
/// - `file_update` events as files are processed
/// - `complete` event when the job reaches a terminal state, then closes
#[utoipa::path(
    get,
    path = "/jobs/{job_id}/stream",
    tag = "jobs",
    params(
        ("job_id" = String, Path, description = "Job identifier")
    ),
    responses(
        (status = 200, description = "SSE event stream"),
        (status = 404, description = "Job not found", body = crate::openapi::ErrorResponse)
    )
)]
pub(crate) async fn stream_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, ServerError> {
    // Validate job exists
    let job_id = crate::api::JobId::from(job_id);
    let initial_info = state
        .control
        .store
        .get(&job_id)
        .await
        .ok_or_else(|| ServerError::JobNotFound(job_id.clone()))?;

    // Subscribe to broadcast BEFORE building the snapshot to avoid missing events.
    let rx = state.control.ws_tx.subscribe();

    let stream = async_stream(rx, String::from(job_id), initial_info);

    Ok(Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15))))
}

/// Build the SSE stream from a broadcast receiver.
fn async_stream(
    rx: tokio::sync::broadcast::Receiver<WsEvent>,
    job_id: String,
    initial_info: JobInfo,
) -> impl tokio_stream::Stream<Item = Result<Event, Infallible>> {
    // Use BroadcastStream to convert the receiver into a Stream.
    let broadcast_stream = tokio_stream::wrappers::BroadcastStream::new(rx);

    // Chain: initial snapshot event, then filtered broadcast events.
    let snapshot_data = serde_json::json!({
        "job_id": initial_info.job_id,
        "status": initial_info.status,
        "file_statuses": initial_info.file_statuses,
        "completed_files": initial_info.completed_files,
        "total_files": initial_info.total_files,
    });
    let snapshot_event: Result<Event, Infallible> = Ok(Event::default()
        .event("snapshot")
        .json_data(snapshot_data)
        .unwrap_or_else(|_| Event::default().event("snapshot").data("{}")));

    // Check if the job is already terminal — if so, send snapshot + complete and close.
    let already_terminal = initial_info.status.is_terminal();

    let initial = tokio_stream::once(snapshot_event);

    if already_terminal {
        let complete_event: Result<Event, Infallible> = Ok(Event::default()
            .event("complete")
            .json_data(serde_json::json!({
                "job_id": initial_info.job_id,
                "status": initial_info.status,
            }))
            .unwrap_or_else(|_| Event::default().event("complete").data("{}")));
        // Return snapshot + complete, then stop.
        let tail = tokio_stream::once(complete_event);
        // Use StreamExt to chain, then take(2) to ensure bounded.
        return EitherStream::Left(initial.chain(tail));
    }

    // Filter broadcast events for this job_id.
    let job_id_clone = job_id.clone();
    let filtered = broadcast_stream.filter_map(move |result| {
        let event = match result {
            Ok(event) => event,
            Err(_) => return None, // Lagged or closed
        };

        match &event {
            WsEvent::FileUpdate {
                job_id: eid,
                file,
                completed_files,
            } if *eid == job_id_clone => {
                let data = serde_json::json!({
                    "job_id": eid,
                    "file": file,
                    "completed_files": completed_files,
                });
                Some(Ok(Event::default()
                    .event("file_update")
                    .json_data(data)
                    .unwrap_or_else(|_| {
                        Event::default().event("file_update").data("{}")
                    })))
            }
            WsEvent::JobUpdate { job } => {
                let event_job_id = job.get("job_id").and_then(|v| v.as_str()).unwrap_or("");
                if event_job_id != job_id_clone {
                    return None;
                }
                let status_str = job.get("status").and_then(|v| v.as_str()).unwrap_or("");
                let is_terminal = status_str
                    .parse::<JobStatus>()
                    .map(|s| s.is_terminal())
                    .unwrap_or(false);
                if is_terminal {
                    Some(Ok(Event::default()
                        .event("complete")
                        .json_data(serde_json::json!({
                            "job_id": event_job_id,
                            "status": status_str,
                        }))
                        .unwrap_or_else(|_| {
                            Event::default().event("complete").data("{}")
                        })))
                } else {
                    Some(Ok(Event::default()
                        .event("job_update")
                        .json_data(job.clone())
                        .unwrap_or_else(|_| {
                            Event::default().event("job_update").data("{}")
                        })))
                }
            }
            _ => None,
        }
    });

    // The stream runs until the broadcast channel closes or the client disconnects.
    // The client closes on receiving a 'complete' event.
    EitherStream::Right(initial.chain(filtered))
}

/// Unifies two stream types so `async_stream` can return a bounded
/// snapshot-only stream (`Left`) for already-terminal jobs or a live
/// broadcast-backed stream (`Right`) for in-progress jobs, without boxing.
enum EitherStream<L, R> {
    Left(L),
    Right(R),
}

impl<L, R, T> tokio_stream::Stream for EitherStream<L, R>
where
    L: tokio_stream::Stream<Item = T> + Unpin,
    R: tokio_stream::Stream<Item = T> + Unpin,
{
    type Item = T;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        match self.get_mut() {
            EitherStream::Left(s) => std::pin::Pin::new(s).poll_next(cx),
            EitherStream::Right(s) => std::pin::Pin::new(s).poll_next(cx),
        }
    }
}
