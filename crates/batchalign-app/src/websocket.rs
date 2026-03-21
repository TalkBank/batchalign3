//! WebSocket route and handler for real-time job updates.

use std::sync::Arc;

use axum::Router;
use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::routing::get;
use tokio::sync::broadcast;
use tracing::warn;

use crate::state::AppState;
use crate::ws::WsEvent;

/// Build the WebSocket route as a separate router.
pub(crate) fn ws_route(_state: Arc<AppState>) -> Router<Arc<AppState>> {
    Router::new().route("/ws", get(ws_handler))
}

async fn ws_handler(State(state): State<Arc<AppState>>, ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state))
}

async fn handle_ws(mut socket: WebSocket, state: Arc<AppState>) {
    // Send initial snapshot
    let jobs = state.control.store.list_all().await;
    let mut jobs_json = Vec::with_capacity(jobs.len());
    for job in &jobs {
        match serde_json::to_value(job) {
            Ok(value) => jobs_json.push(value),
            Err(error) => {
                warn!(
                    job_id = %job.job_id,
                    error = %error,
                    "Skipping job in WS snapshot because serialization failed"
                );
            }
        }
    }

    let (
        worker_crashes,
        attempts_started,
        attempts_retried,
        deferred_work_units,
        forced_terminal_errors,
        memory_gate_aborts,
    ) = state.control.store.operational_counters().await;
    let workers_available = state.control.store.workers_available().await;
    let active_jobs = state.control.store.active_jobs().await;
    let live_workers = state.workers.pool.worker_count() as i64;
    let live_worker_keys = state.workers.pool.worker_keys();
    let worker_summary = state.workers.pool.worker_summary();

    let health = serde_json::json!({
        "status": "ok",
        "version": state.build.version.clone(),
        "free_threaded": false,
        "capabilities": state.workers.capabilities.clone(),
        "loaded_pipelines": worker_summary,
        "media_roots": state.environment.config.media_roots.clone(),
        "media_mapping_keys": state.environment.config.media_mappings.keys().collect::<Vec<_>>(),
        "workers_available": workers_available,
        "job_slots_available": workers_available,
        "live_workers": live_workers,
        "live_worker_keys": live_worker_keys,
        "active_jobs": active_jobs,
        "cache_backend": "sqlite",
        "redis_cache_enabled": false,
        "redis_cache_connected": false,
        "worker_crashes": worker_crashes,
        "attempts_started": attempts_started,
        "attempts_retried": attempts_retried,
        "deferred_work_units": deferred_work_units,
        "forced_terminal_errors": forced_terminal_errors,
        "memory_gate_aborts": memory_gate_aborts,
        "build_hash": state.build.build_hash.clone(),
    });

    let snapshot = WsEvent::Snapshot {
        jobs: jobs_json,
        health,
    };

    match serde_json::to_string(&snapshot) {
        Ok(json) => {
            if socket.send(Message::Text(json.into())).await.is_err() {
                return;
            }
        }
        Err(error) => {
            warn!(error = %error, "Failed to serialize initial WS snapshot");
            return;
        }
    }

    // Subscribe to broadcast channel
    let mut rx = state.control.ws_tx.subscribe();

    // Main loop: forward broadcast events and handle pings
    loop {
        tokio::select! {
            // Forward broadcast events to this client
            msg = rx.recv() => {
                match msg {
                    Ok(event) => {
                        match serde_json::to_string(&event) {
                            Ok(json) => {
                                if socket.send(Message::Text(json.into())).await.is_err() {
                                    break;
                                }
                            }
                            Err(error) => {
                                warn!(error = %error, "Failed to serialize WS event");
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(lagged = n, "WS client lagged, skipping messages");
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        break;
                    }
                }
            }
            // Handle incoming messages from client
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if text == "ping"
                            && socket.send(Message::Text("pong".into())).await.is_err()
                        {
                            break;
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        break;
                    }
                    _ => {}
                }
            }
        }
    }
}
