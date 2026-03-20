//! `GET /health` — health check endpoint.
//!
//! Provides a point-in-time snapshot of server health for use by the CLI
//! (server discovery) and the dashboard (at-a-glance status). The response
//! includes worker availability, operational error counters, and loaded
//! pipeline summaries.

use std::sync::Arc;

use crate::api::{HealthResponse, HealthStatus, MemoryMb};
use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};

use crate::AppState;

/// Build the health-check router (`GET /health`).
pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/health", get(health))
}

/// Return a point-in-time health snapshot used by the CLI for server discovery
/// and the dashboard for at-a-glance status.
///
/// The response includes worker availability, operational error counters
/// (crashes, forced-terminal errors, memory-gate aborts), and the list of
/// loaded pipelines so callers can decide whether this server is suitable
/// for a given command without probing individual workers.
#[utoipa::path(
    get,
    path = "/health",
    tag = "health",
    responses(
        (status = 200, description = "Server health snapshot", body = HealthResponse)
    )
)]
pub(crate) async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
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

    // System memory snapshot for the dashboard memory panel.
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let total_mb = MemoryMb(sys.total_memory() / (1024 * 1024));
    let available_mb = MemoryMb(sys.available_memory() / (1024 * 1024));
    let used_mb = MemoryMb(total_mb.0.saturating_sub(available_mb.0));
    let gate_mb = state.environment.config.memory_gate_mb;

    Json(HealthResponse {
        status: HealthStatus::Ok,
        version: state.build.version.clone(),
        node_id: state.control.store.node_id().clone(),
        free_threaded: false, // Rust server dispatches to Python workers
        capabilities: state.workers.capabilities.clone(),
        loaded_pipelines: worker_summary,
        media_roots: state.environment.config.media_roots.clone(),
        media_mapping_keys: state
            .environment
            .config
            .media_mappings
            .keys()
            .cloned()
            .collect(),
        workers_available,
        job_slots_available: workers_available,
        live_workers,
        live_worker_keys,
        active_jobs,
        cache_backend: "sqlite".into(),
        redis_cache_enabled: false,
        redis_cache_connected: false,
        worker_crashes,
        attempts_started,
        attempts_retried,
        deferred_work_units,
        forced_terminal_errors,
        memory_gate_aborts,
        build_hash: state.build.build_hash.clone(),
        warmup_status: state.workers.pool.warmup_status(),
        system_memory_total_mb: total_mb,
        system_memory_available_mb: available_mb,
        system_memory_used_mb: used_mb,
        memory_gate_threshold_mb: gate_mb,
    })
}
