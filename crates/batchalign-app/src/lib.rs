#![warn(missing_docs)]
//! Axum-based HTTP server for the batchalign3 processing pipeline.
//!
//! This crate implements the batchalign3 server: an axum REST API that
//! accepts NLP processing jobs (morphosyntax, forced alignment, utterance
//! segmentation, translation, coreference resolution), dispatches them to
//! Python worker processes via [`batchalign_app::worker::pool::WorkerPool`], and
//! returns results over HTTP and WebSocket.
//!
//! The server **never loads ML models directly**. All inference runs in
//! Python worker processes managed by the worker pool. The server owns the
//! CHAT format lifecycle (parse, validate, cache lookup, inject results,
//! serialize) so that Python workers are stateless `(text, lang) -> NLP`
//! inference endpoints.
//!
//! # Architecture
//!
//! ```text
//!                       +-----------+
//!                       |  CLI /    |
//!                       |  Browser  |
//!                       +-----+-----+
//!                             |
//!                    HTTP / WebSocket
//!                             |
//!                   +---------v---------+
//!                   |    axum Router    |
//!                   |  (routes, middleware)
//!                   +---------+---------+
//!                             |
//!          +------------------+------------------+
//!          |                  |                   |
//!   +------v------+   +------v------+   +--------v--------+
//!   |   JobStore  |   |   runner    |   |   WebSocket /   |
//!   | (in-memory  |   | (per-job    |   |   SSE stream    |
//!   |  + SQLite)  |   |  tokio task)|   |  (broadcast)    |
//!   +------+------+   +------+------+   +-----------------+
//!          |                  |
//!          |           +------v------+
//!          |           | WorkerPool  |
//!          |           | (semaphore  |
//!          |           |  + channel) |
//!          |           +------+------+
//!          |                  |
//!          |           stdio JSON-lines IPC
//!          |                  |
//!          |         +--------v--------+
//!          |         | Python workers  |
//!          |         | (Stanza, Whisper|
//!          |         |  Rev.AI, etc.)  |
//!          |         +-----------------+
//!          |
//!   +------v------+
//!   |   SQLite    |
//!   |  (jobs.db)  |
//!   +-------------+
//! ```
//!
//! # Endpoints
//!
//! | Method | Path                        | Description                               |
//! |--------|-----------------------------|-------------------------------------------|
//! | GET    | `/health`                   | Server version, capabilities, worker state |
//! | POST   | `/jobs/submit`              | Submit a new processing job                |
//! | GET    | `/jobs`                     | List all jobs                              |
//! | GET    | `/jobs/{id}`                | Get job details                            |
//! | GET    | `/jobs/{id}/results`        | Download completed results                 |
//! | GET    | `/jobs/{id}/results/{file}` | Download a single result file              |
//! | GET    | `/jobs/{id}/stream`         | SSE stream of real-time job progress       |
//! | DELETE | `/jobs/{id}`                | Cancel a running job                       |
//! | POST   | `/jobs/{id}/restart`        | Restart a failed/completed job             |
//! | DELETE | `/jobs/{id}/delete`         | Permanently delete a job                   |
//! | GET    | `/media/list`               | List media files from configured roots     |
//! | GET    | `/bug-reports`              | List filed bug reports                     |
//! | GET    | `/bug-reports/{id}`         | Get a single bug report                    |
//! | GET    | `/dashboard/**`             | Static dashboard SPA                       |
//! | GET    | `/ws`                       | WebSocket for real-time updates            |
//!
//! # Usage
//!
//! The primary entry point is [`create_app`], which builds the axum router
//! and shared application state. For production use, [`serve`] binds to a
//! TCP listener with graceful shutdown handling.
//!
//! ```rust,no_run
//! use batchalign_app::{create_app, serve};
//! use batchalign_app::config::ServerConfig;
//! use batchalign_app::worker::pool::PoolConfig;
//!
//! # async fn example() -> Result<(), batchalign_app::error::ServerError> {
//! // Load or construct a server config
//! let config = ServerConfig::default();
//! let pool_config = PoolConfig::default();
//!
//! // Option A: get the router for custom binding / testing
//! let (router, state) = create_app(
//!     config.clone(),
//!     pool_config.clone(),
//!     None,  // jobs_dir (default: ~/.batchalign3/jobs/)
//!     None,  // db_dir   (default: ~/.batchalign3/)
//!     None,  // build_hash
//! ).await?;
//!
//! // Option B: serve on the configured host:port with graceful shutdown
//! serve(config, pool_config, None).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Module map
//!
//! | Module          | Responsibility                                                   |
//! |-----------------|------------------------------------------------------------------|
//! | [`routes`]      | HTTP route handlers with middleware (CORS, tracing)               |
//! | [`store`]       | JobStore control plane: JobRegistry, counters, SQLite write-through, conflict detection |
//! | [`runner`]      | Per-job async tasks: dispatch to workers, track per-file progress |
//! | [`db`]          | SQLite persistence layer (WAL mode, crash recovery, TTL pruning) |
//! | [`ws`]          | WebSocket broadcast event types and channel setup                |
//! | [`media`]       | Media file resolution across configured roots with walk cache    |
//! | [`runtime_supervisor`] | Owns queue-dispatch and per-job background tasks         |
//! | [`error`]       | Typed server errors mapping to HTTP status codes                 |
//! | [`hostname`]    | Tailscale-based IP-to-hostname resolution                        |
//! | [`openapi`]     | OpenAPI 3.0 schema generation via utoipa                         |
//! | [`morphosyntax`]| Server-side morphosyntax orchestrator (parse/cache/infer/inject) |
//! | [`utseg`]       | Server-side utterance segmentation orchestrator                  |
//! | [`translate`]   | Server-side translation orchestrator                             |
//! | [`coref`]       | Server-side coreference resolution orchestrator (document-level) |
//! | [`fa`]          | Server-side forced alignment orchestrator (per-file, audio-aware)|

pub mod types;
// Re-export non-conflicting types modules at crate root for flat access.
// `types::worker` is NOT re-exported because it conflicts with `crate::worker`
// (the WorkerHandle/WorkerPool module). Access types::worker items via
// `crate::types::worker::` or the re-exports below.
pub use types::{api, config, options, params, runtime, scheduling, traces};

pub mod benchmark;
pub mod cache;
pub mod compare;
pub mod coref;
pub mod db;
pub mod ensure_wav;
pub mod error;
pub mod fa;
pub mod hostname;
mod infer_retry;
pub mod media;
pub mod morphosyntax;
pub mod openapi;
mod pipeline;
mod queue;
pub(crate) mod revai;
pub mod routes;
pub mod runner;
pub mod runtime_paths;
pub(crate) mod runtime_supervisor;
pub mod server;
pub mod state;
pub mod store;
pub mod trace_store;
pub mod transcribe;
pub mod translate;
pub mod utseg;
pub(crate) mod websocket;
pub mod worker;
pub mod ws;

// Re-export primary API surface from submodules.
pub use server::{
    PreparedWorkers, create_app, create_app_with_prepared_workers, create_app_with_runtime,
    prepare_workers, serve, serve_with_runtime,
};
pub use state::AppState;
pub(crate) use websocket::ws_route;
