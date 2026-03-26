//! Python worker process manager: spawn, health-check, dispatch.
//!
//! This crate manages Python worker child processes that do the actual ML
//! inference. The Rust binary is the control plane; Python workers are the
//! data plane.
//!
//! # Architecture
//!
//! ```text
//! WorkerPool
//!   ├── WorkerHandle("infer:morphosyntax", "eng") → morphosyntax model host
//!   ├── WorkerHandle("infer:fa", "eng")           → forced-alignment model host
//!   └── WorkerHandle("infer:asr", "eng")          → ASR model host
//! ```
//!
//! Workers are spawned lazily on first request, health-checked periodically,
//! restarted on failure, and idle-timed out after inactivity.

pub mod artifacts_v2;
pub mod asr_request_v2;
pub mod asr_result_v2;
pub mod avqi_request_v2;
pub mod error;
pub mod fa_result_v2;
pub mod handle;
pub mod memory_guard;
pub mod opensmile_request_v2;
pub mod pool;
pub(crate) mod provider_credentials;
pub mod python;
pub mod registry;
pub mod request_builder_v2;
pub mod speaker_request_v2;
pub mod speaker_result_v2;
pub(crate) mod target;
pub mod tcp_handle;
pub mod text_request_v2;
pub mod text_result_v2;

// Re-export wire-format types from types::worker so that
// `crate::worker::InferTask` etc. continues to resolve.
pub use crate::types::worker::*;
pub use target::{WorkerBootstrapMode, WorkerProfile, WorkerTarget};
