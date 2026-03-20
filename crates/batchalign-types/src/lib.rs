//! Shared domain newtypes and worker IPC types for batchalign3.
//!
//! This crate contains pure data types with zero server logic — only serde,
//! utoipa, schemars, and thiserror dependencies. It exists so that downstream
//! crates (notably `batchalign-pyo3`) can use these types without pulling in
//! the full server stack from `batchalign-app`.

#[macro_use]
mod macros;

pub mod domain;
pub mod worker;
pub mod worker_v2;

// Re-export all domain types at crate root for convenience.
pub use domain::*;
