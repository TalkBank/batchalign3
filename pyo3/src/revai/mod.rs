//! PyO3 wrappers over the shared Rust Rev.AI client.
//!
//! The actual HTTP client lives in the workspace crate
//! `batchalign-revai`. This module only translates between PyO3 function
//! signatures and the shared Rust client so the Python extension does not own
//! any Rev.AI transport logic itself.

mod pybridge;

pub(crate) use pybridge::*;
