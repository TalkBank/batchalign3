//! Thin proxy — delegates to `cargo xtask lint-ci-hygiene`.
//!
//! The actual hygiene checks (version sync, legacy terms, retired packages)
//! live in `xtask/src/ci_hygiene.rs` to avoid compiling a full integration
//! test binary just for structural lints.

use std::process::Command;

#[test]
fn ci_hygiene_passes() {
    let status = Command::new("cargo")
        .args(["xtask", "lint-ci-hygiene"])
        .status()
        .expect("failed to run cargo xtask lint-ci-hygiene");
    assert!(status.success(), "cargo xtask lint-ci-hygiene failed");
}
