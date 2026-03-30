//! Single test binary for all ML-dependent integration tests.
//!
//! Consolidating all ML tests into one binary ensures one process = one
//! shared warmed worker backend for both direct and server-specific fixture
//! sessions. This prevents the OOM crashes caused by independent binaries each
//! spawning their own worker pools (multiple Whisper/Stanza model copies).
//!
//! Run: `cargo nextest run -p batchalign-app --test ml_golden --profile ml`
//! Update golden snapshots: `cargo insta review`

mod common;

// ML test submodules — each was previously a separate binary with its own
// worker pool. Now they share one process-global LazyLock<LiveFixtureBackend>.
mod ml_golden {
    pub mod compare_master_parity;
    pub mod direct_mode_verification;
    pub mod error_paths;
    pub mod golden;
    pub mod golden_audio;
    pub mod golden_parity;
    pub mod live_server_fixture;
    pub mod option_receipt;
    pub mod profile_verification;
}
