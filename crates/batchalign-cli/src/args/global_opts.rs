use clap::{ArgAction, Args};

use super::parse_engine_overrides_json;

fn validate_engine_overrides_json(value: &str) -> Result<String, String> {
    parse_engine_overrides_json(value)?;
    Ok(value.to_string())
}

/// Global options that apply to every command.
#[derive(Args, Debug, Clone)]
pub struct GlobalOpts {
    /// Increase verbosity (-v, -vv, -vvv).
    #[arg(short, long, action = ArgAction::Count, global = true)]
    pub verbose: u8,

    /// Maximum worker processes.
    #[arg(long, global = true)]
    pub workers: Option<usize>,

    /// BA2 compatibility flag (currently a no-op in Rust CLI).
    #[arg(long, global = true, hide = true)]
    pub memlog: bool,

    /// BA2 compatibility flag (currently a no-op in Rust CLI).
    #[arg(long = "mem-guard", global = true, hide = true)]
    pub mem_guard: bool,

    /// BA2 compatibility flag (currently a no-op in Rust CLI).
    #[arg(long = "adaptive-workers", global = true, hide = true)]
    pub adaptive_workers: bool,

    /// BA2 compatibility flag (currently a no-op in Rust CLI).
    #[arg(long = "no-adaptive-workers", global = true, hide = true)]
    pub no_adaptive_workers: bool,

    /// BA2 compatibility flag (currently a no-op in Rust CLI).
    #[arg(long, global = true, hide = true)]
    pub pool: bool,

    /// BA2 compatibility flag (currently a no-op in Rust CLI).
    #[arg(long = "no-pool", global = true, hide = true)]
    pub no_pool: bool,

    /// BA2 compatibility option (currently a no-op in Rust CLI).
    #[arg(long = "adaptive-safety-factor", global = true, hide = true)]
    pub adaptive_safety_factor: Option<f64>,

    /// BA2 compatibility option (currently a no-op in Rust CLI).
    #[arg(long = "adaptive-warmup", global = true, hide = true)]
    pub adaptive_warmup: Option<usize>,

    /// Disable MPS/CUDA and force CPU-only models.
    #[arg(long, global = true)]
    pub force_cpu: bool,

    /// Compatibility no-op (explicitly disable --force-cpu).
    #[arg(long = "no-force-cpu", action = ArgAction::SetTrue, global = true)]
    pub no_force_cpu: bool,

    /// BA2 compatibility flag (currently a no-op in Rust CLI).
    #[arg(long = "shared-models", global = true, hide = true)]
    pub shared_models: bool,

    /// BA2 compatibility flag (currently a no-op in Rust CLI).
    #[arg(long = "no-shared-models", global = true, hide = true)]
    pub no_shared_models: bool,

    /// Remote server URL (or set BATCHALIGN_SERVER env var).
    #[arg(long, env = "BATCHALIGN_SERVER", global = true)]
    pub server: Option<String>,

    /// Bypass the utterance analysis cache.
    #[arg(long, global = true)]
    pub override_cache: bool,

    /// Compatibility no-op (explicitly disable --override-cache).
    #[arg(long = "use-cache", action = ArgAction::SetTrue, global = true)]
    pub use_cache: bool,

    /// Lazy audio loading for alignment/ASR.
    #[arg(long, action = ArgAction::SetTrue, default_value_t = true, global = true)]
    pub lazy_audio: bool,

    /// Disable lazy audio loading for alignment/ASR.
    #[arg(long = "no-lazy-audio", action = ArgAction::SetTrue, global = true)]
    pub no_lazy_audio: bool,

    /// Use full-screen TUI dashboard instead of progress bars (default for
    /// interactive terminals).
    #[arg(long, action = ArgAction::SetTrue, default_value_t = true, global = true)]
    pub tui: bool,

    /// Disable full-screen TUI; use simple progress bars instead.
    #[arg(long = "no-tui", action = ArgAction::SetTrue, global = true)]
    pub no_tui: bool,

    /// Auto-open the submitted job in the browser dashboard after submission.
    ///
    /// Currently only macOS launches a browser automatically; other platforms
    /// still print the dashboard URL for manual use.
    #[arg(long, action = ArgAction::SetTrue, default_value_t = true, global = true)]
    pub open_dashboard: bool,

    /// Disable browser auto-open for submitted dashboard job pages.
    #[arg(long = "no-open-dashboard", action = ArgAction::SetTrue, global = true)]
    pub no_open_dashboard: bool,

    /// Directory for pipeline debug artifacts (CHAT/JSON fixtures for
    /// offline replay). Also enables dashboard algorithm trace collection.
    /// Env fallback: BATCHALIGN_DEBUG_DIR.
    #[arg(long, env = "BATCHALIGN_DEBUG_DIR", value_name = "PATH", global = true)]
    pub debug_dir: Option<std::path::PathBuf>,

    /// Bypass cache only for specific tasks (comma-separated).
    /// Valid: morphosyntax, utr_asr, forced_alignment, utterance_segmentation, translation.
    #[arg(long, value_name = "TASKS", global = true, value_delimiter = ',')]
    pub override_cache_tasks: Vec<String>,

    /// Engine overrides as JSON (e.g. '{"asr": "tencent", "fa": "cantonese_fa"}').
    #[arg(
        long,
        value_name = "JSON",
        value_parser = validate_engine_overrides_json,
        global = true
    )]
    pub engine_overrides: Option<String>,
}
