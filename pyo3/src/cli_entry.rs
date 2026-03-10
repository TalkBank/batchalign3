//! CLI entry point exposed as a `#[pyfunction]`.
//!
//! This allows `batchalign3` to be installed as a single Python package
//! that provides both the `batchalign_core` library AND the `batchalign3`
//! console command via `[project.scripts]` in pyproject.toml.
//!
//! The function reads `sys.argv`, builds a tokio runtime, and delegates
//! to [`batchalign_cli::run_command`] — the single source of truth for
//! CLI dispatch, shared with the standalone binary.

use clap::Parser;
use pyo3::prelude::*;

use batchalign_cli::args::Cli;

/// Run the batchalign3 CLI.
///
/// Called from the console_scripts entry point. Reads `sys.argv` for
/// argument parsing and runs the full CLI dispatch loop.
#[pyfunction]
pub(crate) fn cli_main(py: Python<'_>) -> PyResult<()> {
    let sys = py.import("sys")?;
    let argv: Vec<String> = sys.getattr("argv")?.extract()?;

    // Release the GIL — the CLI is pure Rust from here on.
    let result = py.detach(move || run_cli(argv));

    match result {
        Ok(()) => Ok(()),
        Err(code) => std::process::exit(code),
    }
}

/// Build a tokio runtime and run the CLI, returning an exit code on failure.
fn run_cli(argv: Vec<String>) -> Result<(), i32> {
    let cli = Cli::parse_from(argv);

    init_tracing(cli.global.verbose);

    let rt = tokio::runtime::Runtime::new().map_err(|e| {
        eprintln!("error: failed to create async runtime: {e}");
        6
    })?;

    rt.block_on(async {
        match batchalign_cli::run_command(cli).await {
            Ok(()) => Ok(()),
            Err(e) => {
                eprintln!("error: {e}");
                Err(e.exit_code())
            }
        }
    })
}

fn init_tracing(verbose: u8) {
    use tracing_subscriber::EnvFilter;

    let filter = match verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter));

    // Note: OTLP tracing is only available through the standalone binary.
    // The console_scripts entry point uses basic stderr logging.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init();
}
