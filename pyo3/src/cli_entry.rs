//! Native CLI bridge exposed as a `#[pyfunction]`.
//!
//! The installed `batchalign3` console script now lives in a tiny Python
//! wrapper (`batchalign/_cli.py`) which imports this function and delegates to
//! it. That keeps packaging ownership on the Python side while the CLI
//! implementation itself remains in Rust.
//!
//! The function reads `sys.argv` and delegates to
//! [`batchalign_cli::run_embedded_cli_from_argv`] — the single source of truth
//! for the embedded CLI bootstrap path.

use pyo3::prelude::*;

/// Run the batchalign3 CLI.
///
/// Called from the console_scripts entry point. Reads `sys.argv` for
/// argument parsing and runs the full CLI dispatch loop.
#[pyfunction]
pub(crate) fn cli_main(py: Python<'_>) -> PyResult<()> {
    let sys = py.import("sys")?;
    let argv: Vec<String> = sys.getattr("argv")?.extract()?;

    // Release the GIL — the CLI is pure Rust from here on.
    let result = py.detach(move || batchalign_cli::run_embedded_cli_from_argv(argv));

    match result {
        Ok(()) => Ok(()),
        Err(code) => std::process::exit(code),
    }
}
