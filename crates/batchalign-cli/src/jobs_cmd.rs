//! `batchalign3 jobs` -- query and display jobs on remote servers.
//!
//! This module implements the `jobs` subcommand, which lets users inspect
//! processing jobs without opening the dashboard. It operates in two modes:
//!
//! - **Single-server list** -- When `--server` is given, lists all jobs on
//!   that server with their status, command, and file progress counts.
//!
//! - **Single-job detail** -- When a job ID is provided as a positional argument,
//!   fetches and displays the full job record including per-file statuses and
//!   error messages.

use crate::args::JobsArgs;
use crate::client::BatchalignClient;
use crate::error::CliError;

/// Execute the `jobs` command.
pub async fn run(args: &JobsArgs) -> Result<(), CliError> {
    let client = BatchalignClient::new();

    let Some(ref server) = args.server else {
        eprintln!("error: --server URL required (or set BATCHALIGN_SERVER env var)");
        return Ok(());
    };

    let server = server.trim_end_matches('/');

    if let Some(ref id) = args.job_id {
        show_job(&client, server, id).await
    } else {
        list_jobs(&client, server).await
    }
}

async fn list_jobs(client: &BatchalignClient, server: &str) -> Result<(), CliError> {
    let jobs = client.list_jobs(server).await?;

    if jobs.is_empty() {
        eprintln!("No jobs found.");
        return Ok(());
    }

    eprintln!("\nJobs on {server}\n");
    for j in &jobs {
        let status = j.status.to_string();
        eprintln!(
            "  {}  {:<10}  {:<12}  {}/{} files",
            j.job_id, status, j.command, j.completed_files, j.total_files
        );
    }
    eprintln!();

    Ok(())
}

async fn show_job(client: &BatchalignClient, server: &str, job_id: &str) -> Result<(), CliError> {
    let info = client.get_job(server, job_id).await?;

    eprintln!();
    eprintln!("Job {}", info.job_id);
    eprintln!("{}", "-".repeat(40));
    eprintln!("Status:   {}", info.status);
    eprintln!("Command:  {}", info.command);
    eprintln!("Files:    {}/{}", info.completed_files, info.total_files);
    if let Some(ref current) = info.current_file {
        eprintln!("Current:  {current}");
    }
    if let Some(ref error) = info.error {
        eprintln!("Error:    {error}");
    }

    if !info.file_statuses.is_empty() {
        eprintln!();
        for entry in &info.file_statuses {
            let status = &entry.status;
            let error = entry
                .error
                .as_deref()
                .map(|e| format!(" — {e}"))
                .unwrap_or_default();
            eprintln!("  {:<30} {status}{error}", entry.filename);
        }
    }

    eprintln!();

    Ok(())
}
