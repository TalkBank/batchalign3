//! `batchalign3 jobs` -- inspect remote jobs or local debug artifacts by job ID.
//!
//! This module implements the `jobs` subcommand, which lets users inspect
//! processing jobs without opening the dashboard. It operates in three modes:
//!
//! - **Single-server list** -- When `--server` is given, lists all jobs on
//!   that server with their status, command, and file progress counts.
//!
//! - **Single-job detail** -- When a job ID is provided as a positional argument,
//!   fetches and displays the full job record including per-file statuses and
//!   error messages.
//!
//! - **Local debug inspection** -- When `--server` is omitted but a job ID is
//!   provided, inspects the local runtime state under `~/.batchalign3/jobs/`
//!   (or `BATCHALIGN_STATE_DIR`) and reports stable artifact handles for later
//!   human or agent inspection.

use std::fs;
use std::path::{Path, PathBuf};

use batchalign_app::config::RuntimeLayout;
use batchalign_app::debug_artifacts::JobDebugArtifacts;
use serde::Serialize;

use crate::args::JobsArgs;
use crate::client::BatchalignClient;
use crate::error::CliError;

const DEBUG_ARTIFACTS_FILENAME: &str = "debug-artifacts.json";
const DEBUG_TRACES_FILENAME: &str = "debug-traces.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct LocalJobInspection {
    job_id: String,
    staging_dir: PathBuf,
    debug_summary_file: Option<PathBuf>,
    trace_file: Option<PathBuf>,
    bug_report_ids: Vec<String>,
    bug_report_files: Vec<PathBuf>,
    persisted_summary: bool,
}

/// Execute the `jobs` command.
pub async fn run(args: &JobsArgs) -> Result<(), CliError> {
    if let Some(ref server) = args.server {
        let client = BatchalignClient::new();
        let server = server.trim_end_matches('/');

        if let Some(ref id) = args.job_id {
            show_job(&client, server, id, args.json).await
        } else {
            list_jobs(&client, server, args.json).await
        }
    } else if let Some(ref job_id) = args.job_id {
        let layout = RuntimeLayout::from_env();
        let inspection = inspect_local_job(&layout, job_id)?;
        print_local_job(&inspection, args.json)
    } else {
        Err(CliError::InvalidArgument(
            "JOB_ID required when --server is omitted".into(),
        ))
    }
}

async fn list_jobs(client: &BatchalignClient, server: &str, json: bool) -> Result<(), CliError> {
    let jobs = client.list_jobs(server).await?;

    if json {
        println!("{}", serde_json::to_string_pretty(&jobs)?);
        return Ok(());
    }

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

async fn show_job(
    client: &BatchalignClient,
    server: &str,
    job_id: &str,
    json: bool,
) -> Result<(), CliError> {
    let info = client.get_job(server, job_id).await?;

    if json {
        println!("{}", serde_json::to_string_pretty(&info)?);
        return Ok(());
    }

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

fn inspect_local_job(layout: &RuntimeLayout, job_id: &str) -> Result<LocalJobInspection, CliError> {
    let staging_dir = layout.jobs_dir().join(job_id);
    if !staging_dir.is_dir() {
        return Err(CliError::InvalidArgument(format!(
            "local job not found: {job_id} (expected {})",
            staging_dir.display()
        )));
    }

    let debug_summary_file = staging_dir.join(DEBUG_ARTIFACTS_FILENAME);
    if debug_summary_file.is_file() {
        let artifacts = load_debug_artifacts(&debug_summary_file)?;
        return Ok(LocalJobInspection {
            job_id: artifacts.job_id.to_string(),
            staging_dir: artifacts.staging_dir,
            debug_summary_file: Some(debug_summary_file),
            trace_file: artifacts.trace_file,
            bug_report_ids: artifacts.bug_report_ids,
            bug_report_files: artifacts.bug_report_files,
            persisted_summary: true,
        });
    }

    let trace_file = candidate_file(&staging_dir, DEBUG_TRACES_FILENAME);
    Ok(LocalJobInspection {
        job_id: job_id.to_string(),
        staging_dir,
        debug_summary_file: None,
        trace_file,
        bug_report_ids: Vec::new(),
        bug_report_files: Vec::new(),
        persisted_summary: false,
    })
}

fn load_debug_artifacts(path: &Path) -> Result<JobDebugArtifacts, CliError> {
    let content = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&content)?)
}

fn candidate_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let path = dir.join(name);
    path.is_file().then_some(path)
}

fn print_local_job(inspection: &LocalJobInspection, json: bool) -> Result<(), CliError> {
    if json {
        println!("{}", serde_json::to_string_pretty(inspection)?);
        return Ok(());
    }

    eprintln!();
    eprintln!("Local job {}", inspection.job_id);
    eprintln!("{}", "-".repeat(40));
    eprintln!("Artifacts: {}", inspection.staging_dir.display());
    eprintln!(
        "Summary:   {}",
        if inspection.persisted_summary {
            "persisted debug-artifacts.json"
        } else {
            "staging-dir fallback"
        }
    );
    if let Some(ref summary) = inspection.debug_summary_file {
        eprintln!("Debug:     {}", summary.display());
    }
    if let Some(ref trace_file) = inspection.trace_file {
        eprintln!("Traces:    {}", trace_file.display());
    }
    for bug_report_id in &inspection.bug_report_ids {
        eprintln!("Bug ID:    {bug_report_id}");
    }
    for bug_report_file in &inspection.bug_report_files {
        eprintln!("Bug file:  {}", bug_report_file.display());
    }
    eprintln!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use batchalign_app::api::JobId;

    #[test]
    fn inspect_local_job_prefers_persisted_summary() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let layout = RuntimeLayout::from_state_dir(tempdir.path().join("state"));
        let job_id = "job-local-summary";
        let staging_dir = layout.jobs_dir().join(job_id);
        let bug_reports_dir = layout.bug_reports_dir();
        fs::create_dir_all(&staging_dir).expect("create staging dir");
        fs::create_dir_all(&bug_reports_dir).expect("create bug reports dir");

        let trace_file = staging_dir.join(DEBUG_TRACES_FILENAME);
        let bug_report_file = bug_reports_dir.join("bug-123.json");
        let artifacts = JobDebugArtifacts {
            job_id: JobId::from(job_id),
            staging_dir: staging_dir.clone(),
            trace_file: Some(trace_file.clone()),
            bug_report_ids: vec!["bug-123".into()],
            bug_report_files: vec![bug_report_file.clone()],
        };
        fs::write(
            staging_dir.join(DEBUG_ARTIFACTS_FILENAME),
            serde_json::to_vec_pretty(&artifacts).expect("serialize artifacts"),
        )
        .expect("write debug summary");

        let inspection = inspect_local_job(&layout, job_id).expect("inspect local job");
        assert!(inspection.persisted_summary);
        assert_eq!(inspection.job_id, job_id);
        assert_eq!(inspection.staging_dir, staging_dir);
        assert_eq!(
            inspection.debug_summary_file,
            Some(
                layout
                    .jobs_dir()
                    .join(job_id)
                    .join(DEBUG_ARTIFACTS_FILENAME)
            )
        );
        assert_eq!(inspection.trace_file, Some(trace_file));
        assert_eq!(inspection.bug_report_ids, vec!["bug-123"]);
        assert_eq!(inspection.bug_report_files, vec![bug_report_file]);
    }

    #[test]
    fn inspect_local_job_falls_back_to_staging_dir_when_summary_is_missing() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let layout = RuntimeLayout::from_state_dir(tempdir.path().join("state"));
        let job_id = "job-local-fallback";
        let staging_dir = layout.jobs_dir().join(job_id);
        fs::create_dir_all(&staging_dir).expect("create staging dir");
        let trace_file = staging_dir.join(DEBUG_TRACES_FILENAME);
        fs::write(&trace_file, "{}").expect("write trace file");

        let inspection = inspect_local_job(&layout, job_id).expect("inspect local job");
        assert!(!inspection.persisted_summary);
        assert_eq!(inspection.job_id, job_id);
        assert_eq!(inspection.staging_dir, staging_dir);
        assert_eq!(inspection.debug_summary_file, None);
        assert_eq!(inspection.trace_file, Some(trace_file));
        assert!(inspection.bug_report_ids.is_empty());
        assert!(inspection.bug_report_files.is_empty());
    }

    #[test]
    fn local_job_json_includes_summary_mode_and_bug_reports() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let layout = RuntimeLayout::from_state_dir(tempdir.path().join("state"));
        let job_id = "job-local-json";
        let staging_dir = layout.jobs_dir().join(job_id);
        let bug_reports_dir = layout.bug_reports_dir();
        fs::create_dir_all(&staging_dir).expect("create staging dir");
        fs::create_dir_all(&bug_reports_dir).expect("create bug reports dir");

        let inspection = LocalJobInspection {
            job_id: job_id.into(),
            staging_dir: staging_dir.clone(),
            debug_summary_file: Some(staging_dir.join(DEBUG_ARTIFACTS_FILENAME)),
            trace_file: Some(staging_dir.join(DEBUG_TRACES_FILENAME)),
            bug_report_ids: vec!["bug-123".into()],
            bug_report_files: vec![bug_reports_dir.join("bug-123.json")],
            persisted_summary: true,
        };

        let value = serde_json::to_value(&inspection).expect("serialize inspection");
        assert_eq!(value["job_id"], job_id);
        assert_eq!(value["persisted_summary"], true);
        assert_eq!(value["bug_report_ids"][0], "bug-123");
    }
}
