//! Dispatch router — routes processing commands to direct or server hosts.
//!
//! Mirrors `dispatch.py` + `dispatch_server.py`.
//!
//! Explicit `--server` runs against an HTTP server. Without `--server`, local
//! processing commands execute inline through the shared direct host.

mod helpers;
mod paths;
mod single;

use std::time::Duration;

use batchalign_app::config::{RuntimeLayout, load_validated_config_from_layout};
use batchalign_app::host_memory::HostMemoryRuntimeConfig;
use batchalign_app::host_policy::HostExecutionPolicy;
use batchalign_app::options::CommandOptions;
use batchalign_app::worker::handle::WorkerRuntimeConfig;
use batchalign_app::worker::pool::PoolConfig;
use batchalign_app::{DirectHost, ReleasedCommand, prepare_direct_workers};
use batchalign_app::{api::JobInfo, api::JobStatus, config::ServerConfig};

use crate::client::{self, BatchalignClient, server_label};
use crate::error::CliError;
use crate::progress::BatchProgress;
use crate::python::resolve_python_executable;

use helpers::{DirectProgressTracker, file_error_details, finish_terminal_job};
use paths::prepare_paths_submission;
use single::dispatch_single_server;

// ---------------------------------------------------------------------------
// Top-level dispatch router
// ---------------------------------------------------------------------------

/// Named dispatch request for one CLI processing invocation.
#[derive(Debug)]
pub struct DispatchRequest<'a> {
    /// Canonical processing command name.
    pub command: ReleasedCommand,
    /// Primary language for the command.
    pub lang: &'a str,
    /// Requested number of speakers.
    pub num_speakers: u32,
    /// File extensions to discover.
    pub extensions: &'static [&'static str],
    /// Explicit remote server URL, if any.
    pub server_arg: Option<&'a str>,
    /// Input paths supplied on the CLI.
    pub inputs: &'a [std::path::PathBuf],
    /// Optional output directory.
    pub out_dir: Option<&'a std::path::Path>,
    /// Typed command options for submission.
    pub options: Option<CommandOptions>,
    /// Optional TalkBank bank name.
    pub bank: Option<&'a str>,
    /// Optional bank subdirectory.
    pub subdir: Option<&'a str>,
    /// Optional lexicon path.
    pub lexicon: Option<&'a str>,
    /// Whether to use the TUI.
    pub use_tui: bool,
    /// Whether to auto-open the dashboard.
    pub open_dashboard: bool,
    /// Whether to force CPU execution for local worker processes.
    pub force_cpu: bool,
    /// Skip auto-detection of a local server (force direct mode).
    pub no_server: bool,
    /// Optional before-path input for incremental workflows.
    pub before: Option<&'a std::path::Path>,
    /// Optional explicit worker count.
    pub workers: Option<usize>,
    /// Optional daemon startup timeout.
    pub timeout: Option<u64>,
}

/// Route a processing command to the appropriate execution host.
///
/// This is the main entry point for all CLI processing commands. It resolves
/// where to send work using the following priority chain:
///
/// 1. **Explicit `--server URL`** -- single-server dispatch
///    via HTTP. Text commands submit content and download results. Audio
///    commands submit shared-filesystem paths (`paths_mode`), so the execution
///    host must be able to read the same input paths and write the requested
///    output paths.
/// 2. **Direct local execution** -- local filesystem processing goes through
///    the shared direct host with no daemon/queue layer.
///
/// # Parameters
///
/// Takes one [`DispatchRequest`] describing the command profile, input/output
/// paths, typed options, and UI/runtime toggles for this CLI invocation.
///
/// # Errors
///
/// Returns [`CliError`] on I/O failures, HTTP errors, job failures, or direct
/// execution failures.
pub async fn dispatch(request: DispatchRequest<'_>) -> Result<(), CliError> {
    let DispatchRequest {
        command,
        lang,
        num_speakers,
        extensions,
        server_arg,
        inputs,
        out_dir,
        options,
        bank,
        subdir,
        lexicon,
        use_tui,
        open_dashboard,
        force_cpu,
        no_server,
        before,
        workers,
        timeout,
    } = request;
    let layout = RuntimeLayout::from_env();

    if bank.is_some() || subdir.is_some() {
        eprintln!(
            "error: --bank/--subdir remote media selection is no longer supported.\n\
             Pass filesystem paths that are visible on the execution host instead."
        );
        return Ok(());
    }

    // 1. Explicit --server
    if let Some(server) = server_arg {
        let client = BatchalignClient::new();
        let urls = client::parse_servers(server);
        if urls.is_empty() {
            eprintln!("error: no server URL provided");
            return Ok(());
        }

        if urls.len() == 1 {
            return dispatch_single_server(
                &client,
                &urls[0],
                command,
                lang,
                num_speakers,
                extensions,
                inputs,
                out_dir,
                options.as_ref(),
                lexicon,
                before,
                use_tui,
                open_dashboard,
            )
            .await;
        }

        eprintln!(
            "error: multi-server dispatch (--server URL1,URL2) is not available in this version.\n\
             Use --server with a single URL instead."
        );
        return Ok(());
    }

    // 2. Auto-detect local server
    //
    // If a batchalign3 server is running locally (e.g., as a launchd service
    // connected to the Temporal fleet), route work through it automatically.
    // This gives the user fleet benefits (distributed processing, crash
    // recovery, warm models) without requiring `--server`.
    let (cfg, warnings) = load_validated_config_from_layout(&layout, None)?;
    for warning in warnings {
        eprintln!("warning: {warning}");
    }

    let local_url = format!("http://127.0.0.1:{}", cfg.port);
    if !no_server && let Some(health) = probe_local_server(&local_url).await {
        eprintln!("Using local server at {} ({})\n", local_url, health,);
        let client = BatchalignClient::new();
        return dispatch_single_server(
            &client,
            &local_url,
            command,
            lang,
            num_speakers,
            extensions,
            inputs,
            out_dir,
            options.as_ref(),
            lexicon,
            before,
            use_tui,
            open_dashboard,
        )
        .await;
    }

    // 3. Direct local execution (no server available)
    dispatch_direct_mode(
        cfg,
        layout,
        command,
        lang,
        num_speakers,
        extensions,
        inputs,
        out_dir,
        options.as_ref(),
        lexicon,
        before,
        force_cpu,
        workers,
        timeout,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn dispatch_direct_mode(
    mut cfg: ServerConfig,
    layout: RuntimeLayout,
    command: ReleasedCommand,
    lang: &str,
    num_speakers: u32,
    extensions: &[&str],
    inputs: &[std::path::PathBuf],
    out_dir: Option<&std::path::Path>,
    options: Option<&CommandOptions>,
    lexicon: Option<&str>,
    before: Option<&std::path::Path>,
    force_cpu: bool,
    workers: Option<usize>,
    timeout: Option<u64>,
) -> Result<(), CliError> {
    if let Some(workers) = workers {
        cfg.max_workers_per_job = workers as i32;
    }
    if let Some(timeout) = timeout {
        cfg.audio_task_timeout_s = timeout;
    }

    let mapping_keys: Vec<String> = cfg
        .media_mappings
        .keys()
        .map(|k| k.as_str().to_owned())
        .collect();
    let Some(prepared) = prepare_paths_submission(
        command,
        lang,
        num_speakers,
        extensions,
        inputs,
        out_dir,
        options,
        lexicon,
        before,
        &mapping_keys,
    )?
    else {
        eprintln!("warning: no files found with extensions {extensions:?}");
        return Ok(());
    };

    eprintln!("Found {} file(s) to process.\n", prepared.total_files);
    eprintln!("Running locally (direct mode)...\n");

    let direct_workers = prepare_direct_workers(&cfg, build_direct_pool_config(&cfg, force_cpu))
        .await
        .map_err(CliError::from)?;
    let host = DirectHost::new(cfg, layout, None, None, &direct_workers)
        .await
        .map_err(CliError::from)?;
    let job_id = host
        .submit_submission(prepared.submission)
        .await
        .map_err(CliError::from)?;
    let submitted_debug = host
        .job_debug_artifacts(&job_id)
        .await
        .map_err(CliError::from)?;
    eprintln!("Direct job prepared.\n");
    helpers::print_job_debug_artifacts(&submitted_debug);
    eprintln!();

    let progress = BatchProgress::new(prepared.total_files as u64, command.as_wire_name());
    let mut tracker = DirectProgressTracker::default();
    if let Ok(initial) = host.job_info(&job_id).await {
        tracker.observe(&progress, &initial);
    }

    let runner_host = host.clone();
    let run_job_id = job_id.clone();
    let run_fut = async move {
        runner_host
            .run_job(&run_job_id)
            .await
            .map_err(CliError::from)
    };
    tokio::pin!(run_fut);
    let mut poll_interval = tokio::time::interval(Duration::from_millis(120));
    poll_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    let final_info: JobInfo = loop {
        tokio::select! {
            result = &mut run_fut => {
                if let Err(error) = result {
                    progress.finish();
                    return Err(error);
                }
                let info = match host.job_info(&job_id).await {
                    Ok(info) => info,
                    Err(error) => {
                        progress.finish();
                        return Err(CliError::from(error));
                    }
                };
                tracker.observe(&progress, &info);
                break info;
            }
            _ = poll_interval.tick() => {
                if let Ok(info) = host.job_info(&job_id).await {
                    tracker.observe(&progress, &info);
                }
            }
        }
    };
    progress.finish();

    let error_details = file_error_details(&final_info);
    let clean_success = final_info.status == JobStatus::Completed
        && error_details.is_empty()
        && final_info
            .error
            .as_ref()
            .is_none_or(|s| s.trim().is_empty());
    if !clean_success {
        match host.job_debug_artifacts(&job_id).await {
            Ok(artifacts) => {
                eprintln!();
                helpers::print_job_debug_artifacts(&artifacts);
                eprintln!();
            }
            Err(error) => eprintln!("warning: failed to collect direct debug artifacts: {error}"),
        }
    }
    finish_terminal_job(
        &final_info,
        &error_details,
        prepared.total_files as u64,
        &prepared.effective_out,
    )
}

fn build_direct_pool_config(cfg: &ServerConfig, force_cpu: bool) -> PoolConfig {
    let tier = cfg.resolved_memory_tier();
    let host_policy = HostExecutionPolicy::from_server_config(cfg);
    let idle_timeout_s = cfg.resolved_worker_idle_timeout_s();
    let worker_runtime = WorkerRuntimeConfig {
        force_cpu,
        gpu_thread_pool_size: cfg.gpu_thread_pool_size,
        host_memory: HostMemoryRuntimeConfig::from_server_config(cfg),
        memory_tier: tier,
        bootstrap_mode: host_policy.bootstrap_mode,
        ..WorkerRuntimeConfig::default()
    };
    PoolConfig {
        python_path: resolve_python_executable(),
        idle_timeout_s,
        health_check_interval_s: if cfg.worker_health_interval_s > 0 {
            cfg.worker_health_interval_s
        } else {
            PoolConfig::default().health_check_interval_s
        },
        verbose: 0,
        engine_overrides: String::new(),
        runtime: worker_runtime,
        max_workers_per_key: if cfg.max_workers_per_key > 0 {
            cfg.max_workers_per_key as usize
        } else {
            PoolConfig::default().max_workers_per_key
        },
        ready_timeout_s: if cfg.worker_ready_timeout_s > 0 {
            cfg.worker_ready_timeout_s
        } else {
            PoolConfig::default().ready_timeout_s
        },
        max_total_workers: if cfg.max_total_workers > 0 {
            cfg.max_total_workers as usize
        } else {
            0
        },
        audio_task_timeout_s: cfg.audio_task_timeout_s,
        analysis_task_timeout_s: cfg.analysis_task_timeout_s,
        worker_registry_path: cfg.worker_registry_path.clone(),
        ..PoolConfig::default()
    }
}

fn server_supports_command(capabilities: &[String], command: ReleasedCommand) -> bool {
    capabilities.is_empty()
        || capabilities
            .iter()
            .any(|c| c == command.as_str() || c == "test-echo")
}

/// Warn (but don't block) if the server's build hash differs from the CLI's.
///
/// This warning only applies to explicit `--server` connections.
fn warn_stale_server(server_url: &str, health: &batchalign_app::api::HealthResponse) {
    if !health.build_hash.is_empty() && health.build_hash != crate::build_hash() {
        eprintln!(
            "warning: server {} has a different build ({}) than this CLI ({}).\n\
             Results may differ from what the current binary expects.\n\
             Restart the server to pick up the new binary.",
            server_label(server_url),
            health.build_hash,
            crate::build_hash(),
        );
    }
}

/// Probe the local server with a short timeout.
///
/// Returns a human-readable status string if the server is healthy,
/// `None` if unreachable or unhealthy. Uses a 500ms timeout so direct
/// mode startup is not noticeably delayed when no server is running.
async fn probe_local_server(url: &str) -> Option<String> {
    let health_url = format!("{url}/health");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(500))
        .build()
        .ok()?;
    let resp = client.get(&health_url).send().await.ok()?;
    let health: batchalign_app::api::HealthResponse = resp.json().await.ok()?;
    if health.status != batchalign_app::api::HealthStatus::Ok {
        return None;
    }
    let label = if health.active_jobs > 0 {
        format!(
            "{} workers, {} active job(s)",
            health.workers_available, health.active_jobs
        )
    } else {
        format!("{} workers available", health.workers_available)
    };
    Some(label)
}

#[cfg(test)]
mod tests {
    use batchalign_app::{ReleasedCommand, released_command_uses_local_audio};

    #[test]
    fn benchmark_and_align_are_treated_as_local_audio_commands() {
        assert!(released_command_uses_local_audio(
            ReleasedCommand::Benchmark
        ));
        assert!(released_command_uses_local_audio(
            ReleasedCommand::Transcribe
        ));
        assert!(released_command_uses_local_audio(ReleasedCommand::Align));
        assert!(!released_command_uses_local_audio(
            ReleasedCommand::Morphotag
        ));
    }
}
