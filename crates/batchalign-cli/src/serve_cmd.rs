//! `batchalign3 serve` -- manage the batchalign processing server.
//!
//! This module implements the three `serve` subcommands:
//!
//! - **`serve start`** -- Launch the HTTP server that accepts processing jobs.
//!   In foreground mode (`--foreground`) the server runs in the current process,
//!   blocking until shutdown. In background mode (the default) a detached child
//!   process is spawned in a new session (`setsid`) so it survives CLI exit, and
//!   a PID file is written for later cleanup. CLI flags (port, host, warmup,
//!   Python path, test-echo) override values from `server.yaml`.
//!
//! - **`serve stop`** -- Shut down any running server and local daemon. Reads the
//!   PID file, sends `SIGTERM` to the process group, and cleans up state files.
//!
//! - **`serve status`** -- Probe a running server's `/health` endpoint and print
//!   version, worker count, active jobs, and media root configuration. Discovers
//!   the server URL from `--server`, a local daemon info file, or falls back to
//!   the configured local server URL.

use batchalign_app::config::{self, RuntimeLayout, WARMUP_PRESET_FULL, WARMUP_PRESET_MINIMAL};
use batchalign_app::worker::handle::WorkerRuntimeConfig;
use batchalign_app::worker::pool::PoolConfig;

use crate::args::{ServeStartArgs, ServeStatusArgs};
use crate::client::BatchalignClient;
use crate::daemon;
use crate::error::CliError;
use crate::python::resolve_python_executable;
use crate::self_exe::resolve_self_exe;

/// `serve start` — start the processing server.
pub async fn start(
    args: &ServeStartArgs,
    verbose: u8,
    force_cpu: bool,
    engine_overrides: Option<&str>,
) -> Result<(), CliError> {
    let layout = RuntimeLayout::from_env();
    let mut cfg =
        config::load_config_from_layout(&layout, args.config.as_deref().map(std::path::Path::new))?;
    let worker_python = args
        .python
        .clone()
        .unwrap_or_else(resolve_python_executable);

    // Override config values only when explicitly passed via CLI.
    if let Some(port) = args.port {
        cfg.port = port;
    }
    if let Some(ref host) = args.host {
        cfg.host = host.clone();
    }

    if let Some(workers) = args.workers {
        cfg.max_workers_per_job = workers as i32;
    }
    if let Some(timeout) = args.timeout {
        cfg.audio_task_timeout_s = timeout;
    }

    if let Some(ref warmup) = args.warmup {
        apply_warmup_flag(warmup, &mut cfg);
    }

    let warnings = cfg.validate();
    for w in &warnings {
        eprintln!("warning: {w}");
    }

    if cfg.media_roots.is_empty() && cfg.media_mappings.is_empty() {
        eprintln!(
            "warning: no media_roots or media_mappings configured. \
             Align/transcribe commands will fail unless CHAT files reference \
             accessible media paths."
        );
    }

    if args.foreground {
        eprintln!("\nStarting server on {}:{}...\n", cfg.host, cfg.port);

        let idle_timeout_s = args.worker_idle_timeout_s.unwrap_or_else(|| {
            if cfg.worker_idle_timeout_s > 0 {
                cfg.worker_idle_timeout_s
            } else {
                PoolConfig::default().idle_timeout_s
            }
        });
        let worker_runtime = WorkerRuntimeConfig {
            force_cpu,
            ..WorkerRuntimeConfig::default()
        };
        let pool_config = PoolConfig {
            python_path: worker_python.clone(),
            test_echo: args.test_echo,
            idle_timeout_s,
            health_check_interval_s: if cfg.worker_health_interval_s > 0 {
                cfg.worker_health_interval_s
            } else {
                PoolConfig::default().health_check_interval_s
            },
            verbose,
            engine_overrides: engine_overrides.unwrap_or("").to_string(),
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
                0 // auto-compute from available RAM
            },
            audio_task_timeout_s: cfg.audio_task_timeout_s,
            analysis_task_timeout_s: cfg.analysis_task_timeout_s,
            worker_registry_path: cfg.worker_registry_path.clone(),
        };
        batchalign_app::serve_with_runtime(
            cfg,
            pool_config,
            layout,
            Some(crate::build_hash().to_string()),
        )
        .await?;
    } else {
        // Background mode: spawn self with --foreground
        let exe = resolve_self_exe();

        std::fs::create_dir_all(layout.state_dir())?;

        // Stop any existing server
        let _ = stop_server(&layout);

        let log_path = layout.server_log_path();
        let log_file = std::fs::File::create(&log_path)?;

        let mut cmd = std::process::Command::new(&exe);
        cmd.args([
            "serve",
            "start",
            "--foreground",
            "--port",
            &cfg.port.to_string(),
            "--host",
            &cfg.host,
        ]);
        if let Some(ref config_path) = args.config {
            cmd.args(["--config", config_path]);
        }
        cmd.args(["--python", &worker_python]);
        // Forward warmup configuration to the background server process.
        if let Some(ref warmup) = args.warmup {
            cmd.args(["--warmup", warmup]);
        }
        if args.test_echo {
            cmd.arg("--test-echo");
        }
        if force_cpu {
            cmd.arg("--force-cpu");
        }
        // Forward verbosity to the background server process.
        for _ in 0..verbose {
            cmd.arg("-v");
        }
        // Forward engine overrides to the background server process.
        if let Some(overrides) = engine_overrides {
            cmd.args(["--engine-overrides", overrides]);
        }
        // Forward workers to the background server process.
        if let Some(workers) = args.workers {
            cmd.args(["--workers", &workers.to_string()]);
        }
        // Forward timeout to the background server process.
        if let Some(timeout) = args.timeout {
            cmd.args(["--timeout", &timeout.to_string()]);
        }

        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(log_file);

        // Start new session so it survives CLI exit
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            unsafe {
                cmd.pre_exec(|| {
                    libc::setsid();
                    Ok(())
                });
            }
        }

        let proc = cmd.spawn()?;
        let pid = proc.id();

        // Write PID file
        let pid_path = layout.server_pid_path();
        std::fs::write(&pid_path, pid.to_string())?;

        // Brief health check
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        eprintln!("\nServer started (PID {pid})");
        eprintln!("Listening on http://{}:{}", cfg.host, cfg.port);
        eprintln!("\nPID file: {}", pid_path.display());
        eprintln!("Log file: {}", log_path.display());
        eprintln!(
            "\nClients can now use: batchalign3 <command> ... --server http://<this-machine>:{}",
            cfg.port
        );
    }

    Ok(())
}

/// `serve stop` — stop the server and daemon.
pub async fn stop() -> Result<(), CliError> {
    let layout = RuntimeLayout::from_env();

    // Stop daemon first
    if daemon::stop_daemon().await? {
        eprintln!("Local daemon stopped.");
    }
    if daemon::stop_sidecar_daemon().await? {
        eprintln!("Sidecar daemon stopped.");
    }

    let stopped = stop_server(&layout);
    if stopped {
        eprintln!("Server stopped.");
    } else {
        eprintln!("No server process found.");
    }

    Ok(())
}

/// `serve status` — check server health.
pub async fn status(args: &ServeStatusArgs) -> Result<(), CliError> {
    let client = BatchalignClient::new();
    let layout = RuntimeLayout::from_env();
    let configured_port = config::load_config_from_layout(&layout, None)
        .unwrap_or_default()
        .port;

    let server = if let Some(ref s) = args.server {
        s.trim_end_matches('/').to_string()
    } else {
        // Try local daemon first
        if let Some(info) = daemon::read_daemon_info() {
            if client
                .health_check(&format!("http://127.0.0.1:{}", info.port))
                .await
                .is_ok()
            {
                eprintln!("Using local daemon (PID {})", info.pid);
                format!("http://127.0.0.1:{}", info.port)
            } else {
                format!("http://localhost:{configured_port}")
            }
        } else {
            format!("http://localhost:{configured_port}")
        }
    };

    match client.health_check(&server).await {
        Ok(health) => {
            eprintln!();
            eprintln!("Batchalign Server Status");
            eprintln!("{}", "-".repeat(40));
            eprintln!("URL:              {server}");
            eprintln!("Status:           {}", health.status);
            eprintln!("Version:          {}", health.version);
            if !health.build_hash.is_empty() {
                eprintln!("Build:            {}", health.build_hash);
            }
            eprintln!("Workers free:     {}", health.workers_available);
            eprintln!("Active jobs:      {}", health.active_jobs);
            if !health.media_roots.is_empty() {
                eprintln!("Media:            {}", health.media_roots.join(", "));
            }
            eprintln!();
        }
        Err(e) => {
            eprintln!("error: cannot reach server at {server}: {e}");
        }
    }

    Ok(())
}

fn stop_server(layout: &RuntimeLayout) -> bool {
    let pid_path = layout.server_pid_path();
    let pid_str = match std::fs::read_to_string(&pid_path) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let pid: u32 = match pid_str.trim().parse() {
        Ok(p) => p,
        Err(_) => return false,
    };

    let killed = kill_pid(pid);
    let _ = std::fs::remove_file(&pid_path);
    killed
}

/// Parse the `--warmup` CLI flag value and apply it to the server config.
///
/// Accepts preset names (`off`, `minimal`, `full`) or a comma-separated list
/// of command names (e.g. `align,morphotag`).  The resolved list is written
/// to `warmup_commands`.
fn apply_warmup_flag(value: &str, cfg: &mut batchalign_app::config::ServerConfig) {
    cfg.warmup_commands = match value.to_ascii_lowercase().as_str() {
        "off" => Vec::new(),
        "minimal" => WARMUP_PRESET_MINIMAL
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
        "full" => WARMUP_PRESET_FULL
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
        _ => value
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
    };
}

fn kill_pid(pid: u32) -> bool {
    #[cfg(unix)]
    {
        unsafe {
            if libc::killpg(pid as i32, libc::SIGTERM) == 0 {
                return true;
            }
            if libc::kill(pid as i32, libc::SIGTERM) == 0 {
                return true;
            }
        }
        false
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}
