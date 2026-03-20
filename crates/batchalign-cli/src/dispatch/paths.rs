//! Paths-mode dispatch — local daemon reads/writes files directly.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use batchalign_app::api::{FileStatusKind, JobSubmission};
use batchalign_app::options::CommandOptions;

use crate::client::{self, BatchalignClient, MAX_POLL_FAILURES, POLL_MAX, POLL_MIN, POLL_STEP};
use crate::discover::{build_server_names, copy_nonmatching, infer_base_dir};
use crate::error::CliError;
use crate::progress::{BatchProgress, ProgressSink};
use crate::tui::TuiProgress;

use super::helpers::{
    filter_files_for_command, finish_terminal_job, inject_lexicon, maybe_open_dashboard,
};

/// Dispatch via local daemon using paths mode.
///
/// Sends only filesystem paths (not content) to the daemon.
/// The daemon reads/writes files directly via the shared filesystem.
#[allow(clippy::too_many_arguments)]
pub(super) async fn dispatch_paths_mode(
    client: &BatchalignClient,
    server_url: &str,
    command: &str,
    lang: &str,
    num_speakers: u32,
    extensions: &[&str],
    inputs: &[String],
    out_dir: Option<&str>,
    options: Option<&CommandOptions>,
    bank: Option<&str>,
    subdir: Option<&str>,
    lexicon: Option<&str>,
    use_tui: bool,
    open_dashboard: bool,
    before: Option<&str>,
) -> Result<(), CliError> {
    // Health check
    client.health_check(server_url).await.map_err(|e| {
        eprintln!("error: cannot reach local daemon at {server_url}: {e}");
        e
    })?;

    // Discover files
    let (files, outputs) = crate::discover::discover_server_inputs(inputs, out_dir, extensions);
    let (files, outputs) = filter_files_for_command(command, files, outputs);

    if let Some(od) = out_dir {
        for inp in inputs {
            if Path::new(inp).is_dir() {
                copy_nonmatching(Path::new(inp), Path::new(od), extensions, command);
            }
        }
    }

    if files.is_empty() {
        eprintln!("warning: no files found with extensions {extensions:?}");
        return Ok(());
    }

    eprintln!("Found {} file(s) to process.\n", files.len());

    let (server_names, _) = build_server_names(&files, &outputs, inputs);

    let source_paths: Vec<String> = files
        .iter()
        .filter_map(|f| std::fs::canonicalize(f).ok())
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    let output_paths: Vec<String> = outputs
        .iter()
        .filter_map(|f| {
            if let Some(parent) = f.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            std::fs::canonicalize(f.parent()?)
                .ok()
                .map(|p| p.join(f.file_name().unwrap_or_default()))
        })
        .map(|p| p.to_string_lossy().to_string())
        .collect();

    let base_dir = infer_base_dir(inputs);

    let (mapping_key, mapping_subdir) = if let Some(bk) = bank {
        (bk.to_string(), subdir.unwrap_or("").to_string())
    } else {
        match client.health_check(server_url).await {
            Ok(h) => client::detect_media_mapping(&base_dir, &h.media_mapping_keys),
            Err(_) => (String::new(), String::new()),
        }
    };

    let mut opts = options.cloned().unwrap_or_else(|| {
        CommandOptions::Morphotag(batchalign_app::options::MorphotagOptions {
            common: Default::default(),
            retokenize: false,
            skipmultilang: false,
            merge_abbrev: false.into(),
        })
    });
    inject_lexicon(&mut opts, lexicon)?;
    let debug_traces = opts.common().debug_dir.is_some();

    let effective_out = out_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| base_dir.clone());

    // Resolve before_paths for incremental processing
    let before_paths = if let Some(before_arg) = before {
        let before_path = Path::new(before_arg);
        if before_path.is_dir() {
            // Match each source file to its counterpart in the before directory
            files
                .iter()
                .filter_map(|src| {
                    let src_path = Path::new(src);
                    let filename = src_path.file_name()?;
                    let candidate = before_path.join(filename);
                    if candidate.exists() {
                        std::fs::canonicalize(&candidate)
                            .ok()
                            .map(|p| p.to_string_lossy().to_string())
                    } else {
                        None
                    }
                })
                .collect()
        } else if before_path.is_file() && files.len() == 1 {
            // Single file: before_arg is the before file itself
            std::fs::canonicalize(before_path)
                .ok()
                .map(|p| vec![p.to_string_lossy().to_string()])
                .unwrap_or_default()
        } else {
            eprintln!("warning: --before path is not a valid file or directory, ignoring");
            Vec::new()
        }
    } else {
        Vec::new()
    };

    let submission = JobSubmission {
        command: command.into(),
        lang: lang.into(),
        num_speakers: num_speakers.into(),
        files: vec![],
        media_files: vec![],
        media_mapping: mapping_key,
        media_subdir: mapping_subdir,
        source_dir: base_dir.to_string_lossy().to_string(),
        options: opts,
        paths_mode: true,
        source_paths,
        output_paths,
        display_names: server_names,
        debug_traces,
        before_paths,
    };

    eprintln!("Submitting to local daemon at {server_url}...");
    let info = client.submit_job(server_url, &submission).await?;
    let job_id = &info.job_id;
    let total_files = info.total_files;
    eprintln!("Job {job_id} submitted ({total_files} file(s))");

    let dashboard_url = format!("{server_url}/dashboard/jobs/{job_id}");
    eprintln!("Dashboard: {dashboard_url}\n");

    maybe_open_dashboard(&dashboard_url, open_dashboard);

    // Poll for completion — files are written directly by daemon
    if !info.status.is_terminal() {
        if use_tui && std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            let (tui_progress, tui_runtime) = TuiProgress::new(total_files as u64, command);
            let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();

            let cc = client.clone();
            let cu = server_url.to_string();
            let cj = job_id.to_string();
            tokio::spawn(async move {
                if cancel_rx.await.is_ok() {
                    let _ = cc.cancel_job(&cu, &cj).await;
                }
            });

            let mut tui_handle = tokio::task::spawn_blocking(move || {
                crate::tui::run_tui_loop(tui_runtime, Some(cancel_tx))
            });

            let poll_result = poll_paths_mode(
                client,
                server_url,
                job_id,
                total_files as u64,
                &effective_out,
                command,
                &tui_progress,
            );

            tokio::select! {
                result = poll_result => {
                    result?;
                    let _ = tui_handle.await;
                }
                _ = &mut tui_handle => {}
            }
        } else {
            let progress = BatchProgress::new(total_files as u64, command);
            poll_paths_mode(
                client,
                server_url,
                job_id,
                total_files as u64,
                &effective_out,
                command,
                &progress,
            )
            .await?;
        }
    }

    Ok(())
}

/// Poll a paths-mode job until completion.
///
/// Files are written directly by the server — no HTTP fetch needed.
pub(super) async fn poll_paths_mode(
    client: &BatchalignClient,
    server_url: &str,
    job_id: &str,
    total_files: u64,
    out_dir: &Path,
    _command: &str,
    progress: &dyn ProgressSink,
) -> Result<(), CliError> {
    let mut error_details: Vec<(String, String)> = Vec::new();
    let mut done_count: u64 = 0;
    let mut seen_files: HashSet<String> = HashSet::new();
    let mut consecutive_failures: u32 = 0;
    let mut poll_interval = POLL_MIN;
    let mut last_completed: i64 = 0;
    let mut last_health_poll = std::time::Instant::now()
        .checked_sub(Duration::from_secs(10))
        .unwrap_or_else(std::time::Instant::now);

    loop {
        match client.get_job(server_url, job_id).await {
            Ok(info) => {
                consecutive_failures = 0;

                for entry in &info.file_statuses {
                    let fn_ = &entry.filename;
                    if seen_files.contains(&**fn_) {
                        continue;
                    }
                    if entry.status == FileStatusKind::Done {
                        seen_files.insert(fn_.to_string());
                        done_count += 1;
                        progress.log_done(fn_);
                    } else if entry.status == FileStatusKind::Error {
                        seen_files.insert(fn_.to_string());
                        let error_msg = entry
                            .error
                            .clone()
                            .unwrap_or_else(|| "unknown error".into());
                        progress.log_error(fn_, &error_msg);
                        error_details.push((fn_.to_string(), error_msg));
                    }
                }

                let done_so_far = done_count + error_details.len() as u64;
                progress.update(done_so_far, &info.file_statuses);

                if info.status.is_terminal() {
                    progress.finish();
                    return finish_terminal_job(&info, &error_details, total_files, out_dir);
                }

                let current = info.completed_files;
                if current > last_completed {
                    poll_interval = POLL_MIN;
                    last_completed = current;
                } else {
                    poll_interval = (poll_interval + POLL_STEP).min(POLL_MAX);
                }
            }
            Err(err @ CliError::JobLost { .. }) => {
                progress.finish();
                return Err(err);
            }
            Err(_) => {
                consecutive_failures += 1;
                if consecutive_failures >= MAX_POLL_FAILURES {
                    progress.finish();
                    return Err(CliError::PollExhausted {
                        attempts: MAX_POLL_FAILURES,
                    });
                }
            }
        }

        // Poll health on a slower cadence (~5s) for TUI metrics
        if last_health_poll.elapsed() >= Duration::from_secs(5) {
            if let Ok(h) = client.health_check(server_url).await {
                progress.update_health(&h);
            }
            last_health_poll = std::time::Instant::now();
        }

        tokio::time::sleep(Duration::from_secs_f64(poll_interval)).await;
    }
}
