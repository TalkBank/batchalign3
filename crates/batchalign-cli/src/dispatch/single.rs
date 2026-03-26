//! Single-server dispatch — submit files to one server, poll, write results.

use std::path::{Path, PathBuf};

use batchalign_app::ReleasedCommand;
use batchalign_app::api::JobSubmission;
use batchalign_app::options::CommandOptions;

use crate::client::{self, BatchalignClient, server_label};
use crate::discover::{build_server_names, copy_nonmatching, infer_base_dir};
use crate::error::CliError;
use crate::progress::BatchProgress;
use crate::tui::TuiProgress;

use super::helpers::{
    classify_files, filter_files_for_command, inject_lexicon, maybe_open_dashboard,
    poll_and_write_incrementally,
};
use super::{server_supports_command, warn_stale_server};

/// Submit files to a single server, poll for completion, write results.
#[allow(clippy::too_many_arguments)]
pub(super) async fn dispatch_single_server(
    client: &BatchalignClient,
    server_url: &str,
    command: ReleasedCommand,
    lang: &str,
    num_speakers: u32,
    extensions: &[&str],
    inputs: &[std::path::PathBuf],
    out_dir: Option<&std::path::Path>,
    options: Option<&CommandOptions>,
    bank: Option<&str>,
    subdir: Option<&str>,
    lexicon: Option<&str>,
    use_tui: bool,
    open_dashboard: bool,
) -> Result<(), CliError> {
    // Health check
    let health = match client.health_check(server_url).await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("error: cannot reach server at {server_url}: {e}");
            return Ok(());
        }
    };
    warn_stale_server(server_url, &health);

    // Check capabilities
    if !server_supports_command(&health.capabilities, command) {
        eprintln!(
            "error: server {} does not support '{command}'. Supported: {}",
            server_label(server_url),
            health.capabilities.join(", ")
        );
        return Ok(());
    }

    // Discover files
    let (files, outputs) = crate::discover::discover_server_inputs(inputs, out_dir, extensions)?;
    let (files, outputs) = filter_files_for_command(command, files, outputs);

    // Copy non-matching files
    if let Some(od) = out_dir {
        for inp in inputs {
            if Path::new(inp).is_dir() {
                copy_nonmatching(Path::new(inp), Path::new(od), extensions, command)?;
            }
        }
    }

    // Build server names and result map
    let base_dir = infer_base_dir(inputs)?;
    let (server_names, result_map) = build_server_names(&files, &outputs, inputs)?;

    // Classify files: CHAT → content, media → names
    let (file_payloads, media_file_names) = classify_files(&files, &server_names)?;

    // Handle --bank for remote media
    let (mapping_key, mapping_subdir, media_file_names) = if let Some(bk) = bank {
        let remote_files = client.list_media(server_url, bk, subdir).await?;
        if remote_files.is_empty() {
            eprintln!(
                "warning: no media files found on server in bank '{bk}' / '{}'",
                subdir.unwrap_or("/")
            );
            return Ok(());
        }
        eprintln!(
            "Found {} media file(s) on server (bank={bk}, subdir={}).",
            remote_files.len(),
            subdir.unwrap_or("/")
        );
        (
            bk.to_string(),
            subdir.unwrap_or("").to_string(),
            remote_files,
        )
    } else {
        let mapping = client::detect_media_mapping(&base_dir, &health.media_mapping_keys)?;
        if !mapping.key.is_empty() {
            eprintln!("Media mapping: {} / {}", mapping.key, mapping.subdir);
        }
        (mapping.key, mapping.subdir, media_file_names)
    };

    if file_payloads.is_empty() && media_file_names.is_empty() {
        eprintln!("warning: no files found with extensions {extensions:?}");
        return Ok(());
    }

    let total_count = file_payloads.len() + media_file_names.len();
    eprintln!("Found {total_count} file(s) to submit.\n");

    // Build options with lexicon
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

    let submission = JobSubmission {
        command,
        lang: batchalign_app::api::LanguageSpec::try_from(lang)
            .map_err(|e| CliError::InvalidArgument(format!("invalid language: {e}")))?,
        num_speakers: num_speakers.into(),
        files: file_payloads,
        media_files: media_file_names,
        media_mapping: mapping_key,
        media_subdir: mapping_subdir,
        source_dir: base_dir.to_string_lossy().to_string(),
        options: opts,
        paths_mode: false,
        source_paths: vec![],
        output_paths: vec![],
        display_names: vec![],
        debug_traces,
        before_paths: vec![],
    };

    // Submit
    eprintln!("Submitting to {server_url}...");
    let info = client.submit_job(server_url, &submission).await?;
    let job_id = &info.job_id;
    let total_files = info.total_files;
    eprintln!("Job {job_id} submitted ({total_files} file(s))");

    let dashboard_url = format!("{server_url}/dashboard/jobs/{job_id}");
    eprintln!("Dashboard: {dashboard_url}\n");

    maybe_open_dashboard(&dashboard_url, open_dashboard);

    // Poll and write incrementally
    if !info.status.is_terminal() {
        if use_tui && std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            let (tui_progress, tui_runtime) =
                TuiProgress::new(total_files as u64, command.as_wire_name());
            let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();

            // Cancel task — awaits signal from TUI, sends DELETE cancel
            let cc = client.clone();
            let cu = server_url.to_string();
            let cj = job_id.to_string();
            tokio::spawn(async move {
                if cancel_rx.await.is_ok() {
                    let _ = cc.cancel_job(&cu, &cj).await;
                }
            });

            // TUI on blocking thread
            let mut tui_handle = tokio::task::spawn_blocking(move || {
                crate::tui::run_tui_loop(tui_runtime, Some(cancel_tx))
            });

            // Poll on current task — pinned so it survives TUI exit
            let poll_fut = poll_and_write_incrementally(
                client,
                server_url,
                job_id,
                total_files as u64,
                &result_map,
                &effective_out,
                command.as_wire_name(),
                &tui_progress,
            );
            tokio::pin!(poll_fut);

            tokio::select! {
                result = &mut poll_fut => {
                    result?;
                    // Job finished — wait for TUI to exit
                    let _ = tui_handle.await;
                }
                _ = &mut tui_handle => {
                    // User closed TUI — continue writing results to disk
                    eprintln!("\nDashboard closed — still writing results...");
                    poll_fut.await?;
                }
            }
        } else {
            let progress = BatchProgress::new(total_files as u64, command.as_wire_name());
            poll_and_write_incrementally(
                client,
                server_url,
                job_id,
                total_files as u64,
                &result_map,
                &effective_out,
                command.as_wire_name(),
                &progress,
            )
            .await?;
        }
    }

    Ok(())
}
