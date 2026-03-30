//! Single-server dispatch — submit files to one server, poll, write results.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use batchalign_app::ReleasedCommand;
use batchalign_app::api::JobSubmission;
use batchalign_app::options::CommandOptions;
use batchalign_app::released_command_uses_local_audio;

use crate::client::{BatchalignClient, server_label};
use crate::discover::{build_server_names, copy_nonmatching, infer_base_dir};
use crate::error::CliError;
use crate::progress::BatchProgress;
use crate::tui::TuiProgress;

/// Check if a server URL points to the local machine.
///
/// Returns `true` for localhost and 127.0.0.1 (the auto-daemon addresses).
/// Used to decide between paths mode (shared filesystem, for local daemons)
/// and content mode (HTTP upload, for explicit remote `--server`).
fn is_local_server(url: &str) -> bool {
    let after_scheme = url
        .trim_start_matches("http://")
        .trim_start_matches("https://");

    // Handle IPv6 bracket notation: [::1]:8001
    let host = if after_scheme.starts_with('[') {
        after_scheme
            .find(']')
            .map(|i| &after_scheme[..=i])
            .unwrap_or(after_scheme)
    } else {
        after_scheme.split(':').next().unwrap_or("")
    };

    matches!(host, "localhost" | "127.0.0.1" | "::1" | "[::1]")
}

#[cfg(test)]
mod tests {
    use super::is_local_server;

    #[test]
    fn localhost_is_local() {
        assert!(is_local_server("http://localhost:8001"));
        assert!(is_local_server("http://127.0.0.1:8001"));
        assert!(is_local_server("http://[::1]:8001"));
    }

    #[test]
    fn remote_hosts_are_not_local() {
        assert!(!is_local_server("http://net:8001"));
        assert!(!is_local_server("http://bilbo:8001"));
        assert!(!is_local_server("http://192.168.1.100:8001"));
        assert!(!is_local_server("http://talkbank.org:8001"));
    }
}

use super::helpers::{
    classify_files, filter_files_for_command, inject_lexicon, maybe_open_dashboard,
    poll_and_write_incrementally,
};
use super::paths::prepare_paths_submission;
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
    lexicon: Option<&str>,
    before: Option<&std::path::Path>,
    use_tui: bool,
    open_dashboard: bool,
) -> Result<(), CliError> {
    // Health check
    let health = match client.health_check(server_url).await {
        Ok(h) => h,
        Err(e) => {
            return Err(e);
        }
    };
    warn_stale_server(server_url, &health);

    // Check capabilities
    if !server_supports_command(&health.capabilities, command) {
        return Err(CliError::UnsupportedCommand {
            server: server_label(server_url).to_string(),
            command,
        });
    }

    // Paths mode is only valid when client and server share a filesystem.
    // For explicit remote --server URLs, always use content mode — the CHAT
    // file content is sent over HTTP, and the server resolves media via its
    // own media_mappings. Paths mode is for local daemons only.
    let server_is_local = is_local_server(server_url);
    let use_paths_mode = released_command_uses_local_audio(command) && server_is_local;

    let (submission, effective_out, result_map, paths_mode) = if use_paths_mode {
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
            &health.media_mapping_keys,
        )?
        else {
            eprintln!("warning: no files found with extensions {extensions:?}");
            return Ok(());
        };

        eprintln!("Found {} file(s) to submit.\n", prepared.total_files);
        eprintln!("Submitting shared-filesystem job to {server_url}...");
        eprintln!(
            "note: the server must be able to read these input paths. Successful outputs will also be copied back to this machine.\n"
        );

        (
            prepared.submission,
            prepared.effective_out,
            HashMap::new(),
            true,
        )
    } else {
        let (files, outputs) =
            crate::discover::discover_server_inputs(inputs, out_dir, extensions)?;
        let (files, outputs) = filter_files_for_command(command, files, outputs);

        if let Some(od) = out_dir {
            for inp in inputs {
                if Path::new(inp).is_dir() {
                    copy_nonmatching(Path::new(inp), Path::new(od), extensions, command)?;
                }
            }
        }

        let base_dir = infer_base_dir(inputs)?;
        let (server_names, result_map) = build_server_names(&files, &outputs, inputs)?;
        let (file_payloads, media_file_names) = classify_files(&files, &server_names)?;
        if file_payloads.is_empty() && media_file_names.is_empty() {
            eprintln!("warning: no files found with extensions {extensions:?}");
            return Ok(());
        }

        let total_count = file_payloads.len() + media_file_names.len();
        eprintln!("Found {total_count} file(s) to submit.\n");

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

        (
            JobSubmission {
                command,
                lang: batchalign_app::api::LanguageSpec::try_from(lang)
                    .map_err(|e| CliError::InvalidArgument(format!("invalid language: {e}")))?,
                num_speakers: num_speakers.into(),
                files: file_payloads,
                media_files: media_file_names,
                media_mapping: Default::default(),
                media_subdir: Default::default(),
                source_dir: base_dir.to_string_lossy().to_string().into(),
                options: opts,
                paths_mode: false,
                source_paths: vec![],
                output_paths: vec![],
                display_names: vec![],
                debug_traces,
                before_paths: vec![],
            },
            effective_out,
            result_map,
            false,
        )
    };

    if !paths_mode {
        eprintln!("Submitting to {server_url}...");
    }
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
            let cj = job_id.clone();
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
