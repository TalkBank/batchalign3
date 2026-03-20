//! Dispatch router — routes processing commands to servers.
//!
//! Mirrors `dispatch.py` + `dispatch_server.py`.
//!
//! The Rust CLI is always an HTTP client — it never loads ML models.
//! All processing goes through a server (remote or local daemon).

mod helpers;
mod paths;
mod single;

use batchalign_app::config::{RuntimeLayout, load_config_from_layout};
use batchalign_app::options::CommandOptions;
use tracing::debug;

use crate::client::{self, BatchalignClient, server_label};
use crate::daemon;
use crate::error::CliError;

use paths::dispatch_paths_mode;
use single::dispatch_single_server;

// ---------------------------------------------------------------------------
// Top-level dispatch router
// ---------------------------------------------------------------------------

/// Route a processing command to the appropriate server(s).
///
/// This is the main entry point for all CLI processing commands. It resolves
/// where to send work using the following priority chain:
///
/// 1. **Explicit `--server URL`** -- single-server dispatch
///    via HTTP content mode (CHAT text posted to server, results downloaded).
///    Audio-dependent commands (`transcribe`, `transcribe_s`, `benchmark`,
///    `avqi`) fall back to the local daemon even if `--server` is set,
///    because the remote content-mode path cannot access local audio files.
/// 2. **Auto-daemon** (if `auto_daemon` is enabled in `server.yaml`) --
///    paths-mode dispatch to a local daemon that reads/writes files directly.
/// 3. **Error** -- no server available; an error message is printed.
///
/// # Parameters
///
/// - `command`: the processing command name (e.g. `"morphotag"`, `"align"`).
/// - `lang`: ISO 639-3 language code (e.g. `"eng"`).
/// - `num_speakers`: expected number of speakers (used for diarization).
/// - `extensions`: file extensions to discover (e.g. `&["cha"]`, `&["wav", "mp3"]`).
/// - `server_arg`: explicit `--server` URL, if provided.
/// - `inputs`: input paths (files or directories) from the command line.
/// - `out_dir`: optional output directory; `None` means in-place processing.
/// - `options`: typed command options. `None` for non-processing commands.
/// - `bank`: optional `--bank` name for TalkBank corpus filtering (requires `--server`).
/// - `subdir`: optional `--subdir` within a bank.
/// - `lexicon`: optional lexicon path.
/// - `use_tui`: if `true`, renders a full-screen ratatui TUI instead of progress bars.
/// - `open_dashboard`: if `true`, auto-launches the submitted job URL in a
///   browser on supported platforms after submission.
///
/// # Errors
///
/// Returns [`CliError`] on I/O failures, HTTP errors, job failures, or if
/// no server can be resolved.
#[allow(clippy::too_many_arguments)]
pub async fn dispatch(
    command: &str,
    lang: &str,
    num_speakers: u32,
    extensions: &[&str],
    server_arg: Option<&str>,
    inputs: &[String],
    out_dir: Option<&str>,
    options: Option<CommandOptions>,
    bank: Option<&str>,
    subdir: Option<&str>,
    lexicon: Option<&str>,
    use_tui: bool,
    open_dashboard: bool,
    force_cpu: bool,
    before: Option<&str>,
    workers: Option<usize>,
    timeout: Option<u64>,
) -> Result<(), CliError> {
    let client = BatchalignClient::new();
    let layout = RuntimeLayout::from_env();
    let daemon_log_path = layout.state_dir().join("daemon.log");

    // --bank requires --server
    if bank.is_some() && server_arg.is_none() {
        eprintln!("error: --bank requires --server");
        return Ok(());
    }

    // 1. Explicit --server
    if let Some(server) = server_arg {
        if command_uses_local_audio(command) {
            eprintln!(
                "warning: {command} uses local audio — ignoring --server and using local daemon."
            );
            match resolve_local_daemon_for_command(&client, command, force_cpu, workers, timeout)
                .await
            {
                Ok(Some(daemon_url)) => {
                    return dispatch_paths_mode(
                        &client,
                        &daemon_url,
                        command,
                        lang,
                        num_speakers,
                        extensions,
                        inputs,
                        out_dir,
                        options.as_ref(),
                        bank,
                        subdir,
                        lexicon,
                        use_tui,
                        open_dashboard,
                        before,
                    )
                    .await;
                }
                Ok(None) => {
                    eprintln!(
                        "warning: local daemon unavailable for {command}. \
                         Check {} or start one with `batchalign3 serve start`.",
                        daemon_log_path.display()
                    );
                    return Err(CliError::DaemonStartFailed);
                }
                Err(e) => return Err(e),
            }
        }

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
                bank,
                subdir,
                lexicon,
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

    // 2. Auto-daemon
    let cfg = load_config_from_layout(&layout, None).unwrap_or_default();
    if cfg.auto_daemon {
        match resolve_local_daemon_for_command(&client, command, force_cpu, workers, timeout).await
        {
            Ok(Some(daemon_url)) => {
                return dispatch_paths_mode(
                    &client,
                    &daemon_url,
                    command,
                    lang,
                    num_speakers,
                    extensions,
                    inputs,
                    out_dir,
                    options.as_ref(),
                    bank,
                    subdir,
                    lexicon,
                    use_tui,
                    open_dashboard,
                    before,
                )
                .await;
            }
            Ok(None) => {
                eprintln!(
                    "warning: could not start local daemon. Check {} for details.",
                    daemon_log_path.display()
                );
                return Err(CliError::DaemonStartFailed);
            }
            Err(e) => {
                debug!(error = %e, "Daemon startup failed");
                return Err(e);
            }
        }
    }

    // 3. Only reached when auto_daemon is explicitly false
    eprintln!("error: no server available. Use --server URL or enable auto_daemon in server.yaml.");
    Ok(())
}

async fn resolve_local_daemon_for_command(
    client: &BatchalignClient,
    command: &str,
    force_cpu: bool,
    workers: Option<usize>,
    timeout: Option<u64>,
) -> Result<Option<String>, CliError> {
    let main = daemon::ensure_daemon(force_cpu, workers, timeout).await?;
    if let Some(url) = main {
        if daemon_supports_command(client, &url, command).await {
            return Ok(Some(url));
        }

        let can_use_sidecar = command_uses_local_audio(command);
        if can_use_sidecar {
            eprintln!("warning: main daemon lacks '{command}', trying sidecar daemon.");
            if let Some(sidecar_url) =
                daemon::ensure_sidecar_daemon(force_cpu, workers, timeout).await?
                && daemon_supports_command(client, &sidecar_url, command).await
            {
                return Ok(Some(sidecar_url));
            }
        } else {
            eprintln!(
                "warning: local daemon does not advertise support for '{command}'. \
                 Check worker dependencies."
            );
        }
    } else if command_uses_local_audio(command)
        && let Some(sidecar_url) =
            daemon::ensure_sidecar_daemon(force_cpu, workers, timeout).await?
        && daemon_supports_command(client, &sidecar_url, command).await
    {
        return Ok(Some(sidecar_url));
    }

    Ok(None)
}

async fn daemon_supports_command(client: &BatchalignClient, url: &str, command: &str) -> bool {
    match client.health_check(url).await {
        Ok(health) => server_supports_command(&health.capabilities, command),
        Err(_) => false,
    }
}

fn server_supports_command(capabilities: &[String], command: &str) -> bool {
    capabilities.is_empty()
        || capabilities
            .iter()
            .any(|c| c == command || c == "test-echo")
}

fn command_uses_local_audio(command: &str) -> bool {
    matches!(
        command,
        "transcribe" | "transcribe_s" | "benchmark" | "avqi"
    )
}

/// Warn (but don't block) if the server's build hash differs from the CLI's.
///
/// For auto-daemon connections, stale detection is handled by
/// `daemon::ensure_daemon_locked()` (auto-restart).  This warning covers
/// explicit `--server` connections where auto-restart isn't possible.
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

#[cfg(test)]
mod tests {
    use super::command_uses_local_audio;

    #[test]
    fn benchmark_is_treated_as_local_audio_command() {
        assert!(command_uses_local_audio("benchmark"));
        assert!(command_uses_local_audio("transcribe"));
        assert!(!command_uses_local_audio("morphotag"));
    }
}
