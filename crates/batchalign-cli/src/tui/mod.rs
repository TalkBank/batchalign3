//! Full-screen TUI dashboard for job progress (opt-in via `--tui`).

pub mod app;
pub mod event;
pub mod ui;

use std::io;
use std::time::{Duration, Instant};

use batchalign_app::api::{FileStatusEntry, HealthResponse};
use crossterm::event::KeyCode;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

use crate::progress::ProgressSink;

use self::app::{AppState, ServerHealth, TuiUpdate};
use self::event::TuiEvent;

/// RAII guard for terminal raw mode + alternate screen.
///
/// On drop, restores the terminal to its normal state. This ensures
/// cleanup even if the TUI loop panics.
struct TerminalGuard;

impl TerminalGuard {
    /// Enter raw mode and switch the terminal into the alternate screen.
    fn new() -> io::Result<Self> {
        enable_raw_mode()?;
        crossterm::execute!(io::stdout(), EnterAlternateScreen)?;
        // Hide cursor
        crossterm::execute!(io::stdout(), crossterm::cursor::Hide)?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = crossterm::execute!(io::stdout(), crossterm::cursor::Show);
        let _ = crossterm::execute!(io::stdout(), LeaveAlternateScreen);
        let _ = disable_raw_mode();
    }
}

/// Progress sink that forwards reducer messages into the TUI runtime.
///
/// The polling code owns this sender-side adapter while the blocking TUI loop
/// owns the corresponding [`TuiRuntime`]. This keeps the render state local to
/// one thread instead of sharing it behind a mutex.
pub struct TuiProgress {
    updates: UnboundedSender<TuiUpdate>,
}

/// Owned runtime for the blocking TUI render loop.
///
/// The render thread owns both the current [`AppState`] and the inbound
/// reducer-message queue. Progress producers never mutate UI state directly;
/// they can only enqueue [`TuiUpdate`] values through [`TuiProgress`].
pub struct TuiRuntime {
    state: AppState,
    updates: UnboundedReceiver<TuiUpdate>,
}

impl TuiProgress {
    /// Create a new sender/runtime pair for one job TUI session.
    pub fn new(total_files: u64, command: &str) -> (Self, TuiRuntime) {
        let (updates, receiver) = unbounded_channel();
        (
            Self { updates },
            TuiRuntime::new(total_files, command, receiver),
        )
    }

    /// Forward one reducer message to the TUI runtime, ignoring closed sessions.
    fn send_update(&self, update: TuiUpdate) {
        let _ = self.updates.send(update);
    }
}

impl TuiRuntime {
    /// Create the state owner for one TUI session.
    fn new(total_files: u64, command: &str, updates: UnboundedReceiver<TuiUpdate>) -> Self {
        Self {
            state: AppState::new(total_files, command),
            updates,
        }
    }

    /// Apply all currently queued progress updates to the owned UI state.
    fn apply_pending_updates(&mut self) {
        while let Ok(update) = self.updates.try_recv() {
            self.state.apply_update(update);
        }
    }
}

impl ProgressSink for TuiProgress {
    fn update(&self, done: u64, file_statuses: &[FileStatusEntry]) {
        self.send_update(TuiUpdate::PollSnapshot {
            done,
            file_statuses: file_statuses.to_vec(),
        });
    }

    fn log_done(&self, _filename: &str) {
        // State already updated via update() — no additional action needed.
    }

    fn log_error(&self, filename: &str, msg: &str) {
        self.send_update(TuiUpdate::FileError {
            filename: filename.to_string(),
            message: msg.to_string(),
        });
    }

    fn finish(&self) {
        self.send_update(TuiUpdate::Finished);
    }

    fn update_health(&self, health: &HealthResponse) {
        let warmup_label = serde_json::to_value(health.warmup_status)
            .ok()
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_else(|| "unknown".into());

        self.send_update(TuiUpdate::HealthSnapshot(ServerHealth {
            live_workers: health.live_workers,
            live_worker_keys: health.live_worker_keys.clone(),
            system_memory_total_mb: health.system_memory_total_mb,
            system_memory_available_mb: health.system_memory_available_mb,
            system_memory_used_mb: health.system_memory_used_mb,
            memory_gate_threshold_mb: health.memory_gate_threshold_mb,
            warmup_status: warmup_label,
        }));
    }
}

/// Run the TUI rendering + input loop on a blocking thread.
///
/// - `runtime`: owned TUI runtime updated by the poll task.
/// - `cancel_tx`: oneshot sender to signal job cancellation.
///
/// Returns when the user presses 'q' or the job finishes.
pub fn run_tui_loop(
    mut runtime: TuiRuntime,
    cancel_tx: Option<tokio::sync::oneshot::Sender<()>>,
) -> io::Result<()> {
    let _guard = TerminalGuard::new()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    let mut last_tick = Instant::now();
    let mut cancel_tx = cancel_tx;

    loop {
        runtime.apply_pending_updates();

        // Draw
        terminal.draw(|f| {
            ui::draw(f, &runtime.state);
        })?;

        // Check if finished — show summary and pause
        if runtime.state.is_finished() {
            // Redraw one final time with summary overlay
            terminal.draw(|f| {
                ui::draw(f, &runtime.state);
            })?;
            std::thread::sleep(Duration::from_secs(2));
            break;
        }

        // Poll input
        let timeout = Duration::from_millis(100).saturating_sub(last_tick.elapsed());
        if let Some(evt) = event::poll_event(timeout) {
            match evt {
                TuiEvent::Key(KeyCode::Char('q'), _) => break,
                TuiEvent::Key(KeyCode::Char('c'), modifiers)
                    if modifiers.contains(crossterm::event::KeyModifiers::CONTROL) =>
                {
                    break;
                }
                TuiEvent::Key(KeyCode::Char('c'), _) => {
                    runtime.state.request_cancel_confirmation();
                }
                TuiEvent::Key(KeyCode::Char('y'), _) if runtime.state.cancel_confirm_active() => {
                    runtime.state.clear_cancel_confirmation();
                    if let Some(tx) = cancel_tx.take() {
                        let _ = tx.send(());
                    }
                }
                TuiEvent::Key(KeyCode::Char('n'), _) if runtime.state.cancel_confirm_active() => {
                    runtime.state.clear_cancel_confirmation();
                }
                TuiEvent::Key(KeyCode::Up, _) => runtime.state.scroll_up(),
                TuiEvent::Key(KeyCode::Down, _) => runtime.state.scroll_down(),
                TuiEvent::Key(KeyCode::Tab, _) => runtime.state.cycle_group(),
                TuiEvent::Key(KeyCode::Char('e'), _) => runtime.state.toggle_errors(),
                TuiEvent::Key(KeyCode::Char('m'), _) => runtime.state.toggle_metrics(),
                TuiEvent::Key(KeyCode::Esc, _) => {
                    if runtime.state.cancel_confirm_active() {
                        runtime.state.clear_cancel_confirmation();
                    }
                }
                _ => {}
            }
        }

        // Tick spinner
        if last_tick.elapsed() >= Duration::from_millis(100) {
            runtime.state.tick_spinner();
            last_tick = Instant::now();
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use batchalign_app::api::FileStatusKind;

    use super::*;

    fn make_entry(filename: &str, status: FileStatusKind) -> FileStatusEntry {
        FileStatusEntry {
            filename: filename.into(),
            status,
            error: None,
            error_category: None,
            error_codes: None,
            error_line: None,
            bug_report_id: None,
            started_at: None,
            finished_at: None,
            next_eligible_at: None,
            progress_current: None,
            progress_total: None,
            progress_stage: None,
            progress_label: None,
        }
    }

    #[test]
    fn progress_sink_forwards_updates_into_runtime() {
        let (progress, mut runtime) = TuiProgress::new(1, "morphotag");

        progress.update(1, &[make_entry("eng/a.cha", FileStatusKind::Done)]);
        progress.log_error("eng/a.cha", "parse failed");
        progress.finish();
        runtime.apply_pending_updates();

        assert_eq!(runtime.state.progress.completed, 1);
        assert_eq!(runtime.state.directories.groups.len(), 1);
        assert_eq!(runtime.state.errors.entries.len(), 1);
        assert!(runtime.state.is_finished());
    }
}
