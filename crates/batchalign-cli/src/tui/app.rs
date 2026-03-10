//! TUI application state machine for tracking job progress.
//!
//! This module models the full state needed to render the interactive terminal
//! UI during a processing job. The state is driven by periodic HTTP poll
//! responses from the server: each poll delivers a snapshot of file statuses,
//! and [`AppState::update_from_poll`] rebuilds the directory-grouped view
//! while preserving user navigation (focus, scroll offset).
//!
//! The state machine transitions are:
//!
//! 1. **Created** -- `AppState::new()`, progress starts at zero.
//! 2. **Polling** -- `update_from_poll()` called on each tick, groups rebuilt.
//! 3. **Finished** -- `finished` set to true, TUI shows final summary.
//!
//! User input (arrow keys, tab, `e`, `c`) mutates navigation fields only;
//! it never affects the job itself (cancellation is a separate server call).

use std::collections::HashMap;
use std::time::Instant;

use batchalign_app::api::{FileStatusEntry, FileStatusKind};

/// Reducer message sent from polling code into the TUI state owner.
pub enum TuiUpdate {
    /// Replace the current grouped file snapshot with the latest poll result.
    PollSnapshot {
        /// Completed file count.
        done: u64,
        /// Current server-reported file statuses.
        file_statuses: Vec<FileStatusEntry>,
    },
    /// Append one error to the persistent error summary panel.
    FileError {
        /// File that produced the error.
        filename: String,
        /// Human-readable error message.
        message: String,
    },
    /// Mark the TUI as finished so the loop can exit after showing final state.
    Finished,
}

/// Overall TUI state, updated from poll results.
///
/// A single instance is created per job and mutated in place on every poll
/// tick. The render loop reads this struct to draw the terminal UI.
pub struct AppState {
    /// Command/progress summary for the current job.
    pub progress: JobProgressState,
    /// Directory-grouped file view plus navigation state.
    pub directories: DirectoryViewState,
    /// Error summary panel state.
    pub errors: ErrorPanelState,
    /// Confirmation and other interaction-only state.
    pub interaction: InteractionState,
}

/// Job-level progress summary shown in the header.
pub struct JobProgressState {
    /// The batchalign command being run (e.g. `"morphotag"`, `"align"`).
    pub command: String,
    /// Total number of files in this job.
    pub total_files: u64,
    /// Number of files that have finished processing (success or error).
    pub completed: u64,
    /// Wall-clock instant when the job was submitted.
    pub start_time: Instant,
    /// True once the job has reached a terminal state locally.
    pub finished: bool,
}

/// Directory-grouped file list plus current navigation state.
pub struct DirectoryViewState {
    /// Files grouped by their parent directory path, sorted lexicographically.
    pub groups: Vec<DirGroup>,
    /// Index into `groups` indicating which directory group has keyboard focus.
    pub focused_group: usize,
    /// Scroll offset within the focused group's file list.
    pub scroll_offset: usize,
    /// Monotonically increasing counter driving the spinner animation.
    pub spinner_tick: usize,
}

/// State for the collapsible error summary panel.
pub struct ErrorPanelState {
    /// Accumulated error entries shown in the summary panel.
    pub entries: Vec<ErrorEntry>,
    /// Whether the error summary panel is expanded.
    pub expanded: bool,
}

/// Local interaction-only flags that do not affect the job itself.
pub struct InteractionState {
    /// True while the UI is showing a "press c again to confirm cancel" prompt.
    pub cancel_confirm: bool,
}

/// A group of files sharing a common parent directory.
///
/// Files are grouped by splitting each path at its last `/` separator.
/// Files without a directory component are grouped under `"."`.
/// Groups are sorted lexicographically by `dir` for stable display order.
pub struct DirGroup {
    /// The directory prefix shared by all files in this group
    /// (e.g. `"eng/Eng-NA"` or `"."` for root-level files).
    pub dir: String,

    /// All files in this directory, sorted alphabetically by filename.
    pub files: Vec<FileState>,

    /// Number of files with status `Done`. Invariant: `done_count <= files.len()`.
    pub done_count: usize,

    /// Number of files with status `Processing`. These are the files
    /// currently being worked on by a server worker.
    pub active_count: usize,

    /// Number of files with status `Error`. These files failed and will
    /// not be retried.
    pub error_count: usize,
}

/// Per-file processing status, extracted from the server's poll response.
///
/// Each file appears exactly once across all `DirGroup`s. Fields are
/// populated from `FileStatusEntry` on every poll, so values may change
/// between ticks (e.g. `status` transitions from `Queued` to `Processing`
/// to `Done`).
pub struct FileState {
    /// Filename without the directory prefix (e.g. `"test.cha"`).
    pub name: String,

    /// Full path as reported by the server (e.g. `"eng/Eng-NA/test.cha"`).
    pub full_path: String,

    /// Current processing status of this file.
    pub status: FileStatusKind,

    /// Wall-clock processing time in seconds, computed from the server's
    /// `started_at` and `finished_at` timestamps. `None` while the file
    /// is still queued or processing.
    pub duration_s: Option<f64>,

    /// Current step in a multi-step file operation (e.g. Rev.AI polling).
    /// `None` if the server does not report sub-file progress.
    pub progress_current: Option<i64>,

    /// Total number of steps for this file. `None` if unknown.
    pub progress_total: Option<i64>,

    /// Human-readable label for the current progress step
    /// (e.g. `"uploading"`, `"aligning"`). `None` if not reported.
    pub progress_label: Option<String>,

    /// Error message if the file failed. `None` for successful or
    /// in-progress files.
    pub error_msg: Option<String>,

    /// Structured error codes attached to the failure (e.g. `["E362"]`).
    /// Empty vec for files that have not failed.
    pub error_codes: Vec<String>,
}

/// An error entry displayed in the collapsible error summary panel.
///
/// Error entries are appended as they are discovered during polling and
/// are never removed. They provide a persistent record of every failure
/// in the current job, even after the file list has scrolled past.
pub struct ErrorEntry {
    /// The filename that produced this error (display name, not full path).
    pub filename: String,

    /// Structured error code if available (e.g. `"E362"`). `None` for
    /// errors that do not carry a CHAT-spec error code.
    pub code: Option<String>,

    /// Human-readable error description from the server or worker.
    pub message: String,
}

impl AppState {
    /// Create initial state for a new job.
    pub fn new(total_files: u64, command: &str) -> Self {
        Self {
            progress: JobProgressState {
                command: command.to_string(),
                total_files,
                completed: 0,
                start_time: Instant::now(),
                finished: false,
            },
            directories: DirectoryViewState {
                groups: Vec::new(),
                focused_group: 0,
                scroll_offset: 0,
                spinner_tick: 0,
            },
            errors: ErrorPanelState {
                entries: Vec::new(),
                expanded: false,
            },
            interaction: InteractionState {
                cancel_confirm: false,
            },
        }
    }

    /// Update state from poll results.
    pub fn update_from_poll(&mut self, done: u64, file_statuses: &[FileStatusEntry]) {
        self.progress.completed = done;

        // Group files by parent directory
        let mut dir_map: HashMap<String, Vec<FileState>> = HashMap::new();

        for entry in file_statuses {
            let (dir, name) = split_dir_file(&entry.filename);
            let status = entry.status;

            let duration_s = match (entry.started_at, entry.finished_at) {
                (Some(start), Some(end)) => Some(end.0 - start.0),
                _ => None,
            };

            let file_state = FileState {
                name: name.to_string(),
                full_path: entry.filename.to_string(),
                status,
                duration_s,
                progress_current: entry.progress_current,
                progress_total: entry.progress_total,
                progress_label: entry.progress_label.clone(),
                error_msg: entry.error.clone(),
                error_codes: entry.error_codes.clone().unwrap_or_default(),
            };

            dir_map.entry(dir.to_string()).or_default().push(file_state);
        }

        // Build sorted groups, preserving UI state
        let mut groups: Vec<DirGroup> = dir_map
            .into_iter()
            .map(|(dir, mut files)| {
                files.sort_by(|a, b| a.name.cmp(&b.name));
                let done_count = files
                    .iter()
                    .filter(|f| f.status == FileStatusKind::Done)
                    .count();
                let active_count = files
                    .iter()
                    .filter(|f| f.status == FileStatusKind::Processing)
                    .count();
                let error_count = files
                    .iter()
                    .filter(|f| f.status == FileStatusKind::Error)
                    .count();
                DirGroup {
                    dir,
                    files,
                    done_count,
                    active_count,
                    error_count,
                }
            })
            .collect();
        groups.sort_by(|a, b| a.dir.cmp(&b.dir));

        self.directories.groups = groups;

        // Clamp focus
        if !self.directories.groups.is_empty()
            && self.directories.focused_group >= self.directories.groups.len()
        {
            self.directories.focused_group = self.directories.groups.len() - 1;
        }
    }

    /// Apply one reducer message produced by the poll side of the TUI boundary.
    pub fn apply_update(&mut self, update: TuiUpdate) {
        match update {
            TuiUpdate::PollSnapshot {
                done,
                file_statuses,
            } => {
                self.update_from_poll(done, &file_statuses);
            }
            TuiUpdate::FileError { filename, message } => {
                self.add_error(&filename, &message);
            }
            TuiUpdate::Finished => {
                self.progress.finished = true;
            }
        }
    }

    /// Add an error entry from a poll error callback.
    pub fn add_error(&mut self, filename: &str, msg: &str) {
        self.errors.entries.push(ErrorEntry {
            filename: filename.to_string(),
            code: None,
            message: msg.to_string(),
        });
    }

    /// Scroll up within the focused group.
    pub fn scroll_up(&mut self) {
        self.directories.scroll_offset = self.directories.scroll_offset.saturating_sub(1);
    }

    /// Scroll down within the focused group.
    pub fn scroll_down(&mut self) {
        if let Some(group) = self.directories.groups.get(self.directories.focused_group) {
            let max = group.files.len().saturating_sub(1);
            if self.directories.scroll_offset < max {
                self.directories.scroll_offset += 1;
            }
        }
    }

    /// Cycle to the next directory group.
    pub fn cycle_group(&mut self) {
        if !self.directories.groups.is_empty() {
            self.directories.focused_group =
                (self.directories.focused_group + 1) % self.directories.groups.len();
            self.directories.scroll_offset = 0;
        }
    }

    /// Toggle the error panel expansion.
    pub fn toggle_errors(&mut self) {
        self.errors.expanded = !self.errors.expanded;
    }

    /// Return whether the job is locally marked as finished.
    pub fn is_finished(&self) -> bool {
        self.progress.finished
    }

    /// Return whether the cancel-confirm prompt is currently visible.
    pub fn cancel_confirm_active(&self) -> bool {
        self.interaction.cancel_confirm
    }

    /// Show the cancel-confirm prompt when the job is still active.
    pub fn request_cancel_confirmation(&mut self) {
        if !self.is_finished() {
            self.interaction.cancel_confirm = true;
        }
    }

    /// Clear the cancel-confirm prompt.
    pub fn clear_cancel_confirmation(&mut self) {
        self.interaction.cancel_confirm = false;
    }

    /// Advance the spinner animation one frame.
    pub fn tick_spinner(&mut self) {
        self.directories.spinner_tick = self.directories.spinner_tick.wrapping_add(1);
    }
}

/// Split "dir/subdir/file.cha" into ("dir/subdir", "file.cha").
/// If no directory component, returns (".", filename).
fn split_dir_file(path: &str) -> (&str, &str) {
    match path.rfind('/') {
        Some(idx) => (&path[..idx], &path[idx + 1..]),
        None => (".", path),
    }
}

#[cfg(test)]
mod tests {
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
    fn files_sorted_alphabetically_within_group() {
        let mut state = AppState::new(3, "align");
        // Deliberately out of order — server returns them in arbitrary order.
        let entries = vec![
            make_entry("PWA/ACWT10a.cha", FileStatusKind::Queued),
            make_entry("PWA/ACWT02a.cha", FileStatusKind::Processing),
            make_entry("PWA/ACWT05a.cha", FileStatusKind::Done),
        ];
        state.update_from_poll(1, &entries);

        let files = &state.directories.groups[0].files;
        assert_eq!(files[0].name, "ACWT02a.cha");
        assert_eq!(files[1].name, "ACWT05a.cha");
        assert_eq!(files[2].name, "ACWT10a.cha");
    }

    #[test]
    fn groups_files_by_directory() {
        let mut state = AppState::new(4, "morphotag");
        let entries = vec![
            make_entry("eng/a.cha", FileStatusKind::Done),
            make_entry("eng/b.cha", FileStatusKind::Processing),
            make_entry("spa/c.cha", FileStatusKind::Queued),
            make_entry("spa/d.cha", FileStatusKind::Error),
        ];
        state.update_from_poll(1, &entries);

        assert_eq!(state.directories.groups.len(), 2);
        assert_eq!(state.directories.groups[0].dir, "eng");
        assert_eq!(state.directories.groups[0].files.len(), 2);
        assert_eq!(state.directories.groups[0].done_count, 1);
        assert_eq!(state.directories.groups[0].active_count, 1);
        assert_eq!(state.directories.groups[1].dir, "spa");
        assert_eq!(state.directories.groups[1].error_count, 1);
    }

    #[test]
    fn group_counts_correct() {
        let mut state = AppState::new(3, "morphotag");
        let entries = vec![
            make_entry("d/a.cha", FileStatusKind::Done),
            make_entry("d/b.cha", FileStatusKind::Done),
            make_entry("d/c.cha", FileStatusKind::Error),
        ];
        state.update_from_poll(2, &entries);

        assert_eq!(state.directories.groups.len(), 1);
        assert_eq!(state.directories.groups[0].done_count, 2);
        assert_eq!(state.directories.groups[0].error_count, 1);
        assert_eq!(state.directories.groups[0].active_count, 0);
    }

    #[test]
    fn scroll_and_focus_preserved() {
        let mut state = AppState::new(4, "morphotag");
        let entries = vec![
            make_entry("a/x.cha", FileStatusKind::Queued),
            make_entry("b/y.cha", FileStatusKind::Queued),
        ];
        state.update_from_poll(0, &entries);
        state.directories.focused_group = 1;
        state.directories.scroll_offset = 0;

        // Update again — focus should be preserved
        state.update_from_poll(0, &entries);
        assert_eq!(state.directories.focused_group, 1);
    }

    #[test]
    fn focus_clamped_when_groups_shrink() {
        let mut state = AppState::new(2, "morphotag");
        state.directories.focused_group = 5;
        let entries = vec![make_entry("d/a.cha", FileStatusKind::Done)];
        state.update_from_poll(1, &entries);

        assert_eq!(state.directories.focused_group, 0);
    }

    #[test]
    fn split_dir_file_basic() {
        assert_eq!(
            split_dir_file("eng/Eng-NA/test.cha"),
            ("eng/Eng-NA", "test.cha")
        );
        assert_eq!(split_dir_file("test.cha"), (".", "test.cha"));
    }

    #[test]
    fn reducer_applies_poll_snapshot() {
        let mut state = AppState::new(1, "morphotag");
        state.apply_update(TuiUpdate::PollSnapshot {
            done: 1,
            file_statuses: vec![make_entry("eng/a.cha", FileStatusKind::Done)],
        });

        assert_eq!(state.progress.completed, 1);
        assert_eq!(state.directories.groups.len(), 1);
        assert_eq!(state.directories.groups[0].done_count, 1);
    }

    #[test]
    fn reducer_marks_finished() {
        let mut state = AppState::new(1, "morphotag");
        assert!(!state.is_finished());

        state.apply_update(TuiUpdate::Finished);
        assert!(state.is_finished());
    }

    #[test]
    fn cycle_group_wraps() {
        let mut state = AppState::new(2, "morphotag");
        let entries = vec![
            make_entry("a/x.cha", FileStatusKind::Queued),
            make_entry("b/y.cha", FileStatusKind::Queued),
        ];
        state.update_from_poll(0, &entries);

        state.cycle_group();
        assert_eq!(state.directories.focused_group, 1);
        state.cycle_group();
        assert_eq!(state.directories.focused_group, 0);
    }
}
