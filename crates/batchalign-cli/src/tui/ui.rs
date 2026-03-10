//! Ratatui rendering — layout, widgets, colors.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, List, ListItem, Paragraph};

use batchalign_app::api::FileStatusKind;

use super::app::AppState;

/// Braille spinner characters.
const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Draw the full TUI frame.
pub fn draw(f: &mut Frame, state: &AppState) {
    let area = f.area();

    // Determine error panel height
    let error_height = if state.errors.entries.is_empty() {
        0
    } else if state.errors.expanded {
        (state.errors.entries.len() as u16 + 2).min(8)
    } else {
        1
    };

    let chunks = Layout::vertical([
        Constraint::Length(3),            // header + gauge
        Constraint::Min(4),               // directory groups
        Constraint::Length(error_height), // error summary
        Constraint::Length(1),            // keybind bar
    ])
    .split(area);

    draw_header(f, state, chunks[0]);
    draw_groups(f, state, chunks[1]);
    if error_height > 0 {
        draw_errors(f, state, chunks[2]);
    }
    draw_keybinds(f, state, chunks[3]);
}

/// Header: command badge + progress gauge + file count + elapsed.
fn draw_header(f: &mut Frame, state: &AppState, area: Rect) {
    let elapsed = state.progress.start_time.elapsed();
    let mins = elapsed.as_secs() / 60;
    let secs = elapsed.as_secs() % 60;

    let ratio = if state.progress.total_files > 0 {
        (state.progress.completed as f64) / (state.progress.total_files as f64)
    } else {
        0.0
    };

    let label = format!(
        " {} — {}/{} files  [{mins:02}:{secs:02}]",
        state.progress.command, state.progress.completed, state.progress.total_files
    );

    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .gauge_style(Style::default().fg(Color::Cyan))
        .ratio(ratio.clamp(0.0, 1.0))
        .label(label);

    f.render_widget(gauge, area);
}

/// Directory groups — bordered sections with file rows.
fn draw_groups(f: &mut Frame, state: &AppState, area: Rect) {
    if state.directories.groups.is_empty() {
        let msg =
            Paragraph::new("  Waiting for server…").style(Style::default().fg(Color::DarkGray));
        f.render_widget(msg, area);
        return;
    }

    // Split area evenly across groups (up to what fits)
    let group_count = state
        .directories
        .groups
        .len()
        .min(area.height as usize / 3)
        .max(1);
    let constraints: Vec<Constraint> = (0..group_count).map(|_| Constraint::Min(3)).collect();
    let group_areas = Layout::vertical(constraints).split(area);

    for (i, group_area) in group_areas.iter().enumerate() {
        let group_idx = if state.directories.focused_group < group_count {
            i
        } else {
            // Scroll groups so focused is visible
            let start = state
                .directories
                .focused_group
                .saturating_sub(group_count - 1);
            start + i
        };

        if group_idx >= state.directories.groups.len() {
            break;
        }

        let group = &state.directories.groups[group_idx];
        let is_focused = group_idx == state.directories.focused_group;

        let title = format!(
            " {} ({}/{}) ",
            group.dir,
            group.done_count + group.error_count,
            group.files.len()
        );

        let border_style = if is_focused {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(border_style)
            .title(title);

        let inner = block.inner(*group_area);
        f.render_widget(block, *group_area);

        // File rows with scrolling
        let visible_rows = inner.height as usize;
        let scroll = if is_focused {
            state.directories.scroll_offset
        } else {
            0
        };
        let items: Vec<ListItem> = group
            .files
            .iter()
            .skip(scroll)
            .take(visible_rows)
            .map(|file| {
                let line = render_file_line(file, state.directories.spinner_tick, inner.width);
                ListItem::new(line)
            })
            .collect();

        let list = List::new(items);
        f.render_widget(list, inner);
    }
}

/// Render a single file line with status glyph, name, progress info.
fn render_file_line(
    file: &super::app::FileState,
    spinner_tick: usize,
    width: u16,
) -> Line<'static> {
    let w = width as usize;

    match file.status {
        FileStatusKind::Queued | FileStatusKind::Interrupted => {
            let text = format!("  · {}", file.name);
            Line::from(Span::styled(
                pad_or_truncate(&text, w),
                Style::default().fg(Color::DarkGray),
            ))
        }
        FileStatusKind::Processing => {
            let spinner = SPINNER[spinner_tick % SPINNER.len()];
            let label = file.progress_label.as_deref().unwrap_or("");
            let pct = match (file.progress_current, file.progress_total) {
                (Some(c), Some(t)) if t > 0 => format!("  {c}/{t}"),
                _ => String::new(),
            };
            let text = format!("  {spinner} {:<30} {label}{pct}", file.name);
            Line::from(Span::styled(
                pad_or_truncate(&text, w),
                Style::default().fg(Color::Cyan),
            ))
        }
        FileStatusKind::Done => {
            let dur = file
                .duration_s
                .map(|d| format!("{d:.1}s"))
                .unwrap_or_default();
            let name_part = format!("  ✓ {}", file.name);
            let padding = w.saturating_sub(name_part.len() + dur.len());
            let text = format!("{name_part}{:>pad$}{dur}", "", pad = padding);
            Line::from(Span::styled(
                pad_or_truncate(&text, w),
                Style::default().fg(Color::Green),
            ))
        }
        FileStatusKind::Error => {
            let code = file
                .error_codes
                .first()
                .map(|c| format!("  {c}"))
                .unwrap_or_default();
            let msg = file
                .error_msg
                .as_deref()
                .and_then(|m| m.split('\n').next())
                .unwrap_or("");
            let msg_short = if msg.len() > 30 {
                format!("{}…", &msg[..29])
            } else {
                msg.to_string()
            };
            let text = format!("  ✗ {}{code}   {msg_short}", file.name);
            Line::from(Span::styled(
                pad_or_truncate(&text, w),
                Style::default().fg(Color::Red),
            ))
        }
    }
}

/// Error summary panel.
fn draw_errors(f: &mut Frame, state: &AppState, area: Rect) {
    if state.errors.entries.is_empty() {
        return;
    }

    if !state.errors.expanded {
        let summary = format!(
            "  {} error(s) — press 'e' to expand",
            state.errors.entries.len()
        );
        let p = Paragraph::new(summary).style(Style::default().fg(Color::Red));
        f.render_widget(p, area);
        return;
    }

    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(format!(" Errors ({}) ", state.errors.entries.len()));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let items: Vec<ListItem> = state
        .errors
        .entries
        .iter()
        .take(inner.height as usize)
        .map(|err| {
            let first_line = err.message.split('\n').next().unwrap_or("unknown");
            let code_str = err
                .code
                .as_deref()
                .map(|c| format!("[{c}] "))
                .unwrap_or_default();
            let text = format!("  ✗ {}: {code_str}{first_line}", err.filename);
            ListItem::new(Line::from(Span::styled(
                text,
                Style::default().fg(Color::Red),
            )))
        })
        .collect();

    f.render_widget(List::new(items), inner);
}

/// Bottom keybind bar.
fn draw_keybinds(f: &mut Frame, state: &AppState, area: Rect) {
    let line = if state.interaction.cancel_confirm {
        Line::from(vec![
            Span::styled(
                "  Cancel job? ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "y",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("/", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "n",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
        ])
    } else {
        Line::from(vec![
            Span::styled("  q", Style::default().fg(Color::Cyan)),
            Span::styled(" quit  ", Style::default().fg(Color::DarkGray)),
            Span::styled("c", Style::default().fg(Color::Cyan)),
            Span::styled(" cancel  ", Style::default().fg(Color::DarkGray)),
            Span::styled("↑↓", Style::default().fg(Color::Cyan)),
            Span::styled(" scroll  ", Style::default().fg(Color::DarkGray)),
            Span::styled("tab", Style::default().fg(Color::Cyan)),
            Span::styled(" group  ", Style::default().fg(Color::DarkGray)),
            Span::styled("e", Style::default().fg(Color::Cyan)),
            Span::styled(" errors", Style::default().fg(Color::DarkGray)),
        ])
    };

    f.render_widget(Paragraph::new(line), area);
}

/// Pad or truncate a string to exactly `width` characters.
fn pad_or_truncate(s: &str, width: usize) -> String {
    if s.len() >= width {
        s[..width].to_string()
    } else {
        format!("{s:<width$}")
    }
}

#[cfg(test)]
mod tests {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use batchalign_app::api::UnixTimestamp;

    use super::*;
    use crate::tui::app::AppState;

    fn make_entry(filename: &str, status: FileStatusKind) -> batchalign_app::api::FileStatusEntry {
        batchalign_app::api::FileStatusEntry {
            filename: filename.into(),
            status,
            error: if status == FileStatusKind::Error {
                Some("morph lookup failed".into())
            } else {
                None
            },
            error_category: None,
            error_codes: if status == FileStatusKind::Error {
                Some(vec!["E4012".into()])
            } else {
                None
            },
            error_line: None,
            bug_report_id: None,
            started_at: if status == FileStatusKind::Done {
                Some(UnixTimestamp(0.0))
            } else {
                None
            },
            finished_at: if status == FileStatusKind::Done {
                Some(UnixTimestamp(1.2))
            } else {
                None
            },
            progress_current: if status == FileStatusKind::Processing {
                Some(12)
            } else {
                None
            },
            progress_total: if status == FileStatusKind::Processing {
                Some(45)
            } else {
                None
            },
            next_eligible_at: None,
            progress_stage: None,
            progress_label: if status == FileStatusKind::Processing {
                Some("stanza".into())
            } else {
                None
            },
        }
    }

    #[test]
    fn render_empty_state() {
        let state = AppState::new(10, "morphotag");
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| draw(f, &state))
            .expect("draw should not panic");
    }

    #[test]
    fn render_mixed_statuses() {
        let mut state = AppState::new(4, "morphotag");
        let entries = vec![
            make_entry("eng/a.cha", FileStatusKind::Done),
            make_entry("eng/b.cha", FileStatusKind::Processing),
            make_entry("eng/c.cha", FileStatusKind::Queued),
            make_entry("eng/d.cha", FileStatusKind::Error),
        ];
        state.update_from_poll(1, &entries);

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| draw(f, &state))
            .expect("draw should not panic");
    }

    #[test]
    fn render_error_expanded() {
        let mut state = AppState::new(2, "morphotag");
        state.add_error("test.cha", "something broke");
        state.errors.expanded = true;

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| draw(f, &state))
            .expect("draw should not panic");
    }

    #[test]
    fn render_cancel_confirm() {
        let mut state = AppState::new(2, "morphotag");
        state.interaction.cancel_confirm = true;

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| draw(f, &state))
            .expect("draw should not panic");
    }
}
