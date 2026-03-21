//! Shared helper functions for dispatch modes.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use batchalign_app::ReleasedCommand;
use batchalign_app::api::{FileName, FilePayload, FileStatusKind, JobInfo, JobStatus};

use crate::client::{BatchalignClient, MAX_POLL_FAILURES, POLL_MAX, POLL_MIN, POLL_STEP};
use crate::error::CliError;
use crate::output;
use crate::progress::ProgressSink;

// ---------------------------------------------------------------------------
// Single-server incremental poll
// ---------------------------------------------------------------------------

/// Named CLI-visible file failure detail.
#[derive(Debug, Clone)]
pub(super) struct FileErrorDetail {
    /// File identity reported to the user.
    pub filename: FileName,
    /// Human-readable failure explanation.
    pub message: String,
}

impl FileErrorDetail {
    /// Construct one failure detail from a file identity and message.
    pub(super) fn new(filename: impl Into<FileName>, message: impl Into<String>) -> Self {
        Self {
            filename: filename.into(),
            message: message.into(),
        }
    }
}

/// Poll a single-server job, writing results incrementally as files complete.
#[allow(clippy::too_many_arguments)]
pub(super) async fn poll_and_write_incrementally(
    client: &BatchalignClient,
    server_url: &str,
    job_id: &str,
    total_files: u64,
    result_map: &HashMap<String, PathBuf>,
    out_dir: &Path,
    _command: &str,
    progress: &dyn ProgressSink,
) -> Result<(), CliError> {
    let mut written_files: HashSet<String> = HashSet::new();
    let mut written_count: u64 = 0;
    let mut error_details: Vec<FileErrorDetail> = Vec::new();
    let mut consecutive_failures: u32 = 0;
    let mut poll_interval = POLL_MIN;
    let mut last_completed: i64 = 0;
    let mut last_health_poll = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_secs(10))
        .unwrap_or_else(std::time::Instant::now);

    loop {
        match client.get_job(server_url, job_id).await {
            Ok(info) => {
                consecutive_failures = 0;

                for entry in &info.file_statuses {
                    let fn_ = &entry.filename;
                    if written_files.contains(&**fn_) {
                        continue;
                    }

                    if entry.status == FileStatusKind::Done {
                        match client.get_file_result(server_url, job_id, fn_).await {
                            Ok(result) => {
                                match output::write_result(&result, result_map, out_dir) {
                                    Ok(true) => {
                                        written_count += 1;
                                        progress.log_done(fn_);
                                    }
                                    Ok(false) => {
                                        let error_msg = result.error.unwrap_or_default();
                                        progress.log_error(fn_, &error_msg);
                                        error_details.push(FileErrorDetail::new(
                                            fn_.clone(),
                                            error_msg,
                                        ));
                                    }
                                    Err(e) => {
                                        let error_msg = format!("{e}");
                                        progress.log_error(fn_, &error_msg);
                                        error_details.push(FileErrorDetail::new(
                                            fn_.clone(),
                                            error_msg,
                                        ));
                                    }
                                }
                            }
                            Err(e) => {
                                let error_msg = format!("{e}");
                                progress.log_error(fn_, &error_msg);
                                error_details.push(FileErrorDetail::new(fn_.clone(), error_msg));
                            }
                        }
                        written_files.insert(fn_.to_string());
                    } else if entry.status == FileStatusKind::Error {
                        written_files.insert(fn_.to_string());
                        let error_msg = entry
                            .error
                            .clone()
                            .unwrap_or_else(|| "unknown error".into());
                        progress.log_error(fn_, &error_msg);
                        error_details.push(FileErrorDetail::new(fn_.clone(), error_msg));
                    }
                }

                let done_so_far = written_count + error_details.len() as u64;
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
        if last_health_poll.elapsed() >= std::time::Duration::from_secs(5) {
            if let Ok(h) = client.health_check(server_url).await {
                progress.update_health(&h);
            }
            last_health_poll = std::time::Instant::now();
        }

        tokio::time::sleep(Duration::from_secs_f64(poll_interval)).await;
    }
}

/// Resolve whether the CLI should auto-open a submitted dashboard URL.
///
/// The public CLI flag is the main user-facing control, while the
/// `BATCHALIGN_NO_BROWSER` environment variable remains a hidden backstop for
/// tests and harnesses that must suppress browser launch.
pub(super) fn dashboard_auto_open_enabled(cli_enabled: bool, no_browser_env: bool) -> bool {
    cli_enabled && !no_browser_env
}

/// Launch the submitted job's dashboard URL in the local browser when enabled.
pub(super) fn maybe_open_dashboard(dashboard_url: &str, cli_enabled: bool) {
    #[cfg(target_os = "macos")]
    {
        if !dashboard_auto_open_enabled(
            cli_enabled,
            std::env::var_os("BATCHALIGN_NO_BROWSER").is_some(),
        ) {
            return;
        }

        let _ = std::process::Command::new("open")
            .arg(dashboard_url)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (dashboard_url, cli_enabled);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn terminal_job_detail(info: &JobInfo, error_details: &[FileErrorDetail]) -> String {
    if let Some(job_error) = info.error.as_ref().filter(|s| !s.trim().is_empty()) {
        return job_error.clone();
    }
    if let Some(detail) = error_details.first() {
        let first_line = detail.message.lines().next().unwrap_or("unknown error");
        let filename = detail.filename.as_ref();
        return format!("{filename}: {first_line}");
    }
    match info.status {
        JobStatus::Cancelled => "job was cancelled".into(),
        JobStatus::Interrupted => "job was interrupted".into(),
        JobStatus::Failed => "job failed without a reported error".into(),
        JobStatus::Completed | JobStatus::Queued | JobStatus::Running => {
            "job reported success without a detailed error".into()
        }
    }
}

fn print_job_terminal_failure(
    info: &JobInfo,
    error_details: &[FileErrorDetail],
    total_files: u64,
    out_dir: &Path,
) {
    if !error_details.is_empty() {
        print_failure_summary(error_details, total_files, out_dir);
        if let Some(job_error) = info.error.as_ref().filter(|s| !s.trim().is_empty()) {
            eprintln!("job error: {job_error}");
        }
        return;
    }

    let detail = terminal_job_detail(info, error_details);
    let bar = "\u{2501}".repeat(50);
    eprintln!("\n{bar}");
    eprintln!("  JOB {}: {}", info.status, detail);
    eprintln!("{bar}\n");
}

pub(super) fn finish_terminal_job(
    info: &JobInfo,
    error_details: &[FileErrorDetail],
    total_files: u64,
    out_dir: &Path,
) -> Result<(), CliError> {
    let clean_success = info.status == JobStatus::Completed
        && error_details.is_empty()
        && info.error.as_ref().is_none_or(|s| s.trim().is_empty());
    if clean_success {
        print_failure_summary(error_details, total_files, out_dir);
        return Ok(());
    }

    let detail = terminal_job_detail(info, error_details);
    print_job_terminal_failure(info, error_details, total_files, out_dir);
    Err(CliError::JobFailed {
        job_id: info.job_id.clone(),
        status: info.status.to_string(),
        detail,
    })
}

/// Command-specific file filtering after extension-based discovery.
///
/// AVQI operates on paired `.cs/.sv` files and should only process the
/// continuous-speech side (`*.cs.<ext>`). The sustained-vowel partner is
/// resolved server-side by filename convention. Compare uses `*.gold.cha`
/// companions as references and should not submit them as primary inputs.
pub(super) fn filter_files_for_command(
    command: ReleasedCommand,
    files: Vec<PathBuf>,
    outputs: Vec<PathBuf>,
) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut kept_files = Vec::new();
    let mut kept_outputs = Vec::new();
    for (f, o) in files.into_iter().zip(outputs.into_iter()) {
        let name = f.file_name().and_then(|s| s.to_str()).unwrap_or_default();
        let lower = name.to_ascii_lowercase();
        let keep = match command {
            ReleasedCommand::Avqi => lower.contains(".cs."),
            ReleasedCommand::Compare => !lower.ends_with(".gold.cha"),
            _ => true,
        };
        if keep {
            kept_files.push(f);
            kept_outputs.push(o);
        }
    }

    (kept_files, kept_outputs)
}

/// Classify files into CHAT payloads and media filenames.
pub(super) fn classify_files(
    files: &[PathBuf],
    server_names: &[String],
) -> Result<(Vec<FilePayload>, Vec<String>), CliError> {
    let mut payloads = Vec::new();
    let mut media_names = Vec::new();

    for (path, name) in files.iter().zip(server_names.iter()) {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if ext == "cha" {
            let content = std::fs::read_to_string(path)?;
            payloads.push(FilePayload {
                filename: batchalign_app::api::FileName::from(name.as_str()),
                content,
            });
        } else {
            media_names.push(name.clone());
        }
    }

    Ok((payloads, media_names))
}

/// Read a lexicon CSV and inject as MWT data into typed options.
pub(super) fn inject_lexicon(
    opts: &mut batchalign_app::options::CommandOptions,
    lexicon: Option<&str>,
) -> Result<(), CliError> {
    let Some(path) = lexicon else {
        return Ok(());
    };
    let path = path.trim();
    if path.is_empty() {
        return Ok(());
    }

    let content = std::fs::read_to_string(path)?;
    let mwt = &mut opts.common_mut().mwt;
    for line in content.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            let key = parts[0].trim().to_string();
            let vals: Vec<String> = parts[1..].iter().map(|s| s.trim().to_string()).collect();
            mwt.insert(key, vals);
        }
    }
    Ok(())
}

/// Query /health on each server and return capacity weights.
#[allow(dead_code)] // Fleet support disabled in this version
pub(super) async fn query_server_weights(client: &BatchalignClient, urls: &[String]) -> Vec<i64> {
    let mut weights = Vec::with_capacity(urls.len());
    for url in urls {
        let w = match client.health_check(url).await {
            Ok(h) => h.workers_available.max(1),
            Err(_) => 1,
        };
        weights.push(w);
    }
    weights
}

/// Distribute files across servers proportional to capacity weights.
#[allow(dead_code)] // Fleet support disabled in this version
pub(super) fn distribute_files_weighted(
    files: &[PathBuf],
    outputs: &[PathBuf],
    weights: &[i64],
) -> Vec<Vec<(PathBuf, PathBuf)>> {
    let n = weights.len();
    if n == 0 {
        return vec![];
    }

    let total_weight: i64 = weights.iter().sum();
    let mut allocations = vec![0usize; n];
    let mut remaining = files.len();

    for i in 0..n {
        if i == n - 1 {
            allocations[i] = remaining;
        } else {
            let count = ((files.len() as f64) * (weights[i] as f64) / (total_weight as f64)).round()
                as usize;
            let count = count.min(remaining);
            allocations[i] = count;
            remaining -= count;
        }
    }

    let mut buckets = Vec::with_capacity(n);
    let mut offset = 0;
    for alloc in allocations {
        let bucket: Vec<_> = files[offset..offset + alloc]
            .iter()
            .zip(outputs[offset..offset + alloc].iter())
            .map(|(f, o)| (f.clone(), o.clone()))
            .collect();
        buckets.push(bucket);
        offset += alloc;
    }

    buckets
}

/// Print a structured failure summary.
pub(super) fn print_failure_summary(
    errors: &[FileErrorDetail],
    total_files: u64,
    out_dir: &Path,
) {
    if errors.is_empty() {
        eprintln!(
            "\nAll done! {total_files} file(s) written to {}",
            out_dir.display()
        );
        return;
    }

    let succeeded = total_files - errors.len() as u64;
    let bar = "\u{2501}".repeat(50);
    eprintln!("\n{bar}");
    eprintln!(
        "  RESULTS: {succeeded} succeeded, {} failed (of {total_files} files)",
        errors.len()
    );
    eprintln!("{bar}");

    for error in errors {
        let first_line = error.message.split('\n').next().unwrap_or("unknown error");
        let filename = error.filename.as_ref();
        eprintln!("  \u{2717} {filename}: {first_line}");
    }

    eprintln!("{bar}");
    eprintln!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use batchalign_app::api::{CommandName, FileStatusEntry, JobId, LanguageCode3, LanguageSpec};
    use batchalign_app::options::{CommandOptions, CommonOptions, MorphotagOptions};
    use batchalign_app::ReleasedCommand;

    fn test_job_info(status: JobStatus, error: Option<&str>) -> JobInfo {
        JobInfo {
            job_id: JobId::from("job123"),
            status,
            command: CommandName::from("benchmark"),
            options: CommandOptions::Morphotag(MorphotagOptions {
                common: CommonOptions::default(),
                retokenize: false,
                skipmultilang: false,
                merge_abbrev: false.into(),
            }),
            lang: LanguageSpec::Resolved(LanguageCode3::eng()),
            source_dir: "/tmp/in".into(),
            total_files: 1,
            completed_files: 0,
            current_file: None,
            error: error.map(str::to_string),
            file_statuses: vec![FileStatusEntry {
                filename: "clip.cha".into(),
                status: FileStatusKind::Error,
                error: Some("worker failed".into()),
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
            }],
            submitted_at: None,
            submitted_by: None,
            submitted_by_name: None,
            completed_at: None,
            duration_s: None,
            next_eligible_at: None,
            num_workers: None,
            active_lease: None,
        }
    }

    #[test]
    fn distribute_weighted_proportional() {
        let files: Vec<PathBuf> = (0..10).map(|i| PathBuf::from(format!("f{i}"))).collect();
        let outputs: Vec<PathBuf> = (0..10).map(|i| PathBuf::from(format!("o{i}"))).collect();
        let weights = vec![3, 1]; // 75/25 split

        let buckets = distribute_files_weighted(&files, &outputs, &weights);
        assert_eq!(buckets.len(), 2);
        // First server should get ~8, second ~2 (3/(3+1) * 10 = 7.5 → 8)
        assert!(buckets[0].len() >= 7);
        assert_eq!(buckets[0].len() + buckets[1].len(), 10);
    }

    #[test]
    fn distribute_weighted_single_server() {
        let files: Vec<PathBuf> = (0..5).map(|i| PathBuf::from(format!("f{i}"))).collect();
        let outputs: Vec<PathBuf> = (0..5).map(|i| PathBuf::from(format!("o{i}"))).collect();
        let weights = vec![4];

        let buckets = distribute_files_weighted(&files, &outputs, &weights);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].len(), 5);
    }

    #[test]
    fn dashboard_auto_open_enabled_when_cli_allows_and_env_clear() {
        assert!(dashboard_auto_open_enabled(true, false));
    }

    #[test]
    fn dashboard_auto_open_disabled_when_cli_disables() {
        assert!(!dashboard_auto_open_enabled(false, false));
    }

    #[test]
    fn dashboard_auto_open_disabled_by_env_backstop() {
        assert!(!dashboard_auto_open_enabled(true, true));
    }

    #[test]
    fn classify_cha_vs_media() {
        let dir = tempfile::tempdir().unwrap();
        let cha = dir.path().join("test.cha");
        std::fs::write(&cha, "@Begin\n@End\n").unwrap();

        let files = vec![cha, PathBuf::from("audio.mp3")];
        let names = vec!["test.cha".to_string(), "audio.mp3".to_string()];

        let (payloads, media) = classify_files(&files, &names).unwrap();
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].filename, "test.cha");
        assert_eq!(media, vec!["audio.mp3"]);
    }

    #[test]
    fn filter_avqi_keeps_only_cs_files() {
        let files = vec![
            PathBuf::from("sample.cs.wav"),
            PathBuf::from("sample.sv.wav"),
            PathBuf::from("other.CS.MP3"),
        ];
        let outputs = vec![PathBuf::from("a"), PathBuf::from("b"), PathBuf::from("c")];

        let (f, o) = filter_files_for_command(ReleasedCommand::Avqi, files, outputs);
        assert_eq!(f.len(), 2);
        assert_eq!(o.len(), 2);
        assert!(f[0].to_string_lossy().contains(".cs."));
        assert!(f[1].to_string_lossy().to_ascii_lowercase().contains(".cs."));
    }

    #[test]
    fn filter_compare_skips_gold_chat_companions() {
        let files = vec![
            PathBuf::from("sample.cha"),
            PathBuf::from("sample.gold.cha"),
            PathBuf::from("other.GOLD.CHA"),
            PathBuf::from("other.cha"),
        ];
        let outputs = vec![
            PathBuf::from("a"),
            PathBuf::from("b"),
            PathBuf::from("c"),
            PathBuf::from("d"),
        ];

        let (f, o) = filter_files_for_command(ReleasedCommand::Compare, files, outputs);
        assert_eq!(f.len(), 2);
        assert_eq!(o.len(), 2);
        assert!(f.iter().all(|path| {
            !path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_ascii_lowercase()
                .ends_with(".gold.cha")
        }));
    }

    #[test]
    fn inject_lexicon_csv() {
        use batchalign_app::options::{CommandOptions, CommonOptions, MorphotagOptions};

        let dir = tempfile::tempdir().unwrap();
        let lex = dir.path().join("lex.csv");
        std::fs::write(&lex, "gonna,going,to\nwanna,want,to\n").unwrap();

        let mut opts = CommandOptions::Morphotag(MorphotagOptions {
            common: CommonOptions::default(),
            retokenize: false,
            skipmultilang: false,
            merge_abbrev: false.into(),
        });
        inject_lexicon(&mut opts, Some(lex.to_str().unwrap())).unwrap();

        let mwt = &opts.common().mwt;
        assert_eq!(mwt.len(), 2);
        assert!(mwt.contains_key("gonna"));
        assert_eq!(mwt["gonna"], vec!["going", "to"]);
    }

    #[test]
    fn capability_check_allows_test_echo() {
        use batchalign_app::ReleasedCommand;

        let caps = vec!["test-echo".to_string()];
        assert!(super::super::server_supports_command(
            &caps,
            ReleasedCommand::Morphotag
        ));
    }

    #[test]
    fn capability_check_rejects_missing_command() {
        use batchalign_app::ReleasedCommand;

        let caps = vec!["align".to_string(), "transcribe".to_string()];
        assert!(!super::super::server_supports_command(
            &caps,
            ReleasedCommand::Morphotag
        ));
    }

    #[test]
    fn distribute_weighted_empty_servers() {
        let files: Vec<PathBuf> = vec![PathBuf::from("f0")];
        let outputs: Vec<PathBuf> = vec![PathBuf::from("o0")];
        let weights: Vec<i64> = vec![];
        let buckets = distribute_files_weighted(&files, &outputs, &weights);
        assert!(buckets.is_empty());
    }

    #[test]
    fn distribute_weighted_single_file_many_servers() {
        let files = vec![PathBuf::from("f0")];
        let outputs = vec![PathBuf::from("o0")];
        let weights = vec![1, 1, 1];
        let buckets = distribute_files_weighted(&files, &outputs, &weights);
        assert_eq!(buckets.len(), 3);
        // With 1 file and 3 equal-weight servers:
        // round(1 * 1/3) = 0 for first two, remainder = 1 for last
        let total: usize = buckets.iter().map(|b| b.len()).sum();
        assert_eq!(total, 1);
        assert_eq!(buckets[2].len(), 1); // remainder goes to last
    }

    #[test]
    fn distribute_weighted_equal_weights() {
        let files: Vec<PathBuf> = (0..6).map(|i| PathBuf::from(format!("f{i}"))).collect();
        let outputs: Vec<PathBuf> = (0..6).map(|i| PathBuf::from(format!("o{i}"))).collect();
        let weights = vec![1, 1, 1];
        let buckets = distribute_files_weighted(&files, &outputs, &weights);
        assert_eq!(buckets.len(), 3);
        assert_eq!(buckets[0].len(), 2);
        assert_eq!(buckets[1].len(), 2);
        assert_eq!(buckets[2].len(), 2);
    }

    #[test]
    fn inject_lexicon_missing_file() {
        use batchalign_app::options::{CommandOptions, CommonOptions, MorphotagOptions};

        let mut opts = CommandOptions::Morphotag(MorphotagOptions {
            common: CommonOptions::default(),
            retokenize: false,
            skipmultilang: false,
            merge_abbrev: false.into(),
        });
        let result = inject_lexicon(&mut opts, Some("/nonexistent/lexicon.csv"));
        assert!(result.is_err());
    }

    #[test]
    fn inject_lexicon_empty_path() {
        let mut opts = CommandOptions::Morphotag(MorphotagOptions {
            common: CommonOptions::default(),
            retokenize: false,
            skipmultilang: false,
            merge_abbrev: false.into(),
        });
        inject_lexicon(&mut opts, Some("  ")).unwrap();
        // Empty/whitespace path is a no-op
        assert!(opts.common().mwt.is_empty());
    }

    #[test]
    fn finish_terminal_job_accepts_clean_completed_job() {
        let mut info = test_job_info(JobStatus::Completed, None);
        info.file_statuses[0].status = FileStatusKind::Done;
        info.file_statuses[0].error = None;
        info.completed_files = 1;
        let out_dir = tempfile::tempdir().unwrap();

        let result = finish_terminal_job(&info, &[], 1, out_dir.path());

        assert!(result.is_ok());
    }

    #[test]
    fn finish_terminal_job_rejects_failed_job_status() {
        let info = test_job_info(JobStatus::Failed, Some("worker pool exploded"));
        let out_dir = tempfile::tempdir().unwrap();

        let result = finish_terminal_job(&info, &[], 1, out_dir.path());

        match result {
            Err(CliError::JobFailed {
                job_id,
                status,
                detail,
            }) => {
                assert_eq!(job_id, "job123");
                assert_eq!(status, "failed");
                assert_eq!(detail, "worker pool exploded");
            }
            other => panic!("expected JobFailed, got {other:?}"),
        }
    }

    #[test]
    fn finish_terminal_job_rejects_completed_job_with_file_errors() {
        let info = test_job_info(JobStatus::Completed, None);
        let out_dir = tempfile::tempdir().unwrap();
        let errors = vec![FileErrorDetail::new("clip.cha", "decoder failed\ntrace")];

        let result = finish_terminal_job(&info, &errors, 1, out_dir.path());

        match result {
            Err(CliError::JobFailed {
                job_id,
                status,
                detail,
            }) => {
                assert_eq!(job_id, "job123");
                assert_eq!(status, "completed");
                assert_eq!(detail, "clip.cha: decoder failed");
            }
            other => panic!("expected JobFailed, got {other:?}"),
        }
    }
}
