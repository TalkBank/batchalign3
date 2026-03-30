//! `batchalign3 doctor` — pre-flight diagnostic for the worker pipeline.
//!
//! Spawns a test worker, sends known inputs through the morphosyntax
//! pipeline, and validates the output structure. Catches machine-specific
//! issues (stale models, missing processors, MWT quirks) before they
//! become production failures.

use crate::args::DoctorArgs;
use crate::error::CliError;
use crate::python::resolve_python_executable;

use std::io::{BufRead, Write};
use std::process::{Command, Stdio};
use std::time::Instant;

/// Result of a single diagnostic check.
#[derive(Debug, serde::Serialize)]
struct CheckResult {
    name: String,
    status: CheckStatus,
    detail: String,
    duration_ms: u64,
}

/// Outcome of a diagnostic check.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum CheckStatus {
    Pass,
    Fail,
    Skip,
}

impl std::fmt::Display for CheckStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::Fail => write!(f, "FAIL"),
            Self::Skip => write!(f, "SKIP"),
        }
    }
}

/// Run the doctor command.
pub async fn run(args: &DoctorArgs) -> Result<(), CliError> {
    let mut results: Vec<CheckResult> = Vec::new();
    let python = args
        .python
        .clone()
        .unwrap_or_else(resolve_python_executable);

    // --- Check 1: Python availability ---
    let start = Instant::now();
    let python_check = Command::new(&python)
        .args(["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"])
        .output();

    results.push(match python_check {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            CheckResult {
                name: "python".into(),
                status: CheckStatus::Pass,
                detail: format!("{python} -> Python {version}"),
                duration_ms: start.elapsed().as_millis() as u64,
            }
        }
        Ok(output) => CheckResult {
            name: "python".into(),
            status: CheckStatus::Fail,
            detail: format!(
                "Python exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            duration_ms: start.elapsed().as_millis() as u64,
        },
        Err(e) => CheckResult {
            name: "python".into(),
            status: CheckStatus::Fail,
            detail: format!("Cannot spawn {python}: {e}"),
            duration_ms: start.elapsed().as_millis() as u64,
        },
    });

    // --- Check 2: Worker module importable ---
    let start = Instant::now();
    let import_check = Command::new(&python)
        .args(["-c", "from batchalign.worker import main; print('ok')"])
        .output();

    results.push(match import_check {
        Ok(output) if output.status.success() => CheckResult {
            name: "worker_import".into(),
            status: CheckStatus::Pass,
            detail: "batchalign.worker importable".into(),
            duration_ms: start.elapsed().as_millis() as u64,
        },
        Ok(output) => CheckResult {
            name: "worker_import".into(),
            status: CheckStatus::Fail,
            detail: format!(
                "Import failed: {}",
                String::from_utf8_lossy(&output.stderr)
                    .trim()
                    .chars()
                    .take(200)
                    .collect::<String>()
            ),
            duration_ms: start.elapsed().as_millis() as u64,
        },
        Err(e) => CheckResult {
            name: "worker_import".into(),
            status: CheckStatus::Fail,
            detail: format!("Cannot spawn: {e}"),
            duration_ms: start.elapsed().as_millis() as u64,
        },
    });

    // --- Check 3: Worker ready signal (test-echo mode) ---
    let start = Instant::now();
    let echo_check = spawn_worker_and_check_ready(
        &python,
        &[
            "--test-echo",
            "--task",
            "morphosyntax",
            "--lang",
            &args.lang,
        ],
    );
    results.push(match echo_check {
        Ok(detail) => CheckResult {
            name: "worker_ready_echo".into(),
            status: CheckStatus::Pass,
            detail,
            duration_ms: start.elapsed().as_millis() as u64,
        },
        Err(detail) => CheckResult {
            name: "worker_ready_echo".into(),
            status: CheckStatus::Fail,
            detail,
            duration_ms: start.elapsed().as_millis() as u64,
        },
    });

    // --- Check 4: Real morphosyntax worker (loads Stanza model) ---
    let start = Instant::now();
    let morpho_check = spawn_worker_and_send_batch(
        &python,
        &args.lang,
        &[
            // English test sentence
            vec!["the", "dog", "runs"],
            // Contraction (MWT candidate)
            vec!["I", "dont", "know"],
            // Single letter (edge case)
            vec!["a"],
        ],
    );
    results.push(match morpho_check {
        Ok(detail) => CheckResult {
            name: "morphosyntax_smoke".into(),
            status: CheckStatus::Pass,
            detail,
            duration_ms: start.elapsed().as_millis() as u64,
        },
        Err(detail) => CheckResult {
            name: "morphosyntax_smoke".into(),
            status: CheckStatus::Fail,
            detail,
            duration_ms: start.elapsed().as_millis() as u64,
        },
    });

    // --- Check 5: Memory ---
    let mem_info = sysinfo::System::new_all();
    let total_mb = mem_info.total_memory() / (1024 * 1024);
    let available_mb = mem_info.available_memory() / (1024 * 1024);
    results.push(CheckResult {
        name: "memory".into(),
        status: if available_mb >= 4096 {
            CheckStatus::Pass
        } else {
            CheckStatus::Fail
        },
        detail: format!("{available_mb} MB available / {total_mb} MB total"),
        duration_ms: 0,
    });

    // --- Output ---
    let any_fail = results
        .iter()
        .any(|r| matches!(r.status, CheckStatus::Fail));

    match args.format {
        crate::args::DoctorFormat::Human => {
            for r in &results {
                let icon = match r.status {
                    CheckStatus::Pass => "\u{2713}",
                    CheckStatus::Fail => "\u{2717}",
                    CheckStatus::Skip => "-",
                };
                eprintln!(
                    "  {icon} [{:>4}] {:25} {} ({} ms)",
                    r.status, r.name, r.detail, r.duration_ms
                );
            }
            if any_fail {
                eprintln!(
                    "\nSome checks FAILED. Fix the issues above before using this machine for production."
                );
            } else {
                eprintln!("\nAll checks passed.");
            }
        }
        crate::args::DoctorFormat::Json => {
            let json = serde_json::to_string_pretty(&results).map_err(|e| {
                CliError::InvalidArgument(format!("JSON serialization failed: {e}"))
            })?;
            println!("{json}");
        }
    }

    if any_fail {
        Err(CliError::InvalidArgument("doctor checks failed".into()))
    } else {
        Ok(())
    }
}

/// Spawn a worker and check it emits a valid ready signal.
fn spawn_worker_and_check_ready(python: &str, args: &[&str]) -> Result<String, String> {
    let mut cmd = Command::new(python);
    cmd.args(["-m", "batchalign.worker"]);
    cmd.args(args);
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| format!("Spawn failed: {e}"))?;
    let stdout = child.stdout.take().ok_or("No stdout")?;
    let reader = std::io::BufReader::new(stdout);

    let deadline = Instant::now() + std::time::Duration::from_secs(60);
    for line in reader.lines() {
        if Instant::now() > deadline {
            let _ = child.kill();
            return Err("Timeout (60s) waiting for ready signal".into());
        }
        let line = line.map_err(|e| format!("Read error: {e}"))?;
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
            if val.get("ready") == Some(&serde_json::Value::Bool(true)) {
                let pid = val.get("pid").and_then(|v| v.as_u64()).unwrap_or(0);
                // Send shutdown
                if let Some(mut stdin) = child.stdin.take() {
                    let _ = writeln!(stdin, r#"{{"op":"shutdown"}}"#);
                }
                let _ = child.wait();
                return Ok(format!("Ready signal received (pid {pid})"));
            }
        }
    }
    let _ = child.kill();
    Err("Worker exited without ready signal".into())
}

/// Spawn a real morphosyntax worker, send test sentences, validate output.
fn spawn_worker_and_send_batch(
    python: &str,
    lang: &str,
    test_sentences: &[Vec<&str>],
) -> Result<String, String> {
    let mut cmd = Command::new(python);
    cmd.args([
        "-m",
        "batchalign.worker",
        "--task",
        "morphosyntax",
        "--lang",
        lang,
    ]);
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| format!("Spawn failed: {e}"))?;
    let stdout = child.stdout.take().ok_or("No stdout")?;
    let mut stdin = child.stdin.take().ok_or("No stdin")?;
    let reader = std::io::BufReader::new(stdout);

    // Wait for ready
    let mut lines = reader.lines();
    let deadline = Instant::now() + std::time::Duration::from_secs(120);
    let mut ready = false;
    while let Some(Ok(line)) = lines.next() {
        if Instant::now() > deadline {
            let _ = child.kill();
            return Err("Timeout (120s) waiting for ready".into());
        }
        if line.contains("\"ready\"") && line.contains("true") {
            ready = true;
            break;
        }
    }
    if !ready {
        let _ = child.kill();
        return Err("Worker exited without ready signal".into());
    }

    // Build batch_infer request
    let items: Vec<serde_json::Value> = test_sentences
        .iter()
        .map(|words| {
            serde_json::json!({
                "words": words,
                "lang": lang,
                "retokenize": false,
            })
        })
        .collect();

    let request = serde_json::json!({
        "op": "batch_infer",
        "request": {
            "task": "morphosyntax",
            "lang": lang,
            "items": items,
        }
    });

    writeln!(stdin, "{}", serde_json::to_string(&request).unwrap())
        .map_err(|e| format!("Write failed: {e}"))?;
    stdin.flush().map_err(|e| format!("Flush failed: {e}"))?;

    // Read response
    let response_deadline = Instant::now() + std::time::Duration::from_secs(120);
    while let Some(Ok(line)) = lines.next() {
        if Instant::now() > response_deadline {
            let _ = child.kill();
            return Err("Timeout (120s) waiting for batch_infer response".into());
        }

        let val: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue, // skip noise
        };

        if val.get("op").and_then(|v| v.as_str()) == Some("error") {
            let err = val
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let _ = child.kill();
            return Err(format!("Worker error: {err}"));
        }

        if val.get("op").and_then(|v| v.as_str()) == Some("batch_infer") {
            // Validate response structure
            let results = val
                .pointer("/response/results")
                .and_then(|v| v.as_array())
                .ok_or("No results array in response")?;

            if results.len() != test_sentences.len() {
                let _ = child.kill();
                return Err(format!(
                    "Expected {} results, got {}",
                    test_sentences.len(),
                    results.len()
                ));
            }

            let mut total_words = 0usize;
            let mut missing_fields: Vec<String> = Vec::new();

            for (ri, result) in results.iter().enumerate() {
                let sents = result
                    .pointer("/result/raw_sentences")
                    .and_then(|v| v.as_array());

                if let Some(sents) = sents {
                    for (si, sent) in sents.iter().enumerate() {
                        if let Some(words) = sent.as_array() {
                            for (wi, word) in words.iter().enumerate() {
                                total_words += 1;
                                for field in ["text", "lemma", "upos", "deprel"] {
                                    if word.get(field).is_none()
                                        || word.get(field) == Some(&serde_json::Value::Null)
                                    {
                                        // Check if MWT range token (expected to lack some fields)
                                        let is_range = word
                                            .get("id")
                                            .and_then(|v| v.as_array())
                                            .is_some_and(|a| a.len() > 1);
                                        if !is_range || field == "text" {
                                            let text = word
                                                .get("text")
                                                .and_then(|v| v.as_str())
                                                .unwrap_or("?");
                                            missing_fields.push(format!(
                                                "result {ri} sent {si} word {wi} ('{text}'): missing {field}"
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else if let Some(err) = result.get("error").and_then(|v| v.as_str()) {
                    missing_fields.push(format!("result {ri}: worker error: {err}"));
                }
            }

            // Shutdown
            let _ = writeln!(stdin, r#"{{"op":"shutdown"}}"#);
            let _ = child.wait();

            if missing_fields.is_empty() {
                return Ok(format!(
                    "{} sentences, {total_words} words — all fields present",
                    test_sentences.len()
                ));
            } else {
                return Err(format!(
                    "{} field issues: {}",
                    missing_fields.len(),
                    missing_fields.join("; ")
                ));
            }
        }
    }

    let _ = child.kill();
    Err("Worker exited without batch_infer response".into())
}
