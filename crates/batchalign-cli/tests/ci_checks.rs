//! CI hygiene checks ported from Python scripts.
//!
//! These tests replace:
//! - `scripts/check_cli_version_sync.py`
//! - `scripts/check_legacy_terms.py`
//! - `scripts/check_retired_packages.py`

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Resolve the batchalign3 repo root.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent() // crates/
        .unwrap()
        .parent() // batchalign3/
        .unwrap()
        .to_path_buf()
}

// ---------------------------------------------------------------------------
// CLI version sync (replaces check_cli_version_sync.py)
// ---------------------------------------------------------------------------

#[test]
fn cli_version_sync() {
    let root = repo_root();

    let pyproject_path = root.join("pyproject.toml");
    let pyproject_str = std::fs::read_to_string(&pyproject_path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", pyproject_path.display()));
    let pyproject: toml::Value = toml::from_str(&pyproject_str).unwrap();
    let py_version = pyproject["project"]["version"]
        .as_str()
        .expect("pyproject.toml missing [project].version");

    let cargo_path = root.join("crates/batchalign-cli/Cargo.toml");
    let cargo_str = std::fs::read_to_string(&cargo_path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", cargo_path.display()));
    let cargo: toml::Value = toml::from_str(&cargo_str).unwrap();
    let cargo_version = cargo["package"]["version"]
        .as_str()
        .expect("Cargo.toml missing [package].version");

    assert_eq!(
        py_version, cargo_version,
        "Version mismatch: pyproject.toml={py_version} != batchalign-cli/Cargo.toml={cargo_version}"
    );
}

// ---------------------------------------------------------------------------
// Legacy terms (replaces check_legacy_terms.py)
// ---------------------------------------------------------------------------

const SCAN_SUFFIXES: &[&str] = &[
    ".css", ".js", ".jsx", ".md", ".py", ".rs", ".toml", ".ts", ".tsx", ".yaml", ".yml",
];

const SKIP_DIRS: &[&str] = &[
    ".git",
    ".venv",
    ".venv-314t",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
];

struct BannedPattern {
    pattern: &'static str,
    reason: &'static str,
}

const BANNED_PATTERNS: &[BannedPattern] = &[
    BannedPattern {
        pattern: "batchalign-next",
        reason: "retired command name",
    },
    BannedPattern {
        pattern: "batchalign_next",
        reason: "retired package/module name",
    },
    BannedPattern {
        pattern: "/opt/python/bin/python",
        reason: "hardcoded interpreter path",
    },
    BannedPattern {
        pattern: "batchalign.cli",
        reason: "retired Python CLI package path",
    },
    BannedPattern {
        pattern: "pip install 'batchalign-hk-plugin",
        reason: "retired HK plugin package install guidance",
    },
    BannedPattern {
        pattern: "pip install \"batchalign-hk-plugin",
        reason: "retired HK plugin package install guidance",
    },
    BannedPattern {
        pattern: "batchalign.providers.models",
        reason: "nonexistent public module path",
    },
    BannedPattern {
        pattern: "plugin discovery still happens in `batchalign.plugins`",
        reason: "entry-point plugin discovery was removed",
    },
    BannedPattern {
        pattern: "Entry-point plugin system (`batchalign.plugins`)",
        reason: "current release has no public entry-point plugin system",
    },
    BannedPattern {
        pattern: "batchalign-hk-plugin/common.py",
        reason: "retired HK plugin source path in current docs",
    },
    BannedPattern {
        pattern: "batchalign-hk-plugin/cantonese_fa.py",
        reason: "retired HK plugin source path in current docs",
    },
];

const DOC_BANNED: &[BannedPattern] = &[
    BannedPattern {
        pattern: "batchalign2",
        reason: "legacy repository/name in active docs",
    },
    BannedPattern {
        pattern: "BA2-usage.pdf",
        reason: "historical Batchalign2 PDF linked from active docs",
    },
    BannedPattern {
        pattern: "BA2-cleanup.pdf",
        reason: "historical Batchalign2 PDF linked from active docs",
    },
    BannedPattern {
        pattern: "--whisper-oai",
        reason: "retired public CLI flag form; use --asr-engine whisper-oai",
    },
    BannedPattern {
        pattern: concat!("rust", "-next/"),
        reason: "retired public workspace path",
    },
    BannedPattern {
        pattern: "worker.py",
        reason: "retired Python worker module path in active docs",
    },
    BannedPattern {
        pattern: "test_worker.py",
        reason: "retired Python test file path in active docs",
    },
];

const DOC_ACTIVE_PREFIXES: &[&str] = &[
    "README.md",
    "examples/launchd.plist",
    "examples/server.yaml",
    "book/src/introduction.md",
    "book/src/user-guide/",
    "book/src/developer/building.md",
    "book/src/developer/testing.md",
];

/// Line substrings that are intentionally allowed (migration logic).
fn allowlist() -> HashMap<&'static str, Vec<&'static str>> {
    HashMap::from([(
        "batchalign/runtime.py",
        vec![
            "One-time migration: ~/.batchalign-next",
            "old = Path.home() / \".batchalign-next\"",
        ],
    )])
}

fn should_scan(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let dotted = format!(".{ext}");
    if !SCAN_SUFFIXES.contains(&dotted.as_str()) {
        return false;
    }
    !path
        .components()
        .any(|c| SKIP_DIRS.contains(&c.as_os_str().to_str().unwrap_or("")))
}

/// Paths to exclude from the legacy term scan (this test file contains the
/// banned strings as literal test data).
const SELF_EXCLUDE: &str = "crates/batchalign-cli/tests/ci_checks.rs";

fn scan_files(root: &Path) -> Vec<PathBuf> {
    let active_paths: Vec<PathBuf> = [
        "README.md",
        ".github/workflows",
        "batchalign",
        "crates",
        "frontend/src",
        "book/src/introduction.md",
        "book/src/migration",
        "book/src/architecture",
        "book/src/reference",
        "book/src/user-guide",
        "book/src/developer",
    ]
    .iter()
    .map(|p| root.join(p))
    .collect();

    let self_path = root.join(SELF_EXCLUDE);

    let mut files = Vec::new();
    for base in &active_paths {
        if !base.exists() {
            continue;
        }
        if base.is_file() {
            if should_scan(base) && *base != self_path {
                files.push(base.clone());
            }
            continue;
        }
        if base.is_dir() {
            for entry in walkdir(base) {
                if should_scan(&entry) && entry != self_path {
                    files.push(entry);
                }
            }
        }
    }
    files.sort();
    files
}

/// Simple recursive directory walk (no external crate needed for tests).
fn walkdir(dir: &Path) -> Vec<PathBuf> {
    let mut result = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().unwrap_or_default().to_str().unwrap_or("");
                if !SKIP_DIRS.contains(&name) {
                    result.extend(walkdir(&path));
                }
            } else {
                result.push(path);
            }
        }
    }
    result
}

/// Check if a pattern match is a word boundary match (not embedded in a larger word).
fn is_word_boundary_match(line: &str, start: usize, end: usize) -> bool {
    let before_ok = start == 0
        || !line.as_bytes()[start - 1].is_ascii_alphanumeric()
            && line.as_bytes()[start - 1] != b'_';
    let after_ok = end >= line.len()
        || !line.as_bytes()[end].is_ascii_alphanumeric() && line.as_bytes()[end] != b'_';
    before_ok && after_ok
}

#[test]
fn legacy_terms_absent() {
    let root = repo_root();
    let allow = allowlist();
    let mut failures = Vec::new();

    for path in scan_files(&root) {
        let rel = path
            .strip_prefix(&root)
            .unwrap()
            .to_str()
            .unwrap_or("")
            .to_string();
        let allow_subs = allow.get(rel.as_str()).cloned().unwrap_or_default();

        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => continue,
        };

        for (line_no, line) in text.lines().enumerate() {
            let line_no = line_no + 1;

            for bp in BANNED_PATTERNS {
                if let Some(start) = line.find(bp.pattern) {
                    let end = start + bp.pattern.len();
                    if !is_word_boundary_match(line, start, end) {
                        continue;
                    }
                    if allow_subs.iter().any(|s| line.contains(s)) {
                        continue;
                    }
                    failures.push(format!(
                        "{rel}:{line_no}: `{}` ({})\n  {}",
                        bp.pattern,
                        bp.reason,
                        line.trim()
                    ));
                }
            }

            if DOC_ACTIVE_PREFIXES.iter().any(|p| rel.starts_with(p)) {
                for bp in DOC_BANNED {
                    if let Some(start) = line.find(bp.pattern) {
                        let end = start + bp.pattern.len();
                        if !is_word_boundary_match(line, start, end) {
                            continue;
                        }
                        failures.push(format!(
                            "{rel}:{line_no}: `{}` ({})\n  {}",
                            bp.pattern,
                            bp.reason,
                            line.trim()
                        ));
                    }
                }

                if line.contains("uv tool install batchalign3-cli") {
                    failures.push(format!(
                        "{rel}:{line_no}: `uv tool install batchalign3-cli` (retired: CLI is now part of the batchalign3 package)\n  {}",
                        line.trim()
                    ));
                }
            }
        }
    }

    if !failures.is_empty() {
        let msg = failures.join("\n- ");
        panic!("Legacy term check failed:\n- {msg}");
    }
}

// ---------------------------------------------------------------------------
// Retired packages (replaces check_retired_packages.py)
// ---------------------------------------------------------------------------

/// Verify that retired Python package paths have zero tracked files.
///
/// These packages were fully deleted in the radical Python simplification
/// (commit 6766d0e8). No files should re-appear under them.
#[test]
fn retired_packages_stay_deleted() {
    let root = repo_root();

    let retired_paths = ["batchalign/cli", "batchalign/serve"];
    let mut failures = Vec::new();

    for pathspec in &retired_paths {
        let output = Command::new("git")
            .args(["ls-files", pathspec])
            .current_dir(&root)
            .output()
            .expect("git ls-files failed");

        let tracked: Vec<String> = String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && root.join(l).exists())
            .collect();

        for file in &tracked {
            failures.push(format!(
                "{file}: unexpected tracked file under retired package path `{pathspec}`"
            ));
        }
    }

    if !failures.is_empty() {
        let msg = failures.join("\n- ");
        panic!("Retired package boundary check failed:\n- {msg}");
    }
}
