//! Audit wide Rust structs so field-bag growth stays explicit and reviewed.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Number of named fields that triggers a wide-struct audit entry.
const WIDE_STRUCT_THRESHOLD: usize = 10;

/// Audit classification for one intentionally wide struct.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WideStructDisposition {
    /// Flat only because clap, JSON, or similar boundary tooling wants it.
    BoundaryShim,
    /// Mirrors a transport, DB, or wire-format shape.
    TransportRecord,
    /// A real aggregate with acceptable cohesion for now.
    RealAggregate,
}

impl WideStructDisposition {
    /// Render a short human-readable label for test failures.
    fn label(self) -> &'static str {
        match self {
            Self::BoundaryShim => "boundary shim",
            Self::TransportRecord => "transport record",
            Self::RealAggregate => "real aggregate",
        }
    }
}

/// One reviewed wide-struct entry in the repo audit.
#[derive(Clone, Copy, Debug)]
struct WideStructAllowance {
    /// Repo-relative Rust path containing the struct.
    path: &'static str,
    /// Struct name as written in source.
    struct_name: &'static str,
    /// Maximum reviewed named-field count.
    max_fields: usize,
    /// Maximum reviewed boolean-field count.
    max_bool_fields: usize,
    /// Audit classification for this shape.
    disposition: WideStructDisposition,
    /// Brief rationale for why it currently exists in this form.
    reason: &'static str,
}

/// Parsed metadata for one named Rust struct.
#[derive(Clone, Debug, Eq, PartialEq)]
struct NamedStructInfo {
    /// Repo-relative file path.
    path: String,
    /// One-based declaration line.
    line: usize,
    /// Struct identifier.
    struct_name: String,
    /// Number of named fields.
    field_count: usize,
    /// Number of fields whose type includes `bool`.
    bool_field_count: usize,
}

/// Reviewed wide structs in `batchalign3`.
const WIDE_STRUCT_ALLOWANCES: &[WideStructAllowance] = &[
    WideStructAllowance {
        path: "crates/batchalign-app/src/db/schema.rs",
        struct_name: "JobRow",
        max_fields: 27,
        max_bool_fields: 2,
        disposition: WideStructDisposition::TransportRecord,
        reason: "SQL row shape at the persistence boundary",
    },
    WideStructAllowance {
        path: "crates/batchalign-app/src/types/response.rs",
        struct_name: "HealthResponse",
        max_fields: 24,
        max_bool_fields: 3,
        disposition: WideStructDisposition::TransportRecord,
        reason: "HTTP health snapshot exposed as one response payload",
    },
    WideStructAllowance {
        path: "crates/batchalign-cli/src/args/global_opts.rs",
        struct_name: "GlobalOpts",
        max_fields: 26,
        max_bool_fields: 18,
        disposition: WideStructDisposition::BoundaryShim,
        reason: "flat clap surface that should convert immediately into typed policies",
    },
    WideStructAllowance {
        path: "crates/batchalign-app/src/db/insert.rs",
        struct_name: "NewJobRecord",
        max_fields: 19,
        max_bool_fields: 2,
        disposition: WideStructDisposition::TransportRecord,
        reason: "DB insert payload rather than interior runtime shape",
    },
    WideStructAllowance {
        path: "crates/batchalign-app/src/types/response.rs",
        struct_name: "JobInfo",
        max_fields: 19,
        max_bool_fields: 0,
        disposition: WideStructDisposition::TransportRecord,
        reason: "API response model for one job",
    },
    WideStructAllowance {
        path: "crates/batchalign-cli/src/args/commands.rs",
        struct_name: "TranscribeArgs",
        max_fields: 17,
        max_bool_fields: 10,
        disposition: WideStructDisposition::BoundaryShim,
        reason: "clap boundary type that still needs immediate interior conversion",
    },
    WideStructAllowance {
        path: "crates/batchalign-cli/src/args/commands.rs",
        struct_name: "AlignArgs",
        max_fields: 17,
        max_bool_fields: 11,
        disposition: WideStructDisposition::BoundaryShim,
        reason: "clap boundary type with several behavior switches",
    },
    WideStructAllowance {
        path: "crates/batchalign-app/src/types/config.rs",
        struct_name: "ServerConfig",
        max_fields: 16,
        max_bool_fields: 2,
        disposition: WideStructDisposition::RealAggregate,
        reason: "owned server configuration aggregate",
    },
    WideStructAllowance {
        path: "crates/batchalign-app/src/types/response.rs",
        struct_name: "JobListItem",
        max_fields: 16,
        max_bool_fields: 0,
        disposition: WideStructDisposition::TransportRecord,
        reason: "compact API list-view payload for jobs",
    },
    WideStructAllowance {
        path: "crates/batchalign-app/src/types/request.rs",
        struct_name: "JobSubmission",
        max_fields: 15,
        max_bool_fields: 2,
        disposition: WideStructDisposition::TransportRecord,
        reason: "submission payload at the HTTP boundary",
    },
    WideStructAllowance {
        path: "crates/batchalign-app/src/store/mod.rs",
        struct_name: "FileStatus",
        max_fields: 14,
        max_bool_fields: 0,
        disposition: WideStructDisposition::RealAggregate,
        reason: "cohesive per-file lifecycle state",
    },
    WideStructAllowance {
        path: "crates/batchalign-app/src/transcribe.rs",
        struct_name: "TranscribeOptions",
        max_fields: 11,
        max_bool_fields: 5,
        disposition: WideStructDisposition::RealAggregate,
        reason: "owned transcribe orchestration policy carried through the Rust pipeline",
    },
    WideStructAllowance {
        path: "crates/batchalign-cli/src/args/commands.rs",
        struct_name: "BenchmarkArgs",
        max_fields: 14,
        max_bool_fields: 7,
        disposition: WideStructDisposition::BoundaryShim,
        reason: "clap boundary type for benchmark command wiring",
    },
    WideStructAllowance {
        path: "crates/batchalign-app/src/types/response.rs",
        struct_name: "FileStatusEntry",
        max_fields: 14,
        max_bool_fields: 0,
        disposition: WideStructDisposition::TransportRecord,
        reason: "HTTP response model for file status details",
    },
    WideStructAllowance {
        path: "crates/batchalign-app/src/db/schema.rs",
        struct_name: "AttemptRow",
        max_fields: 12,
        max_bool_fields: 0,
        disposition: WideStructDisposition::TransportRecord,
        reason: "SQL row shape for durable attempts",
    },
    WideStructAllowance {
        path: "crates/batchalign-app/src/types/scheduling.rs",
        struct_name: "AttemptRecord",
        max_fields: 12,
        max_bool_fields: 0,
        disposition: WideStructDisposition::RealAggregate,
        reason: "domain-level durable attempt record",
    },
    WideStructAllowance {
        path: "crates/batchalign-app/src/types/worker_v2.rs",
        struct_name: "AvqiResultV2",
        max_fields: 11,
        max_bool_fields: 1,
        disposition: WideStructDisposition::TransportRecord,
        reason: "raw AVQI metric payload from the Python model host",
    },
    WideStructAllowance {
        path: "crates/batchalign-app/src/runner/dispatch/fa_pipeline.rs",
        struct_name: "FaFileContext",
        max_fields: 11,
        max_bool_fields: 1,
        disposition: WideStructDisposition::RealAggregate,
        reason: "per-file FA context carrying job snapshot, services, and UTR config",
    },
    WideStructAllowance {
        path: "crates/batchalign-chat-ops/src/nlp/types.rs",
        struct_name: "UdWord",
        max_fields: 10,
        max_bool_fields: 0,
        disposition: WideStructDisposition::TransportRecord,
        reason: "mirrors one external NLP word record",
    },
    WideStructAllowance {
        path: "crates/batchalign-cli/src/args/commands.rs",
        struct_name: "BenchArgs",
        max_fields: 10,
        max_bool_fields: 4,
        disposition: WideStructDisposition::BoundaryShim,
        reason: "clap boundary type for bench orchestration",
    },
];

/// Resolve the repo root from the test crate.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("tests live under crates/")
        .parent()
        .expect("repo root lives above crates/")
        .to_path_buf()
}

/// Return the Rust source roots covered by this audit.
fn rust_scan_roots(root: &Path) -> Vec<PathBuf> {
    ["crates", "pyo3"]
        .iter()
        .map(|relative| root.join(relative))
        .collect()
}

/// Recursively walk one directory without pulling in extra test dependencies.
fn walkdir(dir: &Path) -> Vec<PathBuf> {
    let mut result = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or("");
                if !matches!(
                    name,
                    ".git" | "target" | "node_modules" | "dist" | "__pycache__"
                ) {
                    result.extend(walkdir(&path));
                }
            } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
                result.push(path);
            }
        }
    }
    result
}

/// Parse all named Rust structs under the audit roots.
fn scan_named_structs(root: &Path) -> Vec<NamedStructInfo> {
    let mut structs = Vec::new();

    for base in rust_scan_roots(root) {
        if !base.exists() {
            continue;
        }
        for path in walkdir(&base) {
            let relative = path
                .strip_prefix(root)
                .expect("scan path should be inside repo")
                .to_string_lossy()
                .into_owned();
            let text = match std::fs::read_to_string(&path) {
                Ok(text) => text,
                Err(_) => continue,
            };
            structs.extend(parse_named_structs_in_file(&relative, &text));
        }
    }

    structs.sort_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then(left.struct_name.cmp(&right.struct_name))
    });
    structs
}

/// Parse named structs from one Rust source file using a lightweight line scan.
fn parse_named_structs_in_file(relative_path: &str, text: &str) -> Vec<NamedStructInfo> {
    let lines: Vec<&str> = text.lines().collect();
    let mut result = Vec::new();
    let mut index = 0;

    while index < lines.len() {
        let line = lines[index].trim();
        let Some(struct_name) = struct_name_from_declaration(line) else {
            index += 1;
            continue;
        };

        let mut depth = brace_delta(line);
        let mut field_count = 0;
        let mut bool_field_count = 0;
        let start_line = index + 1;
        index += 1;

        while index < lines.len() && depth > 0 {
            let current = lines[index];
            let trimmed = current.trim();
            if depth == 1 && is_named_field(trimmed) {
                field_count += 1;
                if field_type(trimmed).is_some_and(|value| value.contains("bool")) {
                    bool_field_count += 1;
                }
            }
            depth += brace_delta(current);
            index += 1;
        }

        result.push(NamedStructInfo {
            path: relative_path.to_string(),
            line: start_line,
            struct_name,
            field_count,
            bool_field_count,
        });
    }

    result
}

/// Return the struct name if the line starts a named-struct declaration.
fn struct_name_from_declaration(line: &str) -> Option<String> {
    let declaration = line
        .strip_prefix("pub struct ")
        .or_else(|| line.strip_prefix("struct "))?;
    if !declaration.contains('{') {
        return None;
    }

    let name = declaration.split('{').next()?.trim();
    let name = name.split('<').next()?.trim();
    let name = name.split_whitespace().next()?.trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

/// Return the brace delta for one source line.
fn brace_delta(line: &str) -> isize {
    line.chars().fold(0isize, |delta, ch| match ch {
        '{' => delta + 1,
        '}' => delta - 1,
        _ => delta,
    })
}

/// Determine whether one trimmed line looks like a named struct field.
fn is_named_field(line: &str) -> bool {
    if line.is_empty()
        || line.starts_with("//")
        || line.starts_with("///")
        || line.starts_with("#[")
        || line.starts_with("pub use ")
    {
        return false;
    }
    line.contains(':') && !line.starts_with("fn ") && !line.starts_with("where ")
}

/// Extract the field type from a simple named-field line.
fn field_type(line: &str) -> Option<&str> {
    let (_, ty) = line.split_once(':')?;
    Some(ty.trim().trim_end_matches(','))
}

/// Ensure every wide Rust struct is explicitly classified in the audit allowlist.
#[test]
fn wide_structs_are_reviewed_and_capped() {
    let root = repo_root();
    let wide_structs: Vec<NamedStructInfo> = scan_named_structs(&root)
        .into_iter()
        .filter(|info| info.field_count >= WIDE_STRUCT_THRESHOLD)
        .collect();

    let actual_by_key: BTreeMap<(String, String), NamedStructInfo> = wide_structs
        .iter()
        .cloned()
        .map(|info| ((info.path.clone(), info.struct_name.clone()), info))
        .collect();
    let expected_keys: BTreeSet<(String, String)> = WIDE_STRUCT_ALLOWANCES
        .iter()
        .map(|entry| (entry.path.to_string(), entry.struct_name.to_string()))
        .collect();

    let mut failures = Vec::new();

    for info in &wide_structs {
        let Some(allowance) = WIDE_STRUCT_ALLOWANCES
            .iter()
            .find(|entry| entry.path == info.path && entry.struct_name == info.struct_name)
        else {
            failures.push(format!(
                "{}:{}: {} has {} fields and {} bool fields but no audit entry",
                info.path, info.line, info.struct_name, info.field_count, info.bool_field_count
            ));
            continue;
        };

        if info.field_count > allowance.max_fields {
            failures.push(format!(
                "{}:{}: {} grew from reviewed max {} fields to {} ({}, {})",
                info.path,
                info.line,
                info.struct_name,
                allowance.max_fields,
                info.field_count,
                allowance.disposition.label(),
                allowance.reason
            ));
        }

        if info.bool_field_count > allowance.max_bool_fields {
            failures.push(format!(
                "{}:{}: {} grew from reviewed max {} bool fields to {} ({}, {})",
                info.path,
                info.line,
                info.struct_name,
                allowance.max_bool_fields,
                info.bool_field_count,
                allowance.disposition.label(),
                allowance.reason
            ));
        }
    }

    for allowance in WIDE_STRUCT_ALLOWANCES {
        let key = (
            allowance.path.to_string(),
            allowance.struct_name.to_string(),
        );
        if !actual_by_key.contains_key(&key) {
            failures.push(format!(
                "{}: stale audit entry for {} ({}, {})",
                allowance.path,
                allowance.struct_name,
                allowance.disposition.label(),
                allowance.reason
            ));
        }
    }

    for key in actual_by_key.keys() {
        if !expected_keys.contains(key) {
            failures.push(format!(
                "{}: unexpected wide struct audit state for {}",
                key.0, key.1
            ));
        }
    }

    if !failures.is_empty() {
        panic!("wide struct audit failures:\n- {}", failures.join("\n- "));
    }
}
