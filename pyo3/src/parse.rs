//! Pure-Rust parse helpers (no PyResult — safe for use in allow_threads closures).

use pyo3::PyResult;
use talkbank_model::alignment::helpers::AlignmentDomain;
use talkbank_model::model::Line;

pub use batchalign_chat_ops::fa::strip_timing_from_content;

pub(crate) fn parse_strict_pure(
    chat_text: &str,
) -> Result<talkbank_model::model::ChatFile, String> {
    talkbank_parser::parse_chat_file(chat_text).map_err(|errors| {
        let msgs: Vec<String> = errors.errors.iter().map(|e| format!("{}", e)).collect();
        format!("Parse error: {}", msgs.join("\n"))
    })
}

pub(crate) fn parse_lenient_pure(
    chat_text: &str,
) -> Result<talkbank_model::model::ChatFile, String> {
    parse_lenient_with_warnings(chat_text).map(|(cf, _)| cf)
}

pub(crate) fn parse_lenient_with_warnings(
    chat_text: &str,
) -> Result<
    (
        talkbank_model::model::ChatFile,
        Vec<talkbank_model::ParseError>,
    ),
    String,
> {
    use talkbank_model::ErrorCollector;

    let errors = ErrorCollector::new();
    let chat_file = talkbank_parser::parse_chat_file_streaming(chat_text, &errors);

    // Streaming parse always returns a ChatFile, but if there are fatal errors
    // and the result is empty, treat as rejection.
    let error_vec = errors.into_vec();
    if chat_file.lines.is_empty() && !error_vec.is_empty() {
        let msgs: Vec<String> = error_vec.iter().map(|e| format!("{}", e)).collect();
        Err(format!("Parse error:\n{}", msgs.join("\n")))
    } else {
        Ok((chat_file, error_vec))
    }
}

/// Serialize a slice of `ParseError` to a JSON array string.
///
/// Each element is an object with: code, severity, line, column, message, suggestion.
/// Uses serde_json with only the fields Python callers need (strips source_cache etc).
pub(crate) fn errors_to_json(errors: &[talkbank_model::ParseError]) -> String {
    #[derive(serde::Serialize)]
    struct ErrorEntry<'a> {
        code: String,
        severity: &'a str,
        line: Option<usize>,
        column: Option<usize>,
        message: &'a str,
        suggestion: Option<&'a str>,
    }

    let entries: Vec<ErrorEntry<'_>> = errors
        .iter()
        .map(|e| ErrorEntry {
            code: format!("{}", e.code),
            severity: if e.severity == talkbank_model::Severity::Warning {
                "warning"
            } else {
                "error"
            },
            line: e.location.line,
            column: e.location.column,
            message: &e.message,
            suggestion: e.suggestion.as_deref(),
        })
        .collect();
    serde_json::to_string(&entries).unwrap_or_else(|e| {
        tracing::error!(error = %e, "failed to serialize error entries");
        "[]".to_string()
    })
}

pub(crate) fn strip_timing_on_chat_file(chat_file: &mut talkbank_model::model::ChatFile) {
    use talkbank_model::model::DependentTier;
    for line in &mut chat_file.lines {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };
        utt.main.content.bullet = None;
        strip_timing_from_content(&mut utt.main.content.content.0);
        // Remove %wor tiers.
        utt.dependent_tiers
            .retain(|t| !matches!(t, DependentTier::Wor(_)));
    }
}

pub(crate) fn parse_alignment_domain(domain: &str) -> PyResult<AlignmentDomain> {
    match domain {
        "mor" => Ok(AlignmentDomain::Mor),
        "wor" => Ok(AlignmentDomain::Wor),
        "pho" => Ok(AlignmentDomain::Pho),
        "sin" => Ok(AlignmentDomain::Sin),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid domain: {domain:?}. Must be one of: mor, wor, pho, sin"
        ))),
    }
}
