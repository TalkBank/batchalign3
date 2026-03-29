//! REST API request models — `POST /jobs` submission types.
//!
//! These are re-exported from [`super::api`] for backward compatibility.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::options::{AsrEngineName, CommandOptions};
use crate::revai::try_revai_language_hint;

use super::domain::{DisplayPath, LanguageCode3, LanguageSpec, NumSpeakers, ReleasedCommand};

// ---------------------------------------------------------------------------
// Request models
// ---------------------------------------------------------------------------

/// A single CHAT file submitted by the client.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct FilePayload {
    /// Original filename (e.g. "01DM_18.cha").
    pub filename: DisplayPath,
    /// Full CHAT file text.
    pub content: String,
}

/// `POST /jobs` request body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct JobSubmission {
    /// Batchalign command (align, morphotag, etc.).
    pub command: ReleasedCommand,
    /// Language specification: a 3-letter ISO code or `"auto"` for
    /// ASR-driven detection.
    #[serde(default = "default_lang")]
    pub lang: LanguageSpec,
    /// Number of speakers.
    #[serde(default = "default_num_speakers")]
    pub num_speakers: NumSpeakers,
    /// CHAT files to process.
    #[serde(default)]
    pub files: Vec<FilePayload>,
    /// Media filenames for the server to resolve from media_roots (transcribe only).
    #[serde(default)]
    pub media_files: Vec<String>,
    /// Key into server's media_mappings config (e.g. "childes-data").
    #[serde(default)]
    pub media_mapping: batchalign_types::paths::MediaMappingKey,
    /// Subdirectory under the mapped root (e.g. "Eng-NA/MacWhinney/0young-ASR").
    #[serde(default)]
    pub media_subdir: batchalign_types::paths::RepoRelativePath,
    /// Client's input directory path (for dashboard display).
    #[serde(default)]
    pub source_dir: batchalign_types::paths::ClientPath,
    /// Typed command options (engine selections, processing flags, etc.).
    #[schema(value_type = serde_json::Value)]
    pub options: CommandOptions,

    // Paths mode — local daemon sends filesystem paths instead of content.
    /// When true, server reads/writes files directly via source_paths/output_paths.
    #[serde(default)]
    pub paths_mode: bool,
    /// Absolute paths to read input files from (paths_mode only).
    #[serde(default)]
    pub source_paths: Vec<batchalign_types::paths::ClientPath>,
    /// Absolute paths to write output files to (paths_mode only).
    #[serde(default)]
    pub output_paths: Vec<batchalign_types::paths::ClientPath>,
    /// Human-readable filenames for display (paths_mode only, optional).
    #[serde(default)]
    pub display_names: Vec<String>,

    /// When true, the server collects detailed algorithm traces for
    /// visualization (DP alignment matrices, ASR pipeline stages, FA
    /// timelines, retokenization mappings). Defaults to false — zero
    /// overhead when off.
    #[serde(default)]
    pub debug_traces: bool,

    /// Absolute paths to "before" files for incremental processing
    /// (paths_mode only). When non-empty, the diff engine compares each
    /// before file against its corresponding source_path and only
    /// reprocesses changed utterances.
    ///
    /// Must be the same length as `source_paths` when non-empty.
    #[serde(default)]
    pub before_paths: Vec<batchalign_types::paths::ClientPath>,
}

pub(crate) fn default_lang() -> LanguageSpec {
    LanguageSpec::Resolved(LanguageCode3::eng())
}

pub(crate) fn default_num_speakers() -> NumSpeakers {
    NumSpeakers(1)
}

impl JobSubmission {
    /// Validate submission constraints (paths_mode, command consistency).
    pub fn validate(&self) -> Result<(), ValidationError> {
        // Validate options command tag matches the command field.
        if self.command != self.options.command_name() {
            return Err(ValidationError(format!(
                "options command tag '{}' does not match submission command '{}'",
                self.options.command_name(),
                self.command
            )));
        }

        // Validate language support for engines the command will use.
        self.validate_language_support()?;

        if self.paths_mode {
            if self.source_paths.is_empty() || self.output_paths.is_empty() {
                return Err(ValidationError(
                    "paths_mode requires non-empty source_paths and output_paths".into(),
                ));
            }
            if self.source_paths.len() != self.output_paths.len() {
                return Err(ValidationError(
                    "source_paths and output_paths must have equal length".into(),
                ));
            }
            if !self.before_paths.is_empty() && self.before_paths.len() != self.source_paths.len() {
                return Err(ValidationError(
                    "before_paths must have the same length as source_paths when non-empty".into(),
                ));
            }
            if !self.files.is_empty() || !self.media_files.is_empty() {
                return Err(ValidationError(
                    "paths_mode is mutually exclusive with files/media_files".into(),
                ));
            }
        }
        Ok(())
    }

    /// Check that the job's language is supported by all engines the command
    /// will use.
    ///
    /// Called at job submission time to fail fast with a clear diagnostic
    /// rather than letting errors surface deep in the pipeline (Rev.AI HTTP
    /// 400, Whisper wrong-language transcription, Stanza model-not-found).
    fn validate_language_support(&self) -> Result<(), ValidationError> {
        // Auto-detect: can't validate engine support until ASR runs.
        let lang = match &self.lang {
            LanguageSpec::Auto => return Ok(()),
            LanguageSpec::Resolved(code) => code,
        };

        // Commands that use ASR: transcribe, transcribe_s, align, benchmark
        let asr_engine = match &self.options {
            CommandOptions::Transcribe(opts) | CommandOptions::TranscribeS(opts) => {
                Some(opts.effective_asr_engine())
            }
            CommandOptions::Align(opts) => {
                // Align uses ASR for UTR pre-pass
                Some(
                    opts.common
                        .engine_overrides
                        .asr
                        .clone()
                        .unwrap_or(AsrEngineName::RevAi),
                )
            }
            CommandOptions::Benchmark(opts) => Some(opts.effective_asr_engine()),
            _ => None,
        };

        // Check Rev.AI language support
        if let Some(AsrEngineName::RevAi) = &asr_engine
            && try_revai_language_hint(lang).is_none()
        {
            return Err(ValidationError(format!(
                "Language '{}' is not supported by Rev.AI ASR. Alternatives:\n\
                 - Use --asr-engine whisper for local Whisper ASR (supports most languages)\n\
                 - Use --asr-engine-custom tencent for Chinese/Hakka via Tencent\n\
                 - Check supported languages: book/src/reference/language-code-resolution.md",
                lang
            )));
        }

        // Commands that use Stanza: morphotag, utseg, coref, compare
        let uses_stanza = matches!(
            &self.options,
            CommandOptions::Morphotag(_)
                | CommandOptions::Utseg(_)
                | CommandOptions::Coref(_)
                | CommandOptions::Compare(_)
        );
        if uses_stanza && !is_stanza_supported_language(lang) {
            return Err(ValidationError(format!(
                "Language '{}' is not supported by Stanza. Supported languages:\n\
                 {}",
                lang,
                stanza_supported_languages_help()
            )));
        }

        // Check HK ASR engine language constraints
        if let Some(engine) = &asr_engine {
            let chinese_codes = ["zho", "yue", "wuu", "nan", "hak", "cmn"];
            match engine {
                AsrEngineName::HkTencent if !chinese_codes.contains(&lang.as_ref()) => {
                    return Err(ValidationError(format!(
                        "Language '{}' is not supported by Tencent ASR (Chinese variants only: {}). \
                         Use --asr-engine whisper or --asr-engine rev instead.",
                        lang,
                        chinese_codes.join(", ")
                    )));
                }
                AsrEngineName::HkAliyun if lang.as_ref() != "yue" => {
                    return Err(ValidationError(format!(
                        "Language '{}' is not supported by Aliyun ASR (Cantonese 'yue' only). \
                         Use --asr-engine whisper or --asr-engine rev instead.",
                        lang
                    )));
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// Validation error for request models.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct ValidationError(pub String);

// ---------------------------------------------------------------------------
// Stanza language support — hardcoded fallback table
// ---------------------------------------------------------------------------

/// Hardcoded fallback table of ISO 639-3 codes supported by Stanza.
///
/// **DEPRECATED as the primary check.** The authoritative source is now
/// the `StanzaRegistry` built from Stanza's `resources.json` at worker
/// startup. This table is ONLY used as a pre-validation safety net when
/// the registry hasn't been populated yet (before first worker spawn).
///
/// Use `StanzaRegistry::supports_morphosyntax()` for authoritative checks
/// when the registry is available (via `WorkerPool::stanza_registry()`).
///
/// See `batchalign/worker/_stanza_capabilities.py` for the authoritative
/// Python-side table builder.
const STANZA_SUPPORTED_ISO3: &[&str] = &[
    "ara", "ben", "bul", "cat", "ces", "cmn", "cym", "dan", "deu", "ell", "eng", "est", "eus",
    "fas", "fin", "fra", "gla", "gle", "glg", "heb", "hin", "hrv", "hun", "hye", "ind", "isl",
    "ita", "jpn", "kan", "kat", "kor", "lav", "lit", "mal", "mlt", "msa", "nld", "nor", "pol",
    "por", "ron", "slk", "slv", "spa", "swe", "tam", "tel", "tgl", "tha", "tur", "ukr", "urd",
    "vie", "yue", "zho",
];

/// Check whether an ISO 639-3 language code is supported by Stanza.
fn is_stanza_supported_language(lang: &LanguageCode3) -> bool {
    STANZA_SUPPORTED_ISO3.contains(&lang.as_ref())
}

/// Format a help string listing supported Stanza languages for error messages.
fn stanza_supported_languages_help() -> String {
    // Group in rows of 10 for readability.
    STANZA_SUPPORTED_ISO3
        .chunks(10)
        .map(|chunk| chunk.join(", "))
        .collect::<Vec<_>>()
        .join(",\n  ")
}

/// Validate a job's language support using the runtime Stanza registry.
///
/// This is the **authoritative** language validation, called from
/// `materialize_submission_job()` where the registry is available.
/// It supersedes the hardcoded `is_stanza_supported_language()` check
/// in `validate_language_support()`, which acts as a conservative
/// pre-filter only.
///
/// Returns `Ok(())` when:
/// - The command doesn't use Stanza
/// - The language is auto-detect
/// - The registry confirms the language has required processors
/// - The registry is not populated (fallback to hardcoded table)
pub fn validate_language_with_registry(
    submission: &JobSubmission,
    registry: Option<&crate::stanza_registry::StanzaRegistry>,
) -> Result<(), ValidationError> {
    let lang = match &submission.lang {
        LanguageSpec::Auto => return Ok(()),
        LanguageSpec::Resolved(code) => code,
    };

    let uses_stanza = matches!(
        &submission.options,
        CommandOptions::Morphotag(_)
            | CommandOptions::Utseg(_)
            | CommandOptions::Coref(_)
            | CommandOptions::Compare(_)
    );

    if !uses_stanza {
        return Ok(());
    }

    let Some(reg) = registry else {
        // Registry not populated — the hardcoded table in validate() already
        // caught obviously unsupported languages.
        return Ok(());
    };

    if !reg.supports_morphosyntax(lang.as_ref()) {
        let supported = reg.supported_languages().join(", ");
        return Err(ValidationError(format!(
            "Language '{}' is not supported by Stanza on this server. \
             Supported languages: {}",
            lang, supported
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::{CommonOptions, MorphotagOptions, UtsegOptions};

    /// Build a minimal `JobSubmission` for testing language validation.
    fn morphotag_submission(lang: &str) -> JobSubmission {
        JobSubmission {
            command: ReleasedCommand::Morphotag,
            lang: LanguageSpec::Resolved(LanguageCode3::try_new(lang).expect("test lang")),
            num_speakers: NumSpeakers(1),
            files: vec![],
            media_files: vec![],
            media_mapping: Default::default(),
            media_subdir: Default::default(),
            source_dir: Default::default(),
            options: CommandOptions::Morphotag(MorphotagOptions {
                common: CommonOptions::default(),
                retokenize: false,
                skipmultilang: false,
                merge_abbrev: false.into(),
            }),
            paths_mode: false,
            source_paths: vec![],
            output_paths: vec![],
            display_names: vec![],
            debug_traces: false,
            before_paths: vec![],
        }
    }

    fn utseg_submission(lang: &str) -> JobSubmission {
        JobSubmission {
            command: ReleasedCommand::Utseg,
            lang: LanguageSpec::Resolved(LanguageCode3::try_new(lang).expect("test lang")),
            num_speakers: NumSpeakers(1),
            files: vec![],
            media_files: vec![],
            media_mapping: Default::default(),
            media_subdir: Default::default(),
            source_dir: Default::default(),
            options: CommandOptions::Utseg(UtsegOptions {
                common: CommonOptions::default(),
                merge_abbrev: Default::default(),
            }),
            paths_mode: false,
            source_paths: vec![],
            output_paths: vec![],
            display_names: vec![],
            debug_traces: false,
            before_paths: vec![],
        }
    }

    #[test]
    fn stanza_table_is_sorted() {
        // Keep the table sorted so binary search or visual inspection is easy.
        let mut sorted = STANZA_SUPPORTED_ISO3.to_vec();
        sorted.sort();
        assert_eq!(STANZA_SUPPORTED_ISO3, sorted.as_slice());
    }

    #[test]
    fn stanza_table_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for code in STANZA_SUPPORTED_ISO3 {
            assert!(seen.insert(code), "duplicate Stanza language code: {code}");
        }
    }

    #[test]
    fn morphotag_with_supported_language_passes() {
        let submission = morphotag_submission("eng");
        assert!(submission.validate().is_ok());
    }

    #[test]
    fn morphotag_with_unsupported_language_fails() {
        let submission = morphotag_submission("xyz");
        let err = submission.validate().unwrap_err();
        assert!(
            err.to_string().contains("not supported by Stanza"),
            "expected Stanza error, got: {err}"
        );
    }

    #[test]
    fn utseg_with_unsupported_language_fails() {
        let submission = utseg_submission("xyz");
        let err = submission.validate().unwrap_err();
        assert!(
            err.to_string().contains("not supported by Stanza"),
            "expected Stanza error, got: {err}"
        );
    }

    #[test]
    fn morphotag_with_all_supported_languages_passes() {
        for code in STANZA_SUPPORTED_ISO3 {
            let submission = morphotag_submission(code);
            assert!(
                submission.validate().is_ok(),
                "expected language '{code}' to pass Stanza validation"
            );
        }
    }
}
