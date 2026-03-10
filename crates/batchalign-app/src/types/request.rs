//! REST API request models — `POST /jobs` submission types.
//!
//! These are re-exported from [`super::api`] for backward compatibility.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::options::CommandOptions;

use super::domain::{CommandName, FileName, LanguageCode3, NumSpeakers};

// ---------------------------------------------------------------------------
// Request models
// ---------------------------------------------------------------------------

/// A single CHAT file submitted by the client.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct FilePayload {
    /// Original filename (e.g. "01DM_18.cha").
    pub filename: FileName,
    /// Full CHAT file text.
    pub content: String,
}

/// `POST /jobs` request body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct JobSubmission {
    /// Batchalign command (align, morphotag, etc.).
    pub command: CommandName,
    /// 3-letter ISO language code.
    #[serde(default = "default_lang")]
    pub lang: LanguageCode3,
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
    pub media_mapping: String,
    /// Subdirectory under the mapped root (e.g. "Eng-NA/MacWhinney/0young-ASR").
    #[serde(default)]
    pub media_subdir: String,
    /// Client's input directory path (for dashboard display).
    #[serde(default)]
    pub source_dir: String,
    /// Typed command options (engine selections, processing flags, etc.).
    #[schema(value_type = serde_json::Value)]
    pub options: CommandOptions,

    // Paths mode — local daemon sends filesystem paths instead of content.
    /// When true, server reads/writes files directly via source_paths/output_paths.
    #[serde(default)]
    pub paths_mode: bool,
    /// Absolute paths to read input files from (paths_mode only).
    #[serde(default)]
    pub source_paths: Vec<String>,
    /// Absolute paths to write output files to (paths_mode only).
    #[serde(default)]
    pub output_paths: Vec<String>,
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
    pub before_paths: Vec<String>,
}

pub(crate) fn default_lang() -> LanguageCode3 {
    LanguageCode3::from("eng")
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
}

/// Validation error for request models.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct ValidationError(pub String);
