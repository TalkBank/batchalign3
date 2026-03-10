//! Domain newtypes and small enums shared across modules.
//!
//! These are re-exported from [`super::api`] for backward compatibility.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

// ---------------------------------------------------------------------------
// Domain newtypes (shared across modules, re-exported from lib.rs)
// ---------------------------------------------------------------------------

string_id!(
    /// Server-assigned UUID (v4) for a job.
    pub JobId
);

string_id!(
    /// Batchalign command name (e.g. `"morphotag"`, `"align"`).
    pub CommandName
);

string_id!(
    /// 3-letter ISO 639-3 language code (e.g. `"eng"`, `"spa"`).
    pub LanguageCode3
);

string_id!(
    /// Basename of a file being processed (e.g. `"sample.cha"`).
    pub FileName
);

string_id!(
    /// Identifier of a server/fleet node.
    pub NodeId
);

numeric_id!(
    /// Number of speakers in a recording.
    pub NumSpeakers(u32) [Eq]
);

numeric_id!(
    /// Duration measured in fractional seconds.
    pub DurationSeconds(f64)
);

numeric_id!(
    /// Unix timestamp as fractional seconds since epoch.
    pub UnixTimestamp(f64)
);

string_id!(
    /// ML engine version string for cache keying (e.g. `"stanza-1.9.2"`).
    pub EngineVersion
);

string_id!(
    /// Correlation ID for tracing a job across log entries.
    ///
    /// Usually the same as `JobId` but may differ for retried or cloned jobs.
    pub CorrelationId
);

/// MIME-like content discriminator for file results.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ContentType {
    /// CHAT format output.
    #[default]
    Chat,
    /// Tabular CSV output (e.g. opensmile features).
    Csv,
}

impl std::fmt::Display for ContentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chat => write!(f, "chat"),
            Self::Csv => write!(f, "csv"),
        }
    }
}

/// Server health status.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    /// Server is accepting work.
    #[default]
    Ok,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ok => write!(f, "ok"),
        }
    }
}
