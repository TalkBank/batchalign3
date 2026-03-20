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

// ---------------------------------------------------------------------------
// LanguageCode3 — validated 3-letter ISO 639-3 language code
// ---------------------------------------------------------------------------

/// 3-letter ISO 639-3 language code (e.g. `"eng"`, `"spa"`).
///
/// Construction validates that the value is exactly 3 ASCII alphabetic
/// characters, lowercased. Sentinel values like `"auto"` are rejected — use
/// [`LanguageSpec`] at boundaries where auto-detection is meaningful.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, utoipa::ToSchema, schemars::JsonSchema,
)]
#[serde(transparent)]
pub struct LanguageCode3(pub String);

/// Error returned when a string is not a valid 3-letter ISO 639-3 code.
#[derive(Debug, Clone, thiserror::Error)]
#[error("invalid language code \"{0}\": expected 3 ASCII letters (e.g. \"eng\", \"spa\")")]
pub struct InvalidLanguageCode(pub String);

impl LanguageCode3 {
    /// Create a validated language code, panicking if invalid.
    ///
    /// Use [`Self::try_new`] at trust boundaries (CLI args, JSON
    /// deserialization) where the input may be user-provided.
    pub fn new(s: &str) -> Self {
        Self::try_new(s).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Create a language code for worker dispatch, accepting `"auto"` as a
    /// special sentinel value for ASR auto-detection.
    ///
    /// Normal code: only valid ISO 639-3 code paths should construct
    /// `LanguageCode3` via [`From`] or [`try_new`], which reject `"auto"`.
    /// This constructor is the *only* path that accepts `"auto"`, and it
    /// should only be used for GPU worker pool keys where `"auto"` is a
    /// valid dispatch directive.
    pub fn from_worker_lang(s: &str) -> Self {
        Self(s.to_ascii_lowercase())
    }

    /// Try to create a validated language code.
    ///
    /// Validation: exactly 3 ASCII alphabetic characters, lowercased.
    /// Rejects `"auto"`, `""`, `"en"`, `"english"`, etc.
    pub fn try_new(s: &str) -> Result<Self, InvalidLanguageCode> {
        let s = s.trim();
        if s.len() == 3 && s.bytes().all(|b| b.is_ascii_alphabetic()) {
            Ok(Self(s.to_ascii_lowercase()))
        } else {
            Err(InvalidLanguageCode(s.to_string()))
        }
    }
}

impl std::fmt::Display for LanguageCode3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for LanguageCode3 {
    fn from(s: String) -> Self {
        assert!(
            s.len() == 3 && s.bytes().all(|b| b.is_ascii_alphabetic()),
            "LanguageCode3::from() called with invalid code: {s:?}"
        );
        Self(s.to_ascii_lowercase())
    }
}

impl From<&str> for LanguageCode3 {
    fn from(s: &str) -> Self {
        assert!(
            s.len() == 3 && s.bytes().all(|b| b.is_ascii_alphabetic()),
            "LanguageCode3::from() called with invalid code: {s:?}"
        );
        Self(s.to_ascii_lowercase())
    }
}

impl From<LanguageCode3> for String {
    fn from(v: LanguageCode3) -> String {
        v.0
    }
}

impl std::ops::Deref for LanguageCode3 {
    type Target = str;
    fn deref(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for LanguageCode3 {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl PartialEq<&str> for LanguageCode3 {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

impl std::borrow::Borrow<str> for LanguageCode3 {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl Default for LanguageCode3 {
    fn default() -> Self {
        Self("eng".to_string())
    }
}

impl<'de> serde::Deserialize<'de> for LanguageCode3 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::try_new(&s).map_err(serde::de::Error::custom)
    }
}

// ---------------------------------------------------------------------------
// LanguageSpec — Auto vs Resolved(LanguageCode3)
// ---------------------------------------------------------------------------

/// Language specification from the CLI or job submission.
///
/// `Auto` means the ASR engine should detect the language. This variant must
/// be resolved to a concrete [`LanguageCode3`] before any CHAT construction
/// or NLP dispatch that requires a known language.
#[derive(Debug, Clone, PartialEq, Eq, Hash, ToSchema)]
pub enum LanguageSpec {
    /// Let the ASR engine auto-detect the language.
    Auto,
    /// A concrete ISO 639-3 language code.
    Resolved(LanguageCode3),
}

impl LanguageSpec {
    /// Return the resolved language code, or `None` if `Auto`.
    pub fn as_resolved(&self) -> Option<&LanguageCode3> {
        match self {
            Self::Auto => None,
            Self::Resolved(code) => Some(code),
        }
    }

    /// Return the resolved language code, falling back to `fallback` if
    /// `Auto`.
    pub fn resolve_or(&self, fallback: &LanguageCode3) -> LanguageCode3 {
        match self {
            Self::Auto => fallback.clone(),
            Self::Resolved(code) => code.clone(),
        }
    }

    /// Return `true` if this is `Auto`.
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }

    /// Parse from a DB string column. `"auto"` → `Auto`, anything else
    /// → `Resolved` (best-effort, since the DB may contain legacy values).
    pub fn parse_from_db(s: &str) -> Self {
        if s.eq_ignore_ascii_case("auto") {
            Self::Auto
        } else {
            Self::Resolved(LanguageCode3::from(s))
        }
    }
}

impl std::fmt::Display for LanguageSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Resolved(code) => write!(f, "{code}"),
        }
    }
}

impl From<LanguageCode3> for LanguageSpec {
    fn from(code: LanguageCode3) -> Self {
        Self::Resolved(code)
    }
}

impl From<&str> for LanguageSpec {
    fn from(s: &str) -> Self {
        if s.eq_ignore_ascii_case("auto") {
            Self::Auto
        } else {
            Self::Resolved(LanguageCode3::from(s))
        }
    }
}

impl Serialize for LanguageSpec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::Resolved(code) => serializer.serialize_str(&code.0),
        }
    }
}

impl<'de> Deserialize<'de> for LanguageSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.eq_ignore_ascii_case("auto") {
            Ok(Self::Auto)
        } else {
            LanguageCode3::try_new(&s)
                .map(Self::Resolved)
                .map_err(serde::de::Error::custom)
        }
    }
}

impl schemars::JsonSchema for LanguageSpec {
    fn schema_name() -> String {
        "LanguageSpec".to_string()
    }

    fn json_schema(g: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        // Reuse the string schema — "auto" or a 3-letter code.
        <String as schemars::JsonSchema>::json_schema(g)
    }
}

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

numeric_id!(
    /// Duration or audio position measured in milliseconds.
    ///
    /// Used for audio timestamps (`start_ms`, `end_ms`) and durations
    /// (`max_group_ms`, `tight_buffer_ms`) throughout the FA, ASR, and
    /// speaker pipelines. All ML worker IPC timing fields use this type.
    pub DurationMs(u64) [Eq]
);

numeric_id!(
    /// Physical memory quantity in megabytes.
    ///
    /// Used for memory gate thresholds and health-response memory readings.
    pub MemoryMb(u64) [Eq]
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

numeric_id!(
    /// Number of parallel file-processing workers for a job.
    ///
    /// Computed by `compute_job_workers()` based on available memory and CPU.
    /// Used in dispatch runtime structs to bound concurrency via a semaphore.
    pub NumWorkers(usize) [Eq]
);

string_id!(
    /// A Rev.AI server-side job identifier returned after audio submission.
    ///
    /// Obtained during preflight batch upload and passed to polling calls so
    /// individual file tasks can retrieve results without re-uploading audio.
    pub RevAiJobId
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
    /// Plain text output (e.g. AVQI voice quality reports).
    Text,
}

impl std::fmt::Display for ContentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chat => write!(f, "chat"),
            Self::Csv => write!(f, "csv"),
            Self::Text => write!(f, "text"),
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

#[cfg(test)]
mod tests {
    use super::*;

    // ---- LanguageCode3 validation ----

    #[test]
    fn language_code3_valid() {
        assert_eq!(LanguageCode3::try_new("eng").unwrap().0, "eng");
        assert_eq!(LanguageCode3::try_new("SPA").unwrap().0, "spa");
        assert_eq!(LanguageCode3::try_new("Zho").unwrap().0, "zho");
    }

    #[test]
    fn language_code3_rejects_auto() {
        assert!(LanguageCode3::try_new("auto").is_err());
    }

    #[test]
    fn language_code3_rejects_empty() {
        assert!(LanguageCode3::try_new("").is_err());
    }

    #[test]
    fn language_code3_rejects_two_letter() {
        assert!(LanguageCode3::try_new("en").is_err());
    }

    #[test]
    fn language_code3_rejects_four_letter() {
        assert!(LanguageCode3::try_new("engl").is_err());
    }

    #[test]
    fn language_code3_rejects_digits() {
        assert!(LanguageCode3::try_new("e1g").is_err());
    }

    #[test]
    fn language_code3_serde_roundtrip() {
        let code = LanguageCode3::from("eng");
        let json = serde_json::to_string(&code).unwrap();
        assert_eq!(json, "\"eng\"");
        let back: LanguageCode3 = serde_json::from_str(&json).unwrap();
        assert_eq!(back, code);
    }

    #[test]
    fn language_code3_deserialize_rejects_auto() {
        let result: Result<LanguageCode3, _> = serde_json::from_str("\"auto\"");
        assert!(result.is_err());
    }

    #[test]
    #[should_panic(expected = "invalid code")]
    fn language_code3_from_str_panics_on_auto() {
        let _ = LanguageCode3::from("auto");
    }

    #[test]
    #[should_panic(expected = "invalid code")]
    fn language_code3_from_string_panics_on_auto() {
        let _ = LanguageCode3::from("auto".to_string());
    }

    // ---- LanguageSpec ----

    #[test]
    fn language_spec_deserializes_auto() {
        let spec: LanguageSpec = serde_json::from_str("\"auto\"").unwrap();
        assert_eq!(spec, LanguageSpec::Auto);
    }

    #[test]
    fn language_spec_deserializes_auto_case_insensitive() {
        let spec: LanguageSpec = serde_json::from_str("\"AUTO\"").unwrap();
        assert_eq!(spec, LanguageSpec::Auto);
    }

    #[test]
    fn language_spec_deserializes_resolved() {
        let spec: LanguageSpec = serde_json::from_str("\"eng\"").unwrap();
        assert_eq!(spec, LanguageSpec::Resolved(LanguageCode3::from("eng")));
    }

    #[test]
    fn language_spec_serializes_auto() {
        let json = serde_json::to_string(&LanguageSpec::Auto).unwrap();
        assert_eq!(json, "\"auto\"");
    }

    #[test]
    fn language_spec_serializes_resolved() {
        let json =
            serde_json::to_string(&LanguageSpec::Resolved(LanguageCode3::from("spa"))).unwrap();
        assert_eq!(json, "\"spa\"");
    }

    #[test]
    fn language_spec_roundtrip_auto() {
        let spec = LanguageSpec::Auto;
        let json = serde_json::to_string(&spec).unwrap();
        let back: LanguageSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn language_spec_roundtrip_resolved() {
        let spec = LanguageSpec::Resolved(LanguageCode3::from("fra"));
        let json = serde_json::to_string(&spec).unwrap();
        let back: LanguageSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn language_spec_rejects_invalid_code() {
        let result: Result<LanguageSpec, _> = serde_json::from_str("\"xx\"");
        assert!(result.is_err());
    }

    #[test]
    fn language_spec_from_str_auto() {
        assert_eq!(LanguageSpec::from("auto"), LanguageSpec::Auto);
    }

    #[test]
    fn language_spec_from_str_resolved() {
        assert_eq!(
            LanguageSpec::from("eng"),
            LanguageSpec::Resolved(LanguageCode3::from("eng"))
        );
    }

    #[test]
    fn language_spec_resolve_or_returns_resolved() {
        let spec = LanguageSpec::Resolved(LanguageCode3::from("spa"));
        let fallback = LanguageCode3::from("eng");
        assert_eq!(spec.resolve_or(&fallback), LanguageCode3::from("spa"));
    }

    #[test]
    fn language_spec_resolve_or_returns_fallback_for_auto() {
        let spec = LanguageSpec::Auto;
        let fallback = LanguageCode3::from("eng");
        assert_eq!(spec.resolve_or(&fallback), LanguageCode3::from("eng"));
    }

    #[test]
    fn language_spec_display() {
        assert_eq!(LanguageSpec::Auto.to_string(), "auto");
        assert_eq!(
            LanguageSpec::Resolved(LanguageCode3::from("eng")).to_string(),
            "eng"
        );
    }

    #[test]
    fn language_spec_parse_from_db() {
        assert_eq!(LanguageSpec::parse_from_db("auto"), LanguageSpec::Auto);
        assert_eq!(
            LanguageSpec::parse_from_db("eng"),
            LanguageSpec::Resolved(LanguageCode3::from("eng"))
        );
    }
}
