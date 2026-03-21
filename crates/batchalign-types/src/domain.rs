//! Domain newtypes and small enums shared across batchalign crates.
//!
//! These are re-exported from [`super::api`] for backward compatibility.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

// ---------------------------------------------------------------------------
// Domain newtypes (shared across modules, re-exported from lib.rs)
// ---------------------------------------------------------------------------

validated_string_id!(
    /// Server-assigned identifier for a job (non-empty).
    pub JobId
);

string_id!(
    /// Batchalign command name (e.g. `"morphotag"`, `"align"`).
    pub CommandName
);

/// Closed released command vocabulary used at contributor-facing Rust seams.
///
/// This is intentionally distinct from [`CommandName`]:
///
/// - [`CommandName`] remains the open string type used at trust boundaries
///   such as HTTP payloads, SQLite persistence, and interop with older
///   callers.
/// - [`ReleasedCommand`] is the closed set of commands that contributors
///   should read in the code when reasoning about workflow families, dispatch
///   policy, and CLI behavior.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize,
    utoipa::ToSchema,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ReleasedCommand {
    Align,
    Transcribe,
    TranscribeS,
    Translate,
    Morphotag,
    Coref,
    Utseg,
    Benchmark,
    Opensmile,
    Compare,
    Avqi,
}

/// Error returned when one string is not a released command name.
#[derive(Debug, Clone, thiserror::Error)]
#[error("unknown released command \"{0}\"")]
pub struct InvalidReleasedCommand(pub String);

impl ReleasedCommand {
    /// All released commands in a stable contributor-facing order.
    pub const ALL: [Self; 11] = [
        Self::Align,
        Self::Transcribe,
        Self::TranscribeS,
        Self::Translate,
        Self::Morphotag,
        Self::Coref,
        Self::Utseg,
        Self::Benchmark,
        Self::Opensmile,
        Self::Compare,
        Self::Avqi,
    ];

    /// Parse one untrusted released-command token.
    pub fn parse_untrusted(value: &str) -> Result<Self, InvalidReleasedCommand> {
        Self::try_from(value.trim())
    }

    /// Return the canonical snake_case released command name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Align => "align",
            Self::Transcribe => "transcribe",
            Self::TranscribeS => "transcribe_s",
            Self::Translate => "translate",
            Self::Morphotag => "morphotag",
            Self::Coref => "coref",
            Self::Utseg => "utseg",
            Self::Benchmark => "benchmark",
            Self::Opensmile => "opensmile",
            Self::Compare => "compare",
            Self::Avqi => "avqi",
        }
    }

    /// Return the canonical wire/storage spelling.
    pub const fn as_wire_name(self) -> &'static str {
        self.as_str()
    }

    /// Return whether this released command requires client-local audio access.
    pub const fn uses_local_audio(self) -> bool {
        matches!(
            self,
            Self::Transcribe | Self::TranscribeS | Self::Benchmark | Self::Avqi
        )
    }
}

impl std::fmt::Display for ReleasedCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl AsRef<str> for ReleasedCommand {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl PartialEq<&str> for ReleasedCommand {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl TryFrom<&str> for ReleasedCommand {
    type Error = InvalidReleasedCommand;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "align" => Ok(Self::Align),
            "transcribe" => Ok(Self::Transcribe),
            "transcribe_s" => Ok(Self::TranscribeS),
            "translate" => Ok(Self::Translate),
            "morphotag" => Ok(Self::Morphotag),
            "coref" => Ok(Self::Coref),
            "utseg" => Ok(Self::Utseg),
            "benchmark" => Ok(Self::Benchmark),
            "opensmile" => Ok(Self::Opensmile),
            "compare" => Ok(Self::Compare),
            "avqi" => Ok(Self::Avqi),
            other => Err(InvalidReleasedCommand(other.to_owned())),
        }
    }
}

impl TryFrom<&CommandName> for ReleasedCommand {
    type Error = InvalidReleasedCommand;

    fn try_from(value: &CommandName) -> Result<Self, Self::Error> {
        Self::try_from(value.as_ref())
    }
}

impl From<ReleasedCommand> for CommandName {
    fn from(value: ReleasedCommand) -> Self {
        Self::from(value.as_str())
    }
}

/// Borrowed CHAT document text at a contributor-facing boundary.
///
/// This wrapper is intentionally lightweight: it prevents workflow/request
/// types from collapsing back into raw `&str` while still borrowing the
/// underlying document text without allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChatText<'a>(&'a str);

impl<'a> ChatText<'a> {
    /// Wrap one borrowed CHAT document string.
    pub fn new(text: &'a str) -> Self {
        Self(text)
    }

    /// Borrow the underlying CHAT string.
    pub fn as_str(self) -> &'a str {
        self.0
    }
}

impl std::fmt::Display for ChatText<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl<'a> From<&'a str> for ChatText<'a> {
    fn from(value: &'a str) -> Self {
        Self::new(value)
    }
}

impl std::ops::Deref for ChatText<'_> {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl AsRef<str> for ChatText<'_> {
    fn as_ref(&self) -> &str {
        self.0
    }
}

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
    // -- Well-known language codes (use these instead of string literals) --

    /// English (`"eng"`).
    pub fn eng() -> Self { Self("eng".to_owned()) }
    /// Spanish (`"spa"`).
    pub fn spa() -> Self { Self("spa".to_owned()) }
    /// French (`"fra"`).
    pub fn fra() -> Self { Self("fra".to_owned()) }
    /// Chinese / Mandarin (`"zho"`).
    pub fn zho() -> Self { Self("zho".to_owned()) }
    /// Cantonese (`"yue"`).
    pub fn yue() -> Self { Self("yue".to_owned()) }
    /// Japanese (`"jpn"`).
    pub fn jpn() -> Self { Self("jpn".to_owned()) }

    // -- Construction --

    /// Try to create a validated language code.
    ///
    /// Validation: exactly 3 ASCII alphabetic characters, lowercased.
    /// Rejects `"auto"`, `""`, `"en"`, `"english"`, etc.
    ///
    /// This is the **only** way to construct a `LanguageCode3` from
    /// untrusted input. Use well-known constants (e.g. [`Self::eng()`])
    /// for compile-time-known values.
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

impl TryFrom<String> for LanguageCode3 {
    type Error = InvalidLanguageCode;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::try_new(&s)
    }
}

impl TryFrom<&str> for LanguageCode3 {
    type Error = InvalidLanguageCode;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::try_new(s)
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
        Self::eng()
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
// WorkerLanguage — worker-runtime language routing, not a domain language code
// ---------------------------------------------------------------------------

/// Worker-runtime language routed to Python workers.
///
/// This is intentionally distinct from [`LanguageCode3`]. The worker runtime
/// accepts a small sentinel vocabulary that is meaningful only at the process
/// bootstrap/dispatch boundary:
///
/// - `Resolved(code)` for a concrete ISO 639-3 language
/// - `Auto` for ASR auto-detection
/// - `Unspecified` when the worker task does not consume a language hint
#[derive(Debug, Clone, PartialEq, Eq, Hash, utoipa::ToSchema)]
pub enum WorkerLanguage {
    /// Concrete ISO 639-3 language code.
    Resolved(LanguageCode3),
    /// ASR auto-detection sentinel.
    Auto,
    /// No worker language hint should be provided.
    Unspecified,
}

/// Error returned when a worker-runtime language string is invalid.
#[derive(Debug, Clone, thiserror::Error)]
#[error("invalid worker language \"{0}\": expected 3 ASCII letters, \"auto\", or an empty string")]
pub struct InvalidWorkerLanguage(pub String);

impl WorkerLanguage {
    /// Parse one untrusted worker-runtime language string.
    pub fn parse_untrusted(s: &str) -> Result<Self, InvalidWorkerLanguage> {
        let s = s.trim();
        if s.is_empty() {
            Ok(Self::Unspecified)
        } else if s.eq_ignore_ascii_case("auto") {
            Ok(Self::Auto)
        } else {
            LanguageCode3::try_new(s)
                .map(Self::Resolved)
                .map_err(|_| InvalidWorkerLanguage(s.to_string()))
        }
    }

    /// Return the CLI/registry string form used by the worker runtime.
    pub fn as_worker_arg(&self) -> &str {
        match self {
            Self::Resolved(code) => code.as_ref(),
            Self::Auto => "auto",
            Self::Unspecified => "",
        }
    }

    /// Return the resolved ISO language code, if present.
    pub fn as_resolved(&self) -> Option<&LanguageCode3> {
        match self {
            Self::Resolved(code) => Some(code),
            Self::Auto | Self::Unspecified => None,
        }
    }

    /// Return `true` when the worker should auto-detect the language.
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }

    /// Return `true` when the worker should receive no language hint.
    pub fn is_unspecified(&self) -> bool {
        matches!(self, Self::Unspecified)
    }
}

impl std::fmt::Display for WorkerLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_worker_arg())
    }
}

impl TryFrom<String> for WorkerLanguage {
    type Error = InvalidWorkerLanguage;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse_untrusted(&value)
    }
}

impl TryFrom<&str> for WorkerLanguage {
    type Error = InvalidWorkerLanguage;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::parse_untrusted(value)
    }
}

impl From<LanguageCode3> for WorkerLanguage {
    fn from(code: LanguageCode3) -> Self {
        Self::Resolved(code)
    }
}

impl From<&LanguageCode3> for WorkerLanguage {
    fn from(code: &LanguageCode3) -> Self {
        Self::Resolved(code.clone())
    }
}

impl From<&WorkerLanguage> for WorkerLanguage {
    fn from(value: &WorkerLanguage) -> Self {
        value.clone()
    }
}

impl serde::Serialize for WorkerLanguage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_worker_arg())
    }
}

impl<'de> serde::Deserialize<'de> for WorkerLanguage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::parse_untrusted(&s).map_err(serde::de::Error::custom)
    }
}

impl schemars::JsonSchema for WorkerLanguage {
    fn schema_name() -> String {
        "WorkerLanguage".to_string()
    }

    fn json_schema(_generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        use schemars::schema::{InstanceType, Metadata, Schema, SchemaObject};

        Schema::Object(SchemaObject {
            instance_type: Some(InstanceType::String.into()),
            metadata: Some(Box::new(Metadata {
                description: Some(
                    "Worker-runtime language string: ISO 639-3 code, \"auto\", or empty string."
                        .to_string(),
                ),
                ..Default::default()
            })),
            ..Default::default()
        })
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

    /// Convert this submission/runtime language into the worker-runtime language domain.
    pub fn to_worker_language(&self) -> WorkerLanguage {
        match self {
            Self::Auto => WorkerLanguage::Auto,
            Self::Resolved(code) => WorkerLanguage::Resolved(code.clone()),
        }
    }

    /// Parse from a DB string column. `"auto"` → `Auto`, anything else
    /// → `Resolved`.
    ///
    /// Returns `(spec, true)` if the value was valid, `(spec, false)` if
    /// the stored value was invalid and fell back to `eng`. Callers should
    /// log the fallback so corrupt DB values are visible.
    pub fn parse_from_db(s: &str) -> (Self, bool) {
        if s.eq_ignore_ascii_case("auto") {
            (Self::Auto, true)
        } else {
            match LanguageCode3::try_new(s) {
                Ok(code) => (Self::Resolved(code), true),
                Err(_) => (Self::Resolved(LanguageCode3::eng()), false),
            }
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

impl TryFrom<&str> for LanguageSpec {
    type Error = InvalidLanguageCode;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        if s.eq_ignore_ascii_case("auto") {
            Ok(Self::Auto)
        } else {
            LanguageCode3::try_new(s).map(Self::Resolved)
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

validated_string_id!(
    /// Basename of a file being processed (e.g. `"sample.cha"`).
    /// Rejects empty strings and path separators.
    pub FileName
    |s| !s.contains('/') && !s.contains('\\'), "must not contain path separators"
);

string_id!(
    /// Identifier of a server/fleet node.
    /// Empty when the node does not report an identity (older server versions).
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

validated_string_id!(
    /// ML engine version string for cache keying (e.g. `"stanza-1.9.2"`, non-empty).
    pub EngineVersion
);

validated_string_id!(
    /// Correlation ID for tracing a job across log entries (non-empty).
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

validated_string_id!(
    /// A Rev.AI server-side job identifier returned after audio submission (non-empty).
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
        let code = LanguageCode3::eng();
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
    fn language_code3_try_from_str_rejects_auto() {
        assert!(LanguageCode3::try_from("auto").is_err());
    }

    #[test]
    fn language_code3_try_from_string_rejects_auto() {
        assert!(LanguageCode3::try_from("auto".to_string()).is_err());
    }

    // ---- WorkerLanguage ----

    #[test]
    fn worker_language_parses_resolved_auto_and_unspecified() {
        assert_eq!(
            WorkerLanguage::parse_untrusted("eng").unwrap(),
            WorkerLanguage::Resolved(LanguageCode3::eng())
        );
        assert_eq!(
            WorkerLanguage::parse_untrusted("AUTO").unwrap(),
            WorkerLanguage::Auto
        );
        assert_eq!(
            WorkerLanguage::parse_untrusted("").unwrap(),
            WorkerLanguage::Unspecified
        );
    }

    #[test]
    fn worker_language_rejects_invalid_values() {
        assert!(WorkerLanguage::parse_untrusted("english").is_err());
        assert!(WorkerLanguage::parse_untrusted("12").is_err());
    }

    #[test]
    fn worker_language_serde_roundtrip() {
        let auto = WorkerLanguage::Auto;
        assert_eq!(serde_json::to_string(&auto).unwrap(), "\"auto\"");
        assert_eq!(
            serde_json::from_str::<WorkerLanguage>("\"\"").unwrap(),
            WorkerLanguage::Unspecified
        );
        assert_eq!(
            serde_json::from_str::<WorkerLanguage>("\"yue\"").unwrap(),
            WorkerLanguage::Resolved(LanguageCode3::yue())
        );
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
        assert_eq!(spec, LanguageSpec::Resolved(LanguageCode3::eng()));
    }

    #[test]
    fn language_spec_serializes_auto() {
        let json = serde_json::to_string(&LanguageSpec::Auto).unwrap();
        assert_eq!(json, "\"auto\"");
    }

    #[test]
    fn language_spec_serializes_resolved() {
        let json =
            serde_json::to_string(&LanguageSpec::Resolved(LanguageCode3::spa())).unwrap();
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
        let spec = LanguageSpec::Resolved(LanguageCode3::fra());
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
    fn language_spec_try_from_str_auto() {
        assert_eq!(LanguageSpec::try_from("auto").unwrap(), LanguageSpec::Auto);
    }

    #[test]
    fn language_spec_try_from_str_resolved() {
        assert_eq!(
            LanguageSpec::try_from("eng").unwrap(),
            LanguageSpec::Resolved(LanguageCode3::eng())
        );
    }

    #[test]
    fn language_spec_resolve_or_returns_resolved() {
        let spec = LanguageSpec::Resolved(LanguageCode3::spa());
        let fallback = LanguageCode3::eng();
        assert_eq!(spec.resolve_or(&fallback), LanguageCode3::spa());
    }

    #[test]
    fn language_spec_resolve_or_returns_fallback_for_auto() {
        let spec = LanguageSpec::Auto;
        let fallback = LanguageCode3::eng();
        assert_eq!(spec.resolve_or(&fallback), LanguageCode3::eng());
    }

    #[test]
    fn language_spec_display() {
        assert_eq!(LanguageSpec::Auto.to_string(), "auto");
        assert_eq!(
            LanguageSpec::Resolved(LanguageCode3::eng()).to_string(),
            "eng"
        );
    }

    #[test]
    fn language_spec_parse_from_db_valid() {
        let (spec, valid) = LanguageSpec::parse_from_db("auto");
        assert_eq!(spec, LanguageSpec::Auto);
        assert!(valid);

        let (spec, valid) = LanguageSpec::parse_from_db("eng");
        assert_eq!(spec, LanguageSpec::Resolved(LanguageCode3::eng()));
        assert!(valid);
    }

    #[test]
    fn language_spec_parse_from_db_invalid_falls_back() {
        let (spec, valid) = LanguageSpec::parse_from_db("not-a-lang");
        assert_eq!(spec, LanguageSpec::Resolved(LanguageCode3::eng()));
        assert!(!valid, "invalid DB value should report fallback");
    }

    #[test]
    fn language_spec_maps_to_worker_language() {
        assert_eq!(
            LanguageSpec::Auto.to_worker_language(),
            WorkerLanguage::Auto
        );
        assert_eq!(
            LanguageSpec::Resolved(LanguageCode3::eng()).to_worker_language(),
            WorkerLanguage::Resolved(LanguageCode3::eng())
        );
    }
}
