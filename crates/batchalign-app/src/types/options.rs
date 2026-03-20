//! Typed per-command options for job submission and processing.
//!
//! Replaces the stringly-typed `HashMap<String, serde_json::Value>` that
//! previously carried command options through the system. Each command has
//! a dedicated struct with compile-time checked fields and serde defaults
//! matching the CLI defaults.
//!
//! # Wire format
//!
//! [`CommandOptions`] serializes as an internally-tagged JSON object:
//!
//! ```json
//! {
//!   "command": "morphotag",
//!   "retokenize": true,
//!   "skipmultilang": false,
//!   "merge_abbrev": false,
//!   "override_cache": false,
//!   "lazy_audio": true,
//!   "engine_overrides": {}
//! }
//! ```
//!
//! The `command` tag doubles as the command name for routing.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub use super::params::{MergeAbbrevPolicy, WorTierPolicy};

// ---------------------------------------------------------------------------
// Default helpers
// ---------------------------------------------------------------------------

/// Default helper for boolean flags that opt in by default.
fn default_true() -> bool {
    true
}

/// Default forced-alignment engine for serialized command options.
fn default_fa_engine() -> FaEngineName {
    FaEngineName::Wave2Vec
}

/// Default ASR engine for serialized command options.
fn default_asr_engine() -> AsrEngineName {
    AsrEngineName::RevAi
}

/// Default Whisper batch size.
fn default_batch_size() -> i32 {
    8
}

/// Default `%wor` policy for commands that enable the tier by default.
fn default_wor_tier_include() -> WorTierPolicy {
    WorTierPolicy::Include
}

/// Default openSMILE feature set.
fn default_feature_set() -> String {
    "eGeMAPSv02".to_string()
}

/// Wraps a plugin-defined engine identifier so the rest of the system does not
/// pass anonymous strings around when a built-in enum variant is not available.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CustomEngineName(String);

impl CustomEngineName {
    /// Build a custom engine identifier from owned storage.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Borrow the engine identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for CustomEngineName {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for CustomEngineName {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

/// Typed UTR engine selector.
///
/// The wire format still uses the legacy string tokens (`"rev_utr"`,
/// `"whisper_utr"`, or a plugin-provided name), but the server runtime works
/// with this enum so the control plane stops branching on anonymous strings.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UtrEngine {
    /// Rust-owned Rev.AI timed-word path.
    RevAi,
    /// Python-worker ASR path with the built-in Whisper profile.
    Whisper,
    /// Python-worker ASR path with a plugin-defined engine/profile name.
    Custom(CustomEngineName),
}

impl UtrEngine {
    /// Parse one persisted wire-format token into the typed runtime selector.
    pub fn from_wire_name(name: &str) -> Self {
        match name {
            "rev_utr" => Self::RevAi,
            "whisper_utr" => Self::Whisper,
            other => Self::Custom(CustomEngineName::from(other)),
        }
    }

    /// Borrow the legacy wire-format token used in JSON payloads and SQLite.
    pub fn as_wire_name(&self) -> &str {
        match self {
            Self::RevAi => "rev_utr",
            Self::Whisper => "whisper_utr",
            Self::Custom(name) => name.as_str(),
        }
    }

    /// Whether this engine is fully Rust-owned and should never route through
    /// the Python worker transport in server mode.
    pub fn is_rust_owned(&self) -> bool {
        matches!(self, Self::RevAi)
    }

    /// Whether the current engine can reuse the worker-side segment strategy
    /// for partial-window UTR.
    pub fn supports_partial_windows(&self) -> bool {
        !self.is_rust_owned()
    }
}

impl Serialize for UtrEngine {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_wire_name())
    }
}

impl<'de> Deserialize<'de> for UtrEngine {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Ok(Self::from_wire_name(&name))
    }
}

/// Typed forced-alignment engine selector.
///
/// The wire format still uses the legacy string tokens (`"wav2vec_fa"`,
/// `"whisper_fa"`, or a plugin-provided name), but the control plane works
/// with this enum so dispatch does not branch on anonymous strings.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FaEngineName {
    /// MMS Wave2Vec forced alignment.
    Wave2Vec,
    /// Whisper token-timestamp forced alignment.
    Whisper,
    /// Plugin-defined forced-alignment backend/profile.
    Custom(CustomEngineName),
}

impl FaEngineName {
    /// Parse one persisted wire-format token into the typed runtime selector.
    pub fn from_wire_name(name: &str) -> Self {
        match name {
            "wav2vec_fa" => Self::Wave2Vec,
            "whisper_fa" => Self::Whisper,
            other => Self::Custom(CustomEngineName::from(other)),
        }
    }

    /// Borrow the legacy wire-format token used in JSON payloads and SQLite.
    pub fn as_wire_name(&self) -> &str {
        match self {
            Self::Wave2Vec => "wav2vec_fa",
            Self::Whisper => "whisper_fa",
            Self::Custom(name) => name.as_str(),
        }
    }
}

impl Serialize for FaEngineName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_wire_name())
    }
}

impl<'de> Deserialize<'de> for FaEngineName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Ok(Self::from_wire_name(&name))
    }
}

impl From<String> for FaEngineName {
    fn from(value: String) -> Self {
        Self::from_wire_name(&value)
    }
}

impl From<&str> for FaEngineName {
    fn from(value: &str) -> Self {
        Self::from_wire_name(value)
    }
}

/// Typed ASR engine selector.
///
/// The wire format still uses the legacy string tokens (`"rev"`,
/// `"whisper"`, `"whisperx"`, `"whisper_oai"`, or a plugin-provided name), but
/// the control plane works with this enum so backend selection is explicit.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AsrEngineName {
    /// Rust-owned Rev.AI backend.
    RevAi,
    /// Local Whisper worker backend.
    Whisper,
    /// WhisperX worker backend.
    WhisperX,
    /// OpenAI Whisper API backend.
    WhisperOai,
    /// Plugin-defined ASR backend/profile.
    Custom(CustomEngineName),
}

impl AsrEngineName {
    /// Parse one persisted wire-format token into the typed runtime selector.
    pub fn from_wire_name(name: &str) -> Self {
        match name {
            "rev" => Self::RevAi,
            "whisper" => Self::Whisper,
            "whisperx" => Self::WhisperX,
            "whisper_oai" => Self::WhisperOai,
            other => Self::Custom(CustomEngineName::from(other)),
        }
    }

    /// Borrow the legacy wire-format token used in JSON payloads and SQLite.
    pub fn as_wire_name(&self) -> &str {
        match self {
            Self::RevAi => "rev",
            Self::Whisper => "whisper",
            Self::WhisperX => "whisperx",
            Self::WhisperOai => "whisper_oai",
            Self::Custom(name) => name.as_str(),
        }
    }

    /// Whether this engine is the Rust-owned Rev.AI path.
    pub fn is_revai(&self) -> bool {
        matches!(self, Self::RevAi)
    }
}

impl Serialize for AsrEngineName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_wire_name())
    }
}

impl<'de> Deserialize<'de> for AsrEngineName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Ok(Self::from_wire_name(&name))
    }
}

impl From<String> for AsrEngineName {
    fn from(value: String) -> Self {
        Self::from_wire_name(&value)
    }
}

impl From<&str> for AsrEngineName {
    fn from(value: &str) -> Self {
        Self::from_wire_name(value)
    }
}

// ---------------------------------------------------------------------------
// CommonOptions
// ---------------------------------------------------------------------------

/// Options shared by all processing commands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommonOptions {
    /// Bypass the utterance analysis cache.
    #[serde(default)]
    pub override_cache: bool,

    /// Lazy audio loading for alignment/ASR.
    #[serde(default = "default_true")]
    pub lazy_audio: bool,

    /// Engine overrides keyed by task name (e.g. `{"asr": "tencent"}`).
    /// Values are dynamic strings because plugin names are runtime-determined.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub engine_overrides: BTreeMap<String, String>,

    /// Multi-word token (MWT) lexicon: maps a surface form (e.g. "gonna")
    /// to its expansion tokens (e.g. `["going", "to"]`).
    /// Loaded from `--lexicon` CSV on the CLI side.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mwt: BTreeMap<String, Vec<String>>,

    /// Optional directory for pipeline debug artifact dumps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debug_dir: Option<String>,

    /// Per-task cache override specifications (comma-separated task names).
    /// When non-empty, only the listed tasks skip cache; others use cache normally.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub override_cache_tasks: Vec<String>,
}

impl CommonOptions {
    /// Serialize `engine_overrides` to a JSON string for pool worker keying.
    ///
    /// Returns an empty string when no overrides are set (matching the pool
    /// config's default). This ensures `pre_scale_with_overrides` produces
    /// the same key that `dispatch_execute_v2` will look up.
    pub fn engine_overrides_json(&self) -> String {
        if self.engine_overrides.is_empty() {
            String::new()
        } else {
            serde_json::to_string(&self.engine_overrides).unwrap_or_default()
        }
    }
}

impl Default for CommonOptions {
    fn default() -> Self {
        Self {
            override_cache: false,
            lazy_audio: true,
            engine_overrides: BTreeMap::new(),
            mwt: BTreeMap::new(),
            debug_dir: None,
            override_cache_tasks: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-command option structs
// ---------------------------------------------------------------------------

/// How `+<` overlap utterances are handled during UTR.
///
/// Selects the alignment strategy for utterance timing recovery. The trait-based
/// architecture in `batchalign-chat-ops` allows plugging in different strategies
/// at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UtrOverlapStrategy {
    /// Automatically select: two-pass when `+<` utterances are present,
    /// global otherwise.
    #[default]
    Auto,
    /// Single global DP pass. `+<` utterances get no special treatment.
    Global,
    /// Two-pass overlap-aware strategy. Pass 1 excludes `+<` utterances,
    /// pass 2 recovers their timing from the predecessor's audio window.
    TwoPass,
}

/// Options for the `align` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// FA engine selector (`wav2vec_fa`, `whisper_fa`, or plugin name).
    #[serde(default = "default_fa_engine")]
    pub fa_engine: FaEngineName,

    /// UTR engine selection.
    ///
    /// `None` means utterance timing recovery is disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub utr_engine: Option<UtrEngine>,

    /// How `+<` overlap utterances are handled during UTR.
    #[serde(default)]
    pub utr_overlap_strategy: UtrOverlapStrategy,

    /// Two-pass UTR configuration (CA markers, density threshold, buffers).
    #[serde(default)]
    pub utr_two_pass: batchalign_chat_ops::fa::TwoPassConfig,

    /// Include pause durations in forced alignment.
    #[serde(default)]
    pub pauses: bool,

    /// Generate `%wor` tier with word-level timing bullets.
    #[serde(default = "default_wor_tier_include")]
    pub wor: WorTierPolicy,

    /// Merge abbreviated forms during processing.
    #[serde(default)]
    pub merge_abbrev: MergeAbbrevPolicy,

    /// Directory to search for media files (audio/video).
    /// When set, the aligner looks here in addition to the standard
    /// media resolution paths (alongside .cha file, server media roots).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_dir: Option<String>,
}

impl Default for AlignOptions {
    fn default() -> Self {
        Self {
            common: CommonOptions::default(),
            fa_engine: default_fa_engine(),
            utr_engine: None,
            utr_overlap_strategy: UtrOverlapStrategy::default(),
            utr_two_pass: Default::default(),
            pauses: false,
            wor: default_wor_tier_include(),
            merge_abbrev: MergeAbbrevPolicy::default(),
            media_dir: None,
        }
    }
}

impl AlignOptions {
    /// Get the two-pass UTR configuration.
    pub fn two_pass_config(&self) -> &batchalign_chat_ops::fa::TwoPassConfig {
        &self.utr_two_pass
    }

    /// Return the effective FA engine after applying any shared `fa` override.
    pub fn effective_fa_engine(&self) -> FaEngineName {
        self.common
            .engine_overrides
            .get("fa")
            .map(|value| FaEngineName::from_wire_name(value))
            .unwrap_or_else(|| self.fa_engine.clone())
    }
}

/// Options for the `transcribe` and `transcribe_s` commands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscribeOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// ASR engine selector (`rev`, `whisper`, `whisperx`, `whisper_oai`, or
    /// plugin name).
    #[serde(default = "default_asr_engine")]
    pub asr_engine: AsrEngineName,

    /// Enable speaker diarization.
    #[serde(default)]
    pub diarize: bool,

    /// Generate `%wor` tier with word-level timing bullets.
    #[serde(default)]
    pub wor: WorTierPolicy,

    /// Merge abbreviated forms during processing.
    #[serde(default)]
    pub merge_abbrev: MergeAbbrevPolicy,

    /// Whisper batch size.
    #[serde(default = "default_batch_size")]
    pub batch_size: i32,
}

impl TranscribeOptions {
    /// Return the effective ASR engine after applying any shared `asr` override.
    pub fn effective_asr_engine(&self) -> AsrEngineName {
        self.common
            .engine_overrides
            .get("asr")
            .map(|value| AsrEngineName::from_wire_name(value))
            .unwrap_or_else(|| self.asr_engine.clone())
    }
}

/// Options for the `morphotag` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MorphotagOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// Re-tokenize words before morphosyntactic analysis.
    #[serde(default)]
    pub retokenize: bool,

    /// Skip files with multiple `@Languages`.
    #[serde(default)]
    pub skipmultilang: bool,

    /// Merge abbreviated forms during processing.
    #[serde(default)]
    pub merge_abbrev: MergeAbbrevPolicy,
}

/// Options for the `translate` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslateOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// Merge abbreviated forms during processing.
    #[serde(default)]
    pub merge_abbrev: MergeAbbrevPolicy,
}

/// Options for the `coref` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorefOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// Merge abbreviated forms during processing.
    #[serde(default)]
    pub merge_abbrev: MergeAbbrevPolicy,
}

/// Options for the `utseg` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UtsegOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// Merge abbreviated forms during processing.
    #[serde(default)]
    pub merge_abbrev: MergeAbbrevPolicy,
}

/// Options for the `benchmark` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// ASR engine selector.
    #[serde(default = "default_asr_engine")]
    pub asr_engine: AsrEngineName,

    /// Generate `%wor` tier with word-level timing bullets.
    #[serde(default)]
    pub wor: WorTierPolicy,

    /// Merge abbreviated forms during processing.
    #[serde(default)]
    pub merge_abbrev: MergeAbbrevPolicy,
}

impl BenchmarkOptions {
    /// Return the effective ASR engine after applying any shared `asr` override.
    pub fn effective_asr_engine(&self) -> AsrEngineName {
        self.common
            .engine_overrides
            .get("asr")
            .map(|value| AsrEngineName::from_wire_name(value))
            .unwrap_or_else(|| self.asr_engine.clone())
    }
}

/// Options for the `opensmile` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpensmileOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// Feature set to extract (e.g. `"eGeMAPSv02"`, `"ComParE_2016"`).
    #[serde(default = "default_feature_set")]
    pub feature_set: String,
}

/// Options for the `compare` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompareOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// Merge abbreviated forms during processing.
    #[serde(default)]
    pub merge_abbrev: MergeAbbrevPolicy,
}

/// Options for the `avqi` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AvqiOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,
}

// ---------------------------------------------------------------------------
// CommandOptions tagged enum
// ---------------------------------------------------------------------------

/// Typed per-command options with an internally-tagged `command` discriminator.
///
/// Each variant holds a struct with all options for that command. The `command`
/// tag in the JSON matches the job submission command name, enabling
/// deserialization from the wire format without a separate `command` field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "lowercase")]
pub enum CommandOptions {
    /// `align` — forced alignment.
    Align(AlignOptions),
    /// `transcribe` — ASR transcription.
    Transcribe(TranscribeOptions),
    /// `transcribe_s` — ASR with speaker diarization.
    #[serde(rename = "transcribe_s")]
    TranscribeS(TranscribeOptions),
    /// `translate` — translation.
    Translate(TranslateOptions),
    /// `morphotag` — morphosyntactic analysis.
    Morphotag(MorphotagOptions),
    /// `coref` — coreference resolution.
    Coref(CorefOptions),
    /// `utseg` — utterance segmentation.
    Utseg(UtsegOptions),
    /// `benchmark` — ASR benchmarking.
    Benchmark(BenchmarkOptions),
    /// `opensmile` — audio feature extraction.
    Opensmile(OpensmileOptions),
    /// `compare` — transcript comparison against gold standard.
    Compare(CompareOptions),
    /// `avqi` — voice quality index.
    Avqi(AvqiOptions),
}

impl CommandOptions {
    /// Get the common options shared by all commands.
    pub fn common(&self) -> &CommonOptions {
        match self {
            Self::Align(o) => &o.common,
            Self::Transcribe(o) | Self::TranscribeS(o) => &o.common,
            Self::Translate(o) => &o.common,
            Self::Morphotag(o) => &o.common,
            Self::Coref(o) => &o.common,
            Self::Utseg(o) => &o.common,
            Self::Benchmark(o) => &o.common,
            Self::Opensmile(o) => &o.common,
            Self::Compare(o) => &o.common,
            Self::Avqi(o) => &o.common,
        }
    }

    /// Get a mutable reference to the common options.
    pub fn common_mut(&mut self) -> &mut CommonOptions {
        match self {
            Self::Align(o) => &mut o.common,
            Self::Transcribe(o) | Self::TranscribeS(o) => &mut o.common,
            Self::Translate(o) => &mut o.common,
            Self::Morphotag(o) => &mut o.common,
            Self::Coref(o) => &mut o.common,
            Self::Utseg(o) => &mut o.common,
            Self::Benchmark(o) => &mut o.common,
            Self::Opensmile(o) => &mut o.common,
            Self::Compare(o) => &mut o.common,
            Self::Avqi(o) => &mut o.common,
        }
    }

    /// Abbreviation-merging policy for this command.
    ///
    /// Commands without this option use [`MergeAbbrevPolicy::Keep`].
    pub fn merge_abbrev_policy(&self) -> MergeAbbrevPolicy {
        match self {
            Self::Align(o) => o.merge_abbrev,
            Self::Transcribe(o) | Self::TranscribeS(o) => o.merge_abbrev,
            Self::Translate(o) => o.merge_abbrev,
            Self::Morphotag(o) => o.merge_abbrev,
            Self::Coref(o) => o.merge_abbrev,
            Self::Utseg(o) => o.merge_abbrev,
            Self::Benchmark(o) => o.merge_abbrev,
            Self::Compare(o) => o.merge_abbrev,
            Self::Opensmile(_) | Self::Avqi(_) => MergeAbbrevPolicy::Keep,
        }
    }

    /// Whether abbreviation merging is enabled for this command.
    pub fn merge_abbrev(&self) -> bool {
        self.merge_abbrev_policy().should_merge()
    }

    /// Get the command name as a string (matches the serde tag value).
    pub fn command_name(&self) -> &'static str {
        match self {
            Self::Align(_) => "align",
            Self::Transcribe(_) => "transcribe",
            Self::TranscribeS(_) => "transcribe_s",
            Self::Translate(_) => "translate",
            Self::Morphotag(_) => "morphotag",
            Self::Coref(_) => "coref",
            Self::Utseg(_) => "utseg",
            Self::Benchmark(_) => "benchmark",
            Self::Opensmile(_) => "opensmile",
            Self::Compare(_) => "compare",
            Self::Avqi(_) => "avqi",
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morphotag_roundtrip() {
        let opts = CommandOptions::Morphotag(MorphotagOptions {
            common: CommonOptions::default(),
            retokenize: true,
            skipmultilang: false,
            merge_abbrev: false.into(),
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn align_roundtrip() {
        let opts = CommandOptions::Align(AlignOptions {
            common: CommonOptions::default(),
            fa_engine: "whisper_fa".into(),
            utr_engine: Some(UtrEngine::RevAi),
            utr_overlap_strategy: Default::default(),
            utr_two_pass: Default::default(),
            pauses: true,
            wor: true.into(),
            merge_abbrev: false.into(),
            media_dir: None,
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn transcribe_roundtrip() {
        let opts = CommandOptions::Transcribe(TranscribeOptions {
            common: CommonOptions::default(),
            asr_engine: "whisperx".into(),
            diarize: true,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 16,
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn transcribe_s_roundtrip() {
        let opts = CommandOptions::TranscribeS(TranscribeOptions {
            common: CommonOptions::default(),
            asr_engine: "rev".into(),
            diarize: true,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 8,
        });
        let json = serde_json::to_string(&opts).unwrap();
        assert!(json.contains("\"command\":\"transcribe_s\""));
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn command_name_matches_tag() {
        let cases: Vec<(CommandOptions, &str)> = vec![
            (
                CommandOptions::Align(AlignOptions {
                    common: CommonOptions::default(),
                    fa_engine: "wav2vec_fa".into(),
                    utr_engine: None,
                    utr_overlap_strategy: Default::default(),
            utr_two_pass: Default::default(),
                    pauses: false,
                    wor: true.into(),
                    merge_abbrev: false.into(),
                    media_dir: None,
                }),
                "align",
            ),
            (
                CommandOptions::Morphotag(MorphotagOptions {
                    common: CommonOptions::default(),
                    retokenize: false,
                    skipmultilang: false,
                    merge_abbrev: false.into(),
                }),
                "morphotag",
            ),
            (
                CommandOptions::Opensmile(OpensmileOptions {
                    common: CommonOptions::default(),
                    feature_set: "eGeMAPSv02".into(),
                }),
                "opensmile",
            ),
            (
                CommandOptions::Compare(CompareOptions {
                    common: CommonOptions::default(),
                    merge_abbrev: false.into(),
                }),
                "compare",
            ),
            (
                CommandOptions::Avqi(AvqiOptions {
                    common: CommonOptions::default(),
                }),
                "avqi",
            ),
        ];

        for (opts, expected_name) in cases {
            assert_eq!(opts.command_name(), expected_name);
            let json = serde_json::to_string(&opts).unwrap();
            assert!(
                json.contains(&format!("\"command\":\"{expected_name}\"")),
                "JSON should contain command tag '{expected_name}': {json}"
            );
        }
    }

    #[test]
    fn common_accessor() {
        let opts = CommandOptions::Morphotag(MorphotagOptions {
            common: CommonOptions {
                override_cache: true,
                lazy_audio: false,
                engine_overrides: BTreeMap::new(),
                mwt: BTreeMap::new(),
                ..Default::default()
            },
            retokenize: true,
            skipmultilang: false,
            merge_abbrev: false.into(),
        });
        assert!(opts.common().override_cache);
        assert!(!opts.common().lazy_audio);
    }

    #[test]
    fn engine_overrides_roundtrip() {
        let mut overrides = BTreeMap::new();
        overrides.insert("asr".into(), "tencent".into());
        overrides.insert("fa".into(), "cantonese_fa".into());

        let opts = CommandOptions::Align(AlignOptions {
            common: CommonOptions {
                override_cache: false,
                lazy_audio: true,
                engine_overrides: overrides.clone(),
                mwt: BTreeMap::new(),
                ..Default::default()
            },
            fa_engine: "cantonese_fa".into(),
            ..AlignOptions::default()
        });

        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(back.common().engine_overrides, overrides);
    }

    #[test]
    fn transcribe_asr_override_effective_engine_prefers_override() {
        let mut overrides = BTreeMap::new();
        overrides.insert("asr".into(), "tencent".into());
        let opts = TranscribeOptions {
            common: CommonOptions {
                engine_overrides: overrides,
                ..CommonOptions::default()
            },
            asr_engine: "rev".into(),
            diarize: false,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 8,
        };

        assert_eq!(
            opts.effective_asr_engine(),
            AsrEngineName::Custom(CustomEngineName::new("tencent"))
        );
    }

    #[test]
    fn benchmark_asr_override_effective_engine_prefers_override() {
        let mut overrides = BTreeMap::new();
        overrides.insert("asr".into(), "aliyun".into());
        let opts = BenchmarkOptions {
            common: CommonOptions {
                engine_overrides: overrides,
                ..CommonOptions::default()
            },
            asr_engine: "rev".into(),
            wor: true.into(),
            merge_abbrev: false.into(),
        };

        assert_eq!(
            opts.effective_asr_engine(),
            AsrEngineName::Custom(CustomEngineName::new("aliyun"))
        );
    }

    #[test]
    fn minimal_json_deserializes_with_defaults() {
        let json = r#"{"command": "morphotag"}"#;
        let opts: CommandOptions = serde_json::from_str(json).unwrap();
        assert_eq!(opts.command_name(), "morphotag");
        if let CommandOptions::Morphotag(m) = &opts {
            assert!(!m.retokenize);
            assert!(!m.skipmultilang);
            assert!(!m.merge_abbrev.should_merge());
            assert!(m.common.lazy_audio);
        } else {
            panic!("expected Morphotag");
        }
    }

    #[test]
    fn avqi_roundtrip() {
        let opts = CommandOptions::Avqi(AvqiOptions {
            common: CommonOptions::default(),
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn compare_roundtrip() {
        let opts = CommandOptions::Compare(CompareOptions {
            common: CommonOptions::default(),
            merge_abbrev: true.into(),
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn translate_roundtrip() {
        let opts = CommandOptions::Translate(TranslateOptions {
            common: CommonOptions::default(),
            merge_abbrev: true.into(),
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn coref_roundtrip() {
        let opts = CommandOptions::Coref(CorefOptions {
            common: CommonOptions::default(),
            merge_abbrev: false.into(),
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn utseg_roundtrip() {
        let opts = CommandOptions::Utseg(UtsegOptions {
            common: CommonOptions::default(),
            merge_abbrev: true.into(),
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn benchmark_roundtrip() {
        let opts = CommandOptions::Benchmark(BenchmarkOptions {
            common: CommonOptions::default(),
            asr_engine: "whisper_oai".into(),
            wor: true.into(),
            merge_abbrev: false.into(),
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn opensmile_roundtrip() {
        let opts = CommandOptions::Opensmile(OpensmileOptions {
            common: CommonOptions::default(),
            feature_set: "ComParE_2016".into(),
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn utr_engine_roundtrip_preserves_wire_names() {
        let rev_json = serde_json::to_string(&UtrEngine::RevAi).unwrap();
        let whisper_json = serde_json::to_string(&UtrEngine::Whisper).unwrap();
        let custom_json =
            serde_json::to_string(&UtrEngine::Custom(CustomEngineName::new("tencent_utr")))
                .unwrap();

        assert_eq!(rev_json, "\"rev_utr\"");
        assert_eq!(whisper_json, "\"whisper_utr\"");
        assert_eq!(custom_json, "\"tencent_utr\"");

        assert_eq!(
            serde_json::from_str::<UtrEngine>(&rev_json).unwrap(),
            UtrEngine::RevAi
        );
        assert_eq!(
            serde_json::from_str::<UtrEngine>(&whisper_json).unwrap(),
            UtrEngine::Whisper
        );
        assert_eq!(
            serde_json::from_str::<UtrEngine>(&custom_json).unwrap(),
            UtrEngine::Custom(CustomEngineName::new("tencent_utr"))
        );
    }

    #[test]
    fn fa_engine_roundtrip_preserves_wire_names() {
        let wav2vec_json = serde_json::to_string(&FaEngineName::Wave2Vec).unwrap();
        let whisper_json = serde_json::to_string(&FaEngineName::Whisper).unwrap();
        let custom_json =
            serde_json::to_string(&FaEngineName::Custom(CustomEngineName::new("cantonese_fa")))
                .unwrap();

        assert_eq!(wav2vec_json, "\"wav2vec_fa\"");
        assert_eq!(whisper_json, "\"whisper_fa\"");
        assert_eq!(custom_json, "\"cantonese_fa\"");

        assert_eq!(
            serde_json::from_str::<FaEngineName>(&wav2vec_json).unwrap(),
            FaEngineName::Wave2Vec
        );
        assert_eq!(
            serde_json::from_str::<FaEngineName>(&whisper_json).unwrap(),
            FaEngineName::Whisper
        );
        assert_eq!(
            serde_json::from_str::<FaEngineName>(&custom_json).unwrap(),
            FaEngineName::Custom(CustomEngineName::new("cantonese_fa"))
        );
    }

    #[test]
    fn asr_engine_roundtrip_preserves_wire_names() {
        let rev_json = serde_json::to_string(&AsrEngineName::RevAi).unwrap();
        let whisperx_json = serde_json::to_string(&AsrEngineName::WhisperX).unwrap();
        let custom_json =
            serde_json::to_string(&AsrEngineName::Custom(CustomEngineName::new("tencent")))
                .unwrap();

        assert_eq!(rev_json, "\"rev\"");
        assert_eq!(whisperx_json, "\"whisperx\"");
        assert_eq!(custom_json, "\"tencent\"");

        assert_eq!(
            serde_json::from_str::<AsrEngineName>(&rev_json).unwrap(),
            AsrEngineName::RevAi
        );
        assert_eq!(
            serde_json::from_str::<AsrEngineName>(&whisperx_json).unwrap(),
            AsrEngineName::WhisperX
        );
        assert_eq!(
            serde_json::from_str::<AsrEngineName>(&custom_json).unwrap(),
            AsrEngineName::Custom(CustomEngineName::new("tencent"))
        );
    }
}
