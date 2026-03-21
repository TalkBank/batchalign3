//! Engine backend types and traits.
//!
//! Closed enum sets for ASR, FA, and UTR engine selection.
//! No external plugin system — all engines are built-in.
//! The [`EngineBackend`] trait provides a common interface.

use serde::{Deserialize, Serialize};

/// Shared behavior for all engine backend selectors.
///
/// Implement this on each engine enum so generic code can work across
/// engine categories without knowing which specific enum it holds.
pub trait EngineBackend: std::fmt::Debug + Clone + Send + Sync + 'static {
    /// Stable wire-format name used in JSON, CLI args, and SQLite.
    fn wire_name(&self) -> &str;

    /// Whether this engine's inference is fully Rust-owned (no Python worker).
    fn is_rust_owned(&self) -> bool;

    /// Parse a wire-format name. Returns `None` for unrecognized names.
    fn try_from_wire_name(name: &str) -> Option<Self>
    where
        Self: Sized;
}

/// Error returned when a wire-format engine name is not recognized.
#[derive(Debug, Clone, thiserror::Error)]
#[error("unknown engine name \"{name}\" for {category}")]
pub struct UnknownEngineName {
    /// The unrecognized wire name.
    pub name: String,
    /// Which engine category was being parsed (e.g. "ASR", "FA", "UTR").
    pub category: &'static str,
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
    /// Tencent UTR (HK/Cantonese).
    HkTencent,
}

impl EngineBackend for UtrEngine {
    fn wire_name(&self) -> &str {
        match self {
            Self::RevAi => "rev_utr",
            Self::Whisper => "whisper_utr",
            Self::HkTencent => "tencent_utr",
        }
    }

    fn is_rust_owned(&self) -> bool {
        matches!(self, Self::RevAi)
    }

    fn try_from_wire_name(name: &str) -> Option<Self> {
        match name {
            "rev_utr" => Some(Self::RevAi),
            "whisper_utr" => Some(Self::Whisper),
            "tencent_utr" => Some(Self::HkTencent),
            _ => None,
        }
    }
}

impl UtrEngine {
    /// Parse one persisted wire-format token.
    pub fn from_wire_name(name: &str) -> Result<Self, UnknownEngineName> {
        Self::try_from_wire_name(name).ok_or_else(|| UnknownEngineName {
            name: name.to_owned(),
            category: "UTR",
        })
    }

    /// Borrow the wire-format token for JSON/SQLite.
    pub fn as_wire_name(&self) -> &str {
        self.wire_name()
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
        Self::from_wire_name(&name).map_err(serde::de::Error::custom)
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
    /// Wav2Vec Cantonese forced alignment (HK).
    Wav2vecCanto,
}

impl EngineBackend for FaEngineName {
    fn wire_name(&self) -> &str {
        match self {
            Self::Wave2Vec => "wav2vec_fa",
            Self::Whisper => "whisper_fa",
            Self::Wav2vecCanto => "cantonese_fa",
        }
    }

    fn is_rust_owned(&self) -> bool {
        false
    }

    fn try_from_wire_name(name: &str) -> Option<Self> {
        match name {
            "wav2vec_fa" | "wave2vec" => Some(Self::Wave2Vec),
            "whisper_fa" | "whisper" => Some(Self::Whisper),
            "cantonese_fa" | "wav2vec_canto" | "wav2vec_fa_canto" => Some(Self::Wav2vecCanto),
            _ => None,
        }
    }
}

impl FaEngineName {
    /// Parse one persisted wire-format token.
    pub fn from_wire_name(name: &str) -> Result<Self, UnknownEngineName> {
        Self::try_from_wire_name(name).ok_or_else(|| UnknownEngineName {
            name: name.to_owned(),
            category: "FA",
        })
    }

    /// Borrow the wire-format token for JSON/SQLite.
    pub fn as_wire_name(&self) -> &str {
        self.wire_name()
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
        Self::from_wire_name(&name).map_err(serde::de::Error::custom)
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
    /// Tencent Cloud ASR (HK/Cantonese).
    HkTencent,
    /// Aliyun ASR (HK/Cantonese).
    HkAliyun,
    /// FunAudio ASR (HK/Cantonese).
    HkFunaudio,
}

impl EngineBackend for AsrEngineName {
    fn wire_name(&self) -> &str {
        match self {
            Self::RevAi => "rev",
            Self::Whisper => "whisper",
            Self::WhisperX => "whisperx",
            Self::WhisperOai => "whisper_oai",
            Self::HkTencent => "tencent",
            Self::HkAliyun => "aliyun",
            Self::HkFunaudio => "funaudio",
        }
    }

    fn is_rust_owned(&self) -> bool {
        matches!(self, Self::RevAi)
    }

    fn try_from_wire_name(name: &str) -> Option<Self> {
        match name {
            "rev" => Some(Self::RevAi),
            "whisper" => Some(Self::Whisper),
            "whisperx" => Some(Self::WhisperX),
            "whisper_oai" => Some(Self::WhisperOai),
            "tencent" => Some(Self::HkTencent),
            "aliyun" => Some(Self::HkAliyun),
            "funaudio" => Some(Self::HkFunaudio),
            _ => None,
        }
    }
}

impl AsrEngineName {
    /// Parse one persisted wire-format token. Falls back to `try_from_wire_name`.
    pub fn from_wire_name(name: &str) -> Result<Self, UnknownEngineName> {
        Self::try_from_wire_name(name).ok_or_else(|| UnknownEngineName {
            name: name.to_owned(),
            category: "ASR",
        })
    }

    /// Borrow the wire-format token for JSON/SQLite.
    pub fn as_wire_name(&self) -> &str {
        self.wire_name()
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
        Self::from_wire_name(&name).map_err(serde::de::Error::custom)
    }
}

