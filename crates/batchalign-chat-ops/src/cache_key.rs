//! Newtypes for cache key and task name, ensuring type safety across the
//! cache boundary.
//!
//! [`CacheKey`] wraps a BLAKE3 hash hex string. There is no constructor from
//! arbitrary strings — the only way to create one is via the task-specific
//! `cache_key()` functions in sibling modules, which compute the hash
//! internally through [`CacheKey::from_content`].
//!
//! [`CacheTaskName`] enumerates every NLP task that stores results in the
//! utterance cache, with wire strings matching the Python `CacheManager`
//! schema.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CacheKey
// ---------------------------------------------------------------------------

/// A content-derived BLAKE3 hash used to index the utterance cache.
///
/// # Invariant
///
/// Always a 64-character lowercase hexadecimal string (256-bit BLAKE3 hash).
/// There is no constructor from arbitrary strings — the only way to create
/// a `CacheKey` is via the task-specific `cache_key()` functions, which
/// compute the hash internally.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CacheKey(String);

impl CacheKey {
    /// Create a cache key by hashing the given content with BLAKE3.
    pub(crate) fn from_content(content: &str) -> Self {
        Self(blake3::hash(content.as_bytes()).to_hex().to_string())
    }

    /// Create a cache key from an incremental hasher that has already been fed
    /// content. Avoids allocating intermediate `String` from `join()`.
    pub(crate) fn from_hasher(hasher: blake3::Hasher) -> Self {
        Self(hasher.finalize().to_hex().to_string())
    }

    /// View the hex string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// CacheTaskName
// ---------------------------------------------------------------------------

/// Identifies the NLP task whose result is being cached.
///
/// Each variant maps to a fixed wire string matching the Python
/// `CacheManager` schema. Used as the `task` parameter in cache lookups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CacheTaskName {
    /// Morphosyntax (%mor/%gra tagging). Wire name: `"morphosyntax_v4"`.
    Morphosyntax,
    /// Utterance segmentation. Wire name: `"utterance_segmentation"`.
    UtteranceSegmentation,
    /// Translation (%xtra tier). Wire name: `"translation"`.
    Translation,
    /// Forced alignment (word timings). Wire name: `"forced_alignment"`.
    ForcedAlignment,
    /// UTR ASR result (full-file ASR for timing recovery). Wire name: `"utr_asr"`.
    UtrAsr,
}

impl CacheTaskName {
    /// The wire string stored in the cache database.
    ///
    /// Changing any of these values invalidates existing cache entries.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Morphosyntax => "morphosyntax_v4",
            Self::UtteranceSegmentation => "utterance_segmentation",
            Self::Translation => "translation",
            Self::ForcedAlignment => "forced_alignment",
            Self::UtrAsr => "utr_asr",
        }
    }
}

impl std::fmt::Display for CacheTaskName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_from_content_is_64_hex_chars() {
        let key = CacheKey::from_content("hello|eng|mwt");
        assert_eq!(key.as_str().len(), 64);
        assert!(key.as_str().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn cache_key_from_content_deterministic() {
        let a = CacheKey::from_content("test input");
        let b = CacheKey::from_content("test input");
        assert_eq!(a, b);
    }

    #[test]
    fn cache_key_from_content_differs_for_different_input() {
        let a = CacheKey::from_content("input A");
        let b = CacheKey::from_content("input B");
        assert_ne!(a, b);
    }

    #[test]
    fn cache_key_from_hasher_matches_from_content() {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"hello|eng|mwt");
        let from_hasher = CacheKey::from_hasher(hasher);
        let from_content = CacheKey::from_content("hello|eng|mwt");
        assert_eq!(from_hasher, from_content);
    }

    #[test]
    fn cache_key_display_matches_as_str() {
        let key = CacheKey::from_content("test");
        assert_eq!(format!("{key}"), key.as_str());
    }

    #[test]
    fn cache_key_serde_roundtrip() {
        let key = CacheKey::from_content("test");
        let json = serde_json::to_string(&key).unwrap();
        let deserialized: CacheKey = serde_json::from_str(&json).unwrap();
        assert_eq!(key, deserialized);
    }

    #[test]
    fn cache_task_name_wire_strings_are_stable() {
        assert_eq!(CacheTaskName::Morphosyntax.as_str(), "morphosyntax_v4");
        assert_eq!(
            CacheTaskName::UtteranceSegmentation.as_str(),
            "utterance_segmentation"
        );
        assert_eq!(CacheTaskName::Translation.as_str(), "translation");
        assert_eq!(CacheTaskName::ForcedAlignment.as_str(), "forced_alignment");
        assert_eq!(CacheTaskName::UtrAsr.as_str(), "utr_asr");
    }

    #[test]
    fn cache_task_name_display_matches_as_str() {
        for variant in [
            CacheTaskName::Morphosyntax,
            CacheTaskName::UtteranceSegmentation,
            CacheTaskName::Translation,
            CacheTaskName::ForcedAlignment,
            CacheTaskName::UtrAsr,
        ] {
            assert_eq!(format!("{variant}"), variant.as_str());
        }
    }
}
