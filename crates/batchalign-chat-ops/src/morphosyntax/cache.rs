//! Cache key computation, cache injection, and string extraction.

use std::collections::BTreeMap;

use talkbank_model::model::{LanguageCode, Line};

use super::{CachedMorphosyntaxEntry, MorphosyntaxStringsEntry};
use crate::CacheKey;

/// Alias for the MWT lexicon: surface form -> expansion tokens.
pub type MwtDict = BTreeMap<String, Vec<String>>;

// ---------------------------------------------------------------------------
// Cache key computation
// ---------------------------------------------------------------------------

/// Morphosyntax pipeline version for cache invalidation.
///
/// Bump this constant whenever the morphosyntax pipeline changes in a way that
/// would produce different results for the same input. Examples:
///
/// - Adding or removing POS post-processing (e.g., PyCantonese POS override)
/// - Changing the UD→CHAT mapping rules
/// - Changing Stanza model selection for a language
///
/// This is appended to the cache key so old cached results are automatically
/// ignored after a pipeline change.
///
/// ## Change Log
///
/// | Version | Date       | Change |
/// |---------|------------|--------|
/// | 1       | pre-2026   | Original Stanza-only pipeline |
/// | 2       | 2026-03-23 | Added PyCantonese POS override for Cantonese (yue) |
/// | 3       | 2026-03-27 | Per-language dispatch for multilingual files |
const MORPHOSYNTAX_PIPELINE_VERSION: u32 = 3;

/// Compute the cache key for a morphosyntax payload.
///
/// Formula: `BLAKE3("v{VERSION}|{words}|{lang}|mwt:{entries}[|retok]")`.
/// The pipeline version prefix ensures cache invalidation when the pipeline
/// changes (e.g., new POS post-processing, model changes).
///
/// Uses incremental hashing to avoid intermediate `String` allocation from `join()`.
pub fn cache_key(
    words: &[String],
    lang: &LanguageCode,
    mwt: &MwtDict,
    retokenize: bool,
) -> CacheKey {
    let mut hasher = blake3::Hasher::new();
    // Pipeline version prefix — invalidates all old cache entries on bump.
    hasher.update(b"v");
    hasher.update(MORPHOSYNTAX_PIPELINE_VERSION.to_string().as_bytes());
    hasher.update(b"|");
    for (i, w) in words.iter().enumerate() {
        if i > 0 {
            hasher.update(b" ");
        }
        hasher.update(w.as_bytes());
    }
    hasher.update(b"|");
    hasher.update(lang.as_bytes());
    hasher.update(b"|mwt:");
    // BTreeMap iteration is sorted by key, so output is deterministic.
    for (key, vals) in mwt {
        hasher.update(key.as_bytes());
        hasher.update(b"=");
        for (i, v) in vals.iter().enumerate() {
            if i > 0 {
                hasher.update(b"+");
            }
            hasher.update(v.as_bytes());
        }
        hasher.update(b";");
    }
    if retokenize {
        hasher.update(b"|retok");
    }
    CacheKey::from_hasher(hasher)
}

// ---------------------------------------------------------------------------
// Cache injection
// ---------------------------------------------------------------------------

/// Inject cached %mor/%gra tiers into specific utterances (pure Rust).
///
/// # Errors
///
/// Returns `Err` if:
/// - `data_json` is not valid JSON or does not match the expected
///   `CachedMorphosyntaxEntry` schema.
/// - A `line_idx` is out of range or does not point to an utterance.
/// - A cached `MorTier` or `GraTier` cannot be deserialized.
/// - Cached `%mor` and `%gra` have mismatched chunk counts (stale cache).
pub fn inject_from_cache(
    chat_file: &mut talkbank_model::model::ChatFile,
    data_json: &str,
) -> Result<(), String> {
    use talkbank_model::WriteChat;
    use talkbank_model::model::{DependentTier, GraTier, MorTier};

    let entries: Vec<CachedMorphosyntaxEntry> = serde_json::from_str(data_json)
        .map_err(|e| format!("Failed to parse cache injection JSON: {e}"))?;

    for entry in entries {
        let line = chat_file
            .lines
            .get_mut(entry.line_idx)
            .ok_or_else(|| format!("Line index {} out of range", entry.line_idx))?;

        let utt = match line {
            Line::Utterance(u) => u,
            _ => {
                return Err(format!(
                    "Line at index {} is not an utterance",
                    entry.line_idx
                ));
            }
        };

        if entry.mor.is_empty() {
            continue;
        }

        let mut mor_tier: MorTier = serde_json::from_str(&entry.mor).map_err(|e| {
            format!(
                "Failed to deserialize cached MorTier at line {}: {e}",
                entry.line_idx
            )
        })?;

        // Patch the cached MorTier's terminator to match the current utterance's
        // main tier.
        mor_tier.terminator = utt
            .main
            .content
            .terminator
            .as_ref()
            .map(|t| t.to_chat_string().into());

        // Validate MOR/GRA chunk alignment before injection.
        if !entry.gra.is_empty() {
            let gra_tier: GraTier = serde_json::from_str(&entry.gra).map_err(|e| {
                format!(
                    "Failed to deserialize cached GraTier at line {}: {e}",
                    entry.line_idx
                )
            })?;

            let mor_chunks = mor_tier.count_chunks();
            let gra_count = gra_tier.len();
            if mor_chunks != gra_count {
                return Err(format!(
                    "Cached MOR/GRA mismatch at line {}: %mor has {} chunks but %gra has {} relations. \
                     Cache may be stale — re-run with --override-cache.",
                    entry.line_idx, mor_chunks, gra_count
                ));
            }

            crate::inject::replace_or_add_tier(
                &mut utt.dependent_tiers,
                DependentTier::Mor(mor_tier),
            );
            crate::inject::replace_or_add_tier(
                &mut utt.dependent_tiers,
                DependentTier::Gra(gra_tier),
            );
        } else {
            crate::inject::replace_or_add_tier(
                &mut utt.dependent_tiers,
                DependentTier::Mor(mor_tier),
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// String extraction
// ---------------------------------------------------------------------------

/// Extract final %mor/%gra content as serde JSON for specified utterances.
///
/// # Errors
///
/// Returns `Err` if a `line_idx` is out of range, does not point to an
/// utterance, or if a `MorTier`/`GraTier` cannot be serialized to JSON.
pub fn extract_strings(
    chat_file: &talkbank_model::model::ChatFile,
    line_indices: &[usize],
) -> Result<Vec<MorphosyntaxStringsEntry>, String> {
    use talkbank_model::model::DependentTier;

    let mut results: Vec<MorphosyntaxStringsEntry> = Vec::with_capacity(line_indices.len());

    for &line_idx in line_indices {
        let line = chat_file
            .lines
            .get(line_idx)
            .ok_or_else(|| format!("Line index {} out of range", line_idx))?;

        let utt = match line {
            Line::Utterance(u) => u,
            _ => {
                return Err(format!("Line at index {} is not an utterance", line_idx));
            }
        };

        let mut mor_json = String::new();
        let mut gra_json = String::new();

        for tier in &utt.dependent_tiers {
            match tier {
                DependentTier::Mor(mor_tier) => {
                    mor_json = serde_json::to_string(mor_tier).map_err(|e| {
                        format!("Failed to serialize MorTier at line {line_idx}: {e}")
                    })?;
                }
                DependentTier::Gra(gra_tier) => {
                    gra_json = serde_json::to_string(gra_tier).map_err(|e| {
                        format!("Failed to serialize GraTier at line {line_idx}: {e}")
                    })?;
                }
                _ => {}
            }
        }

        results.push(MorphosyntaxStringsEntry {
            line_idx,
            mor: mor_json,
            gra: gra_json,
        });
    }

    Ok(results)
}
