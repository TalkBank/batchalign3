//! Stanza language support registry.
//!
//! Maintains the canonical set of ISO 639-3 language codes that the Python
//! Stanza worker can process for morphosyntax.  This is the Rust-side mirror
//! of `batchalign/worker/_stanza_loading.py::iso3_to_alpha2()`.
//!
//! Used by the batch dispatcher to reject unsupported languages **before**
//! attempting to spawn workers, preventing deadlocks and wasted resources.

use std::collections::HashSet;
use std::sync::LazyLock;

use crate::LanguageCode;

/// ISO 639-3 codes that have a known Stanza alpha-2 mapping.
///
/// This table must stay in sync with `_stanza_loading.py::iso3_to_alpha2()`.
/// If a code is missing here but present in Python, utterances with that
/// language will be silently skipped during morphotag.  If a code is present
/// here but missing in Python, the worker will crash on model load.
///
/// The single source of truth is the Python mapping table.  This Rust-side
/// copy is a preflight filter to avoid dispatching work that will fail.
static SUPPORTED_STANZA_CODES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        // Western European
        "eng", "spa", "fra", "deu", "ita", "por", "nld", "cat", "glg",
        // Nordic / Baltic
        "dan", "swe", "nor", "fin", "est", "lav", "lit", "isl",
        // Central / Eastern European
        "pol", "ces", "ron", "hun", "bul", "hrv", "slk", "slv", "ukr", "rus",
        // Greek, Celtic, Maltese
        "ell", "cym", "gle", "gla", "eus", "mlt", // Middle Eastern / South Asian
        "ara", "heb", "fas", "hin", "urd", "tur", // South / Southeast Asian
        "ben", "tam", "tel", "kan", "mal", "tha", "vie", "ind", "msa", "tgl",
        // East Asian
        "zho", "cmn", "yue", "jpn", "kor", // Caucasian / Armenian
        "kat", "hye", // African
        "afr", // Classical
        "lat", // Luxembourgish
        "ltz",
    ]
    .into_iter()
    .collect()
});

/// Check whether a language code is supported by the Stanza worker.
///
/// Returns `true` for codes that have a known mapping to a Stanza alpha-2
/// code in `_stanza_loading.py`.  Returns `false` for unknown or truly
/// unsupported languages (e.g. Quechua, Jamaican Creole, Tamasheq).
pub fn is_stanza_supported(lang: &LanguageCode) -> bool {
    SUPPORTED_STANZA_CODES.contains(lang.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_languages_supported() {
        for code in ["eng", "spa", "fra", "deu", "zho", "jpn", "rus", "ara"] {
            assert!(
                is_stanza_supported(&LanguageCode::new(code)),
                "{code} should be supported"
            );
        }
    }

    #[test]
    fn unsupported_languages_rejected() {
        for code in ["que", "jam", "nan", "taq", "und", "xmm", "jav", "wuu"] {
            assert!(
                !is_stanza_supported(&LanguageCode::new(code)),
                "{code} should NOT be supported"
            );
        }
    }

    #[test]
    fn yue_and_cmn_both_supported() {
        assert!(is_stanza_supported(&LanguageCode::new("yue")));
        assert!(is_stanza_supported(&LanguageCode::new("cmn")));
    }
}
