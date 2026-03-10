//! Mapping context and language-code normalization.

/// Context for the UD-to-CHAT mapping process.
pub struct MappingContext {
    /// Language code used to select language-specific override rules.
    pub lang: talkbank_model::model::LanguageCode,
}

/// Normalize a language code to its 2-letter form.
///
/// The pipeline passes 3-letter ISO 639-3 codes ("eng", "fra", "jpn")
/// but Python master's UD handler uses 2-letter ISO 639-1 codes ("en", "fr", "ja").
/// This helper maps the common 3-letter codes to 2-letter equivalents.
/// Unknown codes are returned as-is (truncated to 2 chars if ≥3).
pub(crate) fn lang2(code: &str) -> &str {
    match code {
        "eng" => "en",
        "fra" | "fre" => "fr",
        "jpn" => "ja",
        "deu" | "ger" => "de",
        "ita" => "it",
        "spa" => "es",
        "por" => "pt",
        "zho" | "cmn" | "chi" => "zh",
        "heb" => "he",
        "ara" => "ar",
        "nld" | "dut" => "nl",
        "cat" => "ca",
        // If already a 2-letter code, return as-is
        s if s.len() <= 2 => s,
        // Fallback: return the full code (won't match known checks but won't panic)
        s => s,
    }
}
