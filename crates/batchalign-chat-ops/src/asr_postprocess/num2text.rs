//! Number-to-text expansion for ASR post-processing.
//!
//! Converts digit strings in ASR output to their word-form equivalents.
//! Supports 12 languages via fallback lookup tables (NUM2LANG) plus
//! Chinese/Japanese via `num2chinese`.
//!
//! Also handles currency-prefixed numbers (e.g. "$12" → "twelve dollars").
//!
//! The expansion chain:
//! 1. Strip recognized currency prefix/suffix → expand digits → append currency word
//! 2. If not all digits → return as-is
//! 3. If Chinese/Japanese/Cantonese → `num2chinese`
//! 4. Otherwise → NUM2LANG table lookup (reverse key order, substring replacement)
//! 5. If no table → return original string

use std::collections::BTreeMap;
use std::sync::LazyLock;

use super::num2chinese::{ChineseScript, num2chinese};

/// Language-specific number-to-word tables loaded from JSON at compile time.
///
/// Keys are ISO 639-3 language codes. Values are BTreeMaps where keys are
/// number strings (e.g. "1", "21", "100") and values are word forms.
static NUM2LANG: LazyLock<BTreeMap<String, BTreeMap<String, String>>> =
    LazyLock::new(|| serde_json::from_str(include_str!("../../data/num2lang.json")).unwrap());

/// Currency symbols we recognize, mapped to their English word forms.
///
/// For non-English languages, ASR models overwhelmingly produce these same
/// ASCII/Unicode symbols. We expand to the English currency word since CHAT
/// format just needs *some* non-digit word — the morphosyntax pass will tag
/// it properly later.
const CURRENCY_PREFIXES: &[(&str, &str)] = &[
    ("$", "dollars"),
    ("€", "euros"),
    ("£", "pounds"),
    ("¥", "yen"),
    ("₹", "rupees"),
    ("₩", "won"),
    ("₽", "rubles"),
];

const CURRENCY_SUFFIXES: &[(&str, &str)] = &[("€", "euros"), ("₹", "rupees")];

/// Try to strip a currency symbol and expand the remaining digits.
///
/// Returns `Some(expanded)` if a currency symbol was found and the remainder
/// is a valid digit string. Returns `None` otherwise.
fn try_expand_currency(word: &str, lang: &str) -> Option<String> {
    // Check prefixes: "$12" → "twelve dollars"
    for &(symbol, currency_word) in CURRENCY_PREFIXES {
        if let Some(rest) = word.strip_prefix(symbol) {
            let rest = rest.trim();
            if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
                let expanded = expand_single_number(rest, lang);
                // Only use currency word if the number actually expanded
                if expanded != rest {
                    return Some(format!("{expanded} {currency_word}"));
                }
                // Number didn't expand (unknown language) — still strip the symbol
                return Some(format!("{rest} {currency_word}"));
            }
        }
    }

    // Check suffixes: "12€" → "twelve euros"
    for &(symbol, currency_word) in CURRENCY_SUFFIXES {
        if let Some(rest) = word.strip_suffix(symbol) {
            let rest = rest.trim();
            if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
                let expanded = expand_single_number(rest, lang);
                if expanded != rest {
                    return Some(format!("{expanded} {currency_word}"));
                }
                return Some(format!("{rest} {currency_word}"));
            }
        }
    }

    None
}

/// Expand a digit string to its word-form equivalent in the given language.
///
/// Handles:
/// - Pure digit strings ("42" → "forty-two")
/// - Dash-separated digit groups ("21-22" → "twenty-one twenty-two")
/// - Currency-prefixed numbers ("$12" → "twelve dollars")
///
/// Returns the original string if expansion is not possible.
///
/// # Arguments
/// * `word` - The word to potentially expand.
/// * `lang` - ISO 639-3 language code.
pub fn expand_number(word: &str, lang: &str) -> String {
    // Normalize dashes
    let normalized = word.replace(['—', '–'], "-");

    // Handle dash-separated digit groups (e.g. "21-22", "5—6").
    // Only split if ALL parts are pure digit strings — words containing
    // dashes (like French "quatre-vingt") must not be split.
    if normalized.contains('-') {
        let parts: Vec<&str> = normalized.split('-').collect();
        let all_digit_parts = parts.len() > 1
            && parts
                .iter()
                .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()));
        if all_digit_parts {
            let expanded: Vec<String> = parts
                .iter()
                .map(|part| expand_single_number(part, lang))
                .collect();
            return expanded.join(" ");
        }
    }

    // Try currency expansion before pure-digit check
    if let Some(expanded) = try_expand_currency(&normalized, lang) {
        return expanded;
    }

    expand_single_number(&normalized, lang)
}

/// Expand a single number string (no dashes).
fn expand_single_number(word: &str, lang: &str) -> String {
    if !word.chars().all(|c| c.is_ascii_digit()) || word.is_empty() {
        return word.to_string();
    }

    let lang_lower = lang.to_lowercase();

    // Chinese/Japanese/Cantonese: use num2chinese
    match lang_lower.as_str() {
        "zho" | "cmn" => {
            if let Ok(n) = word.parse::<u64>() {
                return num2chinese(n, ChineseScript::Simplified);
            }
        }
        "jpn" | "yue" => {
            if let Ok(n) = word.parse::<u64>() {
                return num2chinese(n, ChineseScript::Traditional);
            }
        }
        _ => {}
    }

    // NUM2LANG lookup: exact match first, then integer decomposition
    if let Some(table) = NUM2LANG.get(&lang_lower) {
        // 1. Exact table lookup (handles 1-99, hundreds, etc.)
        if let Some(value) = table.get(word) {
            return value.clone();
        }

        // 2. Integer decomposition for numbers beyond exact table entries
        if let Ok(n) = word.parse::<u64>()
            && let Some(expanded) = decompose_with_table(n, table)
        {
            return expanded;
        }
    }

    // No expansion possible — return original
    word.to_string()
}

/// Decompose a number into words using a lookup table.
///
/// Strategy: greedily subtract the largest table entry that fits,
/// building up the word form. E.g., 1234 → "one thousand two hundred
/// thirty-four" (if table has 1000, 200, 34 or 30+4).
fn decompose_with_table(mut n: u64, table: &BTreeMap<String, String>) -> Option<String> {
    if n == 0 {
        return table.get("0").cloned();
    }

    // Build a sorted list of (numeric_key, word) pairs, largest first
    let mut entries: Vec<(u64, &str)> = table
        .iter()
        .filter_map(|(k, v)| k.parse::<u64>().ok().map(|num| (num, v.as_str())))
        .filter(|(num, _)| *num > 0)
        .collect();
    entries.sort_by(|a, b| b.0.cmp(&a.0));

    let mut parts: Vec<String> = Vec::new();

    for &(key_num, word_form) in &entries {
        if key_num == 0 {
            continue;
        }
        if key_num <= n {
            if key_num >= 100 {
                // For hundreds/thousands: "two hundred", "three thousand"
                let multiplier = n / key_num;
                let remainder = n % key_num;
                if multiplier > 1 {
                    // Recursively expand the multiplier
                    if let Some(mult_word) = decompose_with_table(multiplier, table) {
                        parts.push(format!("{mult_word} {word_form}"));
                    } else {
                        // Can't expand multiplier — bail
                        return None;
                    }
                } else {
                    parts.push(word_form.to_string());
                }
                n = remainder;
                if n == 0 {
                    break;
                }
            } else {
                // For units/teens/tens: exact match
                parts.push(word_form.to_string());
                n -= key_num;
                if n == 0 {
                    break;
                }
            }
        }
    }

    if n > 0 {
        // Couldn't fully decompose
        return None;
    }

    if parts.is_empty() {
        return None;
    }

    Some(parts.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_digit_passthrough() {
        assert_eq!(expand_number("hello", "eng"), "hello");
        assert_eq!(expand_number("abc123", "eng"), "abc123");
    }

    #[test]
    fn test_english_basic() {
        assert_eq!(expand_number("5", "eng"), "five");
        assert_eq!(expand_number("1", "eng"), "one");
        assert_eq!(expand_number("10", "eng"), "ten");
        assert_eq!(expand_number("99", "eng"), "ninety-nine");
    }

    #[test]
    fn test_english_hundreds() {
        let result = expand_number("100", "eng");
        assert!(
            result.contains("hundred") || result.contains("one hundred"),
            "got: {result}"
        );
    }

    #[test]
    fn test_spanish() {
        assert_eq!(expand_number("1", "spa"), "uno");
        assert_eq!(expand_number("5", "spa"), "cinco");
    }

    #[test]
    fn test_chinese_simplified() {
        assert_eq!(expand_number("5", "zho"), "五");
        assert_eq!(expand_number("42", "zho"), "四十二");
        assert_eq!(expand_number("1000", "zho"), "一千");
    }

    #[test]
    fn test_japanese() {
        assert_eq!(expand_number("5", "jpn"), "五");
        assert_eq!(expand_number("10000", "jpn"), "一萬"); // traditional
    }

    #[test]
    fn test_dash_separated() {
        // "21-22" → expand each part
        let result = expand_number("21-22", "eng");
        assert!(result.contains(' '), "expected space-separated: {result}");
    }

    #[test]
    fn test_em_dash_normalized() {
        let result = expand_number("5—6", "eng");
        assert!(result.contains("five"), "got: {result}");
        assert!(result.contains("six"), "got: {result}");
    }

    #[test]
    fn test_unknown_language_passthrough() {
        assert_eq!(expand_number("42", "xxx"), "42");
    }

    #[test]
    fn test_num2lang_tables_loaded() {
        // Verify all 12 languages are present
        let expected = [
            "deu", "ell", "eng", "eus", "fra", "hrv", "ind", "jpn", "nld", "por", "spa", "tha",
        ];
        for lang in &expected {
            assert!(NUM2LANG.contains_key(*lang), "missing language: {lang}");
        }
    }

    #[test]
    fn test_empty_string() {
        assert_eq!(expand_number("", "eng"), "");
    }

    // --- currency expansion ---

    #[test]
    fn test_dollar_prefix() {
        let result = expand_number("$5", "eng");
        assert!(
            result.contains("five") && result.contains("dollars"),
            "got: {result}"
        );
    }

    #[test]
    fn test_dollar_large() {
        let result = expand_number("$100", "eng");
        assert!(
            result.contains("hundred") && result.contains("dollars"),
            "got: {result}"
        );
    }

    #[test]
    fn test_dollar_no_digits_in_output() {
        // The key requirement: currency-prefixed numbers must not contain
        // raw digits in the output (which would trigger E220).
        let result = expand_number("$5", "eng");
        assert!(
            !result.chars().any(|c| c.is_ascii_digit()),
            "output should not contain digits: {result}"
        );
    }

    #[test]
    fn test_euro_prefix() {
        let result = expand_number("€50", "eng");
        assert!(
            result.contains("fifty") && result.contains("euros"),
            "got: {result}"
        );
    }

    #[test]
    fn test_euro_suffix() {
        let result = expand_number("50€", "eng");
        assert!(
            result.contains("fifty") && result.contains("euros"),
            "got: {result}"
        );
    }

    #[test]
    fn test_pound_prefix() {
        let result = expand_number("£5", "eng");
        assert!(
            result.contains("five") && result.contains("pounds"),
            "got: {result}"
        );
    }

    #[test]
    fn test_yen_prefix() {
        let result = expand_number("¥1000", "jpn");
        assert!(
            result.contains("千") && result.contains("yen"),
            "got: {result}"
        );
    }

    #[test]
    fn test_currency_spanish() {
        let result = expand_number("$13", "spa");
        assert!(result.contains("dollars"), "got: {result}");
        assert!(
            !result.contains("13"),
            "digits should be expanded: {result}"
        );
    }

    #[test]
    fn test_dollar_sign_alone_passthrough() {
        // "$" with no digits should pass through
        assert_eq!(expand_number("$", "eng"), "$");
    }

    #[test]
    fn test_dollar_non_digit_passthrough() {
        // "$abc" should pass through
        assert_eq!(expand_number("$abc", "eng"), "$abc");
    }

    #[test]
    fn test_german() {
        assert_eq!(expand_number("1", "deu"), "eins");
        assert_eq!(expand_number("10", "deu"), "zehn");
    }

    #[test]
    fn test_french() {
        assert_eq!(expand_number("1", "fra"), "un");
        assert_eq!(expand_number("5", "fra"), "cinq");
    }

    // --- decomposition regression tests ---

    #[test]
    fn test_twelve_not_onetwo() {
        // Previously broken: substring replacement produced "onetwo"
        assert_eq!(expand_number("12", "eng"), "twelve");
    }

    #[test]
    fn test_thirteen_spanish() {
        assert_eq!(expand_number("13", "spa"), "trece");
    }

    #[test]
    fn test_large_number_english() {
        let result = expand_number("10000", "eng");
        // Should decompose: 10 * 1000 = "ten thousand"
        assert!(
            result.contains("thousand"),
            "10000 should contain 'thousand': {result}"
        );
        assert!(
            !result.chars().any(|c| c.is_ascii_digit()),
            "should not contain digits: {result}"
        );
    }

    #[test]
    fn test_250_english() {
        let result = expand_number("250", "eng");
        assert!(
            result.contains("two hundred") && result.contains("fifty"),
            "250 should be 'two hundred fifty': {result}"
        );
    }

    #[test]
    fn test_1234_english() {
        let result = expand_number("1234", "eng");
        assert!(
            !result.chars().any(|c| c.is_ascii_digit()),
            "should not contain digits: {result}"
        );
    }

    #[test]
    fn test_dollar_12_now_works() {
        // The original production failure: "$12" in Spanish
        let result = expand_number("$12", "spa");
        assert!(
            !result.contains("12"),
            "$12 in Spanish should not contain raw digits: {result}"
        );
        assert!(
            result.contains("dollars"),
            "should contain currency word: {result}"
        );
    }

    #[test]
    fn test_dollar_10000_english() {
        let result = expand_number("$10000", "eng");
        assert!(
            result.contains("thousand") && result.contains("dollars"),
            "got: {result}"
        );
    }

    // --- property tests ---

    use proptest::prelude::*;

    fn known_lang() -> impl Strategy<Value = &'static str> {
        prop_oneof![
            Just("eng"),
            Just("spa"),
            Just("fra"),
            Just("deu"),
            Just("zho"),
            Just("jpn"),
        ]
    }

    proptest! {
        /// Non-digit strings always pass through unchanged.
        #[test]
        fn non_digit_passthrough(word in "[a-zA-Z]{1,8}", lang in known_lang()) {
            let result = expand_number(&word, lang);
            prop_assert_eq!(result, word);
        }

        /// Idempotence: expand(expand(x)) == expand(x).
        /// Once expanded to words, re-expansion is a no-op (no digits).
        #[test]
        fn expand_is_idempotent(n in 0..1000u32, lang in known_lang()) {
            let s = n.to_string();
            let once = expand_number(&s, lang);
            let twice = expand_number(&once, lang);
            prop_assert_eq!(
                &once, &twice,
                "Not idempotent: '{}' -> '{}' -> '{}'", s, once, twice
            );
        }

        /// Small numbers (1-9) in known languages produce non-digit output.
        #[test]
        fn single_digit_expanded(n in 1..10u32, lang in known_lang()) {
            let s = n.to_string();
            let result = expand_number(&s, lang);
            prop_assert!(
                !result.chars().all(|c| c.is_ascii_digit()),
                "Digit {} not expanded for lang {}: '{}'", n, lang, result
            );
        }

        /// Empty string always returns empty string.
        #[test]
        fn empty_passthrough(_lang in known_lang()) {
            prop_assert_eq!(expand_number("", "eng"), "");
        }
    }
}
