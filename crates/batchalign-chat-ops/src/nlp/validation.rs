//! Low-level sanitization checks for text inserted into %mor fields.
//!
//! `%mor` uses structural separators (`|`, `-`, `&`, `$`, `~`) that cannot
//! appear unescaped inside stems/lemmas/features. This helper provides the
//! final guardrail before assembly.
//!
//! # Related CHAT Manual Sections
//!
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Format>
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Headers>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Main_Tier>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Dependent_Tiers>
//!
/// Sanitizes a string for use in a MOR field by replacing structural separators with underscores.
///
/// This fulfills the "Individual Validation then Assembly" principle by ensuring
/// each component of a complex Mor item is syntactically safe before being
/// combined into the final AST node.
pub fn sanitize_mor_text(s: &str) -> String {
    let mut result = s.replace(['|', '#', '-', '&', '$', '~'], "_");
    result.retain(|c| !c.is_whitespace());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_replaces_structural_separators() {
        assert_eq!(sanitize_mor_text("foo|bar"), "foo_bar");
        assert_eq!(sanitize_mor_text("a#b-c&d$e~f"), "a_b_c_d_e_f");
    }

    #[test]
    fn sanitize_strips_whitespace() {
        // Stanza Japanese tokenizer produces lemmas like "ふ す"
        assert_eq!(sanitize_mor_text("ふ す"), "ふす");
        assert_eq!(sanitize_mor_text(" hello world "), "helloworld");
        assert_eq!(sanitize_mor_text("a\tb\nc"), "abc");
    }

    #[test]
    fn sanitize_handles_combined_issues() {
        assert_eq!(sanitize_mor_text("foo | bar"), "foo_bar");
        assert_eq!(sanitize_mor_text("ふ す#test"), "ふす_test");
    }

    #[test]
    fn sanitize_passthrough_clean_text() {
        assert_eq!(sanitize_mor_text("hello"), "hello");
        assert_eq!(sanitize_mor_text("ふす"), "ふす");
    }
}
