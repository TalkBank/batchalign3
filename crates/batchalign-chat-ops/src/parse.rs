//! CHAT parsing wrappers.
//!
//! Thin wrappers around `talkbank-parser` for lenient and strict parsing.
//! Callers provide a `&TreeSitterParser` handle — create one per entry point
//! and share it across all parsing calls. Do not create throwaway handles.

use talkbank_model::model::ChatFile;
pub use talkbank_parser::TreeSitterParser;

/// Parse CHAT text leniently (tree-sitter with error recovery).
///
/// Always returns a `ChatFile` (best-effort), plus any parse warnings/errors.
pub fn parse_lenient(
    parser: &TreeSitterParser,
    chat_text: &str,
) -> (ChatFile, Vec<talkbank_model::ParseError>) {
    let errors = talkbank_model::ErrorCollector::new();
    let chat_file = parser.parse_chat_file_streaming(chat_text, &errors);
    let error_vec = errors.into_vec();
    (chat_file, error_vec)
}

/// Parse CHAT text strictly (tree-sitter, no error recovery).
pub fn parse_strict(
    parser: &TreeSitterParser,
    chat_text: &str,
) -> Result<ChatFile, talkbank_model::ParseErrors> {
    parser.parse_chat_file(chat_text)
}

/// Check whether a parsed CHAT file has `@Options: dummy`.
///
/// Dummy files are pass-through placeholders that should not be processed
/// by any NLP pipeline — they should be output unchanged.
///
/// The `dummy` option was removed from the CHAT spec, so it now parses as
/// `Unsupported("dummy")`. This function detects it for backward compatibility
/// with existing archive files.
pub fn is_dummy(chat_file: &ChatFile) -> bool {
    use talkbank_model::model::ChatOptionFlag;

    chat_file
        .options
        .iter()
        .any(|f| matches!(f, ChatOptionFlag::Unsupported(s) if s == "dummy"))
}

/// Check whether a parsed CHAT file has `@Options: NoAlign`.
///
/// Files with `NoAlign` should skip forced alignment and be output unchanged
/// by the `align` command. Other commands (morphotag, utseg, etc.) still apply.
pub fn is_no_align(chat_file: &ChatFile) -> bool {
    chat_file.options.iter().any(|f| f.skips_alignment())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_dummy_with_dummy_option() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../test-fixtures/eng_hello_world_dummy.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        assert!(is_dummy(&chat_file));
    }

    #[test]
    fn test_is_dummy_without_options() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../test-fixtures/eng_hello_world.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        assert!(!is_dummy(&chat_file));
    }

    #[test]
    fn test_is_dummy_with_other_options() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../test-fixtures/eng_hello_world_bullets_option.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        assert!(!is_dummy(&chat_file));
    }

    #[test]
    fn test_is_no_align_with_noalign_option() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../test-fixtures/eng_hello_world_noalign_option.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        assert!(is_no_align(&chat_file));
    }

    #[test]
    fn test_is_no_align_without_options() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../test-fixtures/eng_hello_world.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        assert!(!is_no_align(&chat_file));
    }

    #[test]
    fn test_is_no_align_with_dummy_option() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../test-fixtures/eng_hello_world_dummy.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        assert!(!is_no_align(&chat_file));
    }
}
