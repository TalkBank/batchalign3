//! Small helpers for parsing Stanza tokens into CHAT AST nodes.

use talkbank_parser::TreeSitterParser;
use talkbank_model::ChatParser;
use talkbank_model::NullErrorSink;
use talkbank_model::model::{BracketedItem, UtteranceContent, Word};

use crate::extract::ExtractedWord;

/// Try to parse a Stanza token into an `UtteranceContent` item.
///
/// Tag-marker separators become `UtteranceContent::Separator`.
/// Terminators are skipped (return `None`).
/// Valid CHAT words become `UtteranceContent::Word`.
/// Unparseable tokens return `None` -- the caller should keep the original word.
pub(super) fn try_parse_token_as_utterance_content(
    text: &str,
    expected_terminator: Option<&str>,
    diagnostics: &mut Vec<String>,
) -> Option<UtteranceContent> {
    use talkbank_model::Span;
    use talkbank_model::model::Separator;

    match text {
        "," => Some(UtteranceContent::Separator(Separator::Comma {
            span: Span::DUMMY,
        })),
        "\u{201E}" => Some(UtteranceContent::Separator(Separator::Tag {
            span: Span::DUMMY,
        })),
        "\u{2021}" => Some(UtteranceContent::Separator(Separator::Vocative {
            span: Span::DUMMY,
        })),
        _ => {
            if handle_ending_punct_skip(text, expected_terminator, diagnostics) {
                None
            } else {
                try_parse_token_as_word(text, diagnostics)
                    .map(|w| UtteranceContent::Word(Box::new(w)))
            }
        }
    }
}

/// Try to parse a Stanza token into a `BracketedItem`.
///
/// Tag-marker separators (`,`, `\u{201E}`, `\u{2021}`) become `BracketedItem::Separator`.
/// Terminators (`.`, `?`, `!`, etc.) are skipped (return `None`).
/// Valid CHAT words become `BracketedItem::Word`.
/// Unparseable tokens return `None` -- the caller should keep the original.
pub(super) fn try_parse_token_as_bracketed_item(
    text: &str,
    expected_terminator: Option<&str>,
    diagnostics: &mut Vec<String>,
) -> Option<BracketedItem> {
    use talkbank_model::Span;
    use talkbank_model::model::Separator;

    // Tag-marker separators are not words.
    match text {
        "," => {
            return Some(BracketedItem::Separator(Separator::Comma {
                span: Span::DUMMY,
            }));
        }
        "\u{201E}" => {
            return Some(BracketedItem::Separator(Separator::Tag {
                span: Span::DUMMY,
            }));
        }
        "\u{2021}" => {
            return Some(BracketedItem::Separator(Separator::Vocative {
                span: Span::DUMMY,
            }));
        }
        _ => {}
    }

    // Terminators are skipped -- the utterance already has one.
    if handle_ending_punct_skip(text, expected_terminator, diagnostics) {
        return None;
    }

    try_parse_token_as_word(text, diagnostics).map(|w| BracketedItem::Word(Box::new(w)))
}

/// Returns true if `text` is a tag-marker separator character.
pub(super) fn is_tag_marker_text(text: &str) -> bool {
    matches!(text, "," | "\u{201E}" | "\u{2021}")
}

/// Returns true if `text` is a terminator character.
pub(super) fn is_ending_punct(text: &str) -> bool {
    matches!(
        text,
        "." | "?" | "!" | "+..." | "+/." | "+//." | "+/?" | "+//?" | "+..?" | "+\"." | "+\"/."
    )
}

/// Return true when the token is an ending punctuation symbol that should be
/// skipped during retokenization. Emits diagnostics if the skipped token does
/// not match the utterance's existing terminator.
pub(super) fn handle_ending_punct_skip(
    text: &str,
    expected_terminator: Option<&str>,
    diagnostics: &mut Vec<String>,
) -> bool {
    if !is_ending_punct(text) {
        return false;
    }

    if let Some(expected) = expected_terminator
        && text != expected
    {
        diagnostics.push(format!(
            "skipped Stanza terminator {text:?} does not match existing terminator {expected:?}; keeping existing terminator"
        ));
    }
    true
}

/// Try to parse a Stanza token as a valid CHAT Word.
///
/// Returns `Some(word)` if the token parses as valid CHAT syntax, `None`
/// otherwise. Callers should keep the original CHAT word when this returns
/// `None` -- that preserves the known-valid AST content rather than admitting
/// unparseable text via `Word::new_unchecked`.
///
/// # Tracing
///
/// Emits `tracing::warn!` when a token fails to parse, making these events
/// visible in `-vv` output for debugging retokenization issues.
pub(super) fn try_parse_token_as_word(text: &str, diagnostics: &mut Vec<String>) -> Option<Word> {
    let parser = TreeSitterParser::new().expect("TreeSitterParser should always construct");
    let errors = NullErrorSink;
    match talkbank_parser::parse_word(text, 0, &errors).into_option() {
        Some(word) => Some(word),
        None => {
            tracing::warn!(
                token = text,
                "Stanza token is not valid CHAT word syntax; keeping original word"
            );
            diagnostics.push(format!(
                "token {text:?} is not valid CHAT word syntax; keeping original word"
            ));
            None
        }
    }
}

/// Resolve the token text for a Stanza token, handling xbxxx restoration.
///
/// If Stanza returned "xbxxx" and the original word had a special_form,
/// restore the original word text.
pub(super) fn resolve_token_text(
    stanza_text: &str,
    orig_word_idx: usize,
    original_words: &[ExtractedWord],
) -> String {
    if stanza_text == "xbxxx"
        && let Some(word) = original_words.get(orig_word_idx)
        && (word.form_type.is_some() || word.lang.is_some())
    {
        return word.text.as_str().to_string();
    }
    stanza_text.to_string()
}
