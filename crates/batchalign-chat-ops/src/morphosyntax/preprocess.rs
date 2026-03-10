//! Text preprocessing for morphosyntax inference.
//!
//! Prepares utterance word lists into text suitable for Stanza input.

/// Join words with spaces and strip parentheses.
///
/// This produces the input text that Stanza expects for morphosyntax analysis.
/// Parentheses are removed because they interfere with Stanza's tokenizer.
pub fn prepare_text(words: &[String]) -> String {
    let joined = words.join(" ");
    joined.replace(['(', ')'], "").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_text_basic() {
        let words = vec!["I".into(), "eat".into(), "cookies".into()];
        assert_eq!(prepare_text(&words), "I eat cookies");
    }

    #[test]
    fn test_prepare_text_with_parens() {
        let words = vec!["(I)".into(), "eat".into(), "(cookies)".into()];
        assert_eq!(prepare_text(&words), "I eat cookies");
    }

    #[test]
    fn test_prepare_text_empty() {
        let words: Vec<String> = vec![];
        assert_eq!(prepare_text(&words), "");
    }

    #[test]
    fn test_prepare_text_whitespace_trim() {
        let words = vec![" hello ".into()];
        assert_eq!(prepare_text(&words), "hello");
    }
}
