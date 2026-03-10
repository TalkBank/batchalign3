//! NLP word extraction from CHAT AST.
//!
//! Walks the parsed CHAT content tree and collects words that are
//! "alignable" for a given domain (Mor, Wor, Pho, Sin).

use talkbank_model::alignment::helpers::{
    AlignmentDomain, ContentLeaf, annotations_have_alignment_ignore, for_each_leaf,
    is_tag_marker_separator, should_align_replaced_word_in_pho_sin, word_is_alignable,
};
use talkbank_model::model::{
    ChatFile, Line, ReplacedWord, Separator, UtteranceContent, Word, WriteChat,
};

use crate::indices::{UtteranceIdx, WordIdx};

/// A word extracted from the CHAT AST for NLP processing.
#[derive(Debug, Clone)]
pub struct ExtractedWord {
    /// Cleaned text suitable for NLP (no CHAT markers).
    pub text: crate::text_types::ChatCleanedText,
    /// Raw text as it appeared in the transcript.
    pub raw_text: crate::text_types::ChatRawText,
    /// Zero-based index among extracted alignable words in this utterance.
    pub utterance_word_index: WordIdx,
    /// Special form marker if the word has @c, @b, @s, etc.
    pub form_type: Option<talkbank_model::model::FormType>,
    /// Language marker (e.g., @s, @s:eng, @s:eng+fra).
    pub lang: Option<talkbank_model::model::WordLanguageMarker>,
}

/// Per-utterance extraction result.
#[derive(Debug, Clone)]
pub struct ExtractedUtterance {
    /// Speaker code (e.g., "CHI", "MOT").
    pub speaker: crate::text_types::SpeakerCode,
    /// Zero-based utterance index in the file.
    pub utterance_index: UtteranceIdx,
    /// Extracted words.
    pub words: Vec<ExtractedWord>,
}

/// Extract NLP-ready words from all utterances in a ChatFile.
///
/// Walks every utterance in the file and collects words that are
/// "alignable" for the given `domain`. Non-utterance lines (headers,
/// comments, etc.) are skipped.
///
/// * `chat_file` - The parsed CHAT file to extract words from.
/// * `domain` - The alignment domain governing which words are
///   considered alignable (`Mor`, `Wor`, `Pho`, or `Sin`).
pub fn extract_words(chat_file: &ChatFile, domain: AlignmentDomain) -> Vec<ExtractedUtterance> {
    let mut results = Vec::new();
    let mut utt_idx = 0;

    for line in &chat_file.lines {
        if let Line::Utterance(utterance) = line {
            let speaker = crate::text_types::SpeakerCode::new(utterance.main.speaker.to_string());
            let mut words = Vec::new();
            collect_utterance_content(&utterance.main.content.content, domain, &mut words);
            results.push(ExtractedUtterance {
                speaker,
                utterance_index: UtteranceIdx(utt_idx),
                words,
            });
            utt_idx += 1;
        }
    }

    results
}

/// Collect NLP-extractable words from a slice of utterance content items.
///
/// This is the inner workhorse called by [`extract_words`] and also used
/// directly by other modules (morphosyntax, utseg, translate, coref) to
/// extract words from a single utterance's content without iterating the
/// entire file.
///
/// * `content` - The top-level content items of an utterance.
/// * `domain` - The alignment domain that determines which words are
///   collected (e.g., `Mor` includes tag-marker separators; `Wor` does not).
/// * `out` - Accumulator that extracted words are pushed into.
pub fn collect_utterance_content(
    content: &[UtteranceContent],
    domain: AlignmentDomain,
    out: &mut Vec<ExtractedWord>,
) {
    for_each_leaf(content, Some(domain), &mut |leaf| match leaf {
        ContentLeaf::Word(word, annotations) => {
            collect_alignable_word(word, annotations, domain, out);
        }
        ContentLeaf::ReplacedWord(replaced) => {
            collect_replaced_word(replaced, domain, out);
        }
        ContentLeaf::Separator(sep) => {
            if domain == AlignmentDomain::Mor && is_tag_marker_separator(sep) {
                let sep_text = render_separator(sep);
                out.push(ExtractedWord {
                    text: crate::text_types::ChatCleanedText::new(sep_text.clone()),
                    raw_text: crate::text_types::ChatRawText::new(sep_text.clone()),
                    utterance_word_index: WordIdx(out.len()),
                    form_type: None,
                    lang: None,
                });
            }
        }
    });
}

fn collect_alignable_word(
    word: &Word,
    annotations: &[talkbank_model::model::ScopedAnnotation],
    domain: AlignmentDomain,
    out: &mut Vec<ExtractedWord>,
) {
    if domain == AlignmentDomain::Mor && annotations_have_alignment_ignore(annotations) {
        return;
    }

    if !word_is_alignable(word, domain) {
        return;
    }

    out.push(ExtractedWord {
        text: crate::text_types::ChatCleanedText::new(word.cleaned_text().to_string()),
        raw_text: crate::text_types::ChatRawText::new(word.raw_text().to_string()),
        utterance_word_index: WordIdx(out.len()),
        form_type: word.form_type.clone(),
        lang: word.lang.clone(),
    });
}

fn collect_replaced_word(
    entry: &ReplacedWord,
    domain: AlignmentDomain,
    out: &mut Vec<ExtractedWord>,
) {
    if domain == AlignmentDomain::Mor
        && annotations_have_alignment_ignore(&entry.scoped_annotations)
    {
        return;
    }

    match domain {
        AlignmentDomain::Mor => {
            if !entry.replacement.words.is_empty() {
                for word in &entry.replacement.words {
                    if word_is_alignable(word, AlignmentDomain::Mor) {
                        out.push(ExtractedWord {
                            text: crate::text_types::ChatCleanedText::new(
                                word.cleaned_text().to_string(),
                            ),
                            raw_text: crate::text_types::ChatRawText::new(
                                word.raw_text().to_string(),
                            ),
                            utterance_word_index: WordIdx(out.len()),
                            form_type: word.form_type.clone(),
                            lang: word.lang.clone(),
                        });
                    }
                }
            } else if word_is_alignable(&entry.word, AlignmentDomain::Mor) {
                out.push(ExtractedWord {
                    text: crate::text_types::ChatCleanedText::new(
                        entry.word.cleaned_text().to_string(),
                    ),
                    raw_text: crate::text_types::ChatRawText::new(
                        entry.word.raw_text().to_string(),
                    ),
                    utterance_word_index: WordIdx(out.len()),
                    form_type: entry.word.form_type.clone(),
                    lang: entry.word.lang.clone(),
                });
            }
        }
        AlignmentDomain::Pho | AlignmentDomain::Sin | AlignmentDomain::Wor => {
            if should_align_replaced_word_in_pho_sin(
                &entry.word,
                !entry.replacement.words.is_empty(),
            ) {
                out.push(ExtractedWord {
                    text: crate::text_types::ChatCleanedText::new(
                        entry.word.cleaned_text().to_string(),
                    ),
                    raw_text: crate::text_types::ChatRawText::new(
                        entry.word.raw_text().to_string(),
                    ),
                    utterance_word_index: WordIdx(out.len()),
                    form_type: entry.word.form_type.clone(),
                    lang: entry.word.lang.clone(),
                });
            }
        }
    }
}

fn render_separator(sep: &Separator) -> String {
    sep.to_chat_string()
}
