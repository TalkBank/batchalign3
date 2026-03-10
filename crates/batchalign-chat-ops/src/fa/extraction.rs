//! Word extraction for forced alignment (Wor domain).

use talkbank_model::alignment::helpers::{
    AlignmentDomain, ContentLeaf, for_each_leaf, word_is_alignable,
};
use talkbank_model::model::UtteranceContent;

/// Collect alignable word texts from utterance content for forced alignment.
///
/// Uses the `Wor` alignment domain to decide which words are alignable.
/// Extracted texts are the cleaned (CHAT-marker-free) forms.
///
/// * `content` - The top-level content items of an utterance.
/// * `out` - Accumulator that word texts are pushed into.
pub fn collect_fa_words(content: &[UtteranceContent], out: &mut Vec<String>) {
    // domain=None: recurse into all groups unconditionally (FA needs all words)
    for_each_leaf(content, None, &mut |leaf| match leaf {
        ContentLeaf::Word(word, _annotations) => {
            if word_is_alignable(word, AlignmentDomain::Wor) {
                out.push(word.cleaned_text().to_string());
            }
        }
        ContentLeaf::ReplacedWord(replaced) => {
            if !replaced.replacement.words.is_empty() {
                for word in &replaced.replacement.words {
                    if word_is_alignable(word, AlignmentDomain::Wor) {
                        out.push(word.cleaned_text().to_string());
                    }
                }
            } else if word_is_alignable(&replaced.word, AlignmentDomain::Wor) {
                out.push(replaced.word.cleaned_text().to_string());
            }
        }
        ContentLeaf::Separator(_) => {}
    });
}
