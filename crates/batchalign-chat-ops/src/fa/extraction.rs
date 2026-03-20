//! Word extraction for forced alignment (Wor domain).

use talkbank_model::alignment::helpers::{TierDomain, WordItem, counts_for_tier, walk_words};
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
    walk_words(content, None, &mut |leaf| match leaf {
        WordItem::Word(word) => {
            if counts_for_tier(word, TierDomain::Wor) {
                out.push(word.cleaned_text().to_string());
            }
        }
        WordItem::ReplacedWord(replaced) => {
            if !replaced.replacement.words.is_empty() {
                for word in &replaced.replacement.words {
                    if counts_for_tier(word, TierDomain::Wor) {
                        out.push(word.cleaned_text().to_string());
                    }
                }
            } else if counts_for_tier(&replaced.word, TierDomain::Wor) {
                out.push(replaced.word.cleaned_text().to_string());
            }
        }
        WordItem::Separator(_) => {}
    });
}
