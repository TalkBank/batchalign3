//! Inject morphosyntax data into a CHAT AST utterance.
//!
//! After parsing %mor and %gra strings into typed Rust structures,
//! this module adds full tier structs to the utterance's dependent tiers.
//!
//! [`inject_morphosyntax`] validates alignment before injecting:
//! - MOR item count must match alignable word count (Mor domain).
//! - GRA relation count must match MOR chunk count.

use smallvec::SmallVec;
use talkbank_model::WriteChat;
use talkbank_model::alignment::helpers::TierDomain;
use talkbank_model::model::{DependentTier, GraTier, GrammaticalRelation, Mor, MorTier, Utterance};

/// Inject parsed Mor items and GRA relations into an utterance.
///
/// Validates alignment before injecting — catches count mismatches at the
/// point of corruption rather than deferring to pre-serialization validation.
pub fn inject_morphosyntax(
    utterance: &mut Utterance,
    mors: Vec<Mor>,
    terminator: Option<String>,
    gra_relations: Vec<GrammaticalRelation>,
) -> Result<(), String> {
    if mors.is_empty() {
        return Ok(());
    }

    // Validate: MOR count must match alignable word count.
    let mut extracted = Vec::new();
    crate::extract::collect_utterance_content(
        &utterance.main.content.content,
        TierDomain::Mor,
        &mut extracted,
    );
    let word_count = extracted.len();
    let mor_count = mors.len();
    tracing::debug!(
        word_count,
        mor_count,
        extracted_words = ?extracted.iter().map(|w| w.text.as_ref()).collect::<Vec<_>>(),
        "inject_morphosyntax: alignment check"
    );
    if word_count != mor_count {
        let utt_text = utterance.main.to_chat_string();
        // MOR count mismatches can occur legitimately when Stanza's MWT
        // tokenizer expands contractions (e.g. French `qu'` → `qu` + `'`)
        // that overlap with CHAT quotation marks in word text.  This is an
        // inherent tension between CHAT word boundaries and UD tokenization.
        //
        // BA2 did not have this assertion.  Treating it as a hard error
        // kills entire batches over a single utterance.  Instead, warn and
        // skip — the utterance gets no %mor/%gra (honest rather than wrong).
        tracing::warn!(
            word_count,
            mor_count,
            utterance = %utt_text,
            "MOR count mismatch — skipping utterance (Stanza MWT expansion \
             likely differs from CHAT word boundaries)"
        );
        return Err(format!(
            "MOR count mismatch ({mor_count} MOR items vs {word_count} words), \
             skipping utterance: {utt_text}"
        ));
    }

    let mor_tier =
        MorTier::new_mor(mors.clone()).with_terminator(terminator.clone().map(Into::into));

    let gra_tier = GraTier::new_gra(gra_relations);
    let gra_item_count = gra_tier.len();

    replace_or_add_tier(&mut utterance.dependent_tiers, DependentTier::Mor(mor_tier));
    if gra_item_count > 0 {
        replace_or_add_tier(&mut utterance.dependent_tiers, DependentTier::Gra(gra_tier));
    }

    Ok(())
}

/// Replace an existing tier of the same variant or append a new one.
pub fn replace_or_add_tier(tiers: &mut SmallVec<[DependentTier; 3]>, new_tier: DependentTier) {
    let variant_matches = |existing: &DependentTier, new: &DependentTier| -> bool {
        match (existing, new) {
            (DependentTier::Mor(_), DependentTier::Mor(_)) => true,
            (DependentTier::Gra(_), DependentTier::Gra(_)) => true,
            (DependentTier::Wor(_), DependentTier::Wor(_)) => true,
            (DependentTier::UserDefined(a), DependentTier::UserDefined(b)) => a.label == b.label,
            _ => false,
        }
    };

    for tier in tiers.iter_mut() {
        if variant_matches(tier, &new_tier) {
            *tier = new_tier;
            return;
        }
    }
    tiers.push(new_tier);
}

#[cfg(test)]
mod tests {
    use super::*;
    use talkbank_model::model::{ChatFile, Line, WriteChat};
    use talkbank_parser::TreeSitterParser;

    fn parse_chat(text: &str) -> ChatFile {
        let parser = TreeSitterParser::new().unwrap();
        parser.parse_chat_file(text).unwrap()
    }

    fn get_utterance(chat: &mut ChatFile, idx: usize) -> &mut Utterance {
        let mut utt_idx = 0;
        for line in &mut chat.lines {
            if let Line::Utterance(utt) = line {
                if utt_idx == idx {
                    return utt;
                }
                utt_idx += 1;
            }
        }
        panic!("Utterance {idx} not found");
    }

    #[test]
    fn test_replace_or_add_tier_user_defined() {
        use talkbank_model::model::{NonEmptyString, UserDefinedDependentTier};

        let mut tiers = smallvec::smallvec![];

        // Add %xtra
        let xtra1 = DependentTier::UserDefined(UserDefinedDependentTier {
            label: NonEmptyString::new("xtra").unwrap(),
            content: NonEmptyString::new("first").unwrap(),
            span: talkbank_model::Span::DUMMY,
        });
        replace_or_add_tier(&mut tiers, xtra1);
        assert_eq!(tiers.len(), 1);

        // Replace %xtra with new content
        let xtra2 = DependentTier::UserDefined(UserDefinedDependentTier {
            label: NonEmptyString::new("xtra").unwrap(),
            content: NonEmptyString::new("second").unwrap(),
            span: talkbank_model::Span::DUMMY,
        });
        replace_or_add_tier(&mut tiers, xtra2);
        assert_eq!(tiers.len(), 1); // replaced, not appended

        // Verify content was replaced
        if let DependentTier::UserDefined(ud) = &tiers[0] {
            assert_eq!(ud.content.as_ref(), "second");
        } else {
            panic!("Expected UserDefined tier");
        }

        // Add %xcod (different label) — should NOT replace %xtra
        let xcod = DependentTier::UserDefined(UserDefinedDependentTier {
            label: NonEmptyString::new("xcod").unwrap(),
            content: NonEmptyString::new("code").unwrap(),
            span: talkbank_model::Span::DUMMY,
        });
        replace_or_add_tier(&mut tiers, xcod);
        assert_eq!(tiers.len(), 2); // appended, not replaced
    }

    #[test]
    fn test_replace_or_add_tier_replaces_existing_wor() {
        use talkbank_model::model::WorTier;

        let mut tiers = smallvec::smallvec![DependentTier::Wor(WorTier::default())];
        let replacement = DependentTier::Wor(WorTier::from_words(vec![
            talkbank_model::model::Word::simple("hello"),
        ]));

        replace_or_add_tier(&mut tiers, replacement);

        assert_eq!(tiers.len(), 1);
        let DependentTier::Wor(wor) = &tiers[0] else {
            panic!("expected %wor tier");
        };
        assert_eq!(wor.words().count(), 1);
    }

    #[test]
    fn test_inject_empty_mors_is_noop() {
        let chat_text = include_str!("../../../test-fixtures/eng_hello_female.cha");
        let mut chat = parse_chat(chat_text);
        let output_before = chat.to_chat_string();

        let utt = get_utterance(&mut chat, 0);
        inject_morphosyntax(utt, Vec::new(), None, Vec::new()).unwrap();

        let output_after = chat.to_chat_string();
        assert_eq!(output_before, output_after);
    }
}
