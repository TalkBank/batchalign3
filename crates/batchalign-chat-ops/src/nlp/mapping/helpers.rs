//! Helper functions for UD-to-CHAT mapping: relation mapping and MWT assembly.

use super::super::UdWord;
use super::super::mor_word::{is_clitic, map_ud_word_to_mor};
use super::context::MappingContext;
use super::errors::MappingError;
use std::collections::HashMap;
use talkbank_model::model::GrammaticalRelation;
use talkbank_model::model::dependent_tier::mor::Mor;

pub(super) fn map_relation(
    ud: &UdWord,
    chat_idx: usize,
    idx_map: &HashMap<usize, usize>,
) -> Result<GrammaticalRelation, MappingError> {
    let head_chat_idx = if ud.head == 0 {
        0 // ROOT head=0 in both UD and CHAT %gra convention
    } else {
        match idx_map.get(&ud.head) {
            Some(&idx) => idx,
            None => {
                return Err(MappingError::InvalidHeadReference {
                    details: format!(
                        "word {} (deprel={}) has head {} which is not in the chunk index map",
                        chat_idx, ud.deprel, ud.head
                    ),
                });
            }
        }
    };

    // TalkBank uses dashes for subtypes (ACL-RELCL), not UD colons (acl:relcl)
    let relation = ud.deprel.to_uppercase().replace(':', "-");
    // Validate: must match tree-sitter grammar rule [A-Z][A-Z0-9\-]*
    if relation.is_empty()
        || !relation.as_bytes()[0].is_ascii_uppercase()
        || !relation
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'-')
    {
        return Err(MappingError::InvalidDeprel {
            details: format!(
                "word {} has deprel {:?} which transforms to {:?} — \
                 not a valid CHAT %gra relation (must match [A-Z][A-Z0-9-]*)",
                chat_idx, ud.deprel, relation
            ),
        });
    }

    Ok(GrammaticalRelation {
        index: chat_idx,
        head: head_chat_idx,
        relation: relation.into(),
    })
}

/// Assembles multiple UD tokens into a single CHAT Mor node with clitics.
pub(super) fn assemble_mors(
    components: &[UdWord],
    ctx: &MappingContext,
) -> Result<Mor, MappingError> {
    if components.is_empty() {
        return Ok(Mor::new(
            talkbank_model::model::dependent_tier::mor::MorWord::new(
                talkbank_model::model::dependent_tier::mor::PosCategory::new("?"),
                talkbank_model::model::dependent_tier::mor::MorStem::new(""),
            ),
        ));
    }

    let mut main_idx = 0;
    for (idx, comp) in components.iter().enumerate() {
        if !is_clitic(&comp.text, ctx) {
            main_idx = idx;
            break;
        }
    }

    let main_mor = map_ud_word_to_mor(&components[main_idx], ctx)?;

    let mut mor = main_mor;

    // Pre-clitics no longer exist in the model; treat them as post-clitics
    // (components before main are prepended as post-clitics in order)
    for comp in &components[..main_idx] {
        let m = map_ud_word_to_mor(comp, ctx)?;
        mor = mor.with_post_clitic(m.main);
    }

    // Add components after main as post-clitics (~)
    for comp in &components[main_idx + 1..] {
        let m = map_ud_word_to_mor(comp, ctx)?;
        mor = mor.with_post_clitic(m.main);
    }

    Ok(mor)
}
