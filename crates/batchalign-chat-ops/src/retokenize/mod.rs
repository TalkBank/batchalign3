//! Retokenize an utterance's main tier to match Stanza's UD tokenization.
//!
//! When `retokenize=true`, Stanza uses its own tokenizer instead of Batchalign's
//! custom `tokenize_postprocessor`. This means Stanza may split or merge words
//! differently from the original CHAT transcript. For example:
//!   - Original: `["don't"]`  -> Stanza: `["do", "n't"]` (1:N split)
//!   - Original: `["gon", "na"]` -> Stanza: `["gonna"]` (N:1 merge)
//!
//! This module replaces the main tier's Word nodes with the Stanza tokenization
//! while preserving all non-word AST content (Groups, Separators, Events, etc.).
//!
//! # Related CHAT Manual Sections
//!
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Format>
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Headers>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Main_Tier>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Dependent_Tiers>

pub mod mapping;
mod parse_helpers;
mod rebuild;
#[cfg(test)]
mod tests;

use crate::extract::ExtractedWord;
use crate::inject;
use talkbank_model::model::{GrammaticalRelation, Mor, ParseHealthTier, Utterance};

use mapping::build_word_token_mapping;
use rebuild::{RetokenizeContext, rebuild_content};

/// Retokenize an utterance to match Stanza's tokenization, then inject morphosyntax.
///
/// 1. Deterministic mapping maps original words -> Stanza tokens
/// 2. Walks the AST, replacing/splicing Word nodes to match the new tokenization
/// 3. Injects %mor/%gra tiers from the parsed morphosyntax
pub fn retokenize_utterance(
    utterance: &mut Utterance,
    original_words: &[ExtractedWord],
    stanza_tokens: &[String],
    mors: Vec<Mor>,
    terminator: Option<String>,
    gra_relations: Vec<GrammaticalRelation>,
) -> Result<(), String> {
    if original_words.is_empty() || stanza_tokens.is_empty() {
        return Ok(());
    }
    let expected_terminator = terminator.as_deref();

    // Step 1: Build mapping from original word index -> list of Stanza token indices
    let mapping = build_word_token_mapping(original_words, stanza_tokens);

    // Step 2: Walk AST and rebuild content with new tokenization
    let mut ctx = RetokenizeContext {
        mapping: &mapping,
        stanza_tokens,
        original_words,
        mors: &mors,
        expected_terminator,
        word_counter: 0,
        mor_cursor: 0,
        diagnostics: Vec::new(),
        emitted_tokens: std::collections::HashSet::new(),
    };

    let old_content = std::mem::take(&mut utterance.main.content.content.0);
    let mut new_content = Vec::with_capacity(old_content.len());

    rebuild_content(old_content, &mut ctx, &mut new_content);

    utterance.main.content.content.0 = new_content;

    if !ctx.diagnostics.is_empty() {
        // Retokenization had to recover from parser mismatches; taint main tier so
        // downstream alignment checks can account for it.
        utterance.mark_parse_taint(ParseHealthTier::Main);
        for warning in &ctx.diagnostics {
            tracing::warn!("retokenize: {warning}");
        }
    }

    // Step 3: Inject tier markers (reuse inject.rs logic)
    inject::inject_morphosyntax(utterance, mors, terminator, gra_relations)
}
