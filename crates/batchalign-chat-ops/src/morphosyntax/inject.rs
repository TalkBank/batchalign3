//! Result injection, clearing, and alignment validation.

use talkbank_model::model::{LanguageCode, Line};

use super::{AlignmentWarning, BatchItemWithPosition, MwtDict, TokenizationMode};

/// Retokenization info collected during injection, for trace visualization.
#[derive(Debug, Clone)]
pub struct RetokenizationInfo {
    /// Utterance ordinal (0-based, among processed utterances).
    pub utterance_ordinal: usize,
    /// Original CHAT words.
    pub original_words: Vec<String>,
    /// Stanza tokens after retokenization.
    pub stanza_tokens: Vec<String>,
    /// Word→token index mapping: `mapping[word_idx]` = list of token indices.
    pub mapping: Vec<Vec<usize>>,
    /// Whether the fallback (length-proportional) mapping was used.
    pub used_fallback: bool,
}

// ---------------------------------------------------------------------------
// Result injection (from NLP callback)
// ---------------------------------------------------------------------------

/// Inject UD NLP results back into utterances.
///
/// Applies special form overrides (@c -> c|, @s -> L2|xxx) and
/// optionally retokenizes the main tier based on the [`TokenizationMode`].
///
/// # Errors
///
/// Returns `Err` if a `line_idx` no longer points to an utterance, or if
/// retokenization or morphosyntax injection fails for any utterance.
pub fn inject_results(
    chat_file: &mut talkbank_model::model::ChatFile,
    batch_items: Vec<BatchItemWithPosition>,
    responses: Vec<crate::nlp::UdResponse>,
    lang: &LanguageCode,
    tokenization_mode: TokenizationMode,
    mwt: &MwtDict,
) -> Result<Vec<RetokenizationInfo>, String> {
    use talkbank_model::model::FormType;
    use talkbank_model::model::dependent_tier::mor::PosCategory;

    let mut retokenization_traces: Vec<RetokenizationInfo> = Vec::new();

    for (ud_resp, (line_idx, utt_ordinal, item, words)) in
        responses.into_iter().zip(batch_items.into_iter())
    {
        if let Some(ud_sentence) = ud_resp.sentences.first() {
            let utt = match &mut chat_file.lines[line_idx] {
                Line::Utterance(u) => u,
                _ => {
                    return Err(format!(
                        "Line at index {line_idx} is no longer an utterance"
                    ));
                }
            };

            let ctx = crate::nlp::MappingContext { lang: lang.clone() };
            let (mut mors, mut gra_relations) = match crate::nlp::map_ud_sentence(ud_sentence, &ctx)
            {
                Ok(result) => result,
                Err(e) => {
                    tracing::warn!(utterance = utt_ordinal, error = %e, "skipping utterance: morphosyntax mapping failed");
                    continue;
                }
            };

            // Apply special form markers
            for (mor, (form_type, resolved_lang)) in mors.iter_mut().zip(item.special_forms.iter())
            {
                if resolved_lang.is_some() {
                    mor.main.pos = PosCategory::new("L2");
                    mor.main.lemma =
                        talkbank_model::model::dependent_tier::mor::MorStem::new("xxx");
                    continue;
                }

                if let Some(ft) = form_type {
                    let pos_tag = match ft {
                        FormType::A => "a",
                        FormType::B => "b",
                        FormType::C => "c",
                        FormType::D => "d",
                        FormType::F => "f",
                        FormType::FP => "fp",
                        FormType::G => "g",
                        FormType::I => "i",
                        FormType::K => "k",
                        FormType::L => "l",
                        FormType::LS => "ls",
                        FormType::N => "n",
                        FormType::O => "o",
                        FormType::P => "p",
                        FormType::Q => "q",
                        FormType::SAS => "sas",
                        FormType::SI => "si",
                        FormType::SL => "sl",
                        FormType::T => "t",
                        FormType::U => "u",
                        FormType::WP => "wp",
                        FormType::X => "x",
                        FormType::UserDefined(name) => {
                            tracing::warn!(form_type = %name, "user-defined form type not mapped to CHAT POS, skipping override");
                            continue;
                        }
                    };

                    mor.main.pos = PosCategory::new(pos_tag);
                }
            }

            if tokenization_mode == TokenizationMode::StanzaRetokenize {
                let mut tokens: Vec<String> = ud_sentence
                    .words
                    .iter()
                    .map(|w| {
                        if w.text.contains(char::is_whitespace) {
                            w.text.chars().filter(|c| !c.is_whitespace()).collect()
                        } else {
                            w.text.clone()
                        }
                    })
                    .collect();

                // Apply MWT lexicon overrides: when a Stanza token matches an
                // MWT entry, splice in the expansion tokens (and duplicate the
                // corresponding Mor/GRA items so counts stay aligned).
                if !mwt.is_empty() {
                    let mut expanded_tokens = Vec::with_capacity(tokens.len());
                    let mut expanded_mors = Vec::with_capacity(mors.len());
                    let mut expanded_gra = Vec::with_capacity(gra_relations.len());

                    for (tok_idx, tok) in tokens.iter().enumerate() {
                        let tok_lower = tok.to_lowercase();
                        if let Some(expansion) =
                            mwt.get(&tok_lower).or_else(|| mwt.get(tok.as_str()))
                        {
                            // Replace this token with the expansion tokens.
                            expanded_tokens.extend(expansion.iter().cloned());

                            // For the first expansion token, keep the original
                            // Mor/GRA. For subsequent tokens, duplicate so the
                            // alignment stays correct.
                            if tok_idx < mors.len() {
                                expanded_mors.push(mors[tok_idx].clone());
                                for _ in 1..expansion.len() {
                                    expanded_mors.push(mors[tok_idx].clone());
                                }
                            }
                            if tok_idx < gra_relations.len() {
                                expanded_gra.push(gra_relations[tok_idx].clone());
                                for _ in 1..expansion.len() {
                                    expanded_gra.push(gra_relations[tok_idx].clone());
                                }
                            }
                        } else {
                            expanded_tokens.push(tok.clone());
                            if tok_idx < mors.len() {
                                expanded_mors.push(mors[tok_idx].clone());
                            }
                            if tok_idx < gra_relations.len() {
                                expanded_gra.push(gra_relations[tok_idx].clone());
                            }
                        }
                    }

                    tokens = expanded_tokens;
                    mors = expanded_mors;
                    gra_relations = expanded_gra;
                }

                // Collect retokenization trace info before modifying the AST.
                {
                    use crate::retokenize::mapping::{
                        build_word_token_mapping, try_deterministic_word_token_mapping,
                    };
                    let mapping = build_word_token_mapping(&words, &tokens);
                    let used_fallback =
                        try_deterministic_word_token_mapping(&words, &tokens).is_none();
                    retokenization_traces.push(RetokenizationInfo {
                        utterance_ordinal: utt_ordinal,
                        original_words: words.iter().map(|w| w.text.as_str().to_string()).collect(),
                        stanza_tokens: tokens.clone(),
                        mapping: (0..words.len())
                            .map(|i| mapping.tokens_for_word(i).to_vec())
                            .collect(),
                        used_fallback,
                    });
                }

                crate::retokenize::retokenize_utterance(
                    utt,
                    &words,
                    &tokens,
                    mors,
                    Some(item.terminator.as_str().to_string()),
                    gra_relations,
                )
                .map_err(|e| format!("Failed to retokenize utterance {utt_ordinal}: {e}"))?;
            } else {
                crate::inject::inject_morphosyntax(
                    utt,
                    mors,
                    Some(item.terminator.as_str().to_string()),
                    gra_relations,
                )
                .map_err(|e| {
                    format!("Failed to inject morphosyntax for utterance {utt_ordinal}: {e}")
                })?;
            }
        } else {
            tracing::warn!(
                utterance = utt_ordinal,
                "NLP model returned no sentences, skipping morphosyntax"
            );
        }
    }

    Ok(retokenization_traces)
}

// ---------------------------------------------------------------------------
// Clear morphosyntax (strip existing %mor/%gra)
// ---------------------------------------------------------------------------

/// Remove all `%mor` and `%gra` dependent tiers from every utterance.
///
/// This is used before `collect_payloads()` so that all utterances are seen
/// as needing (re-)processing.
pub fn clear_morphosyntax(chat_file: &mut talkbank_model::model::ChatFile) {
    for line in chat_file.lines.iter_mut() {
        if let Line::Utterance(utt) = line {
            let keep: Vec<bool> = utt
                .dependent_tiers
                .iter()
                .map(|t| {
                    !matches!(
                        t,
                        talkbank_model::model::DependentTier::Mor(_)
                            | talkbank_model::model::DependentTier::Gra(_)
                    )
                })
                .collect();

            let mut keep_iter = keep.iter();
            utt.dependent_tiers.retain(|_| *keep_iter.next().unwrap());
        }
    }
}

/// Clear %mor/%gra tiers only from utterances at specific ordinals.
///
/// Unlike [`clear_morphosyntax`], this selectively clears only the utterances
/// whose ordinals are in `utterance_ordinals`. Used by incremental processing
/// to clear stale tiers only on changed utterances.
pub fn clear_morphosyntax_selective(
    chat_file: &mut talkbank_model::model::ChatFile,
    utterance_ordinals: &std::collections::HashSet<usize>,
) {
    let mut utt_idx = 0usize;
    for line in chat_file.lines.iter_mut() {
        if let Line::Utterance(utt) = line {
            if utterance_ordinals.contains(&utt_idx) {
                utt.dependent_tiers.retain(|t| {
                    !matches!(
                        t,
                        talkbank_model::model::DependentTier::Mor(_)
                            | talkbank_model::model::DependentTier::Gra(_)
                    )
                });
            }
            utt_idx += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Alignment validation
// ---------------------------------------------------------------------------

/// Validate that every utterance's %mor word count equals the main tier
/// alignable word count.
///
/// Returns a list of mismatches (empty = all good). This is a lightweight
/// check -- it does NOT run full semantic validation (E362, E701/E704, etc.).
pub fn validate_mor_alignment(
    chat_file: &talkbank_model::model::ChatFile,
) -> Vec<AlignmentWarning> {
    use talkbank_model::alignment::helpers::{TierDomain, count_tier_positions};
    use talkbank_model::model::DependentTier;

    let mut warnings = Vec::new();

    for (line_idx, line) in chat_file.lines.iter().enumerate() {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        let mor_tier = utt.dependent_tiers.iter().find_map(|t| match t {
            DependentTier::Mor(m) => Some(m),
            _ => None,
        });

        let Some(mor) = mor_tier else {
            continue; // No %mor tier -- nothing to validate
        };

        let main_count = count_tier_positions(&utt.main.content.content, TierDomain::Mor);
        let mor_count = mor.len();

        if main_count != mor_count {
            warnings.push(AlignmentWarning {
                line_idx,
                main_count,
                mor_count,
            });
        }
    }

    warnings
}
