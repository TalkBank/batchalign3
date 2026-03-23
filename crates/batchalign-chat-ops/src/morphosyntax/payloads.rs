//! Payload extraction and language helpers.

use talkbank_model::WriteChat;
use talkbank_model::alignment::helpers::TierDomain;
use talkbank_model::model::Line;

use super::{
    BatchItemWithPosition, MorphosyntaxBatchItem, MorphosyntaxPayloadJson, MultilingualPolicy,
    MwtDict, cache_key,
};
use crate::extract;

// ---------------------------------------------------------------------------
// Payload extraction
// ---------------------------------------------------------------------------

/// Extract per-utterance word payloads for cache key computation (pure Rust).
///
/// # Errors
///
/// Returns `Err` if the collected payloads cannot be serialized to JSON.
pub fn extract_payloads_json(
    chat_file: &talkbank_model::model::ChatFile,
    lang: &str,
    multilingual_policy: MultilingualPolicy,
    mwt: &MwtDict,
) -> Result<String, String> {
    let primary_lang = talkbank_model::model::LanguageCode::new(lang);

    let declared_languages: Vec<talkbank_model::model::LanguageCode> =
        if chat_file.languages.is_empty() {
            vec![primary_lang.clone()]
        } else {
            chat_file.languages.0.clone()
        };

    let (batch_items, _total) = collect_payloads(
        chat_file,
        &primary_lang,
        &declared_languages,
        multilingual_policy,
    );

    let payloads: Vec<MorphosyntaxPayloadJson> = batch_items
        .into_iter()
        .map(|(line_idx, _utt_ordinal, item, _words)| {
            let key = cache_key(&item.words, &item.lang, mwt, false);
            MorphosyntaxPayloadJson {
                line_idx,
                words: item.words,
                lang: item.lang.as_str().to_string(),
                key: key.to_string(),
            }
        })
        .collect();

    serde_json::to_string(&payloads)
        .map_err(|e| format!("Failed to serialize morphosyntax payloads: {e}"))
}

/// Walk utterances, build typed payloads.
///
/// Returns `(batch_items, total_utterance_count)`.
pub fn collect_payloads(
    chat_file: &talkbank_model::model::ChatFile,
    primary_lang: &talkbank_model::model::LanguageCode,
    declared_languages: &[talkbank_model::model::LanguageCode],
    multilingual_policy: MultilingualPolicy,
) -> (Vec<BatchItemWithPosition>, usize) {
    let total_utts = chat_file
        .lines
        .iter()
        .filter(|l| matches!(l, Line::Utterance(_)))
        .count();

    let mut batch_items: Vec<BatchItemWithPosition> = Vec::new();
    let mut utt_idx = 0usize;

    for (line_idx, line) in chat_file.lines.iter().enumerate() {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        let utterance_lang = utt.main.content.language_code.clone().unwrap_or_else(|| {
            // Prefer the first declared language from @Languages header
            // over the batch-level primary_lang, which may be a generic
            // default ("eng") that doesn't match this file.
            declared_languages
                .first()
                .cloned()
                .unwrap_or_else(|| primary_lang.clone())
        });

        let skip = multilingual_policy.should_skip_non_primary()
            && utt.main.content.language_code.is_some()
            && utt.main.content.language_code.as_ref() != Some(primary_lang);

        // Skip utterances that already have a %mor tier
        let has_mor = utt
            .dependent_tiers
            .iter()
            .any(|t| matches!(t, talkbank_model::model::DependentTier::Mor(_)));

        if !skip && !has_mor {
            let mut words = Vec::new();
            extract::collect_utterance_content(
                &utt.main.content.content,
                TierDomain::Mor,
                &mut words,
            );

            if !words.is_empty() {
                let terminator_str = utt
                    .main
                    .content
                    .terminator
                    .as_ref()
                    .map(|t| t.to_chat_string())
                    .unwrap_or_else(|| ".".to_string());

                let tier_language = utt
                    .main
                    .content
                    .language_code
                    .as_ref()
                    .or(Some(primary_lang));

                let special_forms: Vec<(
                    Option<talkbank_model::model::FormType>,
                    Option<talkbank_model::validation::LanguageResolution>,
                )> = words
                    .iter()
                    .map(|w| {
                        let resolved_lang = if let Some(ref lang_marker) = w.lang {
                            use talkbank_model::model::Word;
                            use talkbank_model::validation::resolve_word_language;

                            let mut temp_word =
                                Word::new_unchecked(w.text.as_str(), w.text.as_str());
                            temp_word.lang = Some(lang_marker.clone());

                            let (resolved, lang_errors) = resolve_word_language(
                                &temp_word,
                                tier_language,
                                declared_languages,
                            );
                            for err in &lang_errors {
                                tracing::warn!(error = %err, "word language resolution issue");
                            }
                            Some(resolved)
                        } else {
                            None
                        };

                        (w.form_type.clone(), resolved_lang)
                    })
                    .collect();

                let word_texts: Vec<String> =
                    words.iter().map(|w| w.text.as_str().to_string()).collect();

                batch_items.push((
                    line_idx,
                    utt_idx,
                    MorphosyntaxBatchItem {
                        words: word_texts,
                        terminator: terminator_str,
                        special_forms,
                        lang: utterance_lang,
                    },
                    words,
                ));
            }
        }

        utt_idx += 1;
    }

    (batch_items, total_utts)
}

// ---------------------------------------------------------------------------
// Declared languages helper
// ---------------------------------------------------------------------------

/// Extract declared languages from the `@Languages` header, with fallback
/// to `primary_lang` if none declared.
pub fn declared_languages(
    chat_file: &talkbank_model::model::ChatFile,
    primary_lang: &talkbank_model::model::LanguageCode,
) -> Vec<talkbank_model::model::LanguageCode> {
    if chat_file.languages.is_empty() {
        vec![primary_lang.clone()]
    } else {
        chat_file.languages.0.clone()
    }
}
