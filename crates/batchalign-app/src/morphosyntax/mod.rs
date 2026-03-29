//! Server-side morphosyntax orchestrator.
//!
//! Owns the full CHAT lifecycle for morphotag jobs:
//! parse → clear → collect → cache check → infer → inject → cache put → serialize.
//!
//! Python workers receive only `(words, lang) → UdResponse` via the infer protocol —
//! pure Stanza inference with zero CHAT awareness.
//!
//! # Call path
//!
//! `batchalign-cli`/API submission
//! → `runner::dispatch_batched_infer(command="morphotag")`
//! → [`process_morphosyntax_batch`] (or [`process_morphosyntax`] for single-file)
//! → `batchalign_chat_ops::morphosyntax::{collect_payloads, inject_results}`
//! → worker `batch_infer(task="morphosyntax")`
//! → validation + serialization.
//!
//! # Invariants for contributors
//!
//! - `line_idx`/utterance positions from payload collection must still address
//!   utterances at injection time.
//! - `TokenizationMode::StanzaRetokenize` changes main-tier token boundaries
//!   during injection; post-injection alignment checks must still pass.
//! - Cache key computation (`cache_key`) and task name (`morphosyntax_v4`) are
//!   compatibility-critical; changing either invalidates existing entries.
//! - Workers must stay CHAT-agnostic: only structured NLP payloads/responses
//!   cross the Rust/Python boundary.

mod batch;
mod worker;

use std::collections::HashMap;

use crate::error::ServerError;
use crate::params::MorphosyntaxParams;
use crate::pipeline::PipelineServices;
use crate::pipeline::morphosyntax::run_morphosyntax_pipeline;
use crate::text_batch::{TextBatchFileInput, TextBatchFileResults};
use batchalign_chat_ops::morphosyntax::{
    BatchItemWithPosition, TokenizationMode, cache_key, clear_morphosyntax_selective,
    collect_payloads, declared_languages, extract_strings, inject_results, validate_mor_alignment,
};
use batchalign_chat_ops::parse::{is_dummy, parse_lenient};
use batchalign_chat_ops::serialize::to_chat_string;
use batchalign_chat_ops::validate::{ValidityLevel, validate_output, validate_to_level};
use batchalign_chat_ops::{CacheKey, CacheTaskName, LanguageCode};
use tracing::{info, warn};

pub(crate) use batch::{inject_cache_hits, partition_by_cache, run_morphosyntax_batch_impl};
pub(crate) use worker::{cache_put_entries, infer_batch};

/// Cache task name — matches Python's `morphosyntax_v4`.
const CACHE_TASK: CacheTaskName = CacheTaskName::Morphosyntax;

// ---------------------------------------------------------------------------
// Per-file morphosyntax processing
// ---------------------------------------------------------------------------

/// Process a single CHAT file through the morphosyntax pipeline.
///
/// Returns the serialized CHAT text with %mor/%gra tiers injected.
///
/// Algorithm outline:
/// 1. Parse and pre-validate to `MainTierValid`.
/// 2. Clear existing `%mor/%gra`.
/// 3. Collect per-utterance payloads with language/special-form metadata.
/// 4. Partition cache hits/misses.
/// 5. Inject cache hits, infer misses, inject results.
/// 6. Validate light alignment checks, persist cache entries.
/// 7. Run full post-validation and serialize.
pub(crate) async fn process_morphosyntax(
    chat_text: &str,
    services: PipelineServices<'_>,
    params: &MorphosyntaxParams<'_>,
) -> Result<String, ServerError> {
    run_morphosyntax_impl(chat_text, services, params).await
}

pub(crate) async fn run_morphosyntax_impl(
    chat_text: &str,
    services: PipelineServices<'_>,
    params: &MorphosyntaxParams<'_>,
) -> Result<String, ServerError> {
    run_morphosyntax_pipeline(
        chat_text,
        params.lang,
        services,
        params.tokenization_mode,
        params.cache_policy,
        params.multilingual_policy,
        params.mwt,
    )
    .await
}

/// Process multiple CHAT files through the cross-file batch morphosyntax workflow.
pub(crate) async fn process_morphosyntax_batch(
    files: &[TextBatchFileInput],
    services: PipelineServices<'_>,
    params: &MorphosyntaxParams<'_>,
) -> TextBatchFileResults {
    run_morphosyntax_batch_impl(files, services, params, None).await
}

// ---------------------------------------------------------------------------
// Incremental morphosyntax processing
// ---------------------------------------------------------------------------

/// Process a CHAT file incrementally by diffing against a "before" version.
///
/// Compares `before_text` (previous file with existing %mor/%gra) against
/// `after_text` (user-edited version) and only reprocesses utterances whose
/// words changed. Unchanged utterances preserve their existing %mor/%gra
/// tiers from the "before" version.
///
/// Returns the serialized CHAT text with %mor/%gra tiers on all utterances.
///
/// # When to use
///
/// Use this when reprocessing a file the user has edited (e.g., fixing
/// words, splitting/merging utterances). The "before" is the file as it
/// was before editing (with %mor/%gra from a previous run), and "after"
/// is the edited version needing updated %mor/%gra.
///
/// Falls back to full processing if no "before" is available (first run).
pub(crate) async fn process_morphosyntax_incremental(
    before_text: &str,
    after_text: &str,
    services: PipelineServices<'_>,
    params: &MorphosyntaxParams<'_>,
) -> Result<String, ServerError> {
    use batchalign_chat_ops::diff::preserve::TierKind;
    use batchalign_chat_ops::diff::{DiffSummary, UtteranceDelta, copy_dependent_tiers, diff_chat};

    let primary_lang = LanguageCode::new(params.lang.as_ref());
    let parser = batchalign_chat_ops::parse::TreeSitterParser::new()
        .expect("tree-sitter CHAT grammar must load");
    let (before_file, _) = parse_lenient(&parser, before_text);
    let (mut after_file, parse_errors) = parse_lenient(&parser, after_text);

    if !parse_errors.is_empty() {
        warn!(
            num_errors = parse_errors.len(),
            "Parse errors in 'after' file (continuing with recovery)"
        );
    }

    if is_dummy(&after_file) {
        return Ok(to_chat_string(&after_file));
    }

    // Pre-validation
    if let Err(errors) = validate_to_level(
        &after_file,
        parse_errors.len(),
        ValidityLevel::MainTierValid,
    ) {
        let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        return Err(ServerError::Validation(format!(
            "morphotag pre-validation failed: {}",
            msgs.join("; ")
        )));
    }

    // Diff before vs after
    let deltas = diff_chat(&before_file, &after_file);
    let summary = DiffSummary::from_deltas(&deltas);

    info!(
        unchanged = summary.unchanged,
        words_changed = summary.words_changed,
        inserted = summary.inserted,
        deleted = summary.deleted,
        timing_only = summary.timing_only,
        speaker_changed = summary.speaker_changed,
        "Incremental morphosyntax diff"
    );

    // If everything changed, fall back to full processing
    if summary.unchanged == 0 && summary.speaker_changed == 0 && summary.timing_only == 0 {
        return process_morphosyntax(after_text, services, params).await;
    }

    // Step 1: Copy %mor/%gra from "before" for unchanged/speaker-only/timing-only utterances
    let tier_kinds = &[TierKind::Mor, TierKind::Gra];
    let mut preserved_count = 0usize;
    for delta in &deltas {
        match delta {
            UtteranceDelta::Unchanged {
                before_idx,
                after_idx,
            }
            | UtteranceDelta::SpeakerChanged {
                before_idx,
                after_idx,
            }
            | UtteranceDelta::TimingOnly {
                before_idx,
                after_idx,
            } => {
                let n = copy_dependent_tiers(
                    &before_file,
                    *before_idx,
                    &mut after_file,
                    *after_idx,
                    tier_kinds,
                );
                if n > 0 {
                    preserved_count += 1;
                }
            }
            _ => {}
        }
    }
    info!(preserved_count, "Preserved %mor/%gra from before file");

    // Step 2: Clear %mor/%gra on utterances that need reprocessing
    // (WordsChanged and Inserted utterances)
    let needs_processing: Vec<usize> = deltas
        .iter()
        .filter(|d| d.needs_nlp_reprocessing())
        .filter_map(|d| d.after_idx())
        .map(|idx| idx.raw())
        .collect();

    if needs_processing.is_empty() {
        // Nothing to reprocess — all utterances preserved from "before"
        if let Err(errors) = validate_output(&after_file, "morphotag") {
            let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            warn!(errors = ?msgs, "morphotag post-validation warnings (non-fatal)");
        }
        return Ok(to_chat_string(&after_file));
    }

    // Build a set of utterance ordinals that need processing
    let needs_set: std::collections::HashSet<usize> = needs_processing.iter().copied().collect();

    // Clear %mor/%gra only on utterances that need reprocessing
    clear_morphosyntax_selective(&mut after_file, &needs_set);

    // Step 3: Collect payloads only for utterances that need reprocessing
    let langs = declared_languages(&after_file, &primary_lang);
    let (all_payloads, _) = collect_payloads(
        &after_file,
        &primary_lang,
        &langs,
        params.multilingual_policy,
    );

    // Filter to only the utterances that need reprocessing
    let filtered_payloads: Vec<BatchItemWithPosition> = all_payloads
        .into_iter()
        .filter(|(_, utt_ordinal, _, _)| needs_set.contains(utt_ordinal))
        .collect();

    if filtered_payloads.is_empty() {
        if let Err(errors) = validate_output(&after_file, "morphotag") {
            let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            warn!(errors = ?msgs, "morphotag post-validation warnings (non-fatal)");
        }
        return Ok(to_chat_string(&after_file));
    }

    info!(
        total_utterances = summary.total(),
        reprocessing = filtered_payloads.len(),
        "Incremental morphosyntax: sending only changed utterances to worker"
    );

    // Warn when Cantonese input appears to be per-character without --retokenize.
    let retokenize = params.tokenization_mode == TokenizationMode::StanzaRetokenize;
    if !retokenize && params.lang.as_ref() == "yue" {
        let per_char_count = filtered_payloads
            .iter()
            .flat_map(|(_, _, item, _)| item.words.iter())
            .filter(|w| w.chars().count() == 1 && w.chars().all(|c| c > '\u{2E80}'))
            .count();
        let total_words: usize = filtered_payloads
            .iter()
            .map(|(_, _, item, _)| item.words.len())
            .sum();
        if total_words > 0 && per_char_count * 100 / total_words > 80 {
            warn!(
                "Cantonese input appears to be per-character tokens ({per_char_count}/{total_words} single-CJK words). \
                 Consider --retokenize for word-level analysis."
            );
        }
    }

    // Step 4: Cache check + infer for the filtered payloads
    let (hits, miss_keys, misses) = if params.cache_policy.should_skip() {
        let keys: Vec<CacheKey> = filtered_payloads
            .iter()
            .map(|(_, _, item, _)| cache_key(&item.words, &item.lang, params.mwt, retokenize))
            .collect();
        (HashMap::new(), keys, filtered_payloads)
    } else {
        partition_by_cache(
            &filtered_payloads,
            services.cache,
            services.engine_version,
            params.mwt,
            retokenize,
        )
        .await
    };

    if !hits.is_empty()
        && let Err(e) = inject_cache_hits(&mut after_file, &hits)
    {
        warn!(error = %e, "Cache injection failed (non-fatal)");
    }

    if !misses.is_empty() {
        let miss_line_indices: Vec<usize> = misses.iter().map(|(idx, ..)| *idx).collect();

        match infer_batch(services.pool, &misses, params.lang, params.mwt, retokenize, None).await {
            Ok(responses) => {
                match inject_results(
                    &parser,
                    &mut after_file,
                    misses,
                    responses,
                    &primary_lang,
                    params.tokenization_mode,
                    params.mwt,
                ) {
                    Ok(_) => {}
                    Err(e) => {
                        return Err(ServerError::Validation(format!(
                            "Result injection failed: {e}"
                        )));
                    }
                }

                let alignment_warnings = validate_mor_alignment(&after_file);
                for w in &alignment_warnings {
                    warn!(warning = %w, "Morphosyntax alignment mismatch");
                }

                match extract_strings(&after_file, &miss_line_indices) {
                    Ok(entries) => {
                        cache_put_entries(
                            services.cache,
                            &miss_keys,
                            &entries,
                            services.engine_version,
                        )
                        .await;
                    }
                    Err(e) => {
                        warn!(error = %e, "Cache extraction failed (non-fatal)");
                    }
                }
            }
            Err(e) => {
                return Err(e);
            }
        }
    }

    if let Err(errors) = validate_output(&after_file, "morphotag") {
        let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        warn!(errors = ?msgs, "morphotag post-validation warnings (non-fatal)");
    }

    Ok(to_chat_string(&after_file))
}

#[cfg(test)]
mod tests {
    use super::*;
    use batchalign_chat_ops::morphosyntax::MultilingualPolicy;
    use batchalign_chat_ops::parse::TreeSitterParser;

    #[test]
    fn test_declared_languages_via_chat_ops() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../../test-fixtures/eng_hello_male.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        let primary = LanguageCode::new("eng");
        let langs = declared_languages(&chat_file, &primary);
        assert!(!langs.is_empty());
    }

    #[test]
    fn test_collect_payloads_skip_non_primary_skips_non_primary() {
        let parser = TreeSitterParser::new().unwrap();
        // File with @Languages: eng, spa and a [- spa] code-switched utterance
        let chat = include_str!("../../../../test-fixtures/eng_spa_bilingual_code_switch.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        let primary = LanguageCode::new("eng");
        let langs = declared_languages(&chat_file, &primary);

        // With SkipNonPrimary, the Spanish utterance should be skipped
        let (items_skip, _) = collect_payloads(
            &chat_file,
            &primary,
            &langs,
            MultilingualPolicy::SkipNonPrimary,
        );
        // With ProcessAll, all utterances should be included
        let (items_all, _) =
            collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll);

        // SkipNonPrimary should produce fewer items than ProcessAll
        assert!(
            items_skip.len() < items_all.len(),
            "SkipNonPrimary should skip non-primary-language utterances: \
             got {} with SkipNonPrimary vs {} with ProcessAll",
            items_skip.len(),
            items_all.len()
        );
    }

    #[test]
    fn test_collect_payloads_process_all_includes_all() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../../test-fixtures/eng_spa_bilingual_code_switch.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        let primary = LanguageCode::new("eng");
        let langs = declared_languages(&chat_file, &primary);

        let (items, _) =
            collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll);
        // Both utterances should be included
        assert_eq!(items.len(), 2, "ProcessAll should include all utterances");
    }

    #[test]
    fn test_dummy_file_skipped_by_is_dummy() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../../test-fixtures/eng_hello_world_dummy.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        assert!(is_dummy(&chat_file), "@Options: dummy should be detected");

        // Collect payloads should return items for non-dummy file
        let primary = LanguageCode::new("eng");
        let langs = declared_languages(&chat_file, &primary);
        let (items, _) =
            collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll);
        // The file has an utterance, but is_dummy tells the orchestrator to skip it
        assert!(
            !items.is_empty(),
            "collect_payloads still collects from dummy files — the orchestrator is what gates on is_dummy"
        );
    }

    #[test]
    fn test_non_dummy_file_not_detected() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../../test-fixtures/eng_hello_world.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        assert!(!is_dummy(&chat_file));
    }
}
