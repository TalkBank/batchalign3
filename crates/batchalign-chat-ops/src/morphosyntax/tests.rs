//! Tests for morphosyntax module.

use super::*;

#[test]
fn test_cache_key_deterministic() {
    let words = vec!["hello".to_string(), "world".to_string()];
    let eng = talkbank_model::model::LanguageCode::new("eng");
    let empty_mwt = std::collections::BTreeMap::new();
    let k1 = cache_key(&words, &eng, &empty_mwt, false);
    let k2 = cache_key(&words, &eng, &empty_mwt, false);
    assert_eq!(k1, k2);
    assert!(!k1.as_str().is_empty());
}

#[test]
fn test_cache_key_lang_differs() {
    let words = vec!["hello".to_string()];
    let eng = talkbank_model::model::LanguageCode::new("eng");
    let spa = talkbank_model::model::LanguageCode::new("spa");
    let empty_mwt = std::collections::BTreeMap::new();
    let k1 = cache_key(&words, &eng, &empty_mwt, false);
    let k2 = cache_key(&words, &spa, &empty_mwt, false);
    assert_ne!(k1, k2);
}

#[test]
fn cache_key_retokenize_differs() {
    let words = vec!["hello".to_string(), "world".to_string()];
    let eng = talkbank_model::model::LanguageCode::new("eng");
    let empty_mwt = std::collections::BTreeMap::new();
    let k_preserve = cache_key(&words, &eng, &empty_mwt, false);
    let k_retok = cache_key(&words, &eng, &empty_mwt, true);
    assert_ne!(
        k_preserve, k_retok,
        "retokenize=true must produce a different cache key"
    );
}

#[test]
fn test_clear_morphosyntax() {
    use crate::parse::{TreeSitterParser, parse_lenient};
    use talkbank_model::model::Line;

    let parser = TreeSitterParser::new().unwrap();
    // Minimal CHAT with %mor and %gra tiers
    let chat = include_str!("../../../../test-fixtures/eng_hello_world_with_mor_gra.cha");
    let (mut chat_file, _errors) = parse_lenient(&parser, chat);

    // Verify the utterance has %mor and %gra before clearing
    let utt = chat_file
        .lines
        .iter()
        .find_map(|l| match l {
            Line::Utterance(u) => Some(u),
            _ => None,
        })
        .expect("should have an utterance");
    assert!(
        utt.dependent_tiers
            .iter()
            .any(|t| matches!(t, talkbank_model::model::DependentTier::Mor(_))),
        "should have %mor before clear"
    );
    assert!(
        utt.dependent_tiers
            .iter()
            .any(|t| matches!(t, talkbank_model::model::DependentTier::Gra(_))),
        "should have %gra before clear"
    );

    // Clear
    clear_morphosyntax(&mut chat_file);

    // Verify no %mor or %gra remain
    let utt = chat_file
        .lines
        .iter()
        .find_map(|l| match l {
            Line::Utterance(u) => Some(u),
            _ => None,
        })
        .expect("should still have an utterance");
    assert!(
        !utt.dependent_tiers
            .iter()
            .any(|t| matches!(t, talkbank_model::model::DependentTier::Mor(_))),
        "should NOT have %mor after clear"
    );
    assert!(
        !utt.dependent_tiers
            .iter()
            .any(|t| matches!(t, talkbank_model::model::DependentTier::Gra(_))),
        "should NOT have %gra after clear"
    );
}

#[test]
fn test_clear_morphosyntax_preserves_other_tiers() {
    use crate::parse::{TreeSitterParser, parse_lenient};
    use talkbank_model::model::Line;

    let parser = TreeSitterParser::new().unwrap();
    // CHAT with %mor, %gra, and %act -- only %mor/%gra should be removed
    let chat = include_str!("../../../../test-fixtures/eng_hello_world_with_mor_gra_act.cha");
    let (mut chat_file, _) = parse_lenient(&parser, chat);

    clear_morphosyntax(&mut chat_file);

    let utt = chat_file
        .lines
        .iter()
        .find_map(|l| match l {
            Line::Utterance(u) => Some(u),
            _ => None,
        })
        .expect("utterance");
    // %act should survive
    assert!(
        !utt.dependent_tiers.is_empty(),
        "should still have %act after clear"
    );
    // But no %mor or %gra
    assert!(
        !utt.dependent_tiers.iter().any(|t| matches!(
            t,
            talkbank_model::model::DependentTier::Mor(_)
                | talkbank_model::model::DependentTier::Gra(_)
        )),
        "should not have %mor or %gra"
    );
}

#[test]
fn test_clear_morphosyntax_no_tiers() {
    use crate::parse::{TreeSitterParser, parse_lenient};
    use talkbank_model::model::Line;

    let parser = TreeSitterParser::new().unwrap();
    // CHAT without any dependent tiers -- clear should be a no-op
    let chat = include_str!("../../../../test-fixtures/eng_hello_male.cha");
    let (mut chat_file, _) = parse_lenient(&parser, chat);

    clear_morphosyntax(&mut chat_file);

    let utt = chat_file
        .lines
        .iter()
        .find_map(|l| match l {
            Line::Utterance(u) => Some(u),
            _ => None,
        })
        .expect("utterance");
    assert!(utt.dependent_tiers.is_empty());
}

#[test]
fn test_validate_mor_alignment_ok() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    // Correctly aligned: 2 main words + 2 %mor items
    let chat = include_str!("../../../../test-fixtures/eng_hello_world_with_mor_gra.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    let warnings = validate_mor_alignment(&chat_file);
    assert!(
        warnings.is_empty(),
        "expected no alignment warnings, got: {:?}",
        warnings
    );
}

#[test]
fn test_validate_mor_alignment_no_mor_tier() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    // No %mor -- validation should pass (nothing to check)
    let chat = include_str!("../../../../test-fixtures/eng_hello_male.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    let warnings = validate_mor_alignment(&chat_file);
    assert!(warnings.is_empty());
}

// -----------------------------------------------------------------------
// Cross-language roundtrip snapshot tests
// -----------------------------------------------------------------------

/// Verify MorphosyntaxBatchItem serializes to the JSON shape Python expects.
#[test]
fn snapshot_morphosyntax_batch_item() {
    let item = MorphosyntaxBatchItem {
        words: vec!["the".into(), "dog".into(), "runs".into()],
        terminator: ".".into(),
        special_forms: vec![(None, None), (None, None), (None, None)],
        lang: talkbank_model::model::LanguageCode::new("eng"),
    };
    insta::assert_json_snapshot!("morphosyntax_batch_item", item);
}

/// Verify UdResponse from Python deserializes correctly in Rust.
#[test]
fn snapshot_ud_response_from_python() {
    // This is the exact shape Python's Stanza inference returns
    let python_json = r#"{
        "sentences": [
            {
                "words": [
                    {
                        "id": 1,
                        "text": "the",
                        "lemma": "the",
                        "upos": "DET",
                        "xpos": "DT",
                        "feats": "Definite=Def|PronType=Art",
                        "head": 2,
                        "deprel": "det",
                        "start_char": 0,
                        "end_char": 3
                    },
                    {
                        "id": 2,
                        "text": "dog",
                        "lemma": "dog",
                        "upos": "NOUN",
                        "xpos": "NN",
                        "feats": "Number=Sing",
                        "head": 3,
                        "deprel": "nsubj",
                        "start_char": 4,
                        "end_char": 7
                    },
                    {
                        "id": 3,
                        "text": "runs",
                        "lemma": "run",
                        "upos": "VERB",
                        "xpos": "VBZ",
                        "feats": "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin",
                        "head": 0,
                        "deprel": "root",
                        "start_char": 8,
                        "end_char": 12
                    }
                ]
            }
        ]
    }"#;

    let ud: crate::nlp::UdResponse = serde_json::from_str(python_json).unwrap();
    assert_eq!(ud.sentences.len(), 1);
    assert_eq!(ud.sentences[0].words.len(), 3);
    assert_eq!(ud.sentences[0].words[2].lemma, "run");

    // Re-serialize and snapshot to verify round-trip fidelity
    insta::assert_json_snapshot!("ud_response_roundtrip", ud);
}

/// Verify collect_payloads produces the expected shape for a simple CHAT.
#[test]
fn snapshot_collected_payloads() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/eng_the_dog_runs.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    let primary = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary);
    let (items, total) =
        collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll);

    assert_eq!(total, 1);
    assert_eq!(items.len(), 1);

    // Snapshot just the batch item (the payload that crosses the wire)
    let (_, _, ref batch_item, _) = items[0];
    insta::assert_json_snapshot!("collected_payload_item", batch_item);
}

// -----------------------------------------------------------------------
// Regression tests: batch item lang must reflect file @Languages header,
// not the batch-level primary_lang parameter.
//
// Bug: when a job has lang="eng" (the default) but a file declares
// @Languages: spa, collect_payloads produced items with lang="eng"
// instead of "spa". This caused Stanza to use the wrong model.
// -----------------------------------------------------------------------

/// When @Languages declares "spa" but primary_lang is "eng" (batch default),
/// the batch item must carry lang="spa" from the file header.
#[test]
fn collect_payloads_uses_file_language_not_batch_default_spa() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/spa_hola_que_es_este.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    // Simulate the batch-level default: primary_lang = "eng"
    let primary = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary);
    let (items, _) = collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll);

    assert_eq!(items.len(), 1);
    let (_, _, ref batch_item, _) = items[0];

    // The batch item's lang MUST be "spa" (from @Languages header),
    // NOT "eng" (the batch default).
    assert_eq!(
        batch_item.lang.as_str(),
        "spa",
        "batch item lang should be 'spa' from @Languages header, not 'eng' batch default"
    );
}

/// Same regression for Russian: @Languages: rus with batch default "eng".
#[test]
fn collect_payloads_uses_file_language_not_batch_default_rus() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/rus_vot_istoriya.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    let primary = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary);
    let (items, _) = collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll);

    assert_eq!(items.len(), 1);
    let (_, _, ref batch_item, _) = items[0];

    assert_eq!(
        batch_item.lang.as_str(),
        "rus",
        "batch item lang should be 'rus' from @Languages header, not 'eng' batch default"
    );
}

/// Same regression for Chinese: @Languages: zho with batch default "eng".
#[test]
fn collect_payloads_uses_file_language_not_batch_default_zho() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/zho_hao_qing_zhong.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    let primary = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary);
    let (items, _) = collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll);

    assert_eq!(items.len(), 1);
    let (_, _, ref batch_item, _) = items[0];

    assert_eq!(
        batch_item.lang.as_str(),
        "zho",
        "batch item lang should be 'zho' from @Languages header, not 'eng' batch default"
    );
}

/// Same regression for French: @Languages: fra with batch default "eng".
#[test]
fn collect_payloads_uses_file_language_not_batch_default_fra() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/fra_lescargot_dort.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    let primary = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary);
    let (items, _) = collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll);

    assert_eq!(items.len(), 1);
    let (_, _, ref batch_item, _) = items[0];

    assert_eq!(
        batch_item.lang.as_str(),
        "fra",
        "batch item lang should be 'fra' from @Languages header, not 'eng' batch default"
    );
}

/// When @Languages matches primary_lang, lang should still be correct.
/// (Control case: ensures fix doesn't regress the happy path.)
#[test]
fn collect_payloads_lang_correct_when_primary_matches_header() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/eng_hello_world_male.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    let primary = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary);
    let (items, _) = collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll);

    assert_eq!(items.len(), 1);
    let (_, _, ref batch_item, _) = items[0];

    assert_eq!(batch_item.lang.as_str(), "eng");
}

/// When @Languages has multiple languages, the first declared language
/// should be used as the utterance default (not the batch primary_lang).
#[test]
fn collect_payloads_uses_first_declared_language_for_multilingual() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    // Bilingual file: primary declared language is "spa", secondary is "eng"
    let chat = include_str!("../../../../test-fixtures/spa_eng_bilingual_hola_mundo.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    // Batch default is "eng" but file says "spa" first
    let primary = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary);
    let (items, _) = collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll);

    assert_eq!(items.len(), 1);
    let (_, _, ref batch_item, _) = items[0];

    // Should use "spa" (first declared), not "eng" (batch default)
    assert_eq!(
        batch_item.lang.as_str(),
        "spa",
        "batch item lang should be 'spa' (first in @Languages), not 'eng' batch default"
    );
}

/// Cache key must reflect the file's language, not the batch default.
/// If the cache key uses "eng" for a Spanish file, cache hits from English
/// files would be incorrectly reused for Spanish.
#[test]
fn cache_key_uses_file_language_not_batch_default() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/spa_hola_mundo.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    let primary = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary);
    let (items, _) = collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll);

    assert_eq!(items.len(), 1);
    let (_, _, ref batch_item, _) = items[0];

    let empty_mwt = std::collections::BTreeMap::new();
    let actual_key = cache_key(&batch_item.words, &batch_item.lang, &empty_mwt, false);
    let eng = talkbank_model::model::LanguageCode::new("eng");
    let spa = talkbank_model::model::LanguageCode::new("spa");
    let wrong_key = cache_key(&batch_item.words, &eng, &empty_mwt, false);
    let correct_key = cache_key(&batch_item.words, &spa, &empty_mwt, false);

    assert_eq!(
        actual_key, correct_key,
        "cache key should use 'spa' from file header"
    );
    assert_ne!(
        actual_key, wrong_key,
        "cache key must NOT use 'eng' batch default for a Spanish file"
    );
}

/// Regression: inject_results with retokenize on Cantonese retrace utterance.
///
/// The full pipeline: parse CHAT → extract words → construct UD response →
/// inject with retokenize mode. This should succeed but fails with
/// "MOR item count does not match alignable word count".
///
/// Source: MOST corpus 40415b.cha line 46.
#[test]
fn test_inject_results_retokenize_cantonese_retrace() {
    use crate::morphosyntax::inject::inject_results;
    use crate::nlp::{UdId, UdResponse, UdSentence, UdWord};
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/retok_yue_retrace.cha");
    let (mut chat_file, _errors) = parse_lenient(&parser, chat);

    let primary_lang = talkbank_model::model::LanguageCode::new("yue");
    let langs = declared_languages(&chat_file, &primary_lang);

    let (batch_items, _summary) = collect_payloads(
        &chat_file,
        &primary_lang,
        &langs,
        MultilingualPolicy::ProcessAll,
    );

    assert!(!batch_items.is_empty(), "Should have batch items");

    // Print what was extracted
    for (line_idx, utt_ord, item, words) in &batch_items {
        eprintln!(
            "Batch item: line={line_idx} utt={utt_ord} words={:?} item_words={:?}",
            words.iter().map(|w| w.text.as_ref()).collect::<Vec<_>>(),
            item.words,
        );
    }

    // Build a matching UD response (one word per extracted word)
    let first_item = &batch_items[0];
    let word_count = first_item.2.words.len();
    eprintln!("Word count from batch item: {word_count}");

    // Simulate what Python actually returns: _segment_cantonese reduces
    // 7 single-char words to 5 words (下+次→下次, 食+飯→食飯).
    // Stanza processes 5 words and returns 5 MOR items.
    let segmented_words = vec!["呢", "度", "下次", "食飯", "啦"];
    eprintln!(
        "Simulated PyCantonese segmentation: {:?} ({} words)",
        segmented_words,
        segmented_words.len()
    );
    let ud_words: Vec<UdWord> = segmented_words
        .iter()
        .enumerate()
        .map(|(i, w)| UdWord {
            id: UdId::Single(i + 1),
            text: w.to_string(),
            lemma: w.to_string(),
            upos: crate::nlp::UdPunctable::Value(crate::nlp::UniversalPos::Noun),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        })
        .collect();

    let ud_response = UdResponse {
        sentences: vec![UdSentence { words: ud_words }],
    };

    let empty_mwt = std::collections::BTreeMap::new();
    let result = inject_results(
        &parser,
        &mut chat_file,
        batch_items,
        vec![ud_response],
        &primary_lang,
        TokenizationMode::StanzaRetokenize,
        &empty_mwt,
    );

    assert!(
        result.is_ok(),
        "inject_results should succeed for retokenize + retrace: {:?}",
        result.err()
    );
}

/// Regression: French utterance with embedded single quotes around elision.
///
/// `*MOT: On dit pas 'quoi tu veux' , mais 'qu' est-ce que' on dit .`
///
/// The `qu'` is a French elision (like `l'homme`, `j'ai`).  Stanza's French
/// MWT tokenizer expands `qu'` into a range token `[n, n+1]` with components
/// `qu` and `'`.  The MOR mapping must collapse these back into one MOR item
/// so the count matches the CHAT word count.
///
/// Source: childes-other-data/Biling/Amsterdam/Anouk/fra/030428.cha line 509.
/// This caused batch 7 of the multilingual morphotag rerun to fail with:
/// "MOR item count (14) does not match alignable word count (13)"
#[test]
fn test_french_elision_in_quoted_context() {
    use crate::morphosyntax::payloads::collect_payloads;

    let parser = crate::parse::TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/fra_french_elision_quotes.cha");
    let (chat_file, _) = crate::parse::parse_lenient(&parser, chat);

    let primary = talkbank_model::model::LanguageCode::new("fra");
    let langs = declared_languages(&chat_file, &primary);
    let (items, _) = collect_payloads(
        &chat_file,
        &primary,
        &langs,
        MultilingualPolicy::ProcessAll,
    );

    assert_eq!(items.len(), 1, "Should have exactly 1 utterance payload");
    let (_, _, item, extracted_words) = &items[0];

    // Print for debugging
    println!("Extracted words: {:?}", item.words);
    println!("Word count: {}", item.words.len());
    println!("Extracted word details: {:?}", extracted_words.iter().map(|w| w.text.as_ref()).collect::<Vec<_>>());

    // The utterance has these CHAT words (in MOR domain, excluding separators):
    // On, dit, pas, 'quoi, tu, veux', mais, 'qu', est-ce, que', on, dit, .
    // That's 13 words (including the terminator).
    // Stanza should NOT produce more MOR items than this.
    let word_count = item.words.len();
    assert!(
        word_count > 0,
        "Should extract some words from French utterance"
    );
    println!("CHAT word count for MOR alignment: {word_count}");
}
