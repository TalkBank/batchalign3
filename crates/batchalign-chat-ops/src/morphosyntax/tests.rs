//! Tests for morphosyntax module.

use super::*;

#[test]
fn test_cache_key_deterministic() {
    let words = vec!["hello".to_string(), "world".to_string()];
    let eng = talkbank_model::model::LanguageCode::new("eng");
    let empty_mwt = std::collections::BTreeMap::new();
    let k1 = cache_key(&words, &eng, &empty_mwt);
    let k2 = cache_key(&words, &eng, &empty_mwt);
    assert_eq!(k1, k2);
    assert!(!k1.as_str().is_empty());
}

#[test]
fn test_cache_key_lang_differs() {
    let words = vec!["hello".to_string()];
    let eng = talkbank_model::model::LanguageCode::new("eng");
    let spa = talkbank_model::model::LanguageCode::new("spa");
    let empty_mwt = std::collections::BTreeMap::new();
    let k1 = cache_key(&words, &eng, &empty_mwt);
    let k2 = cache_key(&words, &spa, &empty_mwt);
    assert_ne!(k1, k2);
}

#[test]
fn test_clear_morphosyntax() {
    use crate::parse::parse_lenient;
    use talkbank_model::model::Line;

    // Minimal CHAT with %mor and %gra tiers
    let chat = include_str!("../../../../test-fixtures/eng_hello_world_with_mor_gra.cha");
    let (mut chat_file, _errors) = parse_lenient(chat);

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
    use crate::parse::parse_lenient;
    use talkbank_model::model::Line;

    // CHAT with %mor, %gra, and %act -- only %mor/%gra should be removed
    let chat = include_str!("../../../../test-fixtures/eng_hello_world_with_mor_gra_act.cha");
    let (mut chat_file, _) = parse_lenient(chat);

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
    use crate::parse::parse_lenient;
    use talkbank_model::model::Line;

    // CHAT without any dependent tiers -- clear should be a no-op
    let chat = include_str!("../../../../test-fixtures/eng_hello_male.cha");
    let (mut chat_file, _) = parse_lenient(chat);

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
    use crate::parse::parse_lenient;

    // Correctly aligned: 2 main words + 2 %mor items
    let chat = include_str!("../../../../test-fixtures/eng_hello_world_with_mor_gra.cha");
    let (chat_file, _) = parse_lenient(chat);

    let warnings = validate_mor_alignment(&chat_file);
    assert!(
        warnings.is_empty(),
        "expected no alignment warnings, got: {:?}",
        warnings
    );
}

#[test]
fn test_validate_mor_alignment_no_mor_tier() {
    use crate::parse::parse_lenient;

    // No %mor -- validation should pass (nothing to check)
    let chat = include_str!("../../../../test-fixtures/eng_hello_male.cha");
    let (chat_file, _) = parse_lenient(chat);

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
    use crate::parse::parse_lenient;

    let chat = include_str!("../../../../test-fixtures/eng_the_dog_runs.cha");
    let (chat_file, _) = parse_lenient(chat);

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
    use crate::parse::parse_lenient;

    let chat = include_str!("../../../../test-fixtures/spa_hola_que_es_este.cha");
    let (chat_file, _) = parse_lenient(chat);

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
    use crate::parse::parse_lenient;

    let chat = include_str!("../../../../test-fixtures/rus_vot_istoriya.cha");
    let (chat_file, _) = parse_lenient(chat);

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
    use crate::parse::parse_lenient;

    let chat = include_str!("../../../../test-fixtures/zho_hao_qing_zhong.cha");
    let (chat_file, _) = parse_lenient(chat);

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
    use crate::parse::parse_lenient;

    let chat = include_str!("../../../../test-fixtures/fra_lescargot_dort.cha");
    let (chat_file, _) = parse_lenient(chat);

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
    use crate::parse::parse_lenient;

    let chat = include_str!("../../../../test-fixtures/eng_hello_world_male.cha");
    let (chat_file, _) = parse_lenient(chat);

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
    use crate::parse::parse_lenient;

    // Bilingual file: primary declared language is "spa", secondary is "eng"
    let chat = include_str!("../../../../test-fixtures/spa_eng_bilingual_hola_mundo.cha");
    let (chat_file, _) = parse_lenient(chat);

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
    use crate::parse::parse_lenient;

    let chat = include_str!("../../../../test-fixtures/spa_hola_mundo.cha");
    let (chat_file, _) = parse_lenient(chat);

    let primary = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary);
    let (items, _) = collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll);

    assert_eq!(items.len(), 1);
    let (_, _, ref batch_item, _) = items[0];

    let empty_mwt = std::collections::BTreeMap::new();
    let actual_key = cache_key(&batch_item.words, &batch_item.lang, &empty_mwt);
    let eng = talkbank_model::model::LanguageCode::new("eng");
    let spa = talkbank_model::model::LanguageCode::new("spa");
    let wrong_key = cache_key(&batch_item.words, &eng, &empty_mwt);
    let correct_key = cache_key(&batch_item.words, &spa, &empty_mwt);

    assert_eq!(
        actual_key, correct_key,
        "cache key should use 'spa' from file header"
    );
    assert_ne!(
        actual_key, wrong_key,
        "cache key must NOT use 'eng' batch default for a Spanish file"
    );
}
