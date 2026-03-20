use super::*;
use crate::extract;
use mapping::{build_word_token_mapping, try_deterministic_word_token_mapping};
use parse_helpers::{handle_ending_punct_skip, try_parse_token_as_word};
use talkbank_direct_parser::DirectParser;
use talkbank_model::alignment::helpers::TierDomain;
use talkbank_model::model::{ChatFile, GrammaticalRelation, Line, Mor, WriteChat};

fn parse_chat(text: &str) -> ChatFile {
    let parser = DirectParser::new().unwrap();
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

fn extract_words(utt: &Utterance) -> Vec<extract::ExtractedWord> {
    let mut words = Vec::new();
    extract::collect_utterance_content(&utt.main.content.content, TierDomain::Mor, &mut words);
    words
}

/// Parse a %mor tier string into (mors, terminator) using the real parser.
fn parse_mor(line: &str) -> (Vec<Mor>, Option<String>) {
    let errors = talkbank_model::NullErrorSink;
    let outcome = talkbank_direct_parser::mor_tier::parse_mor_tier_content(line, 0, &errors);
    let tier = outcome.into_option().unwrap();
    let terminator = tier.terminator.map(|s| s.to_string());
    let items: Vec<Mor> = tier.items.iter().cloned().collect();
    (items, terminator)
}

/// Parse a %gra tier string into GrammaticalRelations using the real parser.
fn parse_gra(line: &str) -> Vec<GrammaticalRelation> {
    let errors = talkbank_model::NullErrorSink;
    let outcome = talkbank_direct_parser::gra_tier::parse_gra_tier_content(line, 0, &errors);
    let tier = outcome.into_option().unwrap();
    tier.relations.0
}

#[test]
fn test_try_deterministic_word_token_mapping_exact() {
    let chat_text = include_str!("../../../../test-fixtures/retok_i_eat.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    let stanza_tokens = vec!["i".to_string(), "eat".to_string()];

    let mapping = try_deterministic_word_token_mapping(&original_words, &stanza_tokens)
        .expect("deterministic mapping should succeed");
    assert_eq!(mapping.len(), 2);
    assert_eq!(mapping[0].as_slice(), &[0]);
    assert_eq!(mapping[1].as_slice(), &[1]);
}

#[test]
fn test_try_deterministic_word_token_mapping_split_and_merge() {
    let chat_text = include_str!("../../../../test-fixtures/retok_gon_na_eat.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    let stanza_tokens = vec!["gonna".to_string(), "eat".to_string()];

    let mapping = try_deterministic_word_token_mapping(&original_words, &stanza_tokens)
        .expect("deterministic mapping should succeed");
    assert_eq!(mapping.len(), 3);
    assert_eq!(mapping[0].as_slice(), &[0]);
    assert_eq!(mapping[1].as_slice(), &[0]);
    assert_eq!(mapping[2].as_slice(), &[1]);
}

#[test]
fn test_try_deterministic_word_token_mapping_none_on_diverged_text() {
    let chat_text = include_str!("../../../../test-fixtures/retok_i_eat.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    let stanza_tokens = vec!["you".to_string(), "eat".to_string()];

    let mapping = try_deterministic_word_token_mapping(&original_words, &stanza_tokens);
    assert!(
        mapping.is_none(),
        "diverged text should require non-deterministic fallback"
    );
}

#[test]
fn test_build_word_token_mapping_diverged_equal_lengths_indexes_by_position() {
    let chat_text = include_str!("../../../../test-fixtures/retok_gonna_eat.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    let stanza_tokens = vec!["going".to_string(), "eat".to_string()];

    let mapping = build_word_token_mapping(&original_words, &stanza_tokens);
    assert_eq!(mapping.word_count(), 2);
    assert_eq!(mapping.tokens_for_word(0), &[0]);
    assert_eq!(mapping.tokens_for_word(1), &[1]);
}

#[test]
fn test_build_word_token_mapping_diverged_length_mismatch_uses_monotonic_bins() {
    let chat_text = include_str!("../../../../test-fixtures/retok_i_dont_know.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    let stanza_tokens = vec!["alpha".to_string(), "beta".to_string()];

    let mapping = build_word_token_mapping(&original_words, &stanza_tokens);
    assert_eq!(mapping.word_count(), 3);
    assert_eq!(mapping.tokens_for_word(0), &[0]);
    assert_eq!(mapping.tokens_for_word(1), &[0]);
    assert_eq!(mapping.tokens_for_word(2), &[1]);
}

#[test]
fn test_same_tokenization() {
    // 1:1 same text -- words unchanged, mor injected
    let chat_text = include_str!("../../../../test-fixtures/retok_i_eat.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    let stanza_tokens = vec!["I".to_string(), "eat".to_string()];

    let (mors, term) = parse_mor("pro:sub|I v|eat .");
    let gra_rels = parse_gra("1|2|SUBJ 2|0|ROOT 3|2|PUNCT");

    retokenize_utterance(utt, &original_words, &stanza_tokens, mors, term, gra_rels).unwrap();

    let output = chat.to_chat_string();
    assert!(
        output.contains("I eat"),
        "Main tier should be unchanged: {output}"
    );
    assert!(
        output.contains("%mor:\tpro:sub|I v|eat ."),
        "Should have %mor tier: {output}"
    );
    assert!(output.contains("%gra:"), "Should have %gra tier: {output}");
}

#[test]
fn test_different_text_1_to_1() {
    // 1:1 different text -- word text updated
    let chat_text = include_str!("../../../../test-fixtures/retok_gonna_eat.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    let stanza_tokens = vec!["going".to_string(), "eat".to_string()];

    let (mors, term) = parse_mor("aux|go&PRESP v|eat .");
    let gra_rels = parse_gra("1|2|AUX 2|0|ROOT 3|2|PUNCT");

    retokenize_utterance(utt, &original_words, &stanza_tokens, mors, term, gra_rels).unwrap();

    let output = chat.to_chat_string();
    assert!(
        output.contains("going eat"),
        "Word text should be updated: {output}"
    );
    assert!(output.contains("%mor:"), "Should have %mor tier: {output}");
}

#[test]
fn test_clitic_split_1_to_n() {
    // 1:N split -- single word "don't" becomes "do" and "n't"
    let chat_text = include_str!("../../../../test-fixtures/retok_i_dont_know.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    assert_eq!(original_words.len(), 3); // I, don't, know

    let stanza_tokens = vec![
        "I".to_string(),
        "do".to_string(),
        "n't".to_string(),
        "know".to_string(),
    ];

    let (mors, term) = parse_mor("pro:sub|I v|do neg|not v|know .");
    let gra_rels = parse_gra("1|2|SUBJ 2|0|ROOT 3|2|NEG 4|2|XCOMP 5|4|PUNCT");

    retokenize_utterance(utt, &original_words, &stanza_tokens, mors, term, gra_rels).unwrap();

    let output = chat.to_chat_string();
    // After retokenization, "don't" should be split into "do" and "n't"
    assert!(
        output.contains("do n't") || output.contains("do n't"),
        "Should contain split tokens: {output}"
    );
    assert!(output.contains("%mor:"), "Should have %mor tier: {output}");
}

#[test]
fn test_preserves_non_word_content() {
    // Retrace group should be untouched, only non-retraced words retokenized
    let chat_text = include_str!("../../../../test-fixtures/retok_retrace_i_need_cookie.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    // In mor domain, retrace is skipped: words are [I, need, cookie]
    assert_eq!(original_words.len(), 3);

    let stanza_tokens = vec!["I".to_string(), "need".to_string(), "cookie".to_string()];

    let (mors, term) = parse_mor("pro:sub|I v|need n|cookie .");
    let gra_rels = parse_gra("1|2|SUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT");

    retokenize_utterance(utt, &original_words, &stanza_tokens, mors, term, gra_rels).unwrap();

    let output = chat.to_chat_string();
    // Retrace group still present
    assert!(
        output.contains("[/]"),
        "Retrace group should be preserved: {output}"
    );
    assert!(
        output.contains("%mor:\tpro:sub|I v|need n|cookie ."),
        "Should have correct %mor: {output}"
    );
}

#[test]
fn test_xbxxx_restoration() {
    // Special form word: Stanza returns "xbxxx", should be restored to original
    let chat_text = include_str!("../../../../test-fixtures/retok_gumma_special_form.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    assert_eq!(original_words[0].text.as_str(), "gumma");
    assert_eq!(
        original_words[0].form_type,
        Some(talkbank_model::model::FormType::C)
    );

    let stanza_tokens = vec!["xbxxx".to_string(), "is".to_string(), "yummy".to_string()];

    let (mors, term) = parse_mor("c|gumma v|be&3S adj|yummy .");
    let gra_rels = parse_gra("1|2|FLAT 2|0|ROOT 3|2|XCOMP 4|2|PUNCT");

    retokenize_utterance(utt, &original_words, &stanza_tokens, mors, term, gra_rels).unwrap();

    let output = chat.to_chat_string();
    // xbxxx should NOT appear in output -- restored to "gumma"
    assert!(
        !output.contains("xbxxx"),
        "xbxxx should be restored: {output}"
    );
    assert!(
        output.contains("gumma"),
        "Original text should be present: {output}"
    );
}

#[test]
fn test_round_trip_serialization() {
    // Parse -> retokenize -> serialize -> verify output
    let chat_text = include_str!("../../../../test-fixtures/retok_i_dont_know.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);

    let stanza_tokens = vec![
        "I".to_string(),
        "do".to_string(),
        "n't".to_string(),
        "know".to_string(),
    ];

    let mor_str = "pro:sub|I v|do neg|not v|know .";
    let gra_str = "1|2|SUBJ 2|0|ROOT 3|2|NEG 4|2|XCOMP 5|4|PUNCT";

    let (mors, term) = parse_mor(mor_str);
    let gra_rels = parse_gra(gra_str);

    retokenize_utterance(utt, &original_words, &stanza_tokens, mors, term, gra_rels).unwrap();

    let output = chat.to_chat_string();
    // The output should be valid CHAT -- verify it re-parses
    let reparsed = parse_chat(&output);
    let line_count = reparsed
        .lines
        .iter()
        .filter(|l| matches!(l, Line::Utterance(_)))
        .count();
    assert_eq!(line_count, 1, "Should still have 1 utterance");

    // Verify tiers are present
    assert!(output.contains("%mor:"), "Missing %mor tier: {output}");
    assert!(output.contains("%gra:"), "Missing %gra tier: {output}");
}

#[test]
fn test_comma_separator_preserves_words() {
    // Regression test: comma separators are NLP words (cm|cm) but are
    // Separator nodes in the AST. The retokenize walk must increment
    // word_counter for them so subsequent words map correctly.
    let chat_text = include_str!("../../../../test-fixtures/retok_mot_okay_comma.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);

    // Extract should include the comma as an NLP word
    let texts: Vec<&str> = original_words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(
        texts,
        vec!["okay", ",", "push", "the", "start", "button"],
        "Comma should be in extracted words"
    );

    // Stanza tokens: same words, no comma (Stanza doesn't tokenize commas as words)
    let stanza_tokens = vec![
        "okay".to_string(),
        "push".to_string(),
        "the".to_string(),
        "start".to_string(),
        "button".to_string(),
    ];

    let mor_str = "intj|okay cm|cm v|push det|the n|start n|button .";
    let gra_str = "1|3|DISCOURSE 2|1|PUNCT 3|6|ROOT 4|6|DET 5|6|COMPOUND 6|3|OBJ 7|3|PUNCT";

    let (mors, term) = parse_mor(mor_str);
    let gra_rels = parse_gra(gra_str);

    retokenize_utterance(utt, &original_words, &stanza_tokens, mors, term, gra_rels).unwrap();

    let output = chat.to_chat_string();

    // The main tier must preserve all words -- "button" must NOT be dropped
    assert!(
        output.contains("button"),
        "Word 'button' should not be dropped after comma: {output}"
    );
    assert!(
        output.contains("push"),
        "Word 'push' should be preserved: {output}"
    );
    assert!(
        output.contains("start"),
        "Word 'start' should be preserved: {output}"
    );
    assert!(output.contains("%mor:"), "Should have %mor tier: {output}");
}

#[test]
fn test_non_nlp_words_preserved() {
    // Regression test: non-NLP words (xxx, &~uh) must be preserved as-is.
    // They are not extracted by collect_utterance_content, so retokenize
    // must skip them to keep word_counter in sync with the mapping.
    let chat_text = include_str!("../../../../test-fixtures/retok_mot_filler_popcorn.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);

    // &~uh is NOT extracted (non-NLP), only "popcorn" is
    let texts: Vec<&str> = original_words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(texts, vec!["popcorn"], "Only NLP words should be extracted");

    let stanza_tokens = vec!["popcorn".to_string()];

    let mor_str = "n|popcorn !";
    let gra_str = "1|0|ROOT 2|1|PUNCT";

    let (mors, term) = parse_mor(mor_str);
    let gra_rels = parse_gra(gra_str);

    retokenize_utterance(utt, &original_words, &stanza_tokens, mors, term, gra_rels).unwrap();

    let output = chat.to_chat_string();

    // &~uh must still be present -- NOT dropped
    assert!(
        output.contains("&~uh"),
        "Non-NLP word &~uh should be preserved: {output}"
    );
    // popcorn must NOT be duplicated
    let popcorn_count = output.matches("popcorn").count();
    assert!(
        popcorn_count <= 2,
        "popcorn should appear in main tier and %mor only, not duplicated: {output}"
    );
    assert!(output.contains("%mor:"), "Should have %mor tier: {output}");
}

#[test]
fn test_xxx_preserved_in_utterance() {
    // Regression test: xxx (unintelligible marker) must not be dropped or
    // cause adjacent words to duplicate.
    let chat_text = include_str!("../../../../test-fixtures/retok_mot_xxx_put_it_here.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);

    // xxx is NOT extracted (non-NLP)
    let texts: Vec<&str> = original_words.iter().map(|w| w.text.as_str()).collect();
    assert!(
        !texts.contains(&"xxx"),
        "xxx should not be in extracted words: {texts:?}"
    );

    // Stanza tokens = the NLP words (without xxx and comma)
    let stanza_tokens: Vec<String> = texts
        .iter()
        .filter(|t| **t != ",")
        .map(|t| t.to_string())
        .collect();

    // Build a simple mor string for the NLP words
    let mor_str = "intj|okay cm|cm co|I~aux|will adv|just v|put pro|it adv|here .";
    let gra_str =
        "1|6|DISCOURSE 2|1|PUNCT 3|6|SUBJ 4|6|AUX 5|6|ADVMOD 6|0|ROOT 7|6|OBJ 8|6|ADVMOD 9|6|PUNCT";

    let (mors, term) = parse_mor(mor_str);
    let gra_rels = parse_gra(gra_str);

    retokenize_utterance(utt, &original_words, &stanza_tokens, mors, term, gra_rels).unwrap();

    let output = chat.to_chat_string();

    // xxx must still be present
    assert!(output.contains("xxx"), "xxx should be preserved: {output}");
    // "here" must NOT be duplicated (the old bug pattern)
    let main_line = output
        .lines()
        .find(|l| l.starts_with("*MOT:"))
        .unwrap_or("");
    let here_count = main_line.matches("here").count();
    assert_eq!(
        here_count, 1,
        "'here' should appear once in main tier, not duplicated: {main_line}"
    );
}

#[test]
fn test_ending_punct_mismatch_emits_diagnostic() {
    let mut diagnostics = Vec::new();
    let skipped = handle_ending_punct_skip("?", Some("."), &mut diagnostics);
    assert!(skipped, "ending punctuation should be skipped");
    assert_eq!(diagnostics.len(), 1, "mismatch should be diagnosed");
}

#[test]
fn test_try_parse_token_as_word_returns_none_for_invalid() {
    let mut diagnostics = Vec::new();
    let result = try_parse_token_as_word("two words", &mut diagnostics);
    assert!(
        result.is_none(),
        "invalid CHAT token should return None, not an unchecked Word"
    );
    assert!(
        !diagnostics.is_empty(),
        "invalid token should emit diagnostics"
    );
}

#[test]
fn test_whitespace_in_stanza_token_stripped() {
    // Regression: Stanza's Japanese tokenizer sometimes merges adjacent CHAT
    // words into a single token with an embedded space (e.g. "ふ す").
    // The whitespace-stripping in lib.rs normalizes this before calling
    // retokenize_utterance. Here we verify that a token WITHOUT space works
    // correctly as the merged form.
    let chat_text = include_str!("../../../../test-fixtures/retok_jpn_fu_su.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    let texts: Vec<&str> = original_words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(texts, vec!["ふ", "す"]);

    // Simulate: Stanza merged "ふ" and "す" into one token "ふす"
    // (whitespace already stripped by lib.rs normalization)
    let stanza_tokens = vec!["ふす".to_string()];

    let mor_str = "n|ふす .";
    let gra_str = "1|0|ROOT 2|1|PUNCT";

    let (mors, term) = parse_mor(mor_str);
    let gra_rels = parse_gra(gra_str);

    retokenize_utterance(utt, &original_words, &stanza_tokens, mors, term, gra_rels).unwrap();

    let output = chat.to_chat_string();
    // The merged word "ふす" should appear exactly once on the main tier
    let main_line = output
        .lines()
        .find(|l| l.starts_with("*CHI:"))
        .unwrap_or("");
    assert!(
        main_line.contains("ふす"),
        "Merged token should appear on main tier: {main_line}"
    );
    // "ふす" should NOT be duplicated (N:1 merge dedup)
    let count = main_line.matches("ふす").count();
    assert_eq!(
        count, 1,
        "Merged token should appear exactly once, not duplicated: {main_line}"
    );
    assert!(output.contains("%mor:"), "Should have %mor tier: {output}");
}

#[test]
fn test_n_to_1_merge_no_duplication() {
    // Regression: when multiple original words map to the same Stanza token
    // (N:1 merge), only the first original word should emit the merged word.
    // Subsequent original words that map to the same token should be skipped.
    let chat_text = include_str!("../../../../test-fixtures/retok_gon_na_eat.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    let texts: Vec<&str> = original_words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(texts, vec!["gon", "na", "eat"]);

    // Stanza merged "gon" + "na" -> "gonna", "eat" unchanged
    let stanza_tokens = vec!["gonna".to_string(), "eat".to_string()];

    let mor_str = "aux|go&PRESP v|eat .";
    let gra_str = "1|2|AUX 2|0|ROOT 3|2|PUNCT";

    let (mors, term) = parse_mor(mor_str);
    let gra_rels = parse_gra(gra_str);

    retokenize_utterance(utt, &original_words, &stanza_tokens, mors, term, gra_rels).unwrap();

    let output = chat.to_chat_string();
    let main_line = output
        .lines()
        .find(|l| l.starts_with("*CHI:"))
        .unwrap_or("");

    // "gonna" should appear exactly once -- NOT duplicated
    let gonna_count = main_line.matches("gonna").count();
    assert_eq!(
        gonna_count, 1,
        "'gonna' should appear once on main tier (N:1 merge dedup): {main_line}"
    );
    // "eat" should still be present
    assert!(
        main_line.contains("eat"),
        "'eat' should be present: {main_line}"
    );
    assert!(output.contains("%mor:"), "Should have %mor tier: {output}");
}
