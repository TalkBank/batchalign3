//! Utterance segmentation helpers.
//!
//! Splits a single utterance into multiple utterances based on word-level
//! assignments from a segmentation callback.
//!
//! Also provides types and functions for the server-side utseg orchestrator:
//! payload collection, cache key computation, and result application.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use talkbank_model::Span;
use talkbank_model::alignment::helpers::TierDomain;
use talkbank_model::model::{ChatFile, Line, MainTier, Terminator, Utterance, UtteranceContent};

use crate::extract;

// ---------------------------------------------------------------------------
// Wire types (match Python's UtsegBatchItem / UtsegResponse)
// ---------------------------------------------------------------------------

/// Input payload for a single utterance segmentation request.
///
/// Matches the Python `UtsegBatchItem` Pydantic model.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct UtsegBatchItem {
    /// Tokenized words from the utterance.
    pub words: Vec<String>,
    /// Full utterance text (for constituency parsing).
    pub text: String,
}

/// Response from utterance segmentation inference.
///
/// Each element in `assignments` is a 0-based utterance group ID, parallel
/// to the `words` in the corresponding `UtsegBatchItem`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtsegResponse {
    /// 0-based utterance group ID per word, parallel to `UtsegBatchItem::words`.
    pub assignments: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Payload collection
// ---------------------------------------------------------------------------

/// Collect utseg payloads from all multi-word utterances in a ChatFile.
///
/// Returns `(utt_ordinal, UtsegBatchItem)` pairs. Single-word utterances
/// are skipped (they trivially get assignment `[0]`).
pub fn collect_utseg_payloads(chat_file: &ChatFile) -> Vec<(usize, UtsegBatchItem)> {
    let mut batch_items = Vec::new();
    let mut utt_idx = 0usize;

    for line in chat_file.lines.iter() {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        let mut words = Vec::new();
        extract::collect_utterance_content(&utt.main.content.content, TierDomain::Mor, &mut words);

        if words.len() > 1 {
            // Single pass: build both `text` (space-joined) and `word_texts` together
            let mut text = String::new();
            let mut word_texts = Vec::with_capacity(words.len());
            for (i, w) in words.iter().enumerate() {
                if i > 0 {
                    text.push(' ');
                }
                let s = w.text.as_str();
                text.push_str(s);
                word_texts.push(s.to_string());
            }

            batch_items.push((
                utt_idx,
                UtsegBatchItem {
                    words: word_texts,
                    text,
                },
            ));
        }

        utt_idx += 1;
    }

    batch_items
}

// ---------------------------------------------------------------------------
// Cache key
// ---------------------------------------------------------------------------

/// Compute a BLAKE3 cache key for utseg.
///
/// Format: `BLAKE3("{words.join(' ')}|{lang}")`.
/// Uses incremental hashing to avoid intermediate `String` allocation from `join()`.
pub fn cache_key(words: &[String], lang: &talkbank_model::model::LanguageCode) -> crate::CacheKey {
    let mut hasher = blake3::Hasher::new();
    for (i, w) in words.iter().enumerate() {
        if i > 0 {
            hasher.update(b" ");
        }
        hasher.update(w.as_bytes());
    }
    hasher.update(b"|");
    hasher.update(lang.as_bytes());
    crate::CacheKey::from_hasher(hasher)
}

// ---------------------------------------------------------------------------
// Result application
// ---------------------------------------------------------------------------

/// Apply utseg assignments to a ChatFile, splitting utterances as needed.
///
/// `assignment_map` maps `utt_ordinal` to assignments (parallel to extracted words).
/// Utterances whose ordinals are not in the map are left unchanged.
pub fn apply_utseg_results(chat_file: &mut ChatFile, assignment_map: &HashMap<usize, Vec<usize>>) {
    if assignment_map.is_empty() {
        return;
    }

    let old_lines = std::mem::take(&mut chat_file.lines.0);
    let mut new_lines: Vec<Line> = Vec::with_capacity(old_lines.len());
    let mut utt_ordinal = 0usize;

    for line in old_lines {
        let utt = match line {
            Line::Utterance(u) => u,
            other => {
                new_lines.push(other);
                continue;
            }
        };

        if let Some(assignments) = assignment_map.get(&utt_ordinal) {
            let split_utts = split_utterance(*utt, assignments);
            for split_utt in split_utts {
                new_lines.push(Line::Utterance(Box::new(split_utt)));
            }
        } else {
            new_lines.push(Line::Utterance(utt));
        }

        utt_ordinal += 1;
    }

    chat_file.lines.0 = new_lines;
}

/// Build a mapping from extracted-word index to top-level content item index.
pub fn build_word_to_content_map(content: &[UtteranceContent]) -> Vec<usize> {
    let mut word_to_content = Vec::new();

    for (content_idx, item) in content.iter().enumerate() {
        let mut words = Vec::new();
        extract::collect_utterance_content(std::slice::from_ref(item), TierDomain::Mor, &mut words);
        for _ in &words {
            word_to_content.push(content_idx);
        }
    }

    word_to_content
}

/// Split an utterance into multiple utterances based on word assignments.
///
/// `assignments` is a Vec parallel to the extracted words, where each element
/// is the 0-based utterance ID that word belongs to.
pub fn split_utterance(utt: Utterance, assignments: &[usize]) -> Vec<Utterance> {
    let content_items = &utt.main.content.content;
    let word_to_content = build_word_to_content_map(content_items);

    if assignments.is_empty() || word_to_content.is_empty() {
        return vec![utt];
    }

    let first = assignments[0];
    if assignments.iter().all(|&a| a == first) {
        return vec![utt];
    }

    let num_content_items = content_items.len();
    let mut content_item_group: Vec<Option<usize>> = vec![None; num_content_items];

    for (word_idx, &content_idx) in word_to_content.iter().enumerate() {
        if word_idx < assignments.len() && content_item_group[content_idx].is_none() {
            content_item_group[content_idx] = Some(assignments[word_idx]);
        }
    }

    // Back-fill unassigned items
    let mut last_group: Option<usize> = None;
    for group in content_item_group.iter_mut() {
        if group.is_some() {
            last_group = *group;
        } else {
            *group = last_group;
        }
    }
    // Forward-fill remaining None at the start
    let mut next_group: Option<usize> = None;
    for group in content_item_group.iter_mut().rev() {
        if group.is_some() {
            next_group = *group;
        } else {
            *group = next_group;
        }
    }

    let max_group = assignments.iter().copied().max().unwrap_or(0);

    let mut groups: Vec<Vec<UtteranceContent>> = vec![Vec::new(); max_group + 1];
    for (content_idx, item) in content_items.iter().enumerate() {
        if content_item_group[content_idx].is_none() {
            tracing::warn!(
                content_idx,
                "content item has no group assignment, defaulting to group 0"
            );
        }
        let group_id = content_item_group[content_idx].unwrap_or(0);
        if group_id <= max_group {
            groups[group_id].push(item.clone());
        }
    }

    let speaker = &utt.main.speaker;
    let mut result = Vec::new();

    for group_content in groups {
        if group_content.is_empty() {
            continue;
        }

        let main = MainTier::new(
            speaker.clone(),
            group_content,
            Terminator::Period { span: Span::DUMMY },
        );
        let new_utt = Utterance::new(main);
        result.push(new_utt);
    }

    if result.is_empty() {
        tracing::warn!("utterance segmentation produced no groups, returning original");
        return vec![utt];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use talkbank_parser::DirectParser;
    use talkbank_model::model::WriteChat;

    fn parse_chat(text: &str) -> ChatFile {
        let parser = DirectParser::new().unwrap();
        parser.parse_chat_file(text).unwrap()
    }

    fn get_utterance(chat: &ChatFile, idx: usize) -> &Utterance {
        let mut utt_idx = 0;
        for line in &chat.lines.0 {
            if let Line::Utterance(utt) = line {
                if utt_idx == idx {
                    return utt;
                }
                utt_idx += 1;
            }
        }
        panic!("Utterance {idx} not found");
    }

    fn count_utterances(chat: &ChatFile) -> usize {
        chat.lines
            .iter()
            .filter(|l| matches!(l, Line::Utterance(_)))
            .count()
    }

    #[test]
    fn test_split_no_change() {
        let chat_text = include_str!("../../../test-fixtures/eng_i_eat_cookies.cha");
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();
        let result = split_utterance(utt, &[0, 0, 0]);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_split_two_groups() {
        let chat_text =
            include_str!("../../../test-fixtures/eng_i_eat_cookies_and_he_likes_cake.cha");
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();
        let result = split_utterance(utt, &[0, 0, 0, 1, 1, 1, 1]);
        assert_eq!(result.len(), 2);

        let out0 = result[0].to_chat_string();
        let out1 = result[1].to_chat_string();
        assert!(out0.contains("I eat cookies"), "First split: {out0}");
        assert!(out1.contains("and he likes cake"), "Second split: {out1}");
    }

    #[test]
    fn test_collect_utseg_payloads() {
        // 3 utterances: 1 single-word, 2 multi-word
        let chat_text = include_str!("../../../test-fixtures/eng_three_utterances.cha");
        let chat = parse_chat(chat_text);
        let payloads = collect_utseg_payloads(&chat);

        // Single-word utterance "hello" should be skipped
        assert_eq!(payloads.len(), 2);
        assert_eq!(payloads[0].0, 1); // utt_ordinal of "I eat cookies"
        assert_eq!(payloads[0].1.words, vec!["I", "eat", "cookies"]);
        assert_eq!(payloads[0].1.text, "I eat cookies");
        assert_eq!(payloads[1].0, 2); // utt_ordinal of "he likes cake too"
        assert_eq!(payloads[1].1.words, vec!["he", "likes", "cake", "too"]);
    }

    #[test]
    fn test_utseg_cache_key() {
        let eng = talkbank_model::model::LanguageCode::new("eng");
        let words = vec!["I".to_string(), "eat".to_string(), "cookies".to_string()];
        let key = cache_key(&words, &eng);
        // BLAKE3 of "I eat cookies|eng"
        let expected = blake3::hash(b"I eat cookies|eng").to_hex().to_string();
        assert_eq!(key.as_str(), expected);
    }

    #[test]
    fn test_apply_utseg_results() {
        let chat_text =
            include_str!("../../../test-fixtures/eng_i_eat_cookies_and_he_likes_cake.cha");
        let mut chat = parse_chat(chat_text);
        assert_eq!(count_utterances(&chat), 1);

        let mut assignment_map = HashMap::new();
        assignment_map.insert(0, vec![0, 0, 0, 1, 1, 1, 1]);

        apply_utseg_results(&mut chat, &assignment_map);
        assert_eq!(count_utterances(&chat), 2);

        let out0 = get_utterance(&chat, 0).to_chat_string();
        let out1 = get_utterance(&chat, 1).to_chat_string();
        assert!(out0.contains("I eat cookies"), "First: {out0}");
        assert!(out1.contains("and he likes cake"), "Second: {out1}");
    }

    #[test]
    fn test_apply_utseg_empty_map() {
        let chat_text = include_str!("../../../test-fixtures/eng_i_eat_cookies.cha");
        let mut chat = parse_chat(chat_text);
        let original_count = count_utterances(&chat);

        apply_utseg_results(&mut chat, &HashMap::new());
        assert_eq!(count_utterances(&chat), original_count);
    }

    #[test]
    fn snapshot_utseg_batch_item() {
        let item = UtsegBatchItem {
            words: vec!["I".into(), "eat".into(), "cookies".into()],
            text: "I eat cookies".into(),
        };
        insta::assert_json_snapshot!(item, @r#"
        {
          "words": [
            "I",
            "eat",
            "cookies"
          ],
          "text": "I eat cookies"
        }
        "#);
    }

    #[test]
    fn snapshot_utseg_response() {
        let resp = UtsegResponse {
            assignments: vec![0, 0, 0, 1, 1, 1, 1],
        };
        insta::assert_json_snapshot!(resp, @r#"
        {
          "assignments": [
            0,
            0,
            0,
            1,
            1,
            1,
            1
          ]
        }
        "#);
    }
}
