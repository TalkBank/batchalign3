//! Forced alignment — delegates to `batchalign_chat_ops::fa`.
//!
//! Only `build_fa_item` is defined locally for the typed PyO3 callback path.
//! All other types and functions are re-exported from the shared crate.

pub use batchalign_chat_ops::fa::{
    FaGroup, FaInferItem, FaTimingMode, WordTiming, add_wor_tier, group_utterances,
    inject_timings_for_utterance, parse_fa_response, postprocess_utterance_timings,
    update_utterance_bullet,
};

/// Build a typed FA payload for the PyO3 callback.
pub fn build_fa_item(group: &FaGroup, timing_mode: FaTimingMode) -> FaInferItem {
    FaInferItem {
        words: group.words.iter().map(|word| word.text.clone()).collect(),
        word_ids: group.words.iter().map(|word| word.stable_id()).collect(),
        word_utterance_indices: group
            .words
            .iter()
            .map(|word| word.utterance_index.raw())
            .collect(),
        word_utterance_word_indices: group
            .words
            .iter()
            .map(|word| word.utterance_word_index.raw())
            .collect(),
        audio_path: String::new(),
        audio_start_ms: group.audio_start_ms(),
        audio_end_ms: group.audio_end_ms(),
        timing_mode,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use batchalign_chat_ops::fa::FaWord;
    use batchalign_chat_ops::indices::{UtteranceIdx, WordIdx};

    #[test]
    fn test_build_fa_item() {
        let group = FaGroup {
            audio_span: batchalign_chat_ops::fa::TimeSpan::new(1500, 3200),
            words: vec![
                FaWord {
                    utterance_index: UtteranceIdx(0),
                    utterance_word_index: WordIdx(0),
                    text: "hello".into(),
                },
                FaWord {
                    utterance_index: UtteranceIdx(0),
                    utterance_word_index: WordIdx(1),
                    text: "world".into(),
                },
            ],
            utterance_indices: vec![UtteranceIdx(0)],
        };

        let payload = build_fa_item(&group, FaTimingMode::Continuous);
        assert_eq!(payload.words, vec!["hello", "world"]);
        assert_eq!(payload.audio_start_ms, 1500);
        assert_eq!(payload.audio_end_ms, 3200);
        assert_eq!(payload.word_ids, vec!["u0:w0", "u0:w1"]);
        assert_eq!(payload.word_utterance_indices, vec![0, 0]);
        assert_eq!(payload.word_utterance_word_indices, vec![0, 1]);
        assert_eq!(payload.timing_mode, FaTimingMode::Continuous);
    }
}
