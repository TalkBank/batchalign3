//! Build a CHAT file from a structured transcript description.
//!
//! This module constructs a [`ChatFile`] AST from structured input — either
//! a JSON transcript description (for PyO3 bridge compatibility) or typed
//! Rust structs (for the Rust server's transcribe orchestrator).
//!
//! # Two entry points
//!
//! - [`build_chat`] — takes a typed [`TranscriptDescription`] struct
//! - [`build_chat_from_json`] — deserializes JSON into `TranscriptDescription`,
//!   then calls `build_chat`. Used by the PyO3 bridge to delegate here.
//!
//! # Convenience
//!
//! - [`transcript_from_asr_utterances`] — converts post-processed ASR
//!   utterances into a `TranscriptDescription` for CHAT assembly.

use std::path::Path;

use serde::Deserialize;
use talkbank_model::Span;
use talkbank_model::model::{
    BracketedContent, BracketedItem, Bullet, ChatFile, DependentTier, Header,
    IDHeader, LanguageCode, LanguageCodes, Line, MediaHeader, MediaType, ParticipantEntries,
    ParticipantEntry, ParticipantName, ParticipantRole, Retrace, RetraceKind, Separator,
    SpeakerCode, Terminator, Utterance, UtteranceContent, Word,
};

use crate::asr_postprocess;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Structured description of a transcript to be assembled into CHAT format.
///
/// Fields mirror the JSON format accepted by the PyO3 `build_chat()` function.
#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptDescription {
    /// ISO 639-3 language codes (e.g. `["eng"]`). Defaults to `["eng"]` if empty.
    #[serde(default)]
    pub langs: Vec<String>,
    /// Participant entries. At least one is required.
    pub participants: Vec<ParticipantDesc>,
    /// Optional media filename (e.g. `"recording.mp3"`).
    pub media_name: Option<String>,
    /// Optional media type (`"audio"` or `"video"`). Defaults to `"audio"`.
    pub media_type: Option<String>,
    /// Utterances to include in the transcript.
    #[serde(default)]
    pub utterances: Vec<UtteranceDesc>,
    /// Whether to generate `%wor` tiers when word-level timing is available.
    ///
    /// Defaults to `false` (BA2 parity: transcribe omits `%wor` unless
    /// explicitly requested via `--wor`). The JSON bridge (PyO3) defaults to
    /// `false` via serde; callers that want `%wor` must set this to `true`.
    #[serde(default)]
    pub write_wor: bool,
}

/// A participant in the transcript.
#[derive(Debug, Clone, Deserialize)]
pub struct ParticipantDesc {
    /// Speaker code (e.g. `"PAR"`, `"INV"`).
    pub id: String,
    /// Participant name. Defaults to `"Participant"`.
    pub name: Option<String>,
    /// Participant role. Defaults to `"Participant"`.
    pub role: Option<String>,
    /// Corpus name. Defaults to `"corpus_name"`.
    pub corpus: Option<String>,
}

/// An utterance in the transcript.
///
/// Either `words` (word-level with individual timings) or `text` (parse as
/// a single CHAT utterance line) should be provided. If both are present,
/// `words` takes precedence (when non-empty).
#[derive(Debug, Clone, Deserialize)]
pub struct UtteranceDesc {
    /// Speaker code for this utterance.
    pub speaker: String,
    /// Word-level tokens with optional per-word timing.
    pub words: Option<Vec<WordDesc>>,
    /// Full utterance text (alternative to word-level). Parsed via tree-sitter.
    ///
    /// This is a public API surface for callers who want to pass pre-formatted
    /// CHAT text rather than individual word tokens. The text is wrapped in a
    /// mini CHAT document and parsed by `build_text_utterance()`. Currently
    /// unused by the ASR pipeline (which always provides `words`), but
    /// preserved for external JSON API consumers.
    pub text: Option<String>,
    /// Utterance-level start time in ms (used with `text` mode).
    pub start_ms: Option<u64>,
    /// Utterance-level end time in ms (used with `text` mode).
    pub end_ms: Option<u64>,
    /// Detected language for this utterance (ISO 639-3). When set and different
    /// from the primary language (`langs[0]`), a `[- lang]` precode is prepended.
    #[serde(default)]
    pub lang: Option<String>,
}

/// A single word token with optional timing.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WordDesc {
    /// Word text (ready for CHAT assembly via TreeSitterParser).
    pub text: asr_postprocess::ChatWordText,
    /// Start time in milliseconds.
    pub start_ms: Option<u64>,
    /// End time in milliseconds.
    pub end_ms: Option<u64>,
    /// What role this word plays (regular, retrace, etc.).
    #[serde(default)]
    pub kind: asr_postprocess::WordKind,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ENDING_PUNCT: &[&str] = &[
    ".", "?", "!", "+...", "+/.", "+//.", "+/?", "+!?", "+\"/.", "+\".", "+//?", "+..?", "+.",
];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build a CHAT file from a JSON transcript description string.
///
/// This is the entry point used by the PyO3 bridge (`build_chat_inner`).
pub fn build_chat_from_json(json: &str) -> Result<ChatFile, String> {
    let desc: TranscriptDescription =
        serde_json::from_str(json).map_err(|e| format!("Invalid JSON: {e}"))?;
    build_chat(&desc)
}

/// Build a CHAT file from a typed transcript description.
pub fn build_chat(desc: &TranscriptDescription) -> Result<ChatFile, String> {
    let parser = talkbank_parser::TreeSitterParser::new()
        .map_err(|e| format!("Failed to create parser: {e}"))?;
    let langs: Vec<String> = if desc.langs.is_empty() {
        vec!["eng".to_string()]
    } else {
        desc.langs.clone()
    };

    if desc.participants.is_empty() {
        return Err("At least one participant is required".to_string());
    }

    // --- Build participant entries and @ID headers ---
    let mut participant_entries: Vec<ParticipantEntry> = Vec::new();
    let mut id_headers: Vec<IDHeader> = Vec::new();

    for p in &desc.participants {
        let name = p.name.as_deref().unwrap_or("Participant");
        let role = p.role.as_deref().unwrap_or("Participant");
        let corpus = p.corpus.as_deref().unwrap_or("corpus_name");

        let entry = ParticipantEntry {
            speaker_code: SpeakerCode::new(p.id.as_str()),
            name: Some(ParticipantName::new(name)),
            role: ParticipantRole::new(role),
        };
        participant_entries.push(entry);

        let lang_code = langs.first().map(String::as_str).unwrap_or("eng");
        let id = IDHeader::new(lang_code, p.id.as_str(), role).with_corpus(corpus);
        id_headers.push(id);
    }

    // --- Build header lines ---
    let mut lines: Vec<Line> = vec![
        Line::header(Header::Utf8),
        Line::header(Header::Begin),
        Line::header(Header::Languages {
            codes: LanguageCodes::new(langs.iter().map(LanguageCode::new).collect()),
        }),
        Line::header(Header::Participants {
            entries: ParticipantEntries::new(participant_entries),
        }),
    ];
    for id in id_headers {
        lines.push(Line::header(Header::ID(id)));
    }

    // --- Optional @Media header ---
    if let Some(ref media_name) = desc.media_name {
        let normalized_media_name = normalize_media_name(media_name);
        let media_type = match desc.media_type.as_deref() {
            Some("video") => MediaType::Video,
            Some("audio") | None => MediaType::Audio,
            other => {
                tracing::warn!(media_type = ?other, "unrecognized media_type, defaulting to audio");
                MediaType::Audio
            }
        };
        lines.push(Line::header(Header::Media(MediaHeader::new(
            normalized_media_name.as_str(),
            media_type,
        ))));
    }

    // --- Build utterances ---
    let primary_lang = langs.first().map(String::as_str).unwrap_or("eng");
    for utt_desc in &desc.utterances {
        let words = utt_desc.words.as_deref().unwrap_or(&[]);

        if words.is_empty() {
            // Text-level utterance: parse via tree-sitter
            if let Some(ref text) = utt_desc.text
                && let Some(utt_line) = build_text_utterance(
                    &parser,
                    &utt_desc.speaker,
                    text,
                    utt_desc.start_ms,
                    utt_desc.end_ms,
                    &langs,
                )?
            {
                lines.push(utt_line);
            }
            continue;
        }

        // Word-level utterance
        if let Some(mut utt_line) = build_word_utterance(&parser, &utt_desc.speaker, words, desc.write_wor)?
        {
            // Set [- lang] precode when utterance language differs from primary
            if let Some(ref utt_lang) = utt_desc.lang
                && utt_lang != primary_lang
                && let Line::Utterance(ref mut utt) = utt_line
            {
                utt.main.content.language_code = Some(LanguageCode::new(utt_lang.as_str()));
            }
            lines.push(utt_line);
        }
    }

    lines.push(Line::header(Header::End));

    Ok(ChatFile::new(lines))
}

fn normalize_media_name(raw: &str) -> String {
    let candidate = Path::new(raw);
    candidate
        .file_stem()
        .filter(|stem| !stem.is_empty())
        .or_else(|| candidate.file_name())
        .filter(|name| !name.is_empty())
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| raw.to_string())
}

/// Convert post-processed ASR utterances into a [`TranscriptDescription`].
///
/// Speaker indices (0-based) are mapped to `participant_ids`. If a speaker
/// index exceeds the participant list, a generated ID like `"SP1"` is used.
///
/// `write_wor` controls whether the resulting CHAT will include `%wor` tiers
/// when word-level timing is present.
pub fn transcript_from_asr_utterances(
    utterances: &[asr_postprocess::Utterance],
    participant_ids: &[String],
    langs: &[String],
    media_name: Option<&str>,
    write_wor: bool,
) -> TranscriptDescription {
    // Collect unique speaker indices to build participant list
    let mut seen_speakers: Vec<asr_postprocess::SpeakerIndex> = Vec::new();
    for utt in utterances {
        if !seen_speakers.contains(&utt.speaker) {
            seen_speakers.push(utt.speaker);
        }
    }
    seen_speakers.sort_unstable();

    let participants: Vec<ParticipantDesc> = seen_speakers
        .iter()
        .map(|&idx| {
            let i = idx.as_usize();
            let id = if i < participant_ids.len() {
                participant_ids[i].clone()
            } else {
                format!("SP{i}")
            };
            ParticipantDesc {
                id,
                name: None,
                role: None,
                corpus: None,
            }
        })
        .collect();

    let utt_descs: Vec<UtteranceDesc> = utterances
        .iter()
        .map(|utt| {
            let i = utt.speaker.as_usize();
            let speaker_id = if i < participant_ids.len() {
                participant_ids[i].clone()
            } else {
                format!("SP{i}")
            };

            let words: Vec<WordDesc> = utt
                .words
                .iter()
                .map(|w| WordDesc {
                    text: asr_postprocess::ChatWordText::new(w.text.as_str()),
                    start_ms: w.start_ms.map(|ms| ms as u64),
                    end_ms: w.end_ms.map(|ms| ms as u64),
                    kind: w.kind,
                })
                .collect();

            UtteranceDesc {
                speaker: speaker_id,
                words: Some(words),
                text: None,
                start_ms: None,
                end_ms: None,
                lang: utt.lang.clone(),
            }
        })
        .collect();

    TranscriptDescription {
        langs: if langs.is_empty() {
            vec!["eng".to_string()]
        } else {
            langs.to_vec()
        },
        participants,
        media_name: media_name.map(String::from),
        media_type: Some("audio".to_string()),
        utterances: utt_descs,
        write_wor,
    }
}

/// If `text` is a tag-marker separator (comma, tag marker, vocative marker),
/// return the corresponding [`Separator`] model type. Otherwise return `None`.
pub fn tag_marker_separator(text: &str) -> Option<Separator> {
    match text {
        "," => Some(Separator::Comma { span: Span::DUMMY }),
        "\u{201E}" => Some(Separator::Tag { span: Span::DUMMY }),
        "\u{2021}" => Some(Separator::Vocative { span: Span::DUMMY }),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Build a text-level utterance by parsing through tree-sitter.
///
/// This path constructs a minimal valid CHAT document around the input text
/// and parses it with `parse_strict()`. The mini-document hack is necessary
/// because tree-sitter requires complete document context (headers, `@Begin`,
/// `@End`) to parse a single utterance correctly.
///
/// **Callers:** This function is used by the `UtteranceDesc.text` API path —
/// when a caller provides a pre-formatted CHAT utterance string instead of
/// word-level tokens. It has zero production callers in the current codebase
/// (the ASR pipeline always uses word-level `WordDesc` tokens), but it
/// preserves the JSON API contract for external callers who construct
/// `TranscriptDescription` directly. The PyO3 bridge tests exercise this path.
fn build_text_utterance(
    parser: &talkbank_parser::TreeSitterParser,
    speaker: &str,
    text: &str,
    start_ms: Option<u64>,
    end_ms: Option<u64>,
    langs: &[String],
) -> Result<Option<Line>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }

    let bullet_str = match (start_ms, end_ms) {
        (Some(s), Some(e)) => format!(" \x15{}_{}\x15", s, e),
        _ => String::new(),
    };

    let lang_code = langs.first().map(String::as_str).unwrap_or("eng");
    let mini_chat = format!(
        "@UTF8\n@Begin\n@Languages:\t{lang}\n@Participants:\t{spk} Participant Participant\n\
         @ID:\t{lang}|corpus_name|{spk}|||||Participant|||\n*{spk}:\t{text}{bullet}\n@End\n",
        lang = lang_code,
        spk = speaker,
        text = text,
        bullet = bullet_str,
    );

    let parsed = crate::parse::parse_strict(parser, &mini_chat)
        .map_err(|e| format!("Failed to parse text utterance for speaker {speaker}: {e}"))?;

    for parsed_line in parsed.lines.into_iter() {
        if let Line::Utterance(utt) = parsed_line {
            return Ok(Some(Line::Utterance(utt)));
        }
    }

    Ok(None)
}

/// Parse a single word, falling back to unchecked for ASR tokens.
fn parse_asr_word(parser: &talkbank_parser::TreeSitterParser, text: &str) -> Word {
    let errors = talkbank_model::NullErrorSink;
    match parser.parse_word_fragment(text, 0, &errors).into_option() {
        Some(parsed) => parsed,
        None => {
            tracing::warn!(
                word = text,
                "ASR word is not valid CHAT syntax; using unchecked fallback"
            );
            Word::new_unchecked(text, text)
        }
    }
}

/// Parse a word and attach inline bullet timing, updating utterance-level
/// timing bookkeeping. Returns the parsed `Word` and whether timing was present.
fn parse_and_time_word(
    parser: &talkbank_parser::TreeSitterParser,
    text: &str,
    start_ms: Option<u64>,
    end_ms: Option<u64>,
    utt_start_ms: &mut Option<u64>,
    utt_end_ms: &mut Option<u64>,
    has_timing: &mut bool,
) -> Word {
    let mut word = parse_asr_word(parser, text);
    if let (Some(s), Some(e)) = (start_ms, end_ms) {
        word.inline_bullet = Some(Bullet::new(s, e));
        *has_timing = true;
        if utt_start_ms.is_none() {
            *utt_start_ms = Some(s);
        }
        *utt_end_ms = Some(e);
    }
    word
}

/// Build a word-level utterance from individual word tokens.
///
/// When `write_wor` is `true` and word-level timing is present, a `%wor`
/// dependent tier is generated. When `false`, the `%wor` tier is omitted
/// regardless of timing (BA2 default for transcribe).
///
/// Words marked with `WordKind::Retrace` are grouped into consecutive runs
/// and wrapped in proper CHAT retrace AST nodes:
/// - Single-word retrace → `UtteranceContent::AnnotatedWord` with `[/]`
/// - Multi-word retrace → `UtteranceContent::AnnotatedGroup` with `<...> [/]`
fn build_word_utterance(
    parser: &talkbank_parser::TreeSitterParser,
    speaker: &str,
    words: &[WordDesc],
    write_wor: bool,
) -> Result<Option<Line>, String> {
    let mut content: Vec<UtteranceContent> = Vec::new();
    let mut utt_start_ms: Option<u64> = None;
    let mut utt_end_ms: Option<u64> = None;
    let mut has_timing = false;

    // Determine terminator from last word
    let last_text = words.last().map(|w| w.text.as_str()).unwrap_or(".");
    let terminator = match last_text {
        "?" => Terminator::Question { span: Span::DUMMY },
        "!" => Terminator::Exclamation { span: Span::DUMMY },
        "+..." => Terminator::TrailingOff { span: Span::DUMMY },
        "+/." => Terminator::Interruption { span: Span::DUMMY },
        "+//." => Terminator::SelfInterruption { span: Span::DUMMY },
        "+/?" => Terminator::InterruptedQuestion { span: Span::DUMMY },
        "+..?" => Terminator::TrailingOffQuestion { span: Span::DUMMY },
        _ => Terminator::Period { span: Span::DUMMY },
    };

    let mut i = 0;
    while i < words.len() {
        let w = &words[i];
        let text = w.text.as_str().trim();

        if text.is_empty() {
            i += 1;
            continue;
        }

        // Skip ending punctuation (it's captured in the terminator)
        if ENDING_PUNCT.contains(&text) {
            i += 1;
            continue;
        }

        // Tag-marker separators are not words
        if let Some(sep) = tag_marker_separator(text) {
            content.push(UtteranceContent::Separator(sep));
            i += 1;
            continue;
        }

        if w.kind == asr_postprocess::WordKind::Retrace {
            // Collect consecutive retrace words.
            let group_start = i;
            while i < words.len() && words[i].kind == asr_postprocess::WordKind::Retrace {
                i += 1;
            }
            let retrace_words = &words[group_start..i];

            // Parse each retrace word with timing.
            let mut parsed: Vec<Word> = Vec::new();
            for rw in retrace_words {
                let t = rw.text.as_str().trim();
                if t.is_empty() {
                    continue;
                }
                let word = parse_and_time_word(
                    parser,
                    t,
                    rw.start_ms,
                    rw.end_ms,
                    &mut utt_start_ms,
                    &mut utt_end_ms,
                    &mut has_timing,
                );
                parsed.push(word);
            }

            if parsed.is_empty() {
                continue;
            }

            if parsed.len() == 1 {
                // Single-word retrace: word [/]
                let word = parsed.pop().unwrap();
                let bracketed = BracketedContent::new(vec![BracketedItem::Word(Box::new(word))]);
                let retrace = Retrace::new(bracketed, RetraceKind::Partial);
                content.push(UtteranceContent::Retrace(Box::new(retrace)));
            } else {
                // Multi-word retrace: <word word> [/]
                let items: Vec<BracketedItem> = parsed
                    .into_iter()
                    .map(|w| BracketedItem::Word(Box::new(w)))
                    .collect();
                let bracketed = BracketedContent::new(items);
                let retrace = Retrace::new(bracketed, RetraceKind::Partial).as_group();
                content.push(UtteranceContent::Retrace(Box::new(retrace)));
            }
            continue;
        }

        // Regular word
        let word = parse_and_time_word(
            parser,
            text,
            w.start_ms,
            w.end_ms,
            &mut utt_start_ms,
            &mut utt_end_ms,
            &mut has_timing,
        );
        content.push(UtteranceContent::Word(Box::new(word)));
        i += 1;
    }

    if content.is_empty() {
        return Ok(None);
    }

    let mut main = talkbank_model::model::MainTier::new(speaker, content, terminator);
    if let (Some(start), Some(end)) = (utt_start_ms, utt_end_ms) {
        main = main.with_bullet(Bullet::new(start, end));
    }

    let mut utt = Utterance::new(main);
    if write_wor && has_timing {
        let wor_tier = utt.main.generate_wor_tier();
        utt.dependent_tiers.push(DependentTier::Wor(wor_tier));
    }

    Ok(Some(Line::utterance(utt)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{parse_lenient, TreeSitterParser};
    use crate::serialize::to_chat_string;

    /// Helper: create a WordDesc with default kind.
    fn wd(text: &str, start_ms: Option<u64>, end_ms: Option<u64>) -> WordDesc {
        WordDesc {
            text: asr_postprocess::ChatWordText::new(text),
            start_ms,
            end_ms,
            ..Default::default()
        }
    }

    #[test]
    fn test_build_chat_minimal() {
        let desc = TranscriptDescription {
            langs: vec!["eng".to_string()],
            participants: vec![ParticipantDesc {
                id: "PAR".to_string(),
                name: None,
                role: None,
                corpus: None,
            }],
            media_name: None,
            media_type: None,
            utterances: vec![UtteranceDesc {
                speaker: "PAR".to_string(),
                words: Some(vec![
                    wd("hello", None, None),
                    wd("world", None, None),
                    wd(".", None, None),
                ]),
                text: None,
                start_ms: None,
                end_ms: None,
                lang: None,
            }],
            write_wor: false,
        };

        let chat_file = build_chat(&desc).unwrap();
        let output = to_chat_string(&chat_file);
        assert!(output.contains("@Languages:\teng"));
        assert!(output.contains("*PAR:\thello world ."));
    }

    #[test]
    fn test_build_chat_with_timing() {
        let parser = TreeSitterParser::new().unwrap();
        let desc = TranscriptDescription {
            langs: vec!["eng".to_string()],
            participants: vec![ParticipantDesc {
                id: "PAR".to_string(),
                name: None,
                role: None,
                corpus: None,
            }],
            media_name: Some("test.mp3".to_string()),
            media_type: Some("audio".to_string()),
            utterances: vec![UtteranceDesc {
                speaker: "PAR".to_string(),
                words: Some(vec![
                    wd("hello", Some(0), Some(500)),
                    wd("world", Some(500), Some(1000)),
                    wd(".", None, None),
                ]),
                text: None,
                start_ms: None,
                end_ms: None,
                lang: None,
            }],
            write_wor: true,
        };

        let chat_file = build_chat(&desc).unwrap();
        let output = to_chat_string(&chat_file);
        assert!(output.contains("@Media:\ttest, audio"), "got: {output}");
        assert!(output.contains("%wor:"));
        let (_parsed, errors) = parse_lenient(&parser, &output);
        assert!(
            errors.is_empty(),
            "serialized CHAT should reparse cleanly: {errors:?}"
        );
    }

    #[test]
    fn test_build_chat_from_json() {
        let json = r#"{
            "langs": ["eng"],
            "participants": [{"id": "PAR"}],
            "utterances": [
                {"speaker": "PAR", "words": [
                    {"text": "hello"},
                    {"text": "."}
                ]}
            ]
        }"#;

        let chat_file = build_chat_from_json(json).unwrap();
        let output = to_chat_string(&chat_file);
        assert!(output.contains("*PAR:\thello ."));
    }

    #[test]
    fn test_build_chat_text_utterance() {
        let desc = TranscriptDescription {
            langs: vec!["eng".to_string()],
            participants: vec![ParticipantDesc {
                id: "PAR".to_string(),
                name: None,
                role: None,
                corpus: None,
            }],
            media_name: None,
            media_type: None,
            utterances: vec![UtteranceDesc {
                speaker: "PAR".to_string(),
                words: None,
                text: Some("hello world .".to_string()),
                start_ms: Some(0),
                end_ms: Some(1000),
                lang: None,
            }],
            write_wor: false,
        };

        let chat_file = build_chat(&desc).unwrap();
        let output = to_chat_string(&chat_file);
        assert!(output.contains("*PAR:\thello world ."));
    }

    #[test]
    fn test_build_chat_question_terminator() {
        let desc = TranscriptDescription {
            langs: vec!["eng".to_string()],
            participants: vec![ParticipantDesc {
                id: "PAR".to_string(),
                name: None,
                role: None,
                corpus: None,
            }],
            media_name: None,
            media_type: None,
            utterances: vec![UtteranceDesc {
                speaker: "PAR".to_string(),
                words: Some(vec![wd("how", None, None), wd("?", None, None)]),
                text: None,
                start_ms: None,
                end_ms: None,
                lang: None,
            }],
            write_wor: false,
        };

        let chat_file = build_chat(&desc).unwrap();
        let output = to_chat_string(&chat_file);
        assert!(output.contains("*PAR:\thow ?"));
    }

    #[test]
    fn test_write_wor_false_suppresses_wor_tier() {
        let desc = TranscriptDescription {
            langs: vec!["eng".to_string()],
            participants: vec![ParticipantDesc {
                id: "PAR".to_string(),
                name: None,
                role: None,
                corpus: None,
            }],
            media_name: Some("test.mp3".to_string()),
            media_type: Some("audio".to_string()),
            utterances: vec![UtteranceDesc {
                speaker: "PAR".to_string(),
                words: Some(vec![
                    wd("hello", Some(0), Some(500)),
                    wd("world", Some(500), Some(1000)),
                    wd(".", None, None),
                ]),
                text: None,
                start_ms: None,
                end_ms: None,
                lang: None,
            }],
            write_wor: false,
        };

        let chat_file = build_chat(&desc).unwrap();
        let output = to_chat_string(&chat_file);
        assert!(
            !output.contains("%wor:"),
            "write_wor=false should suppress %wor tier, got: {output}"
        );
        // Inline word bullets should still be present
        assert!(
            output.contains("\u{15}"),
            "word-level bullets should still appear on the main tier"
        );
    }

    #[test]
    fn test_transcript_from_asr_utterances() {
        let utterances = vec![
            asr_postprocess::Utterance {
                speaker: asr_postprocess::SpeakerIndex(0),
                words: vec![
                    asr_postprocess::AsrWord::new("hello", Some(0), Some(500)),
                    asr_postprocess::AsrWord::new(".", None, None),
                ],
                lang: None,
            },
            asr_postprocess::Utterance {
                speaker: asr_postprocess::SpeakerIndex(1),
                words: vec![
                    asr_postprocess::AsrWord::new("world", Some(500), Some(1000)),
                    asr_postprocess::AsrWord::new(".", None, None),
                ],
                lang: None,
            },
        ];

        let ids = vec!["PAR".to_string(), "INV".to_string()];
        let desc = transcript_from_asr_utterances(
            &utterances,
            &ids,
            &["eng".to_string()],
            Some("test.mp3"),
            false,
        );

        assert_eq!(desc.participants.len(), 2);
        assert_eq!(desc.participants[0].id, "PAR");
        assert_eq!(desc.participants[1].id, "INV");
        assert_eq!(desc.utterances.len(), 2);
        assert_eq!(desc.utterances[0].speaker, "PAR");
        assert_eq!(desc.utterances[1].speaker, "INV");

        // Should build a valid CHAT file
        let chat_file = build_chat(&desc).unwrap();
        let output = to_chat_string(&chat_file);
        assert!(output.contains("*PAR:"));
        assert!(output.contains("*INV:"));
    }

    #[test]
    fn test_transcript_from_asr_auto_generates_speaker_ids() {
        let utterances = vec![asr_postprocess::Utterance {
            speaker: asr_postprocess::SpeakerIndex(5),
            words: vec![asr_postprocess::AsrWord::new("hello", None, None)],
            lang: None,
        }];

        let desc =
            transcript_from_asr_utterances(&utterances, &[], &["eng".to_string()], None, false);
        assert_eq!(desc.participants[0].id, "SP5");
    }

    #[test]
    fn test_tag_marker_separator() {
        assert!(tag_marker_separator(",").is_some());
        assert!(tag_marker_separator("\u{201E}").is_some());
        assert!(tag_marker_separator("\u{2021}").is_some());
        assert!(tag_marker_separator("hello").is_none());
    }

    #[test]
    fn test_empty_participants_error() {
        let desc = TranscriptDescription {
            langs: vec![],
            participants: vec![],
            media_name: None,
            media_type: None,
            utterances: vec![],
            write_wor: false,
        };
        assert!(build_chat(&desc).is_err());
    }

    // -- Retrace AST construction tests --

    /// Helper: create a retrace WordDesc.
    fn wd_retrace(text: &str, start_ms: Option<u64>, end_ms: Option<u64>) -> WordDesc {
        WordDesc {
            text: asr_postprocess::ChatWordText::new(text),
            start_ms,
            end_ms,
            kind: asr_postprocess::WordKind::Retrace,
        }
    }

    /// Helper: build a single-utterance CHAT file and return serialized output.
    fn build_single_utterance(words: Vec<WordDesc>) -> String {
        let desc = TranscriptDescription {
            langs: vec!["eng".to_string()],
            participants: vec![ParticipantDesc {
                id: "PAR".to_string(),
                name: None,
                role: None,
                corpus: None,
            }],
            media_name: None,
            media_type: None,
            utterances: vec![UtteranceDesc {
                speaker: "PAR".to_string(),
                words: Some(words),
                text: None,
                start_ms: None,
                end_ms: None,
                lang: None,
            }],
            write_wor: false,
        };
        let chat = build_chat(&desc).unwrap();
        to_chat_string(&chat)
    }

    #[test]
    fn single_word_retrace_produces_annotated_word() {
        // "I [/] I went ." → AnnotatedWord with PartialRetracing
        let output = build_single_utterance(vec![
            wd_retrace("I", None, None),
            wd("I", None, None),
            wd("went", None, None),
            wd(".", None, None),
        ]);
        assert!(
            output.contains("I [/] I went ."),
            "expected single-word retrace: {output}"
        );
    }

    #[test]
    fn multi_word_retrace_produces_annotated_group() {
        // "<I want> [/] I want cookie ."
        let output = build_single_utterance(vec![
            wd_retrace("I", None, None),
            wd_retrace("want", None, None),
            wd("I", None, None),
            wd("want", None, None),
            wd("cookie", None, None),
            wd(".", None, None),
        ]);
        assert!(
            output.contains("<I want> [/] I want cookie ."),
            "expected multi-word retrace: {output}"
        );
    }

    #[test]
    fn retrace_preserves_per_word_timing() {
        let output = build_single_utterance(vec![
            wd_retrace("go", Some(0), Some(200)),
            wd("go", Some(200), Some(400)),
            wd("home", Some(400), Some(600)),
            wd(".", None, None),
        ]);
        // The retrace word should have an inline bullet.
        assert!(
            output.contains("\u{15}"),
            "retrace word should preserve timing bullets: {output}"
        );
        assert!(output.contains("[/]"), "expected retrace marker: {output}");
    }

    #[test]
    fn retrace_output_reparses_cleanly() {
        let parser = TreeSitterParser::new().unwrap();
        // Single-word retrace
        let output = build_single_utterance(vec![
            wd_retrace("I", None, None),
            wd("I", None, None),
            wd("went", None, None),
            wd(".", None, None),
        ]);
        let (_parsed, errors) = parse_lenient(&parser, &output);
        assert!(
            errors.is_empty(),
            "single-word retrace should reparse: {errors:?}\noutput: {output}"
        );

        // Multi-word retrace
        let output = build_single_utterance(vec![
            wd_retrace("I", None, None),
            wd_retrace("want", None, None),
            wd("I", None, None),
            wd("want", None, None),
            wd("cookie", None, None),
            wd(".", None, None),
        ]);
        let (_parsed, errors) = parse_lenient(&parser, &output);
        assert!(
            errors.is_empty(),
            "multi-word retrace should reparse: {errors:?}\noutput: {output}"
        );
    }

    #[test]
    fn disfluency_and_retrace_end_to_end() {
        let parser = TreeSitterParser::new().unwrap();
        // Full pipeline: raw ASR → process_raw_asr (includes disfluency + retrace)
        // → transcript_from_asr_utterances → build_chat.
        let output = asr_postprocess::AsrOutput {
            monologues: vec![asr_postprocess::AsrMonologue {
                speaker: asr_postprocess::SpeakerIndex(0),
                elements: vec![
                    asr_postprocess::AsrElement {
                        value: asr_postprocess::AsrRawText::new("um"),
                        ts: asr_postprocess::AsrTimestampSecs(0.0),
                        end_ts: asr_postprocess::AsrTimestampSecs(0.2),
                        kind: asr_postprocess::AsrElementKind::Text,
                    },
                    asr_postprocess::AsrElement {
                        value: asr_postprocess::AsrRawText::new("um"),
                        ts: asr_postprocess::AsrTimestampSecs(0.2),
                        end_ts: asr_postprocess::AsrTimestampSecs(0.4),
                        kind: asr_postprocess::AsrElementKind::Text,
                    },
                    asr_postprocess::AsrElement {
                        value: asr_postprocess::AsrRawText::new("I"),
                        ts: asr_postprocess::AsrTimestampSecs(0.4),
                        end_ts: asr_postprocess::AsrTimestampSecs(0.6),
                        kind: asr_postprocess::AsrElementKind::Text,
                    },
                    asr_postprocess::AsrElement {
                        value: asr_postprocess::AsrRawText::new("went"),
                        ts: asr_postprocess::AsrTimestampSecs(0.6),
                        end_ts: asr_postprocess::AsrTimestampSecs(0.8),
                        kind: asr_postprocess::AsrElementKind::Text,
                    },
                ],
            }],
        };
        let utts = asr_postprocess::process_raw_asr(&output, "eng");

        let desc = transcript_from_asr_utterances(
            &utts,
            &["PAR".to_string()],
            &["eng".to_string()],
            None,
            false,
        );
        let chat = build_chat(&desc).unwrap();
        let serialized = to_chat_string(&chat);

        // Should contain filled pause marker and retrace
        assert!(
            serialized.contains("&-um"),
            "expected filled pause: {serialized}"
        );
        assert!(
            serialized.contains("[/]"),
            "expected retrace marker: {serialized}"
        );

        let (_parsed, errors) = parse_lenient(&parser, &serialized);
        assert!(
            errors.is_empty(),
            "disfluency+retrace should reparse cleanly: {errors:?}\noutput: {serialized}"
        );
    }
}
