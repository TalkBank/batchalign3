//! Test helpers — pure-Rust equivalents of #[pyfunction]s (no Python<'_>).

use talkbank_model::WriteChat;
use talkbank_model::alignment::helpers::TierDomain;
use talkbank_model::model::Line;

use crate::ParsedChat;
use crate::build::build_chat_inner;
use crate::metadata::extract_metadata_from_chat_file_pure;
use crate::metadata::serialize_extracted_words;
use crate::parse::{
    parse_lenient_pure, parse_lenient_with_warnings, parse_strict_pure, strip_timing_on_chat_file,
};
use crate::speaker_ops::{add_utterance_timing_inner, reassign_speakers_inner};

pub fn strip_timing(chat_text: &str) -> Result<String, String> {
    let mut chat_file = parse_lenient_pure(chat_text)?;
    strip_timing_on_chat_file(&mut chat_file);
    let val_errors = talkbank_model::ErrorCollector::new();
    chat_file.validate(&val_errors, None);
    Ok(chat_file.to_chat_string())
}

pub fn parse_and_serialize(chat_text: &str) -> Result<String, String> {
    parse_strict_pure(chat_text).map(|cf| cf.to_chat_string())
}

pub fn extract_nlp_words(chat_text: &str, domain: &str) -> Result<String, String> {
    let domain = match domain {
        "mor" => TierDomain::Mor,
        "wor" => TierDomain::Wor,
        "pho" => TierDomain::Pho,
        "sin" => TierDomain::Sin,
        _ => return Err(format!("Invalid domain: {domain:?}")),
    };
    let chat_file = parse_strict_pure(chat_text)?;
    let extracted = crate::extract::extract_words(&chat_file, domain);
    Ok(serialize_extracted_words(&extracted))
}

pub fn extract_metadata(chat_text: &str) -> Result<String, String> {
    let chat_file = parse_lenient_pure(chat_text)?;
    extract_metadata_from_chat_file_pure(&chat_file)
}

pub fn reassign_speakers(
    chat_text: &str,
    segments_json: &str,
    lang: &str,
) -> Result<String, String> {
    let chat_file = parse_lenient_pure(chat_text)?;
    reassign_speakers_inner(chat_file, segments_json, lang)
        .map(|cf| cf.to_chat_string())
        .map_err(|e| e.to_string())
}

pub fn add_utterance_timing(chat_text: &str, asr_words_json: &str) -> Result<String, String> {
    let mut chat_file = parse_strict_pure(chat_text)?;
    add_utterance_timing_inner(&mut chat_file, asr_words_json).map_err(|e| e.to_string())?;
    Ok(chat_file.to_chat_string())
}

pub fn build_chat(transcript_json: &str) -> Result<String, String> {
    use talkbank_model::model::Provenance;
    build_chat_inner(Provenance::new(transcript_json.to_string())).map(|cf| cf.to_chat_string())
}

pub fn make_handle(chat_text: &str) -> ParsedChat {
    ParsedChat {
        inner: parse_strict_pure(chat_text).unwrap(),
        warnings: vec![],
    }
}

pub fn make_handle_lenient(chat_text: &str) -> ParsedChat {
    let (cf, warnings) = parse_lenient_with_warnings(chat_text).unwrap();
    ParsedChat {
        inner: cf,
        warnings,
    }
}

pub fn validate_alignments(chat_text: &str) -> Vec<String> {
    let chat_file = parse_lenient_pure(chat_text).unwrap();
    chat_file
        .validate_alignments()
        .into_iter()
        .map(|e| format!("{}", e))
        .collect()
}

pub fn add_comment(handle: &mut ParsedChat, comment: &str) {
    use talkbank_model::model::{BulletContent, Header};
    let comment_line = Line::header(Header::Comment {
        content: BulletContent::from_text(comment),
    });
    let pos = handle
        .inner
        .lines
        .iter()
        .position(|l| matches!(l, Line::Utterance(_)))
        .unwrap_or(handle.inner.lines.len());
    handle.inner.lines.0.insert(pos, comment_line);
}
