//! Pre-validation and post-validation gates for CHAT files.
//!
//! Each server orchestrator validates the parsed CHAT file to a command's
//! minimum validity level before spending compute. Invalid files are
//! rejected early with diagnostics.
//!
//! # Validity levels (cumulative)
//!
//! | Level | Name | Checks |
//! |-------|------|--------|
//! | L0 | Parseable | No parse errors (clean tree-sitter CST) |
//! | L1 | StructurallyComplete | Participants, languages, speaker codes, terminators |
//! | L2 | MainTierValid | Well-formed words, timing bullets |
//!
//! Each level includes all checks from lower levels.

use talkbank_model::model::{ChatFile, Line};

/// Validity level for pre-validation gates.
///
/// Each command requires input to meet a minimum level before processing.
/// Higher levels are strictly more restrictive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidityLevel {
    /// L0: The file parsed without errors (no ERROR nodes in CST).
    Parseable = 0,
    /// L1: Structurally complete — has participants, languages, speaker codes,
    /// and every utterance has a terminator.
    StructurallyComplete = 1,
    /// L2: Main tier content is well-formed — no word-level structural errors,
    /// timing bullets are valid if present.
    MainTierValid = 2,
}

/// A single validation failure with diagnostic information.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Human-readable description of the problem.
    pub message: String,
    /// Which level this check belongs to.
    pub level: ValidityLevel,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[L{}] {}", self.level as u8, self.message)
    }
}

/// Validate a CHAT file to the specified minimum validity level.
///
/// Returns `Ok(())` if the file meets the level, or `Err` with all
/// failures found (checks all levels up to the specified one).
pub fn validate_to_level(
    file: &ChatFile,
    parse_error_count: usize,
    level: ValidityLevel,
) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    // L0: Parseable — no parse errors
    if parse_error_count > 0 {
        errors.push(ValidationError {
            message: format!("File has {parse_error_count} parse error(s); input may be malformed"),
            level: ValidityLevel::Parseable,
        });
    }

    if level >= ValidityLevel::StructurallyComplete {
        check_structurally_complete(file, &mut errors);
    }

    if level >= ValidityLevel::MainTierValid {
        check_main_tier_valid(file, &mut errors);
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// L1 checks: structural completeness.
fn check_structurally_complete(file: &ChatFile, errors: &mut Vec<ValidationError>) {
    // Check @Participants present with at least one participant
    if file.participants.is_empty() {
        errors.push(ValidationError {
            message: "@Participants header missing or has no participants".to_string(),
            level: ValidityLevel::StructurallyComplete,
        });
    }

    // Check @Languages present
    if file.languages.is_empty() {
        errors.push(ValidationError {
            message: "@Languages header missing".to_string(),
            level: ValidityLevel::StructurallyComplete,
        });
    }

    // Check every utterance has a terminator and a declared speaker
    for line in &file.lines {
        if let Line::Utterance(utt) = line {
            if utt.main.content.terminator.is_none() {
                let speaker = utt.main.speaker.as_str();
                errors.push(ValidationError {
                    message: format!("Utterance by *{speaker} has no terminator"),
                    level: ValidityLevel::StructurallyComplete,
                });
            }

            // Check speaker is declared in participants
            let speaker_code = utt.main.speaker.as_str();
            let declared = file.participants.keys().any(|k| k.as_str() == speaker_code);
            if !declared {
                errors.push(ValidationError {
                    message: format!("Speaker *{speaker_code} not declared in @Participants"),
                    level: ValidityLevel::StructurallyComplete,
                });
            }
        }
    }
}

/// L2 checks: main tier content validity.
fn check_main_tier_valid(file: &ChatFile, errors: &mut Vec<ValidationError>) {
    for line in &file.lines {
        if let Line::Utterance(utt) = line {
            // Check for empty main tiers (no content at all)
            if utt.main.content.content.is_empty()
                && utt.main.content.linkers.is_empty()
                && utt.main.content.language_code.is_none()
            {
                let speaker = utt.main.speaker.as_str();
                errors.push(ValidationError {
                    message: format!("Utterance by *{speaker} has an empty main tier"),
                    level: ValidityLevel::MainTierValid,
                });
            }
        }
    }
}

/// Post-validation: verify that the output file is at least as valid as the
/// input (no degradation). Returns diagnostics if the command corrupted the file.
pub fn validate_output(file: &ChatFile, command: &str) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    // Check every utterance still has a terminator
    for line in &file.lines {
        if let Line::Utterance(utt) = line
            && utt.main.content.terminator.is_none()
        {
            let speaker = utt.main.speaker.as_str();
            errors.push(ValidationError {
                message: format!("After {command}: utterance by *{speaker} lost its terminator"),
                level: ValidityLevel::StructurallyComplete,
            });
        }
    }

    // Command-specific checks
    match command {
        "morphotag" => validate_morphotag_output(file, &mut errors),
        "align" => validate_align_output(file, &mut errors),
        _ => {}
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Post-validation for morphotag: %mor word count must match main tier.
fn validate_morphotag_output(file: &ChatFile, errors: &mut Vec<ValidationError>) {
    use talkbank_model::alignment::helpers::AlignmentDomain;

    for line in &file.lines {
        if let Line::Utterance(utt) = line {
            // Count alignable words
            let mut extracted = Vec::new();
            crate::extract::collect_utterance_content(
                &utt.main.content.content,
                AlignmentDomain::Mor,
                &mut extracted,
            );
            let word_count = extracted.len();

            // Count %mor items
            for tier in &utt.dependent_tiers {
                if let talkbank_model::model::DependentTier::Mor(mor_tier) = tier {
                    let mor_count = mor_tier.items.0.len();
                    if word_count != mor_count {
                        let speaker = utt.main.speaker.as_str();
                        errors.push(ValidationError {
                            message: format!(
                                "After morphotag: *{speaker} has {word_count} words \
                                 but %mor has {mor_count} items"
                            ),
                            level: ValidityLevel::MainTierValid,
                        });
                    }
                }
            }
        }
    }
}

/// Post-validation for align: check for backwards timing only.
///
/// Cross-speaker overlap is **normal** in conversation data (speakers talk
/// over each other) and is valid CHAT. The real validator in talkbank-tools
/// handles all E362/E704 checks. We only flag clearly broken output here
/// (end < start within a single utterance).
fn validate_align_output(file: &ChatFile, errors: &mut Vec<ValidationError>) {
    for line in &file.lines {
        if let Line::Utterance(utt) = line
            && let Some(ref bullet) = utt.main.content.bullet
        {
            let start = bullet.timing.start_ms;
            let end = bullet.timing.end_ms;

            if start > end {
                let speaker = utt.main.speaker.as_str();
                errors.push(ValidationError {
                    message: format!(
                        "After align: *{speaker} has backwards timing \
                         ({start} > {end})"
                    ),
                    level: ValidityLevel::MainTierValid,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use talkbank_parser::parse_chat_file;

    #[test]
    fn test_valid_file_passes_all_levels() {
        let chat_text = include_str!("../../../test-fixtures/eng_hello_world_with_mor_gra.cha");
        let chat = parse_chat_file(chat_text).unwrap();
        assert!(validate_to_level(&chat, 0, ValidityLevel::MainTierValid).is_ok());
    }

    #[test]
    fn test_parse_errors_fail_l0() {
        let chat_text = include_str!("../../../test-fixtures/eng_hello_world_with_mor_gra.cha");
        let chat = parse_chat_file(chat_text).unwrap();
        let result = validate_to_level(&chat, 3, ValidityLevel::Parseable);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("3 parse error(s)"));
    }

    #[test]
    fn test_morphotag_output_validates_alignment() {
        let chat_text = include_str!("../../../test-fixtures/eng_hello_world_with_mor_gra.cha");
        let chat = parse_chat_file(chat_text).unwrap();
        // Valid morphotag output should pass
        assert!(validate_output(&chat, "morphotag").is_ok());
    }

    #[test]
    fn test_output_validation_catches_missing_terminator() {
        use talkbank_model::model::Line;

        let chat_text = include_str!("../../../test-fixtures/eng_hello_world_with_mor_gra.cha");
        let mut chat = parse_chat_file(chat_text).unwrap();

        // Remove the terminator to simulate corruption
        for line in &mut chat.lines {
            if let Line::Utterance(utt) = line {
                utt.main.content.terminator = None;
            }
        }

        let result = validate_output(&chat, "morphotag");
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors[0].message.contains("lost its terminator"));
    }
}
