//! Dependent tier management — add user-defined tiers with validation.

use pyo3::PyResult;
use talkbank_model::model::Line;

use crate::TierEntryJson;

/// Standard CHAT dependent tier labels that have their own structured variants.
/// User-defined tiers must NOT use these — they must start with 'x'.
pub(crate) const STANDARD_TIER_LABELS: &[&str] = &[
    "mor", "gra", "pho", "mod", "sin", "act", "cod", "add", "com", "exp", "gpx", "int", "sit",
    "spa", "alt", "coh", "def", "eng", "err", "fac", "flo", "gls", "ort", "par", "tim", "wor",
    "trn",
];

/// Validate that a tier label is appropriate for a user-defined dependent tier.
pub(crate) fn validate_user_tier_label(label: &str) -> Result<(), String> {
    if STANDARD_TIER_LABELS.contains(&label) {
        return Err(format!(
            "Tier label '{label}' is a standard CHAT tier and cannot be used as a \
             user-defined tier. Use 'x{label}' instead."
        ));
    }
    if !label.starts_with('x') {
        return Err(format!(
            "User-defined tier label '{label}' must start with 'x' (e.g., 'x{label}'). \
             See CHAT manual: User Defined Tiers."
        ));
    }
    Ok(())
}

pub(crate) fn add_dependent_tiers_inner(
    chat_file: &mut talkbank_model::model::ChatFile,
    tiers_json: &str,
) -> PyResult<()> {
    use talkbank_model::model::{DependentTier, NonEmptyString, UserDefinedDependentTier};

    let entries: Vec<TierEntryJson> = serde_json::from_str(tiers_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid tiers JSON: {e}")))?;

    // Validate all labels upfront before mutating any utterances
    for entry in &entries {
        if !entry.label.is_empty() && !entry.content.is_empty() {
            validate_user_tier_label(&entry.label)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
        }
    }

    let mut by_utt: std::collections::HashMap<usize, Vec<&TierEntryJson>> =
        std::collections::HashMap::new();
    for entry in &entries {
        by_utt.entry(entry.utterance_index).or_default().push(entry);
    }

    let mut utt_idx = 0usize;
    for line in chat_file.lines.iter_mut() {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        if let Some(tier_entries) = by_utt.get(&utt_idx) {
            for entry in tier_entries {
                if let (Some(label), Some(content)) = (
                    NonEmptyString::new(&entry.label),
                    NonEmptyString::new(&entry.content),
                ) {
                    let new_tier = DependentTier::UserDefined(UserDefinedDependentTier {
                        label: label.clone(),
                        content,
                        span: talkbank_model::Span::DUMMY,
                    });

                    let mut found = false;
                    for tier in utt.dependent_tiers.iter_mut() {
                        if matches!(tier, DependentTier::UserDefined(t) if t.label.as_ref() == label.as_ref())
                        {
                            *tier = new_tier.clone();
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        utt.dependent_tiers.push(new_tier);
                    }
                }
            }
        }

        utt_idx += 1;
    }

    Ok(())
}
