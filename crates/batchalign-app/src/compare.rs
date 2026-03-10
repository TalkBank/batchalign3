//! Server-side compare orchestrator.
//!
//! Owns the full CHAT lifecycle for compare jobs:
//! 1. Parse main + gold files
//! 2. Run morphosyntax on main (via existing pipeline)
//! 3. DP-align main vs gold words
//! 4. Inject `%xsrep` tiers into main
//! 5. Serialize annotated CHAT + CSV metrics
//!
//! Gold file convention: for each `FILE.cha`, expects `FILE.gold.cha` in the
//! same directory. Files ending in `.gold.cha` are skipped.

use std::path::Path;

use crate::api::LanguageCode3;
use crate::params::{CachePolicy, MorphosyntaxParams};
use crate::pipeline::PipelineServices;
use batchalign_chat_ops::compare::{
    clear_comparison, compare, format_metrics_csv, inject_comparison,
};
use batchalign_chat_ops::morphosyntax::{MultilingualPolicy, MwtDict, TokenizationMode};
use batchalign_chat_ops::parse::parse_lenient;
use batchalign_chat_ops::serialize::to_chat_string;
use tracing::{info, warn};

use crate::error::ServerError;

/// Process a single CHAT file through the compare pipeline.
///
/// Returns `(annotated_chat_text, metrics_csv)`.
///
/// Steps:
/// 1. Run morphosyntax on `main_text` (so it has %mor/%gra).
/// 2. Parse gold file.
/// 3. Compare main vs gold via DP alignment.
/// 4. Inject `%xsrep` tiers.
/// 5. Serialize and return metrics CSV.
pub(crate) async fn process_compare(
    main_text: &str,
    gold_text: &str,
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    cache_policy: CachePolicy,
    mwt: &MwtDict,
) -> Result<(String, String), ServerError> {
    // 1. Run morphosyntax on main
    let mor_params = MorphosyntaxParams {
        lang,
        tokenization_mode: TokenizationMode::Preserve,
        cache_policy,
        multilingual_policy: MultilingualPolicy::ProcessAll,
        mwt,
    };
    let morphotagged =
        crate::morphosyntax::process_morphosyntax(main_text, services, &mor_params).await?;

    // 2. Parse both files
    let (mut main_file, main_errors) = parse_lenient(&morphotagged);
    if !main_errors.is_empty() {
        warn!(
            num_errors = main_errors.len(),
            "Parse errors in morphotagged main (continuing)"
        );
    }

    let (gold_file, gold_errors) = parse_lenient(gold_text);
    if !gold_errors.is_empty() {
        warn!(
            num_errors = gold_errors.len(),
            "Parse errors in gold file (continuing)"
        );
    }

    // 3. Clear any existing %xsrep and run comparison
    clear_comparison(&mut main_file);
    let result = compare(&main_file, &gold_file);

    info!(
        matches = result.metrics.matches,
        insertions = result.metrics.insertions,
        deletions = result.metrics.deletions,
        wer = %format!("{:.4}", result.metrics.wer),
        "Compare alignment complete"
    );

    // 4. Inject %xsrep tiers
    inject_comparison(&mut main_file, &result);

    // 5. Serialize
    let chat_output = to_chat_string(&main_file);
    let csv_output = format_metrics_csv(&result.metrics);

    Ok((chat_output, csv_output))
}

/// Derive the gold file path from a main file path.
///
/// Convention: `FILE.cha` -> `FILE.gold.cha` (in the same directory).
pub fn gold_path_for(main_path: &str) -> String {
    let p = Path::new(main_path);
    let stem = p.file_stem().unwrap_or_default().to_string_lossy();
    let parent = p.parent().unwrap_or_else(|| Path::new(""));
    parent
        .join(format!("{stem}.gold.cha"))
        .to_string_lossy()
        .to_string()
}

/// Returns `true` if the filename is a gold reference file (ends with `.gold.cha`).
pub fn is_gold_file(filename: &str) -> bool {
    filename.ends_with(".gold.cha")
}

/// Process multiple CHAT files through the compare pipeline.
///
/// For each `(filename, chat_text)`:
/// 1. Skip `.gold.cha` files
/// 2. Look up the companion gold file
/// 3. Run morphosyntax + compare
/// 4. Return `(filename, Ok((chat_text, csv_text)) | Err(error_msg))`
#[allow(dead_code)]
pub(crate) async fn process_compare_batch(
    files: &[(String, String)],
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    cache_policy: CachePolicy,
    mwt: &MwtDict,
    read_gold_fn: &dyn Fn(&str) -> Option<String>,
) -> Vec<(String, Result<(String, String), String>)> {
    let mut results = Vec::with_capacity(files.len());

    for (filename, chat_text) in files {
        // Skip gold files — they're companions, not inputs
        if is_gold_file(filename) {
            continue;
        }

        let gold_filename = gold_path_for(filename);
        let gold_text = match read_gold_fn(&gold_filename) {
            Some(text) => text,
            None => {
                results.push((
                    filename.clone(),
                    Err(format!(
                        "No gold .cha file found for comparison. \
                         main: {filename}, expected: {gold_filename}"
                    )),
                ));
                continue;
            }
        };

        match process_compare(chat_text, &gold_text, lang, services, cache_policy, mwt).await {
            Ok(result) => {
                results.push((filename.clone(), Ok(result)));
            }
            Err(e) => {
                results.push((filename.clone(), Err(e.to_string())));
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gold_path_derivation() {
        assert_eq!(gold_path_for("test.cha"), "test.gold.cha");
        assert_eq!(
            gold_path_for("/data/corpus/01DM.cha"),
            "/data/corpus/01DM.gold.cha"
        );
        assert_eq!(gold_path_for("dir/sub/file.cha"), "dir/sub/file.gold.cha");
    }

    #[test]
    fn gold_file_detection() {
        assert!(is_gold_file("test.gold.cha"));
        assert!(is_gold_file("/data/01DM.gold.cha"));
        assert!(!is_gold_file("test.cha"));
        assert!(!is_gold_file("test.gold.txt"));
    }
}
