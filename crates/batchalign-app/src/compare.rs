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

use crate::api::{ChatText, LanguageCode3};
use crate::params::CachePolicy;
use crate::pipeline::PipelineServices;
use batchalign_chat_ops::morphosyntax::MwtDict;

use crate::error::ServerError;
use crate::workflow::ReferenceProjectionWorkflow;
pub(crate) use crate::workflow::compare::CompareMaterializedOutputs;
use crate::workflow::compare::{CompareWorkflow, CompareWorkflowRequest};
use crate::workflow::text_batch::TextBatchFileInput;

/// Process a single CHAT file through the compare pipeline.
///
/// Returns the released compare outputs for the current main-annotated workflow
/// materialization.
///
/// Steps:
/// 1. Run morphosyntax on `main_text` (so it has %mor/%gra).
/// 2. Parse gold file.
/// 3. Build the comparison bundle from main vs gold.
/// 4. Materialize the current main-annotated output.
pub(crate) async fn process_compare(
    main_text: &str,
    gold_text: &str,
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    cache_policy: CachePolicy,
    mwt: &MwtDict,
) -> Result<CompareMaterializedOutputs, ServerError> {
    CompareWorkflow::released()
        .run(CompareWorkflowRequest {
            main_text: ChatText::from(main_text),
            gold_text: ChatText::from(gold_text),
            lang,
            services,
            cache_policy,
            mwt,
        })
        .await
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
/// 4. Return `(filename, Ok(outputs) | Err(error_msg))`
#[allow(dead_code)]
pub(crate) async fn process_compare_batch(
    files: &[TextBatchFileInput],
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    cache_policy: CachePolicy,
    mwt: &MwtDict,
    read_gold_fn: &dyn Fn(&str) -> Option<String>,
) -> Vec<(String, Result<CompareMaterializedOutputs, String>)> {
    let mut results = Vec::with_capacity(files.len());

    for file in files {
        let filename = file.filename.as_ref();
        let chat_text = file.chat_text.as_ref();
        // Skip gold files — they're companions, not inputs
        if is_gold_file(filename) {
            continue;
        }

        let gold_filename = gold_path_for(filename);
        let gold_text = match read_gold_fn(&gold_filename) {
            Some(text) => text,
            None => {
                results.push((
                    file.filename.to_string(),
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
                results.push((file.filename.to_string(), Ok(result)));
            }
            Err(e) => {
                results.push((file.filename.to_string(), Err(e.to_string())));
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
