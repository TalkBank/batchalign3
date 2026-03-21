//! Server-side benchmark orchestrator.
//!
//! Benchmarking is conceptually "transcribe, then compare against a gold CHAT
//! companion". Neither half requires Python-side document orchestration:
//! - transcription already has a Rust-owned pipeline around raw ASR inference
//! - comparison is already Rust-owned morphosyntax + DP alignment
//!
//! This module composes those two existing Rust pipelines so the `benchmark`
//! command no longer depends on a fictitious Python worker benchmark path.

use std::path::Path;

use crate::error::ServerError;
use crate::workflow::CompositeWorkflow;
use crate::workflow::benchmark::BenchmarkWorkflow;
pub(crate) use crate::workflow::benchmark::BenchmarkWorkflowRequest as BenchmarkRequest;
pub(crate) use crate::workflow::compare::CompareMaterializedOutputs as BenchmarkOutputs;

/// Run the benchmark pipeline for one audio file and one gold CHAT transcript.
///
/// Returns a [`BenchmarkOutputs`] containing the annotated CHAT and CSV metrics.
pub(crate) async fn process_benchmark(
    request: BenchmarkRequest<'_>,
) -> Result<BenchmarkOutputs, ServerError> {
    BenchmarkWorkflow::new().run(request).await
}

/// Derive the companion gold CHAT path for one audio file.
///
/// Convention:
/// - `sample.wav` -> `sample.cha`
/// - `/dir/sample.mp3` -> `/dir/sample.cha`
pub(crate) fn gold_chat_path_for_audio(audio_path: &str) -> String {
    let path = Path::new(audio_path);
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    parent
        .join(format!("{stem}.cha"))
        .to_string_lossy()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::gold_chat_path_for_audio;

    #[test]
    fn derives_gold_chat_path_from_audio() {
        assert_eq!(gold_chat_path_for_audio("sample.wav"), "sample.cha");
        assert_eq!(
            gold_chat_path_for_audio("/data/interview.mp3"),
            "/data/interview.cha"
        );
    }
}
