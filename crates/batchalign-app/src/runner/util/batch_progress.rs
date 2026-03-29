//! Batch-level progress tracking for cross-file batched text commands.
//!
//! The per-file progress model (`set_file_progress`) doesn't work for batched
//! morphotag/utseg/translate/coref because all files' utterances are pooled into
//! language groups and processed as one batch.  Files show `0/N` until the entire
//! batch finishes, then jump to `N/N`.
//!
//! This module adds per-language-group progress: how many utterances have been
//! processed for each language within the current batch.  This gives operators
//! visibility into long-running multilingual batches.
//!
//! # Data flow
//!
//! ```text
//! Python worker → heartbeat with utterance count
//!   → Rust worker handle → BatchInferProgress update
//!     → RunnerEventSink → JobStore → WebSocket → Dashboard
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Progress snapshot for one language group within a batched infer job.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct LanguageGroupProgress {
    /// ISO 639-3 language code (e.g. "fra", "eng").
    pub lang: String,
    /// Number of utterances completed so far.
    pub completed_utterances: u64,
    /// Total utterances in this language group.
    pub total_utterances: u64,
}

impl LanguageGroupProgress {
    /// Create a new progress entry.
    pub fn new(lang: impl Into<String>, completed: u64, total: u64) -> Self {
        Self {
            lang: lang.into(),
            completed_utterances: completed,
            total_utterances: total,
        }
    }

    /// Whether this language group has finished processing.
    pub fn is_complete(&self) -> bool {
        self.completed_utterances >= self.total_utterances
    }

    /// Progress as a fraction in [0.0, 1.0].
    pub fn fraction(&self) -> f64 {
        if self.total_utterances == 0 {
            1.0
        } else {
            self.completed_utterances as f64 / self.total_utterances as f64
        }
    }
}

/// Aggregate progress for a batched infer job across all language groups.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct BatchInferProgress {
    /// Per-language-group progress, keyed by language code.
    /// Uses BTreeMap for deterministic JSON serialization.
    pub language_groups: BTreeMap<String, LanguageGroupProgress>,
}

impl BatchInferProgress {
    /// Create an empty progress tracker.
    pub fn new() -> Self {
        Self {
            language_groups: BTreeMap::new(),
        }
    }

    /// Register a language group with its total utterance count.
    /// Called at dispatch time before inference begins.
    pub fn register_group(&mut self, lang: impl Into<String>, total_utterances: u64) {
        let lang = lang.into();
        self.language_groups.insert(
            lang.clone(),
            LanguageGroupProgress::new(lang, 0, total_utterances),
        );
    }

    /// Update the completed utterance count for a language group.
    /// Called when the worker reports progress via heartbeat.
    pub fn update_group(&mut self, lang: &str, completed_utterances: u64) {
        if let Some(group) = self.language_groups.get_mut(lang) {
            group.completed_utterances = completed_utterances;
        }
    }

    /// Mark a language group as complete.
    pub fn complete_group(&mut self, lang: &str) {
        if let Some(group) = self.language_groups.get_mut(lang) {
            group.completed_utterances = group.total_utterances;
        }
    }

    /// Total utterances across all language groups.
    pub fn total_utterances(&self) -> u64 {
        self.language_groups
            .values()
            .map(|g| g.total_utterances)
            .sum()
    }

    /// Total completed utterances across all language groups.
    pub fn completed_utterances(&self) -> u64 {
        self.language_groups
            .values()
            .map(|g| g.completed_utterances)
            .sum()
    }

    /// Overall progress as a fraction in [0.0, 1.0].
    pub fn overall_fraction(&self) -> f64 {
        let total = self.total_utterances();
        if total == 0 {
            1.0
        } else {
            self.completed_utterances() as f64 / total as f64
        }
    }

    /// Whether all language groups have finished.
    pub fn is_complete(&self) -> bool {
        self.language_groups.values().all(|g| g.is_complete())
    }

    /// Returns language codes for groups that have not yet completed.
    pub fn incomplete_groups(&self) -> Vec<&str> {
        self.language_groups
            .iter()
            .filter(|(_, g)| !g.is_complete())
            .map(|(lang, _)| lang.as_str())
            .collect()
    }

    /// Number of language groups that are still in progress.
    pub fn active_groups(&self) -> usize {
        self.language_groups
            .values()
            .filter(|g| !g.is_complete())
            .count()
    }

    /// Human-readable summary for CLI display.
    ///
    /// Example: "3/5 languages done, 1200/1800 utterances (67%)"
    pub fn summary(&self) -> String {
        let total_groups = self.language_groups.len();
        let complete_groups = total_groups - self.active_groups();
        let completed = self.completed_utterances();
        let total = self.total_utterances();
        let pct = if total > 0 {
            (100 * completed / total) as u32
        } else {
            100
        };
        format!(
            "{complete_groups}/{total_groups} languages done, \
             {completed}/{total} utterances ({pct}%)"
        )
    }
}

impl Default for BatchInferProgress {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_progress_is_complete() {
        let p = BatchInferProgress::new();
        assert!(p.is_complete());
        assert_eq!(p.total_utterances(), 0);
        assert_eq!(p.completed_utterances(), 0);
        assert_eq!(p.active_groups(), 0);
    }

    #[test]
    fn register_and_track_groups() {
        let mut p = BatchInferProgress::new();
        p.register_group("eng", 100);
        p.register_group("fra", 50);

        assert_eq!(p.total_utterances(), 150);
        assert_eq!(p.completed_utterances(), 0);
        assert_eq!(p.active_groups(), 2);
        assert!(!p.is_complete());

        p.update_group("eng", 60);
        assert_eq!(p.completed_utterances(), 60);
        assert_eq!(p.active_groups(), 2);

        p.complete_group("eng");
        assert_eq!(p.completed_utterances(), 100);
        assert_eq!(p.active_groups(), 1);

        p.complete_group("fra");
        assert!(p.is_complete());
        assert_eq!(p.completed_utterances(), 150);
        assert_eq!(p.active_groups(), 0);
    }

    #[test]
    fn overall_fraction_tracks_progress() {
        let mut p = BatchInferProgress::new();
        p.register_group("eng", 100);
        p.register_group("fra", 100);

        assert_eq!(p.overall_fraction(), 0.0);

        p.update_group("eng", 50);
        assert_eq!(p.overall_fraction(), 0.25); // 50/200

        p.complete_group("eng");
        assert_eq!(p.overall_fraction(), 0.5); // 100/200

        p.update_group("fra", 100);
        assert_eq!(p.overall_fraction(), 1.0);
    }

    #[test]
    fn summary_format() {
        let mut p = BatchInferProgress::new();
        p.register_group("eng", 1000);
        p.register_group("fra", 500);
        p.register_group("deu", 300);

        p.complete_group("eng");
        p.update_group("fra", 250);

        let s = p.summary();
        assert!(s.contains("1/3 languages done"), "got: {s}");
        assert!(s.contains("1250/1800"), "got: {s}");
        assert!(s.contains("69%"), "got: {s}");
    }

    #[test]
    fn language_group_progress_fraction() {
        let g = LanguageGroupProgress::new("eng", 75, 100);
        assert_eq!(g.fraction(), 0.75);
        assert!(!g.is_complete());

        let g = LanguageGroupProgress::new("eng", 100, 100);
        assert_eq!(g.fraction(), 1.0);
        assert!(g.is_complete());
    }

    #[test]
    fn zero_total_is_complete() {
        let g = LanguageGroupProgress::new("eng", 0, 0);
        assert!(g.is_complete());
        assert_eq!(g.fraction(), 1.0);
    }

    #[test]
    fn update_unknown_language_is_noop() {
        let mut p = BatchInferProgress::new();
        p.register_group("eng", 100);
        p.update_group("xyz", 50); // unknown language
        assert_eq!(p.completed_utterances(), 0);
    }

    #[test]
    fn incomplete_groups_returns_unfinished() {
        let mut p = BatchInferProgress::new();
        p.register_group("eng", 100);
        p.register_group("fra", 50);
        p.register_group("deu", 30);
        p.complete_group("eng");
        let incomplete = p.incomplete_groups();
        assert_eq!(incomplete.len(), 2);
        assert!(incomplete.contains(&"fra"));
        assert!(incomplete.contains(&"deu"));
    }

    #[test]
    fn deterministic_json_serialization() {
        let mut p = BatchInferProgress::new();
        p.register_group("fra", 50);
        p.register_group("eng", 100);
        p.register_group("deu", 30);

        let json = serde_json::to_string(&p).unwrap();
        // BTreeMap ensures alphabetical order
        let keys: Vec<String> = serde_json::from_str::<BatchInferProgress>(&json)
            .unwrap()
            .language_groups
            .keys()
            .cloned()
            .collect();
        assert_eq!(keys, vec!["deu", "eng", "fra"]);
    }
}
