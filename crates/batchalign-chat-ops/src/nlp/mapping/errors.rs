//! Structured error type for UD-to-CHAT mapping failures.

/// Structured error type for UD-to-CHAT mapping failures.
#[derive(Debug, thiserror::Error)]
pub enum MappingError {
    /// A word produced an empty MOR stem after lemma cleaning and sanitization.
    /// Serializes as `pos|` (bare pipe) which is invalid CHAT.
    #[error("Empty MOR stem: word={word:?}, lemma={lemma:?}, upos={upos:?}")]
    EmptyStem {
        /// Original word form.
        word: String,
        /// Lemma after cleaning.
        lemma: String,
        /// Universal POS tag.
        upos: String,
    },

    /// The generated %gra tier has a circular dependency (head chain loops).
    #[error("Circular dependency in generated %gra: {details}")]
    CircularDependency {
        /// Description of the cycle.
        details: String,
    },

    /// The generated %gra tier has an invalid head reference.
    #[error("Invalid head reference in generated %gra: {details}")]
    InvalidHeadReference {
        /// Description of the invalid reference.
        details: String,
    },

    /// Generated %mor and %gra have mismatched chunk counts.
    #[error("%mor has {mor_chunks} chunks but %gra has {gra_count} relations")]
    ChunkCountMismatch {
        /// Number of %mor chunks.
        mor_chunks: usize,
        /// Number of %gra relations.
        gra_count: usize,
    },

    /// The generated %gra tier has no root or multiple roots.
    #[error("Invalid root structure in generated %gra: {details}")]
    InvalidRoot {
        /// Description of the root problem.
        details: String,
    },

    /// A UD word has a deprel value that cannot produce a valid CHAT %gra relation.
    /// After uppercasing and colon→dash transform, the result must match `[A-Z][A-Z0-9\-]*`.
    #[error("Invalid deprel in UD parse: {details}")]
    InvalidDeprel {
        /// Description of the invalid deprel.
        details: String,
    },
}
