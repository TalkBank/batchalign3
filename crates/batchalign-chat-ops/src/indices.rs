//! Domain-specific index newtypes to prevent index confusion.
//!
//! These newtypes wrap `usize` to statically distinguish between different
//! kinds of indices that are frequently passed together (e.g., utterance
//! indices vs word indices within an utterance).

/// Index of an utterance in a CHAT file (among utterances only, 0-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UtteranceIdx(pub usize);

impl UtteranceIdx {
    /// Raw index value.
    pub fn raw(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for UtteranceIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Index into the CHAT word array (after extraction), 0-based within an utterance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WordIdx(pub usize);

impl WordIdx {
    /// Raw index value.
    pub fn raw(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for WordIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
