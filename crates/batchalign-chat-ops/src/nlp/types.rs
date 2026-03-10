//! Typed representation of UD/NLP responses consumed by Batchalign mapping logic.
//!
//! These types mirror the callback JSON contract used by Python pipeline engines,
//! but constrain ambiguous fields (`id`, `upos`, FA response variants) into Rust
//! enums before mapping starts.
//!
//! Keeping this layer strongly typed lets mapping code fail early with structured
//! errors instead of silently accepting malformed payloads.
//!
//! # Related CHAT Manual Sections
//!
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Format>
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Headers>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Main_Tier>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Dependent_Tiers>
//!
use serde::{Deserialize, Serialize};

/// Represents a raw word/token as returned by a Universal Dependencies (UD) engine like Stanza.
///
/// This structure mirrors the CoNLL-U / UD JSON format but enforces strict typing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UdWord {
    /// Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens;
    /// may be a decimal number for empty nodes.
    pub id: UdId,
    /// Word form or punctuation symbol.
    pub text: String,
    /// Lemma or stem of word form.
    pub lemma: String,
    /// Universal part-of-speech tag.
    pub upos: UdPunctable<UniversalPos>,
    /// Language-specific part-of-speech tag; underscore if not available.
    pub xpos: Option<String>,
    /// List of morphological features from the universal feature inventory or from a
    /// language-specific extension; underscore if not available.
    pub feats: Option<String>,
    /// Head of the current word, which is either a value of ID or zero (0).
    pub head: usize,
    /// Universal dependency relation to the HEAD (root iff HEAD = 0) or a
    /// language-specific subtype of one.
    pub deprel: String,
    /// Any other annotation.
    pub deps: Option<String>,
    /// Any other annotation.
    pub misc: Option<String>,
}

/// UD IDs can be single integers (1), ranges (1-2) for MWTs, or decimals (1.1).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum UdId {
    /// Regular word index (e.g., `1`).
    Single(usize),
    /// Multi-word token range (e.g., `1-2`).
    Range(usize, usize),
    /// Empty node index (e.g., `1.1`).
    Decimal(f64),
}

/// A wrapper to handle the fact that sometimes UD engines return punctuation
/// where a semantic tag is expected.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum UdPunctable<T> {
    /// A semantic value (non-punctuation).
    Value(T),
    /// Punctuation token (no semantic content).
    Punct(String),
}

/// The 17 Universal POS tags as defined by UD (Universal Dependencies) v2.
///
/// These tags form a coarse-grained, cross-linguistically consistent part-of-speech
/// tagset. Batchalign maps them to CHAT %mor categories via language-specific rules
/// in the `lang_en`, `lang_fr`, `lang_ja` modules. The mapping uses both `upos` and
/// `deprel` to disambiguate cases where a single UD tag covers multiple CHAT categories
/// (e.g., English `PART` maps to `inf` for infinitival "to" vs. `ptl` for verb particles).
///
/// Reference: <https://universaldependencies.org/u/pos/>
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "UPPERCASE")]
pub enum UniversalPos {
    /// Adjectives -- words that typically modify nouns and specify their properties
    /// or attributes (e.g., "big", "old", "green", "African").
    Adj,
    /// Adpositions -- prepositions and postpositions that express the grammatical
    /// relationship of a noun phrase to another word (e.g., "in", "to", "during").
    Adp,
    /// Adverbs -- words that typically modify verbs, adjectives, or other adverbs
    /// for time, place, manner, or degree (e.g., "very", "tomorrow", "here", "quickly").
    Adv,
    /// Auxiliary verbs -- function words that accompany the lexical verb to express
    /// tense, mood, aspect, voice, or evidentiality (e.g., "has" in "has done",
    /// "will", "should", "was" in passive).
    Aux,
    /// Coordinating conjunctions -- words that link units of equal syntactic status
    /// (e.g., "and", "or", "but").
    Cconj,
    /// Determiners -- words that modify nouns to express reference within context
    /// (e.g., "the", "a", "this", "every", "which").
    Det,
    /// Pronouns -- words that substitute for nouns or noun phrases whose referent
    /// is recoverable from context (e.g., "he", "herself", "who", "something").
    Pron,
    /// Common nouns -- words that denote classes of entities, as opposed to
    /// individual named entities (e.g., "girl", "cat", "tree", "idea").
    Noun,
    /// Proper nouns -- words that are the name of a specific individual, place, or
    /// object (e.g., "Mary", "London", "NATO"). Distinguished from common nouns
    /// by referring to unique entities rather than classes.
    Propn,
    /// Numerals -- words that express a number or a numerical concept, functioning
    /// as determiners, adjectives, or standalone (e.g., "two", "2", "first", "IV").
    Num,
    /// Particles -- function words that do not fit into other closed word classes.
    /// Language-dependent: in English covers infinitival "to" and verb particles
    /// like "up" in "give up"; in Japanese covers case markers and sentence-final
    /// particles.
    Part,
    /// Non-auxiliary verbs -- words that denote actions, events, or states and
    /// typically serve as the main predicate of a clause (e.g., "run", "eat",
    /// "think", "exist").
    Verb,
    /// Subordinating conjunctions -- words that mark a clause as subordinate to
    /// another clause (e.g., "that", "if", "while", "because", "although").
    Sconj,
    /// Punctuation -- non-alphabetical characters that delimit linguistic units
    /// (e.g., period `.`, comma `,`, question mark `?`). In CHAT mapping these
    /// are typically handled separately as terminators or separators.
    Punct,
    /// Symbols -- word-like entities that differ from ordinary words by form, function,
    /// or both (e.g., `$`, `%`, mathematical operators). Does not include punctuation.
    Sym,
    /// Interjections -- words used as exclamations or discourse fillers that
    /// typically do not enter into syntactic relations (e.g., "oh", "wow", "ugh",
    /// "hmm", "yes").
    Intj,
    /// Other -- a catch-all for words that cannot be assigned a real POS category.
    /// Covers foreign words in non-foreign-language text, typos, abbreviations that
    /// resist classification, etc. Should be used sparingly.
    X,
}

/// A single UD sentence: an ordered sequence of token records.
///
/// Corresponds to one utterance's worth of morphosyntactic analysis. The `words`
/// vector is 1-indexed by convention (matching CoNLL-U), but the Rust `Vec`
/// is 0-indexed -- `words[0]` has `id: Single(1)`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UdSentence {
    /// Ordered token records for this sentence. Includes regular words, MWT
    /// (multi-word token) expansions, and potentially empty nodes. The order
    /// matches the linear surface form of the utterance.
    pub words: Vec<UdWord>,
}

/// The top-level NLP callback response: one or more UD sentences.
///
/// In the morphosyntax pipeline, the Python callback (Stanza) returns one
/// `UdResponse` per utterance. Typically `sentences` contains exactly one
/// element, but multi-sentence utterances or Stanza's internal sentence
/// splitting can produce more. The mapping layer uses only the first sentence.
///
/// For batched morphosyntax, the wire format is `Vec<UdResponse>` -- one
/// response per utterance in the batch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UdResponse {
    /// One or more UD sentences produced by the NLP engine for a single
    /// utterance. The mapping layer in `mapping.rs` consumes `sentences[0]`.
    pub sentences: Vec<UdSentence>,
}

/// A raw token with its onset time, as returned by Whisper-style FA models.
///
/// Whisper produces token-level timestamps (one onset per sub-word token) rather
/// than word-level start/end pairs. The downstream DP aligner in `fa.rs`
/// reconstructs word boundaries by merging consecutive tokens and computing
/// durations from adjacent onsets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FaRawToken {
    /// The sub-word or word text fragment (e.g., " hello", " world").
    /// Leading whitespace is significant -- it indicates a word boundary in
    /// Whisper's byte-pair encoding.
    pub text: String,
    /// Onset time of this token in **seconds** (NOT milliseconds).
    /// Downstream code must convert to milliseconds (multiply by 1000) before
    /// injecting into CHAT timing bullets, which use integer milliseconds.
    pub time_s: f64,
}

/// Indexed timing produced when the callback already preserves word order.
///
/// This payload does not repeat word text; each entry corresponds to the same
/// index in the input `words` list supplied by Rust.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FaIndexedTiming {
    /// Start time in **milliseconds**.
    pub start_ms: u64,
    /// End time in **milliseconds**.
    pub end_ms: u64,
    /// Optional per-word confidence.
    pub confidence: Option<f64>,
}

/// Represents the raw data returned by a Forced Alignment "Passive Stub".
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum FaRawResponse {
    /// Indexed word-level timings aligned to callback input words by position.
    IndexedWordLevel {
        /// Per-index timing entries; `None` means no timing for that word.
        indexed_timings: Vec<Option<FaIndexedTiming>>,
    },
    /// Native Whisper format: list of (text, time)
    TokenLevel {
        /// Per-token BPE timing entries.
        tokens: Vec<FaRawToken>,
    },
}
