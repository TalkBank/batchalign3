//! NLP mapping/validation surface shared by language-specific morphosyntax adapters.
//!
//! This module is the typed boundary between external NLP engine output
//! (UD-like JSON from callbacks) and TalkBank-native `%mor`/`%gra` structures.

mod features;
pub mod lang_en;
pub mod lang_fr;
pub mod lang_ja;
pub mod mapping;
mod mor_word;
mod types;
pub mod validation;

pub use mapping::{MappingContext, map_ud_sentence};
pub use types::{
    FaIndexedTiming, FaRawResponse, FaRawToken, UdId, UdPunctable, UdResponse, UdSentence, UdWord,
    UniversalPos,
};
pub use validation::sanitize_mor_text;
