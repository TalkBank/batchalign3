//! NLP word extraction from CHAT AST — delegates to `batchalign_chat_ops::extract`.

pub use batchalign_chat_ops::extract::{
    ExtractedUtterance, collect_utterance_content, extract_words,
};
