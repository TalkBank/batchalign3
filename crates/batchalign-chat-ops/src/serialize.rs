//! CHAT serialization wrapper.

use talkbank_model::WriteChat;
use talkbank_model::model::ChatFile;

/// Serialize a ChatFile back to CHAT text.
pub fn to_chat_string(chat_file: &ChatFile) -> String {
    chat_file.to_chat_string()
}
