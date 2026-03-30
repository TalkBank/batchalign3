#![warn(missing_docs)]
//! CHAT format operations for the batchalign processing pipeline.
//!
//! This crate contains all pure-Rust logic for manipulating CHAT files
//! (Codes for the Human Analysis of Transcripts) during NLP processing. It
//! was extracted from the PyO3 bridge (`batchalign-core`) so that both the
//! PyO3 layer and the standalone Rust server (`batchalign-server`) can share
//! the same CHAT manipulation code without duplication.
//!
//! # Design principle
//!
//! **No text hacking.** Every CHAT transformation goes through the typed
//! [`ChatFile`] AST from `talkbank-model`. This crate provides the
//! extract-modify-inject round-trip pattern that keeps CHAT serialization
//! correct even in the face of complex escaping, continuation lines,
//! multi-word tokens, and dependent tier alignment.
//!
//! # Parse, extract, modify, inject round-trip
//!
//! The fundamental workflow shared by all NLP tasks is:
//!
//! ```text
//!   CHAT text
//!       |  parse::parse_lenient()
//!       v
//!   ChatFile AST
//!       |  extract / collect_payloads()
//!       v
//!   NLP payloads (words, cache keys, positions)
//!       |  send to Python worker (batch_infer IPC)
//!       v
//!   NLP results (UdResponse, UtsegResponse, ...)
//!       |  inject / apply_*_results()
//!       v
//!   Modified ChatFile AST
//!       |  serialize::to_chat_string()
//!       v
//!   CHAT text (round-tripped)
//! ```
//!
//! Each NLP task module (morphosyntax, utseg, translate, coref, fa)
//! provides its own `collect_*_payloads()` and `apply_*_results()` functions
//! that follow this pattern.
//!
//! # Example: morphosyntax round-trip
//!
//! ```rust,no_run
//! use batchalign_chat_ops::parse::{TreeSitterParser, parse_lenient};
//! use batchalign_chat_ops::morphosyntax::{
//!     collect_payloads, clear_morphosyntax,
//! };
//! use batchalign_chat_ops::serialize::to_chat_string;
//! use batchalign_chat_ops::LanguageCode;
//! use batchalign_chat_ops::morphosyntax::MultilingualPolicy;
//!
//! // 1. Parse CHAT text into an AST
//! let parser = TreeSitterParser::new().unwrap();
//! let chat_text = std::fs::read_to_string("example.cha").unwrap();
//! let (mut chat_file, _errors) = parse_lenient(&parser, &chat_text);
//!
//! // 2. Clear any existing %mor/%gra tiers
//! clear_morphosyntax(&mut chat_file);
//!
//! // 3. Extract NLP payloads (words + positions) for the worker
//! let primary = LanguageCode::new("eng");
//! let declared = vec![primary.clone()];
//! let (payloads, _total) = collect_payloads(
//!     &chat_file, &primary, &declared, MultilingualPolicy::ProcessAll,
//! );
//! // payloads: Vec<BatchItemWithPosition>
//! // Each item contains: words, terminator, special_forms, lang
//!
//! // 4. Send payloads to Python worker via batch_infer IPC,
//! //    receive Vec<UdResponse> back...
//! //    (server orchestrator handles this step)
//!
//! // 5. Inject NLP results back into the AST
//! // inject_results(&mut chat_file, &payloads, &ud_responses, retokenize);
//!
//! // 6. Serialize back to CHAT
//! let output = to_chat_string(&chat_file);
//! ```
//!
//! # Module map
//!
//! ## Parsing and serialization
//!
//! | Module         | Responsibility                                                    |
//! |----------------|-------------------------------------------------------------------|
//! | [`parse`]      | Lenient and strict CHAT parsing wrappers over tree-sitter         |
//! | [`serialize`]  | CHAT serialization (AST back to `.cha` text)                      |
//!
//! ## Word extraction and injection
//!
//! | Module         | Responsibility                                                    |
//! |----------------|-------------------------------------------------------------------|
//! | [`extract`]    | Walk the AST to collect NLP-ready words per utterance (Mor, Wor, Pho, Sin domains) |
//! | [`inject`]     | Inject parsed `Mor`/`GraTier` structures into utterance dependent tiers |
//! | [`retokenize`] | Replace main-tier words with Stanza's UD tokenization (1:N splits, N:1 merges) |
//!
//! ## NLP task modules (server-side orchestration payloads)
//!
//! | Module         | Responsibility                                                    |
//! |----------------|-------------------------------------------------------------------|
//! | [`morphosyntax`]| Cache key, payload collection/injection for %mor/%gra tagging    |
//! | [`utseg`]      | Payload collection, cache key, result application for utterance segmentation |
//! | [`translate`]  | Payload collection, cache key, %xtra tier injection for translation |
//! | [`coref`]      | Document-level payload collection, sparse %xcoref injection       |
//! | [`fa`]         | Forced alignment: utterance grouping, DP alignment, timing injection, monotonicity |
//!
//! ## NLP support
//!
//! | Module         | Responsibility                                                    |
//! |----------------|-------------------------------------------------------------------|
//! | [`nlp`]        | UD-to-CHAT mapping, language-specific overrides (en, fr, ja), validation |
//! | [`dp_align`]   | Hirschberg divide-and-conquer sequence alignment (linear space)   |
//! | [`tokenizer_realign`] | Stanza tokenizer→CHAT word realignment + language-specific MWT patches |
//! | [`text_types`] | Provenance-encoding newtypes (`ChatRawText`, `ChatCleanedText`, `SpeakerCode`) |
//!
//! ## Evaluation
//!
//! | Module         | Responsibility                                                    |
//! |----------------|-------------------------------------------------------------------|
//! | [`wer_conform`]| Word normalization for WER benchmark comparison                   |
//! | [`benchmark`]  | Full WER computation: normalize, align, count errors, diff        |

pub mod asr_postprocess;
pub mod benchmark;
pub mod build_chat;
pub mod cache_key;
pub mod compare;
pub mod constituency;
pub mod coref;
pub mod diff;
pub mod dp_align;
pub mod extract;
pub mod fa;
pub mod indices;
pub mod inject;
pub mod merge_abbrev;
pub mod morphosyntax;
pub mod nlp;
pub mod parse;
pub mod retokenize;
pub mod serialize;
pub mod speaker;
pub mod text_types;
pub mod tokenizer_realign;
pub mod translate;
pub mod utseg;
pub mod utseg_compute;
pub mod validate;
pub mod wer_conform;

// Re-export newtypes used by all NLP task modules and the server orchestrators.
pub use cache_key::{CacheKey, CacheTaskName};

// Re-export talkbank_model types commonly needed by downstream crates
// (e.g. batchalign-server) that shouldn't depend on talkbank_model directly.
pub use talkbank_model::Span;
pub use talkbank_model::alignment::helpers::TierDomain;
pub use talkbank_model::header::Header;
pub use talkbank_model::model::BulletContent;
pub use talkbank_model::model::{
    ChatFile, DependentTier, LanguageCode, Line, UserDefinedDependentTier, Utterance,
};
