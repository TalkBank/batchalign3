//! CLI argument definitions using clap derive, mirroring `batchalign/cli/cli.py`.
//!
//! This module defines the complete argument tree for the `batchalign3` binary:
//!
//! - [`Cli`] -- top-level parser with global options and a subcommand.
//! - [`GlobalOpts`] -- flags that apply to every command (verbosity, server
//!   URL, cache bypass, worker count, etc.). Several fields are hidden BA2
//!   compatibility no-ops kept so that existing scripts do not break.
//! - [`Commands`] -- the subcommand enum (align, transcribe, morphotag, ...).
//! - Per-command arg structs ([`AlignArgs`], [`TranscribeArgs`], etc.) that
//!   embed [`CommonOpts`] for shared file I/O flags (input paths, output dir,
//!   file list, in-place mode).
//!
//! [`build_typed_options()`] converts the parsed args into a [`CommandOptions`]
//! enum variant for type-safe job submission, translating boolean flag pairs
//! (e.g. `--retokenize` / `--keeptokens`) into their canonical form.

mod commands;
mod global_opts;
mod options;

pub use commands::*;
pub use global_opts::GlobalOpts;
pub use options::*;

use clap::{Args, Parser, Subcommand};

/// batchalign3 — process .cha and/or audio files.
#[derive(Parser, Debug)]
#[command(name = "batchalign3", version, about)]
pub struct Cli {
    /// Global flags (verbosity, server URL, cache, etc.).
    #[command(flatten)]
    pub global: GlobalOpts,
    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Commands,
}

/// Shared options for file I/O across processing commands.
#[derive(Args, Debug, Clone)]
pub struct CommonOpts {
    /// Input paths (files and/or directories).
    pub paths: Vec<String>,

    /// Output directory. Omit for in-place modification.
    #[arg(short, long)]
    pub output: Option<String>,

    /// Read input file paths from a text file (one per line).
    #[arg(long)]
    pub file_list: Option<String>,

    /// Treat all paths as inputs and modify in-place.
    #[arg(long)]
    pub in_place: bool,

    /// Reference "before" file or directory for incremental processing.
    ///
    /// When provided, the diff engine compares each input file against
    /// its corresponding "before" version and only reprocesses changed
    /// utterances. Unchanged utterances preserve their existing dependent
    /// tiers (%mor, %gra, timing bullets).
    ///
    /// Supported commands: morphotag, align.
    #[arg(long, value_name = "PATH")]
    pub before: Option<String>,
}

/// Top-level command enum.
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Align transcripts against corresponding media files.
    Align(AlignArgs),
    /// Create a transcript from audio files.
    Transcribe(TranscribeArgs),
    /// Translate the transcript to English.
    Translate(TranslateArgs),
    /// Perform morphosyntactic analysis on transcripts.
    Morphotag(MorphotagArgs),
    /// Perform coreference analysis on transcripts.
    Coref(CorefArgs),
    /// Perform utterance segmentation.
    Utseg(UtsegArgs),
    /// Benchmark ASR word accuracy.
    Benchmark(BenchmarkArgs),
    /// Extract openSMILE audio features.
    Opensmile(OpensmileArgs),
    /// Compare transcripts against gold-standard references.
    Compare(CompareArgs),
    /// Calculate AVQI from paired .cs/.sv audio files.
    Avqi(AvqiArgs),
    /// Initialize ~/.batchalign.ini (ASR defaults / Rev.ai key).
    Setup(SetupArgs),
    /// Manage the processing server.
    Serve(ServeArgs),
    /// List or inspect remote jobs.
    Jobs(JobsArgs),
    /// View, export, or clear run logs.
    Logs(LogsArgs),
    /// Emit Rust-server OpenAPI schema.
    Openapi(OpenapiArgs),
    /// Print version info.
    Version,

    /// Manage the analysis and media caches.
    Cache(CacheArgs),
    /// Model training utilities (delegates to Python training runtime).
    Models(ModelsArgs),
    /// Benchmark command execution time across repeated runs.
    Bench(BenchArgs),
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

impl CommonOpts {
    /// Extract the command name, language, num_speakers, and file extensions
    /// that this command operates on. Returns `(command, lang, num_speakers, extensions)`.
    pub fn command_meta(cmd: &Commands) -> (&'static str, &str, u32, &'static [&'static str]) {
        match cmd {
            Commands::Align(_) => ("align", "eng", 1, &["cha"]),
            Commands::Transcribe(a) => {
                let diarize = if a.diarize {
                    true
                } else if a.nodiarize {
                    false
                } else {
                    a.diarization == DiarizationMode::Enabled
                };
                let cmd = if diarize {
                    "transcribe_s"
                } else {
                    "transcribe"
                };
                (cmd, &a.lang, a.num_speakers, &["mp3", "mp4", "wav"])
            }
            Commands::Translate(_) => ("translate", "eng", 1, &["cha"]),
            Commands::Morphotag(_) => ("morphotag", "eng", 1, &["cha"]),
            Commands::Coref(_) => ("coref", "eng", 1, &["cha"]),
            Commands::Compare(a) => ("compare", &a.lang, a.num_speakers, &["cha"]),
            Commands::Utseg(a) => ("utseg", &a.lang, a.num_speakers, &["cha"]),
            Commands::Benchmark(a) => {
                let cmd = "benchmark";
                (cmd, &a.lang, a.num_speakers, &["mp3", "mp4", "wav"])
            }
            Commands::Opensmile(a) => ("opensmile", &a.lang, 1, &["mp3", "mp4", "wav"]),
            Commands::Avqi(a) => ("avqi", &a.lang, 1, &["mp3", "mp4", "wav"]),
            _ => unreachable!("not a processing command"),
        }
    }
}

#[cfg(test)]
mod tests;
