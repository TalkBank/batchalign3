//! Forced alignment dispatch and per-file FA pipeline.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::api::{
    ContentType, DurationMs, EngineVersion, DisplayPath, LanguageCode3, NumWorkers, RevAiJobId,
    UnixTimestamp,
};
use crate::cache::UtteranceCache;
use crate::options::{CommandOptions, EngineBackend as _};
use crate::params::{AudioContext, FaParams};
use crate::pipeline::PipelineServices;
use crate::runner::debug_dumper::DebugDumper;
use crate::scheduling::{FailureCategory, RetryPolicy, WorkUnitKind};
use crate::worker::pool::WorkerPool;
use tracing::{info, warn};

use crate::store::{JobStore, RunnerJobSnapshot, unix_now};

use super::super::util::{
    FileRunTracker, FileStage, FileTaskOutcome, apply_result_filename, classify_server_error,
    compute_audio_identity, drain_supervised_file_tasks, get_audio_duration_ms,
    is_retryable_worker_failure, resolve_audio_for_chat_with_media_dir, spawn_progress_forwarder,
    spawn_supervised_file_task, user_facing_error,
};
use super::FaDispatchPlan;
use super::infer_batched::apply_merge_abbrev;
use super::utr::{UtrPassContext, run_utr_pass};

/// Shared runtime dependencies for top-level FA dispatch.
///
/// The runner always passes this set together when it hands an `align` job to
/// the server-owned FA pipeline, so the bundle is the real boundary rather
/// than eight separate parameters.
pub(in crate::runner) struct FaDispatchRuntime {
    /// Worker pool used for typed V2 FA requests and any worker-owned UTR work.
    pub pool: Arc<WorkerPool>,
    /// Cache used by FA group reuse and worker result persistence.
    pub cache: Arc<UtteranceCache>,
    /// Current engine version string for cache partitioning.
    pub engine_version: EngineVersion,
    /// Optional preflight Rev.AI job ids keyed by original audio path.
    pub rev_job_ids: Arc<HashMap<PathBuf, RevAiJobId>>,
    /// Maximum number of file tasks to run concurrently for this job.
    pub num_workers: NumWorkers,
}

/// Shared per-file FA dependencies.
///
/// Grouping the job snapshot, services, and per-dispatch options here keeps
/// `process_one_fa_file` focused on the file lifecycle rather than repeating a
/// long parameter list for every call site.
struct FaFileContext<'a> {
    /// Immutable runner snapshot for the current job.
    job: &'a RunnerJobSnapshot,
    /// Store handle used for file lifecycle updates and result persistence.
    store: &'a Arc<JobStore>,
    /// Shared worker/cache services for FA and UTR.
    services: PipelineServices<'a>,
    /// Typed FA parameter bundle.
    fa_params: FaParams,
    /// Whether merge-abbrev should run before writing the result.
    should_merge_abbrev: bool,
    /// Optional before-file path for incremental align reruns.
    before_path: Option<&'a std::path::Path>,
    /// Optional UTR engine for the pre-pass and fallback paths.
    utr_engine: Option<&'a crate::options::UtrEngine>,
    /// Overlap strategy for `+<` utterances during UTR.
    utr_overlap_strategy: crate::options::UtrOverlapStrategy,
    /// Rev.AI preflight job ids keyed by original provider audio path.
    rev_job_ids: &'a HashMap<PathBuf, RevAiJobId>,
    /// Job language used for UTR and worker-side FA/ASR requests.
    lang: &'a LanguageCode3,
    /// Debug artifact writer for offline replay.
    dumper: DebugDumper,
    /// Custom media directory from `--media-dir`.
    media_dir: Option<&'a str>,
}

/// Dispatch FA (forced alignment) via the server-side infer path.
///
/// Unlike morphosyntax/utseg/translate/coref, FA is per-file: each file has
/// its own audio, so there is no cross-file batching. Files are processed
/// concurrently up to `num_workers` at a time, bounded by a semaphore.
/// Within each file, utterances are grouped into time windows and batched
/// to the worker.
pub(in crate::runner) async fn dispatch_fa_infer(
    job: &RunnerJobSnapshot,
    store: &Arc<JobStore>,
    runtime: FaDispatchRuntime,
    plan: FaDispatchPlan,
) {
    let job_id = &job.identity.job_id;
    let fallback_lang = LanguageCode3::eng();
    let job_lang = job.dispatch.lang.resolve_or(&fallback_lang);
    let fa_params = plan.options.fa_params;
    let should_merge_abbrev = plan.options.merge_abbrev.should_merge();
    let utr_engine = plan.options.utr_engine;
    let utr_overlap_strategy = plan.options.utr_overlap_strategy;

    // Read before_paths once for incremental FA
    let before_paths = job.filesystem.before_paths.clone();

    // Process files concurrently, bounded by available workers.
    let file_sem = Arc::new(tokio::sync::Semaphore::new(runtime.num_workers.0.max(1)));
    let mut tasks = Vec::new();

    for file in &job.pending_files {
        // Check cancellation before spawning
        if job.cancel_token.is_cancelled() {
            break;
        }

        let Ok(permit) = file_sem.clone().acquire_owned().await else { tracing::warn!("file semaphore closed during shutdown"); break; };
        let store = store.clone();
        let pool = runtime.pool.clone();
        let cache = runtime.cache.clone();
        let job = job.clone();
        let engine_version = runtime.engine_version.clone();
        let file = file.clone();
        let file_index = file.file_index;
        let before_path = if !before_paths.is_empty() && file_index < before_paths.len() {
            Some(before_paths[file_index].clone())
        } else {
            None
        };
        let utr_engine = utr_engine.clone();
        let job_lang = job_lang.clone();
        let rev_job_ids = runtime.rev_job_ids.clone();
        let filename = file.filename.clone();

        tasks.push(spawn_supervised_file_task(
            filename,
            "align file task",
            async move {
                let _permit = permit;
                let services = PipelineServices::new(&pool, &cache, &engine_version);
                let dumper = DebugDumper::new(
                    job.dispatch
                        .options
                        .common()
                        .debug_dir
                        .as_deref()
                        .map(std::path::Path::new),
                );
                let media_dir_str;
                let media_dir_ref = if let CommandOptions::Align(ref opts) = job.dispatch.options {
                    media_dir_str = opts.media_dir.clone();
                    media_dir_str.as_deref()
                } else {
                    None
                };
                let context = FaFileContext {
                    job: &job,
                    store: &store,
                    services,
                    fa_params,
                    should_merge_abbrev,
                    before_path: before_path.as_deref(),
                    utr_engine: utr_engine.as_ref(),
                    utr_overlap_strategy,
                    rev_job_ids: rev_job_ids.as_ref(),
                    lang: &job_lang,
                    dumper,
                    media_dir: media_dir_ref,
                };
                process_one_fa_file(&file, context).await
            },
        ));
    }

    let abnormal_exits = drain_supervised_file_tasks(store, job_id, &job.cancel_token, tasks).await;
    if abnormal_exits > 0 {
        warn!(
            job_id = %job_id,
            abnormal_exits,
            "Supervised align file tasks exited abnormally"
        );
    }
}

/// Process a single CHAT file through the server-side FA pipeline.
async fn process_one_fa_file(
    file: &crate::store::PendingJobFile,
    context: FaFileContext<'_>,
) -> FileTaskOutcome {
    let FaFileContext {
        job,
        store,
        services,
        fa_params,
        should_merge_abbrev,
        before_path,
        utr_engine,
        utr_overlap_strategy: _,
        rev_job_ids,
        lang,
        ref dumper,
        media_dir,
    } = context;
    let job_id = &job.identity.job_id;
    let correlation_id = &*job.identity.correlation_id;
    let file_index = file.file_index;
    let filename = file.filename.as_ref();
    let lifecycle = FileRunTracker::new(store, job_id, filename);
    let started_at = unix_now();

    lifecycle
        .begin_first_attempt(
            WorkUnitKind::FileForcedAlignment,
            started_at,
            FileStage::Reading,
        )
        .await;

    // Read the CHAT file
    let read_path: PathBuf =
        if job.filesystem.paths_mode && file_index < job.filesystem.source_paths.len() {
            job.filesystem.source_paths[file_index].clone()
        } else {
            job.filesystem.staging_dir.join("input").join(filename)
        };
    let paths_mode = job.filesystem.paths_mode;
    let output_paths = job.filesystem.output_paths.clone();
    let staging_dir = job.filesystem.staging_dir.clone();
    let media_mapping = job.filesystem.media_mapping.clone();
    let media_subdir = job.filesystem.media_subdir.clone();
    let source_dir = job.filesystem.source_dir.clone();

    let chat_text = match tokio::fs::read_to_string(&read_path).await {
        Ok(content) => content,
        Err(e) => {
            let err_msg = format!("Failed to read input: {e}");
            lifecycle
                .fail(&err_msg, FailureCategory::InputMissing, unix_now())
                .await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };
    lifecycle.stage(FileStage::ResolvingAudio).await;

    // Resolve audio path.
    // In paths_mode the audio sits alongside the .cha source file.
    // In content mode (remote --server) we try multiple strategies:
    //   1. Media mapping (explicit server-side path mapping)
    //   2. Source directory (client's original path — works when server shares filesystem)
    //   3. Media roots (server-configured search directories)
    //   4. Alongside staged file (last resort)
    let original_audio_path = if paths_mode {
        resolve_audio_for_chat_with_media_dir(&read_path, media_dir.map(Path::new)).await
    } else {
        let mut found: Option<PathBuf> = None;

        // Strategy 1: Media mapping
        if found.is_none() && !media_mapping.is_empty() {
            let mapping_root = store.config().media_mappings.get(&media_mapping).cloned();
            if let Some(root) = mapping_root {
                let file_path = Path::new(filename);
                let stem = file_path.file_stem().unwrap_or_default().to_string_lossy();
                let file_parent = file_path
                    .parent()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                let full_subdir = if file_parent.is_empty() {
                    media_subdir.clone()
                } else if media_subdir.is_empty() {
                    file_parent
                } else {
                    format!("{media_subdir}/{file_parent}")
                };
                let search_dir = PathBuf::from(&root).join(&full_subdir);
                for ext in crate::runner::util::KNOWN_MEDIA_EXTENSIONS {
                    let candidate = search_dir.join(format!("{stem}.{ext}"));
                    if tokio::fs::try_exists(&candidate).await.unwrap_or(false) {
                        found = Some(candidate);
                        break;
                    }
                }
            }
        }

        // Strategy 2: Source directory — the client's original input path.
        // Works when the server shares the same filesystem (e.g. fleet machines
        // with the same /Users/macw/ layout).
        if found.is_none() && !source_dir.as_os_str().is_empty() {
            let source_path = source_dir.join(Path::new(filename).file_name().unwrap_or_default());
            let source_audio =
                resolve_audio_for_chat_with_media_dir(&source_path, media_dir.map(Path::new)).await;
            if source_audio.is_some() {
                info!(
                    filename,
                    source_dir = %source_dir.display(),
                    "Resolved audio via client source directory"
                );
                found = source_audio;
            }
        }

        // Strategy 3: Media roots from server config
        if found.is_none() && !store.config().media_roots.is_empty() {
            let file_stem = Path::new(filename)
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy();
            'roots: for root in &store.config().media_roots {
                for ext in crate::runner::util::KNOWN_MEDIA_EXTENSIONS {
                    let candidate = Path::new(root).join(format!("{file_stem}.{ext}"));
                    if tokio::fs::try_exists(&candidate).await.unwrap_or(false) {
                        found = Some(candidate);
                        break 'roots;
                    }
                }
            }
        }

        // Strategy 4: Alongside staged file (last resort)
        if found.is_none() {
            found =
                resolve_audio_for_chat_with_media_dir(&read_path, media_dir.map(Path::new)).await;
        }

        found
    };
    let original_audio_path = match original_audio_path {
        Some(p) => p,
        None => {
            let err_msg = format!(
                "Cannot find audio file for {filename}. \
                 Searched for media with known extensions (.wav, .mp3, .mp4, etc.) \
                 {}.",
                if !media_mapping.is_empty() {
                    format!("in media mapping '{media_mapping}' subdir '{media_subdir}'")
                } else {
                    "alongside the .cha file".to_string()
                }
            );
            lifecycle
                .fail(&err_msg, FailureCategory::Validation, unix_now())
                .await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };

    let rev_job_id = rev_job_ids.get(&original_audio_path).map(|id| &**id);

    // Convert non-WAV media (e.g. mp4) to WAV via ffmpeg if needed.
    // soundfile (Python) cannot read container formats like mp4 directly.
    let audio_path = match crate::ensure_wav::ensure_wav(&original_audio_path, None).await {
        Ok(p) => p,
        Err(e) => {
            let err_msg = format!("Media conversion failed for {filename}: {e}");
            lifecycle
                .fail(&err_msg, FailureCategory::Validation, unix_now())
                .await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };

    // Compute audio identity for cache keying: path|mtime|size
    let audio_path_str = audio_path.to_string_lossy();
    let audio_identity = compute_audio_identity(&audio_path_str)
        .await
        .unwrap_or_else(|| {
            // Fallback: use path with zeroed metadata
            batchalign_chat_ops::fa::AudioIdentity::from_metadata(&audio_path_str, 0, 0)
        });

    // Get total audio duration via ffprobe (optional -- for untimed utterance estimation)
    let total_audio_ms = get_audio_duration_ms(&audio_path_str).await;
    let utr_audio_path = if utr_engine.as_ref().is_some_and(|e| e.is_rust_owned()) {
        original_audio_path.as_path()
    } else {
        audio_path.as_path()
    };

    // UTR pre-pass: if untimed utterances exist and a UTR engine is configured,
    // run ASR to recover utterance-level timing before FA grouping.
    let (mut chat_text, mut had_unrecovered_untimed) = {
        let fa_parser = batchalign_chat_ops::parse::TreeSitterParser::new()
            .expect("tree-sitter CHAT grammar must load");
        let (chat_file, _) = batchalign_chat_ops::parse::parse_lenient(&fa_parser, &chat_text);
        let (_timed, untimed) = batchalign_chat_ops::fa::count_utterance_timing(&chat_file);

        if untimed == 0 {
            info!(filename, _timed, "All utterances timed, skipping UTR");
            (chat_text, false)
        } else if let Some(utr_engine) = utr_engine {
            lifecycle.stage(FileStage::RecoveringUtteranceTiming).await;

            // Create a progress forwarder so partial-window UTR can report
            // per-window progress (e.g. "Recovering utterance timing 2/5").
            let utr_progress =
                spawn_progress_forwarder(store.clone(), job_id.clone(), filename.to_string());

            match run_utr_pass(
                &chat_text,
                UtrPassContext {
                    audio_path: utr_audio_path,
                    lang,
                    services,
                    audio_identity: &audio_identity,
                    cache_policy: fa_params.cache_policy,
                    total_audio_ms: total_audio_ms.map(DurationMs),
                    max_group_ms: Some(fa_params.max_group_ms),
                    filename,
                    engine: utr_engine,
                    overlap_strategy: context.utr_overlap_strategy,
                    rev_job_id,
                    dumper,
                },
                Some(&utr_progress),
            )
            .await
            {
                Ok((updated_text, utr_result)) => {
                    let still_untimed = utr_result.unmatched > 0;
                    (updated_text, still_untimed)
                }
                Err(original_text) => (original_text, true),
            }
        } else {
            warn!(
                filename,
                untimed,
                "Untimed utterances detected but no UTR engine configured, using interpolation"
            );
            (chat_text, true)
        }
    };

    let mut utr_fallback_attempted = false;
    let retry_policy = RetryPolicy::default();

    for attempt_number in 1..=retry_policy.max_attempts {
        if attempt_number > 1 {
            lifecycle
                .restart_attempt(
                    WorkUnitKind::FileForcedAlignment,
                    unix_now(),
                    FileStage::Aligning,
                )
                .await;
        } else {
            lifecycle.stage(FileStage::Aligning).await;
        }

        // Create a progress forwarder so the FA orchestrator can report
        // per-group progress back to the store.
        let progress_tx =
            spawn_progress_forwarder(store.clone(), job_id.clone(), filename.to_string());

        // Read "before" text for incremental FA if available
        let before_text = if let Some(bp) = before_path {
            tokio::fs::read_to_string(bp).await.ok()
        } else {
            None
        };

        let audio = AudioContext {
            audio_path: &audio_path,
            audio_identity: &audio_identity,
            total_audio_ms: total_audio_ms.map(DurationMs),
        };

        let fa_result = if let Some(ref bt) = before_text {
            crate::fa::process_fa_incremental(
                bt,
                &chat_text,
                &audio,
                lang,
                services,
                &fa_params,
                Some(&progress_tx),
            )
            .await
        } else {
            crate::fa::process_fa(
                &chat_text,
                &audio,
                lang,
                services,
                &fa_params,
                Some(&progress_tx),
            )
            .await
        };

        match fa_result {
            Ok(fa_result) => {
                // Store FA trace if debug_traces is enabled for this job
                let debug_traces = job.dispatch.debug_traces;
                let output_text = if debug_traces {
                    let output_text = fa_result.chat_text.clone();
                    let file_traces = crate::types::traces::FileTraces {
                        filename: DisplayPath::from(filename),
                        dp_alignments: Vec::new(),
                        asr_pipeline: None,
                        fa_timeline: Some(fa_result.into_timeline_trace()),
                        retokenizations: Vec::new(),
                    };
                    store
                        .trace_store()
                        .upsert_file(job_id, file_index, file_traces)
                        .await;
                    output_text
                } else {
                    fa_result.chat_text
                };
                dumper.dump_fa_output(filename, &output_text);

                lifecycle.stage(FileStage::Writing).await;
                let finished_at = unix_now();

                // Optionally merge abbreviations before writing
                let output_text = if should_merge_abbrev {
                    apply_merge_abbrev(&output_text)
                } else {
                    output_text
                };

                // Write output
                let write_path = if paths_mode && file_index < output_paths.len() {
                    apply_result_filename(&output_paths[file_index], filename)
                } else {
                    staging_dir.join("output").join(filename)
                };

                if let Some(parent) = write_path.parent() {
                    let _ = tokio::fs::create_dir_all(parent).await;
                }

                if let Err(e) = tokio::fs::write(&write_path, &output_text).await {
                    warn!(
                        job_id = %job_id,
                        correlation_id = %correlation_id,
                        filename = %filename,
                        error = %e,
                        "Failed to write FA output"
                    );
                }

                lifecycle
                    .complete_with_result(DisplayPath::from(filename), ContentType::Chat, finished_at)
                    .await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
            Err(e) => {
                let finished_at = unix_now();
                let category = classify_server_error(&e);
                let raw_msg = format!("FA processing failed: {e}");
                // Log the raw system error for developers; show users a
                // helpful message instead of "Broken pipe (os error 32)".
                warn!(
                    job_id = %job_id,
                    filename,
                    category = %category,
                    raw_error = %raw_msg,
                    "FA error (raw)"
                );
                let err_msg = user_facing_error(category, "Alignment", filename, &raw_msg);
                let has_retry_budget = attempt_number < retry_policy.max_attempts;

                if matches!(&e, crate::error::ServerError::Worker(_))
                    && is_retryable_worker_failure(category)
                    && has_retry_budget
                {
                    // Fallback UTR: if FA failed and we have unrecovered untimed
                    // utterances, attempt UTR before the next retry.
                    if had_unrecovered_untimed
                        && !utr_fallback_attempted
                        && let Some(utr_engine) = utr_engine
                    {
                        utr_fallback_attempted = true;
                        info!(
                            filename,
                            "FA failed with untimed utterances; attempting fallback UTR"
                        );
                        lifecycle.stage(FileStage::RecoveringTimingFallback).await;

                        if let Ok((updated_text, utr_result)) = run_utr_pass(
                            &chat_text,
                            UtrPassContext {
                                audio_path: utr_audio_path,
                                lang,
                                services,
                                audio_identity: &audio_identity,
                                cache_policy: fa_params.cache_policy,
                                total_audio_ms: total_audio_ms.map(DurationMs),
                                max_group_ms: Some(fa_params.max_group_ms),
                                filename,
                                engine: utr_engine,
                                overlap_strategy: context.utr_overlap_strategy,
                                rev_job_id,
                                dumper,
                            },
                            None,
                        )
                        .await
                            && utr_result.injected > 0
                        {
                            chat_text = updated_text;
                            had_unrecovered_untimed = false;
                            info!(
                                filename,
                                injected = utr_result.injected,
                                "Fallback UTR recovered timing"
                            );
                        }
                    }

                    let backoff_ms = retry_policy.backoff_for_retry(attempt_number);
                    let retry_at = UnixTimestamp(finished_at.0 + (backoff_ms.0 as f64 / 1000.0));
                    lifecycle
                        .retry(
                            retry_at,
                            category,
                            &format!("{err_msg}; retrying in {backoff_ms} ms"),
                            finished_at,
                        )
                        .await;
                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms.0)).await;
                    continue;
                }

                lifecycle.fail(&err_msg, category, finished_at).await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
        }
    }

    FileTaskOutcome::MissingTerminalState
}
