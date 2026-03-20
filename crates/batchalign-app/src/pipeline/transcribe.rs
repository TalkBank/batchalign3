//! Transcribe pipeline built on the internal stage runner.

use batchalign_chat_ops::asr_postprocess::{self, Utterance};
use batchalign_chat_ops::build_chat;
use batchalign_chat_ops::morphosyntax::{MultilingualPolicy, TokenizationMode};
use batchalign_chat_ops::serialize::to_chat_string;
use batchalign_chat_ops::speaker::{SpeakerSegment as ChatSpeakerSegment, reassign_speakers};
use std::path::Path;

use tracing::info;

use crate::api::{LanguageCode3, LanguageSpec, NumSpeakers};
use crate::error::ServerError;
use crate::params::{CachePolicy, MorphosyntaxParams};
use crate::pipeline::PipelineServices;
use crate::pipeline::plan::{PipelinePlan, StageFuture, StageId, StageSpec, run_plan};
use crate::runner::debug_dumper::DebugDumper;
use crate::runner::util::{FileStage, ProgressSender, ProgressUpdate};
use crate::transcribe::{
    AsrInferParams, AsrResponse, SpeakerInferParams, TranscribeOptions, build_empty_chat_text,
    convert_asr_response, generate_participant_ids, infer_asr, infer_speaker,
    insert_transcribe_comment, response_has_speaker_labels,
};
use crate::types::worker_v2::SpeakerSegmentV2;

/// Per-file transcribe pipeline state.
pub(crate) struct TranscribePipelineContext<'a> {
    /// Shared services for the run.
    pub services: PipelineServices<'a>,
    /// Immutable transcribe options.
    pub opts: &'a TranscribeOptions,
    /// Audio path being processed.
    pub audio_path: &'a Path,
    /// Raw ASR worker response.
    pub asr_response: Option<AsrResponse>,
    /// Postprocessed utterances.
    pub utterances: Option<Vec<Utterance>>,
    /// Dedicated diarization segments when Rust composes the speaker task.
    pub speaker_segments: Option<Vec<SpeakerSegmentV2>>,
    /// Current serialized CHAT text.
    pub chat_text: Option<String>,
    /// Debug artifact writer for offline replay.
    pub dumper: DebugDumper,
    /// Language resolved after ASR detection. When `opts.lang` is `Auto`,
    /// this is set by `stage_build_chat` to the ASR-detected language.
    /// Post-ASR stages (utseg, morphotag) use this for concrete dispatch.
    pub resolved_lang: Option<LanguageCode3>,
}

impl<'a> TranscribePipelineContext<'a> {
    fn new(
        audio_path: &'a Path,
        services: PipelineServices<'a>,
        opts: &'a TranscribeOptions,
        dumper: DebugDumper,
    ) -> Self {
        Self {
            services,
            opts,
            audio_path,
            asr_response: None,
            utterances: None,
            speaker_segments: None,
            chat_text: None,
            dumper,
            resolved_lang: None,
        }
    }

    /// Return the resolved language code for NLP stages (utseg, morphotag).
    ///
    /// After ASR, `resolved_lang` is populated from the ASR response's
    /// detected language. If `opts.lang` was already resolved (not Auto),
    /// it's used directly. Panics if called before resolution — this is a
    /// structural guarantee that the pipeline runs ASR before NLP.
    fn lang_for_nlp(&self) -> &LanguageCode3 {
        if let Some(ref resolved) = self.resolved_lang {
            return resolved;
        }
        match &self.opts.lang {
            LanguageSpec::Resolved(code) => code,
            LanguageSpec::Auto => panic!(
                "BUG: lang_for_nlp() called with unresolved Auto language. \
                 ASR must resolve the language before NLP stages run."
            ),
        }
    }
}

/// Run the transcribe pipeline for a single audio file.
pub(crate) async fn run_transcribe_pipeline(
    audio_path: &Path,
    services: PipelineServices<'_>,
    opts: &TranscribeOptions,
    progress: Option<&ProgressSender>,
    debug_dir: Option<&Path>,
) -> Result<String, ServerError> {
    let plan = transcribe_plan(opts.diarize, opts.with_utseg, opts.with_morphosyntax);
    let dumper = DebugDumper::new(debug_dir);
    let mut ctx = TranscribePipelineContext::new(audio_path, services, opts, dumper);

    // Build stage-level progress callback if a sender is provided.
    let on_stage = progress.map(|tx| {
        move |stage: StageId, done: usize, total: usize| {
            let _ = tx.send(ProgressUpdate::new(
                progress_stage_for_stage(stage),
                Some(done as i64),
                Some(total as i64),
            ));
        }
    });

    let on_stage_ref: Option<&(dyn Fn(StageId, usize, usize) + Send + Sync)> =
        on_stage.as_ref().map(|cb| cb as _);
    let _ = run_plan("transcribe", &plan, &mut ctx, on_stage_ref).await?;

    ctx.chat_text.ok_or_else(|| {
        ServerError::Validation("transcribe pipeline completed without output".to_string())
    })
}

/// Map transcribe-pipeline stage ids onto the shared file-progress stage
/// vocabulary.
///
/// This match is intentionally explicit. If the transcribe plan adds a new
/// stage, contributors should decide its operator-facing stage here rather
/// than silently falling back to a generic string.
fn progress_stage_for_stage(stage: StageId) -> FileStage {
    match stage {
        StageId::AsrInfer => FileStage::Transcribing,
        StageId::SpeakerDiarization => FileStage::PostProcessing,
        StageId::AsrPostprocess => FileStage::PostProcessing,
        StageId::BuildChat => FileStage::BuildingChat,
        StageId::OptionalUtseg => FileStage::SegmentingUtterances,
        StageId::OptionalMorphosyntax => FileStage::AnalyzingMorphosyntax,
        StageId::Serialize => FileStage::Finalizing,
        _ => unreachable!("transcribe plan emitted unsupported stage id {stage}"),
    }
}

fn transcribe_plan<'a>(
    diarize: bool,
    with_utseg: bool,
    with_morphosyntax: bool,
) -> PipelinePlan<TranscribePipelineContext<'a>> {
    let postprocess_dep = if diarize {
        StageId::SpeakerDiarization
    } else {
        StageId::AsrInfer
    };
    let mut stages = vec![
        StageSpec::new(StageId::AsrInfer, vec![], always_enabled, stage_asr_infer),
        StageSpec::new(
            StageId::SpeakerDiarization,
            vec![StageId::AsrInfer],
            diarization_requested,
            stage_speaker_diarization,
        ),
        StageSpec::new(
            StageId::AsrPostprocess,
            vec![postprocess_dep],
            always_enabled,
            stage_asr_postprocess,
        ),
        StageSpec::new(
            StageId::BuildChat,
            vec![StageId::AsrPostprocess],
            always_enabled,
            stage_build_chat,
        ),
    ];

    if with_utseg {
        stages.push(StageSpec::new(
            StageId::OptionalUtseg,
            vec![StageId::BuildChat],
            always_enabled,
            stage_run_utseg,
        ));
    }

    if with_morphosyntax {
        let dep = if with_utseg {
            StageId::OptionalUtseg
        } else {
            StageId::BuildChat
        };
        stages.push(StageSpec::new(
            StageId::OptionalMorphosyntax,
            vec![dep],
            always_enabled,
            stage_run_morphosyntax,
        ));
    }

    let final_dep = if with_morphosyntax {
        StageId::OptionalMorphosyntax
    } else if with_utseg {
        StageId::OptionalUtseg
    } else {
        StageId::BuildChat
    };
    stages.push(StageSpec::new(
        StageId::Serialize,
        vec![final_dep],
        always_enabled,
        stage_serialize,
    ));

    PipelinePlan::new(stages)
}

fn always_enabled(_: &TranscribePipelineContext<'_>) -> bool {
    true
}

fn diarization_requested(ctx: &TranscribePipelineContext<'_>) -> bool {
    ctx.opts.diarize
}

fn stage_asr_infer<'a, 'ctx>(ctx: &'a mut TranscribePipelineContext<'ctx>) -> StageFuture<'a> {
    Box::pin(async move {
        info!(
            audio_path = %ctx.audio_path.display(),
            lang = %ctx.opts.lang,
            num_speakers = ctx.opts.num_speakers,
            "Starting ASR inference"
        );

        let response = infer_asr(
            ctx.services.pool,
            &AsrInferParams {
                backend: ctx.opts.backend,
                audio_path: ctx.audio_path,
                lang: &ctx.opts.lang,
                num_speakers: NumSpeakers(ctx.opts.num_speakers as u32),
                rev_job_id: ctx.opts.rev_job_id.as_deref(),
            },
        )
        .await?;
        let filename = ctx
            .audio_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        ctx.dumper.dump_asr_response(filename, &response);
        ctx.asr_response = Some(response);
        Ok(())
    })
}

fn stage_asr_postprocess<'a, 'ctx>(
    ctx: &'a mut TranscribePipelineContext<'ctx>,
) -> StageFuture<'a> {
    Box::pin(async move {
        let response = ctx.asr_response.as_ref().ok_or_else(|| {
            ServerError::Validation("ASR response missing before post-processing".to_string())
        })?;

        if response.tokens.is_empty() {
            return Ok(());
        }

        let asr_output = convert_asr_response(response);
        info!(
            num_tokens = response.tokens.len(),
            num_monologues = asr_output.monologues.len(),
            "ASR response received, starting post-processing"
        );

        let lang_str = ctx.opts.lang.to_string();
        let utterances = asr_postprocess::process_raw_asr(&asr_output, &lang_str);
        info!(
            num_utterances = utterances.len(),
            "Post-processing complete, building CHAT"
        );

        ctx.utterances = Some(utterances);
        Ok(())
    })
}

fn stage_speaker_diarization<'a, 'ctx>(
    ctx: &'a mut TranscribePipelineContext<'ctx>,
) -> StageFuture<'a> {
    Box::pin(async move {
        let response = ctx.asr_response.as_ref().ok_or_else(|| {
            ServerError::Validation("ASR response missing before speaker diarization".to_string())
        })?;

        if response.tokens.is_empty() || response_has_speaker_labels(response) {
            return Ok(());
        }

        let Some(speaker_backend) = ctx.opts.speaker_backend else {
            return Ok(());
        };

        let lang = &ctx.opts.lang;
        info!(
            audio_path = %ctx.audio_path.display(),
            speaker_backend = ?speaker_backend,
            num_speakers = ctx.opts.num_speakers,
            "Running dedicated speaker diarization"
        );
        let segments = infer_speaker(
            ctx.services.pool,
            &SpeakerInferParams {
                audio_path: ctx.audio_path,
                lang,
                expected_speakers: NumSpeakers(ctx.opts.num_speakers as u32),
                backend: speaker_backend,
            },
        )
        .await?;
        info!(
            num_segments = segments.len(),
            "Speaker diarization complete"
        );
        ctx.speaker_segments = Some(segments);
        Ok(())
    })
}

fn stage_build_chat<'a, 'ctx>(ctx: &'a mut TranscribePipelineContext<'ctx>) -> StageFuture<'a> {
    Box::pin(async move {
        let response = ctx.asr_response.as_ref().ok_or_else(|| {
            ServerError::Validation("ASR response missing before CHAT build".to_string())
        })?;

        // Resolve Auto → ASR-detected language for CHAT headers and NLP.
        // When the user passed --lang auto, opts.lang is Auto. The ASR
        // response carries the engine's detected language code (e.g. "spa").
        // Store the resolved language so post-ASR stages (utseg, morphotag)
        // use the real language, not Auto.
        let resolved_lang = match &ctx.opts.lang {
            LanguageSpec::Auto => {
                let detected = response.lang.clone();
                // If the ASR engine echoed back "auto" (e.g. Whisper path),
                // use trigram detection on the full transcript text as fallback.
                if &*detected == "auto" || detected.is_empty() {
                    let all_text: String = response
                        .tokens
                        .iter()
                        .map(|t| t.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    let detected_iso3 =
                        batchalign_chat_ops::asr_postprocess::lang_detect::detect_primary_language(
                            &[&all_text],
                        )
                        .unwrap_or_else(|| "eng".to_string());
                    LanguageCode3::from(detected_iso3)
                } else {
                    detected
                }
            }
            LanguageSpec::Resolved(code) => code.clone(),
        };
        ctx.resolved_lang = Some(resolved_lang.clone());

        if response.tokens.is_empty() {
            // Build empty CHAT with resolved language.
            let mut opts_resolved = ctx.opts.clone();
            opts_resolved.lang = LanguageSpec::Resolved(resolved_lang.clone());
            ctx.chat_text = Some(build_empty_chat_text(&opts_resolved)?);
            return Ok(());
        }

        let utterances = ctx.utterances.as_mut().ok_or_else(|| {
            ServerError::Validation("Utterances missing before CHAT build".to_string())
        })?;

        // When auto-detecting language, run per-utterance language detection
        // for code-switching markup and multi-language headers.
        let is_auto = matches!(&ctx.opts.lang, LanguageSpec::Auto);
        let langs: Vec<String> = if is_auto {
            use batchalign_chat_ops::asr_postprocess::lang_detect;

            // Concatenate each utterance's words for language detection
            let utt_texts: Vec<String> = utterances
                .iter()
                .map(|utt| {
                    utt.words
                        .iter()
                        .map(|w| w.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect();
            let utt_text_refs: Vec<&str> = utt_texts.iter().map(String::as_str).collect();

            // Tag each utterance with its detected language
            for (utt, text) in utterances.iter_mut().zip(utt_text_refs.iter()) {
                utt.lang = lang_detect::detect_utterance_language(text);
            }

            // Collect all detected languages for @Languages header
            lang_detect::collect_detected_languages(&utt_text_refs, &resolved_lang)
        } else {
            vec![resolved_lang.to_string()]
        };

        let diarization_speaker_count = ctx
            .speaker_segments
            .as_deref()
            .map(unique_diarization_speaker_count)
            .unwrap_or(0);
        let participant_ids = generate_participant_ids(
            utterances,
            ctx.opts.num_speakers.max(diarization_speaker_count),
        );
        let desc = build_chat::transcript_from_asr_utterances(
            utterances,
            &participant_ids,
            &langs,
            ctx.opts.media_name.as_deref(),
            ctx.opts.write_wor,
        );

        let mut chat_file = build_chat::build_chat(&desc)
            .map_err(|e| ServerError::Validation(format!("Failed to build CHAT: {e}")))?;
        if let Some(segments) = ctx.speaker_segments.as_deref() {
            let diarization_segments: Vec<ChatSpeakerSegment> = segments
                .iter()
                .map(|segment| ChatSpeakerSegment {
                    start_ms: segment.start_ms.0,
                    end_ms: segment.end_ms.0,
                    speaker: segment.speaker.clone(),
                })
                .collect();
            reassign_speakers(
                &mut chat_file,
                &diarization_segments,
                &resolved_lang,
                &participant_ids,
            );
        }
        let chat_text = to_chat_string(&chat_file);
        let chat_text = insert_transcribe_comment(&chat_text, ctx.opts);
        let filename = ctx
            .audio_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        ctx.dumper.dump_post_asr_chat(filename, &chat_text);
        ctx.chat_text = Some(chat_text);
        Ok(())
    })
}

fn unique_diarization_speaker_count(segments: &[SpeakerSegmentV2]) -> usize {
    let mut seen: Vec<&str> = Vec::new();
    for segment in segments {
        if !seen.contains(&segment.speaker.as_str()) {
            seen.push(segment.speaker.as_str());
        }
    }
    seen.len()
}

fn stage_run_utseg<'a, 'ctx>(ctx: &'a mut TranscribePipelineContext<'ctx>) -> StageFuture<'a> {
    Box::pin(async move {
        info!("Running utterance segmentation");
        let input = ctx
            .chat_text
            .as_deref()
            .ok_or_else(|| ServerError::Validation("CHAT text missing before utseg".to_string()))?;
        let filename = ctx
            .audio_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        ctx.dumper.dump_pre_utseg_chat(filename, input);
        let utseg_lang = ctx.lang_for_nlp().clone();
        let result = crate::utseg::process_utseg(
            input,
            &utseg_lang,
            ctx.services.pool,
            ctx.services.cache,
            ctx.services.engine_version,
            CachePolicy::from(ctx.opts.override_cache),
        )
        .await?;
        ctx.dumper.dump_post_utseg_chat(filename, &result);
        ctx.chat_text = Some(result);
        Ok(())
    })
}

fn stage_run_morphosyntax<'a, 'ctx>(
    ctx: &'a mut TranscribePipelineContext<'ctx>,
) -> StageFuture<'a> {
    Box::pin(async move {
        info!("Running morphosyntax");
        let input = ctx.chat_text.as_deref().ok_or_else(|| {
            ServerError::Validation("CHAT text missing before morphosyntax".to_string())
        })?;
        let filename = ctx
            .audio_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        ctx.dumper.dump_pre_morphosyntax_chat(filename, input);
        let empty_mwt = std::collections::BTreeMap::new();
        let mor_lang = ctx.lang_for_nlp().clone();
        let mor_params = MorphosyntaxParams {
            lang: &mor_lang,
            tokenization_mode: TokenizationMode::Preserve,
            cache_policy: CachePolicy::from(ctx.opts.override_cache),
            multilingual_policy: MultilingualPolicy::ProcessAll,
            mwt: &empty_mwt,
        };
        ctx.chat_text = Some(
            crate::morphosyntax::process_morphosyntax(input, ctx.services, &mor_params).await?,
        );
        Ok(())
    })
}

fn stage_serialize<'a, 'ctx>(ctx: &'a mut TranscribePipelineContext<'ctx>) -> StageFuture<'a> {
    Box::pin(async move {
        if ctx.chat_text.is_none() {
            return Err(ServerError::Validation(
                "CHAT text missing before serialize".to_string(),
            ));
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{DurationSeconds, EngineVersion};
    use crate::cache::UtteranceCache;
    use crate::transcribe::{AsrBackend, AsrToken};
    use crate::types::worker_v2::SpeakerBackendV2;
    use crate::worker::pool::{PoolConfig, WorkerPool};

    #[test]
    fn transcribe_stage_progress_labels_are_stable() {
        assert_eq!(
            progress_stage_for_stage(StageId::AsrInfer),
            FileStage::Transcribing
        );
        assert_eq!(
            progress_stage_for_stage(StageId::SpeakerDiarization),
            FileStage::PostProcessing
        );
        assert_eq!(
            progress_stage_for_stage(StageId::AsrPostprocess),
            FileStage::PostProcessing
        );
        assert_eq!(
            progress_stage_for_stage(StageId::BuildChat),
            FileStage::BuildingChat
        );
        assert_eq!(
            progress_stage_for_stage(StageId::OptionalUtseg),
            FileStage::SegmentingUtterances
        );
        assert_eq!(
            progress_stage_for_stage(StageId::OptionalMorphosyntax),
            FileStage::AnalyzingMorphosyntax
        );
        assert_eq!(
            progress_stage_for_stage(StageId::Serialize),
            FileStage::Finalizing
        );
    }

    fn test_transcribe_options(speaker_backend: Option<SpeakerBackendV2>) -> TranscribeOptions {
        TranscribeOptions {
            backend: AsrBackend::RustRevAi,
            diarize: true,
            speaker_backend,
            lang: LanguageCode3::new("eng").into(),
            num_speakers: 2,
            with_utseg: false,
            with_morphosyntax: false,
            override_cache: false,
            write_wor: false,
            media_name: Some("sample".into()),
            rev_job_id: None,
        }
    }

    #[tokio::test]
    async fn speaker_diarization_stage_skips_when_asr_already_has_speaker_labels() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let pool = WorkerPool::new(PoolConfig::default());
        let engine_version = EngineVersion::from("test-asr");
        let services = PipelineServices {
            pool: &pool,
            cache: &cache,
            engine_version: &engine_version,
        };
        let audio_path = tempdir.path().join("sample.wav");
        let opts = test_transcribe_options(Some(SpeakerBackendV2::Pyannote));
        let mut ctx =
            TranscribePipelineContext::new(&audio_path, services, &opts, DebugDumper::disabled());
        ctx.asr_response = Some(AsrResponse {
            tokens: vec![AsrToken {
                text: "hello".into(),
                start_s: Some(DurationSeconds(0.0)),
                end_s: Some(DurationSeconds(0.5)),
                speaker: Some("SPEAKER_1".into()),
                confidence: None,
            }],
            lang: LanguageCode3::new("eng"),
        });

        stage_speaker_diarization(&mut ctx)
            .await
            .expect("speaker stage should succeed");

        assert!(
            ctx.speaker_segments.is_none(),
            "dedicated speaker inference should be skipped when ASR already carries speaker labels"
        );
    }

    #[tokio::test]
    async fn speaker_diarization_stage_skips_when_backend_is_unavailable() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let pool = WorkerPool::new(PoolConfig::default());
        let engine_version = EngineVersion::from("test-asr");
        let services = PipelineServices {
            pool: &pool,
            cache: &cache,
            engine_version: &engine_version,
        };
        let audio_path = tempdir.path().join("sample.wav");
        let opts = test_transcribe_options(None);
        let mut ctx =
            TranscribePipelineContext::new(&audio_path, services, &opts, DebugDumper::disabled());
        ctx.asr_response = Some(AsrResponse {
            tokens: vec![AsrToken {
                text: "hello".into(),
                start_s: Some(DurationSeconds(0.0)),
                end_s: Some(DurationSeconds(0.5)),
                speaker: None,
                confidence: None,
            }],
            lang: LanguageCode3::new("eng"),
        });

        stage_speaker_diarization(&mut ctx)
            .await
            .expect("speaker stage should succeed");

        assert!(
            ctx.speaker_segments.is_none(),
            "dedicated speaker inference should be skipped when no speaker backend is configured"
        );
    }

    /// When opts.lang is "auto", stage_build_chat must resolve to the
    /// ASR-detected language for CHAT headers (regression test for job
    /// 696870c7-02b where `@Languages: auto` leaked into output).
    #[tokio::test]
    async fn build_chat_stage_resolves_auto_to_detected_language() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let pool = WorkerPool::new(PoolConfig::default());
        let engine_version = EngineVersion::from("test-asr");
        let services = PipelineServices {
            pool: &pool,
            cache: &cache,
            engine_version: &engine_version,
        };
        let audio_path = tempdir.path().join("sample.wav");

        // Opts with lang="auto" — simulates --lang auto from CLI
        let mut opts = test_transcribe_options(None);
        opts.lang = LanguageSpec::Auto;
        opts.diarize = false;

        let mut ctx =
            TranscribePipelineContext::new(&audio_path, services, &opts, DebugDumper::disabled());

        // ASR response with detected language "spa"
        ctx.asr_response = Some(AsrResponse {
            tokens: vec![AsrToken {
                text: "hola".into(),
                start_s: Some(DurationSeconds(0.0)),
                end_s: Some(DurationSeconds(0.5)),
                speaker: None,
                confidence: None,
            }],
            lang: LanguageCode3::new("spa"),
        });

        // Run post-processing to generate utterances
        stage_asr_postprocess(&mut ctx).await.expect("postprocess");

        // Run build_chat — this should resolve "auto" → "spa"
        stage_build_chat(&mut ctx).await.expect("build_chat");

        let chat_text = ctx.chat_text.as_deref().expect("CHAT text should be set");

        // The @Languages header must contain the detected language, NOT "auto"
        let languages_line = chat_text
            .lines()
            .find(|l| l.starts_with("@Languages:"))
            .expect("@Languages header missing");
        assert!(
            languages_line.contains("spa"),
            "@Languages should contain detected 'spa', got: {languages_line}"
        );
        assert!(
            !languages_line.contains("auto"),
            "@Languages must NOT contain sentinel 'auto', got: {languages_line}"
        );
    }

    /// When opts.lang is "auto" and ASR returns empty tokens,
    /// build_chat should still resolve to the ASR response language.
    #[tokio::test]
    async fn build_chat_stage_resolves_auto_for_empty_response() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let pool = WorkerPool::new(PoolConfig::default());
        let engine_version = EngineVersion::from("test-asr");
        let services = PipelineServices {
            pool: &pool,
            cache: &cache,
            engine_version: &engine_version,
        };
        let audio_path = tempdir.path().join("sample.wav");

        let mut opts = test_transcribe_options(None);
        opts.lang = LanguageSpec::Auto;

        let mut ctx =
            TranscribePipelineContext::new(&audio_path, services, &opts, DebugDumper::disabled());
        ctx.asr_response = Some(AsrResponse {
            tokens: vec![],
            lang: LanguageCode3::new("fra"),
        });

        stage_build_chat(&mut ctx).await.expect("build_chat");

        let chat_text = ctx.chat_text.as_deref().expect("CHAT text should be set");
        let languages_line = chat_text
            .lines()
            .find(|l| l.starts_with("@Languages:"))
            .expect("@Languages header missing");
        assert!(
            languages_line.contains("fra"),
            "empty-response @Languages should contain 'fra', got: {languages_line}"
        );
        assert!(
            !languages_line.contains("auto"),
            "empty-response @Languages must NOT contain 'auto', got: {languages_line}"
        );
    }
}
