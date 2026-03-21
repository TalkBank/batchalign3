//! V2 execute dispatch helpers — task mapping and engine override resolution.
//!
//! These pure functions bridge the V2 execute protocol (typed per-backend
//! requests) to the pool's worker-key abstraction (profile + lang + engine
//! overrides). Extracted from `mod.rs` for browsability.

use crate::api::WorkerLanguage;
use crate::types::worker_v2::{
    AsrBackendV2, ExecuteRequestV2, FaBackendV2, InferenceTaskV2, TaskRequestV2,
};
use crate::worker::{InferTask, WorkerProfile};
use crate::worker::error::WorkerError;

/// Map a V2 inference task enum to the pool's infer-task vocabulary.
pub(super) fn infer_task_for_execute_v2(task: InferenceTaskV2) -> Result<InferTask, WorkerError> {
    match task {
        InferenceTaskV2::Morphosyntax => Ok(InferTask::Morphosyntax),
        InferenceTaskV2::Utseg => Ok(InferTask::Utseg),
        InferenceTaskV2::Translate => Ok(InferTask::Translate),
        InferenceTaskV2::Coref => Ok(InferTask::Coref),
        InferenceTaskV2::Asr => Ok(InferTask::Asr),
        InferenceTaskV2::ForcedAlignment => Ok(InferTask::Fa),
        InferenceTaskV2::Speaker => Ok(InferTask::Speaker),
        InferenceTaskV2::Opensmile => Ok(InferTask::Opensmile),
        InferenceTaskV2::Avqi => Ok(InferTask::Avqi),
    }
}

/// Derive the worker-pool key (profile, lang, engine overrides) for one V2
/// execute request.
pub(super) fn execute_v2_worker_key(
    lang: WorkerLanguage,
    request: &ExecuteRequestV2,
    default_engine_overrides: &str,
) -> Result<(WorkerProfile, WorkerLanguage, String), WorkerError> {
    let infer_task = infer_task_for_execute_v2(request.task)?;
    let engine_overrides =
        execute_v2_engine_overrides(request).unwrap_or_else(|| default_engine_overrides.to_owned());
    Ok((
        WorkerProfile::for_task(infer_task),
        lang.clone(),
        engine_overrides,
    ))
}

/// Extract backend-specific engine override JSON from a V2 execute request.
pub(super) fn execute_v2_engine_overrides(request: &ExecuteRequestV2) -> Option<String> {
    match &request.payload {
        TaskRequestV2::Asr(request) => asr_backend_override_name(request.backend)
            .map(|backend| format!(r#"{{"asr":"{backend}"}}"#)),
        TaskRequestV2::ForcedAlignment(request) => Some(format!(
            r#"{{"fa":"{}"}}"#,
            fa_backend_override_name(request.backend)
        )),
        _ => None,
    }
}

fn asr_backend_override_name(backend: AsrBackendV2) -> Option<&'static str> {
    match backend {
        AsrBackendV2::LocalWhisper => Some("whisper"),
        AsrBackendV2::HkTencent => Some("tencent"),
        AsrBackendV2::HkAliyun => Some("aliyun"),
        AsrBackendV2::HkFunaudio => Some("funaudio"),
        AsrBackendV2::Revai => None,
    }
}

fn fa_backend_override_name(backend: FaBackendV2) -> &'static str {
    match backend {
        FaBackendV2::Whisper => "whisper",
        FaBackendV2::Wave2vec => "wave2vec",
        FaBackendV2::Wav2vecCanto => "wav2vec_canto",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::LanguageCode3;
    use crate::types::worker_v2::{
        AsrInputV2, AsrRequestV2, FaTextModeV2, ForcedAlignmentRequestV2, PreparedAudioInputV2,
        WorkerArtifactIdV2, WorkerRequestIdV2,
    };

    fn request_with_payload(task: InferenceTaskV2, payload: TaskRequestV2) -> ExecuteRequestV2 {
        ExecuteRequestV2 {
            request_id: WorkerRequestIdV2::from("req-1"),
            task,
            payload,
            attachments: Vec::new(),
        }
    }

    #[test]
    fn maps_forced_alignment_execute_v2_to_fa_worker_profile() {
        assert_eq!(
            infer_task_for_execute_v2(InferenceTaskV2::ForcedAlignment).unwrap(),
            InferTask::Fa
        );
    }

    #[test]
    fn execute_v2_asr_worker_key_uses_request_backend_override() {
        let request = request_with_payload(
            InferenceTaskV2::Asr,
            TaskRequestV2::Asr(AsrRequestV2 {
                lang: WorkerLanguage::from(LanguageCode3::fra()),
                backend: AsrBackendV2::LocalWhisper,
                input: AsrInputV2::PreparedAudio(PreparedAudioInputV2 {
                    audio_ref_id: WorkerArtifactIdV2::from("audio-1"),
                }),
            }),
        );

        let key = execute_v2_worker_key(
            WorkerLanguage::from(LanguageCode3::fra()),
            &request,
            r#"{"asr":"tencent"}"#,
        )
        .unwrap();

        assert_eq!(key.0, WorkerProfile::for_task(InferTask::Asr));
        assert_eq!(key.1, WorkerLanguage::from(LanguageCode3::fra()));
        assert_eq!(key.2, r#"{"asr":"whisper"}"#);
    }

    #[test]
    fn execute_v2_fa_worker_key_uses_request_backend_override() {
        let request = request_with_payload(
            InferenceTaskV2::ForcedAlignment,
            TaskRequestV2::ForcedAlignment(ForcedAlignmentRequestV2 {
                backend: FaBackendV2::Wave2vec,
                payload_ref_id: WorkerArtifactIdV2::from("payload-1"),
                audio_ref_id: WorkerArtifactIdV2::from("audio-1"),
                text_mode: FaTextModeV2::SpaceJoined,
                pauses: false,
            }),
        );

        let key =
            execute_v2_worker_key(WorkerLanguage::from(LanguageCode3::eng()), &request, r#"{"fa":"whisper"}"#)
                .unwrap();

        assert_eq!(key.0, WorkerProfile::for_task(InferTask::Fa));
        assert_eq!(key.1, WorkerLanguage::from(LanguageCode3::eng()));
        assert_eq!(key.2, r#"{"fa":"wave2vec"}"#);
    }
}
