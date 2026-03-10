"""Operation handlers for worker metadata and capability reporting."""

from __future__ import annotations

import logging
import os
import time
from batchalign.worker._types import (
    CapabilitiesResponse,
    HealthResponse,
    InferTask,
    _state,
)

L = logging.getLogger("batchalign.worker")

def _health() -> HealthResponse:
    """Health check with worker metadata."""
    is_ready = _state.ready or _state.test_echo
    return HealthResponse(
        status="ok" if is_ready else "loading",
        command=_state.command,
        lang=_state.lang,
        pid=os.getpid(),
        uptime_s=time.monotonic() - _state.started_at,
    )


def _capabilities() -> CapabilitiesResponse:
    """Report available commands and runtime info.

    Command advertisement is intentionally narrower than infer-task
    advertisement. Server-owned compositions such as ``transcribe`` are not
    exposed as Python commands; Rust synthesizes them from lower-level
    capability signals.
    """
    if _state.test_echo:
        from batchalign.runtime import Cmd2Task

        commands = sorted(set(Cmd2Task.keys()) | {"test-echo"})
        return CapabilitiesResponse(
            commands=commands,
            free_threaded=False,
            infer_tasks=[],
            engine_versions={},
        )

    from batchalign.runtime import is_free_threaded

    infer_tasks: list[InferTask] = []
    engine_versions: dict[InferTask, str] = {}

    # Infer task probes: map each InferTask to the imports required to prove
    # the system *can* run it.  The probe worker only loads morphotag models,
    # so we must NOT gate on loaded model state — otherwise FA, translate,
    # utseg, ASR, etc. are silently excluded from server capabilities.
    _INFER_TASK_PROBES: dict[InferTask, tuple[tuple[str, ...], str]] = {
        InferTask.MORPHOSYNTAX: (("stanza",), "stanza"),
        InferTask.UTSEG:        (("stanza",), "stanza"),
        InferTask.COREF:        (("stanza",), "stanza"),
        InferTask.TRANSLATE:    (("googletrans",), "googletrans-v1"),
        InferTask.FA:           (("torch", "torchaudio"), "whisper"),
        InferTask.OPENSMILE:    (("opensmile",), "opensmile"),
        InferTask.AVQI:         (("parselmouth", "torchaudio"), "praat"),
    }

    import importlib

    def _module_importable(module_name: str) -> bool:
        try:
            importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            return False
        return True

    for task, (deps, default_version) in _INFER_TASK_PROBES.items():
        importable = all(_module_importable(dep) for dep in deps)
        if importable:
            infer_tasks.append(task)
            # Use loaded model info when available, otherwise the default
            if task == InferTask.MORPHOSYNTAX or task == InferTask.COREF:
                engine_versions[task] = _state.stanza_version or default_version
            elif task == InferTask.UTSEG:
                engine_versions[task] = _state.utseg_version or default_version
            elif task == InferTask.FA:
                engine_versions[task] = _state.fa_model_name or _state.fa_engine.value
            elif task == InferTask.ASR:
                engine_versions[task] = _state.asr_engine.value
            elif task == InferTask.TRANSLATE:
                from batchalign.inference._domain_types import TranslationBackend
                if _state.translate_backend == TranslationBackend.GOOGLE:
                    engine_versions[task] = "googletrans-v1"
                else:
                    engine_versions[task] = default_version
            else:
                engine_versions[task] = default_version

    speaker_versions: list[str] = []
    if _module_importable("pyannote.audio"):
        speaker_versions.append("pyannote")
    if _module_importable("nemo.collections.asr"):
        speaker_versions.append("nemo")
    if speaker_versions:
        infer_tasks.append(InferTask.SPEAKER)
        engine_versions[InferTask.SPEAKER] = speaker_versions[0]

    # ASR is special now: the server can satisfy ASR through either
    # Python-hosted local engines (for example Whisper) or the Rust-owned
    # Rev.AI path when the control plane has already injected credentials.
    from batchalign.worker._model_loading.asr import resolve_injected_revai_api_key

    has_revai_key = bool(
        (_state.bootstrap and _state.bootstrap.revai_api_key)
        or resolve_injected_revai_api_key()
    )

    has_whisper = _module_importable("whisper")

    if has_whisper or has_revai_key:
        infer_tasks.append(InferTask.ASR)
        engine_versions[InferTask.ASR] = (
            _state.asr_engine.value
            if _state.ready and _state.asr_engine.value
            else "rev" if has_revai_key else "whisper"
        )

    return CapabilitiesResponse(
        commands=[],
        free_threaded=is_free_threaded(),
        infer_tasks=infer_tasks,
        engine_versions=engine_versions,
    )
