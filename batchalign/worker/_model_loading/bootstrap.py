"""Top-level worker bootstrap orchestration.

This module is the narrow control-plane entry for worker startup. Workers are
bootstrapped for one infer task at a time; Rust owns any higher-level command
composition.
"""

from __future__ import annotations

import json
import logging
import os

from batchalign.inference._domain_types import LanguageCode, NumSpeakers
from batchalign.worker._infer_hosts import (
    build_morphosyntax_batch_infer_handler,
    build_translate_batch_infer_handler,
    build_utseg_batch_infer_handler,
)
from batchalign.worker._model_loading.asr import load_asr_engine
from batchalign.worker._model_loading.forced_alignment import load_fa_engine
from batchalign.worker._model_loading.translation import load_translation_engine
from batchalign.worker._model_loading.utterance import load_utterance_model
from batchalign.worker._stanza_loading import load_stanza_models, load_utseg_builder
from batchalign.worker._types import (
    PROFILE_TASKS,
    InferTask,
    WorkerBootstrapRuntime,
    WorkerProfile,
    _state,
)

L = logging.getLogger("batchalign.worker")


def _configure_loaded_tasks(
    tasks: set[str],
    bootstrap: WorkerBootstrapRuntime,
    *,
    target_label: str,
) -> None:
    """Load one explicit task set and register the resulting handlers."""
    lang = bootstrap.lang
    num_speakers = bootstrap.num_speakers
    engine_overrides = bootstrap.engine_overrides or None
    L.info(
        "Loading models: target=%s lang=%s num_speakers=%d pid=%d",
        target_label,
        lang,
        num_speakers,
        os.getpid(),
    )
    _state.clear_batch_infer_handlers()
    if "morphosyntax" in tasks or "coref" in tasks:
        load_stanza_models(lang)
        _state.register_batch_infer_handler(
            InferTask.MORPHOSYNTAX,
            build_morphosyntax_batch_infer_handler(),
        )
    if "utterance" in tasks or "utseg" in tasks:
        load_utseg_builder(lang)
        load_utterance_model(lang)
        _state.register_batch_infer_handler(
            InferTask.UTSEG,
            build_utseg_batch_infer_handler(),
        )
    if "translate" in tasks:
        load_translation_engine(engine_overrides)
        _state.register_batch_infer_handler(
            InferTask.TRANSLATE,
            build_translate_batch_infer_handler(),
        )
    if "fa" in tasks:
        load_fa_engine(bootstrap)
    if "asr" in tasks:
        load_asr_engine(bootstrap)

    _state.command = target_label
    _state.lang = lang
    _state.num_speakers = num_speakers
    _state.ready = True
    L.info("Models ready: target=%s pid=%d", target_label, os.getpid())

def load_worker_profile(bootstrap: WorkerBootstrapRuntime) -> None:
    """Load the ML/runtime state for a profile-based worker.

    Profile-based workers load models for all tasks in the profile, enabling
    model sharing within a single process. GPU-only models (speaker, opensmile,
    avqi) use lazy loading on first request — only ASR/FA/Stanza/translation
    models are loaded eagerly here.
    """
    if bootstrap.profile is None:
        raise ValueError("worker bootstrap runtime requires a profile")

    profile = bootstrap.profile
    tasks = PROFILE_TASKS[profile]
    _state.bootstrap = bootstrap
    _configure_loaded_tasks(
        tasks,
        bootstrap,
        target_label=f"profile:{profile.value}",
    )


def load_worker_task(bootstrap: WorkerBootstrapRuntime) -> None:
    """Load the ML/runtime state needed for one pure infer-task worker."""
    if bootstrap.task is None:
        raise ValueError("worker bootstrap runtime requires a task")

    task = bootstrap.task
    task_name = task.value
    _state.bootstrap = bootstrap
    _configure_loaded_tasks(
        {task_name},
        bootstrap,
        target_label=f"infer:{task_name}",
    )


def enable_test_echo(target_label: str, lang: LanguageCode) -> None:
    """Configure the worker to echo requests without loading models."""
    _state.test_echo = True
    _state.clear_batch_infer_handlers()
    _state.command = target_label or "test-echo"
    _state.lang = lang
    _state.ready = True


def parse_engine_overrides(raw: str) -> dict[str, str] | None:
    """Parse the engine-override JSON payload from CLI args.

    The Rust caller serializes a ``BTreeMap<String, String>``; we validate
    the shape here at the IPC boundary.
    """
    if not raw:
        return None
    parsed: object = json.loads(raw)
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
    ):
        raise ValueError(
            f"engine overrides must be a flat {{str: str}} object, got: {raw!r}"
        )
    return parsed
