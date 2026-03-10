"""Worker model-loading package.

This package keeps worker bootstrap concerns split by responsibility:

- `bootstrap.py` owns infer-task startup orchestration
- `translation.py` owns translation-engine bootstrap
- `forced_alignment.py` owns FA-engine bootstrap
- `asr.py` owns ASR-engine bootstrap

The import surface stays stable so the worker entrypoint can still depend on
`batchalign.worker._model_loading` as one module-level boundary.
"""

from batchalign.worker._model_loading.asr import (
    load_asr_engine,
    resolve_injected_revai_api_key,
)
from batchalign.worker._model_loading.bootstrap import (
    enable_test_echo,
    load_worker_task,
    parse_engine_overrides,
)
from batchalign.worker._model_loading.forced_alignment import load_fa_engine
from batchalign.worker._model_loading.translation import load_translation_engine

__all__ = [
    "enable_test_echo",
    "load_asr_engine",
    "load_fa_engine",
    "load_translation_engine",
    "load_worker_task",
    "parse_engine_overrides",
    "resolve_injected_revai_api_key",
]
