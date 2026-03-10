"""Type stubs for the batchalign_core Rust extension (pyo3/maturin).

This stub exposes type information for mypy to check Python code
that calls into the Rust backend.  Callback signatures are typed
precisely to match the Rust calling convention.

Callback Payload Types
----------------------
Each callback receives a Python dict constructed by Rust.  The TypedDict
definitions below document the exact keys Rust provides at the FFI boundary.

- ``MorphosyntaxPayload``: per-utterance morphosyntax (words only).
- ``MorphosyntaxBatchPayload``: batched morphosyntax (words + terminator + lang).
- ``TranslationPayload``: text + speaker for translation.
- ``FaPayload``: forced alignment input (words, IDs, audio range).
- ``UtsegPayload``: utterance segmentation (words + joined text).

Response Types
--------------
- ``FaIndexedTimingDict``: one word-level timing entry.
- ``FaIndexedResponse``: response with ``indexed_timings`` key.
- ``FaTokenDict``: one raw Whisper token.
- ``FaTokenResponse``: response with ``tokens`` key.
- ``TranslationResponse``: response with ``translation`` key.
- ``UtsegResponse``: response with ``assignments`` key.
"""

from collections.abc import Callable
from typing import TypedDict


# ---------------------------------------------------------------------------
# Callback payload TypedDicts (what Rust sends to Python)
# ---------------------------------------------------------------------------

class MorphosyntaxPayload(TypedDict):
    """Per-utterance morphosyntax payload (add_morphosyntax)."""
    words: list[str]


class MorphosyntaxBatchPayload(TypedDict):
    """One item in a batched morphosyntax payload (add_morphosyntax_batched)."""
    words: list[str]
    terminator: str
    lang: str


class TranslationPayload(TypedDict):
    """Translation callback payload."""
    text: str
    speaker: str


class FaPayload(TypedDict):
    """Forced alignment callback payload."""
    words: list[str]
    word_ids: list[str]
    word_utterance_indices: list[int]
    word_utterance_word_indices: list[int]
    audio_start_ms: int
    audio_end_ms: int
    pauses: bool


class UtsegPayload(TypedDict):
    """Utterance segmentation callback payload."""
    words: list[str]
    text: str


# ---------------------------------------------------------------------------
# Callback response TypedDicts (what Python returns to Rust)
# ---------------------------------------------------------------------------

class FaIndexedTimingDict(TypedDict, total=False):
    """One word-level timing result."""
    start_ms: int
    end_ms: int
    confidence: float | None


class FaIndexedResponse(TypedDict):
    """FA response: indexed word-level timings (Wave2Vec path)."""
    indexed_timings: list[FaIndexedTimingDict | None]


class FaTokenDict(TypedDict):
    """One raw Whisper token with timestamp."""
    text: str
    time_s: float


class FaTokenResponse(TypedDict):
    """FA response: raw tokens (Whisper path)."""
    tokens: list[FaTokenDict]


class TranslationResponseDict(TypedDict):
    """Translation callback response."""
    translation: str


class UtsegResponseDict(TypedDict):
    """Utterance segmentation callback response."""
    assignments: list[int]


# Morphosyntax responses are flexible dicts (Stanza raw_sentences or
# validated UD JSON).  The Rust side accepts either format via
# parse_morphosyntax_response, so we type the return as a plain dict.
MorphosyntaxResponseDict = dict[str, object]
"""Morphosyntax callback return: either ``{"raw_sentences": [...]}`` (Stanza raw)
or ``{"sentences": [{"words": [...]}]}`` (validated UD)."""


# ---------------------------------------------------------------------------
# Callback type aliases
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, int], None]
"""Progress callback: ``(current, total) -> None``."""

MorphosyntaxCallback = Callable[[MorphosyntaxPayload, str], MorphosyntaxResponseDict]
"""Per-utterance morphosyntax callback: ``(payload, lang) -> ud_response``."""

MorphosyntaxBatchCallback = Callable[
    [list[MorphosyntaxBatchPayload], str], list[MorphosyntaxResponseDict]
]
"""Batched morphosyntax callback: ``(payloads, lang) -> [ud_response, ...]``."""

TranslationCallback = Callable[[TranslationPayload], TranslationResponseDict]
"""Translation callback: ``(payload) -> {"translation": str}``."""

FaCallback = Callable[[FaPayload], FaIndexedResponse | FaTokenResponse]
"""Forced alignment callback: ``(payload) -> indexed_timings | tokens``."""

UtsegCallback = Callable[[UtsegPayload], UtsegResponseDict]
"""Utterance segmentation callback: ``(payload) -> {"assignments": [int]}``."""

UtsegBatchCallback = Callable[[list[UtsegPayload]], list[UtsegResponseDict]]
"""Batched utterance segmentation callback."""

ProviderBatchRunner = Callable[
    [str, str, list[dict[str, object]]],
    list[dict[str, object] | None],
]
"""Generic provider batch runner used by ``run_provider_pipeline``."""

JsonObject = dict[str, object]
"""Loose JSON-like mapping used for provider-boundary payloads."""


# ---------------------------------------------------------------------------
# Token alignment return type
# ---------------------------------------------------------------------------

AlignedToken = str | tuple[str, bool]
"""A token from ``align_tokens``: plain ``str`` (unchanged) or
``(merged_text, is_contraction)`` for multi-word tokens."""


class ParsedChat:
    """Opaque handle wrapping a Rust ``ChatFile`` AST."""

    # Construction --------------------------------------------------------
    @staticmethod
    def parse(chat_text: str) -> ParsedChat: ...
    @staticmethod
    def parse_lenient(chat_text: str) -> ParsedChat: ...
    @staticmethod
    def build(transcript_json: str) -> ParsedChat: ...

    # Serialization -------------------------------------------------------
    def serialize(self) -> str: ...
    def replace_inner(self, other: ParsedChat) -> None: ...

    # Validation -----------------------------------------------------------
    def validate(self) -> list[str]: ...
    def validate_structured(self) -> str: ...
    def validate_chat_structured(self) -> str: ...
    def parse_warnings(self) -> str: ...

    # Option flag queries ---------------------------------------------------
    def is_no_align(self) -> bool: ...

    # Metadata / extraction -----------------------------------------------
    def extract_languages(self) -> list[str]: ...
    def extract_metadata(self) -> str: ...
    def extract_nlp_words(self, domain: str) -> str: ...
    def strip_timing(self) -> None: ...
    def clear_morphosyntax(self) -> None: ...

    # PyO3 morphosyntax cache methods -------------------------------------
    def extract_morphosyntax_payloads(
        self, lang: str, *, skipmultilang: bool = False
    ) -> str: ...
    def inject_morphosyntax_from_cache(self, injections_json: str) -> None: ...
    def extract_morphosyntax_strings(self, line_indices_json: str) -> str: ...

    # Mutation via callbacks -----------------------------------------------
    def add_morphosyntax(
        self,
        lang: str,
        morphosyntax_fn: MorphosyntaxCallback,
        progress_fn: ProgressCallback | None = None,
        skipmultilang: bool = False,
        retokenize: bool = False,
    ) -> None: ...

    def add_morphosyntax_batched(
        self,
        lang: str,
        batch_fn: MorphosyntaxBatchCallback,
        progress_fn: ProgressCallback | None = None,
        skipmultilang: bool = False,
        retokenize: bool = False,
    ) -> None: ...

    def add_forced_alignment(
        self,
        fa_callback: FaCallback,
        progress_fn: ProgressCallback | None = None,
        pauses: bool = False,
        max_group_ms: int = 20000,
        total_audio_ms: int | None = None,
    ) -> None: ...

    def add_translation(
        self,
        translation_fn: TranslationCallback,
        progress_fn: ProgressCallback | None = None,
    ) -> None: ...

    def add_utterance_segmentation(
        self,
        segmentation_fn: UtsegCallback,
        progress_fn: ProgressCallback | None = None,
    ) -> None: ...

    def add_utterance_segmentation_batched(
        self,
        batch_fn: UtsegBatchCallback,
        progress_fn: ProgressCallback | None = None,
    ) -> None: ...

    def add_utterance_timing(self, asr_words_json: str) -> None: ...
    def add_retrace_markers(self, lang: str) -> None: ...
    def add_disfluency_markers(
        self, filled_pauses_json: str, replacements_json: str
    ) -> None: ...
    def add_comment(self, comment: str) -> None: ...
    def add_dependent_tiers(self, tiers_json: str) -> None: ...
    def reassign_speakers(self, segments_json: str, lang: str) -> None: ...

    def extract_document_structure(self) -> str:
        """Extract document structure as JSON for compatibility shim subscript access.

        Returns a JSON array of utterances with per-word morphology (%mor)
        and grammatical relations (%gra). See ``batchalign.compat`` for the
        Python wrapper classes that consume this output.
        """
        ...


# Module-level free functions (text-in / text-out) -------------------------

def add_morphosyntax(
    chat_text: str,
    lang: str,
    morphosyntax_fn: MorphosyntaxCallback,
    progress_fn: ProgressCallback | None = None,
    skipmultilang: bool = False,
    retokenize: bool = False,
) -> str: ...

def add_morphosyntax_batched(
    chat_text: str,
    lang: str,
    batch_fn: MorphosyntaxBatchCallback,
    progress_fn: ProgressCallback | None = None,
    skipmultilang: bool = False,
    retokenize: bool = False,
) -> str: ...

def run_provider_pipeline(
    chat_text: str,
    *,
    lang: str,
    provider_batch_fn: ProviderBatchRunner,
    operations: list[dict[str, object]],
    lenient: bool = False,
) -> str: ...

def unwrap_batch_infer_results(
    task: str,
    response: object,
) -> str: ...

def call_batch_infer_provider(
    task: str,
    lang: str,
    items: object,
    infer_fn: Callable[..., object],
) -> str: ...

def dispatch_protocol_message(
    message: object,
    *,
    health_fn: Callable[..., object],
    capabilities_fn: Callable[..., object],
    infer_fn: Callable[..., object],
    batch_infer_fn: Callable[..., object],
    execute_v2_fn: Callable[..., object],
    infer_request_model: object,
    batch_infer_request_model: object,
    execute_v2_request_model: object,
    validation_error_type: object,
) -> tuple[dict[str, object], bool]: ...

def find_worker_attachment_by_id(
    attachments: object,
    artifact_id: str,
) -> str: ...

def load_worker_json_attachment(
    attachments: object,
    artifact_id: str,
) -> str: ...

def load_worker_prepared_text_json(
    attachment: object,
) -> str: ...

def load_worker_prepared_audio_f32le_bytes(
    attachment: object,
) -> bytes: ...

def execute_asr_request_v2(
    request: object,
    local_whisper_runner: Callable[..., object] | None = None,
    hk_tencent_runner: Callable[..., object] | None = None,
    hk_aliyun_runner: Callable[..., object] | None = None,
    hk_funaudio_runner: Callable[..., object] | None = None,
) -> str: ...

def execute_forced_alignment_request_v2(
    request: object,
    whisper_runner: Callable[..., object] | None = None,
    wave2vec_runner: Callable[..., object] | None = None,
    canto_runner: Callable[..., object] | None = None,
) -> str: ...

def execute_opensmile_request_v2(
    request: object,
    prepared_audio_runner: Callable[..., object] | None = None,
) -> str: ...

def execute_avqi_request_v2(
    request: object,
    prepared_audio_runner: Callable[..., object] | None = None,
) -> str: ...

def execute_speaker_request_v2(
    request: object,
    pyannote_prepared_audio_runner: Callable[..., object] | None = None,
    nemo_prepared_audio_runner: Callable[..., object] | None = None,
) -> str: ...

def normalize_text_task_result(
    task: str,
    response: object,
    expected_count: int,
) -> str: ...

def add_forced_alignment(
    chat_text: str,
    fa_callback: FaCallback,
    progress_fn: ProgressCallback | None = None,
    pauses: bool = False,
    max_group_ms: int = 20000,
    total_audio_ms: int | None = None,
) -> str: ...

def add_translation(
    chat_text: str,
    translation_fn: TranslationCallback,
    progress_fn: ProgressCallback | None = None,
) -> str: ...

def add_utterance_segmentation(
    chat_text: str,
    segmentation_fn: UtsegCallback,
    progress_fn: ProgressCallback | None = None,
) -> str: ...

def add_utterance_segmentation_batched(
    chat_text: str,
    batch_fn: UtsegBatchCallback,
    progress_fn: ProgressCallback | None = None,
) -> str: ...

def add_utterance_timing(chat_text: str, asr_words_json: str) -> str: ...
def add_retrace_markers(chat_text: str, lang: str) -> str: ...
def add_disfluency_markers(
    chat_text: str, filled_pauses_json: str, replacements_json: str
) -> str: ...
def add_dependent_tiers(chat_text: str, tiers_json: str) -> str: ...
def reassign_speakers(chat_text: str, segments_json: str, lang: str) -> str: ...

def align_tokens(
    original_words: list[str], stanza_tokens: list[str], alpha2: str = ""
) -> list[AlignedToken]: ...
def build_chat(transcript_json: str) -> str: ...
def parse_and_serialize(chat_text: str) -> str: ...
def extract_metadata(chat_text: str) -> str: ...
def extract_nlp_words(chat_text: str, domain: str) -> str: ...
def extract_timed_tiers(chat_text: str, by_word: bool) -> str: ...
def strip_timing(chat_text: str) -> str: ...
def chat_terminators() -> list[str]: ...
def chat_mor_punct() -> list[str]: ...
def dp_align(
    payload: list[str], reference: list[str], case_insensitive: bool = False
) -> str: ...
def wer_conform(words: list[str]) -> list[str]: ...

def wer_compute(
    hypothesis: list[str],
    reference: list[str],
    langs: list[str] | None = None,
) -> str:
    """Compute WER. Returns JSON: {wer, total, matches, diff}."""
    ...

def wer_metrics(
    hypothesis: list[str],
    reference: list[str],
    langs: list[str] | None = None,
) -> str:
    """Compute structured WER metrics. Returns JSON: {wer, cer, accuracy, matches, total, error}."""
    ...

def clean_funaudio_segment_text(text: str) -> str:
    """Rust-owned FunASR segment cleanup helper used by HK ASR adapters."""
    ...

def funaudio_segments_to_asr(segments: object, lang: str) -> str:
    """Project raw FunASR output into JSON with ``monologues`` and ``timed_words``."""
    ...

def tencent_result_detail_to_asr(result_detail: object, lang: str) -> str:
    """Project Tencent ``ResultDetail`` objects into JSON with ``monologues`` and ``timed_words``."""
    ...

def aliyun_sentences_to_asr(sentences: object, lang: str) -> str:
    """Project Aliyun sentence results into JSON with ``monologues`` and ``timed_words``."""
    ...

def normalize_cantonese(text: str) -> str:
    """Normalize Cantonese text: simplified -> HK traditional + domain replacements."""
    ...

def cantonese_char_tokens(text: str) -> list[str]:
    """Normalize Cantonese text and split into per-character tokens (CJK punct stripped)."""
    ...

# Rev.AI native client ---------------------------------------------------

def rev_transcribe(
    audio_path: str,
    api_key: str,
    language: str,
    speakers_count: int | None = None,
    skip_postprocessing: bool = False,
    metadata: str | None = None,
) -> str: ...

def rev_get_timed_words(
    audio_path: str,
    api_key: str,
    language: str,
) -> str: ...

def rev_submit(
    audio_path: str,
    api_key: str,
    language: str,
    speakers_count: int | None = None,
    skip_postprocessing: bool = False,
    metadata: str | None = None,
) -> str: ...

def rev_poll(
    job_id: str,
    api_key: str,
    max_poll_secs: int = 30,
) -> str: ...

def rev_poll_timed_words(
    job_id: str,
    api_key: str,
    poll_secs: int = 15,
) -> str: ...

# CLI entry point (used by [project.scripts] console command) ----------------

def cli_main() -> None:
    """Run the batchalign3 CLI. Reads sys.argv for argument parsing."""
    ...
