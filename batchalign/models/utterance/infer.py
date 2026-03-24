"""Runtime inference for BA2-style utterance boundary models."""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch
from transformers import AutoTokenizer, BertForTokenClassification

from batchalign.inference._domain_types import LanguageCode
from batchalign.models.resolve import resolve
from batchalign.models.utterance.dataset import BOUNDARIES

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
_STRIP_PUNCT_RE = re.compile(r"[.?!,]")


def resolve_utterance_model(lang: LanguageCode) -> str | None:
    """Resolve the BA2 utterance model id for one language."""
    return resolve("utterance", lang)


def _normalize_utterance_word_mapping(
    words: Sequence[str],
) -> tuple[list[str], list[int]]:
    """Normalize ASR words and keep the original-index mapping."""
    normalized: list[str] = []
    original_indices: list[int] = []
    for original_index, word in enumerate(words):
        lowered = word.lower().strip()
        if not lowered:
            continue
        cleaned = _STRIP_PUNCT_RE.sub("", lowered)
        if cleaned:
            normalized.append(cleaned)
            original_indices.append(original_index)
    return normalized, original_indices


def normalize_utterance_words(words: Sequence[str]) -> list[str]:
    """Normalize ASR words for BA2-style utterance model inference."""
    normalized, _ = _normalize_utterance_word_mapping(words)
    return normalized


class BertUtteranceModel:
    """Typed BA2-style utterance boundary classifier.

    The model predicts one action per input word. BA3 consumes only the
    utterance-boundary decisions as typed group assignments; punctuation
    reconstruction remains a Rust concern.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BertForTokenClassification.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    def predict_actions(self, words: Sequence[str]) -> list[int]:
        """Predict BA2-style token actions for one pretokenized word sequence."""
        normalized_words, original_indices = _normalize_utterance_word_mapping(words)
        if len(normalized_words) <= 1:
            return [0] * len(words)

        tokenized = self.tokenizer(
            [normalized_words],
            return_tensors="pt",
            is_split_into_words=True,
        ).to(DEVICE)
        logits = self.model(**tokenized).logits
        classified_targets = torch.argmax(logits, dim=2).cpu()

        raw_actions: list[int] = [0] * len(normalized_words)
        previous_word_idx: int | None = None
        for token_idx, word_idx in enumerate(tokenized.word_ids(0)):
            if word_idx is None or word_idx == previous_word_idx:
                continue
            previous_word_idx = word_idx
            raw_actions[word_idx] = int(classified_targets[0][token_idx])

        actions = raw_actions[:]
        for word_idx, action in enumerate(raw_actions[:-1]):
            if action > 0 and raw_actions[word_idx + 1] > 0:
                actions[word_idx] = 0

        expanded_actions = [0] * len(words)
        for normalized_index, original_index in enumerate(original_indices):
            expanded_actions[original_index] = actions[normalized_index]
        return expanded_actions

    def predict_assignments(self, words: Sequence[str]) -> list[int]:
        """Predict typed utterance-group assignments for one word sequence."""
        if len(words) <= 1:
            return [0] * len(words)
        actions = self.predict_actions(words)
        assignments: list[int] = []
        current_group = 0
        for action in actions:
            assignments.append(current_group)
            if action in BOUNDARIES:
                current_group += 1
        return assignments
