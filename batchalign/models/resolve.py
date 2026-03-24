"""Resolve model shortcodes + language to HuggingFace IDs."""

from __future__ import annotations

from batchalign.inference._domain_types import LanguageCode

_RESOLVER: dict[str, dict[LanguageCode, str]] = {
    "utterance": {
        "eng": "talkbank/CHATUtterance-en",
        "zho": "talkbank/CHATUtterance-zh_CN",
        "yue": "PolyU-AngelChanLab/Cantonese-Utterance-Segmentation",
    },
}


def resolve(model_class: str, lang_code: LanguageCode) -> str | None:
    """Resolve one model family/language pair to a concrete model identifier."""
    return _RESOLVER.get(model_class, {}).get(lang_code)
