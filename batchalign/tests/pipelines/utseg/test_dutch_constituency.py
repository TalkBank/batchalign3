"""RED/GREEN test: Dutch utseg must not crash on missing constituency model.

Source: Brian's bug report, 2026-03-28.
"""


def test_utseg_config_builder_skips_constituency_for_dutch():
    """The config builder for Dutch should NOT include constituency."""
    from batchalign.worker._stanza_loading import load_utseg_builder
    from batchalign.worker._types import _state

    load_utseg_builder("nld")
    assert _state.utseg_config_builder is not None

    lang_alpha2, configs = _state.utseg_config_builder(["nld"])
    assert "nl" in lang_alpha2
    nl_config = configs.get("nl", {})
    processors = nl_config.get("processors", "")

    assert "constituency" not in processors, (
        f"Dutch should NOT include constituency, got: {processors}"
    )


def test_utseg_config_builder_includes_constituency_for_english():
    """English SHOULD include constituency."""
    from batchalign.worker._stanza_loading import load_utseg_builder
    from batchalign.worker._types import _state

    load_utseg_builder("eng")
    assert _state.utseg_config_builder is not None

    lang_alpha2, configs = _state.utseg_config_builder(["eng"])
    en_config = configs.get("en", {})
    processors = en_config.get("processors", "")

    assert "constituency" in processors, (
        f"English SHOULD include constituency, got: {processors}"
    )
