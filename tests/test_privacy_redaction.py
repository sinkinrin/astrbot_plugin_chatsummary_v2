from main import ChatSummary


def build_plugin(settings=None):
    plugin = ChatSummary.__new__(ChatSummary)
    plugin.settings = settings or {}
    return plugin


def test_prepare_chat_text_redacts_sensitive_info_by_default():
    plugin = build_plugin()

    text = plugin._prepare_chat_text_for_llm(
        "link https://example.com?a=1 email test@example.com phone 13800138000 key sk-abc1234567890",
        topic=None,
        max_chars=0,
    )

    assert "[URL]" in text
    assert "[EMAIL]" in text
    assert "[PHONE]" in text
    assert "[SECRET]" in text


def test_prepare_chat_text_can_disable_sensitive_info_redaction():
    plugin = build_plugin({"privacy": {"redact_sensitive_info": False}})

    raw = "link https://example.com?a=1 email test@example.com phone 13800138000 key sk-abc1234567890"
    text = plugin._prepare_chat_text_for_llm(raw, topic=None, max_chars=0)

    assert text == raw
