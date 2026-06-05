from main import ChatSummary


def build_plugin(settings=None):
    plugin = ChatSummary.__new__(ChatSummary)
    plugin.settings = settings or {}
    return plugin


def test_summary_time_instruction_is_enabled_by_default():
    plugin = build_plugin()

    instruction = plugin._with_summary_time_instruction("请总结重点。")

    assert "请总结重点。" in instruction
    assert "每条总结要点" in instruction
    assert "时间范围" in instruction
    assert "不要编造" in instruction


def test_summary_time_instruction_can_be_disabled():
    plugin = build_plugin({"summary_include_time": False})

    instruction = plugin._with_summary_time_instruction("请总结重点。")

    assert instruction == "请总结重点。"
