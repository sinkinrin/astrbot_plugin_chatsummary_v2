from astrbot.api.message_components import At, Plain, Reply
from main import ChatSummary


class DummyEvent:
    def __init__(self, outline: str = "", messages=None):
        self._outline = outline
        self._messages = list(messages or [])

    def get_message_outline(self):
        return self._outline

    def get_messages(self):
        return self._messages


def build_plugin(reminder_cfg: dict) -> ChatSummary:
    plugin = ChatSummary.__new__(ChatSummary)
    plugin.settings = {"important_message_reminder": reminder_cfg}
    return plugin


def test_is_related_to_me_matches_at_target_user():
    plugin = build_plugin(
        {
            "target_user_id": "1954984488",
            "mention_me": {"enabled": True, "aliases": []},
        }
    )
    event = DummyEvent(messages=[At(qq="1954984488")])

    matched, label = plugin._is_related_to_me(event)

    assert matched is True
    assert label == "@我"


def test_is_related_to_me_matches_reply_sender():
    plugin = build_plugin(
        {
            "target_user_id": "1954984488",
            "mention_me": {"enabled": True, "aliases": []},
        }
    )
    event = DummyEvent(messages=[Reply(id="1", sender_id="1954984488")])

    matched, label = plugin._is_related_to_me(event)

    assert matched is True
    assert label == "回复我的消息"


def test_is_related_to_me_matches_alias_case_insensitive():
    plugin = build_plugin(
        {
            "target_user_id": "1954984488",
            "mention_me": {"enabled": True, "aliases": ["XIII", "十三"]},
        }
    )
    event = DummyEvent(outline="今天找xiii确认一下")

    matched, label = plugin._is_related_to_me(event)

    assert matched is True
    assert label == "XIII"


def test_match_reminder_rules_returns_alias_first():
    plugin = build_plugin(
        {
            "rules": [
                {"pattern": r"bug|报错", "alias": "异常反馈"},
            ]
        }
    )
    event = DummyEvent(outline="这里有个报错，帮忙看看")

    matched, label = plugin._match_reminder_rules(event)

    assert matched is True
    assert label == "异常反馈"


def test_match_reminder_rules_falls_back_to_pattern_when_alias_empty():
    plugin = build_plugin(
        {
            "rules": [
                {"pattern": r"TODO|待办", "alias": ""},
            ]
        }
    )
    event = DummyEvent(outline="新增一个待办项")

    matched, label = plugin._match_reminder_rules(event)

    assert matched is True
    assert label == r"TODO|待办"


def test_is_interested_prioritizes_related_over_regex_rule():
    plugin = build_plugin(
        {
            "target_user_id": "1954984488",
            "mention_me": {"enabled": True, "aliases": ["XIII"]},
            "rules": [{"pattern": r"bug|报错", "alias": "异常反馈"}],
        }
    )
    event = DummyEvent(
        outline="XIII 这里有个报错",
        messages=[Plain("XIII 这里有个报错")],
    )

    matched, label = plugin._is_interested(event)

    assert matched is True
    assert label == "XIII"
