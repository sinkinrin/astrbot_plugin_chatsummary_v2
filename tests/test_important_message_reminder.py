import asyncio
from types import SimpleNamespace

from astrbot.api.message_components import At, Plain, Reply
from main import ChatSummary


class DummyContext:
    def __init__(self):
        self.calls = []

    async def send_message(self, target_umo, message):
        self.calls.append((target_umo, message))


class DummyEvent:
    def __init__(
        self,
        outline: str | None = "",
        messages=None,
        *,
        group_id: str = "123456",
        sender_id: str = "10001",
        sender_name: str = "Tester",
        timestamp: int = 1710000000,
    ):
        self._outline = outline
        self._messages = list(messages or [])
        self._group_id = group_id
        self._sender_id = sender_id
        self._sender_name = sender_name
        self.message_obj = SimpleNamespace(timestamp=timestamp)

    def get_message_outline(self):
        return self._outline

    def get_messages(self):
        return self._messages

    def get_group_id(self):
        return self._group_id

    def get_sender_id(self):
        return self._sender_id

    def get_sender_name(self):
        return self._sender_name


def build_plugin(reminder_cfg: dict | None) -> ChatSummary:
    plugin = ChatSummary.__new__(ChatSummary)
    plugin.settings = {"important_message_reminder": reminder_cfg}
    plugin.context = DummyContext()
    return plugin


def prepare_handler_plugin(plugin: ChatSummary):
    plugin.sent_plain_messages = []

    async def fake_send_plain_message_to_user(user_id, text):
        plugin.sent_plain_messages.append((user_id, text))
        return True

    async def fake_safe_group_info(client, group_id):
        return {"group_name": f"group-{group_id}"}

    plugin._send_plain_message_to_user = fake_send_plain_message_to_user
    plugin._safe_group_info = fake_safe_group_info
    plugin._get_aiocqhttp_client = lambda: object()
    plugin._reload_settings = lambda: plugin.settings
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


def test_is_related_to_me_handles_none_config_safely():
    plugin = build_plugin(None)
    event = DummyEvent(messages=[At(qq="1954984488")])

    matched, label = plugin._is_related_to_me(event)

    assert matched is False
    assert label is None


def test_match_reminder_rules_handles_none_config_safely():
    plugin = build_plugin(None)
    event = DummyEvent(outline="这里有个报错，帮忙看看")

    matched, label = plugin._match_reminder_rules(event)

    assert matched is False
    assert label is None


def test_is_related_to_me_requires_target_user_id_for_at_or_reply_matching():
    plugin = build_plugin(
        {
            "target_user_id": "",
            "mention_me": {"enabled": True, "aliases": []},
        }
    )
    event = DummyEvent(messages=[At(qq="")])

    matched, label = plugin._is_related_to_me(event)

    assert matched is False
    assert label is None


def test_handle_group_message_for_reminder_skips_when_watch_groups_empty():
    plugin = prepare_handler_plugin(
        build_plugin(
            {
                "enabled": True,
                "watch_groups": [],
                "target_user_id": "1954984488",
                "rules": [{"pattern": r"bug|报错", "alias": "异常反馈"}],
                "mention_me": {"enabled": False, "aliases": []},
            }
        )
    )
    event = DummyEvent(outline="这里有个报错，帮忙看看")

    asyncio.run(plugin.handle_group_message_for_reminder(event))

    assert plugin.sent_plain_messages == []
    assert plugin.context.calls == []


def test_handle_group_message_for_reminder_reloads_settings_before_matching():
    plugin = prepare_handler_plugin(build_plugin({"enabled": False}))
    refreshed_cfg = {
        "enabled": True,
        "watch_groups": ["123456"],
        "target_user_id": "1954984488",
        "rules": [{"pattern": r"bug|报错", "alias": "异常反馈"}],
        "mention_me": {"enabled": False, "aliases": []},
    }

    def fake_reload_settings():
        plugin.settings["important_message_reminder"] = refreshed_cfg
        return plugin.settings

    plugin._reload_settings = fake_reload_settings
    event = DummyEvent(outline="这里有个报错，帮忙看看")

    asyncio.run(plugin.handle_group_message_for_reminder(event))

    assert plugin.sent_plain_messages
    assert plugin.context.calls


def test_handle_group_message_for_reminder_handles_none_outline_after_at_match():
    plugin = prepare_handler_plugin(
        build_plugin(
            {
                "enabled": True,
                "watch_groups": ["123456"],
                "target_user_id": "1954984488",
                "rules": [],
                "mention_me": {"enabled": True, "aliases": []},
            }
        )
    )
    event = DummyEvent(outline=None, messages=[At(qq="1954984488")])

    asyncio.run(plugin.handle_group_message_for_reminder(event))

    assert plugin.sent_plain_messages
    assert plugin.context.calls
