import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from main import CachedMessage, ChatSummary


class FakeMessageCache:
    def __init__(self):
        self.messages = []
        self.state = {}

    def add_message(self, message):
        self.messages.append(message)
        return True

    def get_messages_after(self, group_id, *, after_timestamp, limit):
        return self.get_messages_after_cursor(
            group_id,
            after_timestamp=after_timestamp,
            after_message_id="",
            limit=limit,
        )

    def get_messages_after_cursor(self, group_id, *, after_timestamp, after_message_id, limit):
        return [
            {
                "group_id": str(group_id),
                "message_id": "m2",
                "user_id": "10002",
                "nickname": "Bob",
                "time": datetime.fromtimestamp(200),
                "timestamp": 200,
                "text": "cached new message",
                "raw_type_summary": "text",
            }
        ]

    def get_summary_state(self, group_id):
        return self.state.get(str(group_id))


class ExplodingMessageCache:
    def add_message(self, message):
        raise AssertionError("cache should not be written")

    def get_messages_after_cursor(self, *args, **kwargs):
        raise AssertionError("cache should not be read")

    def get_summary_state(self, *args, **kwargs):
        raise AssertionError("summary state should not be read")

    def set_summary_state(self, *args, **kwargs):
        raise AssertionError("summary state should not be written")

    def prune_before(self, *args, **kwargs):
        raise AssertionError("cache should not be pruned")


class DummyGroupEvent:
    def __init__(
        self,
        outline="hello from target group",
        *,
        group_id="868381808",
        sender_id="10001",
        sender_name="Alice",
        message_id="msg-1",
        timestamp=1710000000,
        self_id=None,
    ):
        self._outline = outline
        self._group_id = group_id
        self._sender_id = sender_id
        self._sender_name = sender_name
        self.message_obj = SimpleNamespace(message_id=message_id, timestamp=timestamp, self_id=self_id)

    def get_message_outline(self):
        return self._outline

    def get_group_id(self):
        return self._group_id

    def get_sender_id(self):
        return self._sender_id

    def get_sender_name(self):
        return self._sender_name

    def get_messages(self):
        return []


def build_plugin(auto_cfg=None):
    plugin = ChatSummary.__new__(ChatSummary)
    plugin.settings = {
        "auto_summary": auto_cfg
        or {
            "enabled": True,
            "target_groups": ["868381808"],
            "cache_enabled": True,
        }
    }
    plugin.wake_prefix = ["/"]
    plugin._message_cache = FakeMessageCache()
    plugin._reload_settings = lambda: plugin.settings
    return plugin


def test_group_message_handler_writes_target_group_message_to_cache():
    plugin = build_plugin()
    event = DummyGroupEvent()

    asyncio.run(plugin.handle_group_message_for_summary_cache(event))

    assert len(plugin._message_cache.messages) == 1
    cached = plugin._message_cache.messages[0]
    assert isinstance(cached, CachedMessage)
    assert cached.group_id == "868381808"
    assert cached.message_id == "msg-1"
    assert cached.text == "hello from target group"


def test_group_message_handler_ignores_groups_outside_auto_summary_targets():
    plugin = build_plugin()
    event = DummyGroupEvent(group_id="999")

    asyncio.run(plugin.handle_group_message_for_summary_cache(event))

    assert plugin._message_cache.messages == []


def test_group_message_handler_ignores_self_messages():
    plugin = build_plugin()
    event = DummyGroupEvent(sender_id="10001", self_id="10001")

    asyncio.run(plugin.handle_group_message_for_summary_cache(event))

    assert plugin._message_cache.messages == []


def test_group_message_handler_caches_when_auto_summary_disabled_but_cache_enabled():
    """解耦：关闭自动总结(enabled=False)但开启缓存时，目标群消息仍应写入 SQLite。"""
    plugin = build_plugin(
        {
            "enabled": False,
            "target_groups": ["868381808"],
            "cache_enabled": True,
        }
    )
    event = DummyGroupEvent()

    asyncio.run(plugin.handle_group_message_for_summary_cache(event))

    assert len(plugin._message_cache.messages) == 1
    assert plugin._message_cache.messages[0].text == "hello from target group"


def test_group_message_handler_does_not_cache_when_cache_disabled():
    """cache_enabled=False 时不写缓存，即使自动总结开启。"""
    plugin = build_plugin(
        {
            "enabled": True,
            "target_groups": ["868381808"],
            "cache_enabled": False,
        }
    )
    event = DummyGroupEvent()

    asyncio.run(plugin.handle_group_message_for_summary_cache(event))

    assert plugin._message_cache.messages == []


def test_collect_auto_summary_messages_reads_local_cache_before_history_api():
    plugin = build_plugin()
    plugin._message_cache.state["868381808"] = {
        "last_summarized_ts": 100,
        "last_summarized_message_id": "m1",
        "last_summary_hash": "old",
    }

    async def fail_if_history_is_called(*args, **kwargs):
        raise AssertionError("history API should not be used when local cache has messages")

    plugin._collect_group_messages = fail_if_history_is_called

    chat_text, structured = asyncio.run(
        plugin._collect_auto_summary_messages(
            object(),
            "868381808",
            max_records=200,
            backfill_count=50,
        )
    )

    assert "cached new message" in chat_text
    assert len(structured) == 1
    assert structured[0]["message_id"] == "m2"


def test_collect_auto_summary_messages_uses_history_when_cache_disabled():
    plugin = build_plugin()
    calls = []

    async def fake_history(client, group_id, *, count, expand_forwards=True):
        calls.append({"count": count, "expand_forwards": expand_forwards})
        return "history text", [
            {
                "time": datetime.fromtimestamp(400),
                "user_id": "9",
                "nickname": "History",
                "text": "history text",
            }
        ]

    plugin._collect_group_messages = fake_history

    chat_text, structured = asyncio.run(
        plugin._collect_auto_summary_messages(
            object(),
            "868381808",
            max_records=200,
            backfill_count=50,
            cache_enabled=False,
        )
    )

    assert chat_text == "history text"
    assert structured[0]["nickname"] == "History"
    assert calls == [{"count": 50, "expand_forwards": False}]


def test_collect_auto_summary_messages_caps_history_backfill_to_max_records():
    plugin = build_plugin()
    plugin._message_cache.state["868381808"] = {
        "last_summarized_ts": 100,
        "last_summarized_message_id": "m1",
        "last_summary_hash": "old",
    }

    def empty_cache(*args, **kwargs):
        return []

    calls = []

    async def fake_history(client, group_id, *, count, expand_forwards=True):
        calls.append({"count": count, "expand_forwards": expand_forwards})
        return "", []

    plugin._message_cache.get_messages_after_cursor = empty_cache
    plugin._collect_group_messages = fake_history

    asyncio.run(
        plugin._collect_auto_summary_messages(
            object(),
            "868381808",
            max_records=20,
            backfill_count=500,
        )
    )

    assert calls == [{"count": 20, "expand_forwards": False}]


def test_execute_auto_summary_does_not_touch_cache_when_cache_disabled(tmp_path):
    plugin = build_plugin(
        {
            "enabled": True,
            "target_groups": ["868381808"],
            "cache_enabled": False,
            "history_backfill_count": 50,
            "min_messages": 1,
            "broadcast": False,
            "summary_mode": "message_count",
            "message_chunk_size": 30,
            "time_window_minutes": 15,
        }
    )
    plugin._message_cache = ExplodingMessageCache()
    plugin._get_aiocqhttp_client = lambda: object()
    plugin._persist_summary_file = lambda **kwargs: Path(tmp_path / "summary.md")

    async def fake_history(client, group_id, *, count, expand_forwards=True):
        return "history text", [
            {
                "time": datetime.fromtimestamp(400),
                "user_id": "9",
                "nickname": "History",
                "text": "history text",
            }
        ]

    async def fake_summarize_text(*args, **kwargs):
        return "summary"

    async def fake_group_info(*args, **kwargs):
        return {"group_name": "group"}

    plugin._collect_group_messages = fake_history
    plugin._summarize_text = fake_summarize_text
    plugin._safe_group_info = fake_group_info

    asyncio.run(
        plugin._execute_auto_summary(
            plugin.settings["auto_summary"],
            {
                "limits": {
                    "max_chat_records": 200,
                    "max_tokens": 1000,
                    "max_input_chars": 2000,
                },
                "auto_summary": plugin.settings["auto_summary"],
                "render_as_image": False,
            },
        )
    )


def test_cached_messages_after_cursor_are_not_filtered_out_by_same_timestamp():
    plugin = build_plugin()
    timestamp = 1710000300
    structured = [
        {
            "message_id": "m2",
            "time": datetime.fromtimestamp(timestamp),
            "timestamp": timestamp,
            "user_id": "1",
            "nickname": "Alice",
            "text": "same second but newer id",
        }
    ]

    new_messages = plugin._filter_new_auto_summary_messages(
        structured,
        last_summary_time=datetime.fromtimestamp(timestamp),
        last_summary_message_id="m1",
    )

    assert new_messages == structured


def test_cached_messages_with_empty_last_message_id_keep_same_timestamp_messages():
    plugin = build_plugin()
    timestamp = 1710000300
    structured = [
        {
            "message_id": "10",
            "time": datetime.fromtimestamp(timestamp),
            "timestamp": timestamp,
            "user_id": "1",
            "nickname": "Alice",
            "text": "same second after history fallback",
        }
    ]

    new_messages = plugin._filter_new_auto_summary_messages(
        structured,
        last_summary_time=datetime.fromtimestamp(timestamp),
        last_summary_message_id="",
    )

    assert new_messages == structured


def test_history_backfill_messages_are_filtered_by_persistent_cursor():
    plugin = build_plugin()
    timestamp = 1710000300
    structured = [
        {
            "message_id": "9",
            "time": datetime.fromtimestamp(timestamp),
            "timestamp": timestamp,
            "user_id": "1",
            "nickname": "Old",
            "text": "already summarized",
        },
        {
            "message_id": "11",
            "time": datetime.fromtimestamp(timestamp),
            "timestamp": timestamp,
            "user_id": "2",
            "nickname": "New",
            "text": "new same-second message",
        },
    ]

    new_messages = plugin._filter_new_auto_summary_messages(
        structured,
        last_summary_time=datetime.fromtimestamp(timestamp),
        last_summary_message_id="10",
    )

    assert [msg["message_id"] for msg in new_messages] == ["11"]


def test_execute_auto_summary_does_not_advance_state_when_llm_fails(tmp_path):
    plugin = build_plugin(
        {
            "enabled": True,
            "target_groups": ["868381808"],
            "cache_enabled": True,
            "history_backfill_count": 0,
            "min_messages": 1,
            "broadcast": False,
            "summary_mode": "message_count",
            "message_chunk_size": 30,
            "time_window_minutes": 15,
        }
    )
    plugin._last_summary_time = {}
    plugin._last_summary_hash = {}
    plugin.state_updates = []
    plugin.persisted_files = []

    async def fake_collect_auto_summary_messages(*args, **kwargs):
        return "cached text", [
            {
                "message_id": "10",
                "time": datetime.fromtimestamp(1710000300),
                "timestamp": 1710000300,
                "user_id": "2",
                "nickname": "New",
                "text": "cached text",
            }
        ]

    async def fake_summarize_text(*args, **kwargs):
        return "LLM 调用失败，请检查模型配置后重试。"

    def fake_store_state(group_id, structured, content_hash):
        plugin.state_updates.append((group_id, structured, content_hash))

    def fake_persist_summary_file(**kwargs):
        plugin.persisted_files.append(kwargs)
        return Path(tmp_path / "summary.md")

    plugin._get_aiocqhttp_client = lambda: object()
    plugin._prune_message_cache = lambda *args, **kwargs: None
    plugin._get_persistent_summary_state = lambda *args, **kwargs: None
    plugin._collect_auto_summary_messages = fake_collect_auto_summary_messages
    plugin._summarize_text = fake_summarize_text
    plugin._store_persistent_summary_state = fake_store_state
    plugin._persist_summary_file = fake_persist_summary_file

    asyncio.run(
        plugin._execute_auto_summary(
            plugin.settings["auto_summary"],
            {
                "limits": {
                    "max_chat_records": 200,
                    "max_tokens": 1000,
                    "max_input_chars": 2000,
                },
                "auto_summary": plugin.settings["auto_summary"],
                "render_as_image": False,
            },
        )
    )

    assert plugin.state_updates == []
    assert plugin.persisted_files == []


def test_execute_auto_summary_uses_filtered_new_messages_not_old_context(tmp_path):
    plugin = build_plugin(
        {
            "enabled": True,
            "target_groups": ["868381808"],
            "cache_enabled": True,
            "history_backfill_count": 100,
            "min_messages": 1,
            "broadcast": False,
            "summary_mode": "message_count",
            "message_chunk_size": 30,
            "time_window_minutes": 15,
        }
    )
    timestamp = 1710000300
    plugin._last_summary_time = {}
    plugin._last_summary_hash = {}
    plugin.persisted_files = []
    plugin.state_updates = []

    async def fake_collect_auto_summary_messages(*args, **kwargs):
        return "old text\nnew text", [
            {
                "message_id": "9",
                "time": datetime.fromtimestamp(timestamp),
                "timestamp": timestamp,
                "user_id": "1",
                "nickname": "Old",
                "text": "old text",
            },
            {
                "message_id": "11",
                "time": datetime.fromtimestamp(timestamp),
                "timestamp": timestamp,
                "user_id": "2",
                "nickname": "New",
                "text": "new text",
            },
        ]

    async def fake_summarize_text(*args, **kwargs):
        return "summary"

    def fake_persist_summary_file(**kwargs):
        plugin.persisted_files.append(kwargs)
        return Path(tmp_path / "summary.md")

    def fake_store_state(group_id, structured, content_hash):
        plugin.state_updates.append((group_id, structured, content_hash))

    plugin._get_aiocqhttp_client = lambda: object()
    plugin._prune_message_cache = lambda *args, **kwargs: None
    plugin._get_persistent_summary_state = lambda group_id: {
        "last_summarized_ts": timestamp,
        "last_summarized_message_id": "10",
        "last_summary_hash": "",
    }
    plugin._collect_auto_summary_messages = fake_collect_auto_summary_messages
    plugin._summarize_text = fake_summarize_text
    async def fake_group_info(*args, **kwargs):
        return {"group_name": "group"}

    plugin._safe_group_info = fake_group_info
    plugin._persist_summary_file = fake_persist_summary_file
    plugin._store_persistent_summary_state = fake_store_state

    asyncio.run(
        plugin._execute_auto_summary(
            plugin.settings["auto_summary"],
            {
                "limits": {
                    "max_chat_records": 200,
                    "max_tokens": 1000,
                    "max_input_chars": 2000,
                },
                "auto_summary": plugin.settings["auto_summary"],
                "render_as_image": False,
            },
        )
    )

    assert [msg["message_id"] for msg in plugin.persisted_files[0]["messages"]] == ["11"]
    assert [msg["message_id"] for msg in plugin.state_updates[0][1]] == ["11"]
