from datetime import datetime
import shutil

from message_cache import CachedMessage, MessageCache


def test_message_cache_deduplicates_by_group_and_message_id(tmp_path):
    cache = MessageCache(tmp_path / "messages.sqlite3")
    first = CachedMessage(
        group_id="868381808",
        message_id="100",
        user_id="42",
        nickname="Alice",
        timestamp=1710000000,
        text="first version",
        raw_type_summary="text",
    )
    duplicate = CachedMessage(
        group_id="868381808",
        message_id="100",
        user_id="42",
        nickname="Alice",
        timestamp=1710000001,
        text="duplicate version",
        raw_type_summary="text",
    )

    assert cache.add_message(first) is True
    assert cache.add_message(duplicate) is False

    messages = cache.get_messages_after("868381808", after_timestamp=0, limit=10)
    assert len(messages) == 1
    assert messages[0]["text"] == "first version"


def test_message_cache_reads_messages_after_summary_cursor(tmp_path):
    cache = MessageCache(tmp_path / "messages.sqlite3")
    cache.add_message(
        CachedMessage(
            group_id="868381808",
            message_id="old",
            user_id="1",
            nickname="Old",
            timestamp=100,
            text="old message",
            raw_type_summary="text",
        )
    )
    cache.add_message(
        CachedMessage(
            group_id="868381808",
            message_id="new",
            user_id="2",
            nickname="New",
            timestamp=200,
            text="new message",
            raw_type_summary="text",
        )
    )
    cache.set_summary_state(
        group_id="868381808",
        last_summarized_ts=100,
        last_summarized_message_id="old",
        last_summary_hash="abc",
    )

    state = cache.get_summary_state("868381808")
    messages = cache.get_messages_after(
        "868381808",
        after_timestamp=state["last_summarized_ts"],
        limit=10,
    )

    assert len(messages) == 1
    assert messages[0]["message_id"] == "new"
    assert messages[0]["time"] == datetime.fromtimestamp(200)


def test_message_cache_cursor_uses_message_id_as_timestamp_tie_breaker(tmp_path):
    cache = MessageCache(tmp_path / "messages.sqlite3")
    for message_id in ["m1", "m2", "m3"]:
        cache.add_message(
            CachedMessage(
                group_id="868381808",
                message_id=message_id,
                user_id="1",
                nickname="SameSecond",
                timestamp=300,
                text=f"{message_id} text",
                raw_type_summary="text",
            )
        )

    messages = cache.get_messages_after_cursor(
        "868381808",
        after_timestamp=300,
        after_message_id="m1",
        limit=10,
    )

    assert [msg["message_id"] for msg in messages] == ["m2", "m3"]


def test_message_cache_cursor_orders_numeric_message_ids_numerically(tmp_path):
    cache = MessageCache(tmp_path / "messages.sqlite3")
    for message_id in ["9", "10"]:
        cache.add_message(
            CachedMessage(
                group_id="868381808",
                message_id=message_id,
                user_id="1",
                nickname="NumericId",
                timestamp=300,
                text=f"{message_id} text",
                raw_type_summary="text",
            )
        )

    messages = cache.get_messages_after_cursor(
        "868381808",
        after_timestamp=300,
        after_message_id="9",
        limit=10,
    )

    assert [msg["message_id"] for msg in messages] == ["10"]


def test_message_cache_prunes_old_messages_per_group(tmp_path):
    cache = MessageCache(tmp_path / "messages.sqlite3")
    cache.add_message(
        CachedMessage(
            group_id="868381808",
            message_id="old",
            user_id="1",
            nickname="Old",
            timestamp=100,
            text="old text",
            raw_type_summary="text",
        )
    )
    cache.add_message(
        CachedMessage(
            group_id="868381808",
            message_id="new",
            user_id="1",
            nickname="New",
            timestamp=300,
            text="new text",
            raw_type_summary="text",
        )
    )

    deleted = cache.prune_before("868381808", before_timestamp=200)

    assert deleted == 1
    assert [msg["message_id"] for msg in cache.get_recent_messages("868381808", limit=10)] == ["new"]


def test_message_cache_closes_connections_after_operations(tmp_path):
    cache_dir = tmp_path / "cache-dir"
    cache = MessageCache(cache_dir / "messages.sqlite3")
    cache.add_message(
        CachedMessage(
            group_id="868381808",
            message_id="1",
            user_id="1",
            nickname="Alice",
            timestamp=100,
            text="hello",
            raw_type_summary="text",
        )
    )
    cache.get_recent_messages("868381808", limit=10)

    shutil.rmtree(cache_dir)
