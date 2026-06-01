import hashlib
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, List


@dataclass(frozen=True)
class CachedMessage:
    group_id: str
    message_id: str
    user_id: str
    nickname: str
    timestamp: int
    text: str
    raw_type_summary: str = ""


class MessageCache:
    """SQLite-backed local message cache for event-driven auto summaries."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    group_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    nickname TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    raw_type_summary TEXT NOT NULL DEFAULT '',
                    message_id_num INTEGER,
                    message_id_is_numeric INTEGER NOT NULL DEFAULT 0,
                    hash TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY (group_id, message_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_group_timestamp
                ON messages (group_id, timestamp, message_id)
                """
            )
            self._ensure_column(conn, "messages", "message_id_num", "INTEGER")
            self._ensure_column(conn, "messages", "message_id_is_numeric", "INTEGER NOT NULL DEFAULT 0")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS summary_state (
                    group_id TEXT PRIMARY KEY,
                    last_summarized_ts INTEGER NOT NULL DEFAULT 0,
                    last_summarized_message_id TEXT NOT NULL DEFAULT '',
                    last_summary_hash TEXT NOT NULL DEFAULT '',
                    updated_at INTEGER NOT NULL
                )
                """
            )

    def _ensure_column(self, conn: sqlite3.Connection, table_name: str, column_name: str, ddl: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        if any(row["name"] == column_name for row in rows):
            return
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}")

    def add_message(self, message: CachedMessage) -> bool:
        if not message.group_id or not message.message_id or not message.text.strip():
            return False

        content_hash = self._message_hash(message)
        message_id_num = self._message_id_num(message.message_id)
        message_id_is_numeric = 1 if message_id_num is not None else 0
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO messages (
                    group_id, message_id, user_id, nickname, timestamp, text,
                    raw_type_summary, message_id_num, message_id_is_numeric, hash, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.group_id,
                    message.message_id,
                    message.user_id,
                    message.nickname,
                    int(message.timestamp),
                    message.text.strip(),
                    message.raw_type_summary,
                    message_id_num,
                    message_id_is_numeric,
                    content_hash,
                    int(time.time()),
                ),
            )
            return cursor.rowcount == 1

    def get_messages_after(self, group_id: str, *, after_timestamp: int, limit: int) -> List[dict]:
        safe_limit = max(1, int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT group_id, message_id, user_id, nickname, timestamp, text, raw_type_summary
                FROM messages
                WHERE group_id = ? AND timestamp > ?
                ORDER BY timestamp ASC, message_id ASC
                LIMIT ?
                """,
                (str(group_id), int(after_timestamp), safe_limit),
            ).fetchall()

        return [self._row_to_message(row) for row in rows]

    def get_messages_after_cursor(
        self,
        group_id: str,
        *,
        after_timestamp: int,
        after_message_id: str,
        limit: int,
    ) -> List[dict]:
        safe_limit = max(1, int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT group_id, message_id, user_id, nickname, timestamp, text, raw_type_summary
                FROM messages
                WHERE group_id = ?
                  AND (
                    timestamp > ?
                    OR (
                        timestamp = ?
                        AND (
                            message_id_is_numeric > ?
                            OR (
                                message_id_is_numeric = ?
                                AND COALESCE(message_id_num, 0) > ?
                            )
                            OR (
                                message_id_is_numeric = ?
                                AND COALESCE(message_id_num, 0) = ?
                                AND message_id > ?
                            )
                        )
                    )
                  )
                ORDER BY timestamp ASC, message_id_is_numeric ASC, COALESCE(message_id_num, 0) ASC, message_id ASC
                LIMIT ?
                """,
                self._cursor_params(group_id, after_timestamp, after_message_id, safe_limit),
            ).fetchall()

        return [self._row_to_message(row) for row in rows]

    def get_recent_messages(self, group_id: str, *, limit: int) -> List[dict]:
        safe_limit = max(1, int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT group_id, message_id, user_id, nickname, timestamp, text, raw_type_summary
                FROM messages
                WHERE group_id = ?
                ORDER BY timestamp DESC, message_id_is_numeric DESC, COALESCE(message_id_num, 0) DESC, message_id DESC
                LIMIT ?
                """,
                (str(group_id), safe_limit),
            ).fetchall()

        return [self._row_to_message(row) for row in reversed(rows)]

    def get_summary_state(self, group_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT group_id, last_summarized_ts, last_summarized_message_id,
                       last_summary_hash, updated_at
                FROM summary_state
                WHERE group_id = ?
                """,
                (str(group_id),),
            ).fetchone()

        if row is None:
            return None
        return dict(row)

    def set_summary_state(
        self,
        *,
        group_id: str,
        last_summarized_ts: int,
        last_summarized_message_id: str,
        last_summary_hash: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO summary_state (
                    group_id, last_summarized_ts, last_summarized_message_id,
                    last_summary_hash, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(group_id) DO UPDATE SET
                    last_summarized_ts = excluded.last_summarized_ts,
                    last_summarized_message_id = excluded.last_summarized_message_id,
                    last_summary_hash = excluded.last_summary_hash,
                    updated_at = excluded.updated_at
                """,
                (
                    str(group_id),
                    int(last_summarized_ts),
                    str(last_summarized_message_id or ""),
                    str(last_summary_hash or ""),
                    int(time.time()),
                ),
            )

    def prune_before(self, group_id: str, *, before_timestamp: int) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM messages WHERE group_id = ? AND timestamp < ?",
                (str(group_id), int(before_timestamp)),
            )
            return cursor.rowcount

    def _row_to_message(self, row: sqlite3.Row) -> dict:
        timestamp = int(row["timestamp"])
        return {
            "group_id": row["group_id"],
            "message_id": row["message_id"],
            "user_id": row["user_id"],
            "nickname": row["nickname"],
            "time": datetime.fromtimestamp(timestamp),
            "timestamp": timestamp,
            "text": row["text"],
            "raw_type_summary": row["raw_type_summary"],
        }

    def _message_hash(self, message: CachedMessage) -> str:
        content = f"{message.group_id}:{message.message_id}:{message.user_id}:{message.timestamp}:{message.text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _message_id_num(self, message_id: str) -> int | None:
        text = str(message_id or "").strip()
        if not text.isdigit():
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def _cursor_params(
        self,
        group_id: str,
        after_timestamp: int,
        after_message_id: str,
        limit: int,
    ) -> tuple:
        after_num = self._message_id_num(after_message_id)
        after_is_numeric = 1 if after_num is not None else 0
        normalized_after_num = after_num if after_num is not None else 0
        return (
            str(group_id),
            int(after_timestamp),
            int(after_timestamp),
            after_is_numeric,
            after_is_numeric,
            normalized_after_num,
            after_is_numeric,
            normalized_after_num,
            str(after_message_id or ""),
            limit,
        )
