# event_log.py — Immutable append-only SQLite event log (Essence Governance Plane)
# All system events are written here. Nothing is ever deleted.
# Part of the Essence personal AI agent system.

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Optional

# ---------------------------------------------------------------------------
# UUID v7-style helper (timestamp-prefixed UUID4 for sortability)
# ---------------------------------------------------------------------------

def _new_event_id() -> str:
    """Return a sortable UUID: 13-hex-char ms timestamp + random UUID4 suffix."""
    ts_hex = format(int(time.time() * 1000), "013x")
    rand_part = uuid.uuid4().hex[13:]
    raw = ts_hex + rand_part          # 13 + 19 = 32 hex chars
    # Format as UUID: 8-4-4-4-12
    return f"{raw[0:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:32]}"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Migration: add columns introduced in v1.1 if they don't exist yet
_MIGRATIONS = [
    "ALTER TABLE event_log ADD COLUMN sequence_number INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE event_log ADD COLUMN signature TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE event_log ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'",
]

_DDL = """
CREATE TABLE IF NOT EXISTS event_log (
    id              TEXT    PRIMARY KEY,
    topic           TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    task_id         TEXT,
    command_id      TEXT,
    data_json       TEXT    NOT NULL,
    sequence_number INTEGER NOT NULL DEFAULT 0,
    signature       TEXT    NOT NULL DEFAULT '',
    metadata_json   TEXT    NOT NULL DEFAULT '{}',
    created_at      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_el_topic ON event_log(topic, created_at);
CREATE INDEX IF NOT EXISTS idx_el_task  ON event_log(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_el_seq   ON event_log(topic, sequence_number);

CREATE TABLE IF NOT EXISTS command_results (
    command_id  TEXT    PRIMARY KEY,
    result_json TEXT    NOT NULL,
    created_at  INTEGER NOT NULL,
    expires_at  INTEGER NOT NULL
);
"""


# ---------------------------------------------------------------------------
# EventLog class
# ---------------------------------------------------------------------------

class EventLog:
    """Append-only SQLite event log for the Essence governance plane.

    Thread-safe via a single write lock; reads use their own short-lived
    connections so they never block writers.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        with self._connect() as conn:
            conn.executescript(_DDL)
            # Apply any missing column migrations (idempotent — ignores "duplicate column")
            for stmt in _MIGRATIONS:
                try:
                    conn.execute(stmt)
                    conn.commit()
                except sqlite3.OperationalError:
                    pass  # column already exists
        self.prune_cache()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        try:
            d["data"] = json.loads(d.pop("data_json"))
        except (KeyError, json.JSONDecodeError):
            pass
        try:
            d["metadata"] = json.loads(d.pop("metadata_json", "{}") or "{}")
        except json.JSONDecodeError:
            d["metadata"] = {}
        return d

    # ------------------------------------------------------------------
    # Core write operations
    # ------------------------------------------------------------------

    def emit(
        self,
        topic: str,
        data: dict,
        source: str = "system",
        task_id: str = "",
        command_id: str = "",
        sequence_number: int = 0,
        signature: str = "",
        metadata: dict | None = None,
    ) -> str:
        """Append one event and return its generated id."""
        event_id = _new_event_id()
        now_ms = int(time.time() * 1000)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO event_log
                        (id, topic, source, task_id, command_id, data_json,
                         sequence_number, signature, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        topic,
                        source,
                        task_id or None,
                        command_id or None,
                        json.dumps(data, ensure_ascii=False),
                        sequence_number,
                        signature,
                        json.dumps(metadata or {}, ensure_ascii=False),
                        now_ms,
                    ),
                )
                conn.commit()
        return event_id

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def tail(self, n: int = 50, topic: str = "") -> list[dict]:
        """Return the most-recent *n* events, optionally filtered by topic."""
        with self._connect() as conn:
            if topic:
                rows = conn.execute(
                    """
                    SELECT * FROM event_log
                    WHERE topic = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (topic, n),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM event_log
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (n,),
                ).fetchall()
        return [self._row_to_dict(r) for r in reversed(rows)]

    def since(self, ts: float, topic: str = "") -> list[dict]:
        """Return all events with created_at >= *ts* (unix seconds float)."""
        ts_ms = int(ts * 1000)
        with self._connect() as conn:
            if topic:
                rows = conn.execute(
                    """
                    SELECT * FROM event_log
                    WHERE created_at >= ? AND topic = ?
                    ORDER BY created_at ASC, id ASC
                    """,
                    (ts_ms, topic),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM event_log
                    WHERE created_at >= ?
                    ORDER BY created_at ASC, id ASC
                    """,
                    (ts_ms,),
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_by_task(self, task_id: str) -> list[dict]:
        """Return all events belonging to a causal task group."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM event_log
                WHERE task_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (task_id,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Command-result cache (idempotency store)
    # ------------------------------------------------------------------

    def cache_result(
        self,
        command_id: str,
        result: dict,
        ttl_s: float = 300,
    ) -> None:
        """Store a command result for TTL-based idempotency replay."""
        now_ms = int(time.time() * 1000)
        expires_ms = now_ms + int(ttl_s * 1000)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO command_results
                        (command_id, result_json, created_at, expires_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (command_id, json.dumps(result, ensure_ascii=False), now_ms, expires_ms),
                )
                conn.commit()

    def get_cached(self, command_id: str) -> dict | None:
        """Return a cached result if it still exists and has not expired."""
        now_ms = int(time.time() * 1000)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT result_json FROM command_results
                WHERE command_id = ? AND expires_at > ?
                """,
                (command_id, now_ms),
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["result_json"])
        except json.JSONDecodeError:
            return None

    def prune_cache(self) -> int:
        """Delete expired command_results rows. Returns the number removed."""
        now_ms = int(time.time() * 1000)
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM command_results WHERE expires_at <= ?",
                    (now_ms,),
                )
                conn.commit()
                return cur.rowcount

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return aggregate statistics about the event log."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM event_log").fetchone()[0]
            topics_rows = conn.execute(
                "SELECT topic, COUNT(*) as cnt FROM event_log GROUP BY topic ORDER BY cnt DESC"
            ).fetchall()
            oldest_row = conn.execute(
                "SELECT created_at FROM event_log ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
            newest_row = conn.execute(
                "SELECT created_at FROM event_log ORDER BY created_at DESC LIMIT 1"
            ).fetchone()

        topics = {r["topic"]: r["cnt"] for r in topics_rows}
        oldest = (oldest_row["created_at"] / 1000.0) if oldest_row else None
        newest = (newest_row["created_at"] / 1000.0) if newest_row else None

        return {
            "total_events": total,
            "topics": topics,
            "oldest": oldest,
            "newest": newest,
        }


# ---------------------------------------------------------------------------
# Shared singleton — lazily initialised when the router is first imported
# ---------------------------------------------------------------------------

_DEFAULT_DB = Path(__file__).parent.parent / "data" / "event_log.db"
_event_log: Optional[EventLog] = None


def get_event_log() -> EventLog:
    global _event_log
    if _event_log is None:
        _event_log = EventLog(_DEFAULT_DB)
    return _event_log


