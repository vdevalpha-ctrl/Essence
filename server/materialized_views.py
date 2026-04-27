"""
materialized_views.py — Deterministic read projections over the event log.

Views are rebuilt from the append-only event log, so they can always be
regenerated — they are NOT authoritative state, only cached projections.

Views:
  view_activity      — aggregate event counts per topic per day
  view_skill_health  — per-skill invocation counts and last-error
  view_memory_stats  — gravity memory tier distribution
  view_violations    — governance violations from the event log
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Any


_DDL = """
CREATE TABLE IF NOT EXISTS view_activity (
    date        TEXT    NOT NULL,
    topic       TEXT    NOT NULL,
    event_count INTEGER NOT NULL DEFAULT 0,
    rebuilt_at  INTEGER NOT NULL,
    PRIMARY KEY (date, topic)
);

CREATE TABLE IF NOT EXISTS view_skill_health (
    skill_id       TEXT PRIMARY KEY,
    invoke_count   INTEGER NOT NULL DEFAULT 0,
    error_count    INTEGER NOT NULL DEFAULT 0,
    last_error     TEXT,
    last_invoked   INTEGER,
    avg_latency_ms REAL,
    rebuilt_at     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS view_memory_stats (
    tier          TEXT PRIMARY KEY,
    entry_count   INTEGER NOT NULL DEFAULT 0,
    avg_gravity   REAL,
    rebuilt_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS view_violations (
    event_id    TEXT PRIMARY KEY,
    topic       TEXT NOT NULL,
    source      TEXT NOT NULL,
    severity    TEXT,
    description TEXT,
    task_id     TEXT,
    occurred_at INTEGER NOT NULL,
    rebuilt_at  INTEGER NOT NULL
);
"""


class MaterializedViews:
    """
    Rebuilds read-projection tables from the event log on demand.
    Each rebuild() call is idempotent — safe to call at any frequency.
    """

    def __init__(self, views_db: Path, event_log: Any, gravity_memory: Any = None) -> None:
        self._path = views_db
        self._event_log = event_log
        self._gravity = gravity_memory
        self._lock = Lock()
        views_db.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_DDL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ── Rebuild ──────────────────────────────────────────────────────────

    def rebuild_activity(self, days: int = 30) -> int:
        """Rebuild view_activity from the last *days* of event log entries."""
        since_ts = time.time() - days * 86400
        events = self._event_log.since(since_ts)
        now = int(time.time())

        # Aggregate: (date, topic) → count
        counts: dict[tuple[str, str], int] = {}
        for ev in events:
            ts = ev.get("created_at", 0)
            date_str = _ms_to_date(ts)
            topic = ev.get("topic", "unknown")
            counts[(date_str, topic)] = counts.get((date_str, topic), 0) + 1

        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM view_activity WHERE rebuilt_at < ?", (now - 1,))
                for (date_str, topic), cnt in counts.items():
                    conn.execute(
                        """INSERT OR REPLACE INTO view_activity (date, topic, event_count, rebuilt_at)
                           VALUES (?, ?, ?, ?)""",
                        (date_str, topic, cnt, now),
                    )
                conn.commit()
        return len(counts)

    def rebuild_skill_health(self) -> int:
        """Rebuild view_skill_health from skill.execute / skill.result events."""
        execute_events = self._event_log.tail(n=5000, topic="skill.execute")
        result_events  = self._event_log.tail(n=5000, topic="skill.result")
        now = int(time.time())

        skills: dict[str, dict] = {}

        for ev in execute_events:
            data = ev.get("data", {})
            sid  = data.get("skill_id", "unknown")
            s = skills.setdefault(sid, {"invoke": 0, "errors": 0, "last_error": None,
                                        "last_invoked": None, "latencies": []})
            s["invoke"] += 1
            ts = ev.get("created_at", 0)
            if s["last_invoked"] is None or ts > s["last_invoked"]:
                s["last_invoked"] = ts

        for ev in result_events:
            data = ev.get("data", {})
            sid  = data.get("skill_id", "unknown")
            s = skills.setdefault(sid, {"invoke": 0, "errors": 0, "last_error": None,
                                        "last_invoked": None, "latencies": []})
            if data.get("status") == "error":
                s["errors"] += 1
                s["last_error"] = data.get("error", "")
            if "latency_ms" in data:
                s["latencies"].append(float(data["latency_ms"]))

        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM view_skill_health")
                for sid, s in skills.items():
                    avg_lat = (sum(s["latencies"]) / len(s["latencies"])) if s["latencies"] else None
                    conn.execute(
                        """INSERT OR REPLACE INTO view_skill_health
                           (skill_id, invoke_count, error_count, last_error, last_invoked, avg_latency_ms, rebuilt_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (sid, s["invoke"], s["errors"], s["last_error"], s["last_invoked"], avg_lat, now),
                    )
                conn.commit()
        return len(skills)

    def rebuild_memory_stats(self) -> int:
        """Rebuild view_memory_stats from GravityMemory.stats() if available."""
        if self._gravity is None:
            return 0
        stats = self._gravity.stats()
        now   = int(time.time())
        tiers = [
            ("volatile", stats.get("volatile_count", 0)),
            ("weighted", stats.get("weighted_count", 0)),
            ("anchored", stats.get("anchored_count", 0)),
        ]
        avg_g = stats.get("avg_gravity")

        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM view_memory_stats")
                for tier, cnt in tiers:
                    conn.execute(
                        "INSERT OR REPLACE INTO view_memory_stats (tier, entry_count, avg_gravity, rebuilt_at) VALUES (?, ?, ?, ?)",
                        (tier, cnt, avg_g, now),
                    )
                conn.commit()
        return len(tiers)

    def rebuild_violations(self) -> int:
        """Rebuild view_violations from governance.violation events."""
        events = self._event_log.tail(n=2000, topic="governance.violation")
        now = int(time.time())

        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM view_violations")
                for ev in events:
                    data = ev.get("data", {})
                    conn.execute(
                        """INSERT OR REPLACE INTO view_violations
                           (event_id, topic, source, severity, description, task_id, occurred_at, rebuilt_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            ev.get("id", ""),
                            ev.get("topic", "governance.violation"),
                            ev.get("source", ""),
                            data.get("severity", ""),
                            data.get("description", ""),
                            ev.get("task_id", ""),
                            ev.get("created_at", 0),
                            now,
                        ),
                    )
                conn.commit()
        return len(events)

    def rebuild_all(self) -> dict[str, int]:
        """Rebuild all views and return row counts."""
        return {
            "activity":    self.rebuild_activity(),
            "skill_health": self.rebuild_skill_health(),
            "memory_stats": self.rebuild_memory_stats(),
            "violations":  self.rebuild_violations(),
        }

    # ── Read ─────────────────────────────────────────────────────────────

    def get_activity(self, days: int = 7) -> list[dict]:
        cutoff = _ts_to_date(time.time() - days * 86400)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM view_activity WHERE date >= ? ORDER BY date DESC, event_count DESC",
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_skill_health(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM view_skill_health ORDER BY invoke_count DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_memory_stats(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM view_memory_stats").fetchall()
        return [dict(r) for r in rows]

    def get_violations(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM view_violations ORDER BY occurred_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms_to_date(ts_ms: int) -> str:
    """Convert millisecond timestamp to YYYY-MM-DD string."""
    return _ts_to_date(ts_ms / 1000.0)


def _ts_to_date(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "views.db"
_views: MaterializedViews | None = None


def init_materialized_views(
    event_log: Any, gravity_memory: Any = None, db_path: Path | None = None
) -> MaterializedViews:
    """Initialise the global materialized views. Call once at startup."""
    global _views
    _views = MaterializedViews(db_path or _DEFAULT_DB, event_log, gravity_memory)
    return _views


def get_materialized_views() -> MaterializedViews:
    if _views is None:
        raise RuntimeError("Materialized views not initialised. Call init_materialized_views() at startup.")
    return _views
