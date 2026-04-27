"""essence_ledger.quiet_ledger — Accountable silence ledger (ported from AURA v4.8.0)."""
from __future__ import annotations
import json, logging, sqlite3, uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger("essence.quiet_ledger")


class RollbackTier(str, Enum):
    R1 = "R1"   # fully reversible
    R2 = "R2"   # reversible with confirmation
    R3 = "R3"   # irreversible


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS quiet_ledger (
    ledger_id TEXT PRIMARY KEY, session_id TEXT NOT NULL DEFAULT '',
    entry_type TEXT NOT NULL DEFAULT 'observation',
    action_type TEXT NOT NULL DEFAULT '', plain_summary TEXT NOT NULL DEFAULT '',
    reasoning TEXT NOT NULL DEFAULT '', outcome TEXT, rollback_tier TEXT,
    rollback_token TEXT, rollback_executed INTEGER NOT NULL DEFAULT 0,
    is_abstention INTEGER NOT NULL DEFAULT 0, abstention_reason TEXT,
    endorsement_score REAL NOT NULL DEFAULT 0.0, resonance_score REAL NOT NULL DEFAULT 0.0,
    domain TEXT NOT NULL DEFAULT '', step_id TEXT NOT NULL DEFAULT '',
    extra TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ql_session    ON quiet_ledger (session_id);
CREATE INDEX IF NOT EXISTS idx_ql_type       ON quiet_ledger (entry_type);
CREATE INDEX IF NOT EXISTS idx_ql_abstention ON quiet_ledger (is_abstention);
"""


class QuietLedger:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c: c.executescript(_CREATE_SQL)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try: yield conn; conn.commit()
        finally: conn.close()

    def record_action(self, *, session_id: str, step_id: str = "", action_type: str,
                      plain_summary: str, reasoning: str = "", outcome: str | None = None,
                      rollback_tier: RollbackTier = RollbackTier.R3,
                      rollback_token: str | None = None,
                      endorsement_score: float = 0.0, domain: str = "",
                      extra: dict | None = None) -> str:
        lid = str(uuid.uuid4())
        with self._conn() as c:
            c.execute("""INSERT INTO quiet_ledger
                (ledger_id,session_id,entry_type,action_type,plain_summary,reasoning,
                 outcome,rollback_tier,rollback_token,rollback_executed,is_abstention,
                 endorsement_score,domain,step_id,extra,created_at)
                VALUES (?,?,'action',?,?,?,?,?,?,0,0,?,?,?,?,?)""",
                (lid,session_id,action_type,plain_summary,reasoning,outcome,
                 rollback_tier.value if rollback_tier else None,rollback_token,
                 endorsement_score,domain,step_id,json.dumps(extra or {}),
                 datetime.now(timezone.utc).isoformat()))
        return lid

    def record_abstention(self, *, session_id: str, step_id: str = "", action_type: str,
                          plain_summary: str, abstention_reason: str, reasoning: str = "",
                          domain: str = "", extra: dict | None = None) -> str:
        lid = str(uuid.uuid4())
        with self._conn() as c:
            c.execute("""INSERT INTO quiet_ledger
                (ledger_id,session_id,entry_type,action_type,plain_summary,reasoning,
                 is_abstention,abstention_reason,domain,step_id,extra,created_at)
                VALUES (?,?,'abstention',?,?,?,1,?,?,?,?,?)""",
                (lid,session_id,action_type,plain_summary,reasoning,
                 abstention_reason,domain,step_id,json.dumps(extra or {}),
                 datetime.now(timezone.utc).isoformat()))
        return lid

    def record_observation(self, *, session_id: str, plain_summary: str,
                           extra: dict | None = None) -> str:
        lid = str(uuid.uuid4())
        with self._conn() as c:
            c.execute("""INSERT INTO quiet_ledger
                (ledger_id,session_id,entry_type,plain_summary,extra,created_at)
                VALUES (?,?,'observation',?,?,?)""",
                (lid,session_id,plain_summary,json.dumps(extra or {}),
                 datetime.now(timezone.utc).isoformat()))
        return lid

    def actions_taken(self, session_id: str | None = None, undoable_only: bool = False,
                      limit: int = 100) -> list[dict]:
        sql = "SELECT * FROM quiet_ledger WHERE entry_type='action'"
        params: list = []
        if session_id: sql += " AND session_id=?"; params.append(session_id)
        if undoable_only: sql += " AND rollback_tier IN ('R1','R2') AND rollback_executed=0"
        sql += f" ORDER BY created_at DESC LIMIT {int(limit)}"
        with self._conn() as c: rows = c.execute(sql, params).fetchall()
        return [self._to_dict(r) for r in rows]

    def deliberate_abstentions(self, session_id: str | None = None, limit: int = 100) -> list[dict]:
        sql = "SELECT * FROM quiet_ledger WHERE entry_type='abstention'"
        params: list = []
        if session_id: sql += " AND session_id=?"; params.append(session_id)
        sql += f" ORDER BY created_at DESC LIMIT {int(limit)}"
        with self._conn() as c: rows = c.execute(sql, params).fetchall()
        return [self._to_dict(r) for r in rows]

    def mark_rollback_executed(self, ledger_id: str) -> bool:
        with self._conn() as c:
            row = c.execute("SELECT rollback_tier,rollback_executed FROM quiet_ledger WHERE ledger_id=?",
                            (ledger_id,)).fetchone()
            if not row or row["rollback_tier"] not in ("R1","R2") or row["rollback_executed"]:
                return False
            c.execute("UPDATE quiet_ledger SET rollback_executed=1 WHERE ledger_id=?", (ledger_id,))
        return True

    def why(self, ledger_id: str) -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT plain_summary,reasoning,abstention_reason FROM quiet_ledger WHERE ledger_id=?",
                            (ledger_id,)).fetchone()
        if not row: return None
        parts = [row["plain_summary"]]
        if row["reasoning"]: parts.append(f"Why: {row['reasoning']}")
        if row["abstention_reason"]: parts.append(f"Abstention: {row['abstention_reason']}")
        return "  ".join(parts)

    def recent_summary(self, session_id: str, hours: int = 24) -> dict:
        with self._conn() as c:
            row = c.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE entry_type='action')     AS action_count,
                    COUNT(*) FILTER (WHERE entry_type='abstention') AS abstention_count,
                    COUNT(*) FILTER (WHERE entry_type='observation') AS observation_count,
                    COUNT(*) FILTER (WHERE rollback_executed=1)     AS undo_count
                FROM quiet_ledger
                WHERE session_id=? AND created_at >= datetime('now', ? || ' hours')""",
                (session_id, f"-{hours}")).fetchone()
        return dict(row) if row else {}

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        try: d["extra"] = json.loads(d.get("extra") or "{}")
        except Exception: pass
        return d
