"""
audit_logger.py — Essence Audit Trail
=======================================
Records every governance-relevant action to a dedicated SQLite table:
  • Tool approvals (grant / deny)
  • Model switches
  • Configuration changes
  • Governance violations

The table is append-only and separate from the event log so compliance
reports can be extracted without re-parsing the full event stream.

Usage
-----
    logger = get_audit_logger()
    logger.log_approval("shell", {"command": "ls"}, granted=True, actor="user")
    logger.log_model_switch("ollama", "qwen3:4b", "groq", "llama3-70b")
    logger.log_config_change("persona.tone", "concise", "formal", actor="user")
    logger.log_violation("missing_capability_tokens", "WARN", skill_id="research")

    report = logger.export_compliance_report(fmt="json")
    Path("report.json").write_text(report)
"""
from __future__ import annotations

import csv
import io
import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Optional

log = logging.getLogger("essence.audit")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_DDL = """
CREATE TABLE IF NOT EXISTS audit_logs (
    id           TEXT    PRIMARY KEY,
    event_type   TEXT    NOT NULL,   -- approval|model_switch|config_change|violation
    occurred_at  INTEGER NOT NULL,   -- epoch ms
    actor        TEXT    NOT NULL DEFAULT 'system',
    skill_id     TEXT,
    tool_name    TEXT,
    granted      INTEGER,            -- NULL if N/A, 1=granted, 0=denied
    old_value    TEXT,
    new_value    TEXT,
    severity     TEXT,
    description  TEXT    NOT NULL DEFAULT '',
    metadata_json TEXT   NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_al_type ON audit_logs(event_type, occurred_at);
CREATE INDEX IF NOT EXISTS idx_al_time ON audit_logs(occurred_at);
"""


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------
class AuditLogger:
    """Thread-safe append-only audit log backed by SQLite."""

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        with self._connect() as conn:
            conn.executescript(_DDL)

    # ── Internal ─────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _insert(self, **kwargs) -> str:
        row_id = uuid.uuid4().hex
        now_ms = int(time.time() * 1000)
        meta   = json.dumps(kwargs.pop("metadata", {}), ensure_ascii=False)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO audit_logs
                        (id, event_type, occurred_at, actor, skill_id, tool_name,
                         granted, old_value, new_value, severity, description, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row_id,
                        kwargs.get("event_type", "unknown"),
                        now_ms,
                        kwargs.get("actor", "system"),
                        kwargs.get("skill_id"),
                        kwargs.get("tool_name"),
                        kwargs.get("granted"),
                        kwargs.get("old_value"),
                        kwargs.get("new_value"),
                        kwargs.get("severity"),
                        kwargs.get("description", ""),
                        meta,
                    ),
                )
                conn.commit()
        return row_id

    # ── Public write API ─────────────────────────────────────────────

    def log_approval(
        self,
        tool_name:  str,
        args:       dict,
        granted:    bool,
        actor:      str = "user",
        skill_id:   str = "",
    ) -> str:
        desc = f"Tool '{tool_name}' {'granted' if granted else 'denied'} by {actor}"
        log.info("Audit: %s", desc)
        return self._insert(
            event_type="approval",
            actor=actor,
            tool_name=tool_name,
            skill_id=skill_id or None,
            granted=int(granted),
            description=desc,
            metadata={"args": args},
        )

    def log_model_switch(
        self,
        old_provider: str,
        old_model:    str,
        new_provider: str,
        new_model:    str,
        reason:       str = "",
        actor:        str = "system",
    ) -> str:
        desc = f"Model switched from {old_provider}/{old_model} → {new_provider}/{new_model}"
        if reason:
            desc += f" ({reason})"
        log.info("Audit: %s", desc)
        return self._insert(
            event_type="model_switch",
            actor=actor,
            old_value=f"{old_provider}/{old_model}",
            new_value=f"{new_provider}/{new_model}",
            description=desc,
            metadata={"reason": reason},
        )

    def log_config_change(
        self,
        key:       str,
        old_value: str,
        new_value: str,
        actor:     str = "user",
    ) -> str:
        desc = f"Config '{key}' changed from '{old_value}' → '{new_value}'"
        log.info("Audit: %s", desc)
        return self._insert(
            event_type="config_change",
            actor=actor,
            old_value=str(old_value),
            new_value=str(new_value),
            description=desc,
            metadata={"key": key},
        )

    def log_violation(
        self,
        reason:    str,
        severity:  str = "WARN",
        skill_id:  str = "",
        actor:     str = "governance",
    ) -> str:
        desc = f"Governance violation [{severity}]: {reason}"
        log.warning("Audit: %s", desc)
        return self._insert(
            event_type="violation",
            actor=actor,
            skill_id=skill_id or None,
            severity=severity,
            description=desc,
            metadata={"reason": reason},
        )

    # ── Read API ─────────────────────────────────────────────────────

    def tail(self, n: int = 100, event_type: str = "") -> list[dict]:
        with self._connect() as conn:
            if event_type:
                rows = conn.execute(
                    "SELECT * FROM audit_logs WHERE event_type=? ORDER BY occurred_at DESC LIMIT ?",
                    (event_type, n),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM audit_logs ORDER BY occurred_at DESC LIMIT ?",
                    (n,),
                ).fetchall()
        return [self._row_to_dict(r) for r in reversed(rows)]

    def since(self, ts: float, event_type: str = "") -> list[dict]:
        ts_ms = int(ts * 1000)
        with self._connect() as conn:
            if event_type:
                rows = conn.execute(
                    "SELECT * FROM audit_logs WHERE occurred_at >= ? AND event_type=? ORDER BY occurred_at ASC",
                    (ts_ms, event_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM audit_logs WHERE occurred_at >= ? ORDER BY occurred_at ASC",
                    (ts_ms,),
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def stats(self) -> dict:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]
            by_type = dict(conn.execute(
                "SELECT event_type, COUNT(*) FROM audit_logs GROUP BY event_type"
            ).fetchall())
            violations = conn.execute(
                "SELECT COUNT(*) FROM audit_logs WHERE event_type='violation'"
            ).fetchone()[0]
            denials = conn.execute(
                "SELECT COUNT(*) FROM audit_logs WHERE event_type='approval' AND granted=0"
            ).fetchone()[0]
        return {
            "total": total, "by_type": by_type,
            "violations": violations, "denials": denials,
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d.pop("metadata_json", "{}") or "{}")
        except json.JSONDecodeError:
            d["metadata"] = {}
        d["occurred_at_iso"] = _ms_to_iso(d["occurred_at"])
        return d

    # ── Export ───────────────────────────────────────────────────────

    def export_compliance_report(
        self,
        fmt:       str   = "json",
        since_ts:  float = 0.0,
        event_type: str  = "",
    ) -> str:
        """
        Export audit records as JSON or CSV string.

        Parameters
        ----------
        fmt        : "json" | "csv"
        since_ts   : unix epoch float — only records after this timestamp
        event_type : filter to one event type (empty = all)
        """
        rows = self.since(since_ts, event_type) if since_ts else self.tail(n=100_000, event_type=event_type)
        st   = self.stats()

        if fmt == "json":
            return json.dumps({
                "generated_at": _ms_to_iso(int(time.time() * 1000)),
                "summary":      st,
                "records":      rows,
            }, indent=2, ensure_ascii=False)

        # CSV
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "id", "event_type", "occurred_at", "actor",
            "skill_id", "tool_name", "granted", "old_value",
            "new_value", "severity", "description",
        ])
        for r in rows:
            writer.writerow([
                r.get("id", ""),
                r.get("event_type", ""),
                r.get("occurred_at_iso", ""),
                r.get("actor", ""),
                r.get("skill_id", ""),
                r.get("tool_name", ""),
                r.get("granted", ""),
                r.get("old_value", ""),
                r.get("new_value", ""),
                r.get("severity", ""),
                r.get("description", ""),
            ])
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ms_to_iso(ms: int) -> str:
    import datetime
    return datetime.datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_DEFAULT_DB = Path(__file__).parent.parent / "data" / "audit_log.db"
_audit_logger: Optional[AuditLogger] = None


def init_audit_logger(db_path: Path | None = None) -> AuditLogger:
    """Initialise the global audit logger. Call once at startup."""
    global _audit_logger
    _audit_logger = AuditLogger(db_path or _DEFAULT_DB)
    return _audit_logger


def get_audit_logger(db_path: Path | None = None) -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(db_path or _DEFAULT_DB)
    return _audit_logger
