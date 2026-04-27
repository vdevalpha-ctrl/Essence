"""essence_ledger.trust_ledger — 3-axis trust ledger (ported from AURA v4.8.0)."""
from __future__ import annotations
import logging, sqlite3, uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger("essence.trust_ledger")

_TIER_ADVANCE = 0.75
_TIER_REGRESS = 0.25
_MAX_TIER     = 4
_MIN_TIER     = 0


class TrustAxis(str, Enum):
    COMPETENCE = "COMPETENCE"
    VALUES     = "VALUES"
    JUDGMENT   = "JUDGMENT"


class TrustEventType(str, Enum):
    ENDORSEMENT      = "ENDORSEMENT"
    OBSERVATION      = "OBSERVATION"
    OVERRIDE         = "OVERRIDE"
    REJECTION        = "REJECTION"
    CORRECTION       = "CORRECTION"
    TRUST_GRANT      = "TRUST_GRANT"
    TRUST_REVOKE     = "TRUST_REVOKE"
    SEVERE_VIOLATION = "SEVERE_VIOLATION"


_DELTAS: dict[TrustEventType, float] = {
    TrustEventType.ENDORSEMENT:      +0.08,
    TrustEventType.OBSERVATION:      +0.02,
    TrustEventType.OVERRIDE:         -0.10,
    TrustEventType.REJECTION:        -0.15,
    TrustEventType.CORRECTION:       -0.08,
    TrustEventType.TRUST_GRANT:      +0.25,
    TrustEventType.TRUST_REVOKE:     -0.30,
    TrustEventType.SEVERE_VIOLATION: -1.00,
}

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS domain_trust (
    domain TEXT NOT NULL, axis TEXT NOT NULL, tier INTEGER NOT NULL DEFAULT 0,
    score REAL NOT NULL DEFAULT 0.0, event_count INTEGER NOT NULL DEFAULT 0,
    never_graduate INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL,
    PRIMARY KEY (domain, axis)
);
CREATE TABLE IF NOT EXISTS trust_events (
    event_id TEXT PRIMARY KEY, domain TEXT NOT NULL, axis TEXT NOT NULL,
    event_type TEXT NOT NULL, delta REAL NOT NULL DEFAULT 0.0,
    description TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '',
    action_ref TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_te_domain  ON trust_events (domain, axis);
CREATE INDEX IF NOT EXISTS idx_te_created ON trust_events (created_at);
"""


class TrustLedger:
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

    def get_domain_trust(self, domain: str) -> dict[str, Any]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM domain_trust WHERE domain=?", (domain,)).fetchall()
        result: dict[str, Any] = {"domain": domain}
        for row in rows:
            result[row["axis"].lower()] = {
                "tier": row["tier"], "score": round(row["score"], 4),
                "event_count": row["event_count"], "never_graduate": bool(row["never_graduate"]),
            }
        for axis in TrustAxis:
            if axis.value.lower() not in result:
                result[axis.value.lower()] = {"tier": 0, "score": 0.0, "event_count": 0, "never_graduate": False}
        return result

    def record_event(self, *, domain: str, axis: TrustAxis, event_type: TrustEventType,
                     description: str = "", session_id: str = "",
                     action_ref: str | None = None, delta_override: float = 0.0) -> dict[str, Any]:
        now   = datetime.now(timezone.utc).isoformat()
        delta = delta_override if delta_override != 0.0 else _DELTAS.get(event_type, 0.0)
        eid   = str(uuid.uuid4())
        with self._conn() as c:
            c.execute("""INSERT INTO domain_trust (domain,axis,tier,score,event_count,never_graduate,updated_at)
                VALUES (?,?,0,0.0,0,0,?) ON CONFLICT(domain,axis) DO NOTHING""",
                (domain, axis.value, now))
            if event_type == TrustEventType.SEVERE_VIOLATION:
                c.execute("UPDATE domain_trust SET never_graduate=1,score=0.0,tier=0,updated_at=? WHERE domain=? AND axis=?",
                          (now, domain, axis.value))
            else:
                c.execute("""UPDATE domain_trust
                    SET score=MAX(0.0,MIN(1.0,score+?)),event_count=event_count+1,updated_at=?
                    WHERE domain=? AND axis=? AND never_graduate=0""",
                    (delta, now, domain, axis.value))
            c.execute("""INSERT INTO trust_events
                (event_id,domain,axis,event_type,delta,description,session_id,action_ref,created_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (eid,domain,axis.value,event_type.value,delta,description,session_id,action_ref,now))
        self._check_tier(domain, axis)
        return self.get_domain_trust(domain)

    def _check_tier(self, domain: str, axis: TrustAxis) -> None:
        with self._conn() as c:
            row = c.execute("SELECT tier,score,never_graduate FROM domain_trust WHERE domain=? AND axis=?",
                            (domain, axis.value)).fetchone()
            if not row or row["never_graduate"]: return
            tier, score = row["tier"], row["score"]
            now = datetime.now(timezone.utc).isoformat()
            if score >= _TIER_ADVANCE and tier < _MAX_TIER:
                c.execute("UPDATE domain_trust SET tier=?,score=0.5,updated_at=? WHERE domain=? AND axis=?",
                          (tier+1, now, domain, axis.value))
                log.info("Trust advance: %s/%s T%d→T%d", domain, axis.value, tier, tier+1)
            elif score <= _TIER_REGRESS and tier > _MIN_TIER:
                c.execute("UPDATE domain_trust SET tier=?,score=0.5,updated_at=? WHERE domain=? AND axis=?",
                          (tier-1, now, domain, axis.value))
                log.warning("Trust regress: %s/%s T%d→T%d", domain, axis.value, tier, tier-1)

    def recent_events(self, domain: str, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM trust_events WHERE domain=? ORDER BY created_at DESC LIMIT ?",
                             (domain, limit)).fetchall()
        return [dict(r) for r in rows]

    def list_all_domains(self) -> list[str]:
        with self._conn() as c:
            rows = c.execute("SELECT DISTINCT domain FROM domain_trust ORDER BY domain").fetchall()
        return [r["domain"] for r in rows]

    def is_domain_blocked(self, domain: str, axis: TrustAxis) -> bool:
        with self._conn() as c:
            row = c.execute("SELECT tier,never_graduate FROM domain_trust WHERE domain=? AND axis=?",
                            (domain, axis.value)).fetchone()
        if not row: return False
        return bool(row["never_graduate"]) or row["tier"] == 0
