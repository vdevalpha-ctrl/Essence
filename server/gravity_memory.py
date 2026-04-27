"""
gravity_memory.py — Contextual Gravity Memory System for Essence
================================================================
Implements the Gravity Score formula:
    G = (frequency * 0.30) + (recency * 0.25) + (user_signal * 0.30) + (emotional_wt * 0.15)

Persistence Tiers:
    volatile  — session-only, all entries auto-added
    weighted  — G >= 0.40, 30-day TTL, -0.08 decay per 7 days
    anchored  — G >= 0.85 or user pins, permanent
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

log = logging.getLogger("essence.memory")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHTED_THRESHOLD = 0.40
ANCHORED_THRESHOLD = 0.85
WEIGHTED_TTL_DAYS = 30
DECAY_PER_7_DAYS = 0.08
DECAY_INTERVAL_SECONDS = 7 * 24 * 3600

# Signal addends applied directly to G (spec §5.2)
# confirmed → +0.20, corrected → +0.15, neutral → 0.0, ignored → -0.05
SIGNAL_ADDENDS: dict[str, float] = {
    "confirmed": 0.20,
    "corrected": 0.15,
    "neutral": 0.0,
    "ignored": -0.05,
}

VALID_SIGNALS = set(SIGNAL_ADDENDS.keys())


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    tier TEXT NOT NULL DEFAULT 'volatile',
    skill_source TEXT,
    topic_tags TEXT,          -- JSON array
    gravity REAL DEFAULT 0.5,
    frequency INTEGER DEFAULT 1,
    last_referenced INTEGER,
    user_signal TEXT DEFAULT 'neutral',  -- confirmed|corrected|ignored|neutral
    emotional_weight REAL DEFAULT 0.3,
    data_schema TEXT DEFAULT '',
    created_at INTEGER NOT NULL,
    expires_at INTEGER,
    is_archived INTEGER DEFAULT 0,
    is_pinned INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS memory_corrections (
    id TEXT PRIMARY KEY,
    memory_id TEXT REFERENCES memories(id),
    old_value TEXT,
    new_value TEXT,
    corrected_at INTEGER
);
"""


_MIGRATIONS = [
    "ALTER TABLE memories ADD COLUMN data_schema TEXT DEFAULT ''",
]


# ---------------------------------------------------------------------------
# Helper: row → dict
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    if d.get("topic_tags"):
        try:
            d["topic_tags"] = json.loads(d["topic_tags"])
        except (json.JSONDecodeError, TypeError):
            d["topic_tags"] = []
    else:
        d["topic_tags"] = []
    d["is_archived"] = bool(d.get("is_archived", 0))
    d["is_pinned"] = bool(d.get("is_pinned", 0))
    return d


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class GravityMemory:
    """SQLite-backed Contextual Gravity Memory for Essence."""

    def __init__(self, db_path: Path, event_bus: Any = None) -> None:
        self.db_path = db_path
        self._lock = Lock()
        self._bus = event_bus  # optional EventBus; set after bus init if needed
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_DDL)
            for stmt in _MIGRATIONS:
                try:
                    conn.execute(stmt)
                    conn.commit()
                except sqlite3.OperationalError:
                    pass  # column already exists

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _now(self) -> int:
        return int(time.time())

    def _calc_gravity(
        self,
        frequency: int,
        last_referenced: int | None,
        user_signal: str,
        emotional_weight: float,
        created_at: int | None = None,
    ) -> float:
        """
        G = (frequency × 0.30) + (recency × 0.25) + (emotional_wt × 0.15) + signal_addend

        frequency  — normalized with a soft cap: min(freq / 20, 1.0)
        recency    — exponential decay: exp(-hours_since_ref / 168)  (168 h = 1 week half-life)
        signal_addend — confirmed +0.20, corrected +0.15, neutral 0.0, ignored −0.05 (spec §5.2)
        emotional_weight — passed through directly [0, 1]
        """
        now = self._now()

        freq_norm = min(frequency / 20.0, 1.0)

        ref_ts = last_referenced or created_at or now
        hours_since = max((now - ref_ts) / 3600.0, 0.0)
        recency_norm = math.exp(-hours_since / 168.0)

        signal_add = SIGNAL_ADDENDS.get(user_signal, 0.0)
        emo_norm = max(0.0, min(1.0, emotional_weight))

        g = (
            freq_norm * 0.30
            + recency_norm * 0.25
            + emo_norm * 0.15
            + signal_add
        )
        return round(min(max(g, 0.0), 1.0), 6)

    def _resolve_tier(self, gravity: float, is_pinned: bool) -> str:
        if is_pinned or gravity >= ANCHORED_THRESHOLD:
            return "anchored"
        if gravity >= WEIGHTED_THRESHOLD:
            return "weighted"
        return "volatile"

    def _expires_at(self, tier: str) -> int | None:
        if tier == "weighted":
            return self._now() + WEIGHTED_TTL_DAYS * 86400
        return None

    def _get_by_key(self, key: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE key = ? AND is_archived = 0 ORDER BY created_at DESC LIMIT 1",
                (key,),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def _get_by_id(self, memory_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def _update_gravity_and_tier(self, memory_id: str, mem: dict) -> dict:
        g = self._calc_gravity(
            frequency=mem["frequency"],
            last_referenced=mem.get("last_referenced"),
            user_signal=mem.get("user_signal", "neutral"),
            emotional_weight=mem.get("emotional_weight", 0.3),
            created_at=mem.get("created_at"),
        )
        is_pinned = bool(mem.get("is_pinned", False))
        tier = self._resolve_tier(g, is_pinned)
        expires_at = self._expires_at(tier)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE memories SET gravity = ?, tier = ?, expires_at = ? WHERE id = ?",
                    (g, tier, expires_at, memory_id),
                )
                conn.commit()
        mem["gravity"] = g
        mem["tier"] = tier
        mem["expires_at"] = expires_at
        return mem

    def _publish_write(self, mem: dict) -> None:
        """Fire memory.write event on the bus if one is wired up."""
        if self._bus is None:
            return
        try:
            from server.event_bus import Envelope
            env = Envelope(
                topic="memory.write",
                source_component="gravity_memory",
                data={"key": mem.get("key"), "tier": mem.get("tier"), "gravity": mem.get("gravity")},
                task_id="",
            )
            self._bus.publish_sync(env)
        except Exception as exc:
            log.debug("memory.write bus publish skipped: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(
        self,
        key: str,
        value: str,
        skill_source: str = "",
        tags: list | None = None,
        emotional_weight: float = 0.3,
        data_schema: str = "",
    ) -> dict:
        """Write a new memory entry (always starts as volatile)."""
        now = self._now()
        mem_id = str(uuid.uuid4())
        emo = max(0.0, min(1.0, emotional_weight))
        g = self._calc_gravity(
            frequency=1, last_referenced=now, user_signal="neutral",
            emotional_weight=emo, created_at=now,
        )
        tier = self._resolve_tier(g, False)
        expires_at = self._expires_at(tier)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO memories
                       (id, key, value, tier, skill_source, topic_tags, gravity,
                        frequency, last_referenced, user_signal, emotional_weight,
                        data_schema, created_at, expires_at, is_archived, is_pinned)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)""",
                    (
                        mem_id, key, value, tier, skill_source or "",
                        json.dumps(tags or []), g, 1, now, "neutral", emo,
                        data_schema, now, expires_at,
                    ),
                )
                conn.commit()
        result = self._get_by_id(mem_id)
        self._publish_write(result or {})
        return result  # type: ignore[return-value]

    def get(self, key: str) -> dict | None:
        """Retrieve a memory by key (most recent non-archived)."""
        return self._get_by_key(key)

    def reference(self, key: str) -> dict:
        """Increment frequency and last_referenced, then recalculate G."""
        mem = self._get_by_key(key)
        if not mem:
            raise KeyError(f"Memory key not found: {key!r}")
        now = self._now()
        new_freq = mem["frequency"] + 1
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE memories SET frequency = ?, last_referenced = ? WHERE id = ?",
                    (new_freq, now, mem["id"]),
                )
                conn.commit()
        mem["frequency"] = new_freq
        mem["last_referenced"] = now
        return self._update_gravity_and_tier(mem["id"], mem)

    def signal(self, key: str, signal: str) -> dict:
        """Update user_signal for a memory (confirmed|corrected|ignored|neutral)."""
        if signal not in VALID_SIGNALS:
            raise ValueError(f"Invalid signal {signal!r}. Must be one of: {VALID_SIGNALS}")
        mem = self._get_by_key(key)
        if not mem:
            raise KeyError(f"Memory key not found: {key!r}")
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE memories SET user_signal = ? WHERE id = ?",
                    (signal, mem["id"]),
                )
                conn.commit()
        mem["user_signal"] = signal
        return self._update_gravity_and_tier(mem["id"], mem)

    def correct(self, key: str, new_value: str) -> dict:
        """Archive the current memory and write a corrected version."""
        old_mem = self._get_by_key(key)
        if not old_mem:
            raise KeyError(f"Memory key not found: {key!r}")
        now = self._now()
        corr_id = str(uuid.uuid4())
        with self._lock:
            with self._connect() as conn:
                conn.execute("UPDATE memories SET is_archived = 1 WHERE id = ?", (old_mem["id"],))
                conn.execute(
                    "INSERT INTO memory_corrections (id, memory_id, old_value, new_value, corrected_at) VALUES (?, ?, ?, ?, ?)",
                    (corr_id, old_mem["id"], old_mem["value"], new_value, now),
                )
                conn.commit()
        return self.write(
            key=key, value=new_value,
            skill_source=old_mem.get("skill_source", ""),
            tags=old_mem.get("topic_tags", []),
            emotional_weight=old_mem.get("emotional_weight", 0.3),
        )

    def pin(self, key: str) -> dict:
        """Pin a memory to anchored tier permanently."""
        mem = self._get_by_key(key)
        if not mem:
            raise KeyError(f"Memory key not found: {key!r}")
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE memories SET is_pinned = 1, tier = 'anchored', expires_at = NULL WHERE id = ?",
                    (mem["id"],),
                )
                conn.commit()
        mem["is_pinned"] = True
        mem["tier"] = "anchored"
        mem["expires_at"] = None
        return self._update_gravity_and_tier(mem["id"], mem)

    def expire(self, key: str) -> bool:
        """Soft-delete (archive) a memory. Returns True if found."""
        mem = self._get_by_key(key)
        if not mem:
            return False
        with self._lock:
            with self._connect() as conn:
                conn.execute("UPDATE memories SET is_archived = 1 WHERE id = ?", (mem["id"],))
                conn.commit()
        return True

    def search(self, query: str, tier: str = "", limit: int = 20) -> list[dict]:
        """Substring search on key + value. Optionally filter by tier."""
        like = f"%{query}%"
        with self._connect() as conn:
            if tier:
                rows = conn.execute(
                    """SELECT * FROM memories WHERE is_archived = 0 AND tier = ?
                       AND (key LIKE ? OR value LIKE ?) ORDER BY gravity DESC LIMIT ?""",
                    (tier, like, like, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM memories WHERE is_archived = 0
                       AND (key LIKE ? OR value LIKE ?) ORDER BY gravity DESC LIMIT ?""",
                    (like, like, limit),
                ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def top_by_gravity(self, n: int = 10, tier: str = "") -> list[dict]:
        """Return top N non-archived memories sorted by gravity descending."""
        with self._connect() as conn:
            if tier:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE is_archived = 0 AND tier = ? ORDER BY gravity DESC LIMIT ?",
                    (tier, n),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE is_archived = 0 ORDER BY gravity DESC LIMIT ?",
                    (n,),
                ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def decay_tick(self) -> int:
        """Decay weighted-tier entries not referenced in >= 7 days by -0.08 G."""
        cutoff = self._now() - DECAY_INTERVAL_SECONDS
        with self._connect() as conn:
            rows = [_row_to_dict(r) for r in conn.execute(
                """SELECT * FROM memories WHERE is_archived = 0 AND tier = 'weighted'
                   AND (last_referenced IS NULL OR last_referenced <= ?)""",
                (cutoff,),
            ).fetchall()]
        count = 0
        with self._lock:
            with self._connect() as conn:
                for mem in rows:
                    new_g = round(max(mem["gravity"] - DECAY_PER_7_DAYS, 0.0), 6)
                    new_tier = self._resolve_tier(new_g, bool(mem.get("is_pinned")))
                    conn.execute(
                        "UPDATE memories SET gravity = ?, tier = ?, expires_at = ? WHERE id = ?",
                        (new_g, new_tier, self._expires_at(new_tier), mem["id"]),
                    )
                    count += 1
                if count:
                    conn.commit()
        return count

    def promote_session(self) -> int:
        """Promote volatile entries with G >= WEIGHTED_THRESHOLD to weighted tier."""
        with self._connect() as conn:
            rows = [_row_to_dict(r) for r in conn.execute(
                "SELECT * FROM memories WHERE is_archived = 0 AND tier = 'volatile' AND gravity >= ?",
                (WEIGHTED_THRESHOLD,),
            ).fetchall()]
        count = 0
        with self._lock:
            with self._connect() as conn:
                for mem in rows:
                    tier = self._resolve_tier(mem["gravity"], bool(mem.get("is_pinned")))
                    if tier == "volatile":
                        continue
                    conn.execute(
                        "UPDATE memories SET tier = ?, expires_at = ? WHERE id = ?",
                        (tier, self._expires_at(tier), mem["id"]),
                    )
                    count += 1
                if count:
                    conn.commit()
        return count

    # ── Multi-signal retrieval (BM25 + entity + gravity fusion) ──────

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Lower-case word tokeniser (strips punctuation)."""
        return re.findall(r"[a-zA-Z0-9_\-]{2,}", text.lower())

    @staticmethod
    def _extract_entities(text: str) -> set[str]:
        """Simple entity heuristic: capitalised words, quoted strings, digits."""
        entities: set[str] = set()
        # Capitalised words (not at sentence start)
        for m in re.finditer(r'\b[A-Z][a-z]{2,}\b', text):
            entities.add(m.group().lower())
        # Quoted strings
        for m in re.finditer(r'["\']([^"\']{3,30})["\']', text):
            entities.add(m.group(1).lower())
        # Numbers that might be identifiers
        for m in re.finditer(r'\b\d[\d.:\-]{1,}\b', text):
            entities.add(m.group())
        return entities

    def multi_signal_search(
        self,
        query: str,
        n: int = 10,
        *,
        alpha: float = 0.45,   # BM25 weight
        beta:  float = 0.35,   # gravity weight
        gamma: float = 0.20,   # entity-link weight
        k1: float = 1.5,
        b:  float = 0.75,
    ) -> list[dict]:
        """Retrieve top-N memories using BM25 + gravity + entity-link fusion.

        Inspired by mem0-main multi-signal retrieval.

        Scoring
        -------
          BM25:         token-level frequency-weighted keyword match
          Gravity:      the memory's existing G score (frequency × recency × signal)
          Entity link:  +1 per named entity shared between query and memory content

        Final score = alpha * bm25_norm + beta * gravity + gamma * entity_link_norm

        Falls back to top_by_gravity() if query is empty.
        """
        if not query or not query.strip():
            return self.top_by_gravity(n=n)

        query_tokens = self._tokenise(query)
        query_entities = self._extract_entities(query)
        if not query_tokens:
            return self.top_by_gravity(n=n)

        # Load all active memories (BM25 needs corpus-level stats)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE is_archived = 0 ORDER BY gravity DESC LIMIT 500"
            ).fetchall()
        if not rows:
            return []

        mems = [_row_to_dict(r) for r in rows]

        # Build corpus for IDF
        N = len(mems)
        df: dict[str, int] = {}
        doc_tokens: list[list[str]] = []
        for mem in mems:
            text = f"{mem['key']} {mem['value']}"
            tokens = self._tokenise(text)
            doc_tokens.append(tokens)
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        avg_dl = sum(len(dt) for dt in doc_tokens) / max(N, 1)

        # BM25 scoring per document
        import math as _math
        bm25_scores: list[float] = []
        for i, (mem, tokens) in enumerate(zip(mems, doc_tokens)):
            dl = len(tokens)
            score = 0.0
            tf_map: dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            for qt in query_tokens:
                if qt not in df:
                    continue
                tf = tf_map.get(qt, 0)
                if tf == 0:
                    continue
                idf = _math.log((N - df[qt] + 0.5) / (df[qt] + 0.5) + 1.0)
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
                score += idf * tf_norm
            bm25_scores.append(score)

        # Entity linking bonus
        entity_scores: list[float] = []
        for mem in mems:
            mem_entities = self._extract_entities(f"{mem['key']} {mem['value']}")
            shared = len(query_entities & mem_entities)
            entity_scores.append(float(shared))

        # Normalise BM25 and entity scores to [0, 1]
        max_bm25   = max(bm25_scores)   if max(bm25_scores)   > 0 else 1.0
        max_entity = max(entity_scores) if max(entity_scores) > 0 else 1.0
        bm25_norm   = [s / max_bm25   for s in bm25_scores]
        entity_norm = [s / max_entity for s in entity_scores]

        # Fuse scores
        fused: list[tuple[float, dict]] = []
        for i, mem in enumerate(mems):
            g = float(mem.get("gravity", 0.5))
            final = alpha * bm25_norm[i] + beta * g + gamma * entity_norm[i]
            if final > 0.05:          # discard near-zero matches
                fused.append((final, mem))

        fused.sort(key=lambda x: -x[0])
        return [m for _, m in fused[:n]]

    def build_context_block(self, n: int = 15, query: str = "") -> str:
        """
        Return a formatted string of the top N memories by gravity for
        injection into an LLM context window.

        When *query* is provided, uses multi-signal retrieval (BM25 + entity
        linking + gravity fusion) to surface the most task-relevant memories.
        When *query* is empty, falls back to pure gravity ranking.
        """
        if query:
            memories = self.multi_signal_search(query, n=n)
        else:
            memories = self.top_by_gravity(n=n)

        if not memories:
            return "[Memory] No memories available."

        lines: list[str] = ["[Memory Context]"]
        for i, mem in enumerate(memories, 1):
            tags_str = ", ".join(mem.get("topic_tags") or [])
            tag_part = f" [{tags_str}]" if tags_str else ""
            lines.append(
                f"{i:>2}. ({mem['tier']}, G={mem['gravity']:.3f}){tag_part}"
                f"\n    {mem['key']}: {mem['value']}"
            )
        return "\n".join(lines)

    def stats(self) -> dict:
        """Return aggregate statistics about the memory store."""
        with self._connect() as conn:
            row = dict(conn.execute(
                """SELECT
                     COUNT(*) AS total,
                     SUM(CASE WHEN is_archived = 0 THEN 1 ELSE 0 END) AS active,
                     SUM(CASE WHEN is_archived = 1 THEN 1 ELSE 0 END) AS archived,
                     SUM(CASE WHEN tier = 'volatile' AND is_archived = 0 THEN 1 ELSE 0 END) AS volatile_count,
                     SUM(CASE WHEN tier = 'weighted' AND is_archived = 0 THEN 1 ELSE 0 END) AS weighted_count,
                     SUM(CASE WHEN tier = 'anchored' AND is_archived = 0 THEN 1 ELSE 0 END) AS anchored_count,
                     SUM(CASE WHEN is_pinned = 1     AND is_archived = 0 THEN 1 ELSE 0 END) AS pinned_count,
                     AVG(CASE WHEN is_archived = 0 THEN gravity ELSE NULL END) AS avg_gravity,
                     MAX(CASE WHEN is_archived = 0 THEN gravity ELSE NULL END) AS max_gravity,
                     MIN(CASE WHEN is_archived = 0 THEN gravity ELSE NULL END) AS min_gravity
                   FROM memories"""
            ).fetchone())
            row["total_corrections"] = conn.execute(
                "SELECT COUNT(*) AS n FROM memory_corrections"
            ).fetchone()["n"]
        for k in ("avg_gravity", "max_gravity", "min_gravity"):
            if row[k] is not None:
                row[k] = round(row[k], 6)
        return row


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_DEFAULT_DB = Path(__file__).parent.parent / "data" / "gravity_memory.db"
_gm_instance: GravityMemory | None = None


def get_gravity_memory() -> GravityMemory:
    global _gm_instance
    if _gm_instance is None:
        _gm_instance = GravityMemory(_DEFAULT_DB)
    return _gm_instance
