"""
action_engine.py — Essence Pattern-Based Action Engine
=======================================================
Implements the proactive action layer per spec §6:

  - Pattern registry: observing → suggested → active → archived lifecycle
  - 3-tier autonomy: silent (G < 0.50), suggest (0.50–0.79), act (≥ 0.80)
  - Trigger types: time/cron, event (bus topic), condition (callable), webhook
  - Heartbeat: publishes heartbeat.tick every interval; drives time-based triggers
  - Quiet window: no proactive actions during configured hours
  - Intent mapper: keyword fast-path, semantic similarity, LLM fallback
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Awaitable

log = logging.getLogger("essence.action_engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUTONOMY_SILENT   = 0.50   # G < this → never fires without explicit confirm
AUTONOMY_SUGGEST  = 0.80   # 0.50 ≤ G < this → surface suggestion
# G ≥ AUTONOMY_ACT  → act autonomously
AUTONOMY_ACT      = AUTONOMY_SUGGEST

HEARTBEAT_INTERVAL = 60.0  # seconds between heartbeat.tick events
MIN_CONFIDENCE     = 0.40  # patterns below this are culled during promotion


# ---------------------------------------------------------------------------
# Pattern states
# ---------------------------------------------------------------------------

class PatternState:
    OBSERVING  = "observing"   # collecting data, not yet actionable
    SUGGESTED  = "suggested"   # surfaced to user, awaiting confirm
    ACTIVE     = "active"      # fires autonomously
    ARCHIVED   = "archived"    # retired / superseded


# ---------------------------------------------------------------------------
# Trigger types
# ---------------------------------------------------------------------------

class TriggerType:
    TIME      = "time"       # cron-style schedule
    EVENT     = "event"      # bus topic match
    CONDITION = "condition"  # Python callable returning bool
    WEBHOOK   = "webhook"    # external HTTP callback (not yet wired)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS patterns (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    trigger_type TEXT NOT NULL,
    trigger_spec TEXT NOT NULL,     -- JSON: cron string / topic / condition name
    action_spec  TEXT NOT NULL,     -- JSON: what to do when fired
    state        TEXT NOT NULL DEFAULT 'observing',
    gravity      REAL NOT NULL DEFAULT 0.50,
    fire_count   INTEGER NOT NULL DEFAULT 0,
    last_fired   INTEGER,
    created_at   INTEGER NOT NULL,
    updated_at   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pat_state ON patterns(state);
CREATE INDEX IF NOT EXISTS idx_pat_type  ON patterns(trigger_type, state);
"""

_MIGRATIONS: list[str] = []  # future schema migrations


# ---------------------------------------------------------------------------
# Pattern dataclass
# ---------------------------------------------------------------------------

@dataclass
class Pattern:
    id:           str
    name:         str
    description:  str
    trigger_type: str
    trigger_spec: dict
    action_spec:  dict
    state:        str  = PatternState.OBSERVING
    gravity:      float = 0.50
    fire_count:   int   = 0
    last_fired:   int | None = None
    created_at:   int   = field(default_factory=lambda: int(time.time()))
    updated_at:   int   = field(default_factory=lambda: int(time.time()))

    def to_row(self) -> tuple:
        return (
            self.id, self.name, self.description, self.trigger_type,
            json.dumps(self.trigger_spec), json.dumps(self.action_spec),
            self.state, self.gravity, self.fire_count,
            self.last_fired, self.created_at, self.updated_at,
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Pattern":
        d = dict(row)
        return cls(
            id=d["id"], name=d["name"], description=d["description"],
            trigger_type=d["trigger_type"],
            trigger_spec=json.loads(d["trigger_spec"] or "{}"),
            action_spec=json.loads(d["action_spec"] or "{}"),
            state=d["state"], gravity=d["gravity"],
            fire_count=d["fire_count"], last_fired=d.get("last_fired"),
            created_at=d["created_at"], updated_at=d["updated_at"],
        )


# ---------------------------------------------------------------------------
# Intent mapper — 3-pass resolution
# ---------------------------------------------------------------------------

class IntentMapper:
    """
    3-pass intent resolution:
      Pass 1 (~70%): keyword matching (fast path)
      Pass 2: basic semantic similarity (token overlap)
      Pass 3: LLM fallback (not wired — returns None)
    """

    def __init__(self, keyword_map: dict[str, str] | None = None) -> None:
        self._kw: dict[str, str] = keyword_map or {}

    def register_keyword(self, keyword: str, intent: str) -> None:
        self._kw[keyword.lower()] = intent

    def resolve(self, text: str) -> str | None:
        lower = text.lower()

        # Pass 1: exact keyword match
        for kw, intent in self._kw.items():
            if kw in lower:
                return intent

        # Pass 2: token overlap (simple bag-of-words)
        tokens = set(lower.split())
        best_intent, best_score = None, 0
        for kw, intent in self._kw.items():
            kw_tokens = set(kw.split())
            overlap = len(tokens & kw_tokens) / max(len(kw_tokens), 1)
            if overlap > best_score:
                best_score, best_intent = overlap, intent

        if best_score >= 0.5:
            return best_intent

        # Pass 3: LLM decomposition is handled by the kernel planner (_llm_plan),
        # not here. IntentMapper returns None to signal "no fast-path match";
        # the kernel then escalates to its own LLM planning pass.
        return None


# ---------------------------------------------------------------------------
# Pattern registry (SQLite-backed)
# ---------------------------------------------------------------------------

class PatternRegistry:
    """Thread-safe SQLite store for all patterns."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._lock = Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_DDL)
            for stmt in _MIGRATIONS:
                try:
                    conn.execute(stmt)
                    conn.commit()
                except sqlite3.OperationalError:
                    pass

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def upsert(self, p: Pattern) -> Pattern:
        now = int(time.time())
        p.updated_at = now
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO patterns
                       (id,name,description,trigger_type,trigger_spec,action_spec,
                        state,gravity,fire_count,last_fired,created_at,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET
                         name=excluded.name, description=excluded.description,
                         trigger_type=excluded.trigger_type,
                         trigger_spec=excluded.trigger_spec,
                         action_spec=excluded.action_spec,
                         state=excluded.state, gravity=excluded.gravity,
                         fire_count=excluded.fire_count, last_fired=excluded.last_fired,
                         updated_at=excluded.updated_at""",
                    p.to_row(),
                )
                conn.commit()
        return p

    def get(self, pattern_id: str) -> Pattern | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM patterns WHERE id = ?", (pattern_id,)).fetchone()
        return Pattern.from_row(row) if row else None

    def list_by_state(self, state: str) -> list[Pattern]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM patterns WHERE state = ? ORDER BY gravity DESC",
                (state,),
            ).fetchall()
        return [Pattern.from_row(r) for r in rows]

    def list_by_trigger(self, trigger_type: str, state: str = PatternState.ACTIVE) -> list[Pattern]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM patterns WHERE trigger_type = ? AND state = ? ORDER BY gravity DESC",
                (trigger_type, state),
            ).fetchall()
        return [Pattern.from_row(r) for r in rows]

    def transition(self, pattern_id: str, new_state: str) -> Pattern | None:
        p = self.get(pattern_id)
        if p is None:
            return None
        p.state = new_state
        return self.upsert(p)

    def record_fire(self, pattern_id: str) -> None:
        now = int(time.time())
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE patterns SET fire_count = fire_count + 1, last_fired = ?, updated_at = ? WHERE id = ?",
                    (now, now, pattern_id),
                )
                conn.commit()

    def promote_observing(self, min_gravity: float = MIN_CONFIDENCE) -> int:
        """Promote observing→suggested for patterns above min_gravity."""
        patterns = self.list_by_state(PatternState.OBSERVING)
        count = 0
        for p in patterns:
            if p.gravity >= min_gravity:
                self.transition(p.id, PatternState.SUGGESTED)
                count += 1
        return count

    def archive_low_gravity(self, threshold: float = 0.20) -> int:
        """Archive active patterns that have decayed below threshold."""
        patterns = self.list_by_state(PatternState.ACTIVE)
        count = 0
        for p in patterns:
            if p.gravity < threshold:
                self.transition(p.id, PatternState.ARCHIVED)
                count += 1
        return count

    def all_patterns(self) -> list[Pattern]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM patterns ORDER BY updated_at DESC").fetchall()
        return [Pattern.from_row(r) for r in rows]


# ---------------------------------------------------------------------------
# Action Engine
# ---------------------------------------------------------------------------

class ActionEngine:
    """
    Proactive action engine — listens on the event bus, fires patterns,
    manages the heartbeat tick, and enforces the 3-tier autonomy model.
    """

    def __init__(
        self,
        registry: PatternRegistry,
        event_bus: Any,
        quiet_window: tuple[int, int] = (22, 7),  # (start_hour, end_hour) local time
    ) -> None:
        self._reg = registry
        self._bus = event_bus
        self._quiet_start, self._quiet_end = quiet_window
        self._intent_mapper = IntentMapper()
        self._condition_registry: dict[str, Callable[[], bool]] = {}
        self._suggest_callbacks: list[Callable[[Pattern], Awaitable[None]]] = []
        self._act_callbacks:     list[Callable[[Pattern], Awaitable[None]]] = []
        self._heartbeat_task: asyncio.Task | None = None

    # ── Quiet window ─────────────────────────────────────────────────

    def _in_quiet_window(self) -> bool:
        hour = datetime.now().hour
        s, e = self._quiet_start, self._quiet_end
        if s < e:
            return s <= hour < e
        return hour >= s or hour < e  # wraps midnight

    # ── Condition registry ────────────────────────────────────────────

    def register_condition(self, name: str, fn: Callable[[], bool]) -> None:
        self._condition_registry[name] = fn

    def register_suggest_callback(self, fn: Callable[[Pattern], Awaitable[None]]) -> None:
        self._suggest_callbacks.append(fn)

    def register_act_callback(self, fn: Callable[[Pattern], Awaitable[None]]) -> None:
        self._act_callbacks.append(fn)

    # ── Pattern CRUD ─────────────────────────────────────────────────

    def add_pattern(
        self,
        name: str,
        trigger_type: str,
        trigger_spec: dict,
        action_spec: dict,
        description: str = "",
        gravity: float = 0.50,
    ) -> Pattern:
        p = Pattern(
            id=uuid.uuid4().hex,
            name=name,
            description=description,
            trigger_type=trigger_type,
            trigger_spec=trigger_spec,
            action_spec=action_spec,
            gravity=gravity,
        )
        return self._reg.upsert(p)

    def confirm_pattern(self, pattern_id: str) -> Pattern | None:
        """User confirmed a suggested pattern — promote to active."""
        p = self._reg.get(pattern_id)
        if p and p.state == PatternState.SUGGESTED:
            p.state = PatternState.ACTIVE
            if p.gravity < AUTONOMY_SUGGEST:
                p.gravity = AUTONOMY_SUGGEST
            return self._reg.upsert(p)
        return p

    def dismiss_pattern(self, pattern_id: str) -> Pattern | None:
        """User dismissed a suggested pattern — archive it."""
        return self._reg.transition(pattern_id, PatternState.ARCHIVED)

    def list_patterns(self, state: str | None = None) -> list[Pattern]:
        if state:
            return self._reg.list_by_state(state)
        return self._reg.all_patterns()

    # ── Autonomy dispatch ─────────────────────────────────────────────

    async def _dispatch(self, pattern: Pattern) -> None:
        """Route a triggered pattern through the 3-tier autonomy model."""
        if self._in_quiet_window():
            log.debug("Quiet window — suppressing pattern %s", pattern.name)
            return

        if pattern.gravity < AUTONOMY_SILENT:
            # Tier 1: silent — record but don't surface
            self._reg.record_fire(pattern.id)
            return

        if pattern.gravity < AUTONOMY_ACT:
            # Tier 2: suggest — surface to user
            self._reg.record_fire(pattern.id)
            for cb in self._suggest_callbacks:
                try:
                    await cb(pattern)
                except Exception as exc:
                    log.warning("suggest callback error on %s: %s", pattern.name, exc)
        else:
            # Tier 3: act autonomously
            self._reg.record_fire(pattern.id)
            for cb in self._act_callbacks:
                try:
                    await cb(pattern)
                except Exception as exc:
                    log.warning("act callback error on %s: %s", pattern.name, exc)

    # ── Event-triggered patterns ──────────────────────────────────────

    async def _on_bus_event(self, env: Any) -> None:
        """Bus subscriber — check all event-triggered active patterns."""
        patterns = self._reg.list_by_trigger(TriggerType.EVENT)
        for p in patterns:
            topic_filter = p.trigger_spec.get("topic", "")
            if topic_filter and env.topic != topic_filter:
                continue
            await self._dispatch(p)

    async def _check_conditions(self) -> None:
        """Evaluate condition-triggered patterns (called on heartbeat)."""
        patterns = self._reg.list_by_trigger(TriggerType.CONDITION)
        for p in patterns:
            cond_name = p.trigger_spec.get("condition", "")
            fn = self._condition_registry.get(cond_name)
            if fn is None:
                continue
            try:
                if fn():
                    await self._dispatch(p)
            except Exception as exc:
                log.warning("Condition %r error: %s", cond_name, exc)

    # ── Heartbeat ─────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Publish heartbeat.tick every HEARTBEAT_INTERVAL seconds."""
        from server.event_bus import Envelope
        while True:
            try:
                env = Envelope(
                    topic="heartbeat.tick",
                    source_component="action_engine",
                    data={"ts": time.time(), "patterns_active": len(self._reg.list_by_state(PatternState.ACTIVE))},
                )
                await self._bus.publish(env)

                # Housekeeping on each tick
                self._reg.promote_observing()
                self._reg.archive_low_gravity()
                await self._check_conditions()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Heartbeat loop error: %s", exc)

            await asyncio.sleep(HEARTBEAT_INTERVAL)

    def start(self) -> None:
        """Wire bus subscriptions and start the heartbeat task."""
        self._bus.subscribe("user.request",    self._on_bus_event)
        self._bus.subscribe("skill.result",    self._on_bus_event)
        self._bus.subscribe("trigger.fired",   self._on_bus_event)
        self._bus.subscribe("heartbeat.tick",  self._on_bus_event)

        try:
            loop = asyncio.get_running_loop()
            self._heartbeat_task = loop.create_task(
                self._heartbeat_loop(), name="action-engine-heartbeat"
            )
        except RuntimeError:
            log.warning("ActionEngine.start() called outside asyncio loop — heartbeat not started")

    def stop(self) -> None:
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "patterns.db"
_engine: ActionEngine | None = None


def init_action_engine(event_bus: Any, db_path: Path | None = None) -> ActionEngine:
    """Initialise and start the global action engine. Call once at startup."""
    global _engine
    registry = PatternRegistry(db_path or _DEFAULT_DB)
    _engine = ActionEngine(registry, event_bus)
    return _engine


def get_action_engine() -> ActionEngine:
    if _engine is None:
        raise RuntimeError("Action engine not initialised. Call init_action_engine() at startup.")
    return _engine
