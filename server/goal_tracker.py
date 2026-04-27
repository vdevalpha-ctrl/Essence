"""
goal_tracker.py — Multi-Session Goal Persistence
=================================================
Detects incomplete tasks from conversation patterns and resurfaces
them at the start of the next session.

Detection heuristics (end-of-session analysis):
  • Last user message contains task-opening verbs AND
  • No closure signal in the last user OR assistant message
  • OR user explicitly deferred ("remind me", "save this", "tomorrow")

Storage: data/pending_goals.json  (human-readable, editable)

Goals auto-expire after STALE_DAYS days and cap at MAX_GOALS total.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from threading import Lock
from typing import List, Optional

log = logging.getLogger("essence.goal_tracker")

_GOALS_FILE = "data/pending_goals.json"
_MAX_GOALS  = 15
_STALE_DAYS = 7

# ---------------------------------------------------------------------------
# Pattern regexes
# ---------------------------------------------------------------------------

# Task-opening verbs that suggest the user started something
_RE_TASK = re.compile(
    r"\b(build|create|implement|write|fix|debug|make|design|set.?up|configure|"
    r"generate|produce|find|research|analyze|plan|help me|can you|could you|"
    r"I need|I want|we need|let[''']?s|let us|finish|complete|continue|"
    r"refactor|migrate|update|add|remove|deploy|test|review)\b",
    re.I,
)

# Closure signals — if present, the session ended cleanly
_RE_CLOSED = re.compile(
    r"\b(thanks?|thank you|perfect|great|done|finished|got it|that works?|"
    r"that[''']?s (it|all|great|perfect)|never mind|forget it|stop|exit|quit|"
    r"all good|looks good|ship it|merged|deployed|resolved|closed)\b",
    re.I,
)

# Explicit deferral phrases
_RE_DEFERRED = re.compile(
    r"\b(remind me|come back|pick up (later|tomorrow)|next (time|session)|"
    r"save this|bookmark|I[''']?ll|we[''']?ll|will do|TODO|FIXME|"
    r"continue (this|later)|let[''']?s (pause|stop|pick))\b",
    re.I,
)

# Questions that are one-off and don't imply ongoing work
_RE_ONOFF_QUESTION = re.compile(
    r"^(what is|who is|where is|when (did|was|is)|how (many|much|old)|"
    r"define |definition of |what does .* mean)",
    re.I,
)


# ---------------------------------------------------------------------------
# PendingGoal dataclass
# ---------------------------------------------------------------------------

@dataclass
class PendingGoal:
    id:              str   = field(default_factory=lambda: uuid.uuid4().hex[:8])
    text:            str   = ""
    context_snippet: str   = ""   # first 120 chars of assistant's last reply
    task_type:       str   = "general"
    session_id:      str   = ""
    resurface_count: int   = 0
    created_at:      float = field(default_factory=time.time)

    def is_stale(self) -> bool:
        return (time.time() - self.created_at) > (_STALE_DAYS * 86400)

    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PendingGoal":
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# GoalTracker
# ---------------------------------------------------------------------------

class GoalTracker:
    """Detects, stores, and resurfaces incomplete tasks across sessions."""

    def __init__(self, workspace: Path) -> None:
        self._ws    = workspace
        self._lock  = Lock()
        self._goals: List[PendingGoal] = self._load()

    # ── Persistence ────────────────────────────────────────────────────

    def _path(self) -> Path:
        p = self._ws / _GOALS_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _load(self) -> List[PendingGoal]:
        try:
            p = self._path()
            if p.exists():
                raw = json.loads(p.read_text("utf-8"))
                return [PendingGoal.from_dict(d) for d in raw if isinstance(d, dict)]
        except Exception as e:
            log.debug("goal_tracker: load failed: %s", e)
        return []

    def _save(self) -> None:
        try:
            self._path().write_text(
                json.dumps([g.to_dict() for g in self._goals], indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            log.debug("goal_tracker: save failed: %s", e)

    def _prune(self) -> None:
        """Remove stale goals and enforce MAX_GOALS cap. Call under lock."""
        self._goals = [g for g in self._goals if not g.is_stale()]
        if len(self._goals) > _MAX_GOALS:
            # Keep the most recent
            self._goals = sorted(self._goals, key=lambda g: g.created_at)[-_MAX_GOALS:]

    # ── Detection helpers ──────────────────────────────────────────────

    @staticmethod
    def _looks_like_task(text: str) -> bool:
        if _RE_ONOFF_QUESTION.match(text.strip()):
            return False
        return bool(_RE_TASK.search(text))

    @staticmethod
    def _looks_closed(text: str) -> bool:
        return bool(_RE_CLOSED.search(text))

    @staticmethod
    def _looks_deferred(text: str) -> bool:
        return bool(_RE_DEFERRED.search(text))

    @staticmethod
    def _is_duplicate(text: str, existing: List[PendingGoal]) -> bool:
        """Simple duplicate check: first 100 chars match (case-insensitive)."""
        key = text[:100].lower().strip()
        return any(g.text[:100].lower().strip() == key for g in existing)

    # ── Public API ─────────────────────────────────────────────────────

    def analyze_session_end(
        self,
        last_user_msg: str,
        last_assistant_msg: str,
        session_id: str = "",
        task_type: str = "general",
    ) -> Optional[PendingGoal]:
        """
        Analyze the final exchange of a session.
        Creates and persists a PendingGoal if the session appears to have
        ended mid-task.  Returns the goal or None.
        """
        um = (last_user_msg or "").strip()
        am = (last_assistant_msg or "").strip()

        if not um:
            return None

        # Don't flag clean endings
        if self._looks_closed(um) or self._looks_closed(am):
            return None

        is_task     = self._looks_like_task(um)
        is_deferred = self._looks_deferred(um)

        if not (is_task or is_deferred):
            return None

        goal = PendingGoal(
            text=um[:300],
            context_snippet=am[:120],
            task_type=task_type,
            session_id=session_id,
        )

        with self._lock:
            self._prune()
            if self._is_duplicate(um, self._goals):
                log.debug("goal_tracker: duplicate goal suppressed")
                return None
            self._goals.append(goal)
            self._save()

        log.info("goal_tracker: saved pending goal [%s]: %r", goal.id, um[:60])
        return goal

    def add_explicit(
        self,
        text: str,
        session_id: str = "",
        task_type: str = "general",
    ) -> PendingGoal:
        """Manually add a pending goal (e.g. /goals save <text>)."""
        goal = PendingGoal(text=text[:300], session_id=session_id, task_type=task_type)
        with self._lock:
            self._prune()
            self._goals.append(goal)
            self._save()
        return goal

    def get_pending(self, max_resurface: int = 3) -> List[PendingGoal]:
        """Return goals that haven't been resurfaced too many times yet."""
        with self._lock:
            self._prune()
            return [g for g in self._goals if g.resurface_count < max_resurface]

    def mark_resurfaced(self, *goal_ids: str) -> None:
        with self._lock:
            for g in self._goals:
                if g.id in goal_ids:
                    g.resurface_count += 1
            self._save()

    def complete(self, goal_id: str) -> bool:
        """Remove a goal by id. Returns True if found."""
        with self._lock:
            before = len(self._goals)
            self._goals = [g for g in self._goals if g.id != goal_id]
            changed = len(self._goals) < before
            if changed:
                self._save()
        return changed

    def clear_all(self) -> int:
        """Clear all goals. Returns count cleared."""
        with self._lock:
            n = len(self._goals)
            self._goals = []
            self._save()
        return n

    def list_all(self) -> List[PendingGoal]:
        with self._lock:
            self._prune()
            return list(self._goals)

    # ── Formatting ─────────────────────────────────────────────────────

    def format_resurface(self, goals: List[PendingGoal]) -> str:
        """
        Generate a session-start notice for pending goals.
        Returns empty string when there are none.
        """
        if not goals:
            return ""

        if len(goals) == 1:
            g = goals[0]
            age_str = f"{g.age_hours():.0f}h ago"
            short   = g.text[:120] + ("…" if len(g.text) > 120 else "")
            return (
                f"\n💭  Unfinished work from {age_str}:\n"
                f"    \"{short}\"\n"
                f"    /goals done {g.id}  to dismiss  ·  just continue to resume\n"
            )

        lines = [f"\n💭  {len(goals)} pending goal(s) from previous sessions:"]
        for i, g in enumerate(goals[:5], 1):
            age_str = f"{g.age_hours():.0f}h"
            short   = g.text[:72] + ("…" if len(g.text) > 72 else "")
            lines.append(f"    {i}. [{g.id}] ({age_str}) {short}")
        lines.append("    /goals list  ·  /goals done <id>  ·  /goals clear\n")
        return "\n".join(lines)

    def format_list(self) -> str:
        """Format all goals for /goals list."""
        goals = self.list_all()
        if not goals:
            return "No pending goals."
        lines = [f"Pending goals ({len(goals)}):"]
        for g in goals:
            age   = f"{g.age_hours():.0f}h"
            short = g.text[:90] + ("…" if len(g.text) > 90 else "")
            lines.append(f"  [{g.id}]  ({g.task_type}, {age})  {short}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_tracker: Optional[GoalTracker] = None
_tlock   = Lock()


def get_goal_tracker(workspace: Optional[Path] = None) -> GoalTracker:
    """Return the process-level GoalTracker singleton."""
    global _tracker
    if _tracker is None:
        with _tlock:
            if _tracker is None:
                ws = workspace or Path(__file__).resolve().parent.parent
                _tracker = GoalTracker(ws)
    return _tracker
