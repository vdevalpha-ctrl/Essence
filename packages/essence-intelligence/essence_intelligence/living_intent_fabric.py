"""
essence_intelligence.living_intent_fabric
=======================================
Living Intent Fabric (LIF) — ported from AURA v4.8.0.

Unique AURA trait: 4-horizon temporal intent model where every user signal
is classified into IMMEDIATE/NEAR/ARC/LIFE gravity wells, then scored by a
Temporal Gravity Score (TGS). TGS > 1.5 = critical priority.

Horizon weights: IMMEDIATE=2.0, NEAR=1.0, ARC=0.3, LIFE=0.1
Priority score:  urgency × importance × confidence
TGS formula:     Σ(priority_score × horizon_weight) for active intents
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LIFHorizon(str, Enum):
    IMMEDIATE = "IMMEDIATE"   # right now / today
    NEAR      = "NEAR"        # this week
    ARC       = "ARC"         # this month / quarter
    LIFE      = "LIFE"        # multi-year / life goals


_HORIZON_WEIGHTS: dict[LIFHorizon, float] = {
    LIFHorizon.IMMEDIATE: 2.0,
    LIFHorizon.NEAR:      1.0,
    LIFHorizon.ARC:       0.3,
    LIFHorizon.LIFE:      0.1,
}


class TensionType(str, Enum):
    TIME_CONFLICT      = "TIME_CONFLICT"
    GOAL_DRIFT         = "GOAL_DRIFT"
    PRIORITY_AMBIGUITY = "PRIORITY_AMBIGUITY"


@dataclass
class LIFTension:
    tension_type: TensionType
    description:  str
    intent_a_id:  str = ""
    intent_b_id:  str = ""


@dataclass
class LIFIntent:
    description:   str
    horizon:       LIFHorizon  = LIFHorizon.NEAR
    urgency:       float       = 0.5   # 0-1
    importance:    float       = 0.5   # 0-1
    confidence:    float       = 0.8   # 0-1
    source:        str         = "inferred"
    intent_id:     str         = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at:    float       = field(default_factory=time.monotonic)
    context:       dict[str, Any] = field(default_factory=dict)

    @property
    def priority_score(self) -> float:
        return self.urgency * self.importance * self.confidence


@dataclass
class LIFSnapshot:
    intents:  list[LIFIntent]
    tensions: list[LIFTension] = field(default_factory=list)
    ts:       float            = field(default_factory=time.monotonic)

    def top_priority(self) -> LIFIntent | None:
        if not self.intents:
            return None
        return max(self.intents, key=lambda i: i.priority_score * _HORIZON_WEIGHTS[i.horizon])


class LIFEngine:
    """
    Living Intent Fabric — maintains a set of active intents across four horizons
    and computes the Temporal Gravity Score for the proactive scheduler.
    """

    def __init__(self, max_intents: int = 50) -> None:
        self._intents: list[LIFIntent] = []
        self._max     = max_intents

    # ── Public API ────────────────────────────────────────────────────────────

    def snapshot(self) -> LIFSnapshot:
        return LIFSnapshot(
            intents=list(self._intents),
            tensions=self.detect_tensions(),
        )

    def extract_from_signal(self, signal: str, context: dict | None = None) -> LIFIntent:
        """Heuristic classifier: map raw text signal → LIFIntent."""
        text  = signal.lower()
        ctx   = context or {}
        urgency    = 0.8 if any(w in text for w in ["urgent", "asap", "now", "today", "immediately"]) else 0.4
        importance = 0.9 if any(w in text for w in ["critical", "important", "must", "need"]) else 0.5

        if any(w in text for w in ["now", "today", "asap", "immediately", "right now"]):
            horizon = LIFHorizon.IMMEDIATE
        elif any(w in text for w in ["this week", "soon", "shortly"]):
            horizon = LIFHorizon.NEAR
        elif any(w in text for w in ["this month", "quarter", "project"]):
            horizon = LIFHorizon.ARC
        else:
            horizon = LIFHorizon.NEAR

        intent = LIFIntent(
            description=signal[:200],
            horizon=horizon,
            urgency=urgency,
            importance=importance,
            confidence=ctx.get("confidence", 0.7),
            source="inferred",
            context=ctx,
        )
        self._add(intent)
        return intent

    def add_verbatim(self, description: str, horizon: LIFHorizon,
                     urgency: float = 0.7, importance: float = 0.7,
                     confidence: float = 0.9, context: dict | None = None) -> LIFIntent:
        """Add an intent stated explicitly by the user."""
        intent = LIFIntent(
            description=description,
            horizon=horizon,
            urgency=min(1.0, max(0.0, urgency)),
            importance=min(1.0, max(0.0, importance)),
            confidence=min(1.0, max(0.0, confidence)),
            source="verbatim",
            context=context or {},
        )
        self._add(intent)
        return intent

    def remove(self, intent_id: str) -> bool:
        before = len(self._intents)
        self._intents = [i for i in self._intents if i.intent_id != intent_id]
        return len(self._intents) < before

    def detect_tensions(self) -> list[LIFTension]:
        tensions: list[LIFTension] = []
        immediates = [i for i in self._intents if i.horizon == LIFHorizon.IMMEDIATE]
        if len(immediates) > 2:
            tensions.append(LIFTension(
                tension_type=TensionType.TIME_CONFLICT,
                description=f"{len(immediates)} IMMEDIATE intents competing for now-bandwidth",
            ))
        high_imp = [i for i in self._intents if i.importance > 0.8]
        high_urg = [i for i in self._intents if i.urgency    > 0.8]
        if len(high_imp) > 1 and len(high_urg) > 1:
            tensions.append(LIFTension(
                tension_type=TensionType.PRIORITY_AMBIGUITY,
                description=f"{len(high_imp)} high-importance intents with unclear ordering",
            ))
        return tensions

    def compute_temporal_gravity(self, snapshot: LIFSnapshot | None = None) -> float:
        """Temporal Gravity Score: 0-3+. >1.5 = critical."""
        snap = snapshot or self.snapshot()
        return round(
            sum(i.priority_score * _HORIZON_WEIGHTS[i.horizon] for i in snap.intents), 4
        )

    @staticmethod
    def horizon_class():
        return LIFHorizon

    # ── Internal ──────────────────────────────────────────────────────────────

    def _add(self, intent: LIFIntent) -> None:
        self._intents.append(intent)
        if len(self._intents) > self._max:
            # evict lowest-priority
            self._intents.sort(
                key=lambda i: i.priority_score * _HORIZON_WEIGHTS[i.horizon], reverse=True
            )
            self._intents = self._intents[: self._max]
