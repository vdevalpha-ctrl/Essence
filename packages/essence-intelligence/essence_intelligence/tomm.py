"""
essence_intelligence.tomm
=======================
Theory of Mind Model (ToMM) — ported from AURA v4.8.0.

Unique AURA trait: 6-dimensional user-state model updated from conversation
signals. Dimensions: cognitive_load, emotional_valence, attention_focus,
stress_index, goal_clarity, social_context.

is_deep_focus = FOCUSED + confidence ≥ 0.6 + load < 0.5
confidence < 0.6 → is_unknown = True (cold-start guard)
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SocialContext(str, Enum):
    SOLO        = "SOLO"
    COLLABORATIVE = "COLLABORATIVE"
    OBSERVED    = "OBSERVED"
    PRESENTING  = "PRESENTING"


class AttentionState(str, Enum):
    SCATTERED = "SCATTERED"
    NORMAL    = "NORMAL"
    FOCUSED   = "FOCUSED"
    DEEP      = "DEEP"


@dataclass
class ToMMDimensionResult:
    value:      float
    confidence: float
    is_unknown: bool = False


@dataclass
class ToMMState:
    dimensions:    dict[str, Any] = field(default_factory=dict)
    confidence:    float = 0.0
    attention:     AttentionState = AttentionState.NORMAL
    social_context: SocialContext = SocialContext.SOLO
    ts:            float = field(default_factory=time.monotonic)

    @property
    def is_deep_focus(self) -> bool:
        load = self.dimensions.get("cognitive_load", 0.5)
        return (
            self.attention in (AttentionState.FOCUSED, AttentionState.DEEP)
            and self.confidence >= 0.6
            and load < 0.5
        )

    @property
    def cognitive_load(self) -> float:
        return self.dimensions.get("cognitive_load", 0.5)

    @property
    def stress_index(self) -> float:
        return self.dimensions.get("stress_index", 0.0)


class ToMMEngine:
    """
    Stateful Theory of Mind engine.
    Call update() after each user message; read state for restraint decisions.
    """

    _MIN_CONFIDENCE = 0.6

    def __init__(self) -> None:
        self._state = ToMMState()
        self._tick_count = 0

    @property
    def state(self) -> ToMMState:
        return self._state

    def update(
        self,
        user_message: str = "",
        response_time_s: float = 0.0,
        override_count_7d: int = 0,
        verbatim_count: int = 0,
        session_messages: int = 0,
    ) -> ToMMState:
        """Re-compute ToMM state from observable signals."""
        self._tick_count += 1
        text = user_message.lower()

        # ── cognitive_load ────────────────────────────────────────────────────
        question_density = text.count("?") / max(1, len(text.split()))
        load = min(1.0, 0.3 + question_density * 2.0 + min(0.3, override_count_7d * 0.05))

        # ── emotional_valence (−1 neg … +1 pos) ──────────────────────────────
        pos_words = sum(1 for w in ["good","great","thanks","perfect","nice","love"] if w in text)
        neg_words = sum(1 for w in ["bad","wrong","error","issue","problem","fail","no","not"] if w in text)
        valence = max(-1.0, min(1.0, (pos_words - neg_words) * 0.2))

        # ── stress_index ──────────────────────────────────────────────────────
        stress = min(1.0, override_count_7d * 0.08 + (0.3 if "urgent" in text or "asap" in text else 0.0))

        # ── attention_focus ───────────────────────────────────────────────────
        if len(user_message) > 300 and "?" not in user_message[:100]:
            attention = AttentionState.DEEP
            focus_val = 0.9
        elif len(user_message) > 80:
            attention = AttentionState.FOCUSED
            focus_val = 0.7
        elif session_messages < 2:
            attention = AttentionState.NORMAL
            focus_val = 0.5
        else:
            attention = AttentionState.SCATTERED
            focus_val = 0.3

        # ── goal_clarity ──────────────────────────────────────────────────────
        goal_clarity = min(1.0, 0.4 + verbatim_count * 0.1)

        # ── social_context ────────────────────────────────────────────────────
        social = SocialContext.SOLO

        # ── confidence grows with observations ─────────────────────────────
        confidence = min(1.0, self._tick_count / 10.0)

        self._state = ToMMState(
            dimensions={
                "cognitive_load":    round(load, 4),
                "emotional_valence": round(valence, 4),
                "attention_focus":   round(focus_val, 4),
                "stress_index":      round(stress, 4),
                "goal_clarity":      round(goal_clarity, 4),
                "social_context":    social.value,
            },
            confidence=round(confidence, 4),
            attention=attention,
            social_context=social,
        )
        return self._state

    def to_dict(self) -> dict[str, Any]:
        s = self._state
        return {
            "cognitive_load":    s.dimensions.get("cognitive_load", 0.5),
            "emotional_valence": s.dimensions.get("emotional_valence", 0.0),
            "attention_focus":   s.dimensions.get("attention_focus", 0.5),
            "stress_index":      s.dimensions.get("stress_index", 0.0),
            "goal_clarity":      s.dimensions.get("goal_clarity", 0.5),
            "social_context":    s.dimensions.get("social_context", "SOLO"),
            "is_deep_focus":     s.is_deep_focus,
            "confidence":        s.confidence,
            "attention":         s.attention.value,
        }
