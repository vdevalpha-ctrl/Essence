"""
essence_intelligence.fis_engine
==============================
Feedback Intent Scorer (FIS) — ported from AURA v4.8.0.

Unique AURA trait: 7-axis rejection/intent classifier with a 500 ms latency
target. Each axis maps to a Essence subsystem that should receive the signal.

Axes: temporal_coherence, value_alignment, goal_relevance, context_fit,
      tone_match, constraint_adherence, user_sovereignty

Routing: axis → subsystem name for downstream trust/ledger updates.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("essence.fis_engine")

FIS_LATENCY_TARGET_MS = 500.0

_AXIS_ROUTES: dict[str, str] = {
    "temporal_coherence":  "living_intent_fabric",
    "value_alignment":     "trust_ledger",
    "goal_relevance":      "living_intent_fabric",
    "context_fit":         "tomm_engine",
    "tone_match":          "user_persona",
    "constraint_adherence":"policy_engine",
    "user_sovereignty":    "trust_ledger",
}


@dataclass
class RejectionEvent:
    axis_scores:    dict[str, float]
    overall_score:  float
    is_rejection:   bool
    routed_systems: list[str] = field(default_factory=list)
    latency_ms:     float = 0.0
    raw_feedback:   str   = ""

    @property
    def all_routes(self) -> list[str]:
        return [_AXIS_ROUTES[ax] for ax in self.axis_scores if self.axis_scores[ax] < 0.4]


class FISEngine:
    """
    7-axis intent/rejection classifier.
    Call classify() on every user override, correction, or rejection signal.
    Results route back to the TrustLedger and LIF automatically.
    """

    _REJECTION_THRESHOLD = 0.4

    def classify(
        self,
        feedback:           str,
        context:            dict[str, Any] | None = None,
        is_explicit_reject: bool = False,
    ) -> RejectionEvent:
        t0 = time.monotonic()
        ctx  = context or {}
        text = feedback.lower()

        scores: dict[str, float] = {
            "temporal_coherence":  self._score_temporal(text, ctx),
            "value_alignment":     self._score_values(text, ctx),
            "goal_relevance":      self._score_goals(text, ctx),
            "context_fit":         self._score_context(text, ctx),
            "tone_match":          self._score_tone(text, ctx),
            "constraint_adherence":self._score_constraints(text, ctx),
            "user_sovereignty":    self._score_sovereignty(text, ctx, is_explicit_reject),
        }
        overall = round(sum(scores.values()) / len(scores), 4)
        is_rej  = is_explicit_reject or overall < self._REJECTION_THRESHOLD

        latency_ms = (time.monotonic() - t0) * 1000
        if latency_ms > FIS_LATENCY_TARGET_MS:
            log.warning("FIS latency %.0f ms exceeded %.0f ms target", latency_ms, FIS_LATENCY_TARGET_MS)

        ev = RejectionEvent(
            axis_scores=scores,
            overall_score=overall,
            is_rejection=is_rej,
            latency_ms=round(latency_ms, 2),
            raw_feedback=feedback[:500],
        )
        ev.routed_systems = ev.all_routes
        return ev

    # ── Axis scorers (heuristic) ──────────────────────────────────────────────

    def _score_temporal(self, text: str, ctx: dict) -> float:
        neg = any(w in text for w in ["wrong time","bad timing","too early","too late","not now"])
        return 0.2 if neg else 0.75

    def _score_values(self, text: str, ctx: dict) -> float:
        neg = any(w in text for w in ["wrong","bad","inappropriate","unethical","disagree","shouldn't"])
        return 0.2 if neg else 0.75

    def _score_goals(self, text: str, ctx: dict) -> float:
        neg = any(w in text for w in ["irrelevant","off-topic","not what i wanted","wrong goal"])
        return 0.2 if neg else 0.75

    def _score_context(self, text: str, ctx: dict) -> float:
        neg = any(w in text for w in ["doesn't make sense","context","misunderstood","out of place"])
        return 0.3 if neg else 0.8

    def _score_tone(self, text: str, ctx: dict) -> float:
        neg = any(w in text for w in ["tone","rude","formal","casual","stop","don't"])
        return 0.4 if neg else 0.85

    def _score_constraints(self, text: str, ctx: dict) -> float:
        neg = any(w in text for w in ["can't","not allowed","restricted","permission","blocked"])
        return 0.1 if neg else 0.8

    def _score_sovereignty(self, text: str, ctx: dict, explicit: bool) -> float:
        if explicit:
            return 0.0
        neg = any(w in text for w in ["stop","don't do that","i didn't ask","no","reject"])
        return 0.1 if neg else 0.9
