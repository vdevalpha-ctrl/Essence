"""
essence_cie.scorer — CIE Interrupt Budget Scorer (ported from Essence v1.0).

Daily surface event quota (default 10/day) resets at midnight.
Confidence formula:
  With LLM score:    0.50 × rule_conf + 0.30 × historical_rate + 0.20 × llm_score
  Without LLM score: 0.625 × rule_conf + 0.375 × historical_rate

SURFACE ≥ 0.85 | QUEUE 0.70–0.84 | SUPPRESS < 0.70
Budget depleted → force-queue even if score ≥ SURFACE.
"""
from __future__ import annotations
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

log = logging.getLogger("essence.cie.scorer")


class Disposition(str, Enum):
    SURFACE  = "surface"
    QUEUE    = "queue"
    SUPPRESS = "suppress"


@dataclass
class ScoredIntent:
    name:        str
    rule_conf:   float
    final_score: float
    disposition: Disposition
    metadata:    dict[str, Any] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":        self.name,
            "rule_conf":   round(self.rule_conf, 4),
            "final_score": round(self.final_score, 4),
            "disposition": self.disposition.value,
        }


class CIEScorer:
    """Thread-safe scorer with daily interrupt budget."""

    SURFACE_THRESHOLD = 0.85
    QUEUE_THRESHOLD   = 0.70

    def __init__(self, daily_budget: int = 10) -> None:
        if daily_budget <= 0:
            daily_budget = 10
        self._budget     = daily_budget
        self._max_budget = daily_budget
        self._reset_at   = self._next_midnight()
        self._lock       = threading.Lock()

    @property
    def budget(self) -> int:
        with self._lock:
            self._maybe_reset()
            return self._budget

    @property
    def max_budget(self) -> int:
        return self._max_budget

    def score(
        self,
        name: str,
        rule_confidence: float,
        historical_rate: float = 0.5,
        llm_score: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ScoredIntent:
        if llm_score > 0:
            final = 0.50 * rule_confidence + 0.30 * historical_rate + 0.20 * llm_score
        else:
            final = 0.625 * rule_confidence + 0.375 * historical_rate
        final = max(0.0, min(1.0, final))

        with self._lock:
            self._maybe_reset()
            disp = self._classify(final)
            if disp == Disposition.SURFACE and self._budget <= 0:
                log.debug("CIE budget depleted — force-queueing '%s' (score=%.2f)", name, final)
                disp = Disposition.QUEUE
            elif disp == Disposition.SURFACE:
                self._budget -= 1

        return ScoredIntent(name=name, rule_conf=rule_confidence,
                            final_score=final, disposition=disp, metadata=metadata or {})

    def set_budget(self, new_max: int) -> None:
        if new_max <= 0:
            return
        with self._lock:
            ratio = self._budget / self._max_budget if self._max_budget > 0 else 1.0
            self._max_budget = new_max
            self._budget = round(ratio * new_max)

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._maybe_reset()
            return {
                "budget_remaining": self._budget,
                "budget_max":       self._max_budget,
                "resets_at":        self._reset_at.isoformat(),
            }

    def _classify(self, score: float) -> Disposition:
        if score >= self.SURFACE_THRESHOLD: return Disposition.SURFACE
        if score >= self.QUEUE_THRESHOLD:   return Disposition.QUEUE
        return Disposition.SUPPRESS

    def _maybe_reset(self) -> None:
        now = datetime.now(timezone.utc)
        if now >= self._reset_at:
            self._budget   = self._max_budget
            self._reset_at = self._next_midnight()
            log.info("CIE interrupt budget reset to %d", self._budget)

    @staticmethod
    def _next_midnight() -> datetime:
        now = datetime.now(timezone.utc)
        tomorrow = now.date() + timedelta(days=1)
        return datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=timezone.utc)
