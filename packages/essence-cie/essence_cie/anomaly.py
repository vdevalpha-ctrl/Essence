"""essence_cie.anomaly — Anomaly Detector (ported from Essence v1.0)."""
from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AnomalyType(str, Enum):
    CONTEXT_SHIFT   = "CONTEXT_SHIFT"
    FOCUS_INTERRUPT = "FOCUS_INTERRUPT"
    PROPOSAL_BURST  = "PROPOSAL_BURST"
    BEHAVIOR_DRIFT  = "BEHAVIOR_DRIFT"


@dataclass
class AnomalyEvent:
    anomaly_type: AnomalyType
    description:  str
    severity:     float
    context:      dict[str, Any] = field(default_factory=dict)
    ts:           float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, Any]:
        return {"anomaly_type": self.anomaly_type.value, "description": self.description,
                "severity": round(self.severity, 3), "context": self.context}


class AnomalyDetector:
    """Moving-baseline anomaly detector. No LLM calls — pure heuristic."""

    def __init__(self, window_secs: float = 60.0, burst_threshold: int = 5,
                 context_shift_threshold: int = 4) -> None:
        self._window       = window_secs
        self._burst_thresh = burst_threshold
        self._shift_thresh = context_shift_threshold
        self._app_history:      deque[tuple[float, str]] = deque(maxlen=50)
        self._proposal_times:   deque[float]              = deque(maxlen=100)
        self._skill_history:    deque[tuple[float, str]]  = deque(maxlen=100)
        self._baseline_switch_rate: float = 0.0

    def observe_context(self, app_name: str, is_deep_focus: bool = False,
                        intent_name: str = "") -> list[AnomalyEvent]:
        now = time.monotonic()
        self._app_history.append((now, app_name))
        cutoff      = now - self._window
        recent_apps = [a for ts, a in self._app_history if ts >= cutoff]
        unique      = len(set(recent_apps))
        self._baseline_switch_rate = 0.05 * unique + 0.95 * self._baseline_switch_rate

        events: list[AnomalyEvent] = []
        if unique >= self._shift_thresh and unique > self._baseline_switch_rate * 1.8:
            events.append(AnomalyEvent(
                anomaly_type=AnomalyType.CONTEXT_SHIFT,
                description=f"{unique} distinct contexts in {self._window:.0f}s — rapid switching",
                severity=min(1.0, unique / (self._shift_thresh * 2)),
                context={"unique_apps": unique, "baseline": round(self._baseline_switch_rate, 1)},
            ))
        if is_deep_focus and intent_name:
            events.append(AnomalyEvent(
                anomaly_type=AnomalyType.FOCUS_INTERRUPT,
                description=f"CIE proposing '{intent_name}' during deep focus",
                severity=0.6,
                context={"intent": intent_name},
            ))
        return events

    def observe_proposal(self, skill_name: str = "") -> list[AnomalyEvent]:
        now = time.monotonic()
        self._proposal_times.append(now)
        burst = sum(1 for t in self._proposal_times if t >= now - self._window)
        if burst >= self._burst_thresh:
            return [AnomalyEvent(
                anomaly_type=AnomalyType.PROPOSAL_BURST,
                description=f"{burst} proposals in {self._window:.0f}s — possible runaway loop",
                severity=min(1.0, burst / (self._burst_thresh * 2)),
                context={"burst_count": burst},
            )]
        return []

    def observe_skill(self, skill_name: str, drift_score: float = 0.0) -> list[AnomalyEvent]:
        self._skill_history.append((time.monotonic(), skill_name))
        if drift_score > 0.55:
            return [AnomalyEvent(
                anomaly_type=AnomalyType.BEHAVIOR_DRIFT,
                description=f"Routine model drift={drift_score:.2f} — behavior deviates from baseline",
                severity=min(1.0, drift_score),
                context={"drift_score": drift_score, "skill": skill_name},
            )]
        return []

    @property
    def any_active(self) -> bool:
        now = time.monotonic()
        return any(ts >= now - self._window for ts in self._proposal_times)

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "proposals_in_window":   sum(1 for t in self._proposal_times if t >= now - self._window),
            "unique_apps_in_window": len({a for ts, a in self._app_history if ts >= now - self._window}),
            "baseline_switch_rate":  round(self._baseline_switch_rate, 2),
            "window_secs":           self._window,
        }
