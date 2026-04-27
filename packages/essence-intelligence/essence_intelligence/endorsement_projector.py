"""
essence_intelligence.endorsement_projector
=========================================
Endorsement Projector — ported from AURA v4.8.0.

Unique AURA trait: multi-factor endorsement score combining goal alignment,
value consistency, historical endorsement rate, and constraint proximity.

Formula: 0.35×goal_alignment + 0.30×value_consistency
       + 0.25×historical_endorsement + 0.10×constraint_proximity

Threshold: 0.65 (production), 0.55 (cold-start or reversible + impact<0.2)

Calibration stages:
  STAGE_0:  0-19 observations  (cold-start)
  STAGE_1: 20-49
  STAGE_2: 50-99
  STAGE_3: 100+  (fully calibrated)
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum


class CalibrationStage(str, Enum):
    STAGE_0 = "STAGE_0"  # cold-start
    STAGE_1 = "STAGE_1"
    STAGE_2 = "STAGE_2"
    STAGE_3 = "STAGE_3"  # fully calibrated


@dataclass
class EndorsementResult:
    score:              float
    approved:           bool
    calibration_stage:  CalibrationStage
    components:         dict[str, float] = field(default_factory=dict)
    reason:             str = ""


class EndorsementProjector:
    """
    Computes weighted endorsement scores for proposed Essence actions.
    Tracks historical endorsement rates for calibration.
    """

    _WEIGHTS = {
        "goal_alignment":         0.35,
        "value_consistency":      0.30,
        "historical_endorsement": 0.25,
        "constraint_proximity":   0.10,
    }
    _THRESHOLD_PRODUCTION  = 0.65
    _THRESHOLD_COLD_START  = 0.55

    def __init__(self) -> None:
        self._endorsements: list[bool] = []   # historical record
        self._goal_vectors:  list[list[float]] = []
        self._value_vectors: list[list[float]] = []

    @property
    def observations(self) -> int:
        return len(self._endorsements)

    @property
    def calibration_stage(self) -> CalibrationStage:
        n = self.observations
        if n < 20:  return CalibrationStage.STAGE_0
        if n < 50:  return CalibrationStage.STAGE_1
        if n < 100: return CalibrationStage.STAGE_2
        return CalibrationStage.STAGE_3

    def evaluate(
        self,
        goal_alignment:     float,
        value_consistency:  float,
        constraint_proximity: float = 0.5,
        is_reversible:      bool  = True,
        impact_estimate:    float = 0.5,
    ) -> EndorsementResult:
        hist = self._historical_rate()
        components = {
            "goal_alignment":         min(1.0, max(0.0, goal_alignment)),
            "value_consistency":      min(1.0, max(0.0, value_consistency)),
            "historical_endorsement": hist,
            "constraint_proximity":   min(1.0, max(0.0, constraint_proximity)),
        }
        score = sum(v * self._WEIGHTS[k] for k, v in components.items())
        score = round(min(1.0, max(0.0, score)), 4)

        # Relaxed threshold for cold-start or low-impact reversible actions
        cold_start = self.calibration_stage == CalibrationStage.STAGE_0
        threshold = (
            self._THRESHOLD_COLD_START
            if (cold_start or (is_reversible and impact_estimate < 0.2))
            else self._THRESHOLD_PRODUCTION
        )

        return EndorsementResult(
            score=score,
            approved=score >= threshold,
            calibration_stage=self.calibration_stage,
            components=components,
            reason=f"score={score:.2f} {'≥' if score >= threshold else '<'} threshold={threshold:.2f}",
        )

    def record_outcome(self, endorsed: bool) -> None:
        """Record whether the action was actually endorsed by the user."""
        self._endorsements.append(endorsed)
        if len(self._endorsements) > 500:
            self._endorsements = self._endorsements[-500:]

    def _historical_rate(self) -> float:
        if not self._endorsements:
            return 0.5  # neutral cold-start
        return round(sum(self._endorsements) / len(self._endorsements), 4)
