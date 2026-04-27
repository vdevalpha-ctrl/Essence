"""
essence-cie — Essence-derived Contextual Intent Engine for Essence.

Unique traits ported from Essence v1.0:
  CIEScorer     — daily interrupt budget (default 10/day, resets at midnight)
  RoutineModel  — first-order Markov habit predictor (α=0.3, cold-start <50 obs)
  AnomalyDetector — context/proposal anomaly detection with moving baselines
"""
from .scorer  import CIEScorer, ScoredIntent, Disposition
from .routine import RoutineModel
from .anomaly import AnomalyDetector, AnomalyEvent, AnomalyType

__all__ = [
    "CIEScorer", "ScoredIntent", "Disposition",
    "RoutineModel",
    "AnomalyDetector", "AnomalyEvent", "AnomalyType",
]
