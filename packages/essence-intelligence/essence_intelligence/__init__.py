"""
essence-intelligence — AURA-derived intelligence layer for Essence.

Unique traits ported from AURA v4.8.0:
  LIFEngine           — 4-horizon temporal intent model (IMMEDIATE/NEAR/ARC/LIFE)
  ToMMEngine          — 6-dimension Theory of Mind user-state model
  RestraintEngine     — ABSTAIN-first decision gate
  EndorsementProjector— multi-factor weighted endorsement scoring
  FISEngine           — 7-axis rejection/intent classifier (<500 ms)
  MorningBriefEngine  — fatigue-adaptive 8-section daily briefing
"""

from .living_intent_fabric  import LIFEngine, LIFHorizon, LIFIntent, LIFSnapshot
from .tomm                  import ToMMEngine, ToMMState
from .restraint_engine      import RestraintEngine, RestraintDecision
from .endorsement_projector import EndorsementProjector, EndorsementResult
from .fis_engine            import FISEngine, RejectionEvent
from .morning_brief         import MorningBriefEngine

__all__ = [
    "LIFEngine", "LIFHorizon", "LIFIntent", "LIFSnapshot",
    "ToMMEngine", "ToMMState",
    "RestraintEngine", "RestraintDecision",
    "EndorsementProjector", "EndorsementResult",
    "FISEngine", "RejectionEvent",
    "MorningBriefEngine",
]
