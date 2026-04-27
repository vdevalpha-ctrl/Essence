"""
essence_intelligence.restraint_engine
====================================
Restraint Engine — ported from AURA v4.8.0.

Unique AURA trait: ABSTAIN is the default decision. The engine checks 8
ordered conditions and returns PROCEED only when all pass. A CRITICAL
override fires when urgency ≥ 0.95 AND NOT deep_focus AND TGS > 1.5.

Abstentions are recorded by the caller to the QuietLedger.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class RestraintDecision(str, Enum):
    ABSTAIN  = "ABSTAIN"
    PROCEED  = "PROCEED"
    CRITICAL = "CRITICAL"   # urgency override


@dataclass
class RestraintResult:
    decision:         RestraintDecision
    reason:           str
    abstention_reason: str | None = None   # filled when ABSTAIN
    confidence:       float = 1.0


class RestraintEngine:
    """
    ABSTAIN-first gate for every proposed Essence action.
    Call evaluate() before executing any proactive action.
    """

    def evaluate(
        self,
        *,
        action_description: str = "",
        urgency:            float = 0.5,
        is_deep_focus:      bool  = False,
        tgs:                float = 0.0,
        budget_remaining:   int   = 10,
        anomaly_active:     bool  = False,
        trust_tier:         int   = 1,
        is_reversible:      bool  = True,
        endorsement_score:  float = 0.65,
        user_present:       bool  = True,
    ) -> RestraintResult:
        """
        Evaluate whether a proposed action should PROCEED, ABSTAIN, or CRITICAL.

        Checks (in priority order):
          1. Budget depleted
          2. Anomaly active
          3. Deep focus + not critical
          4. Trust tier 0 (blocked domain)
          5. Endorsement below threshold
          6. Irreversible + low trust
          7. TGS too low to warrant interruption
          8. User not present (offline safety)
        CRITICAL override: urgency ≥ 0.95 AND NOT deep_focus AND TGS > 1.5
        """
        # CRITICAL override check first
        if urgency >= 0.95 and not is_deep_focus and tgs > 1.5:
            return RestraintResult(
                decision=RestraintDecision.CRITICAL,
                reason="CRITICAL override: urgency ≥ 0.95, not in deep focus, TGS > 1.5",
            )

        # 1. Budget depleted
        if budget_remaining <= 0:
            return RestraintResult(
                decision=RestraintDecision.ABSTAIN,
                reason="Daily interrupt budget depleted",
                abstention_reason="budget_depleted",
            )

        # 2. Active anomaly
        if anomaly_active:
            return RestraintResult(
                decision=RestraintDecision.ABSTAIN,
                reason="CIE anomaly active — holding back until baseline restored",
                abstention_reason="anomaly_active",
            )

        # 3. Deep focus guard (unless urgent)
        if is_deep_focus and urgency < 0.85:
            return RestraintResult(
                decision=RestraintDecision.ABSTAIN,
                reason="User is in deep focus; low-urgency interruption suppressed",
                abstention_reason="deep_focus",
            )

        # 4. Trust tier 0
        if trust_tier == 0:
            return RestraintResult(
                decision=RestraintDecision.ABSTAIN,
                reason="Domain trust tier is T0 (blocked)",
                abstention_reason="trust_blocked",
            )

        # 5. Low endorsement
        threshold = 0.55 if trust_tier <= 1 else 0.65
        if endorsement_score < threshold:
            return RestraintResult(
                decision=RestraintDecision.ABSTAIN,
                reason=f"Endorsement score {endorsement_score:.2f} below threshold {threshold:.2f}",
                abstention_reason="low_endorsement",
            )

        # 6. Irreversible + low trust
        if not is_reversible and trust_tier < 2:
            return RestraintResult(
                decision=RestraintDecision.ABSTAIN,
                reason="Irreversible action requires trust tier ≥ T2",
                abstention_reason="insufficient_trust_for_irreversible",
            )

        # 7. Low TGS — not enough gravity to bother user
        if tgs < 0.3 and urgency < 0.6:
            return RestraintResult(
                decision=RestraintDecision.ABSTAIN,
                reason=f"TGS {tgs:.2f} too low to warrant proactive surface",
                abstention_reason="low_tgs",
            )

        # 8. User not present
        if not user_present and urgency < 0.9:
            return RestraintResult(
                decision=RestraintDecision.ABSTAIN,
                reason="User not present; deferring non-urgent action",
                abstention_reason="user_absent",
            )

        return RestraintResult(decision=RestraintDecision.PROCEED, reason="All gates cleared")
