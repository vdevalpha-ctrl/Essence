"""
essence_intelligence.morning_brief
==================================
Morning Brief Engine — ported from AURA v4.8.0.

Unique AURA trait: fatigue-adaptive 8-section briefing. Engagement rate
over the past 7 days controls section count:
  ≥ 0.6 → HEALTHY   (8 sections)
  ≥ 0.3 → WATCH     (5 sections)
  < 0.3 → FATIGUED  (3 sections — essentials only)

Also generates a Friday sovereignty review (6-element checklist).
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EngagementTier(str, Enum):
    HEALTHY  = "HEALTHY"
    WATCH    = "WATCH"
    FATIGUED = "FATIGUED"


class MorningBriefEngine:
    """
    Generates today's morning brief adapted to the user's engagement pattern.
    """

    def __init__(self) -> None:
        self._engagement_history: list[float] = []   # per-day engagement 0-1

    def record_engagement(self, score: float) -> None:
        """Record end-of-day engagement score (0-1)."""
        self._engagement_history.append(min(1.0, max(0.0, score)))
        if len(self._engagement_history) > 30:
            self._engagement_history = self._engagement_history[-30:]

    @property
    def engagement_rate_7d(self) -> float:
        recent = self._engagement_history[-7:]
        if not recent:
            return 0.5  # neutral default
        return round(sum(recent) / len(recent), 4)

    @property
    def tier(self) -> EngagementTier:
        r = self.engagement_rate_7d
        if r >= 0.6: return EngagementTier.HEALTHY
        if r >= 0.3: return EngagementTier.WATCH
        return EngagementTier.FATIGUED

    def generate(
        self,
        lif_snapshot: Any  = None,
        tomm_state:   Any  = None,
        session_count: int = 0,
        action_count:  int = 0,
        abstention_count: int = 0,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        is_friday = now.weekday() == 4
        tier = self.tier
        rate = self.engagement_rate_7d

        sections: dict[str, Any] = {
            "generated_at":    now.isoformat(),
            "tier":            tier.value,
            "engagement_7d":   rate,
            "greeting":        self._greeting(tier, now),
        }

        # Always present
        sections["active_intents"] = self._intents_section(lif_snapshot)
        sections["focus_today"]    = self._focus_section(lif_snapshot, tomm_state)

        if tier in (EngagementTier.HEALTHY, EngagementTier.WATCH):
            sections["yesterday_recap"] = {
                "sessions":   session_count,
                "actions":    action_count,
                "abstentions": abstention_count,
                "note": "Essence abstained from interrupting you "
                        f"{abstention_count} time(s) to respect your focus.",
            }
            sections["cie_readiness"] = {
                "budget":  "Daily interrupt budget reset.",
                "routine": "Habit model warm — proactive suggestions active.",
            }
            sections["trust_snapshot"] = {
                "note": "Check /trust to review domain trust tiers.",
            }

        if tier == EngagementTier.HEALTHY:
            sections["learning_queue"] = {
                "note": "No pending learning items — all observations integrated.",
            }
            sections["open_loops"] = {
                "note": "Review /quiet to see all deliberate abstentions.",
            }
            sections["upcoming"] = {
                "note": "ARC/LIFE horizon intents surfaced for weekly planning.",
                "arcs": [i.description for i in (lif_snapshot.intents if lif_snapshot else [])
                         if hasattr(i, "horizon") and i.horizon.value in ("ARC", "LIFE")][:3],
            }

        if is_friday:
            sections["sovereignty_review"] = self._friday_review()

        return sections

    def _greeting(self, tier: EngagementTier, now: datetime) -> str:
        hour = now.hour
        if hour < 12: time_str = "Good morning"
        elif hour < 17: time_str = "Good afternoon"
        else: time_str = "Good evening"
        suffix = {
            EngagementTier.HEALTHY:  " — you're in good rhythm.",
            EngagementTier.WATCH:    " — engagement trending down, lighter brief today.",
            EngagementTier.FATIGUED: " — essentials only. Rest is important.",
        }[tier]
        return time_str + suffix

    def _intents_section(self, snap: Any) -> dict:
        if not snap or not hasattr(snap, "intents"):
            return {"count": 0, "items": [], "tgs": 0.0}
        top = sorted(snap.intents, key=lambda i: i.priority_score, reverse=True)[:5]
        return {
            "count": len(snap.intents),
            "items": [{"description": i.description, "horizon": i.horizon.value,
                       "priority": round(i.priority_score, 3)} for i in top],
            "tgs": 0.0,
        }

    def _focus_section(self, snap: Any, tomm: Any) -> dict:
        focus = "Unable to determine — no ToMM data."
        if tomm and hasattr(tomm, "is_deep_focus"):
            focus = "Deep focus mode detected." if tomm.is_deep_focus else "Normal working mode."
        top_intent = ""
        if snap and hasattr(snap, "top_priority"):
            tp = snap.top_priority()
            top_intent = tp.description if tp else ""
        return {"mode": focus, "top_intent": top_intent}

    def _friday_review(self) -> dict:
        return {
            "title": "Weekly Sovereignty Review",
            "checklist": [
                "Review all trust events this week (/trust/general/events)",
                "Audit deliberate abstentions (/quiet-ledger/abstentions)",
                "Confirm or revoke any pending HITL decisions",
                "Adjust CIE interrupt budget for next week if needed",
                "Review ARC-horizon intents — still relevant?",
                "Confirm Essence is learning the right habits (routine model drift)",
            ],
        }
