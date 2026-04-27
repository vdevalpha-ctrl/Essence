"""
governance.py — Essence Governance Enforcer
============================================
The governance layer is a RUNTIME GATE, not a log viewer.

Every WorkItem dispatched through the kernel passes through
GovernanceEnforcer.check() before execution.  If the check fails,
the item is dropped and a governance.violation event is emitted.

Enforced policies
-----------------
  1. Quiet window       — no autonomous actions during configured hours
  2. CIE budget         — daily interrupt budget (10/day, midnight reset)
  3. Skill trust        — skills with trust < threshold are blocked
  4. Capability tokens  — skills that require caps not granted are blocked
  5. Idempotency guard  — duplicate command_ids are short-circuited
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("essence.governance")


# ---------------------------------------------------------------------------
# Trust ledger — 3-axis per skill (competence / values / judgement)
# ---------------------------------------------------------------------------

@dataclass
class TrustRecord:
    skill_id:    str
    competence:  float = 0.70   # task success rate
    values:      float = 0.70   # alignment with stated preferences
    judgement:   float = 0.70   # appropriate action selection
    violations:  int   = 0
    locked:      bool  = False  # SEVERE_VIOLATION → never_graduate

    @property
    def composite(self) -> float:
        return (self.competence * 0.40 + self.values * 0.35 + self.judgement * 0.25)


_TRUST_LEDGER_PATH = Path(__file__).resolve().parent.parent / "data" / "trust_ledger.json"


class TrustLedger:
    """Persisted 3-axis trust ledger for skill agents.

    Scores survive process restarts so trust builds up over time per spec.
    Storage: data/trust_ledger.json (atomic write via .tmp rename).
    """

    def __init__(self, path: "Path | None" = None) -> None:
        self._path    = path or _TRUST_LEDGER_PATH
        self._records: dict[str, TrustRecord] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for row in data.get("records", []):
                    sid = row.get("skill_id", "")
                    if sid:
                        self._records[sid] = TrustRecord(
                            skill_id   = sid,
                            competence = float(row.get("competence", 0.70)),
                            values     = float(row.get("values",     0.70)),
                            judgement  = float(row.get("judgement",  0.70)),
                            violations = int(row.get("violations",   0)),
                            locked     = bool(row.get("locked",      False)),
                        )
        except Exception as e:
            log.warning("TrustLedger: failed to load %s: %s", self._path, e)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"records": self.all_records()}, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except Exception as e:
            log.warning("TrustLedger: failed to save: %s", e)

    # ── Mutations ─────────────────────────────────────────────────────────

    def get(self, skill_id: str) -> TrustRecord:
        if skill_id not in self._records:
            self._records[skill_id] = TrustRecord(skill_id=skill_id)
        return self._records[skill_id]

    def record_success(self, skill_id: str) -> None:
        r = self.get(skill_id)
        if r.locked:
            return
        r.competence = min(1.0, r.competence * 0.95 + 1.0 * 0.05)
        self._save()

    def record_failure(self, skill_id: str) -> None:
        r = self.get(skill_id)
        if r.locked:
            return
        r.competence = max(0.0, r.competence * 0.95 + 0.0 * 0.05)
        r.violations += 1
        self._save()

    def record_severe_violation(self, skill_id: str) -> None:
        r = self.get(skill_id)
        r.locked     = True
        r.competence = 0.0
        r.violations += 1
        log.critical("SEVERE VIOLATION: skill %r locked in TrustLedger", skill_id)
        self._save()

    def is_trusted(self, skill_id: str, threshold: float = 0.30) -> bool:
        r = self.get(skill_id)
        return not r.locked and r.composite >= threshold

    def all_records(self) -> list[dict]:
        return [
            {
                "skill_id":   r.skill_id,
                "composite":  round(r.composite, 3),
                "competence": round(r.competence, 3),
                "values":     round(r.values, 3),
                "judgement":  round(r.judgement, 3),
                "violations": r.violations,
                "locked":     r.locked,
            }
            for r in self._records.values()
        ]


# ---------------------------------------------------------------------------
# CIE scorer — daily interrupt budget
# ---------------------------------------------------------------------------

_CIE_PATH = Path(__file__).resolve().parent.parent / "data" / "cie_budget.json"


class CIEScorer:
    """Cognitive Interrupt Economy — limits autonomous interruptions.

    Daily budget: 10 interrupts. Resets at midnight.
    Persisted to data/cie_budget.json so the budget survives restarts
    (a crash + restart no longer resets the daily counter to 0).
    """

    DAILY_BUDGET = 10

    def __init__(self, path: "Path | None" = None) -> None:
        self._path  = path or _CIE_PATH
        self._count = 0
        self._day   = self._today()
        self._load()

    def _today(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if data.get("day") == self._today():
                    self._count = int(data.get("count", 0))
                    self._day   = data["day"]
        except Exception:
            pass

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps({"day": self._day, "count": self._count}),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _check_reset(self) -> None:
        today = self._today()
        if today != self._day:
            self._day   = today
            self._count = 0
            self._save()

    def consume(self) -> bool:
        """Attempt to consume one interrupt token. Returns True if allowed."""
        self._check_reset()
        if self._count >= self.DAILY_BUDGET:
            return False
        self._count += 1
        self._save()
        return True

    def remaining(self) -> int:
        self._check_reset()
        return max(0, self.DAILY_BUDGET - self._count)

    def exhausted(self) -> bool:
        self._check_reset()
        return self._count >= self.DAILY_BUDGET


# ---------------------------------------------------------------------------
# WorkItem (imported from kernel at runtime to avoid circular)
# ---------------------------------------------------------------------------

# Copied here to avoid circular import — kernel imports governance
@dataclass
class GovernanceContext:
    item_type:    str           # user_request | trigger | heartbeat
    skill_id:     str = ""
    command_id:   str = ""
    autonomy:     bool = False  # True if autonomous (not user-initiated)
    requires_caps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GovernanceEnforcer — the gate
# ---------------------------------------------------------------------------

class GovernanceEnforcer:
    """
    Checks every kernel dispatch against all active policies.
    Returns (allowed: bool, reason: str).
    Emits governance.violation on any rejection.
    """

    TRUST_MIN = 0.30

    def __init__(
        self,
        bus:            Any,
        trust_ledger:   TrustLedger,
        cie:            CIEScorer,
        quiet_window:   tuple[int, int] = (22, 7),    # (start_hour, end_hour) local time
        granted_caps:   set[str] | None = None,
    ) -> None:
        self._bus          = bus
        self._trust        = trust_ledger
        self._cie          = cie
        self._quiet_start, self._quiet_end = quiet_window
        self._granted_caps = granted_caps or {"network", "filesystem", "llm_call", "memory"}
        self._idempotency: dict[str, float] = {}   # command_id → approved_at

    # ── Public gate ───────────────────────────────────────────────────

    def check(self, ctx: GovernanceContext) -> bool:
        """Synchronous check. Returns True if allowed."""
        # User-initiated requests always pass (user is in the loop)
        if not ctx.autonomy:
            return self._idempotency_check(ctx)

        # Autonomous actions go through all gates
        allow, reason, severity = self._evaluate_autonomous(ctx)
        if not allow:
            self._emit_violation(ctx, reason, severity)
        return allow

    # ── Policy checks ─────────────────────────────────────────────────

    def _evaluate_autonomous(self, ctx: GovernanceContext) -> tuple[bool, str, str]:
        # 1. Quiet window
        if self._in_quiet_window():
            return False, "quiet_window: autonomous actions suppressed", "INFO"

        # 2. CIE budget
        if ctx.item_type == "trigger" and not self._cie.consume():
            return False, f"cie_budget_exhausted: {self._cie.remaining()} remaining today", "WARN"

        # 3. Skill trust
        if ctx.skill_id and not self._trust.is_trusted(ctx.skill_id, self.TRUST_MIN):
            return False, f"trust_below_threshold: {ctx.skill_id}", "WARN"

        # 4. Capability tokens
        missing = [c for c in ctx.requires_caps if c not in self._granted_caps]
        if missing:
            return False, f"missing_capability_tokens: {missing}", "WARN"

        return True, "", ""

    def _idempotency_check(self, ctx: GovernanceContext) -> bool:
        if not ctx.command_id:
            return True
        # Check persisted cache first — survives process restarts
        try:
            if self._bus.get_cached(ctx.command_id) is not None:
                log.debug("Governance: command_id %s already has cached result — allow replay", ctx.command_id)
                return True  # kernel will replay the cached response
        except Exception:
            pass
        # In-memory dedup gate for commands not yet completed (no cached result)
        now = time.time()
        if ctx.command_id in self._idempotency:
            age = now - self._idempotency[ctx.command_id]
            if age < 300:   # 5-min dedup window
                log.debug("Governance: duplicate command_id %s in-flight — skipped", ctx.command_id)
                return False
        self._idempotency[ctx.command_id] = now
        self._prune_idempotency(now)
        return True

    def _prune_idempotency(self, now: float) -> None:
        expired = [k for k, t in self._idempotency.items() if now - t > 300]
        for k in expired:
            del self._idempotency[k]

    # ── Helpers ───────────────────────────────────────────────────────

    def _in_quiet_window(self) -> bool:
        hour = datetime.now().hour
        s, e = self._quiet_start, self._quiet_end
        return (hour >= s or hour < e) if s > e else (s <= hour < e)

    def _emit_violation(self, ctx: GovernanceContext, reason: str, severity: str) -> None:
        from server.event_bus import Envelope
        env = Envelope(
            topic="governance.violation",
            source_component="governance",
            data={
                "reason":    reason,
                "severity":  severity,
                "item_type": ctx.item_type,
                "skill_id":  ctx.skill_id,
            },
        )
        self._bus.publish_sync(env)
        log.warning("Governance BLOCK [%s]: %s", severity, reason)
        try:
            from server.audit_logger import get_audit_logger
            get_audit_logger().log_violation(reason, severity=severity, skill_id=ctx.skill_id or "")
        except Exception:
            pass

    # ── Trust feedback (called by reflector) ──────────────────────────

    def on_skill_success(self, skill_id: str) -> None:
        self._trust.record_success(skill_id)

    def on_skill_failure(self, skill_id: str) -> None:
        self._trust.record_failure(skill_id)

    def on_skill_severe_violation(self, skill_id: str) -> None:
        self._trust.record_severe_violation(skill_id)

    # ── Capability management ─────────────────────────────────────────

    def grant_capability(self, cap: str) -> None:
        self._granted_caps.add(cap)

    def revoke_capability(self, cap: str) -> None:
        self._granted_caps.discard(cap)

    # ── Status ───────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "cie_remaining":   self._cie.remaining(),
            "quiet_window":    f"{self._quiet_start:02d}:00–{self._quiet_end:02d}:00",
            "in_quiet_window": self._in_quiet_window(),
            "granted_caps":    sorted(self._granted_caps),
            "trust":           self._trust.all_records(),
        }
