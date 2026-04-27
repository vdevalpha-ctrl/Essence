"""
skill_agent.py — Skill Agent Contract
======================================
Every skill in Essence is a SkillAgent subclass.  Skills are first-class
agents, not passive config blobs:

  • Declare a SkillManifest (capabilities, I/O schema, autonomy threshold)
  • Implement execute(context) → SkillResult
  • Subscribe to bus topics on_load
  • Report health; auto-pause after 3 consecutive errors
  • Hot-loadable by the SkillRegistry

Usage
-----
    class ResearchSkill(SkillAgent):
        manifest = SkillManifest(
            id="research", name="Web Research", version="1.0",
            description="...", category="research",
            capabilities=["http_get", "llm_call"],
            requires_caps=["network"],
            input_schema={"query": "string"},
            output_schema={"summary": "string", "sources": "list"},
        )

        async def execute(self, ctx: SkillContext) -> SkillResult:
            ...
"""
from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("essence.skill_agent")


# ---------------------------------------------------------------------------
# Manifest — capability declaration every skill must provide
# ---------------------------------------------------------------------------

@dataclass
class SkillManifest:
    id:                 str
    name:               str
    version:            str
    description:        str
    category:           str           # research|write|code|memory|tool|notify|calendar
    capabilities:       list[str]     # what it provides: ["web_search","llm_call","file_write"]
    requires_caps:      list[str]     # what the kernel must grant: ["network","filesystem"]
    input_schema:       dict
    output_schema:      dict
    max_latency_ms:     int   = 30_000
    autonomy_threshold: float = 0.80  # min gravity to fire without user confirm
    max_retries:        int   = 2
    timeout_s:          int   = 120
    tags:               list[str] = field(default_factory=list)
    system_prompt:      str   = ""    # injected when this skill uses LLM
    enabled:            bool  = True  # false = excluded from planning and execution


# ---------------------------------------------------------------------------
# Execution context — everything a skill receives at call time
# ---------------------------------------------------------------------------

@dataclass
class SkillContext:
    task_id:        str
    intent:         str
    input_data:     dict
    shared_context: dict            # outputs from upstream steps
    memory_block:   str  = ""       # gravity memory context string
    identity:       dict = field(default_factory=dict)
    command_id:     str  = field(default_factory=lambda: f"cmd-{uuid.uuid4().hex[:12]}")
    session_id:     str  = ""
    metadata:       dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Skill result
# ---------------------------------------------------------------------------

@dataclass
class SkillResult:
    skill_id:  str
    status:    str                  # done | error | partial | skipped | timeout
    output:    Any  = None
    error:     str  = ""
    started:   float = field(default_factory=time.time)
    finished:  float = field(default_factory=time.time)  # always set; overridden by safe_execute
    retries:   int  = 0
    metadata:  dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.started and self.finished:
            return round((self.finished - self.started) * 1000, 1)
        return 0.0

    def to_dict(self) -> dict:
        return {
            "skill_id":    self.skill_id,
            "status":      self.status,
            "output":      self.output,
            "error":       self.error,
            "duration_ms": self.duration_ms,
            "retries":     self.retries,
            "metadata":    self.metadata,
        }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@dataclass
class SkillHealth:
    skill_id:           str
    state:              str   = "active"   # active|degraded|paused|failed
    consecutive_errors: int   = 0
    last_error:         str   = ""
    last_ok:            float = 0.0
    avg_latency_ms:     float = 0.0
    invoke_count:       int   = 0

    def to_dict(self) -> dict:
        return {
            "skill_id":           self.skill_id,
            "state":              self.state,
            "consecutive_errors": self.consecutive_errors,
            "last_error":         self.last_error,
            "avg_latency_ms":     round(self.avg_latency_ms, 1),
            "invoke_count":       self.invoke_count,
        }


# ---------------------------------------------------------------------------
# Base SkillAgent — all skills inherit from this
# ---------------------------------------------------------------------------

ERROR_THRESHOLD = 3   # auto-pause after this many consecutive errors


class SkillAgent(ABC):
    """Abstract base for all Essence skill agents."""

    manifest: SkillManifest   # MUST be set as a class attribute

    def __init__(self) -> None:
        self._bus: Any = None
        self._health = SkillHealth(skill_id=self.manifest.id)

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def on_load(self, bus: Any) -> None:
        """Called by SkillRegistry when the skill is loaded.
        Override to subscribe to bus topics or initialise resources."""
        self._bus = bus

    async def on_unload(self) -> None:
        """Called before the skill is removed. Clean up here."""
        pass

    async def health_check(self) -> SkillHealth:
        return self._health

    # ── Execution ─────────────────────────────────────────────────────

    @abstractmethod
    async def execute(self, context: SkillContext) -> SkillResult:
        """Run the skill. Must be implemented by every subclass."""
        ...

    async def safe_execute(self, context: SkillContext) -> SkillResult:
        """Wrapper: tracks health metrics, handles retries, enforces timeout."""
        import asyncio

        if self._health.state == "paused":
            return SkillResult(
                skill_id=self.manifest.id, status="skipped",
                error="skill paused (too many consecutive errors)",
            )

        retries = 0
        last_err = ""
        while retries <= self.manifest.max_retries:
            start = time.time()
            try:
                result = await asyncio.wait_for(
                    self.execute(context),
                    timeout=self.manifest.timeout_s,
                )
                result.started  = start
                result.finished = time.time()
                result.retries  = retries
                self._record_success(result.duration_ms)
                return result

            except asyncio.TimeoutError:
                last_err = f"timeout after {self.manifest.timeout_s}s"
                retries += 1

            except Exception as exc:
                last_err = str(exc)
                retries += 1
                log.warning("Skill %r error (attempt %d): %s", self.manifest.id, retries, exc)

        self._record_error(last_err)
        return SkillResult(
            skill_id=self.manifest.id,
            status="error",
            error=last_err,
            started=start,
            finished=time.time(),
            retries=retries,
        )

    # ── Health tracking ───────────────────────────────────────────────

    def _record_success(self, latency_ms: float) -> None:
        h = self._health
        h.consecutive_errors = 0
        h.state    = "active"
        h.last_ok  = time.time()
        h.invoke_count += 1
        h.avg_latency_ms = h.avg_latency_ms * 0.8 + latency_ms * 0.2

    def _record_error(self, error: str) -> None:
        h = self._health
        h.consecutive_errors += 1
        h.last_error = error
        if h.consecutive_errors >= ERROR_THRESHOLD:
            h.state = "paused"
            log.warning("Skill %r auto-paused after %d consecutive errors",
                       self.manifest.id, h.consecutive_errors)

    def is_healthy(self) -> bool:
        return self._health.state in ("active", "degraded")

    def reset_health(self) -> None:
        self._health.consecutive_errors = 0
        self._health.state = "active"
        self._health.last_error = ""


# ---------------------------------------------------------------------------
# Skill Registry — loads and manages all skill agents
# ---------------------------------------------------------------------------

class SkillRegistry:
    """
    Runtime registry of all loaded SkillAgent instances.

    Skills are registered programmatically (built-ins) or discovered from
    a plugin directory (dynamic).  The kernel calls match_capability() and
    get() — it never instantiates skills directly.
    """

    def __init__(self) -> None:
        self._agents: dict[str, SkillAgent] = {}
        self._bus: Any = None

    def register(self, agent: SkillAgent) -> None:
        self._agents[agent.manifest.id] = agent

    async def load_all(self, bus: Any) -> None:
        """Call on_load for every registered skill."""
        self._bus = bus
        for agent in self._agents.values():
            try:
                await agent.on_load(bus)
            except Exception as exc:
                log.error("Skill %r on_load failed: %s", agent.manifest.id, exc)

    async def unload_all(self) -> None:
        for agent in self._agents.values():
            try:
                await agent.on_unload()
            except Exception:
                pass

    def get(self, skill_id: str) -> SkillAgent | None:
        agent = self._agents.get(skill_id)
        if agent is None or not agent.manifest.enabled:
            return None
        return agent

    def all_healthy(self) -> list[SkillAgent]:
        return [a for a in self._agents.values() if a.is_healthy() and a.manifest.enabled]

    def match_capability(self, intent: str) -> SkillAgent | None:
        """
        Fast-path: return the first healthy skill whose capabilities or
        tags contain a token from the intent string.
        """
        lower = intent.lower()
        tokens = set(lower.split())
        best: SkillAgent | None = None
        best_score = 0

        for agent in self._agents.values():
            if not agent.is_healthy() or not agent.manifest.enabled:
                continue
            score = 0
            for cap in agent.manifest.capabilities + agent.manifest.tags:
                if cap.lower() in lower:
                    score += 2
            for tok in tokens:
                if tok in agent.manifest.description.lower():
                    score += 1
            if score > best_score:
                best_score, best = score, agent

        return best if best_score >= 2 else None

    def catalog_for_planner(self) -> list[dict]:
        """Return a compact catalog of enabled, healthy skills for the LLM planner."""
        return [
            {
                "id":           a.manifest.id,
                "name":         a.manifest.name,
                "description":  a.manifest.description,
                "capabilities": a.manifest.capabilities,
                "input_schema": a.manifest.input_schema,
                "healthy":      a.is_healthy(),
            }
            for a in self._agents.values()
            if a.manifest.enabled and a.is_healthy()
        ]

    def health_report(self) -> list[dict]:
        return [a._health.to_dict() for a in self._agents.values()]
