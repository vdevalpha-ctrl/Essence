"""
kernel.py — EssenceKernel: The Cognitive Execution Core
=========================================================
The kernel IS the AI.  Everything else is peripheral.

It owns:
  • The cognitive state machine (IDLE→PLANNING→EXECUTING→REFLECTING→IDLE)
  • A priority work queue (user > trigger > heartbeat)
  • The 3-pass planner (procedural → capability → LLM)
  • The skill executor (drives SkillAgent.safe_execute)
  • The reflector (feeds outcomes back into memory + patterns)
  • The streaming response path (when no skill is needed)

Nothing bypasses the kernel.  The TUI publishes user.request.
The kernel processes it and emits user.response token events.
The TUI subscribes to user.response.  The TUI never calls Ollama.

Bus topics owned by the kernel
-------------------------------
  Subscribes: user.request, heartbeat.tick, trigger.fired, skill.result
  Publishes:  user.response, skill.execute, skill.health,
              kernel.state, governance.audit
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger("essence.kernel")

# Pulled from episodic_memory to avoid circular import at module level
CROSS_SESSION_N = 3


# ---------------------------------------------------------------------------
# Cognitive state machine
# ---------------------------------------------------------------------------

class CognitiveState(Enum):
    IDLE        = "idle"
    PLANNING    = "planning"
    EXECUTING   = "executing"
    REFLECTING  = "reflecting"
    INTERRUPTED = "interrupted"


# ---------------------------------------------------------------------------
# Plan / WorkItem
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    skill_id:    str
    input_map:   dict
    depends_on:  list[str] = field(default_factory=list)
    required:    bool       = True
    timeout_s:   int        = 120


@dataclass
class Plan:
    id:       str
    intent:   str
    goal:     str
    steps:    list[PlanStep]
    source:   str = "llm"          # procedural | capability | llm
    fallback: "Plan | None" = None


@dataclass
class WorkItem:
    priority:   int         # 0=user, 2=trigger, 10=heartbeat
    item_type:  str         # user_request | trigger | skill_result | replan
    payload:    dict
    task_id:    str
    command_id: str = ""
    autonomy:   bool = False
    created:    float = field(default_factory=time.time)

    def __lt__(self, other: "WorkItem") -> bool:
        return self.priority < other.priority


# ---------------------------------------------------------------------------
# EssenceKernel
# ---------------------------------------------------------------------------

class EssenceKernel:
    """
    The single cognitive execution core.

    All system components communicate through the event bus.
    The kernel is the only entity that reads user.request and writes
    user.response.
    """

    _MAX_PLAN_TURNS = 3

    def __init__(
        self,
        bus:            Any,           # EventBus
        memory:         Any,           # GravityMemory
        identity:       Any,           # IdentityEngine
        governance:     Any,           # GovernanceEnforcer
        skill_registry: Any,           # SkillRegistry
        ollama_url:     str,
        model:          str,
        episodic:       Any = None,    # EpisodicMemory (optional; created if None)
    ) -> None:
        self._bus      = bus
        self._memory   = memory
        self._identity = identity
        self._gov      = governance
        self._skills   = skill_registry
        self._ollama   = ollama_url
        self._model    = model
        self._provider = "ollama"   # active provider id; switchable at runtime

        # Model router — used for provider-aware API calls
        from server.model_router import ModelRouter
        _ws = Path(__file__).resolve().parent.parent
        self._router = ModelRouter(_ws, ollama_url=ollama_url)

        # Dynamic router — per-request capability-aware provider selection
        # Initialised lazily with provider keys after init_kernel completes
        from server.dynamic_router import get_dynamic_router
        self._dynamic_router = get_dynamic_router()
        # Flag: True = use dynamic routing; False = always use self._provider/self._model
        self._dynamic_routing = True

        # Model tier registry — PRIMARY / FALLBACK / EMERGENCY backbone.
        # Provides the structural fallback chain for every request.
        # init_tier_registry() is called by essence.py at startup (with workspace
        # path so tiers persist to config.toml); get_tier_registry() just fetches
        # the already-created singleton here.
        from server.model_tiers import get_tier_registry
        self._tier_registry = get_tier_registry()

        # Offline resilience
        from server.offline_cache import get_offline_cache
        self._offline = get_offline_cache()
        self._offline.set_flush_callback(self._replay_queued_messages)
        self._offline.start()

        self._state: CognitiveState = CognitiveState.IDLE
        self._work_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._dispatch_task: asyncio.Task | None = None
        self._current_plan: Plan | None = None

        # L2 Episodic memory — persistent, survives restarts
        if episodic is None:
            from server.episodic_memory import get_episodic_memory
            episodic = get_episodic_memory()
        self._episodic = episodic

        # L2.5 Semantic embedding index — attach to episodic memory
        try:
            self._episodic.init_embeddings(ollama_url=ollama_url)
        except Exception as _ee:
            log.debug("Kernel: embedding init skipped: %s", _ee)

        # L4 Procedural memory — load persisted successful plans from disk
        self._procedural: dict[str, Any] = self._episodic.load_plans()
        log.info("Kernel: loaded %d procedural plans from disk", len(self._procedural))

        # Context compressor — reuse module singleton if already init'd, else create
        from server.context_compressor import get_compressor, ContextCompressor
        self._compressor: ContextCompressor | None = get_compressor()

        # Token + cost tracking
        self._tokens_in:  int   = 0
        self._tokens_out: int   = 0
        self._cost:       float = 0.0

        # Wire bus subscriptions
        bus.subscribe("user.request",   self._on_user_request)
        bus.subscribe("heartbeat.tick", self._on_tick)
        bus.subscribe("trigger.fired",  self._on_trigger)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        loop = asyncio.get_event_loop()
        self._dispatch_task = loop.create_task(
            self._dispatch_loop(), name="kernel-dispatch"
        )
        log.info("EssenceKernel started (model=%s)", self._model)

        # Wire MCP servers asynchronously — non-blocking, errors are just logged
        _ws = Path(__file__).resolve().parent.parent
        _mcp_store = _ws / "memory" / "ledger" / "mcp_servers.json"
        if _mcp_store.exists():
            async def _register_mcp() -> None:
                try:
                    from server.tools_engine import register_mcp_tools, init_mcp
                    init_mcp(_mcp_store)
                    n = await register_mcp_tools(_mcp_store)
                    if n:
                        log.info("Kernel: registered %d MCP tool(s)", n)
                except Exception as _e:
                    log.debug("Kernel: MCP registration skipped: %s", _e)
            loop.create_task(_register_mcp(), name="kernel-mcp-init")

        self._emit_state()

    def stop(self) -> None:
        if self._dispatch_task and not self._dispatch_task.done():
            self._dispatch_task.cancel()

    async def shutdown(self) -> None:
        """Graceful shutdown: persist session summary and analyze for pending goals."""
        self.stop()
        try:
            summary = await self._episodic.end_session(self._ollama, self._model)
            if summary:
                log.info("Session summary: %s", summary)
        except Exception as exc:
            log.warning("Session end failed: %s", exc)

        # Analyze final exchange for incomplete tasks and persist as pending goals
        try:
            from server.goal_tracker import get_goal_tracker
            from server.adaptive import detect_task_type
            turns     = self._episodic.get_recent_turns(n=6, include_skill_outcomes=False)
            user_msgs = [t["content"] for t in turns if t.get("role") == "user"]
            asst_msgs = [t["content"] for t in turns if t.get("role") == "assistant"]
            last_user = user_msgs[-1] if user_msgs else ""
            last_asst = asst_msgs[-1] if asst_msgs else ""
            sid       = getattr(self._episodic, "_session_id", "")
            get_goal_tracker().analyze_session_end(
                last_user, last_asst,
                session_id=str(sid),
                task_type=detect_task_type(last_user),
            )
        except Exception as exc:
            log.debug("Goal tracker session-end analysis failed: %s", exc)

    @property
    def state(self) -> str:
        return self._state.value

    def stats(self) -> dict:
        return {
            "state":          self._state.value,
            "tokens_in":      self._tokens_in,
            "tokens_out":     self._tokens_out,
            "cost":           round(self._cost, 6),
            "procedural_hits": len(self._procedural),
            "queue_depth":    self._work_queue.qsize(),
        }

    # ── State machine ─────────────────────────────────────────────────

    def _transition(self, new_state: CognitiveState) -> None:
        if self._state == new_state:
            return
        old = self._state
        self._state = new_state
        log.debug("Kernel: %s → %s", old.value, new_state.value)
        self._emit_state()

    def _emit_state(self) -> None:
        from server.event_bus import Envelope
        env = Envelope(
            topic="kernel.state",
            source_component="kernel",
            data={"state": self._state.value, "ts": time.time()},
        )
        self._bus.publish_sync(env)

    # ── Bus entry points ──────────────────────────────────────────────

    async def _on_user_request(self, env: Any) -> None:
        item = WorkItem(
            priority=0,
            item_type="user_request",
            payload=env.data,
            task_id=env.task_id or uuid.uuid4().hex[:10],
            command_id=env.command_id,
            autonomy=False,
        )
        await self._work_queue.put((item.priority, item))

    async def _on_tick(self, env: Any) -> None:
        # Heartbeat is low-priority housekeeping only
        item = WorkItem(
            priority=10,
            item_type="heartbeat",
            payload=env.data,
            task_id="heartbeat",
            autonomy=True,
        )
        await self._work_queue.put((item.priority, item))

    async def _on_trigger(self, env: Any) -> None:
        item = WorkItem(
            priority=2,
            item_type="trigger",
            payload=env.data,
            task_id=env.task_id or uuid.uuid4().hex[:10],
            command_id=env.command_id,
            autonomy=True,
        )
        await self._work_queue.put((item.priority, item))

    # ── Dispatch loop — the CPU ────────────────────────────────────────

    async def _dispatch_loop(self) -> None:
        """Process one WorkItem at a time — sequential like a CPU instruction pointer."""
        while True:
            try:
                _, item = await asyncio.wait_for(
                    self._work_queue.get(), timeout=5.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._process(item)
            except Exception as exc:
                log.error("Kernel dispatch error on %s: %s", item.item_type, exc)

    async def _process(self, item: WorkItem) -> None:
        from server.governance import GovernanceContext

        # Governance gate
        ctx = GovernanceContext(
            item_type=item.item_type,
            command_id=item.command_id,
            autonomy=item.autonomy,
        )
        if not self._gov.check(ctx):
            return

        if item.item_type == "user_request":
            await self._handle_user_request(item)
        elif item.item_type == "trigger":
            await self._handle_trigger(item)
        elif item.item_type == "heartbeat":
            await self._handle_heartbeat(item)

    # ── User request pipeline ─────────────────────────────────────────

    async def _handle_user_request(self, item: WorkItem) -> None:
        text             = item.payload.get("text", "")
        task_id          = item.task_id
        images           = item.payload.get("images", [])
        temperature      = float(item.payload.get("temperature", 0.7))
        max_tokens       = int(item.payload.get("max_tokens", 2048))
        compress_forced  = bool(item.payload.get("compress_forced", False))
        compress_topic   = str(item.payload.get("compress_topic", ""))
        json_mode        = bool(item.payload.get("json_mode", False))
        reasoning_mode   = str(item.payload.get("reasoning_mode", ""))

        # Idempotency replay: if this command_id already produced a result, re-stream it
        if item.command_id:
            cached = self._bus.get_cached(item.command_id)
            if cached:
                log.debug("Kernel: replaying cached result for command_id %s", item.command_id)
                cached_response = cached.get("response", "")
                if cached_response:
                    await self._emit_response_tokens(cached_response, task_id)
                self._transition(CognitiveState.IDLE)
                return

        # L2: persist user turn immediately (also to offline session cache)
        self._episodic.append(role="user", content=text, entry_type="turn")
        self._offline.save_turn(task_id, "user", text)

        # Compact episodic store if it's getting long (background, non-blocking)
        asyncio.create_task(
            self._episodic.compact_if_needed(self._ollama, self._model),
            name="episodic-compact",
        )

        # ── Service URL detection ─────────────────────────────────────
        # If the user dropped a URL that looks like an API, ingest it in
        # the background and emit a brief notice — the kernel continues
        # processing the rest of the message without blocking.
        try:
            from server.service_ingestor import extract_urls, looks_like_api_url, get_ingestor
            urls_in_msg = extract_urls(text)
            api_urls    = [u for u in urls_in_msg if looks_like_api_url(u)]
            if api_urls:
                from server.event_bus import Envelope
                for api_url in api_urls:
                    notice = f"\n[ingesting service from {api_url} …]\n"
                    self._bus.publish_sync(Envelope(
                        topic="user.response",
                        source_component="kernel",
                        task_id=task_id,
                        data={"token": notice, "done": False},
                    ))
                    async def _ingest_bg(u=api_url):
                        try:
                            ingestor = get_ingestor()
                            ingestor._ollama = self._ollama
                            ingestor._model  = self._model
                            profile  = await ingestor.ingest(u)
                            result_notice = (
                                f"\n[✓ Learned **{profile.name}** — "
                                f"{len(profile.endpoints)} endpoint(s) now available as tools. "
                                f"Service id: `{profile.id}`]\n"
                            )
                            self._bus.publish_sync(Envelope(
                                topic="user.response",
                                source_component="kernel",
                                task_id=task_id,
                                data={"token": result_notice, "done": False},
                            ))
                        except Exception as exc:
                            log.warning("Kernel: background ingestion failed for %s: %s", u, exc)
                    asyncio.create_task(_ingest_bg(), name=f"service-ingest-{task_id}")
        except Exception as exc:
            log.debug("Kernel: service URL detection error: %s", exc)

        # Planning pass
        self._transition(CognitiveState.PLANNING)
        plan = await self._plan(text, task_id)

        if plan is None or not plan.steps:
            # Pure conversational — stream directly
            self._transition(CognitiveState.EXECUTING)
            response = await self._stream_response(
                text, task_id, images, temperature, max_tokens,
                compress_forced=compress_forced, compress_topic=compress_topic,
                json_mode=json_mode, reasoning_mode=reasoning_mode,
            )
            # L2: persist assistant turn
            if response:
                self._episodic.append(role="assistant", content=response, entry_type="turn")
            # Auto-generate session title from first exchange (background thread)
            if response:
                try:
                    from server.title_generator import maybe_auto_title
                    maybe_auto_title(
                        self._episodic, text, response,
                        self._ollama, self._model,
                    )
                except Exception:
                    pass
            # Cache result for idempotency replay
            if response and item.command_id:
                self._bus.cache_result(item.command_id, {"response": response})
            self._transition(CognitiveState.IDLE)
            return

        # Skill execution
        self._transition(CognitiveState.EXECUTING)
        self._current_plan = plan
        results = await self._execute_plan(plan, task_id)

        # Synthesis: stream a final summary response after skill execution
        synthesis_input = self._build_synthesis_prompt(text, plan, results)
        self._transition(CognitiveState.EXECUTING)
        response = await self._stream_response(synthesis_input, task_id, [], temperature, max_tokens)

        # Reflection (also writes to episodic)
        self._transition(CognitiveState.REFLECTING)
        await self._reflect(plan, results, text)

        # L2: persist assistant synthesis
        if response:
            self._episodic.append(role="assistant", content=response, entry_type="turn")
        # Cache result for idempotency replay
        if response and item.command_id:
            self._bus.cache_result(item.command_id, {"response": response})
        self._transition(CognitiveState.IDLE)

    # ── 3-Pass Planner ────────────────────────────────────────────────

    async def _plan(self, intent: str, task_id: str) -> Plan | None:
        """
        Pass 1: Procedural memory  — ~0ms, pure dict lookup
        Pass 2: Capability match   — ~1ms, keyword scoring
        Pass 3: LLM decomposition  — ~1-3s, Ollama call
        """

        # Pass 1: procedural memory (successful past plans)
        key = _intent_key(intent)
        if key in self._procedural:
            p = self._procedural[key]
            log.debug("Planner P1 (procedural): %r", intent[:60])
            return p

        # Pass 2: single-skill capability match
        agent = self._skills.match_capability(intent)
        if agent:
            log.debug("Planner P2 (capability): matched %r", agent.manifest.id)
            return Plan(
                id=uuid.uuid4().hex[:8],
                intent=intent,
                goal=f"Run {agent.manifest.name} for: {intent}",
                steps=[PlanStep(
                    skill_id=agent.manifest.id,
                    input_map={"query": intent, "text": intent},
                )],
                source="capability",
            )

        # Pass 3: LLM decomposition
        log.debug("Planner P3 (LLM): decomposing %r", intent[:60])
        llm_plan = await self._llm_plan(intent, task_id)

        # Pass 3b: MCTS — re-rank LLM plan when intent is complex
        if llm_plan and llm_plan.steps:
            try:
                from server.mcts_planner import get_mcts_planner, intent_complexity
                if intent_complexity(intent) >= 0.6:
                    catalog = self._skills.catalog_for_planner()
                    mem_ctx = self._memory.build_context_block(n=4)
                    mcts    = get_mcts_planner(self._ollama, self._model)
                    best    = await mcts.plan(intent, catalog, mem_ctx)
                    if best and best.get("steps"):
                        log.debug("Planner P3b (MCTS): selected higher-scoring plan")
                        llm_plan = Plan(
                            id=uuid.uuid4().hex[:8],
                            intent=intent,
                            goal=best.get("goal", intent),
                            steps=[
                                PlanStep(
                                    skill_id=s.get("skill_id", ""),
                                    input_map=s.get("input_map", {"query": intent}),
                                    depends_on=s.get("depends_on", []),
                                    required=s.get("required", True),
                                )
                                for s in best["steps"] if s.get("skill_id")
                            ],
                            source="mcts",
                        )
            except Exception as _mce:
                log.debug("Planner P3b (MCTS) skipped: %s", _mce)

        if llm_plan:
            return llm_plan

        # Pass 4: HTN decomposition — pattern-based fallback
        try:
            from server.htn_planner import get_htn_planner, plan_to_steps
            htn_node = get_htn_planner().plan(intent)
            if htn_node:
                log.debug("Planner P4 (HTN): matched method %r", htn_node.name)
                raw_steps = plan_to_steps(htn_node)
                return Plan(
                    id=uuid.uuid4().hex[:8],
                    intent=intent,
                    goal=f"HTN: {htn_node.name} for: {intent}",
                    steps=[
                        PlanStep(
                            skill_id=s["skill_id"],
                            input_map=s.get("input_map", {}),
                            depends_on=s.get("depends_on", []),
                            required=s.get("required", True),
                        )
                        for s in raw_steps if s.get("skill_id")
                    ],
                    source="htn",
                )
        except Exception as _hte:
            log.debug("Planner P4 (HTN) skipped: %s", _hte)

        return None

    async def _llm_plan(self, intent: str, task_id: str) -> Plan | None:
        catalog = self._skills.catalog_for_planner()

        # Merge learned service endpoints as pseudo-skills in the planner catalog
        try:
            from server.service_registry import get_service_registry
            reg      = get_service_registry()
            svc_ctx  = reg.context_for_task(intent)
            svc_defs = reg.get_dynamic_tool_definitions()
            if svc_defs:
                for td in svc_defs[:20]:   # cap to avoid token overrun
                    fn = td["function"]
                    catalog.append({
                        "id":          fn["name"],
                        "name":        fn["name"],
                        "description": fn["description"],
                        "type":        "service_tool",
                    })
        except Exception:
            svc_ctx = ""

        if not catalog:
            return None  # no skills or services → pure chat

        system = (
            "You are a planning assistant for an autonomous AI agent.\n"
            "Given a user intent and available skills (including learned API tools), "
            "produce a concise JSON plan.\n"
            "If the intent is purely conversational (no tool needed), return {\"steps\":[]}.\n\n"
            f"Available skills and service tools:\n{json.dumps(catalog, indent=2)}\n\n"
            "Respond ONLY with valid JSON (no markdown fences):\n"
            '{"goal": "string", "steps": ['
            '{"skill_id": "string", "input_map": {}, "depends_on": [], "required": true}'
            "]}"
        )

        memory_ctx = self._memory.build_context_block(n=6)
        user_msg = f"Intent: {intent}\n\nRelevant memory:\n{memory_ctx}"

        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as c:
                r = await c.post(
                    f"{self._ollama}/api/chat",
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user",   "content": user_msg},
                        ],
                        "stream": False,
                        "options": {"temperature": 0.05, "num_predict": 512},
                    },
                )
            content = r.json().get("message", {}).get("content", "{}") or "{}"

            # Strip any markdown code fences if the model added them
            import re
            m = re.search(r'\{[\s\S]*\}', content)
            if not m:
                return None
            data  = json.loads(m.group())
            steps = data.get("steps", [])
            if not steps:
                return None  # conversational

            return Plan(
                id=uuid.uuid4().hex[:8],
                intent=intent,
                goal=data.get("goal", intent),
                steps=[
                    PlanStep(
                        skill_id=s.get("skill_id", ""),
                        input_map=s.get("input_map", {"query": intent}),
                        depends_on=s.get("depends_on", []),
                        required=s.get("required", True),
                    )
                    for s in steps
                    if s.get("skill_id")
                ],
                source="llm",
            )

        except Exception as exc:
            log.warning("LLM planner failed: %s — falling back to chat", exc)
            return None

    # ── Skill Executor ────────────────────────────────────────────────

    async def _execute_plan(self, plan: Plan, task_id: str) -> list:
        """
        Parallel DAG execution: steps with no unresolved depends_on
        run concurrently; dependent steps wait for their prerequisites.
        Falls back to sequential if all steps have no depends_on.
        """
        from server.event_bus import Envelope
        from server.skill_agent import SkillContext

        shared_ctx: dict = {}
        results:    list = []
        completed:  set[str] = set()

        # Build step-id → step index map for depends_on resolution
        # (PlanStep doesn't have an id, so we use skill_id as proxy)
        remaining = [s for s in plan.steps if s.skill_id]
        aborted   = False

        while remaining and not aborted:
            # Collect steps whose dependencies are all satisfied
            ready = [
                s for s in remaining
                if all(dep in completed for dep in s.depends_on)
            ]
            if not ready:
                # Circular dependency or unresolvable dep — run first step
                ready = [remaining[0]]

            # Execute all ready steps concurrently
            step_results = await asyncio.gather(
                *[self._execute_single_step(s, plan, task_id, shared_ctx)
                  for s in ready],
                return_exceptions=True,
            )

            for step, result in zip(ready, step_results):
                remaining.remove(step)
                if isinstance(result, Exception):
                    from server.skill_agent import SkillResult
                    result = SkillResult(
                        skill_id=step.skill_id, status="error", error=str(result),
                    )
                results.append(result)
                completed.add(step.skill_id)

                if hasattr(result, "status"):
                    if result.status == "done" and result.output:
                        if isinstance(result.output, dict):
                            shared_ctx[step.skill_id] = result.output
                        else:
                            shared_ctx[step.skill_id] = {"result": str(result.output)[:800]}

                    if result.status == "error":
                        self._gov.on_skill_failure(step.skill_id)
                        if step.required:
                            aborted = True
                            break
                    elif result.status == "done":
                        self._gov.on_skill_success(step.skill_id)

        return results

    async def _execute_single_step(
        self,
        step:       "PlanStep",
        plan:       "Plan",
        task_id:    str,
        shared_ctx: dict,
    ):
        """Execute one plan step; returns SkillResult or raises."""
        from server.event_bus import Envelope
        from server.skill_agent import SkillContext, SkillResult

        if not step.skill_id:
            return SkillResult(skill_id="", status="skipped")

        # Service tool step
        if step.skill_id.startswith("svc_"):
            from server.tools_engine import execute_tool
            result_text = await execute_tool(step.skill_id, dict(step.input_map))
            sr = SkillResult(skill_id=step.skill_id, status="done", output=result_text)
            self._bus.publish_sync(Envelope(
                topic="skill.result", source_component="kernel", task_id=task_id,
                data={"skill_id": step.skill_id, "result": result_text},
            ))
            return sr

        agent = self._skills.get(step.skill_id)
        if agent is None:
            log.warning("Kernel: unknown skill %r — skipping", step.skill_id)
            return SkillResult(skill_id=step.skill_id, status="skipped",
                               error="skill not found")

        from server.governance import GovernanceContext
        step_ctx = GovernanceContext(
            item_type="skill_execute",
            skill_id=step.skill_id,
            autonomy=True,
            requires_caps=agent.manifest.requires_caps,
        )
        if not self._gov.check(step_ctx):
            return SkillResult(skill_id=step.skill_id, status="skipped",
                               error="governance blocked")

        self._bus.publish_sync(Envelope(
            topic="skill.execute", source_component="kernel", task_id=task_id,
            data={"skill_id": step.skill_id, "intent": plan.intent},
        ))

        skill_ctx = SkillContext(
            task_id=task_id,
            intent=plan.intent,
            input_data=_resolve_input(step.input_map, shared_ctx),
            shared_context=shared_ctx,
            memory_block=self._memory.build_context_block(n=5),
            identity=self._identity.load(),
            command_id=f"cmd-{uuid.uuid4().hex[:12]}",
        )

        result = await agent.safe_execute(skill_ctx)
        self._bus.publish_sync(Envelope(
            topic="skill.result", source_component="kernel", task_id=task_id,
            data=result.to_dict(),
        ))
        return result

    # ── Reflector ─────────────────────────────────────────────────────

    async def _reflect(self, plan: Plan, results: list, original_intent: str) -> None:
        from server.event_bus import Envelope

        all_ok  = all(r.status in ("done", "skipped") for r in results)

        # --- L2 Episodic: write one entry per skill outcome ---
        for r in results:
            if r.status in ("done", "error"):
                raw_out = r.output
                if isinstance(raw_out, dict):
                    # Pull the most useful text field
                    out_str = (
                        raw_out.get("summary")
                        or raw_out.get("text")
                        or raw_out.get("result", "")
                        or str(raw_out)
                    )
                else:
                    out_str = str(raw_out) if raw_out else ""
                self._episodic.append_skill_outcome(
                    intent   = original_intent,
                    skill_id = r.skill_id,
                    status   = r.status,
                    summary  = (out_str[:400] if out_str else r.error[:200]),
                    plan_id  = plan.id,
                )

        # --- L2 Episodic: write one compact plan-level summary ---
        skill_names = [s.skill_id for s in plan.steps]
        outcome_summary = (
            f"Ran {', '.join(skill_names)}. "
            + ("All steps succeeded." if all_ok
               else f"{sum(1 for r in results if r.status=='error')} step(s) failed.")
        )
        self._episodic.append_plan_summary(
            intent  = original_intent,
            goal    = plan.goal,
            success = all_ok,
            summary = outcome_summary,
        )

        # --- L3 Gravity memory signals ---
        mem_key = original_intent[:100]
        if all_ok:
            try:
                self._memory.signal(mem_key, "confirmed")    # G += 0.20
            except KeyError:
                self._memory.write(mem_key, f"plan:{plan.id}", skill_source="kernel")

            # --- L4 Procedural: persist successful plan to disk ---
            if plan.source != "procedural":
                ikey = _intent_key(original_intent)
                # Serialise plan to dict for storage
                plan_dict = {
                    "id": plan.id, "intent": plan.intent, "goal": plan.goal,
                    "source": "procedural",
                    "steps": [
                        {"skill_id": s.skill_id, "input_map": s.input_map,
                         "depends_on": s.depends_on, "required": s.required}
                        for s in plan.steps
                    ],
                }
                self._procedural[ikey] = plan
                self._episodic.save_plan(ikey, plan_dict)
                log.debug("Reflector: plan persisted to procedural memory (%d total)",
                          len(self._procedural))
        else:
            try:
                self._memory.signal(mem_key, "corrected")    # G += 0.15
            except KeyError:
                pass

        # --- Adaptive: record task outcome into performance ledger ---
        try:
            from server.adaptive import get_adaptive_engine, detect_task_type
            _task_type = detect_task_type(original_intent)
            get_adaptive_engine().record_task(
                provider=self._provider, model=self._model,
                task_type=_task_type, success=all_ok,
            )
        except Exception:
            pass

        # --- Governance audit ---
        self._bus.publish_sync(Envelope(
            topic="governance.audit",
            source_component="kernel",
            data={
                "plan_id":         plan.id,
                "plan_source":     plan.source,
                "success":         all_ok,
                "step_count":      len(plan.steps),
                "result_statuses": [r.status for r in results],
            },
        ))

    # ── Trigger handler ───────────────────────────────────────────────

    async def _handle_trigger(self, item: WorkItem) -> None:
        """Autonomous trigger: run the associated skill directly."""
        skill_id = item.payload.get("skill_id", "")
        if not skill_id:
            return
        from server.governance import GovernanceContext
        from server.skill_agent import SkillContext

        agent = self._skills.get(skill_id)
        if agent is None or not agent.is_healthy():
            return

        ctx = GovernanceContext(
            item_type="trigger",
            skill_id=skill_id,
            autonomy=True,
            requires_caps=agent.manifest.requires_caps,
        )
        if not self._gov.check(ctx):
            return

        skill_ctx = SkillContext(
            task_id=item.task_id,
            intent=item.payload.get("intent", ""),
            input_data=item.payload.get("input", {}),
            shared_context={},
            memory_block=self._memory.build_context_block(n=5),
            identity=self._identity.load(),
        )

        self._transition(CognitiveState.EXECUTING)
        result = await agent.safe_execute(skill_ctx)

        if result.status == "done":
            self._gov.on_skill_success(skill_id)
        else:
            self._gov.on_skill_failure(skill_id)

        # Surface the result to the user via user.response
        if result.output:
            from server.event_bus import Envelope
            text = (result.output.get("text", "") if isinstance(result.output, dict)
                    else str(result.output))
            if text:
                self._bus.publish_sync(Envelope(
                    topic="user.response",
                    source_component="kernel",
                    task_id=item.task_id,
                    data={"token": f"[{agent.manifest.name}] {text}", "done": True,
                          "source": "trigger"},
                ))

        self._transition(CognitiveState.IDLE)

    # ── Heartbeat handler ─────────────────────────────────────────────

    async def _handle_heartbeat(self, item: WorkItem) -> None:
        """
        Periodic housekeeping tick.
        1. GoalTracker: resurface pending goals at session start
        2. Knowledge Graph: extract entities from recent episodic turns
        3. CronExecutor: forward tick to fire matching cron jobs
        """
        # GoalTracker — resurface unfinished goals (once per session)
        try:
            from server.goal_tracker import get_goal_tracker
            tracker = get_goal_tracker()
            pending = tracker.get_pending(max_resurface=2)
            if pending:
                resurface_msg = tracker.format_resurface(pending)
                if resurface_msg:
                    from server.event_bus import Envelope
                    self._bus.publish_sync(Envelope(
                        topic="user.response",
                        source_component="kernel",
                        data={"token": resurface_msg, "done": False, "source": "goal_tracker"},
                    ))
                    tracker.mark_resurfaced(*[g.id for g in pending])
        except Exception as exc:
            log.debug("Kernel heartbeat: GoalTracker error: %s", exc)

        # Knowledge Graph — background entity extraction from recent turns
        try:
            from server.knowledge_graph import get_knowledge_graph
            kg = get_knowledge_graph()
            recent = self._episodic.get_recent_turns(n=3, include_skill_outcomes=False)
            for t in recent:
                if t.get("role") == "assistant" and t.get("content"):
                    kg.extract_from_text(t["content"][:1000], source="episodic")
        except Exception as exc:
            log.debug("Kernel heartbeat: KG extraction error: %s", exc)

        # CronExecutor — forward the tick
        try:
            from server.cron_executor import CronExecutor as _CE
            _ws = _get_workspace()
            _ce = _CE(_ws, self._bus)
            await _ce.on_tick(item.payload.get("ts"))
        except Exception as exc:
            log.debug("Kernel heartbeat: CronExecutor error: %s", exc)

    # ── Conversational streaming ──────────────────────────────────────

    async def _stream_response(
        self,
        text: str,
        task_id: str,
        images: list,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        compress_forced: bool = False,
        compress_topic: str = "",
        json_mode: bool = False,
        reasoning_mode: str = "",
    ) -> str:
        """
        Stream LLM tokens to the TUI via user.response bus events.
        Returns the full response string for episodic memory.
        temperature and max_tokens are forwarded from the originating user.request
        envelope so that TUI settings (/temp, /tokens) take effect immediately.
        """
        from server.event_bus import Envelope

        system_prompt = self._identity.get_system_prompt()
        # L3 semantic memory — multi-signal retrieval (BM25 + entity + gravity fusion)
        gravity_ctx   = self._memory.build_context_block(n=10, query=text)
        episodic_ctx  = self._episodic.build_context_block(        # L2 episodic (recency)
            n_turns=12, n_past_sessions=CROSS_SESSION_N, include_outcomes=True
        )
        # L2 semantic recall — vector similarity over stored episodes (supplements recency)
        semantic_ctx = ""
        try:
            _sem_hits = await self._episodic.semantic_search(text, n=4)
            if _sem_hits:
                _sem_lines = ["## Semantically related past exchanges"]
                for h in _sem_hits:
                    _label = {"user": "User", "assistant": "Essence"}.get(h["role"], "Context")
                    _text  = h["content"][:300] + ("…" if len(h["content"]) > 300 else "")
                    _sem_lines.append(f"{_label}: {_text}")
                semantic_ctx = "\n".join(_sem_lines)
        except Exception:
            pass
        # Learned external services — injected so the model knows what APIs are available
        service_ctx = ""
        try:
            from server.service_registry import get_service_registry
            service_ctx = get_service_registry().context_for_task(text)
        except Exception:
            pass
        # Adaptive style suffix — dynamically calibrated from interaction history
        _style_suffix = ""
        try:
            from server.adaptive import get_adaptive_engine
            _style_suffix = get_adaptive_engine().get_style_suffix()
        except Exception:
            pass

        # Git context — injected when workspace is inside a git repo
        _git_ctx = ""
        try:
            _ws = _get_workspace()
            from server.git_context import get_cached_git_context
            _git_ctx = get_cached_git_context(_ws, max_chars=1000)
        except Exception:
            pass

        # Knowledge Graph context — ego-subgraph of entities in the query
        _kg_ctx = ""
        try:
            from server.knowledge_graph import get_knowledge_graph
            _kg = get_knowledge_graph()
            # Extract first capitalized token as seed entity
            import re as _re
            _seed = _re.search(r'\b[A-Z][a-z]{2,}\b', text)
            if _seed:
                _kg_ctx = _kg.ego_context(_seed.group(), depth=2)
        except Exception:
            pass

        system_full = "\n\n".join(filter(None, [
            system_prompt, gravity_ctx, episodic_ctx, semantic_ctx,
            service_ctx, _git_ctx, _kg_ctx, _style_suffix,
        ]))

        messages = [{"role": "system", "content": system_full}]

        # Inject raw conversation turns from episodic (role=user/assistant only)
        for turn in self._episodic.get_recent_turns(n=10, include_skill_outcomes=False):
            # Avoid re-adding the current user message that's about to be appended
            if turn["content"] == text and turn["role"] == "user":
                continue
            messages.append({"role": turn["role"], "content": turn["content"]})

        # Ensure the current message is last — attach images if provided
        if not messages or messages[-1].get("content") != text:
            if images:
                try:
                    from server.multimodal import build_content_with_images
                    _content = build_content_with_images(text, images, provider="openai")
                except Exception as _me:
                    log.warning("Kernel: image content build failed (%s) — sending text only", _me)
                    _content = text
            else:
                _content = text
            messages.append({"role": "user", "content": _content})

        # Context compression — condense old turns when context is full or forced by /compress
        if self._compressor and (compress_forced or self._compressor.should_compress(messages)):
            try:
                messages = self._compressor.compress(
                    messages, focus_topic=compress_topic, force=compress_forced
                )
                log.info(
                    "Context compressed (#%d): %d msgs in window%s",
                    self._compressor.compression_count, len(messages),
                    f" [focus: {compress_topic}]" if compress_topic else "",
                )
            except Exception as _ce:
                log.warning("Context compression failed (continuing uncompressed): %s", _ce)

        # Offline guard: if backend is known-down, queue and return
        if self._offline.probe.is_online is False:
            msg_id = self._offline.enqueue_message({"text": text, "task_id": task_id})
            offline_msg = f"\n[offline — message queued ({msg_id}); will send when backend reconnects]"
            self._bus.publish_sync(Envelope(
                topic="user.response",
                source_component="kernel",
                task_id=task_id,
                data={"token": offline_msg, "done": True},
            ))
            return offline_msg

        # Lazy imports for new resilience modules
        import httpx
        from server.prompt_caching   import apply_anthropic_cache_control, should_apply_cache_control
        from server.rate_limit_tracker import parse_rate_limit_headers, get_tracker as _get_rl_tracker
        from server.error_classifier import classify_api_error, FailoverReason
        from server.retry_utils      import jittered_backoff
        from server.usage_pricing    import CanonicalUsage, get_usage_tracker
        from server.redact           import redact_messages, should_redact

        # Build ALL tool definitions (will be filtered per-request below)
        _all_tools: list[dict] = []
        try:
            from server.tools_engine import TOOL_DEFINITIONS
            _all_tools = list(TOOL_DEFINITIONS)
        except Exception:
            pass

        # ── Dynamic routing: classify the request and select the best provider ──
        # When self._dynamic_routing is True (default), each request is routed
        # to the most capable available provider for that specific task type.
        # The user-configured provider is still honoured when it scores well.
        _native_tools: list[dict] = _all_tools  # default: send all tools
        if self._dynamic_routing:
            try:
                _dr = self._dynamic_router
                # Update provider keys so the scorer knows what's available
                _dr.update_available(self._router._provider_cfg)
                _profile = _dr.classify(
                    text=text,
                    history=messages,
                    images=[],        # images injected separately via multimodal
                    tool_defs=_all_tools,
                )
                # Route — passes current provider as "preferred" so we don't
                # switch unnecessarily; dynamic router only overrides when
                # the preferred provider scores poorly for this task.
                candidates = _dr.route(
                    profile=_profile,
                    preferred=(self._provider, self._model),
                )
                # Filter tools to only the relevant subset (reduces noise)
                _native_tools = _dr.select_tools(_profile, _all_tools)
                log.debug(
                    "DynamicRouter: task=%s → %s/%s  tools=%s",
                    _profile.task_type,
                    candidates[0][0], candidates[0][1],
                    [t.get("function", t).get("name") for t in _native_tools],
                )
            except Exception as _dre:
                log.debug("Dynamic routing failed (using static): %s", _dre)
                # Fall through to static routing below
                candidates = None

        if not self._dynamic_routing or not candidates:
            # Static fallback: active provider first, then configured alternates
            primary   = (self._provider, self._model)
            fallbacks = self._router.fallback_chain(exclude_provider=self._provider)
            candidates = [primary] + fallbacks

        full_response = ""
        succeeded     = False

        for attempt, (provider, model) in enumerate(candidates):
            # Apply Anthropic prompt cache control (system_and_3 strategy — ~75% cost savings)
            send_messages = messages
            if should_apply_cache_control(provider):
                try:
                    send_messages = apply_anthropic_cache_control(messages)
                except Exception as _ce:
                    log.debug("Prompt cache control failed (using original): %s", _ce)

            # Redact secrets/PII before transmitting to cloud providers
            # (url not yet resolved — provider string is sufficient for local-skip logic)
            if should_redact(provider):
                try:
                    send_messages = redact_messages(send_messages)
                except Exception as _re:
                    log.debug("Redaction failed (sending unredacted): %s", _re)

            from server.tool_bridge import get_tool_bridge as _get_tb
            _tb = _get_tb()

            url, body, headers = self._router.normalize_to_openai(
                provider_id=provider,
                messages=send_messages,
                model=model,
                tools=_native_tools if _tb.supports_tools(provider) else None,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format="json_object" if json_mode else "",
                reasoning_mode=reasoning_mode,
            )

            if attempt > 0:
                notice = f"\n[{provider}/{model} — fallback attempt {attempt}]\n"
                log.warning("Kernel: falling back to %s/%s", provider, model)
                self._bus.publish_sync(Envelope(
                    topic="user.response",
                    source_component="kernel",
                    task_id=task_id,
                    data={"token": notice, "done": False},
                ))

            t_start = time.time()
            try:
                async with httpx.AsyncClient(timeout=180.0) as c:
                    async with c.stream("POST", url, json=body, headers=headers) as resp:
                        resp.raise_for_status()

                        # Capture rate limit headers from each response
                        try:
                            rl_state = parse_rate_limit_headers(dict(resp.headers), provider=provider)
                            if rl_state:
                                _get_rl_tracker().update(provider, rl_state)
                        except Exception:
                            pass

                        accumulator = _tb.new_accumulator(provider)
                        pc = ec = 0
                        usage_dict: dict = {}
                        tool_call_round = 0
                        MAX_TOOL_ROUNDS = 5   # prevent infinite tool-call loops

                        async for line in resp.aiter_lines():
                            if not line or line == "data: [DONE]":
                                continue
                            raw = line.removeprefix("data: ").strip()
                            if not raw:
                                continue
                            try:
                                d = json.loads(raw)
                            except json.JSONDecodeError:
                                continue

                            # Feed chunk into provider-agnostic accumulator
                            tool_calls_ready = accumulator.feed(d)

                            # Emit any text token the accumulator extracted
                            token = accumulator.last_text_token
                            if token:
                                full_response += token
                                self._bus.publish_sync(Envelope(
                                    topic="user.response",
                                    source_component="kernel",
                                    task_id=task_id,
                                    data={"token": token, "done": False},
                                ))
                            # Emit thinking tokens as a separate stream hint (thinking=True flag)
                            thinking_tok = accumulator.last_thinking_token
                            if thinking_tok:
                                self._bus.publish_sync(Envelope(
                                    topic="user.response",
                                    source_component="kernel",
                                    task_id=task_id,
                                    data={"token": thinking_tok, "done": False, "thinking": True},
                                ))

                            # Collect usage from final chunk (OpenAI puts it in usage field)
                            if "usage" in d:
                                usage_dict = d["usage"]
                                pc = usage_dict.get("prompt_tokens", 0)
                                ec = usage_dict.get("completion_tokens", 0)
                            elif d.get("done"):
                                pc = d.get("prompt_eval_count", 0)
                                ec = d.get("eval_count", 0)

                            if tool_calls_ready is not None:
                                # Stream is done for this round
                                if tool_calls_ready and tool_call_round < MAX_TOOL_ROUNDS:
                                    # ── Execute native tool calls ────────────
                                    tool_call_round += 1

                                    # Stream "executing…" progress token per tool
                                    # so the TUI shows activity instead of silence.
                                    for _tc in tool_calls_ready:
                                        _progress = f"\n⚙ `{_tc.name}`…"
                                        self._bus.publish_sync(Envelope(
                                            topic="user.response",
                                            data={"token": _progress, "done": False,
                                                  "tool_progress": True, "tool_name": _tc.name},
                                        ))

                                    # Execute all tool calls concurrently (parallel)
                                    from server.tools_engine import execute_tool
                                    async def _run_one(tc) -> str:
                                        try:
                                            res = await execute_tool(tc.name, tc.arguments)
                                            log.debug(
                                                "Kernel: tool %r → %s…",
                                                tc.name, str(res)[:80],
                                            )
                                            return str(res)
                                        except Exception as te:
                                            log.warning("Kernel: tool %r failed: %s", tc.name, te)
                                            return f"[error: {te}]"

                                    executed_results: list[str] = list(
                                        await asyncio.gather(*(_run_one(tc) for tc in tool_calls_ready))
                                    )

                                    # Build provider-correct tool result messages
                                    result_msgs = _tb.make_tool_result_messages(
                                        provider, tool_calls_ready, executed_results,
                                    )
                                    # Re-append current user turn, assistant tool-call
                                    # turn, and tool results; then re-stream continuation
                                    send_messages = send_messages + result_msgs

                                    # Compress send_messages if it has grown past the
                                    # threshold during this tool-call round (the compressor
                                    # normally checks `messages` but multi-round growth only
                                    # shows up in `send_messages`).
                                    if self._compressor and self._compressor.should_compress(send_messages):
                                        try:
                                            send_messages = await self._compressor.compress(send_messages)
                                        except Exception as _ce:
                                            log.debug("Mid-turn compression failed: %s", _ce)

                                    # Re-normalize for next round (stream=True, no tools
                                    # in follow-up to force a text response)
                                    url, body, headers = self._router.normalize_to_openai(
                                        provider_id=provider,
                                        messages=send_messages,
                                        model=model,
                                        tools=None,      # force text response after tools
                                        stream=True,
                                        temperature=temperature,
                                        max_tokens=max_tokens,
                                    )
                                    # Do NOT break here.  The server will send data:[DONE]
                                    # (or close the connection for Ollama), exhausting
                                    # aiter_lines naturally.  Python's for...else then
                                    # fires the else: block below, which opens the
                                    # follow-up stream using the url/body/headers we
                                    # just prepared.  Calling break would prevent else:
                                    # from running and the follow-up would never be sent.
                                    # Reset tool_calls_ready so we don't re-enter this
                                    # block if an unexpected extra chunk arrives before
                                    # the stream closes.
                                    tool_calls_ready = []

                                else:
                                    # Clean finish (no tool calls or max rounds reached)
                                    latency = time.time() - t_start
                                    canonical = CanonicalUsage.from_response(usage_dict) if usage_dict else CanonicalUsage(
                                        input_tokens=pc, output_tokens=ec
                                    )
                                    call_cost = get_usage_tracker().record(canonical, model=model, provider=provider)
                                    self._tokens_in  += pc
                                    self._tokens_out += ec
                                    from decimal import Decimal as _Dec
                                    if call_cost is not None:
                                        self._cost += float(call_cost)
                                    else:
                                        self._cost += pc * 0.0000002 + ec * 0.0000006
                                    self._router.record_success(provider, pc + ec, latency)
                                    self._bus.publish_sync(Envelope(
                                        topic="user.response",
                                        source_component="kernel",
                                        task_id=task_id,
                                        data={
                                            "token":     "",
                                            "done":      True,
                                            "tokens_in": pc,    "tokens_out": ec,
                                            "total_in":  self._tokens_in,
                                            "total_out": self._tokens_out,
                                            "provider":  provider,
                                            "model":     model,
                                        },
                                    ))
                                    break
                        else:
                            # aiter_lines exhausted naturally (no break).
                            # This is the normal path when tool calls are found —
                            # we deliberately don't break so this else: fires.
                            if tool_call_round > 0:
                                # Re-open the stream with updated messages + no tools
                                async with httpx.AsyncClient(timeout=180.0) as c2:
                                    async with c2.stream("POST", url, json=body, headers=headers) as resp2:
                                        resp2.raise_for_status()
                                        accumulator2 = _tb.new_accumulator(provider)
                                        async for line2 in resp2.aiter_lines():
                                            if not line2 or line2 == "data: [DONE]":
                                                continue
                                            raw2 = line2.removeprefix("data: ").strip()
                                            if not raw2:
                                                continue
                                            try:
                                                d2 = json.loads(raw2)
                                            except json.JSONDecodeError:
                                                continue
                                            done2 = accumulator2.feed(d2)
                                            tok2  = accumulator2.last_text_token
                                            if tok2:
                                                full_response += tok2
                                                self._bus.publish_sync(Envelope(
                                                    topic="user.response",
                                                    source_component="kernel",
                                                    task_id=task_id,
                                                    data={"token": tok2, "done": False},
                                                ))
                                            if "usage" in d2:
                                                usage_dict = d2["usage"]
                                                pc = usage_dict.get("prompt_tokens", 0)
                                                ec = usage_dict.get("completion_tokens", 0)
                                            if done2 is not None:
                                                break
                                        latency = time.time() - t_start
                                        canonical = CanonicalUsage.from_response(usage_dict) if usage_dict else CanonicalUsage(input_tokens=pc, output_tokens=ec)
                                        call_cost = get_usage_tracker().record(canonical, model=model, provider=provider)
                                        self._tokens_in  += pc
                                        self._tokens_out += ec
                                        if call_cost is not None:
                                            self._cost += float(call_cost)
                                        else:
                                            self._cost += pc * 0.0000002 + ec * 0.0000006
                                        self._router.record_success(provider, pc + ec, latency)
                                        self._bus.publish_sync(Envelope(
                                            topic="user.response",
                                            source_component="kernel",
                                            task_id=task_id,
                                            data={"token": "", "done": True,
                                                  "tokens_in": pc, "tokens_out": ec,
                                                  "total_in": self._tokens_in, "total_out": self._tokens_out,
                                                  "provider": provider, "model": model},
                                        ))

                # If we got here without exception, stream succeeded
                succeeded = True
                # Update active provider/model to the one that worked
                if provider != self._provider or model != self._model:
                    log.info("Kernel: active provider updated to %s/%s after fallback", provider, model)
                    self._provider = provider
                    self._model    = model
                break

            except Exception as exc:
                # Structured error classification — determines recovery action
                approx_tokens = sum(len(str(m.get("content", ""))) // 4 for m in messages)
                classified = classify_api_error(
                    exc, provider=provider, model=model,
                    approx_tokens=approx_tokens, num_messages=len(messages),
                )
                log.warning(
                    "Kernel: %s/%s failed (attempt %d): %s",
                    provider, model, attempt + 1, classified,
                )
                self._router.record_error(
                    provider,
                    is_rate_limit=classified.reason == FailoverReason.rate_limit,
                )

                # Context overflow: compress and retry on the SAME provider
                if classified.should_compress and self._compressor:
                    try:
                        messages = self._compressor.compress(messages, force=True)
                        log.info("Kernel: compressed context after overflow (%d msgs)", len(messages))
                        # Retry same provider — don't advance to next candidate
                        continue
                    except Exception as _cce:
                        log.warning("Kernel: compression after overflow failed: %s", _cce)

                # Non-retryable or should-fallback → advance to next candidate
                if not classified.retryable or classified.should_fallback:
                    # Apply jittered backoff before fallback to avoid hammering
                    if attempt < len(candidates) - 1:
                        delay = jittered_backoff(attempt + 1, base_delay=2.0, max_delay=20.0)
                        await asyncio.sleep(delay)
                    # Loop continues to next candidate naturally

                elif classified.retryable:
                    # Same provider retry with backoff (rate limit / transient server error)
                    delay = jittered_backoff(attempt + 1, base_delay=5.0, max_delay=60.0)
                    log.info("Kernel: retrying %s/%s in %.1fs (reason=%s)", provider, model, delay, classified.reason.value)
                    await asyncio.sleep(delay)
                    continue  # retry same provider

                # Credential rotation: try next key in pool before falling to next provider
                if classified.should_rotate_credential:
                    try:
                        from server.credential_pool import rotate_credential
                        new_cred = rotate_credential(
                            provider,
                            status_code=classified.status_code,
                        )
                        if new_cred:
                            # Patch the router with the rotated key and retry same provider
                            self._router.set_provider_key(provider, new_cred.api_key, new_cred.base_url)
                            log.info("Kernel: rotated credential for %s → %s", provider, new_cred.short_label())
                            continue  # retry same provider with new key
                    except Exception as _re:
                        log.debug("Kernel: credential rotation failed: %s", _re)

                if attempt == len(candidates) - 1:
                    # All candidates exhausted
                    if classified.reason == FailoverReason.timeout or classified.should_rotate_credential:
                        msg_id    = self._offline.enqueue_message({"text": text, "task_id": task_id})
                        err_token = f"\n[all providers failed — message queued ({msg_id})]"
                    else:
                        err_token = f"\n[all providers failed: {classified.message[:200]}]"
                    full_response += err_token
                    self._bus.publish_sync(Envelope(
                        topic="user.response",
                        source_component="kernel",
                        task_id=task_id,
                        data={"token": err_token, "done": True},
                    ))

        # Persist the turn to offline session cache for resilience
        if full_response:
            self._offline.save_turn(task_id, "assistant", full_response)

        # Record turn signal for adaptive calibration (non-blocking)
        if full_response:
            try:
                from server.adaptive import get_adaptive_engine
                get_adaptive_engine().record_turn(
                    text, full_response,
                    provider=self._provider, model=self._model,
                )
            except Exception:
                pass

        return full_response

    async def _replay_queued_messages(self, queued: list[dict]) -> None:
        """Called by OfflineCache when the backend reconnects — replays buffered messages."""
        log.info("Kernel: replaying %d queued messages after reconnect", len(queued))
        from server.event_bus import Envelope
        for item in queued:
            payload = item.get("payload", {})
            text    = payload.get("text", "")
            task_id = payload.get("task_id", uuid.uuid4().hex[:10])
            if not text:
                continue
            await self._stream_response(text, task_id, [])

    async def _emit_response_tokens(self, text: str, task_id: str) -> None:
        """Re-emit a previously-produced response for idempotency replay."""
        from server.event_bus import Envelope
        if text:
            self._bus.publish_sync(Envelope(
                topic="user.response",
                source_component="kernel",
                task_id=task_id,
                data={"token": text, "done": False},
            ))
        self._bus.publish_sync(Envelope(
            topic="user.response",
            source_component="kernel",
            task_id=task_id,
            data={"token": "", "done": True},
        ))

    # ── Synthesis prompt ──────────────────────────────────────────────

    def _build_synthesis_prompt(self, intent: str, plan: Plan, results: list) -> str:
        """
        Build the final-answer prompt after skills have run.
        Includes:
          - Episodic context so the LLM knows what happened earlier in the session
          - Each skill's output (truncated)
          - Instruction to synthesise a clear response
        """
        lines: list[str] = []

        # Recent episodic context — gives the LLM session continuity
        ep_ctx = self._episodic.build_context_block(
            n_turns=6, n_past_sessions=1, include_outcomes=False
        )
        if ep_ctx.strip():
            lines.append(ep_ctx.strip())
            lines.append("---")

        lines.append(f"User request: {intent}\n")

        for r in results:
            if r.status == "done" and r.output:
                if isinstance(r.output, dict):
                    # Prefer prose fields over raw JSON
                    text_out = (
                        r.output.get("summary")
                        or r.output.get("text")
                        or r.output.get("result", "")
                    )
                    if text_out:
                        lines.append(f"[{r.skill_id}]\n{str(text_out)[:1200]}")
                    else:
                        lines.append(f"[{r.skill_id}]\n{json.dumps(r.output, indent=2)[:1200]}")
                else:
                    lines.append(f"[{r.skill_id}]\n{str(r.output)[:1200]}")
            elif r.status == "error":
                lines.append(f"[{r.skill_id} failed: {r.error}]")
            elif r.status == "skipped":
                lines.append(f"[{r.skill_id} skipped]")

        lines.append(
            "\nUsing the above results, write a clear and concise response to the user's request. "
            "Do not repeat the raw data — synthesise it into a helpful answer."
        )
        return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_workspace() -> Path:
    """Return workspace path from tools_engine (set at startup)."""
    try:
        from server.tools_engine import workspace as _ws
        return _ws()
    except Exception:
        return Path(__file__).resolve().parent.parent


def _intent_key(intent: str) -> str:
    """Stable 16-char key for procedural memory lookup."""
    return hashlib.sha256(intent.lower().strip()[:200].encode()).hexdigest()[:16]


def _resolve_input(input_map: dict, ctx: dict) -> dict:
    """
    Resolve ${skill_id.field} references in input_map against shared context.
    Falls back to the literal string if the path is not found.
    """
    resolved = {}
    for k, v in input_map.items():
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            path = v[2:-1].split(".")
            val: Any = ctx
            for part in path:
                val = val.get(part, "") if isinstance(val, dict) else ""
            resolved[k] = val or v
        else:
            resolved[k] = v
    return resolved


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_kernel: EssenceKernel | None = None


def init_kernel(
    bus:            Any,
    memory:         Any,
    identity:       Any,
    governance:     Any,
    skill_registry: Any,
    ollama_url:     str,
    model:          str,
    episodic:       Any = None,
) -> EssenceKernel:
    """Initialise and return the global kernel. Call once at startup."""
    global _kernel
    _kernel = EssenceKernel(bus, memory, identity, governance, skill_registry,
                            ollama_url, model, episodic=episodic)
    return _kernel


def get_kernel() -> EssenceKernel:
    if _kernel is None:
        raise RuntimeError("Kernel not initialised. Call init_kernel() at startup.")
    return _kernel
