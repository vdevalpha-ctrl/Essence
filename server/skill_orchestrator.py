"""
skill_orchestrator.py — AURA Skill Orchestration Engine
========================================================
Skills are the fundamental unit of capability in Essence. Each skill is a
named, versioned, composable agent that:

  • Has a clear input/output contract
  • Can be invoked standalone or chained with other skills
  • Runs in its own async context with timeout + retry
  • Reports progress via SSE events
  • Can spawn sub-skills (nested orchestration)

Architecture
------------
  SkillRegistry  — loads and stores skill definitions from disk + built-ins
  SkillRunner    — executes a single skill (LLM-powered or tool-powered)
  Orchestrator   — builds and executes skill chains (DAGs) with deadlines
  TaskQueue      — persistent queue of pending/running/done orchestrated tasks
  NotificationBus— delivers alerts via SSE + (optionally) webhook/email

Built-in skills
---------------
  research    — web search + summarise (http_get × N → llm summary)
  write_doc   — produce structured document from outline + research
  code        — generate, test, and save code to workspace
  summarise   — compress long text into structured summary
  calendar    — read/write calendar events (multi-provider)
  remind      — schedule a future notification
  remember    — persist a fact to MEMORY.md
  extract_pdf — pull text/tables from PDF file
  email       — draft or send email (via configured SMTP)
  search_web  — HTTP GET + HTML strip for multiple queries
  news        — fetch RSS/news feeds and summarise

Skill chain example (research paper by deadline)
-------------------------------------------------
  orchestrator.run_chain([
    {"skill": "research",  "input": {"query": "quantum error correction 2025"}},
    {"skill": "write_doc", "input": {"format": "pdf", "template": "research"}},
    {"skill": "remind",    "input": {"when": "2026-04-25T09:00", "message": "Paper draft ready"}},
  ], deadline="2026-04-25", notify=True)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger("essence.orchestrator")


# ---------------------------------------------------------------------------
# Skill definition
# ---------------------------------------------------------------------------
@dataclass
class SkillDef:
    id:          str
    name:        str
    description: str
    category:    str            # research | write | code | calendar | notify | memory | tool
    input_schema: dict          # JSON Schema of expected input
    output_schema: dict         # JSON Schema of output
    system_prompt: str = ""     # Injected when using LLM to run this skill
    enabled:     bool  = True
    timeout_s:   int   = 120
    max_retries: int   = 2
    tags:        list[str] = field(default_factory=list)
    version:     str   = "1.0"
    skill_type:  str   = "builtin"  # builtin | json_store | mcp


# ---------------------------------------------------------------------------
# Task / step state
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    skill_id:  str
    status:    str            # pending | running | done | error | skipped
    input:     dict
    output:    Any   = None
    error:     str   = ""
    started:   float = 0.0
    finished:  float = 0.0
    retries:   int   = 0

    @property
    def duration_s(self) -> float:
        if self.started and self.finished:
            return round(self.finished - self.started, 2)
        return 0.0


@dataclass
class OrchestrationTask:
    id:          str
    title:       str
    steps:       list[dict]          # raw skill-chain spec
    results:     list[StepResult] = field(default_factory=list)
    status:      str = "pending"     # pending | running | done | error | cancelled
    created:     float = field(default_factory=time.time)
    deadline:    Optional[float] = None   # epoch
    notify:      bool = True
    context:     dict = field(default_factory=dict)   # shared context passed between steps
    error:       str  = ""

    def to_dict(self) -> dict:
        return {
            "id":       self.id,
            "title":    self.title,
            "status":   self.status,
            "steps":    len(self.steps),
            "done":     sum(1 for r in self.results if r.status == "done"),
            "created":  self.created,
            "deadline": self.deadline,
            "notify":   self.notify,
            "error":    self.error,
            "results":  [
                {
                    "skill":    r.skill_id,
                    "status":   r.status,
                    "duration": r.duration_s,
                    "error":    r.error,
                    "output_preview": str(r.output)[:200] if r.output else None,
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# Built-in skill registry
# ---------------------------------------------------------------------------
BUILTIN_SKILLS: list[SkillDef] = [
    SkillDef(
        id="research", name="Web Research", category="research",
        description="Search the web for a topic, fetch multiple sources, and produce a structured summary.",
        input_schema={"query": "string", "depth": "integer (1-5, default 3)"},
        output_schema={"summary": "string", "sources": "list[string]"},
        system_prompt=(
            "You are a research assistant. Given a query, you will:\n"
            "1. Use http_get to fetch relevant pages or search results\n"
            "2. Extract key facts and findings\n"
            "3. Produce a structured summary with source citations\n"
            "Be thorough but concise. Focus on verifiable facts."
        ),
        timeout_s=180, tags=["web", "llm"],
    ),
    SkillDef(
        id="write_doc", name="Document Writer", category="write",
        description="Produce a structured document (Markdown, PDF outline, or plain text) from given content or research output.",
        input_schema={"content": "string", "format": "md|txt|outline", "title": "string"},
        output_schema={"document": "string", "file_path": "string"},
        system_prompt=(
            "You are a professional technical writer. Given content or research output, "
            "produce a well-structured document with clear sections, headings, and conclusions. "
            "Save the document to the workspace using write_file."
        ),
        timeout_s=120, tags=["write", "llm"],
    ),
    SkillDef(
        id="code", name="Code Agent", category="code",
        description="Generate, test, and save code to the workspace. Can execute code to verify correctness.",
        input_schema={"task": "string", "language": "string", "output_path": "string"},
        output_schema={"code": "string", "test_output": "string", "file_path": "string"},
        system_prompt=(
            "You are an expert software engineer. Write clean, tested, production-ready code. "
            "Use write_file to save code. Use shell to run tests. "
            "Report test results and fix issues before finishing."
        ),
        timeout_s=300, tags=["code", "llm", "shell"],
    ),
    SkillDef(
        id="summarise", name="Summariser", category="write",
        description="Compress long text into a structured, concise summary.",
        input_schema={"text": "string", "style": "bullet|paragraph|executive"},
        output_schema={"summary": "string"},
        system_prompt="Produce a clear, accurate summary. Do not add information not in the source.",
        timeout_s=60, tags=["write", "llm"],
    ),
    SkillDef(
        id="calendar", name="Calendar Manager", category="calendar",
        description="Read, create, and manage calendar events across providers (Google, Outlook, iCal).",
        input_schema={"action": "list|create|check_conflicts", "event": "dict", "provider": "string"},
        output_schema={"events": "list", "conflicts": "list"},
        system_prompt="",  # tool-powered, no LLM needed for basic ops
        timeout_s=30, tags=["calendar", "tool"],
    ),
    SkillDef(
        id="remind", name="Reminder", category="notify",
        description="Schedule a future notification to be delivered via SSE (and optionally webhook/email).",
        input_schema={"message": "string", "when": "ISO-8601 datetime or relative (in 2h)"},
        output_schema={"reminder_id": "string", "scheduled_for": "string"},
        system_prompt="",
        timeout_s=5, tags=["notify", "tool"],
    ),
    SkillDef(
        id="extract_pdf", name="PDF Extractor", category="tool",
        description="Extract text, tables, and metadata from a PDF file in the workspace.",
        input_schema={"file_path": "string", "pages": "string (e.g. '1-5', default all)"},
        output_schema={"text": "string", "tables": "list", "metadata": "dict"},
        system_prompt="",
        timeout_s=60, tags=["pdf", "tool"],
    ),
    SkillDef(
        id="search_web", name="Web Search", category="research",
        description="Fetch and strip multiple URLs, returning plain-text content for further processing.",
        input_schema={"urls": "list[string]", "query": "string"},
        output_schema={"results": "list[{url, text}]"},
        system_prompt="",
        timeout_s=60, tags=["web", "tool"],
    ),
    SkillDef(
        id="remember", name="Memory Writer", category="memory",
        description="Persist a fact or insight to MEMORY.md for recall in future sessions.",
        input_schema={"fact": "string", "section": "string"},
        output_schema={"status": "string"},
        system_prompt="",
        timeout_s=5, tags=["memory", "tool"],
    ),
    SkillDef(
        id="news", name="News Fetcher", category="research",
        description="Fetch and summarise recent news from RSS feeds or news APIs on a given topic.",
        input_schema={"topic": "string", "sources": "list[string]", "max_items": "integer"},
        output_schema={"articles": "list", "summary": "string"},
        system_prompt=(
            "You are a news analyst. Fetch relevant news, extract key stories, "
            "and produce a concise briefing with the most important developments."
        ),
        timeout_s=90, tags=["web", "llm"],
    ),
]

SKILLS_BY_ID: dict[str, SkillDef] = {s.id: s for s in BUILTIN_SKILLS}


# ---------------------------------------------------------------------------
# Skill Runner — executes a single skill step
# ---------------------------------------------------------------------------
class SkillRunner:
    """
    Runs a single skill with the given input dict.
    For LLM-powered skills: builds a system prompt + runs the agentic tool loop.
    For tool-powered skills: calls tools_engine directly.
    """

    def __init__(
        self,
        ollama_url:   str,
        model:        str,
        workspace:    Path,
        sse_broadcast: Callable | None = None,
    ) -> None:
        self._ollama_url   = ollama_url
        self._model        = model
        self._workspace    = workspace
        self._sse          = sse_broadcast

    async def run(
        self,
        skill: SkillDef,
        input_data: dict,
        context: dict,
        step_result: StepResult,
    ) -> Any:
        """
        Execute a skill. Returns output value (any JSON-serialisable type).
        Updates step_result in place.
        """
        step_result.started = time.time()
        step_result.status  = "running"
        self._emit("skill_start", {"skill": skill.id, "name": skill.name})

        try:
            if "tool" in skill.tags and "llm" not in skill.tags:
                output = await self._run_tool_skill(skill, input_data, context)
            else:
                output = await self._run_llm_skill(skill, input_data, context)

            step_result.status   = "done"
            step_result.output   = output
            step_result.finished = time.time()
            self._emit("skill_done", {"skill": skill.id, "duration": step_result.duration_s})
            return output

        except asyncio.TimeoutError:
            msg = f"Skill '{skill.id}' timed out after {skill.timeout_s}s"
            log.warning(msg)
            step_result.status = "error"
            step_result.error  = msg
            step_result.finished = time.time()
            self._emit("skill_error", {"skill": skill.id, "error": msg})
            return None

        except Exception as e:
            msg = f"Skill '{skill.id}' error: {e}"
            log.error(msg)
            step_result.status = "error"
            step_result.error  = msg
            step_result.finished = time.time()
            self._emit("skill_error", {"skill": skill.id, "error": msg})
            return None

    # ── Tool-powered skill (no LLM, direct tool execution) ───────────

    async def _run_tool_skill(self, skill: SkillDef, inp: dict, ctx: dict) -> Any:
        from tools_engine import execute_tool

        if skill.id == "remember":
            return await execute_tool("remember", {
                "fact":    inp.get("fact", ""),
                "section": inp.get("section", "Key facts"),
            })

        if skill.id == "remind":
            return await self._schedule_reminder(inp)

        if skill.id == "extract_pdf":
            return await self._extract_pdf(inp)

        if skill.id == "search_web":
            results = []
            for url in (inp.get("urls") or []):
                text = await execute_tool("http_get", {"url": url})
                results.append({"url": url, "text": text[:3000]})
            return {"results": results}

        if skill.id == "calendar":
            return await self._calendar_op(inp)

        return {"status": "ok", "skill": skill.id}

    # ── LLM-powered skill (runs agentic loop via Ollama) ─────────────

    async def _run_llm_skill(self, skill: SkillDef, inp: dict, ctx: dict) -> Any:
        import httpx
        from tools_engine import TOOL_DEFINITIONS, execute_tool

        # Build context string from previous steps
        ctx_str = ""
        if ctx:
            ctx_str = "\n\n## Context from previous steps:\n" + json.dumps(ctx, indent=2)[:2000]

        messages = [
            {"role": "system", "content": skill.system_prompt + ctx_str},
            {"role": "user",   "content": json.dumps(inp, ensure_ascii=False)},
        ]

        max_turns = 5
        for turn in range(max_turns):
            async with httpx.AsyncClient(timeout=skill.timeout_s) as c:
                r = await c.post(
                    f"{self._ollama_url}/api/chat",
                    json={
                        "model":    self._model,
                        "messages": messages,
                        "stream":   False,
                        "tools":    TOOL_DEFINITIONS,
                        "options":  {"temperature": 0.3, "num_predict": 2048},
                    },
                )
                if r.status_code >= 400:
                    raise RuntimeError(f"Ollama {r.status_code}: {r.text[:200]}")
                d = r.json()

            msg = d.get("message", {})
            tool_calls = msg.get("tool_calls", [])

            if not tool_calls:
                return {"text": msg.get("content", ""), "skill": skill.id}

            messages.append({
                "role": "assistant",
                "content": msg.get("content", ""),
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                raw_args  = fn.get("arguments", {})
                if isinstance(raw_args, str):
                    try: raw_args = json.loads(raw_args)
                    except Exception: raw_args = {}

                self._emit("tool_call", {"skill": skill.id, "tool": tool_name})
                result = await execute_tool(tool_name, raw_args)
                messages.append({"role": "tool", "content": result, "name": tool_name})

        return {"text": "Max turns reached", "skill": skill.id}

    # ── Helpers ───────────────────────────────────────────────────────

    async def _schedule_reminder(self, inp: dict) -> dict:
        """Parse 'when' and store a reminder that the background agent will fire."""
        from tools_engine import execute_tool
        rid = uuid.uuid4().hex[:8]
        msg = inp.get("message", "")
        when_str = inp.get("when", "")
        # Persist to reminders file
        reminders_path = self._workspace / "memory" / "reminders.json"
        reminders_path.parent.mkdir(parents=True, exist_ok=True)
        reminders = json.loads(reminders_path.read_text()) if reminders_path.exists() else []
        reminders.append({"id": rid, "message": msg, "when": when_str,
                          "created": time.time(), "fired": False})
        reminders_path.write_text(json.dumps(reminders, indent=2))
        return {"reminder_id": rid, "scheduled_for": when_str, "message": msg}

    async def _extract_pdf(self, inp: dict) -> dict:
        from tools_engine import _safe_path
        path_str = inp.get("file_path", "")
        try:
            p = _safe_path(path_str)
            import io
            try:
                import pypdf  # type: ignore
                reader = pypdf.PdfReader(str(p))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                return {"text": text, "tables": [], "metadata": {"pages": len(reader.pages)}}
            except ImportError:
                return {"text": f"[PDF extraction requires: pip install pypdf]", "tables": [], "metadata": {}}
        except Exception as e:
            return {"text": f"Error: {e}", "tables": [], "metadata": {}}

    async def _calendar_op(self, inp: dict) -> dict:
        """Calendar integration — reads local ICS files + writes events."""
        action = inp.get("action", "list")
        cal_dir = self._workspace / "memory" / "calendar"
        cal_dir.mkdir(parents=True, exist_ok=True)

        if action == "list":
            events = self._read_local_events(cal_dir)
            return {"events": events, "conflicts": self._find_conflicts(events)}

        if action == "create":
            event = inp.get("event", {})
            if not event:
                return {"error": "no event data"}
            events_file = cal_dir / "local.json"
            existing = json.loads(events_file.read_text()) if events_file.exists() else []
            event["id"] = uuid.uuid4().hex[:8]
            event["created"] = time.time()
            existing.append(event)
            events_file.write_text(json.dumps(existing, indent=2))
            return {"created": event, "conflicts": []}

        if action == "check_conflicts":
            events = self._read_local_events(cal_dir)
            return {"events": events, "conflicts": self._find_conflicts(events)}

        return {"error": f"unknown action: {action}"}

    def _read_local_events(self, cal_dir: Path) -> list[dict]:
        events = []
        for f in cal_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                if isinstance(data, list):
                    events.extend(data)
            except Exception:
                pass
        return events

    def _find_conflicts(self, events: list[dict]) -> list[dict]:
        """Simple overlap detection on start/end times."""
        conflicts = []
        timed = [e for e in events if e.get("start") and e.get("end")]
        for i, a in enumerate(timed):
            for b in timed[i+1:]:
                if a["end"] > b["start"] and a["start"] < b["end"]:
                    conflicts.append({"a": a.get("id"), "b": b.get("id"),
                                      "overlap_start": max(a["start"], b["start"])})
        return conflicts

    def _emit(self, event_type: str, data: dict) -> None:
        if self._sse:
            try:
                self._sse({"type": event_type, "ts": int(time.time() * 1000), **data})
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Orchestrator — manages skill chains + task queue
# ---------------------------------------------------------------------------
class Orchestrator:
    """
    Manages the full lifecycle of multi-skill tasks.

    Usage
    -----
        orch = Orchestrator(workspace, ollama_url, model, sse_broadcast)
        task = await orch.submit([
            {"skill": "research",  "input": {"query": "LLM benchmarks 2025"}},
            {"skill": "write_doc", "input": {"format": "md", "title": "LLM Report"}},
            {"skill": "remind",    "input": {"message": "Report ready", "when": "in 1h"}},
        ], title="LLM Research Report", deadline="2026-05-01T17:00")
    """

    def __init__(
        self,
        workspace:     Path,
        ollama_url:    str,
        model:         str,
        sse_broadcast: Callable | None = None,
    ) -> None:
        self._workspace    = workspace
        self._ollama_url   = ollama_url
        self._model        = model
        self._sse          = sse_broadcast
        self._tasks: dict[str, OrchestrationTask] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._augment_skills_from_store()

    def start(self) -> None:
        if not self._worker_task or self._worker_task.done():
            self._augment_skills_from_store()
            self._worker_task = asyncio.create_task(self._worker(), name="orchestrator-worker")
            log.info("Orchestrator worker started")

    def stop(self) -> None:
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()

    async def submit(
        self,
        steps: list[dict],
        title: str = "Untitled task",
        deadline: str | None = None,
        notify: bool = True,
        context: dict | None = None,
    ) -> OrchestrationTask:
        """Submit a skill chain for execution. Returns task immediately."""
        task = OrchestrationTask(
            id       = uuid.uuid4().hex[:10],
            title    = title,
            steps    = steps,
            deadline = self._parse_deadline(deadline),
            notify   = notify,
            context  = context or {},
        )
        self._tasks[task.id] = task
        await self._queue.put(task.id)
        self._persist_tasks()
        log.info("Orchestrator: submitted task '%s' (%d steps)", title, len(steps))
        self._emit("task_queued", {"task_id": task.id, "title": title, "steps": len(steps)})
        return task

    def get_task(self, task_id: str) -> OrchestrationTask | None:
        return self._tasks.get(task_id)

    def list_tasks(self, status: str | None = None) -> list[dict]:
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return [t.to_dict() for t in sorted(tasks, key=lambda x: x.created, reverse=True)]

    async def cancel_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task or task.status in ("done", "error"):
            return False
        task.status = "cancelled"
        self._persist_tasks()
        return True

    # ── Worker loop ───────────────────────────────────────────────────

    async def _worker(self) -> None:
        log.info("Orchestrator worker running")
        while True:
            try:
                task_id = await asyncio.wait_for(self._queue.get(), timeout=60)
            except asyncio.TimeoutError:
                await self._check_reminders()
                continue
            except asyncio.CancelledError:
                break

            task = self._tasks.get(task_id)
            if not task or task.status == "cancelled":
                continue

            await self._execute_task(task)

    async def _execute_task(self, task: OrchestrationTask) -> None:
        task.status = "running"
        self._emit("task_start", {"task_id": task.id, "title": task.title})
        log.info("Orchestrator: executing task '%s'", task.title)

        runner = SkillRunner(
            ollama_url    = self._ollama_url,
            model         = self._model,
            workspace     = self._workspace,
            sse_broadcast = self._sse,
        )
        shared_ctx = dict(task.context)

        for step_spec in task.steps:
            if task.status == "cancelled":
                break

            skill_id = step_spec.get("skill", "")
            skill    = SKILLS_BY_ID.get(skill_id)
            if not skill:
                log.warning("Orchestrator: unknown skill '%s' — skipping", skill_id)
                continue
            if not skill.enabled:
                log.warning("Orchestrator: skill '%s' is disabled — skipping", skill_id)
                continue

            # Merge previous output into input if requested
            inp = dict(step_spec.get("input", {}))
            if step_spec.get("use_context", True) and shared_ctx:
                inp.setdefault("_context", shared_ctx)

            step_result = StepResult(skill_id=skill_id, status="pending", input=inp)
            task.results.append(step_result)

            output = await runner.run(skill, inp, shared_ctx, step_result)

            # Pass output forward as context for next step
            if output and isinstance(output, dict):
                shared_ctx[skill_id] = output
            elif output:
                shared_ctx[skill_id] = {"result": str(output)[:500]}

            # Stop chain on error if step is critical
            if step_result.status == "error" and step_spec.get("required", False):
                task.status = "error"
                task.error  = f"Required step '{skill_id}' failed: {step_result.error}"
                break

        if task.status == "running":
            task.status = "done"

        self._persist_tasks()
        self._emit("task_done", {"task_id": task.id, "title": task.title, "status": task.status})

        if task.notify and task.status == "done":
            self._emit("notification", {
                "type":    "task_complete",
                "message": f"✅ Task complete: {task.title}",
                "task_id": task.id,
            })

        log.info("Orchestrator: task '%s' → %s", task.title, task.status)

    # ── Reminder checker ─────────────────────────────────────────────

    async def _check_reminders(self) -> None:
        """Fire any due reminders."""
        reminders_path = self._workspace / "memory" / "reminders.json"
        if not reminders_path.exists():
            return
        try:
            reminders = json.loads(reminders_path.read_text())
            changed   = False
            now       = time.time()
            for r in reminders:
                if r.get("fired"):
                    continue
                when = self._parse_deadline(r.get("when", ""))
                if when and now >= when:
                    r["fired"] = True
                    changed = True
                    self._emit("reminder", {
                        "type":    "reminder",
                        "message": r.get("message", "Reminder"),
                        "id":      r.get("id"),
                    })
                    log.info("Orchestrator: fired reminder '%s'", r.get("message", ""))
            if changed:
                reminders_path.write_text(json.dumps(reminders, indent=2))
        except Exception as e:
            log.debug("Reminder check error: %s", e)

    # ── Helpers ───────────────────────────────────────────────────────

    def _parse_deadline(self, deadline: str | None) -> float | None:
        if not deadline:
            return None
        import re
        # Relative: "in 2h", "in 30m", "in 1d"
        m = re.match(r"in\s+(\d+)\s*(h|m|d|s)", (deadline or "").lower())
        if m:
            n, unit = int(m.group(1)), m.group(2)
            secs = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
            return time.time() + n * secs
        # ISO-8601 or date string
        try:
            from datetime import datetime
            for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    return datetime.strptime(deadline, fmt).timestamp()
                except ValueError:
                    continue
        except Exception:
            pass
        return None

    # ── Store augmentation ───────────────────────────────────────────

    def _augment_skills_from_store(self) -> None:
        """Load enabled JSON skills from SkillStore into SKILLS_BY_ID."""
        try:
            from server.skillstore import SkillStore
            skills_dir = self._workspace / "memory" / "skills"
            store = SkillStore(skills_dir)
            for s in store.list_all():
                sid = s.get("id", "")
                if not sid or sid in SKILLS_BY_ID:
                    continue
                SKILLS_BY_ID[sid] = SkillDef(
                    id=sid,
                    name=s.get("label", sid),
                    description=s.get("description", ""),
                    category=s.get("category", "custom"),
                    input_schema={"input": "string"},
                    output_schema={"result": "string"},
                    system_prompt=s.get("system_prompt", ""),
                    enabled=bool(s.get("enabled", True)),
                    tags=s.get("tags", []),
                    skill_type="json_store",
                )
            log.debug("Orchestrator: augmented SKILLS_BY_ID with %d store skills", len(SKILLS_BY_ID))
        except Exception as exc:
            log.warning("Orchestrator: store augmentation failed: %s", exc)

    def _persist_tasks(self) -> None:
        try:
            p = self._workspace / "memory" / "orchestrator_tasks.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            data = [t.to_dict() for t in self._tasks.values()]
            p.write_text(json.dumps(data, indent=2))
        except Exception as e:
            log.debug("Orchestrator persist error: %s", e)

    def _emit(self, event_type: str, data: dict) -> None:
        if self._sse:
            try:
                self._sse({"type": event_type, "ts": int(time.time() * 1000), **data})
            except Exception:
                pass

    @property
    def skill_registry(self) -> list[dict]:
        return [
            {
                "id":          s.id,
                "name":        s.name,
                "description": s.description,
                "category":    s.category,
                "tags":        s.tags,
                "enabled":     s.enabled,
                "timeout_s":   s.timeout_s,
            }
            for s in BUILTIN_SKILLS
        ]
