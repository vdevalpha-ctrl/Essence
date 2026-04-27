"""
proactive_agent.py — Essence v1 Proactive Autonomous Agent
=========================================================
A background asyncio task that periodically:

1. Reads workspace context (GOALS.md, PROJECTS.md, MEMORY.md, LEARNED.md)
2. Calls the local Ollama model with a "proactive check" prompt
3. Decides whether to take action (create tasks, update memory, send notifications)
4. Executes any approved actions via tools_engine
5. Broadcasts results via SSE

The agent runs on a configurable tick interval (default 15 minutes).
It can also be triggered on-demand via POST /api/agent/run.

Design principles
-----------------
• Never takes destructive actions without tool safety checks
• Stays focused on the user's declared goals
• Limits each tick to at most 3 tool calls to avoid runaway behaviour
• All actions are logged to LEARNED.md for transparency
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Callable, AsyncIterator

import httpx

log = logging.getLogger("essence.agent")


class ProactiveAgent:
    """
    Autonomous background agent.

    Parameters
    ----------
    workspace     : Path to Essence workspace root
    ollama_url    : Ollama base URL (e.g. http://localhost:11434)
    model         : Model name to use for proactive reasoning
    tick_interval : Seconds between autonomous checks (default 900 = 15 min)
    sse_broadcast : Callable(event_dict) to push SSE events to UI
    """

    def __init__(
        self,
        workspace: Path,
        ollama_url: str,
        model: str,
        tick_interval: float = 900,
        sse_broadcast: Callable | None = None,
    ) -> None:
        self._workspace     = workspace
        self._ollama_url    = ollama_url
        self._model         = model
        self._tick_interval = tick_interval
        self._sse_broadcast = sse_broadcast
        self._running       = False
        self._task: asyncio.Task | None = None
        self._last_tick     = 0.0
        self._tick_count    = 0

    # ── Public API ────────────────────────────────────────────────────

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._loop(), name="proactive-agent")
            log.info("ProactiveAgent started (interval=%.0fs model=%s)", self._tick_interval, self._model)

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def trigger(self) -> dict:
        """Run a single proactive tick immediately and return a summary."""
        return await self._tick()

    def set_model(self, model: str) -> None:
        self._model = model

    # ── Internal loop ─────────────────────────────────────────────────

    async def _loop(self) -> None:
        # Initial delay so server starts up before first tick
        await asyncio.sleep(30)
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("ProactiveAgent tick error: %s", e)
            self._last_tick = time.time()
            self._tick_count += 1
            await asyncio.sleep(self._tick_interval)

    async def _tick(self) -> dict:
        """
        ReAct-style tick: Reason → Act → Observe loop.
        Replaces the old 3-action-cap approach with a trajectory-aware loop
        that stops when the critic gate approves or max_steps is reached.
        """
        log.info("ProactiveAgent tick #%d (ReAct)", self._tick_count + 1)
        ctx = self._build_context()
        if not ctx.strip():
            return {"action": "skip", "reason": "no workspace context"}

        prompt   = self._build_prompt(ctx)
        response = await self._call_ollama(prompt)
        if not response:
            return {"action": "skip", "reason": "no model response"}

        actions  = self._parse_actions(response)
        executed = []
        trajectory: list[dict] = []   # (thought, action, observation) triples

        MAX_STEPS = 5
        step = 0

        for act in actions:
            if step >= MAX_STEPS:
                log.debug("ProactiveAgent: max_steps reached")
                break

            # Critic gate: should this action execute?
            approved, reason = await self._critic_gate(act, trajectory)
            if not approved:
                log.info("ProactiveAgent: action blocked by critic (%s)", reason)
                trajectory.append({
                    "thought": f"Considered: {act.get('type')} but blocked: {reason}",
                    "action":  None,
                    "obs":     "skipped",
                })
                continue

            result = await self._execute_action(act)
            executed.append({"action": act, "result": result})
            trajectory.append({
                "thought": f"Executing {act.get('type')}: {act.get('detail', '')}",
                "action":  act,
                "obs":     result[:200],
            })
            step += 1

            if self._sse_broadcast:
                self._sse_broadcast({
                    "type":   "agent_action",
                    "action": act.get("type", "unknown"),
                    "detail": act.get("detail", ""),
                    "result": result[:200],
                    "ts":     int(time.time() * 1000),
                })

        self._log_tick(response, executed)

        summary = {
            "tick":    self._tick_count,
            "model":   self._model,
            "actions": len(executed),
            "steps":   step,
            "insight": response[:300],
        }
        if self._sse_broadcast and executed:
            self._sse_broadcast({
                "type":    "agent_tick",
                "summary": summary["insight"],
                "actions": len(executed),
                "ts":      int(time.time() * 1000),
            })
        return summary

    async def _critic_gate(
        self,
        action:     dict,
        trajectory: list[dict],
    ) -> tuple[bool, str]:
        """
        Lightweight critic that decides whether an action should execute.

        Rules (no LLM call — purely heuristic to keep ticks fast):
          1. Unknown action types → block
          2. Shell commands that look destructive → block
          3. Same action repeated in trajectory → block (idempotency)
          4. notify actions → always allow
        """
        atype = action.get("type", "")

        # Rule 1: unknown type
        _known = {"notify", "remember", "create_task", "shell"}
        if atype not in _known:
            return False, f"unknown action type '{atype}'"

        # Rule 2: shell safety
        if atype == "shell":
            cmd = action.get("command", "")
            _danger = re.compile(
                r"rm\s+-rf|del\s+/[fF]|format|shutdown|reboot|kill\s+-9\s+1",
                re.IGNORECASE,
            )
            if _danger.search(cmd):
                return False, "destructive shell command blocked"

        # Rule 3: idempotency — skip if identical action already ran this tick
        for past in trajectory:
            if past.get("action") == action:
                return False, "duplicate action suppressed"

        # Rule 4: notify always passes
        if atype == "notify":
            return True, "notify action approved"

        return True, "approved"

    # ── Context builder ───────────────────────────────────────────────

    def _build_context(self) -> str:
        parts = []
        files_to_read = [
            ("GOALS.md",    "User Goals"),
            ("PROJECTS.md", "Active Projects"),
            ("MEMORY.md",   "Memory"),
            ("LEARNED.md",  "Recent Learnings"),
            ("IDENTITY.md", "User Identity"),
        ]
        for fname, label in files_to_read:
            p = self._workspace / fname
            if p.exists():
                txt = p.read_text(encoding="utf-8", errors="replace").strip()
                # Skip empty template files
                real = [l for l in txt.split("\n")
                        if l.strip() and not l.strip().startswith("<!--") and not l.strip().startswith("#")]
                if real:
                    parts.append(f"### {label}\n{txt[:2000]}")

        # Recent sessions summary
        sess_dir = self._workspace / "sessions"
        if sess_dir.exists():
            session_files = sorted(sess_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
            recent = []
            for sf in session_files[:3]:
                try:
                    data = json.loads(sf.read_text(encoding="utf-8"))
                    hist = data.get("history", [])
                    # Last user message
                    for m in reversed(hist):
                        if m.get("role") == "user":
                            recent.append(m["content"][:120])
                            break
                except Exception:
                    pass
            if recent:
                parts.append("### Recent Conversations\n" + "\n".join(f"- {r}" for r in recent))

        return "\n\n".join(parts)

    def _build_prompt(self, context: str) -> str:
        return f"""You are the proactive intelligence core of Essence — a personal AI assistant.
You have just woken up for a scheduled autonomous check-in.

Your job is to review the context below and decide:
1. Is there anything stale, overdue, or worth the user's attention?
2. Should any memory be updated based on what you see?
3. Is there a quick autonomous action you could take to help?

Be concise and practical. If there is nothing useful to do, say "no action needed".

If you want to take an action, output it as JSON at the END of your response in this format:
<actions>
[
  {{"type": "notify", "detail": "brief message to show user"}},
  {{"type": "remember", "section": "Key facts", "fact": "fact to store"}},
  {{"type": "create_task", "title": "task title", "description": "what to do"}}
]
</actions>

Keep actions minimal and targeted. Max 3 actions.

--- WORKSPACE CONTEXT ---
{context}
--- END CONTEXT ---

Your proactive assessment:"""

    # ── Ollama call ───────────────────────────────────────────────────

    async def _call_ollama(self, prompt: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post(
                    f"{self._ollama_url}/api/generate",
                    json={
                        "model":  self._model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3, "num_predict": 512},
                    }
                )
                if r.status_code == 200:
                    return r.json().get("response", "")
        except Exception as e:
            log.warning("ProactiveAgent Ollama error: %s", e)
        return ""

    # ── Action parser ─────────────────────────────────────────────────

    def _parse_actions(self, response: str) -> list[dict]:
        import re
        m = re.search(r"<actions>\s*(\[.*?\])\s*</actions>", response, re.DOTALL)
        if not m:
            return []
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return []

    # ── Action executor ───────────────────────────────────────────────

    async def _execute_action(self, act: dict) -> str:
        atype = act.get("type", "")

        if atype == "notify":
            # Just return the detail — SSE broadcast happens in caller
            return act.get("detail", "")

        if atype == "remember":
            from server.tools_engine import execute_tool
            return await execute_tool("remember", {
                "fact":    act.get("fact", ""),
                "section": act.get("section", "Key facts"),
            })

        if atype == "create_task":
            from server.tools_engine import execute_tool
            return await execute_tool("create_task", {
                "title":       act.get("title", "Untitled"),
                "description": act.get("description", ""),
            })

        if atype == "shell":
            from server.tools_engine import execute_tool
            return await execute_tool("shell", {"command": act.get("command", "")})

        return f"unknown action type: {atype}"

    # ── Logger ────────────────────────────────────────────────────────

    def _log_tick(self, insight: str, executed: list) -> None:
        try:
            p = self._workspace / "LEARNED.md"
            ts = time.strftime("%Y-%m-%d %H:%M")
            lines = [f"\n## Agent Tick — {ts}"]
            lines.append(f"> {insight[:300].strip()}")
            for item in executed:
                act = item.get("action", {})
                res = item.get("result", "")
                lines.append(f"- **{act.get('type','?')}**: {act.get('detail', act.get('fact', act.get('title','')))} → {res[:80]}")
            entry = "\n".join(lines) + "\n"
            existing = p.read_text(encoding="utf-8") if p.exists() else "# LEARNED.md\nAgent observations and actions.\n"
            # Keep last 200 lines max
            all_lines = (existing + entry).split("\n")
            if len(all_lines) > 200:
                all_lines = all_lines[-200:]
            p.write_text("\n".join(all_lines), encoding="utf-8")
        except Exception as e:
            log.debug("ProactiveAgent log error: %s", e)

    @property
    def status(self) -> dict:
        return {
            "running":       self._running,
            "tick_count":    self._tick_count,
            "last_tick_ago": int(time.time() - self._last_tick) if self._last_tick else None,
            "interval_s":    self._tick_interval,
            "model":         self._model,
        }
