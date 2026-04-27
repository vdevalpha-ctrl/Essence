"""
memory_skill.py — Memory Read/Write SkillAgent
"""
from __future__ import annotations

import logging
from server.skill_agent import SkillAgent, SkillManifest, SkillContext, SkillResult

log = logging.getLogger("essence.skill.memory")


class MemorySkill(SkillAgent):
    manifest = SkillManifest(
        id="memory",
        name="Memory Manager",
        version="1.0",
        description="Read, write, search, and signal memories in the gravity memory store.",
        category="memory",
        capabilities=["memory_read", "memory_write"],
        requires_caps=["memory"],
        input_schema={
            "action":  "string (read|write|search|signal)",
            "key":     "string",
            "value":   "string (for write)",
            "signal":  "string (confirmed|corrected|ignored, for signal)",
            "query":   "string (for search)",
        },
        output_schema={"result": "any", "memories": "list"},
        max_latency_ms=500,
        autonomy_threshold=0.60,
        timeout_s=10,
        tags=["memory", "remember", "recall", "store", "forget"],
        system_prompt="",
    )

    async def execute(self, ctx: SkillContext) -> SkillResult:
        try:
            from server.gravity_memory import get_gravity_memory
            mem = get_gravity_memory()
        except Exception as exc:
            return SkillResult(skill_id=self.manifest.id, status="error",
                               error=f"memory unavailable: {exc}")

        action = ctx.input_data.get("action", "write").lower()
        key    = ctx.input_data.get("key", ctx.intent[:80])
        value  = ctx.input_data.get("value", "")
        query  = ctx.input_data.get("query", ctx.intent)
        signal = ctx.input_data.get("signal", "")

        try:
            if action == "read":
                result = mem.get(key)
                return SkillResult(skill_id=self.manifest.id, status="done",
                                   output={"result": result})

            elif action == "write":
                if not value:
                    value = ctx.input_data.get("text", ctx.intent)
                result = mem.write(key=key, value=value, skill_source="memory_skill")
                return SkillResult(skill_id=self.manifest.id, status="done",
                                   output={"result": result, "action": "written"})

            elif action == "search":
                results = mem.search(query=query, limit=10)
                return SkillResult(skill_id=self.manifest.id, status="done",
                                   output={"memories": results, "count": len(results)})

            elif action == "signal":
                if signal not in ("confirmed", "corrected", "ignored", "neutral"):
                    return SkillResult(skill_id=self.manifest.id, status="error",
                                       error=f"invalid signal: {signal!r}")
                result = mem.signal(key, signal)
                return SkillResult(skill_id=self.manifest.id, status="done",
                                   output={"result": result, "signal": signal})

            else:
                return SkillResult(skill_id=self.manifest.id, status="error",
                                   error=f"unknown action: {action!r}")

        except KeyError as exc:
            return SkillResult(skill_id=self.manifest.id, status="error",
                               error=str(exc))
        except Exception as exc:
            log.error("MemorySkill error: %s", exc)
            return SkillResult(skill_id=self.manifest.id, status="error",
                               error=str(exc))
