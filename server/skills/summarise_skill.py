"""
summarise_skill.py — Summarisation SkillAgent
"""
from __future__ import annotations

import logging
from server.skill_agent import SkillAgent, SkillManifest, SkillContext, SkillResult

log = logging.getLogger("essence.skill.summarise")


class SummariseSkill(SkillAgent):
    manifest = SkillManifest(
        id="summarise",
        name="Summariser",
        version="1.0",
        description="Compress long text into a structured, concise summary.",
        category="write",
        capabilities=["llm_call"],
        requires_caps=["llm_call"],
        input_schema={
            "text":  "string — the content to summarise",
            "style": "string — bullet|paragraph|executive (default: bullet)",
            "max_words": "int (default 150)",
        },
        output_schema={"summary": "string"},
        max_latency_ms=20_000,
        autonomy_threshold=0.70,
        timeout_s=60,
        tags=["summarise", "summarize", "summary", "tldr", "condense", "shorten"],
        system_prompt=(
            "You are a precise summarisation engine. "
            "Produce accurate, concise summaries. Do not add information not in the source."
        ),
    )

    async def execute(self, ctx: SkillContext) -> SkillResult:
        text      = ctx.input_data.get("text") or ctx.shared_context.get("research", {}).get("summary", "")
        style     = ctx.input_data.get("style", "bullet")
        max_words = int(ctx.input_data.get("max_words", 150))

        if not text:
            return SkillResult(skill_id=self.manifest.id, status="error",
                               error="no text provided to summarise")

        style_instruction = {
            "bullet":     "Use bullet points. Each point is one key fact.",
            "paragraph":  "Write 2-3 coherent paragraphs.",
            "executive":  "Executive summary: one sentence of context, 3 key findings, one recommendation.",
        }.get(style, "Use bullet points.")

        prompt = (
            f"Summarise the following text in ~{max_words} words.\n"
            f"Style: {style_instruction}\n\n"
            f"TEXT:\n{text[:6000]}"
        )

        try:
            from server.kernel import get_kernel
            k = get_kernel()
            import httpx
            async with httpx.AsyncClient(timeout=self.manifest.timeout_s) as c:
                r = await c.post(
                    f"{k._ollama}/api/chat",
                    json={
                        "model": k._model,
                        "messages": [
                            {"role": "system", "content": self.manifest.system_prompt},
                            {"role": "user",   "content": prompt},
                        ],
                        "stream": False,
                        "options": {"temperature": 0.2, "num_predict": 600},
                    },
                )
            summary = r.json().get("message", {}).get("content", "").strip()
            return SkillResult(skill_id=self.manifest.id, status="done",
                               output={"summary": summary})

        except Exception as exc:
            log.error("SummariseSkill error: %s", exc)
            return SkillResult(skill_id=self.manifest.id, status="error",
                               error=str(exc))
