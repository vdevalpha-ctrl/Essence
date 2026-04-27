"""
research_skill.py — Web Research SkillAgent
"""
from __future__ import annotations

import json
import logging
from server.skill_agent import SkillAgent, SkillManifest, SkillContext, SkillResult

log = logging.getLogger("essence.skill.research")


class ResearchSkill(SkillAgent):
    manifest = SkillManifest(
        id="research",
        name="Web Research",
        version="1.0",
        description="Search the web for a topic and produce a structured summary with sources.",
        category="research",
        capabilities=["web_search", "http_get", "llm_call"],
        requires_caps=["network"],
        input_schema={"query": "string", "depth": "int (1-5, default 2)"},
        output_schema={"summary": "string", "sources": "list[string]"},
        max_latency_ms=60_000,
        autonomy_threshold=0.75,
        timeout_s=90,
        tags=["web", "research", "search", "find", "look up", "information"],
        system_prompt=(
            "You are a research assistant. Given a query, search for information, "
            "extract key facts, and produce a concise structured summary. "
            "Always cite your sources."
        ),
    )

    async def execute(self, ctx: SkillContext) -> SkillResult:
        query = ctx.input_data.get("query") or ctx.intent
        depth = min(int(ctx.input_data.get("depth", 2)), 5)

        try:
            import httpx
        except ImportError:
            return SkillResult(skill_id=self.manifest.id, status="error",
                               error="httpx not installed")

        sources = []
        fetched_texts = []

        # Step 1: Fetch search results via DuckDuckGo HTML (no API key needed)
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as c:
                headers = {"User-Agent": "Mozilla/5.0 (compatible; EssenceBot/1.0)"}
                r = await c.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers=headers,
                )
                if r.status_code == 200:
                    # Extract visible text snippets (simple parse)
                    import re
                    snippets = re.findall(r'class="result__snippet">(.*?)</a>', r.text, re.DOTALL)
                    links    = re.findall(r'class="result__url".*?>(.*?)</a>', r.text, re.DOTALL)
                    for i, (snip, link) in enumerate(zip(snippets, links)):
                        clean = re.sub(r'<[^>]+>', '', snip).strip()
                        if clean:
                            fetched_texts.append(clean)
                            sources.append(link.strip())
                        if i >= depth - 1:
                            break
        except Exception as exc:
            log.warning("Research: fetch failed: %s", exc)

        if not fetched_texts:
            # Fallback: LLM-only answer
            summary = await self._llm_summary(query, [], ctx)
            return SkillResult(skill_id=self.manifest.id, status="done",
                               output={"summary": summary, "sources": []})

        # Step 2: Summarise with LLM
        combined = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(fetched_texts))
        summary  = await self._llm_summary(query, [combined], ctx)

        return SkillResult(
            skill_id=self.manifest.id,
            status="done",
            output={"summary": summary, "sources": sources[:depth]},
        )

    async def _llm_summary(self, query: str, texts: list[str], ctx: SkillContext) -> str:
        try:
            import httpx
            from server.kernel import get_kernel
            k = get_kernel()
            ollama, model = k._ollama, k._model
        except Exception:
            return "\n".join(texts) if texts else f"No results found for: {query}"

        content = "\n\n".join(texts) if texts else ""
        prompt  = (
            f"Query: {query}\n\n"
            f"{'Source material:\n' + content[:3000] if content else 'No web results found.'}\n\n"
            "Produce a concise, factual summary answering the query. "
            "If no sources were found, answer from your training knowledge and say so."
        )
        try:
            async with httpx.AsyncClient(timeout=30.0) as c:
                r = await c.post(
                    f"{ollama}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": self.manifest.system_prompt},
                            {"role": "user",   "content": prompt},
                        ],
                        "stream": False,
                        "options": {"temperature": 0.2, "num_predict": 512},
                    },
                )
            return r.json().get("message", {}).get("content", "").strip() or content
        except Exception as exc:
            return content or f"Summary failed: {exc}"
