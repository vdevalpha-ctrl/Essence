"""
mcts_planner.py — Monte Carlo Tree Search Planner for Essence
==============================================================
Generates N candidate plans via the LLM, scores each with a critic
LLM judge, and selects the best via UCB1 bandit rollouts.

This is Pass 3b in the kernel planner pipeline — it activates when
the standard LLM planner succeeds but the kernel is in a high-stakes
or multi-step situation (detected by complexity score).

Architecture:
  MCTSNode    — plan candidate with visit/value stats
  MCTSPlanner — samples candidates, scores, and selects winner
  llm_judge() — lightweight LLM call that returns a 0-1 quality score

Configuration (all optional):
  ESSENCE_MCTS_CANDIDATES   — number of candidate plans (default: 3)
  ESSENCE_MCTS_ROLLOUTS     — UCB rollout iterations (default: 5)
  ESSENCE_MCTS_THRESHOLD    — min complexity score to activate (default: 0.6)
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger("essence.mcts")

_N_CANDIDATES = int(os.environ.get("ESSENCE_MCTS_CANDIDATES", "3"))
_N_ROLLOUTS   = int(os.environ.get("ESSENCE_MCTS_ROLLOUTS", "5"))
_THRESHOLD    = float(os.environ.get("ESSENCE_MCTS_THRESHOLD", "0.6"))


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class MCTSNode:
    plan:     dict                         # raw plan dict from LLM
    visits:   int   = 0
    value:    float = 0.0
    id:       str   = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def ucb1(self) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = max(self.visits, 1)
        return self.value / self.visits + math.sqrt(2 * math.log(parent_visits) / self.visits)


# ── LLM judge ─────────────────────────────────────────────────────────────

async def llm_judge(plan: dict, intent: str, ollama_url: str, model: str) -> float:
    """
    Ask the LLM to score a plan on a 0-1 scale.
    Returns 0.5 on failure (neutral score).
    """
    try:
        import httpx
        prompt = (
            f"Rate this plan for the intent '{intent[:200]}' on a scale 0.0-1.0.\n"
            f"Plan steps: {json.dumps([s.get('skill_id','?') for s in plan.get('steps', [])], indent=None)}\n"
            f"Goal: {plan.get('goal', '')}\n\n"
            "Respond with ONLY a single number between 0.0 and 1.0 (no explanation)."
        )
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.post(
                f"{ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.1, "num_predict": 8}},
            )
            text = r.json().get("response", "0.5").strip()
            # Extract first number
            import re
            m = re.search(r"[01]\.\d+|\d+\.\d+|[01]", text)
            if m:
                return min(1.0, max(0.0, float(m.group())))
    except Exception as exc:
        log.debug("mcts: llm_judge failed: %s", exc)
    return 0.5


# ── Complexity estimator ───────────────────────────────────────────────────

def intent_complexity(intent: str) -> float:
    """
    Heuristic complexity score 0-1 based on intent characteristics.
    Higher = more beneficial to run MCTS.
    """
    score = 0.0
    words = intent.lower().split()
    n_words = len(words)

    # Length signal
    if n_words > 20:
        score += 0.3
    elif n_words > 10:
        score += 0.15

    # Conjunction signal — AND / then / also suggests multi-step
    conj = sum(1 for w in words if w in ("and", "then", "also", "after", "before", "while"))
    score += min(conj * 0.15, 0.4)

    # High-stakes verbs
    high_stakes = ("deploy", "delete", "migrate", "refactor", "rebuild", "publish", "remove")
    if any(w in words for w in high_stakes):
        score += 0.3

    return min(score, 1.0)


# ── MCTS Planner ───────────────────────────────────────────────────────────

class MCTSPlanner:
    """
    Generate N candidate plans, score them with the LLM judge,
    and select the winner via UCB1 rollouts.
    """

    def __init__(
        self,
        ollama_url:    str,
        model:         str,
        n_candidates:  int   = _N_CANDIDATES,
        n_rollouts:    int   = _N_ROLLOUTS,
        threshold:     float = _THRESHOLD,
    ) -> None:
        self._ollama       = ollama_url
        self._model        = model
        self._n_candidates = n_candidates
        self._n_rollouts   = n_rollouts
        self._threshold    = threshold

    async def plan(
        self,
        intent: str,
        catalog: list[dict],
        memory_ctx: str = "",
    ) -> Optional[dict]:
        """
        Run MCTS and return the best plan dict, or None if complexity
        is below threshold or no valid plans were found.
        """
        complexity = intent_complexity(intent)
        if complexity < self._threshold:
            log.debug("mcts: complexity %.2f < %.2f — skipping", complexity, self._threshold)
            return None

        log.info("mcts: running (complexity=%.2f, N=%d)", complexity, self._n_candidates)

        # Sample N candidates
        candidates = await asyncio.gather(*[
            self._sample_plan(intent, catalog, memory_ctx)
            for _ in range(self._n_candidates)
        ])
        candidates = [c for c in candidates if c and c.get("steps")]

        if not candidates:
            return None

        nodes = [MCTSNode(plan=c) for c in candidates]

        # UCB1 rollouts — score each node with the LLM judge
        for _ in range(self._n_rollouts):
            # Select node with highest UCB1
            node = max(nodes, key=lambda n: n.ucb1)
            score = await llm_judge(node.plan, intent, self._ollama, self._model)
            node.visits += 1
            node.value  += score

        best = max(nodes, key=lambda n: n.value / max(n.visits, 1))
        log.info("mcts: selected plan with score %.2f (%d steps)",
                 best.value / max(best.visits, 1), len(best.plan.get("steps", [])))
        return best.plan

    async def _sample_plan(
        self, intent: str, catalog: list[dict], memory_ctx: str
    ) -> Optional[dict]:
        """Ask the LLM to produce one candidate plan (with slight temperature variation)."""
        try:
            import httpx
            system = (
                "You are a planning assistant. Given a user intent and skill catalog, "
                "produce a concise JSON execution plan.\n"
                "If the intent is conversational, return {\"steps\":[]}.\n\n"
                f"Available skills:\n{json.dumps(catalog, indent=2)}\n\n"
                "Return ONLY valid JSON:\n"
                '{"goal": "...", "steps": ['
                '{"skill_id": "...", "input_map": {}, "depends_on": [], "required": true}'
                "]}"
            )
            temperature = 0.3 + random.uniform(0, 0.4)   # vary per sample
            async with httpx.AsyncClient(timeout=15.0) as c:
                r = await c.post(
                    f"{self._ollama}/api/chat",
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user",   "content": f"Intent: {intent}\n{memory_ctx}"},
                        ],
                        "stream": False,
                        "options": {"temperature": temperature, "num_predict": 512},
                    },
                )
            content = r.json().get("message", {}).get("content", "{}")
            import re
            m = re.search(r"\{[\s\S]*\}", content)
            if not m:
                return None
            return json.loads(m.group())
        except Exception as exc:
            log.debug("mcts: _sample_plan failed: %s", exc)
            return None


# ── Module-level singleton ─────────────────────────────────────────────────

_mcts: Optional[MCTSPlanner] = None


def get_mcts_planner(ollama_url: str = "", model: str = "") -> MCTSPlanner:
    global _mcts
    if _mcts is None:
        _mcts = MCTSPlanner(ollama_url=ollama_url, model=model)
    return _mcts
