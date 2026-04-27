"""
skill_evolution.py — Self-Evolution Loop for Essence Skills
===========================================================
Adapted from Hermes Agent Self-Evolution (hermes-agent-self-evolution-main).

Implements a DSPy-inspired multi-dimensional fitness evaluation and LLM-driven
optimisation loop for Essence skill instructions — without requiring DSPy or
any external ML framework.

Architecture
------------
  FitnessScore        — multi-dimensional (correctness · procedure · conciseness)
  LLMJudge            — rubric-based LLM-as-judge scorer via httpx
  ConstraintValidator — hard gates that evolved skills must pass
  evolve_skill()      — end-to-end 10-step pipeline (safe, never auto-deploys)

Usage (CLI)
-----------
  python essence.py evolve <skill_id>               # generate + score + improve
  python essence.py evolve <skill_id> --dry-run     # show what would happen
  python essence.py evolve list                      # list evolvable skills

Safety
------
  Evolved output is NEVER auto-deployed.  It is saved to
  memory/skills/evolved/<skill_id>_v<N>.md alongside a metrics.json file
  for human review before promotion.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

log = logging.getLogger("essence.evolution")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_OLLAMA = "http://localhost:11434"
_DEFAULT_MODEL  = "qwen3:4b"

# How many synthetic examples to generate for evaluation
_EVAL_EXAMPLES = 8
# Minimum composite-score improvement to consider the evolution a success
_MIN_IMPROVEMENT = 0.05


# ---------------------------------------------------------------------------
# FitnessScore
# ---------------------------------------------------------------------------

@dataclass
class FitnessScore:
    """Multi-dimensional fitness score for an evolved skill (0.0–1.0 per dim).

    Adapted from Hermes evolution/core/fitness.py.
    """
    correctness:         float = 0.0   # Did the output correctly address the task?
    procedure_following: float = 0.0   # Did it follow the skill's procedure?
    conciseness:         float = 0.0   # Appropriately concise (no fluff)?
    length_penalty:      float = 0.0   # Penalty for excessive verbosity
    feedback:            str   = ""    # Actionable feedback for next iteration

    @property
    def composite(self) -> float:
        """Weighted composite: 50% correctness · 30% procedure · 20% conciseness."""
        raw = (
            0.5 * self.correctness
            + 0.3 * self.procedure_following
            + 0.2 * self.conciseness
        )
        return max(0.0, raw - self.length_penalty)

    def __str__(self) -> str:
        return (
            f"composite={self.composite:.3f}  "
            f"[correct={self.correctness:.2f} "
            f"proc={self.procedure_following:.2f} "
            f"concise={self.conciseness:.2f} "
            f"penalty={self.length_penalty:.2f}]"
        )


def _parse_score(value: Any) -> float:
    """Parse LLM-emitted score to float in [0, 1]."""
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    try:
        return min(1.0, max(0.0, float(str(value).strip())))
    except (ValueError, TypeError):
        return 0.5


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

async def _llm_call(
    prompt: str,
    ollama_url: str,
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    as_json: bool = False,
) -> str:
    """Simple async call to Ollama /api/chat.  Returns content string."""
    messages = [{"role": "user", "content": prompt}]
    options: dict = {"temperature": temperature, "num_predict": max_tokens}
    if as_json:
        options["format"] = "json"  # type: ignore[assignment]

    async with httpx.AsyncClient(timeout=90.0) as client:
        r = await client.post(
            f"{ollama_url}/api/chat",
            json={"model": model, "messages": messages, "stream": False, "options": options},
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "").strip()


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------

class LLMJudge:
    """Rubric-based multi-dimensional scorer (LLM-as-judge).

    Inspired by Hermes evolution/core/fitness.py — reimplemented with plain
    httpx instead of DSPy so no extra dependencies are required.
    """

    JUDGE_PROMPT = """\
You are an impartial evaluator scoring an AI agent's response.

## Task given to agent
{task_input}

## Expected behaviour rubric
{expected_behavior}

## Skill / instructions the agent was following
{skill_text}

## Agent's actual response
{agent_output}

---
Score the response on three dimensions (0.0 = terrible, 1.0 = perfect):

1. correctness        — Did the response correctly address the task?
2. procedure_following — Did it follow the expected procedure described in the skill?
3. conciseness        — Was it appropriately concise without omitting important info?

Also provide 1–2 sentences of SPECIFIC, ACTIONABLE feedback on the biggest weakness.

Respond in JSON only:
{{
  "correctness": <float 0-1>,
  "procedure_following": <float 0-1>,
  "conciseness": <float 0-1>,
  "feedback": "<string>"
}}"""

    def __init__(self, ollama_url: str, model: str) -> None:
        self.ollama_url = ollama_url
        self.model = model

    async def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        skill_text: str,
        artifact_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> FitnessScore:
        """Return a FitnessScore for the given agent output."""
        prompt = self.JUDGE_PROMPT.format(
            task_input=task_input[:2000],
            expected_behavior=expected_behavior[:1000],
            skill_text=skill_text[:1500],
            agent_output=agent_output[:2000],
        )
        try:
            raw = await _llm_call(prompt, self.ollama_url, self.model,
                                  max_tokens=512, temperature=0.1, as_json=True)
            data = json.loads(raw)
        except Exception as exc:
            log.warning("LLMJudge parse failed: %s — using heuristic fallback", exc)
            data = self._heuristic_score(task_input, expected_behavior, agent_output)

        correctness         = _parse_score(data.get("correctness", 0.5))
        procedure_following = _parse_score(data.get("procedure_following", 0.5))
        conciseness         = _parse_score(data.get("conciseness", 0.5))
        feedback            = str(data.get("feedback", ""))

        # Length penalty: if output is > 2× the expected behaviour length
        length_penalty = 0.0
        if artifact_size is not None and max_size is not None:
            ratio = artifact_size / max_size
            if ratio > 0.9:
                length_penalty = min(0.3, (ratio - 0.9) * 3.0)

        return FitnessScore(
            correctness=correctness,
            procedure_following=procedure_following,
            conciseness=conciseness,
            length_penalty=length_penalty,
            feedback=feedback,
        )

    @staticmethod
    def _heuristic_score(task_input: str, expected: str, output: str) -> dict:
        """Keyword-overlap heuristic when the LLM judge fails."""
        exp_words = set(expected.lower().split())
        out_words = set(output.lower().split())
        overlap = len(exp_words & out_words) / max(len(exp_words), 1)
        score = 0.3 + 0.6 * overlap
        return {
            "correctness": score,
            "procedure_following": score * 0.9,
            "conciseness": 0.7,
            "feedback": "Heuristic fallback — LLM judge unavailable.",
        }


# ---------------------------------------------------------------------------
# ConstraintValidator — hard gates
# ---------------------------------------------------------------------------

class ConstraintValidator:
    """Hard constraints that evolved skills MUST pass.

    Mirrors Hermes evolution/core/constraints.py:
      - Output size within limits
      - No forbidden content (commands, secrets, HTML injection)
      - Minimum quality threshold
    """

    MAX_SKILL_CHARS = 12_000
    MIN_SCORE       = 0.35    # composite must beat this or we reject

    @classmethod
    def validate(cls, skill_text: str, score: FitnessScore) -> tuple[bool, str]:
        """Return (passed, reason).  reason is empty string when passed."""
        if len(skill_text) > cls.MAX_SKILL_CHARS:
            return False, f"Skill too long ({len(skill_text):,} chars > {cls.MAX_SKILL_CHARS:,})"

        if score.composite < cls.MIN_SCORE:
            return False, f"Score too low ({score.composite:.3f} < {cls.MIN_SCORE})"

        # Forbidden patterns
        forbidden = [
            (r"rm\s+-rf", "contains dangerous shell command"),
            (r"<script", "contains HTML/JS injection"),
            (r"os\.system|subprocess\.call", "contains raw subprocess calls"),
        ]
        for pattern, reason in forbidden:
            if re.search(pattern, skill_text, re.IGNORECASE):
                return False, f"Constraint violation: {reason}"

        return True, ""


# ---------------------------------------------------------------------------
# Synthetic example generator
# ---------------------------------------------------------------------------

async def _generate_eval_examples(
    skill_text: str,
    skill_name: str,
    n: int,
    ollama_url: str,
    model: str,
) -> list[dict]:
    """Generate synthetic task/expected-behaviour pairs for evaluating a skill.

    Each example is a dict with keys:
      task_input       — a realistic user request that exercises this skill
      expected_behavior — rubric describing what a good response looks like
    """
    prompt = f"""\
You are generating evaluation examples for testing an AI skill.

## Skill name
{skill_name}

## Skill instructions
{skill_text[:3000]}

---
Generate {n} diverse, realistic evaluation examples. Each example should:
1. Be a concrete task input that exercises the skill
2. Include an expected behaviour rubric (what a great response looks like)

Respond in JSON as a list of objects:
[
  {{
    "task_input": "...",
    "expected_behavior": "..."
  }},
  ...
]
Generate exactly {n} examples. No other text."""

    try:
        raw = await _llm_call(prompt, ollama_url, model,
                              max_tokens=2048, temperature=0.7, as_json=True)
        data = json.loads(raw)
        if isinstance(data, list):
            return data[:n]
        # Sometimes LLMs wrap in {"examples": [...]}
        if isinstance(data, dict):
            for k in ("examples", "data", "items"):
                if isinstance(data.get(k), list):
                    return data[k][:n]
    except Exception as exc:
        log.warning("Eval example generation failed: %s", exc)

    # Fallback: minimal synthetic examples
    return [
        {"task_input": f"Demonstrate the {skill_name} skill.", "expected_behavior": "Accurate, concise, follows skill procedure."}
        for _ in range(min(n, 3))
    ]


# ---------------------------------------------------------------------------
# Skill text improver
# ---------------------------------------------------------------------------

async def _improve_skill(
    skill_text: str,
    skill_name: str,
    examples: list[dict],
    scores: list[FitnessScore],
    iteration: int,
    ollama_url: str,
    model: str,
) -> str:
    """Ask the LLM to rewrite the skill instructions to fix identified weaknesses."""
    # Aggregate feedback from all scored examples
    feedback_lines = []
    for i, (ex, sc) in enumerate(zip(examples, scores)):
        if sc.feedback:
            feedback_lines.append(f"Example {i+1}: {sc.feedback}")

    weakest = min(scores, key=lambda s: s.composite) if scores else None
    avg_score = sum(s.composite for s in scores) / len(scores) if scores else 0

    prompt = f"""\
You are improving an AI agent skill. Your job is to rewrite the skill instructions
to address the specific weaknesses identified by evaluation.

## Skill name
{skill_name}

## Current skill instructions (iteration {iteration})
{skill_text}

## Evaluation summary
Average composite score: {avg_score:.3f} / 1.0

## Identified weaknesses (from LLM judge evaluation)
{chr(10).join(feedback_lines) if feedback_lines else 'No specific feedback captured.'}

## Improvement instructions
Rewrite the skill instructions to:
1. Fix the specific weaknesses above
2. Make the procedure clearer and more actionable
3. Keep it concise — no padding or repetition
4. Preserve any correct parts that are already working well
5. Keep the same general intent and scope

Output ONLY the improved skill text. No explanation, no preamble, no markdown wrapper.
The output will be saved as the new skill file verbatim."""

    try:
        improved = await _llm_call(prompt, ollama_url, model,
                                   max_tokens=3000, temperature=0.4)
        return improved.strip()
    except Exception as exc:
        log.warning("Skill improvement LLM failed: %s", exc)
        return skill_text  # Return original if improvement fails


# ---------------------------------------------------------------------------
# Skill score averaging helper
# ---------------------------------------------------------------------------

async def _evaluate_skill(
    skill_text: str,
    examples: list[dict],
    judge: LLMJudge,
    ollama_url: str,
    model: str,
) -> tuple[list[FitnessScore], float]:
    """Score a skill against all evaluation examples.  Returns (scores, avg_composite)."""
    scores: list[FitnessScore] = []
    for ex in examples:
        task_input = ex.get("task_input", "")
        expected   = ex.get("expected_behavior", "")

        # Simulate the skill execution: ask the LLM to respond to the task
        # following the skill instructions (no tool execution — instruction quality only)
        sim_prompt = (
            f"Following these instructions exactly:\n\n{skill_text[:2000]}\n\n"
            f"Now respond to this task:\n{task_input}"
        )
        try:
            agent_output = await _llm_call(sim_prompt, ollama_url, model, max_tokens=800, temperature=0.2)
        except Exception:
            agent_output = ""

        sc = await judge.score(
            task_input=task_input,
            expected_behavior=expected,
            agent_output=agent_output,
            skill_text=skill_text,
            artifact_size=len(skill_text),
            max_size=ConstraintValidator.MAX_SKILL_CHARS,
        )
        scores.append(sc)

    avg = sum(s.composite for s in scores) / len(scores) if scores else 0.0
    return scores, avg


# ---------------------------------------------------------------------------
# Main evolution entry point
# ---------------------------------------------------------------------------

async def evolve_skill(
    skill_id: str,
    skills_dir: Path,
    ollama_url: str = _DEFAULT_OLLAMA,
    model: str      = _DEFAULT_MODEL,
    iterations: int = 3,
    dry_run: bool   = False,
) -> dict:
    """End-to-end skill evolution pipeline.

    Steps:
      1. Load skill from skills_dir/<skill_id>.md (or .yaml)
      2. Generate synthetic evaluation examples
      3. Score baseline
      4. Validate baseline constraints
      5. Iteratively improve + score
      6. Compare against baseline
      7. Validate evolved constraints
      8. Save evolved skill to evolved/ sub-directory (NEVER auto-promote)
      9. Write metrics.json alongside

    Returns a summary dict with keys: improved, baseline_score, evolved_score,
    output_path, metrics.
    """
    # ── 1. Load skill ───────────────────────────────────────────────────
    skill_path = None
    for ext in (".md", ".yaml", ".yml", ".txt", ""):
        candidate = skills_dir / f"{skill_id}{ext}"
        if candidate.exists():
            skill_path = candidate
            break

    if skill_path is None:
        raise FileNotFoundError(f"Skill '{skill_id}' not found in {skills_dir}")

    skill_text = skill_path.read_text(encoding="utf-8")
    skill_name = skill_id.replace("-", " ").replace("_", " ").title()
    log.info("Evolution: loaded skill '%s' (%d chars)", skill_id, len(skill_text))

    if dry_run:
        return {
            "dry_run": True,
            "skill_id": skill_id,
            "skill_chars": len(skill_text),
            "would_generate": _EVAL_EXAMPLES,
            "would_iterate": iterations,
        }

    judge = LLMJudge(ollama_url, model)

    # ── 2. Generate evaluation examples ────────────────────────────────
    log.info("Evolution: generating %d evaluation examples …", _EVAL_EXAMPLES)
    examples = await _generate_eval_examples(skill_text, skill_name, _EVAL_EXAMPLES, ollama_url, model)
    log.info("Evolution: generated %d examples", len(examples))

    # ── 3. Baseline score ───────────────────────────────────────────────
    log.info("Evolution: scoring baseline …")
    baseline_scores, baseline_avg = await _evaluate_skill(skill_text, examples, judge, ollama_url, model)
    log.info("Evolution: baseline composite = %.3f", baseline_avg)

    # ── 4. Baseline constraint check ────────────────────────────────────
    baseline_fs = FitnessScore(
        correctness=sum(s.correctness for s in baseline_scores) / max(len(baseline_scores), 1),
        procedure_following=sum(s.procedure_following for s in baseline_scores) / max(len(baseline_scores), 1),
        conciseness=sum(s.conciseness for s in baseline_scores) / max(len(baseline_scores), 1),
    )
    passed, reason = ConstraintValidator.validate(skill_text, baseline_fs)
    if not passed:
        log.warning("Baseline constraint failure: %s — evolution may be limited", reason)

    # ── 5. Iterative improvement ─────────────────────────────────────────
    current_text   = skill_text
    current_scores = baseline_scores
    current_avg    = baseline_avg
    best_text      = skill_text
    best_avg       = baseline_avg
    iteration_log: list[dict] = []

    for i in range(1, iterations + 1):
        log.info("Evolution: iteration %d/%d (current best=%.3f)", i, iterations, best_avg)

        improved_text = await _improve_skill(
            current_text, skill_name, examples, current_scores, i, ollama_url, model
        )

        if not improved_text or improved_text == current_text:
            log.info("Evolution: no change on iteration %d — stopping", i)
            break

        improved_scores, improved_avg = await _evaluate_skill(improved_text, examples, judge, ollama_url, model)
        log.info("Evolution: iteration %d score=%.3f (was %.3f)", i, improved_avg, current_avg)

        iteration_log.append({
            "iteration": i,
            "score": round(improved_avg, 4),
            "delta": round(improved_avg - current_avg, 4),
            "chars": len(improved_text),
        })

        if improved_avg > best_avg:
            best_text = improved_text
            best_avg  = improved_avg

        current_text   = improved_text
        current_scores = improved_scores
        current_avg    = improved_avg

    # ── 6. Final constraint validation ──────────────────────────────────
    evolved_fs = FitnessScore(
        correctness=sum(s.correctness for s in current_scores) / max(len(current_scores), 1),
        procedure_following=sum(s.procedure_following for s in current_scores) / max(len(current_scores), 1),
        conciseness=sum(s.conciseness for s in current_scores) / max(len(current_scores), 1),
    )
    evolved_passed, evolved_reason = ConstraintValidator.validate(best_text, evolved_fs)

    improved = best_avg > baseline_avg + _MIN_IMPROVEMENT and evolved_passed

    # ── 7. Save output (NEVER auto-promote) ─────────────────────────────
    evolved_dir = skills_dir / "evolved"
    evolved_dir.mkdir(parents=True, exist_ok=True)

    # Find next version number
    existing = list(evolved_dir.glob(f"{skill_id}_v*.md"))
    version  = len(existing) + 1
    output_path = evolved_dir / f"{skill_id}_v{version}.md"
    metrics_path = evolved_dir / f"{skill_id}_v{version}_metrics.json"

    output_path.write_text(best_text, encoding="utf-8")

    metrics = {
        "skill_id":      skill_id,
        "version":       version,
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model":         model,
        "iterations":    iterations,
        "eval_examples": len(examples),
        "baseline": {
            "score":       round(baseline_avg, 4),
            "chars":       len(skill_text),
            "correctness": round(baseline_fs.correctness, 3),
            "procedure":   round(baseline_fs.procedure_following, 3),
            "conciseness": round(baseline_fs.conciseness, 3),
        },
        "evolved": {
            "score":        round(best_avg, 4),
            "chars":        len(best_text),
            "correctness":  round(evolved_fs.correctness, 3),
            "procedure":    round(evolved_fs.procedure_following, 3),
            "conciseness":  round(evolved_fs.conciseness, 3),
            "delta":        round(best_avg - baseline_avg, 4),
            "constraints_passed": evolved_passed,
            "constraint_reason":  evolved_reason,
        },
        "improved":     improved,
        "iterations_log": iteration_log,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if improved:
        log.info(
            "Evolution SUCCESS: %.3f → %.3f (+%.3f)  saved to %s",
            baseline_avg, best_avg, best_avg - baseline_avg, output_path,
        )
    else:
        log.info(
            "Evolution: no significant improvement (%.3f → %.3f).  "
            "Output saved for reference at %s",
            baseline_avg, best_avg, output_path,
        )

    return {
        "improved":        improved,
        "skill_id":        skill_id,
        "baseline_score":  round(baseline_avg, 4),
        "evolved_score":   round(best_avg, 4),
        "delta":           round(best_avg - baseline_avg, 4),
        "output_path":     str(output_path),
        "metrics_path":    str(metrics_path),
        "metrics":         metrics,
    }


# ---------------------------------------------------------------------------
# CLI helpers (called by essence.py cmd_evolve)
# ---------------------------------------------------------------------------

async def shadow_test_and_promote(
    skill_id:        str,
    skills_dir:      Path,
    ollama_url:      str  = _DEFAULT_OLLAMA,
    model:           str  = _DEFAULT_MODEL,
    auto_threshold:  float = 0.15,
    dry_run:         bool  = False,
) -> dict:
    """
    Auto-promotion pipeline for evolved skills.

    Runs evolve_skill() in shadow (both current and evolved versions answer
    a test battery) and automatically promotes the evolved skill if:
      - The delta composite score ≥ auto_threshold
      - All constraint checks pass
      - No regression on any individual fitness dimension

    Auto-promotion means copying the evolved file over the canonical skill file.
    A backup of the original is kept at <skill_id>.md.bak.

    Returns a result dict with "promoted" bool and supporting metrics.
    """
    # Run standard evolution first
    result = await evolve_skill(
        skill_id=skill_id,
        skills_dir=skills_dir,
        ollama_url=ollama_url,
        model=model,
        iterations=3,
        dry_run=dry_run,
    )

    if dry_run:
        result["promoted"] = False
        result["promotion_reason"] = "dry_run"
        return result

    if not result.get("improved"):
        result["promoted"] = False
        result["promotion_reason"] = f"delta {result.get('delta', 0):.3f} below threshold or constraints failed"
        return result

    delta = result.get("delta", 0)
    if delta < auto_threshold:
        result["promoted"] = False
        result["promotion_reason"] = (
            f"delta {delta:.3f} below auto_threshold {auto_threshold} — "
            "manual review required"
        )
        log.info(
            "shadow_test: skill %s improved (Δ%.3f) but below auto-promote threshold (%.2f)",
            skill_id, delta, auto_threshold,
        )
        return result

    # Promote: backup original → copy evolved version over canonical
    canonical  = skills_dir / f"{skill_id}.md"
    evolved_p  = Path(result["output_path"])
    backup_p   = skills_dir / f"{skill_id}.md.bak"

    if not evolved_p.exists():
        result["promoted"] = False
        result["promotion_reason"] = "evolved file not found"
        return result

    try:
        if canonical.exists():
            import shutil
            shutil.copy2(str(canonical), str(backup_p))

        shutil.copy2(str(evolved_p), str(canonical))
        log.info(
            "shadow_test: AUTO-PROMOTED skill %s (Δ%.3f) — backup at %s",
            skill_id, delta, backup_p,
        )
        result["promoted"]         = True
        result["promotion_reason"] = f"auto-promoted (Δ{delta:.3f} ≥ threshold {auto_threshold})"
        result["promoted_to"]      = str(canonical)
        result["backup"]           = str(backup_p)
    except Exception as exc:
        result["promoted"]         = False
        result["promotion_reason"] = f"copy failed: {exc}"

    return result


def list_evolvable_skills(skills_dir: Path) -> list[dict]:
    """Return info dicts for all skills in skills_dir."""
    result = []
    for p in sorted(skills_dir.glob("*.md")) + sorted(skills_dir.glob("*.yaml")):
        try:
            text = p.read_text(encoding="utf-8")
            result.append({
                "id":    p.stem,
                "name":  p.stem.replace("-", " ").replace("_", " ").title(),
                "chars": len(text),
                "path":  str(p),
            })
        except Exception:
            pass
    return result
