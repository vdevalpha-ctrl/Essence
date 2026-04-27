"""
htn_planner.py — Hierarchical Task Network Planner for Essence
===============================================================
Decomposes high-level user intents into trees of primitive operations
using a library of task decomposition methods.

Architecture:
  TaskNode          — a node in the task tree (compound or primitive)
  Method            — one way to decompose a compound task
  HTNPlanner        — matches methods to tasks and expands the tree
  plan_to_steps()   — converts a TaskNode tree to a flat list of PlanSteps

The HTN planner runs as Pass 4 of the kernel's 3-pass planner
(procedural → capability → LLM → HTN).  It fires when the LLM
planner returns None (i.e., no single-skill plan could be formed)
and the intent complexity warrants decomposition.

Built-in methods cover:
  research   → [web_search, summarise]
  write      → [outline, draft, review]
  debug      → [read_file, shell, analyze]
  deploy     → [test, build, shell]
  refactor   → [read_file, analyze, write_file]
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("essence.htn")


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class TaskNode:
    name:        str
    task_type:   str                      # compound | primitive
    args:        dict                     = field(default_factory=dict)
    children:    list["TaskNode"]         = field(default_factory=list)
    skill_id:    str                      = ""   # for primitives
    depends_on:  list[str]                = field(default_factory=list)
    id:          str                      = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class Method:
    name:        str
    applies_to:  str                      # regex matched against intent
    subtasks:    list[dict]               # list of subtask dicts
    priority:    int   = 0


# ── Built-in methods ───────────────────────────────────────────────────────

_BUILTIN_METHODS: list[Method] = [
    Method(
        name="research",
        applies_to=r"\b(research|find|look up|investigate|what is|who is|when did)\b",
        subtasks=[
            {"name": "web_search",  "skill_id": "research", "args": {"query": "${intent}"}},
            {"name": "summarise",   "skill_id": "summarise", "args": {"text": "${web_search.result}"}},
        ],
        priority=2,
    ),
    Method(
        name="write_document",
        applies_to=r"\b(write|draft|compose|create)\b.{0,40}\b(doc|report|essay|article|email|letter)\b",
        subtasks=[
            {"name": "outline",   "skill_id": "write",    "args": {"task": "outline", "topic": "${intent}"}},
            {"name": "draft",     "skill_id": "write",    "args": {"task": "draft",   "outline": "${outline.result}"}},
            {"name": "review",    "skill_id": "summarise","args": {"text": "${draft.result}"}},
        ],
        priority=1,
    ),
    Method(
        name="debug_code",
        applies_to=r"\b(debug|fix|error|traceback|exception|crash|failing)\b",
        subtasks=[
            {"name": "read_context", "skill_id": "shell",     "args": {"command": "git diff HEAD~1"}},
            {"name": "analyze",      "skill_id": "research",  "args": {"query": "${intent}"}},
        ],
        priority=3,
    ),
    Method(
        name="refactor_code",
        applies_to=r"\b(refactor|clean up|improve|reorganize|simplify)\b.{0,30}\b(code|function|class|module)\b",
        subtasks=[
            {"name": "read_file",  "skill_id": "shell",   "args": {"command": "cat ${file}"}},
            {"name": "analyze",    "skill_id": "research","args": {"query": "best practices for ${intent}"}},
            {"name": "rewrite",    "skill_id": "write",   "args": {"task": "rewrite", "code": "${read_file.result}"}},
        ],
        priority=1,
    ),
    Method(
        name="deploy_app",
        applies_to=r"\b(deploy|ship|release|publish|push to production|go live)\b",
        subtasks=[
            {"name": "test",   "skill_id": "shell", "args": {"command": "python -m pytest --tb=short -q"}},
            {"name": "build",  "skill_id": "shell", "args": {"command": "python setup.py build"}},
            {"name": "deploy", "skill_id": "shell", "args": {"command": "git push origin main"}},
        ],
        priority=2,
    ),
]


# ── HTN Planner ────────────────────────────────────────────────────────────

class HTNPlanner:
    """
    Matches user intent against method patterns and expands into a TaskNode tree.
    """

    def __init__(self) -> None:
        self._methods: list[Method] = list(_BUILTIN_METHODS)

    def add_method(self, method: Method) -> None:
        self._methods.append(method)

    def plan(self, intent: str) -> Optional[TaskNode]:
        """
        Try to match the intent against built-in methods.
        Returns a compound TaskNode or None if no method matches.
        """
        intent_lower = intent.lower()
        best: Optional[Method] = None
        best_prio = -1

        for m in self._methods:
            if re.search(m.applies_to, intent_lower, re.IGNORECASE):
                if m.priority > best_prio:
                    best, best_prio = m, m.priority

        if best is None:
            return None

        log.debug("HTN: matched method %r for intent %r", best.name, intent[:60])
        return self._expand(best, intent)

    def _expand(self, method: Method, intent: str) -> TaskNode:
        root = TaskNode(
            name=method.name,
            task_type="compound",
            args={"intent": intent},
        )
        prev_id: Optional[str] = None
        for sub in method.subtasks:
            child = TaskNode(
                name=sub["name"],
                task_type="primitive",
                skill_id=sub.get("skill_id", ""),
                args={**sub.get("args", {}), "intent": intent},
                depends_on=[prev_id] if prev_id else [],
            )
            root.children.append(child)
            prev_id = child.id
        return root


# ── Tree → PlanSteps converter ─────────────────────────────────────────────

def plan_to_steps(node: TaskNode) -> list[dict]:
    """
    Flatten a TaskNode tree into a list of PlanStep-compatible dicts.
    Preserves depends_on edges for parallel DAG scheduling.
    """
    steps = []
    if node.task_type == "primitive":
        steps.append({
            "skill_id":   node.skill_id,
            "input_map":  node.args,
            "depends_on": node.depends_on,
            "required":   True,
        })
    for child in node.children:
        steps.extend(plan_to_steps(child))
    return steps


# ── Module-level singleton ─────────────────────────────────────────────────

_planner: Optional[HTNPlanner] = None


def get_htn_planner() -> HTNPlanner:
    global _planner
    if _planner is None:
        _planner = HTNPlanner()
    return _planner
