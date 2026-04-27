"""
dynamic_router.py — Per-Request Capability-Aware Routing
=========================================================
The brain that decides *what model* to use for *each individual request*,
rather than picking one at session start and using it for everything.

Design principles
-----------------
• Zero LLM calls for routing — pure heuristics, < 1 ms
• Provider-agnostic — works with whatever keys are present
• Capability-driven — selects for what the request actually needs
• Privacy-aware — forces local models when sensitive content detected
• Tool-selective — sends only tools relevant to the request (reduces noise)
• Self-healing — routes around rate-limited / erroring providers automatically
• Learns — tracks per-provider success/latency to improve future routing

Request lifecycle
-----------------
  User text
    ↓
  RequestClassifier.classify()  →  RequestProfile
    ↓
  DynamicRouter.route()         →  [(provider, model), ...]   fallback chain
    ↓
  ToolSelector.select()         →  [tool_def, ...]            relevant subset
    ↓
  Kernel streams the request

Integration with kernel.py
--------------------------
  Call get_dynamic_router() once at startup, then per-request:

      profile  = _dr.classify(text, history, images, all_tool_defs)
      chain    = _dr.route(profile)
      tools    = _dr.select_tools(profile, all_tool_defs)
      # pass chain as the candidates list, tools as the tools list
"""
from __future__ import annotations

import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger("essence.dynamic_router")

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    CHAT     = "chat"       # conversational, open-ended
    CODE     = "code"       # code generation, debugging, review
    RESEARCH = "research"   # web search, current events, fact-finding
    ANALYSIS = "analysis"   # reasoning, math, document analysis
    VISION   = "vision"     # image understanding
    CREATIVE = "creative"   # writing, brainstorming, storytelling
    PLAN     = "plan"       # multi-step task planning
    QUICK    = "quick"      # one-liner, needs fastest possible response


class PrivacyLevel(str, Enum):
    LOCAL_ONLY = "local"   # sensitive — never send to cloud
    CLOUD_OK   = "cloud"   # safe to use cloud providers


class CostTier(str, Enum):
    FREE     = "free"      # local / free-tier only
    CHEAP    = "cheap"     # ≤ $0.10 / 1M tokens
    STANDARD = "standard"  # ≤ $0.50 / 1M tokens
    PREMIUM  = "premium"   # any cost


# ---------------------------------------------------------------------------
# Capability set
# ---------------------------------------------------------------------------

@dataclass
class Capabilities:
    tools:     bool = False  # needs function/tool calling
    vision:    bool = False  # request contains images
    long_ctx:  bool = False  # needs > 16k token context
    reasoning: bool = False  # multi-step chain-of-thought
    code:      bool = False  # code generation / analysis
    streaming: bool = True   # streaming output (default yes)


# ---------------------------------------------------------------------------
# Request profile — output of classification
# ---------------------------------------------------------------------------

@dataclass
class RequestProfile:
    task_type:            TaskType
    capabilities:         Capabilities
    privacy:              PrivacyLevel
    cost_tier:            CostTier
    urgency:              float        # 0.0 = batch, 1.0 = real-time
    estimated_tokens:     int
    relevant_tool_names:  list[str]    = field(default_factory=list)
    signals:              dict         = field(default_factory=dict)  # debug info


# ---------------------------------------------------------------------------
# Static capability registry — what each provider/model family can do
# ---------------------------------------------------------------------------

# provider → frozenset of capability tags
_PROVIDER_CAPS: dict[str, frozenset] = {
    "ollama":     frozenset({"tools", "vision", "code", "streaming"}),
    "groq":       frozenset({"tools", "code", "streaming", "fast"}),
    "openai":     frozenset({"tools", "vision", "code", "long_ctx", "streaming", "reasoning"}),
    "anthropic":  frozenset({"tools", "vision", "code", "long_ctx", "streaming", "reasoning"}),
    "mistral":    frozenset({"tools", "code", "streaming"}),
    "together":   frozenset({"tools", "code", "streaming", "fast"}),
    "deepseek":   frozenset({"tools", "code", "streaming"}),
    "gemini":     frozenset({"tools", "vision", "code", "long_ctx", "streaming"}),
    "openrouter": frozenset({"tools", "vision", "code", "long_ctx", "streaming"}),
    "hf_local":  frozenset({"code", "streaming"}),
    "llamacpp":  frozenset({"code", "streaming"}),
}

# Cost per 1M tokens (prompt, completion) — mirrors model_router.py
_PROVIDER_COST: dict[str, tuple[float, float]] = {
    "ollama":     (0.00,  0.00),
    "hf_local":   (0.00,  0.00),
    "llamacpp":   (0.00,  0.00),
    "groq":       (0.05,  0.08),
    "together":   (0.10,  0.30),
    "deepseek":   (0.14,  0.28),
    "gemini":     (0.075, 0.30),
    "mistral":    (0.10,  0.30),
    "openrouter": (0.10,  0.40),
    "openai":     (0.15,  0.60),
    "anthropic":  (0.25,  1.25),
}

# Estimated latency tier: lower = faster
_PROVIDER_LATENCY: dict[str, int] = {
    "groq":       1,   # ultra fast
    "together":   2,
    "ollama":     3,   # depends on hardware
    "hf_local":   3,
    "mistral":    3,
    "deepseek":   3,
    "gemini":     3,
    "openai":     4,
    "anthropic":  4,
    "openrouter": 4,
    "llamacpp":   5,
}

# Task-type → preferred provider order (override the static TASK_PROVIDER_ORDER)
_TASK_PROVIDER_PREFERENCE: dict[str, list[str]] = {
    TaskType.QUICK:    ["groq", "together", "ollama", "mistral", "openai", "anthropic"],
    TaskType.CODE:     ["deepseek", "ollama", "openai", "anthropic", "groq", "mistral"],
    TaskType.RESEARCH: ["openai", "anthropic", "groq", "ollama", "mistral"],
    TaskType.ANALYSIS: ["anthropic", "openai", "deepseek", "groq", "ollama"],
    TaskType.VISION:   ["openai", "anthropic", "gemini", "ollama"],
    TaskType.PLAN:     ["anthropic", "openai", "deepseek", "groq", "ollama"],
    TaskType.CREATIVE: ["anthropic", "openai", "groq", "ollama"],
    TaskType.CHAT:     ["ollama", "groq", "openai", "anthropic", "mistral"],
}

# Default models per provider (better defaults than the router's conservative choices)
_PROVIDER_BEST_MODEL: dict[str, str] = {
    "groq":       "llama-3.3-70b-versatile",
    "openai":     "gpt-4o-mini",
    "anthropic":  "claude-3-5-haiku-20241022",
    "mistral":    "mistral-small-latest",
    "together":   "meta-llama/Llama-3-70b-chat-hf",
    "deepseek":   "deepseek-chat",
    "gemini":     "gemini-1.5-flash",
    "openrouter": "anthropic/claude-3-haiku",
    "ollama":     "qwen3:4b",
    "hf_local":   "local",
    "llamacpp":   "local",
}


# ---------------------------------------------------------------------------
# Request classifier — fast heuristic, zero LLM calls
# ---------------------------------------------------------------------------

# Keyword sets for task type detection
_CODE_SIGNALS = re.compile(
    r'\b(code|function|class|def |import |bug|error|exception|syntax|'
    r'debug|refactor|implement|algorithm|script|program|compile|run|'
    r'python|javascript|typescript|rust|go|java|c\+\+|sql|bash|shell|'
    r'regex|api|endpoint|unittest|test case)\b',
    re.IGNORECASE,
)
_RESEARCH_SIGNALS = re.compile(
    r'\b(search|find|look up|latest|current|today|news|weather|price|'
    r'stock|what is happening|who is|when did|where is|recent|2024|2025|'
    r'2026|wikipedia|article|paper|publication)\b',
    re.IGNORECASE,
)
_ANALYSIS_SIGNALS = re.compile(
    r'\b(analyze|analyse|explain|compare|contrast|summarize|evaluate|'
    r'calculate|compute|reason|logic|proof|derive|infer|conclude|'
    r'statistics|probability|math|equation|formula)\b',
    re.IGNORECASE,
)
_PLAN_SIGNALS = re.compile(
    r'\b(plan|steps|how to|guide|strategy|roadmap|workflow|process|'
    r'procedure|checklist|schedule|organize|breakdown|outline)\b',
    re.IGNORECASE,
)
_CREATIVE_SIGNALS = re.compile(
    r'\b(write|story|poem|essay|creative|imagine|brainstorm|ideas|'
    r'suggest|describe|narrative|character|plot|draft)\b',
    re.IGNORECASE,
)
_QUICK_SIGNALS = re.compile(
    r'^.{0,60}$',   # very short message
)
_PRIVACY_SIGNALS = re.compile(
    r'\b(password|secret|token|api.?key|credential|ssn|social.?security|'
    r'credit.?card|private|confidential|internal|proprietary)\b',
    re.IGNORECASE,
)
_SENSITIVE_PATTERNS = re.compile(
    r'(?:password|passwd|api.?key|secret|token)\s*[=:]\s*\S+',
    re.IGNORECASE,
)


class RequestClassifier:
    """
    Classify a request in < 1 ms using keyword heuristics.
    No LLM calls, no I/O.
    """

    def classify(
        self,
        text:       str,
        history:    list[dict],
        images:     list       = None,
        tool_defs:  list[dict] = None,
    ) -> RequestProfile:
        images    = images or []
        tool_defs = tool_defs or []
        text_l    = (text or "").lower()

        # ── Task type ────────────────────────────────────────────────
        scores: dict[str, float] = {t: 0.0 for t in TaskType}

        if images:
            scores[TaskType.VISION] += 5.0

        code_m     = len(_CODE_SIGNALS.findall(text))
        research_m = len(_RESEARCH_SIGNALS.findall(text))
        analysis_m = len(_ANALYSIS_SIGNALS.findall(text))
        plan_m     = len(_PLAN_SIGNALS.findall(text))
        creative_m = len(_CREATIVE_SIGNALS.findall(text))

        scores[TaskType.CODE]     += code_m * 1.5
        scores[TaskType.RESEARCH] += research_m * 1.5
        scores[TaskType.ANALYSIS] += analysis_m * 1.0
        scores[TaskType.PLAN]     += plan_m * 1.2
        scores[TaskType.CREATIVE] += creative_m * 1.0

        # Code blocks are strong code signals
        if "```" in text or "`" in text:
            scores[TaskType.CODE] += 2.0

        # URLs → likely research
        if re.search(r'https?://', text):
            scores[TaskType.RESEARCH] += 2.0

        # Short message → quick
        if len(text.strip()) < 60 and code_m == 0:
            scores[TaskType.QUICK] += 3.0

        # Fallback to chat
        scores[TaskType.CHAT] += 0.5

        task_type = max(scores, key=scores.__getitem__)

        # ── Capabilities ─────────────────────────────────────────────
        caps = Capabilities()
        caps.vision    = bool(images)
        caps.tools     = (task_type in (TaskType.RESEARCH, TaskType.ANALYSIS) or
                          bool(tool_defs) and research_m > 0)
        caps.code      = (task_type == TaskType.CODE or code_m > 0)
        caps.reasoning = (task_type in (TaskType.ANALYSIS, TaskType.PLAN) or
                          analysis_m >= 2)
        # Estimate token count from text + history
        hist_tokens = sum(len(str(m.get("content", ""))) // 4 for m in (history or []))
        msg_tokens  = len(text) // 4
        est_tokens  = hist_tokens + msg_tokens
        caps.long_ctx  = est_tokens > 8000

        # ── Privacy ──────────────────────────────────────────────────
        is_private = bool(
            _PRIVACY_SIGNALS.search(text) or
            _SENSITIVE_PATTERNS.search(text)
        )
        privacy = PrivacyLevel.LOCAL_ONLY if is_private else PrivacyLevel.CLOUD_OK

        # ── Cost tier ────────────────────────────────────────────────
        # Default to free/cheap; escalate for complex tasks
        if task_type in (TaskType.ANALYSIS, TaskType.PLAN) or caps.reasoning:
            cost_tier = CostTier.STANDARD
        elif task_type in (TaskType.VISION, TaskType.RESEARCH):
            cost_tier = CostTier.CHEAP
        else:
            cost_tier = CostTier.FREE

        # ── Urgency ──────────────────────────────────────────────────
        urgency = 1.0 if task_type == TaskType.QUICK else 0.5

        # ── Relevant tools ───────────────────────────────────────────
        relevant_tool_names = _score_tools(text, tool_defs, task_type)

        signals = {
            "task_scores":  {k: round(v, 2) for k, v in scores.items() if v > 0},
            "code_m":       code_m,
            "research_m":   research_m,
            "est_tokens":   est_tokens,
            "is_private":   is_private,
        }

        return RequestProfile(
            task_type=task_type,
            capabilities=caps,
            privacy=privacy,
            cost_tier=cost_tier,
            urgency=urgency,
            estimated_tokens=est_tokens,
            relevant_tool_names=relevant_tool_names,
            signals=signals,
        )


def _score_tools(text: str, tool_defs: list[dict], task_type: str) -> list[str]:
    """
    Return tool names most relevant to the request (up to 8).
    Uses lightweight keyword overlap between request and tool descriptions.
    """
    if not tool_defs:
        return []

    text_words = set(re.findall(r'\b\w{4,}\b', text.lower()))
    if not text_words:
        return [t.get("function", t).get("name", "") for t in tool_defs[:8]]

    # Always include task-type-implied tools
    always: set[str] = set()
    if task_type in (TaskType.RESEARCH, TaskType.QUICK):
        always.update({"web_search", "fetch_page", "http_get"})
    if task_type == TaskType.CODE:
        always.update({"shell", "read_file", "write_file"})

    scored: list[tuple[float, str]] = []
    for td in tool_defs:
        fn   = td.get("function", td)
        name = fn.get("name", "")
        desc = fn.get("description", "")
        # Keyword overlap score
        tool_words = set(re.findall(r'\b\w{4,}\b', (name + " " + desc).lower()))
        overlap    = len(text_words & tool_words)
        # Name match bonus
        name_score = 2.0 if any(w in name for w in text_words) else 0.0
        score      = overlap + name_score + (3.0 if name in always else 0.0)
        scored.append((score, name))

    # Sort by score descending, return top 8 names with score > 0
    scored.sort(reverse=True)
    result = [name for score, name in scored if score > 0][:8]

    # Ensure always-include tools are present
    for name in always:
        if name not in result:
            result.append(name)

    return result


# ---------------------------------------------------------------------------
# Provider scorer
# ---------------------------------------------------------------------------

class ProviderScorer:
    """
    Score a (provider, model) pair against a RequestProfile.
    Returns 0.0–1.0; higher is better.
    """

    def score(
        self,
        provider:    str,
        model:       str,
        profile:     RequestProfile,
        health:      dict | None = None,  # {provider: {error_rate, avg_latency_ms}}
    ) -> float:
        health = health or {}
        caps   = _PROVIDER_CAPS.get(provider, frozenset())
        cost   = _PROVIDER_COST.get(provider, (1.0, 2.0))
        lat    = _PROVIDER_LATENCY.get(provider, 5)
        s      = 1.0   # base score

        # ── Capability fit ───────────────────────────────────────────
        needed = profile.capabilities
        if needed.tools   and "tools"    not in caps: s -= 0.5
        if needed.vision  and "vision"   not in caps: s -= 0.8   # hard requirement
        if needed.long_ctx and "long_ctx" not in caps: s -= 0.3
        if needed.code    and "code"     not in caps: s -= 0.2
        if needed.reasoning and "reasoning" not in caps: s -= 0.1

        # ── Privacy ──────────────────────────────────────────────────
        if profile.privacy == PrivacyLevel.LOCAL_ONLY and provider not in ("ollama", "hf_local", "llamacpp"):
            return 0.0   # hard exclusion

        # ── Cost tier ────────────────────────────────────────────────
        avg_cost_per_1m = (cost[0] + cost[1]) / 2
        if profile.cost_tier == CostTier.FREE and avg_cost_per_1m > 0.01:
            s -= 0.4
        elif profile.cost_tier == CostTier.CHEAP and avg_cost_per_1m > 0.15:
            s -= 0.2

        # ── Latency fit ───────────────────────────────────────────────
        if profile.urgency > 0.7 and lat > 2:
            s -= (lat - 2) * 0.1   # penalize slow providers for urgent requests
        if profile.urgency > 0.7 and "fast" in caps:
            s += 0.15

        # ── Health / availability ─────────────────────────────────────
        ph = health.get(provider, {})
        error_rate = ph.get("error_rate", 0.0)
        s -= error_rate * 0.5     # penalise recently-erroring providers

        # ── Task-type preference bonus ────────────────────────────────
        pref = _TASK_PROVIDER_PREFERENCE.get(profile.task_type, [])
        if provider in pref:
            rank_bonus = (len(pref) - pref.index(provider)) / len(pref) * 0.2
            s += rank_bonus

        return max(0.0, s)


# ---------------------------------------------------------------------------
# Dynamic router — main entry point
# ---------------------------------------------------------------------------

class DynamicRouter:
    """
    Per-request provider/model selection.

    Usage:
        dr = DynamicRouter(available_providers)
        profile = dr.classify(text, history, images, tool_defs)
        chain   = dr.route(profile)     # [(provider, model), ...] fallback list
        tools   = dr.select_tools(profile, all_tool_defs)
    """

    def __init__(self, available_providers: dict | None = None) -> None:
        """
        available_providers: {provider_id: {"api_key": ..., "base_url": ...}}
        If None, reads from environment.
        """
        self._available  = available_providers or {}
        self._classifier = RequestClassifier()
        self._scorer     = ProviderScorer()
        self._health: dict[str, dict] = {}   # live health data from rate-limit tracker
        self._last_route: dict = {}           # debug info for last routing decision

    # ── Public API ─────────────────────────────────────────────────

    def classify(
        self,
        text:      str,
        history:   list[dict] = None,
        images:    list       = None,
        tool_defs: list[dict] = None,
    ) -> RequestProfile:
        return self._classifier.classify(text, history, images, tool_defs)

    def route(
        self,
        profile:   RequestProfile,
        preferred: tuple[str, str] | None = None,   # (provider, model) from kernel config
    ) -> list[tuple[str, str]]:
        """
        Return ordered fallback chain: [(provider, model), ...].
        First entry is the recommended choice; rest are fallbacks.

        If 'preferred' is set and scores above threshold, it is honoured —
        the system will NOT silently switch providers unless necessary.
        """
        # Refresh health snapshot
        self._refresh_health()

        candidates: list[tuple[float, str, str]] = []   # (score, provider, model)

        for pid, best_model in _PROVIDER_BEST_MODEL.items():
            # Must have an API key or be local
            if not self._is_available(pid):
                continue

            score = self._scorer.score(pid, best_model, profile, self._health)
            if score > 0.0:
                candidates.append((score, pid, best_model))

        # Sort by score descending
        candidates.sort(reverse=True)

        chain = [(pid, mdl) for _, pid, mdl in candidates]

        # Honour preferred if it scores reasonably well (≥ 0.4)
        # — avoids unnecessary switching when the current provider is fine
        if preferred:
            pref_p, pref_m = preferred
            pref_score = self._scorer.score(pref_p, pref_m, profile, self._health)
            best_score = candidates[0][0] if candidates else 0.0
            if pref_score >= 0.4 or pref_score >= best_score * 0.85:
                # Keep preferred at front
                chain = [(pref_p, pref_m)] + [(p, m) for p, m in chain if p != pref_p]

        if not chain:
            # Last resort: whatever the preferred was
            chain = [preferred] if preferred else [("ollama", "qwen3:4b")]

        log.debug(
            "DynamicRouter: task=%s  top3=%s",
            profile.task_type,
            [(p, round(s, 2)) for s, p, _ in candidates[:3]],
        )
        self._last_route = {
            "task":       profile.task_type,
            "top_choice": chain[0],
            "chain_len":  len(chain),
            "signals":    profile.signals,
        }
        return chain

    def select_tools(
        self,
        profile:   RequestProfile,
        all_tools: list[dict],
    ) -> list[dict]:
        """
        Return the subset of all_tools relevant to this request.
        Caps at 8 tools to keep the context tight.
        """
        if not all_tools:
            return []
        if not profile.relevant_tool_names:
            # No relevance signal — return nothing (let model use knowledge)
            return []

        name_set = set(profile.relevant_tool_names)
        result   = [t for t in all_tools
                    if (t.get("function") or t).get("name", "") in name_set]
        return result[:8]

    def update_available(self, providers: dict) -> None:
        """Refresh the available providers dict (called after key changes)."""
        self._available = providers

    @property
    def last_route(self) -> dict:
        return self._last_route

    # ── Internals ──────────────────────────────────────────────────

    def _is_available(self, provider: str) -> bool:
        """True if the provider has a key or is local."""
        if provider in ("ollama", "hf_local", "llamacpp"):
            return True   # always available (may fail at runtime)
        # Check injected config
        cfg = self._available.get(provider, {})
        if cfg.get("api_key"):
            return True
        # Check environment
        env_key = f"{provider.upper()}_API_KEY"
        ess_key = f"ESSENCE_{provider.upper()}_KEY"
        return bool(os.environ.get(env_key) or os.environ.get(ess_key))

    def _refresh_health(self) -> None:
        """Pull latest health data from the rate-limit tracker."""
        try:
            from server.rate_limit_tracker import get_tracker as _get_rl
            tracker = _get_rl()
            for pid in list(_PROVIDER_BEST_MODEL.keys()):
                state = tracker.get(pid)
                if state:
                    self._health[pid] = {
                        "error_rate":    getattr(state, "error_rate", 0.0),
                        "avg_latency_ms": getattr(state, "avg_latency_ms", 0),
                        "is_limited":    getattr(state, "is_limited", False),
                    }
                    # Hard-exclude rate-limited providers
                    if self._health[pid].get("is_limited"):
                        self._health[pid]["error_rate"] = 1.0
        except Exception:
            pass   # health data is best-effort


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_dr: DynamicRouter | None = None


def get_dynamic_router(providers: dict | None = None) -> DynamicRouter:
    global _dr
    if _dr is None:
        _dr = DynamicRouter(providers)
    elif providers:
        _dr.update_available(providers)
    return _dr


def init_dynamic_router(providers: dict) -> DynamicRouter:
    global _dr
    _dr = DynamicRouter(providers)
    return _dr
