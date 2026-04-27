"""
adaptive.py — Adaptive Behavior Engine
========================================
Self-calibrating layer that learns from interaction patterns to continuously
adjust Essence's response style, model selection, and behavior.

What it learns (all implicit — zero explicit ratings required):
  verbosity_pref  (0=terse … 1=detailed)        "be brief" / "elaborate"
  formality       (0=casual … 1=professional)    user writing style heuristic
  reasoning_depth (0=direct answer … 1=show all) "why?" follow-up frequency
  expertise[]     domain confidence              technical vocabulary density
  length_target   preferred response length      accepted-length EMA

What it changes automatically:
  • System prompt suffix injected each turn   (style directive)
  • max_tokens hint passed to kernel          (length calibration)
  • Model preference hints for ModelRouter    (empirical performance bias)

Signal sources (turn-level, after each response):
  record_turn(user_msg, response, provider, model)
    → detects correction/positive/negative signals from the user message
    → infers task type from vocabulary
    → updates profile via exponential moving average

  record_task(provider, model, task_type, success, latency)
    → feeds the performance ledger after plan execution

Storage (human-readable JSON, safely editable):
  data/adaptive_profile.json   — UserProfile dimensions
  data/adaptive_ledger.json    — model × task success rates

Kill-switch: set ESSENCE_ADAPTIVE=0 to disable all adaptation.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("essence.adaptive")

# ---------------------------------------------------------------------------
# Kill-switch (snapshotted at import time)
# ---------------------------------------------------------------------------
_ADAPTIVE_ENABLED = os.getenv("ESSENCE_ADAPTIVE", "1").lower() not in (
    "0", "false", "no", "off"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PROFILE_FILE = "data/adaptive_profile.json"
_LEDGER_FILE  = "data/adaptive_ledger.json"

# Exponential moving average — higher alpha = faster adaptation
_EMA_ALPHA_FAST = 0.20   # verbosity / reasoning (user explicitly corrects)
_EMA_ALPHA_SLOW = 0.07   # formality / length (inferred passively)
_EMA_ALPHA_RATE = 0.05   # positive / correction rates (rolling window feel)

# Minimum turns before the profile influences behavior (cold-start guard)
_MIN_TURNS = 5

# Performance ledger: minimum calls before a model override is trusted
_MIN_LEDGER_CALLS = 5

# ---------------------------------------------------------------------------
# Domain vocabulary — first matching domain wins for task-type classification
# ---------------------------------------------------------------------------
_DOMAIN_VOCAB: Dict[str, List[str]] = {
    "code": [
        "function", "class", "def ", "bug", "error", "exception", "implement",
        "refactor", "variable", "compile", "runtime", "algorithm", "script",
        "import", "python", "javascript", "typescript", "rust", "golang",
        "sql query", "regex", "rest api", "endpoint", "debug", "unittest",
    ],
    "reasoning": [
        "why", "explain", "because", "cause", "reason", "understand",
        "how does", "what if", "hypothesis", "analyze", "compare", "versus",
        "pros", "cons", "tradeoff", "implication", "consequence",
    ],
    "creative": [
        "write", "poem", "story", "creative", "imagine", "fiction", "draft",
        "essay", "narrative", "describe", "metaphor", "generate text",
    ],
    "retrieval": [
        "find", "search", "look up", "what is", "who is", "where is",
        "when did", "definition", "list all", "show me", "fetch", "get",
    ],
    "analysis": [
        "analyze", "summarize", "review", "evaluate", "assess", "breakdown",
        "pattern", "trend", "insight", "report", "audit",
    ],
    "system": [
        "configure", "setup", "install", "deploy", "server", "network",
        "linux", "windows", "docker", "kubernetes", "bash", "shell", "terminal",
        "ssh", "firewall", "port",
    ],
}

# ---------------------------------------------------------------------------
# Implicit signal regexes (applied to the user's NEXT message after a response)
# ---------------------------------------------------------------------------
_RE_SHORTER  = re.compile(r"\b(shorter|briefer?|concis\w*|less detail|too long|tl;?dr|summarize it)\b", re.I)
_RE_LONGER   = re.compile(r"\b(more detail|elaborate|expand|explain more|tell me more|longer|go deeper|in depth)\b", re.I)
_RE_SIMPLER  = re.compile(r"\b(simpler|plainer|layman|eli5|basic|beginner|dumb it down)\b", re.I)
_RE_DEEPER   = re.compile(r"\b(deeper|technical|advanced|step[ -]by[ -]step|show (your |the )?reasoning|why did you)\b", re.I)
_RE_POSITIVE = re.compile(r"\b(thanks?|thank you|perfect|great|excellent|exactly|awesome|helpful|well done|good job|nice|correct|nailed it|spot on)\b", re.I)
_RE_NEGATIVE = re.compile(r"\b(wrong|incorrect|that.s not|no that|you missed|actually,|not quite|try again|re-?do)\b", re.I)


# ---------------------------------------------------------------------------
# Task-type detection
# ---------------------------------------------------------------------------

def detect_task_type(text: str) -> str:
    """Map a message to one of 7 task types via vocabulary heuristic.

    Uses word-boundary matching for single-word keywords to avoid false
    positives (e.g. 'cause' inside 'because', 'api' inside 'capital').
    """
    lower = (text or "").lower()
    for task_type, keywords in _DOMAIN_VOCAB.items():
        for kw in keywords:
            if " " in kw:
                # Multi-word phrase: simple substring match is fine
                if kw in lower:
                    return task_type
            else:
                # Single word: require word boundary to avoid partial matches
                if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                    return task_type
    return "general"


# ---------------------------------------------------------------------------
# UserProfile
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    """
    Persisted, human-editable model of user communication preferences.
    All scalar dimensions are floats in [0.0, 1.0] updated via EMA.
    """
    # Style dimensions
    verbosity:        float = 0.5   # 0=terse, 1=detailed
    formality:        float = 0.5   # 0=casual, 1=professional
    reasoning_depth:  float = 0.40  # 0=direct, 1=show all steps

    # Domain expertise confidence (domain → 0–1)
    expertise: Dict[str, float] = field(default_factory=dict)

    # Length calibration
    avg_user_msg_len: float = 80.0
    avg_accept_len:   float = 400.0   # chars of "accepted" responses (EMA)

    # Quality signals (rolling rates)
    positive_rate:    float = 0.0
    correction_rate:  float = 0.0

    # Metadata
    turn_count: int   = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ── EMA helper ─────────────────────────────────────────────────────

    @staticmethod
    def _ema(current: float, target: float, alpha: float) -> float:
        return current + alpha * (target - current)

    # ── Dimension updates ──────────────────────────────────────────────

    def nudge_verbosity(self, direction: float) -> None:
        """direction: +1=more verbose, -1=more concise."""
        target = min(1.0, self.verbosity + 0.15 * direction)
        self.verbosity = self._ema(self.verbosity, max(0.0, min(1.0, target)), _EMA_ALPHA_FAST)

    def nudge_reasoning(self, direction: float) -> None:
        """direction: +1=more reasoning, -1=less."""
        target = min(1.0, self.reasoning_depth + 0.15 * direction)
        self.reasoning_depth = self._ema(
            self.reasoning_depth, max(0.0, min(1.0, target)), _EMA_ALPHA_FAST
        )

    def nudge_expertise(self, domain: str, delta: float) -> None:
        cur = self.expertise.get(domain, 0.25)
        self.expertise[domain] = max(0.0, min(1.0, self._ema(cur, cur + delta, _EMA_ALPHA_SLOW)))

    def absorb_len(self, response_len: int) -> None:
        self.avg_accept_len = self._ema(self.avg_accept_len, float(response_len), _EMA_ALPHA_SLOW)

    # ── Serialization ──────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# PerformanceLedger — per-model × per-task success/failure tracking
# ---------------------------------------------------------------------------

@dataclass
class _ModelStats:
    success: int   = 0
    failure: int   = 0
    latency: float = 0.0

    @property
    def calls(self) -> int:
        return self.success + self.failure

    @property
    def success_rate(self) -> float:
        return self.success / self.calls if self.calls else 0.5

    @property
    def avg_latency(self) -> float:
        return self.latency / self.calls if self.calls else 0.0


class PerformanceLedger:
    """Thread-safe, serializable per-model × per-task outcome tracker."""

    def __init__(self) -> None:
        # { "provider/model": { "task_type": _ModelStats } }
        self._data: Dict[str, Dict[str, _ModelStats]] = {}

    def record(
        self,
        provider: str,
        model: str,
        task_type: str,
        success: bool,
        latency: float = 0.0,
    ) -> None:
        key = f"{provider}/{model}"
        self._data.setdefault(key, {}).setdefault(task_type, _ModelStats())
        s = self._data[key][task_type]
        if success:
            s.success += 1
        else:
            s.failure += 1
        s.latency += latency

    def best_model_for(self, task_type: str, candidates: List[str]) -> Optional[str]:
        """
        Return the candidate (provider/model) with the highest empirical
        success rate for this task type.

        Requires ≥ _MIN_LEDGER_CALLS to override default ordering.
        Returns None if data is insufficient or the top-2 are within 10%.
        """
        scored: List[Tuple[str, float]] = []
        for key in candidates:
            stats = self._data.get(key, {}).get(task_type)
            if stats and stats.calls >= _MIN_LEDGER_CALLS:
                scored.append((key, stats.success_rate))
        if not scored:
            return None
        scored.sort(key=lambda x: x[1], reverse=True)
        # Only override if there is a meaningful gap
        if len(scored) >= 2 and (scored[0][1] - scored[1][1]) < 0.10:
            return None
        return scored[0][0]

    def summary(self) -> List[Tuple[str, str, float, int]]:
        """(model_key, task_type, success_rate, calls) sorted by calls desc."""
        rows = []
        for mk, tasks in self._data.items():
            for tt, s in tasks.items():
                rows.append((mk, tt, s.success_rate, s.calls))
        rows.sort(key=lambda x: x[3], reverse=True)
        return rows

    # ── Serialization ──────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            mk: {
                tt: {"success": s.success, "failure": s.failure, "latency": s.latency}
                for tt, s in tasks.items()
            }
            for mk, tasks in self._data.items()
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PerformanceLedger":
        ledger = cls()
        for mk, tasks in (d or {}).items():
            for tt, v in tasks.items():
                ledger._data.setdefault(mk, {})[tt] = _ModelStats(
                    success=v.get("success", 0),
                    failure=v.get("failure", 0),
                    latency=v.get("latency", 0.0),
                )
        return ledger


# ---------------------------------------------------------------------------
# BehaviorAdapter — translates UserProfile → system prompt fragment
# ---------------------------------------------------------------------------

class BehaviorAdapter:
    """Generates a compact style directive appended to the system prompt."""

    def get_suffix(self, profile: UserProfile) -> str:
        """
        Return a style directive string or "" during cold-start.
        This is appended to the system prompt each turn so the model
        adjusts its style to match the learned user preferences.
        """
        if profile.turn_count < _MIN_TURNS:
            return ""

        hints: List[str] = []

        # Verbosity
        v = profile.verbosity
        if v < 0.28:
            hints.append("Be extremely concise — terse, direct answers only")
        elif v < 0.44:
            hints.append("Keep responses brief and focused")
        elif v > 0.72:
            hints.append("Provide detailed, comprehensive responses")
        elif v > 0.87:
            hints.append("Be exhaustive — include context, examples, and full explanations")

        # Reasoning depth
        r = profile.reasoning_depth
        if r < 0.22:
            hints.append("Answer directly without showing intermediate steps")
        elif r > 0.65:
            hints.append("Show your reasoning step-by-step")

        # Formality
        f = profile.formality
        if f < 0.28:
            hints.append("Use casual, conversational language")
        elif f > 0.75:
            hints.append("Maintain a professional, precise tone")

        # Domain expertise
        experts = sorted(
            ((d, c) for d, c in profile.expertise.items() if c > 0.60),
            key=lambda x: -x[1],
        )
        if experts:
            names = ", ".join(d for d, _ in experts[:3])
            hints.append(f"User is expert in {names} — skip introductory explanations")

        if not hints:
            return ""

        return "\n\n[Adaptive style: " + " · ".join(hints) + "]"

    def max_tokens_hint(self, profile: UserProfile) -> Optional[int]:
        """Suggest a max_tokens override based on accepted response length."""
        if profile.turn_count < _MIN_TURNS:
            return None
        # chars / 3.5 ≈ tokens
        target = int(profile.avg_accept_len / 3.5)
        if target < 700:
            return max(200, target)
        if target > 3200:
            return min(8192, target)
        return None  # within normal range — no override


# ---------------------------------------------------------------------------
# AdaptiveEngine — top-level facade
# ---------------------------------------------------------------------------

class AdaptiveEngine:
    """
    Central adaptive intelligence layer.
    Thread-safe singleton; persists to disk after every turn.
    """

    def __init__(self, workspace: Path) -> None:
        self._ws      = workspace
        self._lock    = Lock()
        self._profile = self._load_profile()
        self._ledger  = self._load_ledger()
        self._adapter = BehaviorAdapter()

    # ── Paths ──────────────────────────────────────────────────────────

    def _pp(self) -> Path:
        p = self._ws / _PROFILE_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _lp(self) -> Path:
        p = self._ws / _LEDGER_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ── Persistence ────────────────────────────────────────────────────

    def _load_profile(self) -> UserProfile:
        try:
            p = self._pp()
            if p.exists():
                return UserProfile.from_dict(json.loads(p.read_text("utf-8")))
        except Exception as e:
            log.debug("adaptive: profile load failed: %s", e)
        return UserProfile()

    def _load_ledger(self) -> PerformanceLedger:
        try:
            p = self._lp()
            if p.exists():
                return PerformanceLedger.from_dict(json.loads(p.read_text("utf-8")))
        except Exception as e:
            log.debug("adaptive: ledger load failed: %s", e)
        return PerformanceLedger()

    def _flush(self) -> None:
        """Write profile + ledger to disk. Must be called under self._lock."""
        try:
            self._pp().write_text(
                json.dumps(self._profile.to_dict(), indent=2), encoding="utf-8"
            )
            self._lp().write_text(
                json.dumps(self._ledger.to_dict(), indent=2), encoding="utf-8"
            )
        except Exception as e:
            log.debug("adaptive: flush failed: %s", e)

    # ── Signal ingestion ───────────────────────────────────────────────

    def record_turn(
        self,
        user_message: str,
        response: str,
        provider: str = "",
        model: str = "",
    ) -> None:
        """
        Extract implicit signals from a completed turn and update the profile.
        Called from kernel._stream_response() after full_response is built.
        """
        if not _ADAPTIVE_ENABLED:
            return
        um = (user_message or "").strip()
        rs = (response or "").strip()
        if not um:
            return

        with self._lock:
            p = self._profile
            p.turn_count += 1
            p.updated_at = time.time()

            # ── Message length signal ──────────────────────────────────
            p.avg_user_msg_len = p._ema(p.avg_user_msg_len, float(len(um)), _EMA_ALPHA_SLOW)

            # ── Verbosity & length corrections ─────────────────────────
            if _RE_SHORTER.search(um):
                p.nudge_verbosity(-1)
                # Don't absorb the response length — they didn't like it
            elif _RE_LONGER.search(um):
                p.nudge_verbosity(+1)
            else:
                # No correction: this length was implicitly accepted
                if rs:
                    p.absorb_len(len(rs))

            # ── Reasoning depth corrections ────────────────────────────
            if _RE_DEEPER.search(um):
                p.nudge_reasoning(+1)
            elif _RE_SIMPLER.search(um):
                p.nudge_reasoning(-1)

            # ── Positive / negative quality signals ────────────────────
            if _RE_POSITIVE.search(um):
                p.positive_rate    = p._ema(p.positive_rate,    1.0, _EMA_ALPHA_RATE)
                p.correction_rate  = p._ema(p.correction_rate,  0.0, _EMA_ALPHA_RATE)
            elif _RE_NEGATIVE.search(um):
                p.correction_rate  = p._ema(p.correction_rate,  1.0, _EMA_ALPHA_RATE)
                p.positive_rate    = p._ema(p.positive_rate,    0.0, _EMA_ALPHA_RATE)
                # Negative on a long response → they wanted shorter
                if rs and len(rs) > 1000:
                    p.nudge_verbosity(-0.5)
            else:
                p.positive_rate   = p._ema(p.positive_rate,    0.0, _EMA_ALPHA_RATE)
                p.correction_rate = p._ema(p.correction_rate,  0.0, _EMA_ALPHA_RATE)

            # ── Domain expertise inference ─────────────────────────────
            task_type = detect_task_type(um)
            if task_type != "general":
                kws = _DOMAIN_VOCAB.get(task_type, [])
                hits = sum(1 for kw in kws if kw in um.lower())
                if hits >= 3:
                    p.nudge_expertise(task_type, 0.08)
                elif hits >= 2:
                    p.nudge_expertise(task_type, 0.04)
                elif hits == 1:
                    p.nudge_expertise(task_type, 0.01)

            # ── Formality inference ────────────────────────────────────
            # Heuristic: capitalised sentences with punctuation → formal
            sentences = [s.strip() for s in re.split(r"[.!?]", um) if len(s.strip()) > 8]
            if sentences:
                formal_count = sum(1 for s in sentences if s[0].isupper())
                formality_signal = formal_count / len(sentences)
                p.formality = p._ema(p.formality, formality_signal, _EMA_ALPHA_SLOW)

            self._flush()

    def record_task(
        self,
        provider: str,
        model: str,
        task_type: str,
        success: bool,
        latency: float = 0.0,
    ) -> None:
        """
        Record a plan execution outcome into the performance ledger.
        Called from kernel._reflect() after each plan run.
        """
        if not _ADAPTIVE_ENABLED:
            return
        with self._lock:
            self._ledger.record(provider, model, task_type, success, latency)
            self._flush()

    # ── Behavior outputs ───────────────────────────────────────────────

    def get_style_suffix(self) -> str:
        """Dynamic system prompt suffix based on current learned profile."""
        if not _ADAPTIVE_ENABLED:
            return ""
        with self._lock:
            return self._adapter.get_suffix(self._profile)

    def get_max_tokens_hint(self) -> Optional[int]:
        """Suggested max_tokens override, or None if no adjustment needed."""
        if not _ADAPTIVE_ENABLED:
            return None
        with self._lock:
            return self._adapter.max_tokens_hint(self._profile)

    def suggest_model(self, task_type: str, candidates: List[str]) -> Optional[str]:
        """
        Return the empirically best candidate (provider/model string) for
        this task type, or None if data is insufficient.
        """
        if not _ADAPTIVE_ENABLED:
            return None
        with self._lock:
            return self._ledger.best_model_for(task_type, candidates)

    # ── Introspection ──────────────────────────────────────────────────

    def profile_dict(self) -> dict:
        with self._lock:
            p = self._profile
            return {
                "turn_count":      p.turn_count,
                "verbosity":       round(p.verbosity, 3),
                "formality":       round(p.formality, 3),
                "reasoning_depth": round(p.reasoning_depth, 3),
                "expertise":       {k: round(v, 3) for k, v in sorted(p.expertise.items(), key=lambda x: -x[1])},
                "avg_accept_len":  int(p.avg_accept_len),
                "positive_rate":   round(p.positive_rate, 3),
                "correction_rate": round(p.correction_rate, 3),
                "updated_at":      p.updated_at,
                "mature":          p.turn_count >= _MIN_TURNS,
            }

    def ledger_summary(self) -> List[Tuple[str, str, float, int]]:
        with self._lock:
            return self._ledger.summary()

    def format_report(self) -> str:
        """Human-readable ASCII report for /adapt show."""
        data = self.profile_dict()
        ledger = self.ledger_summary()

        def bar(val: float, width: int = 22) -> str:
            filled = round(val * width)
            return "▓" * filled + "░" * (width - filled)

        maturity = (
            "● calibrated"
            if data["mature"]
            else f"○ cold-start ({data['turn_count']}/{_MIN_TURNS} turns)"
        )

        lines = [
            "┌─ Adaptive Profile ────────────────────────────────────────────────┐",
            f"│  Status         : {maturity:<50}│",
            f"│  Total turns    : {data['turn_count']:<50}│",
            "│                                                                    │",
            f"│  Verbosity      : {bar(data['verbosity'])}  {data['verbosity']:.3f}     │",
            f"│  Formality      : {bar(data['formality'])}  {data['formality']:.3f}     │",
            f"│  Reasoning depth: {bar(data['reasoning_depth'])}  {data['reasoning_depth']:.3f}     │",
            "│                                                                    │",
        ]

        if data["expertise"]:
            lines.append("│  Domain expertise:                                                 │")
            for domain, conf in list(data["expertise"].items())[:6]:
                lines.append(f"│    {domain:<14}: {bar(conf, 16)}  {conf:.3f}              │")
        else:
            lines.append("│  Domain expertise : (not yet calibrated)                           │")

        lines += [
            "│                                                                    │",
            f"│  Avg accepted len : ~{data['avg_accept_len']} chars                                 │",
            f"│  Positive signal  : {data['positive_rate']:.0%} of turns                              │",
            f"│  Correction rate  : {data['correction_rate']:.0%} of turns                              │",
        ]

        if ledger:
            lines += [
                "│                                                                    │",
                "│  Performance ledger (by call count):                               │",
            ]
            for model_key, task_type, rate, calls in ledger[:8]:
                short = model_key.split("/")[-1][:18]
                lines.append(
                    f"│    {short:<18}  {task_type:<12}  {rate:.0%} ({calls} calls)          │"
                )

        lines += [
            "└────────────────────────────────────────────────────────────────────┘",
            "  /adapt reset              — reset all learned preferences",
            "  /adapt reset verbosity    — reset one dimension",
            "  /adapt reset formality | reasoning | expertise | ledger",
        ]

        return "\n".join(lines)

    def reset(self, dimension: Optional[str] = None) -> str:
        """Reset profile dimension(s). Returns a status message."""
        with self._lock:
            if not dimension:
                self._profile = UserProfile()
                self._ledger  = PerformanceLedger()
                self._flush()
                return "Adaptive profile fully reset to defaults."

            p = self._profile
            resets = {
                "verbosity":  lambda: setattr(p, "verbosity", 0.5),
                "formality":  lambda: setattr(p, "formality", 0.5),
                "reasoning":  lambda: setattr(p, "reasoning_depth", 0.40),
                "expertise":  lambda: setattr(p, "expertise", {}),
                "rates":      lambda: (
                    setattr(p, "positive_rate", 0.0),
                    setattr(p, "correction_rate", 0.0),
                ),
                "ledger":     lambda: setattr(self, "_ledger", PerformanceLedger()),
            }
            fn = resets.get(dimension.lower())
            if fn:
                fn()
                self._flush()
                return f"Reset: {dimension}."
            return (
                f"Unknown dimension '{dimension}'. "
                "Options: verbosity formality reasoning expertise rates ledger"
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine: Optional[AdaptiveEngine] = None
_lock   = Lock()


def get_adaptive_engine(workspace: Optional[Path] = None) -> AdaptiveEngine:
    """Return the process-level AdaptiveEngine singleton."""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                ws = workspace or Path(__file__).resolve().parent.parent
                _engine = AdaptiveEngine(ws)
    return _engine
