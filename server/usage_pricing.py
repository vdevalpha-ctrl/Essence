"""
usage_pricing.py — Token Usage Tracking & Cost Estimation
==========================================================
Adapted from Hermes agent/usage_pricing.py — all external dependencies
removed; pricing table maintained internally.

Provides:
  CanonicalUsage   — unified token-count struct (input/output/cache)
  estimate_cost()  — USD cost estimate from model + usage
  format_usage()   — human-readable usage + cost summary for /cost command

Pricing table covers the most common providers. For models not in the
table, cost returns None (unknown) rather than silently returning $0.

Integration
-----------
  kernel._stream_response() builds CanonicalUsage from each API response
  and calls get_usage_tracker().record() so the /cost TUI command can
  display cumulative session cost.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from threading import Lock
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# CanonicalUsage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalUsage:
    """Provider-agnostic token usage for one API call."""
    input_tokens:       int = 0
    output_tokens:      int = 0
    cache_read_tokens:  int = 0
    cache_write_tokens: int = 0
    reasoning_tokens:   int = 0   # o-series / thinking models
    request_count:      int = 1
    raw_usage:          Optional[Dict[str, Any]] = None

    @property
    def prompt_tokens(self) -> int:
        """Total tokens sent (including cache hits)."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @staticmethod
    def from_response(usage_dict: Dict[str, Any]) -> "CanonicalUsage":
        """Parse a usage dict from any OpenAI-compatible response body."""
        if not usage_dict:
            return CanonicalUsage()
        return CanonicalUsage(
            input_tokens       = int(usage_dict.get("prompt_tokens", 0)
                                     or usage_dict.get("input_tokens", 0)),
            output_tokens      = int(usage_dict.get("completion_tokens", 0)
                                     or usage_dict.get("output_tokens", 0)),
            cache_read_tokens  = int(
                (usage_dict.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                or usage_dict.get("cache_read_input_tokens", 0)
            ),
            cache_write_tokens = int(usage_dict.get("cache_creation_input_tokens", 0)),
            reasoning_tokens   = int(
                (usage_dict.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
            ),
            raw_usage          = usage_dict,
        )

    def __add__(self, other: "CanonicalUsage") -> "CanonicalUsage":
        return CanonicalUsage(
            input_tokens       = self.input_tokens       + other.input_tokens,
            output_tokens      = self.output_tokens      + other.output_tokens,
            cache_read_tokens  = self.cache_read_tokens  + other.cache_read_tokens,
            cache_write_tokens = self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens   = self.reasoning_tokens   + other.reasoning_tokens,
            request_count      = self.request_count      + other.request_count,
        )


# ---------------------------------------------------------------------------
# Pricing table  (USD per million tokens)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Price:
    inp:   float          # input $/M
    out:   float          # output $/M
    cr:    float = 0.0    # cache-read $/M  (0 = unknown / not supported)
    cw:    float = 0.0    # cache-write $/M

    def cost(self, usage: CanonicalUsage) -> Decimal:
        M = Decimal("1000000")
        total = (
            Decimal(str(self.inp)) * Decimal(str(usage.input_tokens)) / M
            + Decimal(str(self.out)) * Decimal(str(usage.output_tokens)) / M
        )
        if usage.cache_read_tokens and self.cr:
            total += Decimal(str(self.cr)) * Decimal(str(usage.cache_read_tokens)) / M
        if usage.cache_write_tokens and self.cw:
            total += Decimal(str(self.cw)) * Decimal(str(usage.cache_write_tokens)) / M
        return total


# Model slug → price.  Checked with longest-prefix matching so
# "claude-sonnet-4-20250514" also matches "claude-sonnet-4".
_PRICING: Dict[str, _Price] = {
    # ── Anthropic ────────────────────────────────────────────────────────
    "claude-opus-4":           _Price(15.00, 75.00,  1.50, 18.75),
    "claude-sonnet-4":        _Price( 3.00, 15.00,  0.30,  3.75),
    "claude-haiku-3-5":       _Price( 0.80,  4.00,  0.08,  1.00),
    "claude-haiku-3":         _Price( 0.25,  1.25,  0.03,  0.30),
    "claude-opus-3-5":        _Price(15.00, 75.00,  1.50, 18.75),
    "claude-sonnet-3-5":      _Price( 3.00, 15.00,  0.30,  3.75),
    "claude-opus-3":          _Price(15.00, 75.00,  1.50, 18.75),
    "claude-3-opus":          _Price(15.00, 75.00,  1.50, 18.75),
    "claude-3-sonnet":        _Price( 3.00, 15.00,  0.30,  3.75),
    "claude-3-haiku":         _Price( 0.25,  1.25,  0.03,  0.30),
    # ── OpenAI ───────────────────────────────────────────────────────────
    "gpt-4o":                 _Price( 2.50, 10.00,  1.25,  0.00),
    "gpt-4o-mini":            _Price( 0.15,  0.60,  0.08,  0.00),
    "gpt-4-turbo":            _Price(10.00, 30.00,  0.00,  0.00),
    "gpt-4":                  _Price(30.00, 60.00,  0.00,  0.00),
    "gpt-3.5-turbo":          _Price( 0.50,  1.50,  0.00,  0.00),
    "o1":                     _Price(15.00, 60.00,  7.50,  0.00),
    "o1-mini":                _Price( 3.00, 12.00,  1.50,  0.00),
    "o3-mini":                _Price( 1.10,  4.40,  0.55,  0.00),
    # ── Google ───────────────────────────────────────────────────────────
    "gemini-2.0-flash":       _Price( 0.10,  0.40,  0.00,  0.00),
    "gemini-1.5-pro":         _Price( 3.50, 10.50,  0.00,  0.00),
    "gemini-1.5-flash":       _Price( 0.075, 0.30,  0.00,  0.00),
    # ── Groq ─────────────────────────────────────────────────────────────
    "llama3-70b":             _Price( 0.59,  0.79,  0.00,  0.00),
    "llama3-8b":              _Price( 0.05,  0.08,  0.00,  0.00),
    "mixtral-8x7b":           _Price( 0.27,  0.27,  0.00,  0.00),
    "gemma2-9b":              _Price( 0.20,  0.20,  0.00,  0.00),
    # ── Local (free) ─────────────────────────────────────────────────────
    "ollama":                 _Price( 0.00,  0.00),
    "qwen":                   _Price( 0.00,  0.00),
    "llama":                  _Price( 0.00,  0.00),
    "mistral":                _Price( 0.00,  0.00),
    "phi":                    _Price( 0.00,  0.00),
    "gemma":                  _Price( 0.00,  0.00),
    "deepseek":               _Price( 0.00,  0.00),
}


def _lookup_price(model: str, provider: str = "") -> Optional[_Price]:
    """Find the best matching price entry by longest-prefix match."""
    m = (model or "").lower().strip()
    p = (provider or "").lower().strip()

    # Exact match first
    if m in _PRICING:
        return _PRICING[m]

    # Ollama / local — all free
    if p in ("ollama",) or any(m.startswith(k) for k in ("qwen", "llama", "mistral", "phi", "gemma", "deepseek")):
        return _Price(0.0, 0.0)

    # Longest prefix match
    best_key = ""
    for key in _PRICING:
        if m.startswith(key) and len(key) > len(best_key):
            best_key = key

    return _PRICING.get(best_key)


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

def estimate_cost(
    usage:    CanonicalUsage,
    model:    str,
    provider: str = "",
) -> Optional[Decimal]:
    """Return USD cost estimate, or None if the model price is unknown."""
    price = _lookup_price(model, provider)
    if price is None:
        return None
    return price.cost(usage)


def format_cost(amount: Optional[Decimal]) -> str:
    """Format a cost Decimal for display (e.g. '$0.0042' or 'unknown')."""
    if amount is None:
        return "unknown"
    if amount == 0:
        return "$0.00 (local)"
    if amount < Decimal("0.001"):
        return f"${amount:.6f}"
    return f"${amount.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)}"


# ---------------------------------------------------------------------------
# Session usage tracker (singleton)
# ---------------------------------------------------------------------------

@dataclass
class _UsageRecord:
    usage:     CanonicalUsage
    cost_usd:  Optional[Decimal]
    provider:  str
    model:     str
    ts:        float


class UsageTracker:
    """Accumulates per-session token usage and cost across all LLM calls."""

    def __init__(self) -> None:
        self._lock    = Lock()
        self._records: list[_UsageRecord] = []
        self._total   = CanonicalUsage()
        self._total_cost: Decimal = Decimal("0")

    def record(
        self,
        usage:    CanonicalUsage,
        model:    str,
        provider: str = "",
    ) -> Optional[Decimal]:
        """Record a usage event and return the cost for this call."""
        cost = estimate_cost(usage, model, provider)
        with self._lock:
            self._records.append(_UsageRecord(
                usage=usage, cost_usd=cost,
                provider=provider, model=model,
                ts=time.time(),
            ))
            self._total = self._total + usage
            if cost is not None:
                self._total_cost += cost
        return cost

    def totals(self) -> tuple[CanonicalUsage, Decimal]:
        """Return (total_usage, total_cost_usd)."""
        with self._lock:
            return self._total, self._total_cost

    def summary(self) -> str:
        """Format a human-readable cost summary for /cost command."""
        total, cost_usd = self.totals()
        lines = [
            "Token & Cost Summary",
            "",
            f"  Requests:       {total.request_count}",
            f"  Input tokens:   {total.input_tokens:,}",
            f"  Output tokens:  {total.output_tokens:,}",
        ]
        if total.cache_read_tokens:
            pct = total.cache_read_tokens / max(total.prompt_tokens, 1) * 100
            lines.append(f"  Cache reads:    {total.cache_read_tokens:,}  ({pct:.0f}% of prompt)")
        if total.cache_write_tokens:
            lines.append(f"  Cache writes:   {total.cache_write_tokens:,}")
        if total.reasoning_tokens:
            lines.append(f"  Reasoning tok:  {total.reasoning_tokens:,}")
        lines.append(f"  Total tokens:   {total.total_tokens:,}")
        lines.append("")

        # Per-model breakdown
        with self._lock:
            by_model: Dict[str, tuple[CanonicalUsage, Decimal]] = {}
            for r in self._records:
                key = f"{r.provider}/{r.model}"
                prev_u, prev_c = by_model.get(key, (CanonicalUsage(), Decimal("0")))
                by_model[key] = (prev_u + r.usage, prev_c + (r.cost_usd or Decimal("0")))

        if by_model:
            lines.append("  By model:")
            for key, (u, c) in sorted(by_model.items()):
                lines.append(f"    {key:<40}  {u.total_tokens:>8,} tok  {format_cost(c)}")
            lines.append("")

        lines.append(f"  Estimated cost: {format_cost(cost_usd)}")
        if cost_usd == Decimal("0"):
            lines.append("  (local inference — no API cost)")

        return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._total   = CanonicalUsage()
            self._total_cost = Decimal("0")


_tracker: Optional[UsageTracker] = None


def get_usage_tracker() -> UsageTracker:
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker()
    return _tracker
