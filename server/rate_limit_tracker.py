"""
rate_limit_tracker.py — API Rate Limit Header Tracking
========================================================
Adapted from Hermes agent/rate_limit_tracker.py.

Captures x-ratelimit-* response headers from provider API calls and
surfaces them as structured state that the router + kernel can use to
make intelligent backoff and rotation decisions.

Supports the 12-header format used by OpenAI, OpenRouter, Nous Portal,
and most OpenAI-compatible providers.

Integration
-----------
  Kernel captures headers from every streaming response:
    from server.rate_limit_tracker import parse_rate_limit_headers, get_tracker
    state = parse_rate_limit_headers(resp.headers, provider=provider)
    get_tracker().update(provider, state)

  TUI /usage command reads:
    from server.rate_limit_tracker import get_tracker
    print(format_rate_limit_display(get_tracker().latest(provider)))
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Mapping, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RateLimitBucket:
    """One rate-limit window (e.g. requests-per-minute)."""
    limit:         int   = 0
    remaining:     int   = 0
    reset_seconds: float = 0.0
    captured_at:   float = 0.0   # monotonic time when captured

    @property
    def used(self) -> int:
        return max(0, self.limit - self.remaining)

    @property
    def usage_pct(self) -> float:
        return (self.used / self.limit * 100.0) if self.limit > 0 else 0.0

    @property
    def remaining_seconds_now(self) -> float:
        """Remaining seconds until reset, adjusted for elapsed time."""
        elapsed = time.time() - self.captured_at
        return max(0.0, self.reset_seconds - elapsed)


@dataclass
class RateLimitState:
    """Full rate-limit state parsed from one API response's headers."""
    requests_min:  RateLimitBucket = field(default_factory=RateLimitBucket)
    requests_hour: RateLimitBucket = field(default_factory=RateLimitBucket)
    tokens_min:    RateLimitBucket = field(default_factory=RateLimitBucket)
    tokens_hour:   RateLimitBucket = field(default_factory=RateLimitBucket)
    captured_at:   float = 0.0
    provider:      str   = ""

    @property
    def has_data(self) -> bool:
        return self.captured_at > 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.captured_at if self.has_data else float("inf")

    @property
    def is_near_limit(self) -> bool:
        """True if any bucket is >= 80% used — callers should slow down."""
        for bucket in (self.requests_min, self.requests_hour,
                       self.tokens_min, self.tokens_hour):
            if bucket.limit > 0 and bucket.usage_pct >= 80:
                return True
        return False

    @property
    def should_rotate(self) -> bool:
        """True if any minute-window bucket is exhausted (remaining == 0)."""
        for bucket in (self.requests_min, self.tokens_min):
            if bucket.limit > 0 and bucket.remaining == 0:
                return True
        return False


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_rate_limit_headers(
    headers: Mapping[str, str],
    provider: str = "",
) -> Optional[RateLimitState]:
    """Parse x-ratelimit-* response headers into a RateLimitState.

    Returns None if no rate-limit headers are present (e.g. Ollama).
    Header names are normalised to lowercase (HTTP is case-insensitive).
    """
    lowered = {k.lower(): v for k, v in headers.items()}
    if not any(k.startswith("x-ratelimit-") for k in lowered):
        return None

    now = time.time()

    def _bucket(resource: str, suffix: str = "") -> RateLimitBucket:
        tag = f"{resource}{suffix}"
        return RateLimitBucket(
            limit         = _safe_int(lowered.get(f"x-ratelimit-limit-{tag}")),
            remaining     = _safe_int(lowered.get(f"x-ratelimit-remaining-{tag}")),
            reset_seconds = _safe_float(lowered.get(f"x-ratelimit-reset-{tag}")),
            captured_at   = now,
        )

    return RateLimitState(
        requests_min  = _bucket("requests"),
        requests_hour = _bucket("requests", "-1h"),
        tokens_min    = _bucket("tokens"),
        tokens_hour   = _bucket("tokens", "-1h"),
        captured_at   = now,
        provider      = provider,
    )


# ---------------------------------------------------------------------------
# In-process tracker (singleton)
# ---------------------------------------------------------------------------

class RateLimitTracker:
    """Thread-safe store of the latest RateLimitState per provider."""

    def __init__(self) -> None:
        self._states: Dict[str, RateLimitState] = {}
        self._lock   = Lock()

    def update(self, provider: str, state: Optional[RateLimitState]) -> None:
        if state is None:
            return
        with self._lock:
            self._states[provider] = state

    def latest(self, provider: str) -> Optional[RateLimitState]:
        with self._lock:
            return self._states.get(provider)

    def all_states(self) -> Dict[str, RateLimitState]:
        with self._lock:
            return dict(self._states)


_tracker: Optional[RateLimitTracker] = None


def get_tracker() -> RateLimitTracker:
    global _tracker
    if _tracker is None:
        _tracker = RateLimitTracker()
    return _tracker


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_seconds(secs: float) -> str:
    s = max(0, int(secs))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{m}m {sec}s" if sec else f"{m}m"
    h, rem = divmod(s, 3600)
    m = rem // 60
    return f"{h}h {m}m" if m else f"{h}h"


def _bar(pct: float, width: int = 18) -> str:
    filled = max(0, min(width, int(pct / 100.0 * width)))
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def _bucket_line(label: str, bucket: RateLimitBucket) -> str:
    if bucket.limit <= 0:
        return f"  {label:<16}  (no data)"
    pct   = bucket.usage_pct
    reset = _fmt_seconds(bucket.remaining_seconds_now)
    bar   = _bar(pct)
    return (
        f"  {label:<16} {bar} {pct:5.1f}%  "
        f"{_fmt_count(bucket.used)}/{_fmt_count(bucket.limit)} used  "
        f"({_fmt_count(bucket.remaining)} left, resets {reset})"
    )


def format_rate_limit_display(state: Optional[RateLimitState]) -> str:
    """Full multi-line rate limit display for /usage command."""
    if state is None or not state.has_data:
        return "No rate limit data — make an API request first."

    age = state.age_seconds
    freshness = ("just now" if age < 5
                 else f"{int(age)}s ago" if age < 60
                 else f"{_fmt_seconds(age)} ago")

    provider_label = (state.provider or "Provider").title()
    lines = [
        f"{provider_label} Rate Limits  (captured {freshness})",
        "",
        _bucket_line("Requests/min",  state.requests_min),
        _bucket_line("Requests/hr",   state.requests_hour),
        "",
        _bucket_line("Tokens/min",    state.tokens_min),
        _bucket_line("Tokens/hr",     state.tokens_hour),
    ]

    warnings = []
    for label, bucket in [
        ("requests/min",  state.requests_min),
        ("requests/hr",   state.requests_hour),
        ("tokens/min",    state.tokens_min),
        ("tokens/hr",     state.tokens_hour),
    ]:
        if bucket.limit > 0 and bucket.usage_pct >= 80:
            warnings.append(
                f"  ⚠  {label} at {bucket.usage_pct:.0f}% — resets in {_fmt_seconds(bucket.remaining_seconds_now)}"
            )

    if warnings:
        lines += [""] + warnings

    return "\n".join(lines)


def format_rate_limit_compact(state: Optional[RateLimitState]) -> str:
    """One-line compact summary for status bars."""
    if state is None or not state.has_data:
        return "no rate data"
    parts = []
    if state.requests_min.limit > 0:
        parts.append(f"RPM {state.requests_min.remaining}/{state.requests_min.limit}")
    if state.tokens_min.limit > 0:
        parts.append(f"TPM {_fmt_count(state.tokens_min.remaining)}/{_fmt_count(state.tokens_min.limit)}")
    if state.tokens_hour.limit > 0:
        parts.append(f"TPH {_fmt_count(state.tokens_hour.remaining)}/{_fmt_count(state.tokens_hour.limit)}")
    return " | ".join(parts) or "no rate data"
