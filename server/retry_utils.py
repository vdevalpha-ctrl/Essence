"""
retry_utils.py — Jittered Exponential Backoff
===============================================
Adapted from Hermes agent/retry_utils.py.

Replaces fixed exponential backoff with decorrelated jittered delays to
prevent thundering-herd retry spikes when multiple requests hit the same
rate-limited provider concurrently.

Used by kernel._stream_response() and the model router fallback loop.
"""
from __future__ import annotations

import random
import threading
import time

# Monotonic counter — gives each backoff call a unique seed even when
# multiple tasks fire simultaneously on the same coarse-grained clock.
_jitter_counter = 0
_jitter_lock    = threading.Lock()


def jittered_backoff(
    attempt: int,
    *,
    base_delay: float = 5.0,
    max_delay:  float = 120.0,
    jitter_ratio: float = 0.5,
) -> float:
    """Compute a jittered exponential backoff delay in seconds.

    Args:
        attempt:      1-based retry attempt number.
        base_delay:   Base delay for attempt 1 (seconds).
        max_delay:    Hard cap on computed delay (seconds).
        jitter_ratio: Fraction of the capped delay used as jitter range.
                      0.5 → jitter uniform in [0, 0.5 × delay].

    Returns:
        Delay in seconds = min(base × 2^(attempt-1), max) + jitter.

    Jitter decorrelates concurrent retries from separate kernel tasks /
    TUI sessions so they don't all hammer the provider at the same instant.
    """
    global _jitter_counter
    with _jitter_lock:
        _jitter_counter += 1
        tick = _jitter_counter

    exponent = max(0, attempt - 1)
    delay    = max_delay if exponent >= 63 or base_delay <= 0 else min(base_delay * (2 ** exponent), max_delay)

    # Decorrelated seed: wall-clock nanoseconds XOR monotonic counter
    seed = (time.time_ns() ^ (tick * 0x9E3779B9)) & 0xFFFFFFFF
    rng  = random.Random(seed)
    return delay + rng.uniform(0, jitter_ratio * delay)


def retry_delays(
    max_attempts: int   = 4,
    base_delay:   float = 3.0,
    max_delay:    float = 60.0,
) -> list[float]:
    """Return a list of ``max_attempts`` jittered delay values.

    Convenience wrapper for building a retry schedule up-front.
    """
    return [jittered_backoff(i + 1, base_delay=base_delay, max_delay=max_delay)
            for i in range(max_attempts)]
