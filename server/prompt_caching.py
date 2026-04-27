"""
prompt_caching.py — Anthropic Prompt Cache Control
====================================================
Adapted from Hermes agent/prompt_caching.py.

Reduces input token costs by ~75% on multi-turn conversations with
Anthropic models by injecting cache_control breakpoints into messages.

Strategy: system_and_3
  • System prompt                    → breakpoint 1 (always stable)
  • Last 3 non-system messages       → breakpoints 2-4 (rolling window)
  This uses all 4 breakpoints Anthropic supports.

Wire-up
-------
  Called from kernel._stream_response() when provider == "anthropic" (or
  any provider whose base_url matches api.anthropic.com).

Pure functions — no class state, no side effects.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List


def _apply_cache_marker(msg: dict, marker: dict) -> None:
    """Inject a cache_control marker into a single message.

    Handles all content formats:
      - plain string content      → convert to list with text block
      - multimodal list content   → append marker to the last block
      - tool role (no content)    → attach directly to message dict
      - empty content             → attach to message dict
    """
    role = msg.get("role", "")
    content = msg.get("content")

    # Tool results: Anthropic accepts cache_control on the message dict
    if role == "tool":
        msg["cache_control"] = marker
        return

    if content is None or content == "":
        msg["cache_control"] = marker
        return

    if isinstance(content, str):
        # Convert to content-block list with the marker on the text block
        msg["content"] = [{"type": "text", "text": content, "cache_control": marker}]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = marker
        return

    # Unknown format — fall back to message-level marker
    msg["cache_control"] = marker


def apply_anthropic_cache_control(
    messages: List[Dict[str, Any]],
    ttl: str = "5m",        # "5m" (ephemeral) | "1h"
) -> List[Dict[str, Any]]:
    """Apply system_and_3 caching strategy for Anthropic API calls.

    Returns a deep copy of *messages* with up to 4 cache_control breakpoints:
      1. System prompt (index 0 if role == "system")
      2-4. Last 3 non-system messages

    Args:
        messages: Full message list as passed to normalize_to_openai().
        ttl:      "5m" for ephemeral (default) or "1h" for 1-hour TTL.
                  1-hour TTL costs 25% more on cache writes but saves on
                  long-lived sessions that restart within an hour.

    The deep copy ensures the original message list is never mutated —
    callers can safely pass the same list to multiple providers.
    """
    if not messages:
        return messages

    msgs = copy.deepcopy(messages)
    marker: Dict[str, Any] = {"type": "ephemeral"}
    if ttl == "1h":
        marker["ttl"] = "1h"

    breakpoints_used = 0

    # Breakpoint 1: system prompt
    if msgs[0].get("role") == "system":
        _apply_cache_marker(msgs[0], marker)
        breakpoints_used += 1

    # Breakpoints 2-4: last 3 non-system messages
    remaining = 4 - breakpoints_used
    non_sys_indices = [i for i, m in enumerate(msgs) if m.get("role") != "system"]
    for idx in non_sys_indices[-remaining:]:
        _apply_cache_marker(msgs[idx], marker)

    return msgs


def should_apply_cache_control(provider: str, base_url: str = "") -> bool:
    """Return True if this provider supports Anthropic prompt caching."""
    if provider == "anthropic":
        return True
    if "anthropic.com" in base_url:
        return True
    # AWS Bedrock Anthropic
    if "bedrock" in base_url and "anthropic" in base_url.lower():
        return True
    return False
