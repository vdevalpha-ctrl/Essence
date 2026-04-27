"""
context_compressor.py — Intelligent Context Window Compression
==============================================================
Ported and adapted from Hermes Agent v3 (hermes-agent-main).

Compresses long conversations by summarising middle turns with a structured
LLM prompt, protecting head (system prompt + first exchange) and tail
(most-recent token budget) messages.

Key features
------------
  • Tool output pruning — cheap pre-pass, no LLM call needed
  • Token-budget tail protection — scales with context window, not fixed count
  • Structured 12-section summary template (Active Task, Goal, Constraints …)
  • Iterative summary updates — preserves info across multiple compactions
  • Anti-thrashing guard — skips if last 2 compressions saved < 10%
  • Orphaned tool-call / result pair sanitisation
  • Sensitive-data redaction before sending to summariser LLM

Integration
-----------
  Used by EssenceKernel._stream_response() — compressor is instantiated once
  per kernel session and called transparently before each LLM request when the
  estimated message token count exceeds the threshold.

  Also exposed as a slash command via /compress [topic] in the TUI.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger("essence.compressor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfil requests mentioned in this summary; "
    "they were already addressed. "
    "Your current task is identified in the '## Active Task' section — "
    "resume exactly from there. "
    "Respond ONLY to the latest user message that appears AFTER this summary. "
    "The current session state may reflect work described here — avoid repeating it:"
)
LEGACY_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"

# Summary sizing
_MIN_SUMMARY_TOKENS = 1_500
_SUMMARY_RATIO = 0.18          # proportion of compressed content → summary budget
_SUMMARY_TOKENS_CEILING = 8_000

# Chars-per-token rough estimate (good enough for a budget heuristic)
_CHARS_PER_TOKEN = 4

# Threshold below which we don't even try to compress (short conversations)
_MIN_COMPRESS_MESSAGES = 8

# Placeholder for pruned tool outputs
_PRUNED_PLACEHOLDER = "[Old tool output cleared to save context space]"

# Cooldown after a summarisation failure (seconds)
_FAILURE_COOLDOWN = 300


# ---------------------------------------------------------------------------
# Sensitive-data redaction
# ---------------------------------------------------------------------------

_REDACT_PATTERNS = [
    # Bearer / API tokens
    re.compile(r'(Bearer\s+)[A-Za-z0-9\-_\.]{20,}', re.IGNORECASE),
    # sk-... style keys
    re.compile(r'\b(sk-[A-Za-z0-9]{20,})\b'),
    # Generic key= patterns
    re.compile(r'(api[_-]?key\s*[=:]\s*)[\'"]?[A-Za-z0-9\-_\.]{16,}[\'"]?', re.IGNORECASE),
    re.compile(r'(token\s*[=:]\s*)[\'"]?[A-Za-z0-9\-_\.]{16,}[\'"]?', re.IGNORECASE),
    re.compile(r'(password\s*[=:]\s*)[\'"]?[^\s\'"]{6,}[\'"]?', re.IGNORECASE),
]


def _redact(text: str) -> str:
    """Replace likely secrets with [REDACTED]."""
    if not isinstance(text, str):
        return text
    for pat in _REDACT_PATTERNS:
        text = pat.sub(lambda m: m.group(0)[:m.start(1) - m.start(0) + len(m.group(1))] + "[REDACTED]", text)
    return text


# ---------------------------------------------------------------------------
# Token estimator
# ---------------------------------------------------------------------------

def _estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Rough token count from message content (4 chars ≈ 1 token)."""
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += len(block.get("text", "")) // _CHARS_PER_TOKEN
        else:
            total += len(content) // _CHARS_PER_TOKEN
        # Include tool call arguments
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                args = tc.get("function", {}).get("arguments", "")
                total += len(args) // _CHARS_PER_TOKEN
        total += 10  # per-message overhead
    return total


# ---------------------------------------------------------------------------
# Tool-result summariser (no LLM — regex-based)
# ---------------------------------------------------------------------------

def _summarise_tool_result(tool_name: str, tool_args: str, tool_content: str) -> str:
    """Return an informative 1-line summary of a tool call + result.

    Examples::
        [shell] ran `npm test` -> exit 0, 47 lines
        [read_file] read config.py (1,200 chars)
        [http_get] GET https://example.com/api (3,400 chars)
    """
    try:
        args = json.loads(tool_args) if tool_args else {}
    except (json.JSONDecodeError, TypeError):
        args = {}

    content = tool_content or ""
    content_len = len(content)
    line_count = content.count("\n") + 1 if content.strip() else 0

    if tool_name == "shell":
        cmd = args.get("command", "")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        exit_match = re.search(r'exit\s+(\d+)', content, re.IGNORECASE)
        exit_code = exit_match.group(1) if exit_match else "?"
        return f"[shell] ran `{cmd}` -> exit {exit_code}, {line_count} lines"

    if tool_name == "read_file":
        path = args.get("path", "?")
        return f"[read_file] read {path} ({content_len:,} chars)"

    if tool_name == "write_file":
        path = args.get("path", "?")
        lines = (args.get("content") or "").count("\n") + 1
        return f"[write_file] wrote {path} ({lines} lines)"

    if tool_name == "list_dir":
        path = args.get("path", ".")
        return f"[list_dir] listed {path} ({line_count} entries)"

    if tool_name == "http_get":
        url = args.get("url", "?")
        if len(url) > 80:
            url = url[:77] + "..."
        return f"[http_get] GET {url} ({content_len:,} chars)"

    if tool_name == "remember":
        fact = (args.get("fact") or "")[:60]
        return f"[remember] stored: {fact}"

    if tool_name == "search_memory":
        query = args.get("query", "?")
        return f"[search_memory] '{query}' ({line_count} results)"

    if tool_name == "create_task":
        title = args.get("title", "?")
        return f"[create_task] '{title}'"

    if tool_name.startswith("svc_"):
        return f"[{tool_name}] ({content_len:,} chars)"

    # Generic fallback
    first_arg = ""
    for k, v in list(args.items())[:2]:
        first_arg += f" {k}={str(v)[:40]}"
    return f"[{tool_name}]{first_arg} ({content_len:,} chars)"


# ---------------------------------------------------------------------------
# Tool-call argument truncator (JSON-safe)
# ---------------------------------------------------------------------------

def _truncate_tool_args(args_json: str, max_chars: int = 200) -> str:
    """Shrink long string values inside a tool-call arguments JSON blob.

    Parses the JSON, truncates long strings, re-serialises. If args_json
    is not valid JSON, returns it unchanged (some providers use non-JSON args).
    """
    try:
        parsed = json.loads(args_json)
    except (ValueError, TypeError):
        return args_json

    def _shrink(obj: Any) -> Any:
        if isinstance(obj, str) and len(obj) > max_chars:
            return obj[:max_chars] + "...[truncated]"
        if isinstance(obj, dict):
            return {k: _shrink(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_shrink(v) for v in obj]
        return obj

    return json.dumps(_shrink(parsed), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Content helpers for multimodal-safe manipulation
# ---------------------------------------------------------------------------

def _content_text(content: Any) -> str:
    """Return plain-text view of message content (for substring checks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return str(content)


def _prepend_to_content(content: Any, prefix: str) -> Any:
    """Prepend text to message content (string or multimodal list)."""
    if content is None:
        return prefix
    if isinstance(content, str):
        return prefix + content
    if isinstance(content, list):
        return [{"type": "text", "text": prefix}, *content]
    return prefix + str(content)


def _append_to_content(content: Any, suffix: str) -> Any:
    """Append text to message content."""
    if content is None:
        return suffix
    if isinstance(content, str):
        return content + suffix
    if isinstance(content, list):
        return [*content, {"type": "text", "text": suffix}]
    return str(content) + suffix


# ---------------------------------------------------------------------------
# ContextCompressor
# ---------------------------------------------------------------------------

class ContextCompressor:
    """Compress long conversation history via structured LLM summarisation.

    Algorithm (per compress() call)
    --------------------------------
    1. Prune old tool results  (cheap — no LLM, informative 1-line summaries)
    2. Protect head messages   (system prompt + first N exchanges)
    3. Find tail boundary      (token-budget walk-back from most recent message)
    4. Summarise middle turns  (structured 12-section template)
    5. Assemble compressed list and sanitise orphaned tool pairs
    6. On re-compression, iteratively update previous summary
    """

    def __init__(
        self,
        ollama_url: str,
        model: str,
        *,
        context_tokens: int = 32_768,
        threshold_pct: float = 0.55,
        protect_first_n: int = 3,
        tail_pct: float = 0.25,
        quiet: bool = False,
    ) -> None:
        """
        Args:
            ollama_url:     Ollama base URL (e.g. http://localhost:11434)
            model:          Model name to use for summarisation
            context_tokens: Approximate context window in tokens (default 32K)
            threshold_pct:  Compress when estimated tokens exceed this % of context
            protect_first_n: Number of head messages always kept verbatim
            tail_pct:       Proportion of context_tokens to protect as tail
            quiet:          Suppress info-level log messages
        """
        self._ollama = ollama_url
        self._model = model
        self._context_tokens = context_tokens
        self._threshold_pct = threshold_pct
        self._protect_first_n = protect_first_n
        self._tail_budget = int(context_tokens * tail_pct)
        self._quiet = quiet

        self._threshold = max(int(context_tokens * threshold_pct), 4_000)
        self._max_summary_tokens = min(
            int(context_tokens * 0.05), _SUMMARY_TOKENS_CEILING
        )

        # Per-session state
        self.compression_count: int = 0
        self._previous_summary: Optional[str] = None
        self._last_savings_pct: float = 100.0
        self._ineffective_count: int = 0
        self._cooldown_until: float = 0.0

    # ── Public API ────────────────────────────────────────────────────

    def should_compress(self, messages: List[Dict[str, Any]]) -> bool:
        """Return True if messages exceed threshold and compression would help."""
        tokens = _estimate_tokens(messages)
        if tokens < self._threshold:
            return False
        if self._ineffective_count >= 2:
            if not self._quiet:
                log.warning(
                    "Compression skipped — last %d passes each saved <10%%. "
                    "Use /compress <topic> for forced focused compression.",
                    self._ineffective_count,
                )
            return False
        return True

    def reset_session(self) -> None:
        """Reset per-session state (call on /new or session restart)."""
        self._previous_summary = None
        self._last_savings_pct = 100.0
        self._ineffective_count = 0
        self._cooldown_until = 0.0
        self.compression_count = 0

    def compress(
        self,
        messages: List[Dict[str, Any]],
        focus_topic: str = "",
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Compress conversation messages. Returns the compressed list.

        Args:
            messages:    Full message list (system + history + current turn)
            focus_topic: If set, the summariser prioritises this topic
            force:       Compress even if threshold not reached (for /compress cmd)
        """
        n = len(messages)
        if n < _MIN_COMPRESS_MESSAGES and not force:
            return messages

        tokens_before = _estimate_tokens(messages)

        # Phase 1: Tool result pruning (no LLM)
        messages, pruned = self._prune_tool_results(messages)
        if pruned and not self._quiet:
            log.info("Pre-compression pruned %d tool result(s)", pruned)

        # Phase 2: Determine boundaries
        head_end = min(self._protect_first_n, n - 1)
        head_end = self._advance_past_tool_results(messages, head_end)
        tail_start = self._find_tail_start(messages, head_end)

        if head_end >= tail_start:
            return messages

        turns_to_summarise = messages[head_end:tail_start]

        if not self._quiet:
            log.info(
                "Compressing turns %d–%d (%d msgs), protecting %d head + %d tail",
                head_end + 1, tail_start, len(turns_to_summarise),
                head_end, n - tail_start,
            )

        # Phase 3: LLM summary
        summary = self._generate_summary(turns_to_summarise, focus_topic)

        # Phase 4: Assemble
        compressed = self._assemble(messages, head_end, tail_start, summary, n)

        # Phase 5: Sanitise orphaned tool pairs
        compressed = self._sanitise_tool_pairs(compressed)

        self.compression_count += 1
        tokens_after = _estimate_tokens(compressed)
        saved = tokens_before - tokens_after
        pct = saved / tokens_before * 100 if tokens_before else 0
        self._last_savings_pct = pct
        if pct < 10:
            self._ineffective_count += 1
        else:
            self._ineffective_count = 0

        if not self._quiet:
            log.info(
                "Compression #%d: %d → %d msgs (~%d tokens saved, %.0f%%)",
                self.compression_count, n, len(compressed), saved, pct,
            )

        return compressed

    # ── Tool result pruning ───────────────────────────────────────────

    def _prune_tool_results(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], int]:
        """Three-pass pruning: deduplicate → summarise → truncate args."""
        result = [m.copy() for m in messages]
        pruned = 0

        # Build call_id → (name, args) index from assistant messages
        call_index: Dict[str, tuple] = {}
        for msg in result:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        cid = tc.get("id", "")
                        fn = tc.get("function", {})
                        call_index[cid] = (fn.get("name", "unknown"), fn.get("arguments", ""))

        # Tail boundary — protect recent messages within tail_budget tokens
        tail_start = self._find_tail_start(result, 0)

        # Pass 1: Deduplicate identical tool results (keep newest copy)
        seen_hashes: Dict[str, int] = {}  # hash → newest index
        for i in range(len(result) - 1, -1, -1):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content") or ""
            if isinstance(content, list) or len(content) < 200:
                continue
            h = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()[:12]
            if h in seen_hashes:
                result[i] = {**msg, "content": "[Duplicate — see more-recent identical result]"}
                pruned += 1
            else:
                seen_hashes[h] = i

        # Pass 2: Summarise old tool results (outside protected tail)
        for i in range(tail_start):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content") or ""
            if isinstance(content, list):
                continue
            if not content or content == _PRUNED_PLACEHOLDER:
                continue
            if content.startswith("[Duplicate"):
                continue
            if len(content) > 200:
                cid = msg.get("tool_call_id", "")
                name, args = call_index.get(cid, ("unknown", ""))
                result[i] = {**msg, "content": _summarise_tool_result(name, args, content)}
                pruned += 1

        # Pass 3: Truncate large tool_call args in old assistant messages
        for i in range(tail_start):
            msg = result[i]
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            new_tcs = []
            changed = False
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict):
                    args = tc.get("function", {}).get("arguments", "")
                    if len(args) > 500:
                        new_args = _truncate_tool_args(args)
                        if new_args != args:
                            tc = {**tc, "function": {**tc["function"], "arguments": new_args}}
                            changed = True
                new_tcs.append(tc)
            if changed:
                result[i] = {**msg, "tool_calls": new_tcs}

        return result, pruned

    # ── Boundary helpers ──────────────────────────────────────────────

    @staticmethod
    def _advance_past_tool_results(messages: List[Dict[str, Any]], idx: int) -> int:
        """Slide idx forward past any tool-result messages (avoid starting mid-group)."""
        while idx < len(messages) and messages[idx].get("role") == "tool":
            idx += 1
        return idx

    @staticmethod
    def _retreat_before_tool_group(messages: List[Dict[str, Any]], idx: int) -> int:
        """Walk idx backward to avoid splitting an assistant + tool-results group."""
        if idx <= 0 or idx >= len(messages):
            return idx
        check = idx - 1
        while check >= 0 and messages[check].get("role") == "tool":
            check -= 1
        if (check >= 0
                and messages[check].get("role") == "assistant"
                and messages[check].get("tool_calls")):
            idx = check
        return idx

    def _find_tail_start(
        self, messages: List[Dict[str, Any]], head_end: int
    ) -> int:
        """Walk backward accumulating token budget to find tail start index."""
        n = len(messages)
        min_tail = min(3, max(0, n - head_end - 1))
        soft_ceiling = int(self._tail_budget * 1.5)
        accumulated = 0
        cut = n

        for i in range(n - 1, head_end - 1, -1):
            msg = messages[i]
            content = msg.get("content") or ""
            msg_tok = len(content) // _CHARS_PER_TOKEN + 10
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    msg_tok += len(tc.get("function", {}).get("arguments", "")) // _CHARS_PER_TOKEN
            if accumulated + msg_tok > soft_ceiling and (n - i) >= min_tail:
                break
            accumulated += msg_tok
            cut = i

        # Hard minimum
        fallback = n - min_tail
        if cut > fallback:
            cut = fallback
        if cut <= head_end:
            cut = max(fallback, head_end + 1)

        # Align backward to avoid splitting tool groups
        cut = self._retreat_before_tool_group(messages, cut)

        # Ensure the most-recent user message is always in the tail
        # (fixes silent active-task loss after compression)
        last_user = -1
        for i in range(n - 1, head_end - 1, -1):
            if messages[i].get("role") == "user":
                last_user = i
                break
        if 0 <= last_user < cut:
            cut = max(last_user, head_end + 1)

        return max(cut, head_end + 1)

    # ── Summarisation ─────────────────────────────────────────────────

    def _serialise_for_summary(self, turns: List[Dict[str, Any]]) -> str:
        """Render turns as labelled text for the summariser LLM."""
        MAX_MSG = 5_000
        HEAD = 3_500
        TAIL = 1_000
        parts = []
        for msg in turns:
            role = msg.get("role", "unknown")
            raw = _redact(_content_text(msg.get("content") or ""))
            if len(raw) > MAX_MSG:
                raw = raw[:HEAD] + "\n...[truncated]...\n" + raw[-TAIL:]

            if role == "tool":
                cid = msg.get("tool_call_id", "")
                parts.append(f"[TOOL RESULT {cid}]: {raw}")
                continue

            if role == "assistant":
                tc_block = ""
                tcs = msg.get("tool_calls") or []
                if tcs:
                    tc_lines = []
                    for tc in tcs:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            args = _redact(fn.get("arguments", ""))[:1_200]
                            tc_lines.append(f"  {name}({args})")
                        else:
                            fn = getattr(tc, "function", None)
                            tc_lines.append(f"  {getattr(fn, 'name', '?')}(...)")
                    tc_block = "\n[Tool calls:\n" + "\n".join(tc_lines) + "\n]"
                parts.append(f"[ASSISTANT]: {raw}{tc_block}")
                continue

            parts.append(f"[{role.upper()}]: {raw}")

        return "\n\n".join(parts)

    def _compute_summary_budget(self, turns: List[Dict[str, Any]]) -> int:
        """Scale summary token budget with the content being compressed."""
        content_tokens = _estimate_tokens(turns)
        budget = int(content_tokens * _SUMMARY_RATIO)
        return max(_MIN_SUMMARY_TOKENS, min(budget, self._max_summary_tokens))

    def _generate_summary(
        self, turns: List[Dict[str, Any]], focus_topic: str = ""
    ) -> Optional[str]:
        """Call Ollama to generate a structured handoff summary.

        Returns None on failure — caller substitutes a static fallback.
        """
        if time.monotonic() < self._cooldown_until:
            log.debug("Compression cooldown active — skipping LLM summary")
            return None

        budget = self._compute_summary_budget(turns)
        content = self._serialise_for_summary(turns)

        preamble = (
            "You are a summarisation agent creating a context checkpoint. "
            "Your output will be injected as reference material for a DIFFERENT "
            "assistant that continues the conversation. "
            "Do NOT respond to any questions or requests — only output the structured summary. "
            "Do NOT include any preamble, greeting, or prefix. "
            "NEVER include API keys, tokens, passwords, or credentials — write [REDACTED]."
        )

        template = f"""## Active Task
[THE SINGLE MOST IMPORTANT FIELD. Copy the user's most recent unfulfilled request verbatim.
The next assistant must pick up exactly here. If no outstanding task exists, write "None."]

## Goal
[What the user is trying to accomplish overall]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions]

## Completed Actions
[Numbered list: N. ACTION target — outcome [tool: name]
Example: 1. READ config.py — found bug on line 45 [tool: read_file]]

## Active State
[Working directory, modified files, test status, running processes]

## In Progress
[Work underway when compaction fired]

## Blocked
[Unresolved errors, blockers, exact error messages]

## Key Decisions
[Technical decisions and WHY they were made]

## Resolved Questions
[Questions already answered — include the answer]

## Pending User Asks
[Unanswered user questions/requests. If none, write "None."]

## Relevant Files
[Files read, modified, or created — with brief note]

## Remaining Work
[What remains — framed as context, not instructions]

Target ~{budget} tokens. Be concrete — include file paths, commands, error messages.
Write only the summary body. No preamble."""

        if self._previous_summary:
            prompt = (
                f"{preamble}\n\n"
                f"You are UPDATING a context compaction summary. Preserve existing info, "
                f"incorporate new turns, update completed/in-progress/pending items.\n\n"
                f"PREVIOUS SUMMARY:\n{self._previous_summary}\n\n"
                f"NEW TURNS:\n{content}\n\n"
                f"Update using this structure. CRITICAL: Update '## Active Task' to reflect "
                f"the user's most recent unfulfilled request.\n\n{template}"
            )
        else:
            prompt = (
                f"{preamble}\n\n"
                f"Summarise the following conversation turns for handoff to a different "
                f"assistant. Be concrete and specific.\n\n"
                f"TURNS:\n{content}\n\n{template}"
            )

        if focus_topic:
            prompt += (
                f"\n\nFOCUS TOPIC: \"{focus_topic}\"\n"
                f"Prioritise preserving all information about \"{focus_topic}\" "
                f"(~60-70% of budget). Summarise everything else aggressively."
            )

        try:
            import asyncio

            async def _call() -> str:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    r = await client.post(
                        f"{self._ollama}/api/chat",
                        json={
                            "model": self._model,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "options": {"temperature": 0.1, "num_predict": int(budget * 1.4)},
                        },
                    )
                    r.raise_for_status()
                    data = r.json()
                    return data.get("message", {}).get("content", "").strip()

            # Run inside the existing event loop (we're always called from async context)
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(_call(), loop)
                # Give up to 90s for the summary
                raw = future.result(timeout=90)
            else:
                raw = loop.run_until_complete(_call())

            if not raw:
                return None

            summary = _redact(raw)
            self._previous_summary = summary
            self._cooldown_until = 0.0
            return self._with_prefix(summary)

        except Exception as exc:
            log.warning("Compression summary failed: %s", exc)
            self._cooldown_until = time.monotonic() + _FAILURE_COOLDOWN
            return None

    @staticmethod
    def _with_prefix(summary: str) -> str:
        """Wrap summary in the standard handoff prefix."""
        text = (summary or "").strip()
        for pfx in (LEGACY_SUMMARY_PREFIX, SUMMARY_PREFIX):
            if text.startswith(pfx):
                text = text[len(pfx):].lstrip()
                break
        return f"{SUMMARY_PREFIX}\n{text}" if text else SUMMARY_PREFIX

    # ── Assembly ──────────────────────────────────────────────────────

    def _assemble(
        self,
        messages: List[Dict[str, Any]],
        head_end: int,
        tail_start: int,
        summary: Optional[str],
        orig_n: int,
    ) -> List[Dict[str, Any]]:
        """Build compressed message list from head + summary + tail."""
        if not summary:
            n_dropped = tail_start - head_end
            summary = (
                f"{SUMMARY_PREFIX}\n"
                f"Summary unavailable. {n_dropped} turns were removed to free context space. "
                f"Continue based on recent messages and current file state."
            )

        # Add a note to the system message (if present)
        result: List[Dict[str, Any]] = []
        note = (
            "[Note: Earlier conversation turns were compacted into a handoff summary "
            "to preserve context space. Build on that summary and current state.]"
        )
        for i in range(head_end):
            msg = messages[i].copy()
            if i == 0 and msg.get("role") == "system":
                existing = msg.get("content", "")
                if note not in _content_text(existing):
                    msg["content"] = _append_to_content(existing, "\n\n" + note)
            result.append(msg)

        # Pick a role for the summary message that avoids consecutive same-role
        last_head_role = messages[head_end - 1].get("role", "user") if head_end > 0 else "user"
        first_tail_role = messages[tail_start].get("role", "user") if tail_start < orig_n else "user"

        merge_into_tail = False
        if last_head_role in ("assistant", "tool"):
            summary_role = "user"
        else:
            summary_role = "assistant"
        if summary_role == first_tail_role:
            flipped = "assistant" if summary_role == "user" else "user"
            if flipped != last_head_role:
                summary_role = flipped
            else:
                merge_into_tail = True

        if not merge_into_tail:
            result.append({"role": summary_role, "content": summary})

        for i in range(tail_start, orig_n):
            msg = messages[i].copy()
            if merge_into_tail and i == tail_start:
                merged_prefix = (
                    summary
                    + "\n\n--- END OF CONTEXT SUMMARY — "
                    "respond to the message below, not the summary above ---\n\n"
                )
                msg["content"] = _prepend_to_content(msg.get("content"), merged_prefix)
                merge_into_tail = False
            result.append(msg)

        return result

    # ── Tool-pair sanitisation ────────────────────────────────────────

    def _sanitise_tool_pairs(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove orphaned tool results; inject stub results for orphaned calls."""
        surviving_calls: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                    if cid:
                        surviving_calls.add(cid)

        result_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_ids.add(cid)

        orphaned_results = result_ids - surviving_calls
        if orphaned_results:
            messages = [
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]
            if not self._quiet:
                log.info("Sanitiser: removed %d orphaned tool result(s)", len(orphaned_results))

        missing_results = surviving_calls - result_ids
        if missing_results:
            patched: List[Dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        cid = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                        if cid in missing_results:
                            patched.append({
                                "role": "tool",
                                "content": "[Result from earlier session — see context summary above]",
                                "tool_call_id": cid,
                            })
            messages = patched
            if not self._quiet:
                log.info("Sanitiser: injected %d stub tool result(s)", len(missing_results))

        return messages


# ---------------------------------------------------------------------------
# Module-level singleton (one per Essence session)
# ---------------------------------------------------------------------------

_compressor: Optional[ContextCompressor] = None


def init_compressor(
    ollama_url: str,
    model: str,
    *,
    context_tokens: int = 32_768,
    threshold_pct: float = 0.55,
) -> ContextCompressor:
    global _compressor
    _compressor = ContextCompressor(
        ollama_url=ollama_url,
        model=model,
        context_tokens=context_tokens,
        threshold_pct=threshold_pct,
    )
    return _compressor


def get_compressor() -> Optional[ContextCompressor]:
    return _compressor
