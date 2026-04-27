"""
tool_bridge.py — Provider-Agnostic Tool Call Layer
====================================================
Translates between Essence's canonical (OpenAI-style) tool format and
each provider's native wire format, both for request serialisation and
response parsing.

Problem it solves
-----------------
model_router.normalize_to_openai() previously injected tools into the
request body in raw OpenAI format regardless of the target provider.
Anthropic's Messages API has a fundamentally different tool schema, tool
result structure, and streaming protocol — blindly forwarding OpenAI
JSON to Anthropic's endpoint causes API errors or silent misfires.

This module is the single place that knows about each provider's quirks.
The rest of Essence (kernel, model_router) speaks only canonical format.

Canonical (OpenAI) tool definition
-----------------------------------
    {
        "type": "function",
        "function": {
            "name": "tool_name",
            "description": "What it does",
            "parameters": {
                "type": "object",
                "properties": {"arg": {"type": "string", "description": "..."}},
                "required": ["arg"]
            }
        }
    }

Canonical ToolCall (parsed result)
------------------------------------
    ToolCall(id="call_xxx", name="tool_name", arguments={"arg": "value"})

Canonical tool result message (injected back into conversation)
---------------------------------------------------------------
    The bridge produces provider-correct message dicts via
    make_tool_result_messages(); callers append them to messages[] as-is.

Provider matrix
---------------
  openai      → tools array, tool role messages           (reference format)
  anthropic   → anthropic tools array + tool_result       (fully different)
  groq        → openai-compatible                          (pass-through)
  mistral     → openai-compatible                          (pass-through)
  together    → openai-compatible                          (pass-through)
  deepseek    → openai-compatible                          (pass-through)
  gemini      → openai-compatible (via their /openai URL)  (pass-through)
  ollama      → openai-compatible (v0.1.24+)               (pass-through)
  openrouter  → openai-compatible                          (pass-through)
  lmstudio    → openai-compatible                          (pass-through)
  hf_local    → no native tool support → system-prompt fallback
  llamacpp    → no native tool support → system-prompt fallback

Streaming tool-call accumulation
---------------------------------
  StreamAccumulator tracks partial tool-call deltas across SSE chunks.
  Call accumulator.feed(chunk) on each chunk; it returns List[ToolCall]
  the moment a complete set of tool calls has been received, else None.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("essence.tool_bridge")

# ---------------------------------------------------------------------------
# Provider classification
# ---------------------------------------------------------------------------

# Providers that accept native tool calls (OpenAI-compatible wire format)
_OPENAI_TOOL_PROVIDERS = frozenset({
    "openai", "groq", "mistral", "together", "deepseek",
    "gemini", "ollama", "openrouter", "lmstudio",
})

# Providers with Anthropic's native tool format
_ANTHROPIC_TOOL_PROVIDERS = frozenset({"anthropic"})

# Providers with no native tool support — fall back to system-prompt JSON
_NO_NATIVE_TOOLS = frozenset({"hf_local", "llamacpp"})


def supports_native_tools(provider: str) -> bool:
    """Return True if the provider supports native function/tool calling."""
    p = (provider or "").lower()
    return p in _OPENAI_TOOL_PROVIDERS or p in _ANTHROPIC_TOOL_PROVIDERS


def tool_format(provider: str) -> str:
    """Return 'openai', 'anthropic', or 'none' for a provider."""
    p = (provider or "").lower()
    if p in _ANTHROPIC_TOOL_PROVIDERS:
        return "anthropic"
    if p in _NO_NATIVE_TOOLS:
        return "none"
    return "openai"  # default for all other providers


# ---------------------------------------------------------------------------
# Canonical ToolCall dataclass
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """
    Provider-agnostic representation of a single tool invocation returned
    by the LLM.
    """
    id:        str            # provider-assigned call id (may be empty for Ollama)
    name:      str            # function name
    arguments: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_openai_dict(cls, d: dict) -> "ToolCall":
        """Parse one entry from choices[0].message.tool_calls."""
        fn   = d.get("function", {})
        args = fn.get("arguments", "{}")
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
        except json.JSONDecodeError:
            parsed = {"_raw": args}
        return cls(id=d.get("id", ""), name=fn.get("name", ""), arguments=parsed)

    @classmethod
    def from_anthropic_dict(cls, d: dict) -> "ToolCall":
        """Parse one tool_use content block from Anthropic's response."""
        inp = d.get("input", {})
        if isinstance(inp, str):
            try:
                inp = json.loads(inp)
            except json.JSONDecodeError:
                inp = {"_raw": inp}
        return cls(id=d.get("id", ""), name=d.get("name", ""), arguments=inp)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


# ---------------------------------------------------------------------------
# Request-side: adapt tools for each provider
# ---------------------------------------------------------------------------

def _to_openai_tools(tools: List[dict]) -> List[dict]:
    """Ensure tools are in standard OpenAI format (pass-through if already correct)."""
    result = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            result.append(t)  # already correct
        elif "name" in t and "parameters" in t:
            # Bare function object without the wrapper — wrap it
            result.append({"type": "function", "function": t})
        else:
            log.debug("tool_bridge: skipping malformed tool def: %r", list(t.keys()))
    return result


def _to_anthropic_tools(tools: List[dict]) -> List[dict]:
    """
    Convert from OpenAI tool format to Anthropic's tools format:
      OpenAI:    {"type":"function","function":{"name":...,"parameters":{...}}}
      Anthropic: {"name":..., "description":..., "input_schema":{...}}
    """
    result = []
    for t in tools:
        fn = t.get("function", t)  # handle both wrapped and bare
        name   = fn.get("name", "")
        desc   = fn.get("description", "")
        params = fn.get("parameters", {"type": "object", "properties": {}})
        if not name:
            continue
        result.append({
            "name":         name,
            "description":  desc,
            "input_schema": params,
        })
    return result


def adapt_request_body(
    provider: str,
    body: dict,
    tools: List[dict],
) -> dict:
    """
    Mutate *body* in-place with the correct tool fields for *provider*.
    Returns the mutated body (same object) for convenience.

    If the provider has no native tool support, the tools are injected
    as a system-prompt JSON schema so the model can still call them via
    structured text output.
    """
    if not tools:
        return body

    fmt = tool_format(provider)

    if fmt == "openai":
        body["tools"]       = _to_openai_tools(tools)
        body["tool_choice"] = "auto"

    elif fmt == "anthropic":
        body["tools"] = _to_anthropic_tools(tools)
        # Anthropic doesn't use "tool_choice" the same way; omit for default "auto"

    elif fmt == "none":
        # System-prompt fallback: describe tools as JSON and ask for JSON response
        tool_descriptions = _tools_to_system_prompt(tools)
        existing_system = ""
        # Find and extract the system message if present
        messages = body.get("messages", [])
        new_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                existing_system = msg.get("content", "")
            else:
                new_messages.append(msg)
        combined = (existing_system + "\n\n" if existing_system else "") + tool_descriptions
        body["messages"] = [{"role": "system", "content": combined}] + new_messages
        log.debug("tool_bridge: injected %d tools as system prompt for %s", len(tools), provider)

    return body


def _tools_to_system_prompt(tools: List[dict]) -> str:
    """Convert tool definitions to a system-prompt description for non-tool providers."""
    lines = [
        "You have access to the following tools. To call a tool, respond with a JSON "
        "object on its own line in the format: "
        '{"tool_call": {"name": "<tool_name>", "arguments": {<args>}}}',
        "",
        "Available tools:",
    ]
    for t in tools:
        fn   = t.get("function", t)
        name = fn.get("name", "?")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        props  = params.get("properties", {})
        req    = params.get("required", [])
        arg_parts = []
        for pname, pdef in props.items():
            r = " (required)" if pname in req else ""
            arg_parts.append(f"  {pname} ({pdef.get('type','any')}){r}: {pdef.get('description','')}")
        lines.append(f"- {name}: {desc}")
        lines.extend(arg_parts)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response-side: parse tool calls from provider responses
# ---------------------------------------------------------------------------

def parse_tool_calls_from_message(provider: str, message: dict) -> List[ToolCall]:
    """
    Extract ToolCall objects from a *complete* message dict (non-streaming).

    For OpenAI-compatible: reads message["tool_calls"].
    For Anthropic: reads message["content"] for type==tool_use blocks.
    Returns empty list if no tool calls are present.
    """
    fmt = tool_format(provider)
    results: List[ToolCall] = []

    if fmt == "openai":
        for tc in (message.get("tool_calls") or []):
            try:
                results.append(ToolCall.from_openai_dict(tc))
            except Exception as e:
                log.debug("tool_bridge: failed to parse openai tool call: %s", e)

    elif fmt == "anthropic":
        content = message.get("content") or []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    try:
                        results.append(ToolCall.from_anthropic_dict(block))
                    except Exception as e:
                        log.debug("tool_bridge: failed to parse anthropic tool call: %s", e)

    elif fmt == "none":
        # Try to detect JSON tool_call in the text content
        text = message.get("content", "")
        results.extend(_parse_text_tool_calls(text))

    return results


# ---------------------------------------------------------------------------
# Extensible text-format tool call parsers
# ---------------------------------------------------------------------------
# Any plugin or integration can register a custom parser here to handle
# model-specific text formats without modifying core code.
#
# A parser is a callable(text: str) -> List[ToolCall].
# Parsers are called after the built-in formats; results are deduplicated.
#
# Usage (from a plugin):
#   from server.tool_bridge import register_text_tool_parser
#   register_text_tool_parser(my_parser)

_TEXT_TOOL_PARSERS: List = []


def register_text_tool_parser(fn) -> None:
    """Register a custom text-format tool-call parser."""
    if fn not in _TEXT_TOOL_PARSERS:
        _TEXT_TOOL_PARSERS.append(fn)


def _parse_text_tool_calls(text: str) -> List[ToolCall]:
    """
    Parse tool calls embedded in free-form text.
    Used for providers without native tool support AND as fallback when a
    native-tool provider (e.g. Groq + Llama-3.3) outputs function calls as
    text tokens instead of the OpenAI tool_calls delta.

    Handles four common formats:

    1. Essence JSON wrapper  (system-prompt-injected format):
         {"tool_call": {"name": "web_search", "arguments": {...}}}

    2. Llama 3.3 / Groq XML-like format (model's built-in format):
         <function/web_search {"query": "...", "max_results": 6}></function>

    3. Llama 3.1/3.2 pipe-delimited format:
         <|python_tag|>{"name": "web_search", "parameters": {...}}

    4. Tool-call JSON array (some fine-tunes):
         [{"name": "web_search", "arguments": {...}}]

    Deduplicates by (name, canonical-args) to suppress repeated calls.
    """
    import re as _re

    results:  List[ToolCall]  = []
    seen:     set             = set()   # (name, args_repr) dedup
    decoder   = json.JSONDecoder()
    search_text = text or ""

    def _add(name: str, args: dict) -> None:
        if not name:
            return
        key = (name, repr(sorted(args.items()) if isinstance(args, dict) else args))
        if key in seen:
            return
        seen.add(key)
        results.append(ToolCall(id=f"text_tc_{name}_{len(results)}", name=name, arguments=args))

    # ── Format 2: <function/NAME JSON_ARGS></function> ────────────────
    # Llama 3.3 native format when served through Groq/Ollama chat endpoint
    fn_tag_re = _re.compile(
        r'<function/([^\s>{]+)\s*([\s\S]*?)</function>',
        _re.DOTALL,
    )
    for m in fn_tag_re.finditer(search_text):
        fn_name   = m.group(1).strip()
        args_text = m.group(2).strip()
        if args_text:
            try:
                args = json.loads(args_text)
            except json.JSONDecodeError:
                # Try to extract the first JSON object from the args text
                try:
                    args, _ = decoder.raw_decode(args_text)
                except (json.JSONDecodeError, ValueError):
                    args = {"_raw": args_text}
        else:
            args = {}
        if isinstance(args, dict):
            _add(fn_name, args)
        else:
            _add(fn_name, {"_raw": args_text})

    # ── Format 3: <|python_tag|>JSON ──────────────────────────────────
    # Llama 3.1/3.2 tool-call format (sometimes used by Ollama)
    pytag_re = _re.compile(r'<\|python_tag\|>([\s\S]*?)(?:<\|eom_id\|>|$)')
    for m in pytag_re.finditer(search_text):
        payload = m.group(1).strip()
        try:
            obj = json.loads(payload)
            if isinstance(obj, dict):
                name = obj.get("name") or obj.get("function", "")
                args = obj.get("parameters") or obj.get("arguments") or obj.get("args") or {}
                _add(name, args)
        except (json.JSONDecodeError, ValueError):
            pass

    # ── Formats 1 & 4: JSON scanning ─────────────────────────────────
    pos = 0
    while pos < len(search_text):
        idx = search_text.find("{", pos)
        if idx == -1:
            # Also try arrays for format 4
            idx = search_text.find("[", pos)
            if idx == -1:
                break
        try:
            obj, rel_end = decoder.raw_decode(search_text, idx)
            end_pos = idx + rel_end - idx  # absolute end position

            if isinstance(obj, dict):
                # Format 1: {"tool_call": {"name": ..., "arguments": ...}}
                if "tool_call" in obj:
                    tc   = obj["tool_call"]
                    name = tc.get("name", "") if isinstance(tc, dict) else ""
                    args = tc.get("arguments", {}) if isinstance(tc, dict) else {}
                    _add(name, args)
                # Bare tool call: {"name": "...", "arguments": {...}}
                elif "name" in obj and ("arguments" in obj or "parameters" in obj):
                    name = obj.get("name", "")
                    args = obj.get("arguments") or obj.get("parameters") or {}
                    _add(name, args)

            elif isinstance(obj, list):
                # Format 4: [{"name": ..., "arguments": ...}, ...]
                for item in obj:
                    if isinstance(item, dict) and "name" in item:
                        name = item.get("name", "")
                        args = item.get("arguments") or item.get("parameters") or {}
                        _add(name, args)

            pos = end_pos
        except (json.JSONDecodeError, ValueError):
            pos = idx + 1

    # ── Custom parsers (registered by plugins) ────────────────────────
    for _custom_parser in _TEXT_TOOL_PARSERS:
        try:
            for tc in (_custom_parser(search_text) or []):
                if isinstance(tc, ToolCall):
                    _add(tc.name, tc.arguments)
        except Exception as _pe:
            log.debug("custom text-tool parser %r error: %s", _custom_parser, _pe)

    return results


def has_tool_calls(provider: str, response_data: dict) -> bool:
    """
    Return True if *response_data* (a complete SSE chunk or non-streaming
    response) contains tool call signals.

    OpenAI-compatible: finish_reason == "tool_calls"
    Anthropic:         stop_reason  == "tool_use"
    """
    fmt = tool_format(provider)
    if fmt == "openai":
        choices = response_data.get("choices") or []
        if choices:
            return choices[0].get("finish_reason") == "tool_calls"
        return False
    elif fmt == "anthropic":
        return response_data.get("stop_reason") == "tool_use"
    return False


# ---------------------------------------------------------------------------
# Tool result injection — provider-correct message format
# ---------------------------------------------------------------------------

def make_tool_result_messages(
    provider: str,
    tool_calls: List[ToolCall],
    results: List[str],
    assistant_tool_call_message: Optional[dict] = None,
) -> List[dict]:
    """
    Build the messages that must be appended to the conversation after
    executing tool calls.

    Returns a list of dicts ready to extend messages[]:
      - For OpenAI: [assistant_with_tool_calls, tool_result, tool_result, ...]
      - For Anthropic: [assistant_with_tool_use, user_with_tool_results]

    Parameters
    ----------
    provider
        Active provider id.
    tool_calls
        The ToolCall objects that were executed.
    results
        String results in the same order as tool_calls.
    assistant_tool_call_message
        The raw assistant message containing the tool_calls (OpenAI) or
        tool_use content (Anthropic), needed to re-inject it correctly.
        If None, a minimal assistant message is constructed.
    """
    fmt = tool_format(provider)
    messages: List[dict] = []

    if fmt == "openai":
        # 1. The assistant turn that issued the tool calls
        if assistant_tool_call_message:
            messages.append(assistant_tool_call_message)
        else:
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id":       tc.id or f"call_{tc.name}",
                        "type":     "function",
                        "function": {
                            "name":      tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ],
            })

        # 2. One tool result message per call
        for tc, result in zip(tool_calls, results):
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id or f"call_{tc.name}",
                "content":      str(result),
            })

    elif fmt == "anthropic":
        # 1. Assistant message containing tool_use blocks
        if assistant_tool_call_message:
            messages.append(assistant_tool_call_message)
        else:
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type":  "tool_use",
                        "id":    tc.id or f"toolu_{tc.name}",
                        "name":  tc.name,
                        "input": tc.arguments,
                    }
                    for tc in tool_calls
                ],
            })

        # 2. Single user message with all tool_result blocks
        result_blocks = [
            {
                "type":        "tool_result",
                "tool_use_id": tc.id or f"toolu_{tc.name}",
                "content":     str(result),
            }
            for tc, result in zip(tool_calls, results)
        ]
        messages.append({"role": "user", "content": result_blocks})

    else:
        # No native tools: just add the results as a user message
        result_text = "\n\n".join(
            f"[{tc.name} result]: {result}"
            for tc, result in zip(tool_calls, results)
        )
        messages.append({"role": "user", "content": result_text})

    return messages


# ---------------------------------------------------------------------------
# Streaming tool-call accumulator
# ---------------------------------------------------------------------------

class StreamAccumulator:
    """
    Stateful accumulator for streaming tool-call deltas.

    Usage:
        acc = StreamAccumulator(provider)
        for chunk in sse_stream:
            data = json.loads(chunk)
            calls = acc.feed(data)
            if calls is not None:
                # Tool calls complete — execute them
                break
            token = acc.last_text_token
            if token:
                emit(token)

    Once feed() returns a non-None list, the stream for this turn is
    complete and contained tool calls (may be empty list for clean stop).
    """

    def __init__(self, provider: str) -> None:
        self._fmt         = tool_format(provider)
        self._text_buf:   str  = ""
        self._tool_bufs:  Dict[int, dict] = {}   # index → partial call
        self._anth_bufs:  Dict[int, dict] = {}   # Anthropic block index → partial
        self._think_bufs: Dict[int, str]  = {}   # Anthropic thinking block index → text
        self.last_text_token:    str = ""
        self.last_thinking_token: str = ""        # thinking delta (not shown by default)
        self.finish_reason:      Optional[str] = None

    def feed(self, chunk: dict) -> Optional[List[ToolCall]]:
        """
        Process one parsed SSE chunk.

        Returns:
          None             — stream is still ongoing (emit last_text_token if set)
          List[ToolCall]   — stream is done; tool calls (possibly empty) are complete
        """
        self.last_text_token     = ""
        self.last_thinking_token = ""

        if self._fmt == "openai":
            return self._feed_openai(chunk)
        elif self._fmt == "anthropic":
            return self._feed_anthropic(chunk)
        else:
            return self._feed_text(chunk)

    # ── OpenAI-compatible streaming ────────────────────────────────────

    def _feed_openai(self, chunk: dict) -> Optional[List[ToolCall]]:
        choices = chunk.get("choices") or []
        if not choices:
            return None

        choice = choices[0]
        delta  = choice.get("delta") or {}
        finish = choice.get("finish_reason")

        # Text token
        text = delta.get("content") or ""
        if text:
            self._text_buf       += text
            self.last_text_token  = text

        # Tool call deltas
        for tc_delta in (delta.get("tool_calls") or []):
            idx  = tc_delta.get("index", 0)
            buf  = self._tool_bufs.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            buf["id"]        += (tc_delta.get("id") or "")
            fn = tc_delta.get("function") or {}
            buf["name"]      += (fn.get("name") or "")
            buf["arguments"] += (fn.get("arguments") or "")

        if finish:
            self.finish_reason = finish
            if finish == "tool_calls":
                return self._assemble_openai_calls()
            # Some providers (e.g. Groq + Llama-3.3) emit finish_reason="stop"
            # even when the model has embedded function calls in the text content
            # using its trained format (e.g. <function/name {...}></function>).
            # Fall back to text parsing so these calls are not silently dropped.
            text_calls = _parse_text_tool_calls(self._text_buf)
            if text_calls:
                log.debug(
                    "StreamAccumulator: found %d text-embedded tool call(s) in finish_reason=%s response",
                    len(text_calls), finish,
                )
                # Strip the raw function-call markup from the displayed text so
                # the user doesn't see raw <function/...></function> tags.
                import re as _re
                cleaned = _re.sub(r'<function/[^>]*>[\s\S]*?</function>', '', self._text_buf).strip()
                cleaned = _re.sub(r'<\|python_tag\|>[\s\S]*?(?:<\|eom_id\|>|$)', '', cleaned).strip()
                self._text_buf = cleaned
                return text_calls
            return []   # clean stop — no tool calls

        return None

    def _assemble_openai_calls(self) -> List[ToolCall]:
        calls = []
        for buf in sorted(self._tool_bufs.values(), key=lambda b: id(b)):
            try:
                args = json.loads(buf["arguments"]) if buf["arguments"] else {}
            except json.JSONDecodeError:
                args = {"_raw": buf["arguments"]}
            calls.append(ToolCall(id=buf["id"], name=buf["name"], arguments=args))
        return calls

    # ── Anthropic streaming ────────────────────────────────────────────

    def _feed_anthropic(self, chunk: dict) -> Optional[List[ToolCall]]:
        etype = chunk.get("type", "")

        if etype == "content_block_start":
            block = chunk.get("content_block", {})
            idx   = chunk.get("index", 0)
            btype = block.get("type", "")
            if btype == "tool_use":
                self._anth_bufs[idx] = {
                    "id":         block.get("id", ""),
                    "name":       block.get("name", ""),
                    "input_json": "",
                }
            elif btype == "thinking":
                self._think_bufs[idx] = block.get("thinking", "")
            # text block — nothing to record

        elif etype == "content_block_delta":
            idx   = chunk.get("index", 0)
            delta = chunk.get("delta", {})
            dtype = delta.get("type", "")
            if dtype == "text_delta":
                text = delta.get("text", "")
                self._text_buf       += text
                self.last_text_token  = text
            elif dtype == "input_json_delta":
                if idx in self._anth_bufs:
                    self._anth_bufs[idx]["input_json"] += delta.get("partial_json", "")
            elif dtype == "thinking_delta":
                # Accumulate thinking text — exposed via last_thinking_token
                # but NOT mixed into last_text_token (keeps UI output clean)
                thinking_text = delta.get("thinking", "")
                if thinking_text:
                    self._think_bufs.setdefault(idx, "")
                    self._think_bufs[idx] += thinking_text
                    self.last_thinking_token = thinking_text

        elif etype == "message_delta":
            stop_reason = (chunk.get("delta") or {}).get("stop_reason")
            if stop_reason:
                self.finish_reason = stop_reason
                if stop_reason == "tool_use":
                    return self._assemble_anthropic_calls()
                return []  # end_turn or max_tokens — no tool calls

        elif etype == "message_stop":
            if self.finish_reason is None:
                return []

        return None

    def _assemble_anthropic_calls(self) -> List[ToolCall]:
        calls = []
        for buf in self._anth_bufs.values():
            try:
                inp = json.loads(buf["input_json"]) if buf["input_json"] else {}
            except json.JSONDecodeError:
                inp = {"_raw": buf["input_json"]}
            calls.append(ToolCall(id=buf["id"], name=buf["name"], arguments=inp))
        return calls

    # ── System-prompt text fallback streaming ──────────────────────────

    def _feed_text(self, chunk: dict) -> Optional[List[ToolCall]]:
        """For providers without native tool support — detect JSON tool calls in text."""
        # OpenAI-style body
        choices = chunk.get("choices") or []
        if choices:
            delta  = choices[0].get("delta") or {}
            finish = choices[0].get("finish_reason")
            text   = delta.get("content") or ""
            if text:
                self._text_buf       += text
                self.last_text_token  = text
            if finish:
                self.finish_reason = finish
                return self._finalize_text_calls()
        # Ollama native format
        elif chunk.get("done"):
            text = (chunk.get("message") or {}).get("content", "")
            if text:
                self._text_buf += text
            return self._finalize_text_calls()
        else:
            text = (chunk.get("message") or {}).get("content", "")
            if text:
                self._text_buf       += text
                self.last_text_token  = text
        return None

    def _finalize_text_calls(self) -> List[ToolCall]:
        """Parse tool calls from accumulated text, stripping markup from the display buffer."""
        import re as _re
        calls = _parse_text_tool_calls(self._text_buf)
        if calls:
            # Remove raw function-call markup so it doesn't render in the chat
            cleaned = _re.sub(r'<function/[^>]*>[\s\S]*?</function>', '', self._text_buf).strip()
            cleaned = _re.sub(r'<\|python_tag\|>[\s\S]*?(?:<\|eom_id\|>|$)', '', cleaned).strip()
            self._text_buf = cleaned
        return calls

    @property
    def text_so_far(self) -> str:
        return self._text_buf

    @property
    def thinking_so_far(self) -> str:
        """All accumulated thinking/reasoning text (Anthropic extended thinking)."""
        return "\n".join(self._think_bufs.values())


# ---------------------------------------------------------------------------
# Message normalisation helpers
# ---------------------------------------------------------------------------

def normalise_messages_for_provider(provider: str, messages: List[dict]) -> List[dict]:
    """
    Apply any provider-specific message shape corrections.

    Currently handles:
      • Anthropic requires system messages to be in a top-level "system"
        field (handled separately in normalize_to_openai) — this method
        handles the remaining case where tool role messages must be
        converted to user messages with tool_result content.
      • Coerces None content to "" for providers that don't accept null.
      • Translates image content blocks from OpenAI canonical format to
        the target provider's native image format (via multimodal module).
    """
    # Lazy import — avoids circular imports and keeps multimodal optional
    try:
        from server.multimodal import normalise_image_blocks_for_provider as _norm_img
        _has_multimodal = True
    except Exception:
        _has_multimodal = False

    fmt = tool_format(provider)
    result = []

    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content")

        # Convert None content to empty string (some providers reject null)
        if content is None and not msg.get("tool_calls"):
            msg = {**msg, "content": ""}

        if fmt == "anthropic" and role == "tool":
            # OpenAI tool result → Anthropic tool_result content block inside a user message
            tc_id  = msg.get("tool_call_id", "")
            text   = msg.get("content", "")
            # Merge consecutive tool messages into one user message (done in kernel)
            result.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tc_id, "content": str(text)}
                ],
            })
            continue

        # Translate image content blocks for the target provider
        if _has_multimodal and isinstance(msg.get("content"), list):
            translated = _norm_img(msg["content"], provider)
            if translated is not msg["content"]:
                msg = {**msg, "content": translated}

        result.append(msg)

    return result


def extract_system_for_anthropic(messages: List[dict]) -> tuple[str, List[dict]]:
    """
    Anthropic requires the system prompt in a dedicated top-level field,
    not as a message with role=='system'.

    Returns (system_text, remaining_messages).
    """
    system_parts = []
    rest = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multimodal — extract text parts
                system_parts.extend(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            else:
                system_parts.append(str(content))
        else:
            rest.append(msg)
    return "\n\n".join(filter(None, system_parts)), rest


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_bridge_instance: Optional["ToolBridge"] = None


class ToolBridge:
    """
    Facade providing a single object interface to all tool bridge functions.
    Access via get_tool_bridge().
    """

    # ── Request side ───────────────────────────────────────────────────

    def adapt_request(self, provider: str, body: dict, tools: List[dict]) -> dict:
        """Adapt *body* in-place for *provider*'s tool format. Returns body."""
        return adapt_request_body(provider, body, tools)

    def supports_tools(self, provider: str) -> bool:
        return supports_native_tools(provider)

    def tool_format(self, provider: str) -> str:
        return tool_format(provider)

    # ── Response side ──────────────────────────────────────────────────

    def parse_tool_calls(self, provider: str, message: dict) -> List[ToolCall]:
        return parse_tool_calls_from_message(provider, message)

    def has_tool_calls(self, provider: str, response_data: dict) -> bool:
        return has_tool_calls(provider, response_data)

    def make_tool_result_messages(
        self,
        provider: str,
        tool_calls: List[ToolCall],
        results: List[str],
        assistant_msg: Optional[dict] = None,
    ) -> List[dict]:
        return make_tool_result_messages(provider, tool_calls, results, assistant_msg)

    # ── Streaming ──────────────────────────────────────────────────────

    def new_accumulator(self, provider: str) -> StreamAccumulator:
        return StreamAccumulator(provider)

    # ── Message normalisation ──────────────────────────────────────────

    def normalise_messages(self, provider: str, messages: List[dict]) -> List[dict]:
        return normalise_messages_for_provider(provider, messages)

    def extract_anthropic_system(self, messages: List[dict]) -> tuple[str, List[dict]]:
        return extract_system_for_anthropic(messages)


def get_tool_bridge() -> ToolBridge:
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = ToolBridge()
    return _bridge_instance
