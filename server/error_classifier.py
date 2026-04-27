"""
error_classifier.py — API Error Classification for Smart Failover
==================================================================
Adapted from Hermes agent/error_classifier.py (949 lines → 400 lines,
Hermes-specific deps removed, Essence router patterns retained).

Provides a priority-ordered taxonomy that maps any API exception to a
concrete recovery action:
  retryable              → safe to retry with the same provider
  should_compress        → context overflow: compress before retry
  should_rotate_credential → API key exhausted/invalid: swap key
  should_fallback        → give up on this provider, try next one

Integration
-----------
  Used in kernel._stream_response() to replace bare except-and-retry
  with structured, self-healing failure recovery:

    from server.error_classifier import classify_api_error, FailoverReason
    classified = classify_api_error(exc, provider=provider, model=model)
    if classified.should_compress:
        messages = compressor.compress(messages, force=True)
    if classified.should_fallback:
        continue   # next candidate in fallback chain
    if not classified.retryable:
        break
"""
from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

log = logging.getLogger("essence.errors")


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

class FailoverReason(enum.Enum):
    """Why an API call failed — determines recovery strategy."""
    # Auth / billing
    auth                    = "auth"
    auth_permanent          = "auth_permanent"
    billing                 = "billing"
    rate_limit              = "rate_limit"
    # Server-side
    overloaded              = "overloaded"
    server_error            = "server_error"
    # Transport
    timeout                 = "timeout"
    # Context / payload
    context_overflow        = "context_overflow"
    payload_too_large       = "payload_too_large"
    # Model
    model_not_found         = "model_not_found"
    provider_policy_blocked = "provider_policy_blocked"
    # Request format
    format_error            = "format_error"
    # Anthropic-specific
    thinking_signature      = "thinking_signature"
    long_context_tier       = "long_context_tier"
    # Catch-all
    unknown                 = "unknown"


@dataclass
class ClassifiedError:
    """Structured classification with recovery action hints."""
    reason:                  FailoverReason
    status_code:             Optional[int] = None
    provider:                Optional[str] = None
    model:                   Optional[str] = None
    message:                 str           = ""
    error_context:           Dict[str, Any] = field(default_factory=dict)
    # Recovery hints
    retryable:               bool = True
    should_compress:         bool = False
    should_rotate_credential:bool = False
    should_fallback:         bool = False

    @property
    def is_auth(self) -> bool:
        return self.reason in (FailoverReason.auth, FailoverReason.auth_permanent)

    def __str__(self) -> str:
        flags = []
        if self.should_compress:         flags.append("compress")
        if self.should_rotate_credential:flags.append("rotate-key")
        if self.should_fallback:         flags.append("fallback")
        if self.retryable:               flags.append("retryable")
        flag_str = " ".join(flags) or "abort"
        sc = f" HTTP {self.status_code}" if self.status_code else ""
        return f"[{self.reason.value}{sc}] {self.message[:120]} → {flag_str}"


# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------

_BILLING_PATTERNS = [
    "insufficient credits", "insufficient_quota", "credit balance",
    "credits have been exhausted", "top up your credits", "payment required",
    "billing hard limit", "exceeded your current quota", "account is deactivated",
    "plan does not include",
]

_RATE_LIMIT_PATTERNS = [
    "rate limit", "rate_limit", "too many requests", "throttled",
    "requests per minute", "tokens per minute", "requests per day",
    "try again in", "please retry after", "resource_exhausted",
    "rate increased too quickly", "throttlingexception",
    "too many concurrent requests", "servicequotaexceededexception",
]

_USAGE_LIMIT_PATTERNS = [
    "usage limit", "quota", "limit exceeded", "key limit exceeded",
]

_USAGE_LIMIT_TRANSIENT_SIGNALS = [
    "try again", "retry", "resets at", "reset in", "wait",
    "requests remaining", "periodic", "window",
]

_PAYLOAD_TOO_LARGE_PATTERNS = [
    "request entity too large", "payload too large", "error code: 413",
]

_CONTEXT_OVERFLOW_PATTERNS = [
    "context length", "context size", "maximum context", "token limit",
    "too many tokens", "reduce the length", "exceeds the limit",
    "context window", "prompt is too long", "prompt exceeds max length",
    "max_tokens", "maximum number of tokens", "exceeds the max_model_len",
    "max_model_len", "prompt length", "input is too long",
    "maximum model length", "context length exceeded", "truncating input",
    "slot context", "n_ctx_slot", "超过最大长度", "上下文长度",
    "max input token", "input token",
    "exceeds the maximum number of input tokens",
]

_MODEL_NOT_FOUND_PATTERNS = [
    "is not a valid model", "invalid model", "model not found",
    "model_not_found", "does not exist", "no such model",
    "unknown model", "unsupported model",
]

_PROVIDER_POLICY_BLOCKED_PATTERNS = [
    "no endpoints available matching your guardrail",
    "no endpoints available matching your data policy",
    "no endpoints found matching your data policy",
]

_AUTH_PATTERNS = [
    "invalid api key", "invalid_api_key", "authentication",
    "unauthorized", "forbidden", "invalid token",
    "token expired", "token revoked", "access denied",
]

_SERVER_DISCONNECT_PATTERNS = [
    "server disconnected", "peer closed connection",
    "connection reset by peer", "connection was closed",
    "network connection lost", "unexpected eof",
    "incomplete chunked read",
]

_SSL_TRANSIENT_PATTERNS = [
    "bad record mac", "ssl alert", "tls alert",
    "ssl handshake failure", "tlsv1 alert", "sslv3 alert",
    "bad_record_mac", "ssl_alert", "tls_alert",
    "tls_alert_internal_error", "[ssl:",
]

_TRANSPORT_ERROR_TYPES = frozenset({
    "ReadTimeout", "ConnectTimeout", "PoolTimeout",
    "ConnectError", "RemoteProtocolError",
    "ConnectionError", "ConnectionResetError", "ConnectionAbortedError",
    "BrokenPipeError", "TimeoutError", "ReadError",
    "ServerDisconnectedError",
    "SSLError", "SSLZeroReturnError", "SSLWantReadError",
    "SSLWantWriteError", "SSLEOFError", "SSLSyscallError",
    "APIConnectionError", "APITimeoutError",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_status_code(error: Exception) -> Optional[int]:
    current = error
    for _ in range(5):
        code = getattr(current, "status_code", None)
        if isinstance(code, int):
            return code
        code = getattr(current, "status", None)
        if isinstance(code, int) and 100 <= code < 600:
            return code
        cause = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if cause is None or cause is current:
            break
        current = cause
    return None


def _extract_error_body(error: Exception) -> dict:
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        return body
    response = getattr(error, "response", None)
    if response is not None:
        try:
            j = response.json()
            if isinstance(j, dict):
                return j
        except Exception:
            pass
    return {}


def _extract_error_code(body: dict) -> str:
    if not body:
        return ""
    err = body.get("error", {})
    if isinstance(err, dict):
        code = err.get("code") or err.get("type") or ""
        if isinstance(code, str) and code.strip():
            return code.strip()
    code = body.get("code") or body.get("error_code") or ""
    return str(code).strip() if code else ""


def _extract_message(error: Exception, body: dict) -> str:
    if body:
        err = body.get("error", {})
        if isinstance(err, dict):
            msg = err.get("message", "")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()[:500]
        msg = body.get("message", "")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()[:500]
    return str(error)[:500]


def _build_error_msg(error: Exception, body: dict) -> str:
    """Build a comprehensive lowercased string for pattern matching."""
    parts = [str(error).lower()]
    if isinstance(body, dict):
        err = body.get("error", {})
        if isinstance(err, dict):
            body_msg = str(err.get("message") or "").lower()
            if body_msg and body_msg not in parts[0]:
                parts.append(body_msg)
            # Parse OpenRouter's metadata.raw for wrapped provider errors
            meta = err.get("metadata", {})
            if isinstance(meta, dict):
                raw_json = meta.get("raw") or ""
                if isinstance(raw_json, str) and raw_json.strip():
                    try:
                        inner = json.loads(raw_json)
                        if isinstance(inner, dict):
                            inner_err = inner.get("error", {})
                            if isinstance(inner_err, dict):
                                meta_msg = str(inner_err.get("message") or "").lower()
                                if meta_msg and meta_msg not in parts[0]:
                                    parts.append(meta_msg)
                    except (json.JSONDecodeError, TypeError):
                        pass
        if not any(parts[1:]):
            flat = str(body.get("message") or "").lower()
            if flat and flat not in parts[0]:
                parts.append(flat)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_api_error(
    error: Exception,
    *,
    provider:       str = "",
    model:          str = "",
    approx_tokens:  int = 0,
    context_length: int = 32_768,
    num_messages:   int = 0,
) -> ClassifiedError:
    """Classify an API exception into a structured recovery recommendation.

    Priority-ordered pipeline:
      1. Provider-specific patterns (thinking sigs, long-context tier)
      2. HTTP status code + message-aware refinement
      3. Structured error code (from body)
      4. Message pattern matching (billing vs rate_limit vs context vs auth)
      5. SSL/TLS transient patterns → retry (not compress)
      6. Server disconnect + large session → context overflow
      7. Transport error type names
      8. Unknown → retryable with backoff
    """
    status_code = _extract_status_code(error)
    error_type  = type(error).__name__

    # Copilot RateLimitError may omit .status_code
    if status_code is None and error_type == "RateLimitError":
        status_code = 429

    body      = _extract_error_body(error)
    error_code = _extract_error_code(body)
    error_msg  = _build_error_msg(error, body)

    def _result(reason: FailoverReason, **kw) -> ClassifiedError:
        defaults = dict(
            reason      = reason,
            status_code = status_code,
            provider    = provider or None,
            model       = model or None,
            message     = _extract_message(error, body),
        )
        defaults.update(kw)
        return ClassifiedError(**defaults)

    # ── 1. Provider-specific ───────────────────────────────────────────
    if status_code == 400 and "signature" in error_msg and "thinking" in error_msg:
        return _result(FailoverReason.thinking_signature, retryable=True)

    if (status_code == 429
            and "extra usage" in error_msg
            and "long context" in error_msg):
        return _result(FailoverReason.long_context_tier, retryable=True, should_compress=True)

    # ── 2. HTTP status code ────────────────────────────────────────────
    if status_code is not None:
        r = _by_status(status_code, error_msg, error_code, body,
                       provider=provider.lower(), model=model.lower(),
                       approx_tokens=approx_tokens, context_length=context_length,
                       num_messages=num_messages, fn=_result)
        if r is not None:
            return r

    # ── 3. Error code ──────────────────────────────────────────────────
    if error_code:
        r = _by_error_code(error_code, fn=_result)
        if r is not None:
            return r

    # ── 4. Message patterns ────────────────────────────────────────────
    r = _by_message(error_msg, error_type,
                    approx_tokens=approx_tokens, context_length=context_length,
                    fn=_result)
    if r is not None:
        return r

    # ── 5. SSL transient → retry (not compress) ────────────────────────
    if any(p in error_msg for p in _SSL_TRANSIENT_PATTERNS):
        return _result(FailoverReason.timeout, retryable=True)

    # ── 6. Server disconnect: large session → context overflow ─────────
    if any(p in error_msg for p in _SERVER_DISCONNECT_PATTERNS) and not status_code:
        is_large = (approx_tokens > context_length * 0.6
                    or approx_tokens > 120_000
                    or num_messages > 200)
        if is_large:
            return _result(FailoverReason.context_overflow, retryable=True, should_compress=True)
        return _result(FailoverReason.timeout, retryable=True)

    # ── 7. Transport ───────────────────────────────────────────────────
    if error_type in _TRANSPORT_ERROR_TYPES or isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return _result(FailoverReason.timeout, retryable=True)

    # ── 8. Unknown ─────────────────────────────────────────────────────
    return _result(FailoverReason.unknown, retryable=True)


# ---------------------------------------------------------------------------
# Sub-classifiers
# ---------------------------------------------------------------------------

def _by_status(
    status_code, error_msg, error_code, body,
    *, provider, model, approx_tokens, context_length, num_messages, fn,
) -> Optional[ClassifiedError]:
    if status_code == 401:
        return fn(FailoverReason.auth, retryable=False,
                  should_rotate_credential=True, should_fallback=True)
    if status_code == 403:
        if "key limit exceeded" in error_msg or "spending limit" in error_msg:
            return fn(FailoverReason.billing, retryable=False,
                      should_rotate_credential=True, should_fallback=True)
        return fn(FailoverReason.auth, retryable=False, should_fallback=True)
    if status_code == 402:
        return _classify_402(error_msg, fn)
    if status_code == 404:
        if any(p in error_msg for p in _PROVIDER_POLICY_BLOCKED_PATTERNS):
            return fn(FailoverReason.provider_policy_blocked, retryable=False, should_fallback=False)
        if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
            return fn(FailoverReason.model_not_found, retryable=False, should_fallback=True)
        return fn(FailoverReason.unknown, retryable=True)
    if status_code == 413:
        return fn(FailoverReason.payload_too_large, retryable=True, should_compress=True)
    if status_code == 429:
        return fn(FailoverReason.rate_limit, retryable=True,
                  should_rotate_credential=True, should_fallback=True)
    if status_code == 400:
        return _classify_400(error_msg, error_code, body,
                             approx_tokens=approx_tokens,
                             context_length=context_length,
                             num_messages=num_messages, fn=fn)
    if status_code in (500, 502):
        return fn(FailoverReason.server_error, retryable=True)
    if status_code in (503, 529):
        return fn(FailoverReason.overloaded, retryable=True)
    if 400 <= status_code < 500:
        return fn(FailoverReason.format_error, retryable=False, should_fallback=True)
    if 500 <= status_code < 600:
        return fn(FailoverReason.server_error, retryable=True)
    return None


def _classify_402(error_msg: str, fn) -> ClassifiedError:
    if (any(p in error_msg for p in _USAGE_LIMIT_PATTERNS)
            and any(p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS)):
        return fn(FailoverReason.rate_limit, retryable=True,
                  should_rotate_credential=True, should_fallback=True)
    return fn(FailoverReason.billing, retryable=False,
              should_rotate_credential=True, should_fallback=True)


def _classify_400(
    error_msg, error_code, body,
    *, approx_tokens, context_length, num_messages, fn,
) -> ClassifiedError:
    if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
        return fn(FailoverReason.context_overflow, retryable=True, should_compress=True)
    if any(p in error_msg for p in _PROVIDER_POLICY_BLOCKED_PATTERNS):
        return fn(FailoverReason.provider_policy_blocked, retryable=False, should_fallback=False)
    if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
        return fn(FailoverReason.model_not_found, retryable=False, should_fallback=True)
    if any(p in error_msg for p in _RATE_LIMIT_PATTERNS):
        return fn(FailoverReason.rate_limit, retryable=True,
                  should_rotate_credential=True, should_fallback=True)
    if any(p in error_msg for p in _BILLING_PATTERNS):
        return fn(FailoverReason.billing, retryable=False,
                  should_rotate_credential=True, should_fallback=True)
    # Generic 400 + large session → probable context overflow
    err_msg = ""
    if isinstance(body, dict):
        err_obj = body.get("error", {})
        if isinstance(err_obj, dict):
            err_msg = str(err_obj.get("message") or "").strip().lower()
        if not err_msg:
            err_msg = str(body.get("message") or "").strip().lower()
    is_generic = len(err_msg) < 30 or err_msg in ("error", "")
    is_large   = (approx_tokens > context_length * 0.4
                  or approx_tokens > 80_000
                  or num_messages > 80)
    if is_generic and is_large:
        return fn(FailoverReason.context_overflow, retryable=True, should_compress=True)
    return fn(FailoverReason.format_error, retryable=False, should_fallback=True)


def _by_error_code(error_code: str, fn) -> Optional[ClassifiedError]:
    code = error_code.lower()
    if code in ("resource_exhausted", "throttled", "rate_limit_exceeded"):
        return fn(FailoverReason.rate_limit, retryable=True, should_rotate_credential=True)
    if code in ("insufficient_quota", "billing_not_active", "payment_required"):
        return fn(FailoverReason.billing, retryable=False,
                  should_rotate_credential=True, should_fallback=True)
    if code in ("model_not_found", "model_not_available", "invalid_model"):
        return fn(FailoverReason.model_not_found, retryable=False, should_fallback=True)
    if code in ("context_length_exceeded", "max_tokens_exceeded"):
        return fn(FailoverReason.context_overflow, retryable=True, should_compress=True)
    return None


def _by_message(
    error_msg, error_type, *, approx_tokens, context_length, fn,
) -> Optional[ClassifiedError]:
    if any(p in error_msg for p in _PAYLOAD_TOO_LARGE_PATTERNS):
        return fn(FailoverReason.payload_too_large, retryable=True, should_compress=True)
    if any(p in error_msg for p in _USAGE_LIMIT_PATTERNS):
        if any(p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS):
            return fn(FailoverReason.rate_limit, retryable=True,
                      should_rotate_credential=True, should_fallback=True)
        return fn(FailoverReason.billing, retryable=False,
                  should_rotate_credential=True, should_fallback=True)
    if any(p in error_msg for p in _BILLING_PATTERNS):
        return fn(FailoverReason.billing, retryable=False,
                  should_rotate_credential=True, should_fallback=True)
    if any(p in error_msg for p in _RATE_LIMIT_PATTERNS):
        return fn(FailoverReason.rate_limit, retryable=True,
                  should_rotate_credential=True, should_fallback=True)
    if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
        return fn(FailoverReason.context_overflow, retryable=True, should_compress=True)
    if any(p in error_msg for p in _AUTH_PATTERNS):
        return fn(FailoverReason.auth, retryable=False,
                  should_rotate_credential=True, should_fallback=True)
    if any(p in error_msg for p in _PROVIDER_POLICY_BLOCKED_PATTERNS):
        return fn(FailoverReason.provider_policy_blocked, retryable=False, should_fallback=False)
    if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
        return fn(FailoverReason.model_not_found, retryable=False, should_fallback=True)
    return None
