"""
redact.py — Secret & PII Redaction
====================================
Regex-based scrubbing of API keys, tokens, credentials, and sensitive PII
from any text before it reaches cloud LLM providers or log files.

Adapted from hermes-agent/agent/redact.py — all external dependencies
removed; only stdlib (re, os) is used.

Covers:
  • Known API key prefixes (sk-, ghp_, AIza, gsk_, hf_, etc. — 30+ vendors)
  • ENV assignments  (OPENAI_API_KEY=sk-…)
  • JSON fields      ("apiKey": "value")
  • Authorization headers
  • JWT tokens       (eyJ… base64 headers/payloads)
  • Private key blocks (PEM)
  • Database connection strings  (postgres://user:pass@host)
  • URL query params  (?access_token=…)
  • URL userinfo      (https://user:pass@host)
  • Telegram bot tokens
  • E.164 phone numbers
  • Discord snowflake mentions

Integration
-----------
  kernel._stream_response() calls redact_messages(messages) before sending
  to any non-local provider so secrets typed in chat never leave the machine.

  The redacting log formatter is installed in essence.py so log files are
  also scrubbed.

  Disable with environment variable ESSENCE_REDACT_SECRETS=0 (not recommended).
"""
from __future__ import annotations

import logging
import os
import re

# ---------------------------------------------------------------------------
# Kill-switch: set ESSENCE_REDACT_SECRETS=0 to disable (not recommended)
# Snapshot at import time so an LLM-generated export can't disable it mid-session.
# ---------------------------------------------------------------------------
_REDACT_ENABLED = os.getenv("ESSENCE_REDACT_SECRETS", "1").lower() not in (
    "0", "false", "no", "off"
)

# ---------------------------------------------------------------------------
# Sensitive URL query-parameter names (case-insensitive exact match)
# ---------------------------------------------------------------------------
_SENSITIVE_QUERY_PARAMS: frozenset[str] = frozenset({
    "access_token", "refresh_token", "id_token", "token",
    "api_key", "apikey", "client_secret", "password",
    "auth", "jwt", "session", "secret", "key", "code",
    "signature", "x-amz-signature",
})

# Sensitive JSON / form-body key names (exact match — NOT substring)
_SENSITIVE_BODY_KEYS: frozenset[str] = frozenset({
    "access_token", "refresh_token", "id_token", "token",
    "api_key", "apikey", "client_secret", "password",
    "auth", "jwt", "secret", "private_key", "authorization", "key",
})

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Known API-key vendor prefixes — longest-match via alternation
_PREFIX_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{10,}",           # OpenAI / OpenRouter / Anthropic (sk-ant-*)
    r"ghp_[A-Za-z0-9]{10,}",            # GitHub PAT (classic)
    r"github_pat_[A-Za-z0-9_]{10,}",    # GitHub PAT (fine-grained)
    r"gho_[A-Za-z0-9]{10,}",            # GitHub OAuth access token
    r"ghu_[A-Za-z0-9]{10,}",            # GitHub user-to-server token
    r"ghs_[A-Za-z0-9]{10,}",            # GitHub server-to-server token
    r"ghr_[A-Za-z0-9]{10,}",            # GitHub refresh token
    r"xox[baprs]-[A-Za-z0-9-]{10,}",    # Slack tokens
    r"AIza[A-Za-z0-9_-]{30,}",          # Google API keys
    r"pplx-[A-Za-z0-9]{10,}",           # Perplexity
    r"fal_[A-Za-z0-9_-]{10,}",          # Fal.ai
    r"fc-[A-Za-z0-9]{10,}",             # Firecrawl
    r"bb_live_[A-Za-z0-9_-]{10,}",      # BrowserBase
    r"gAAAA[A-Za-z0-9_=-]{20,}",        # Codex encrypted tokens
    r"AKIA[A-Z0-9]{16}",                # AWS Access Key ID
    r"sk_live_[A-Za-z0-9]{10,}",        # Stripe secret key (live)
    r"sk_test_[A-Za-z0-9]{10,}",        # Stripe secret key (test)
    r"rk_live_[A-Za-z0-9]{10,}",        # Stripe restricted key
    r"SG\.[A-Za-z0-9_-]{10,}",          # SendGrid API key
    r"hf_[A-Za-z0-9]{10,}",             # HuggingFace token
    r"r8_[A-Za-z0-9]{10,}",             # Replicate API token
    r"npm_[A-Za-z0-9]{10,}",            # npm access token
    r"pypi-[A-Za-z0-9_-]{10,}",         # PyPI API token
    r"dop_v1_[A-Za-z0-9]{10,}",         # DigitalOcean PAT
    r"doo_v1_[A-Za-z0-9]{10,}",         # DigitalOcean OAuth
    r"am_[A-Za-z0-9_-]{10,}",           # AgentMail API key
    r"tvly-[A-Za-z0-9]{10,}",           # Tavily search API key
    r"exa_[A-Za-z0-9]{10,}",            # Exa search API key
    r"gsk_[A-Za-z0-9]{10,}",            # Groq Cloud API key
    r"hsk-[A-Za-z0-9]{10,}",            # Hindsight API key
    r"mem0_[A-Za-z0-9]{10,}",           # Mem0 Platform API key
]

_PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(" + "|".join(_PREFIX_PATTERNS) + r")(?![A-Za-z0-9_-])"
)

# ENV assignment: OPENAI_API_KEY=sk-abc… or SECRET_TOKEN="value"
_SECRET_ENV_NAMES = r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)"
_ENV_ASSIGN_RE = re.compile(
    rf"([A-Z0-9_]{{0,50}}{_SECRET_ENV_NAMES}[A-Z0-9_]{{0,50}})\s*=\s*(['\"]?)(\S+)\2",
)

# JSON field: "apiKey": "value"
_JSON_KEY_NAMES = (
    r"(?:api_?[Kk]ey|token|secret|password|access_token|refresh_token"
    r"|auth_token|bearer|secret_value|raw_secret|secret_input|key_material)"
)
_JSON_FIELD_RE = re.compile(
    rf'("{_JSON_KEY_NAMES}")\s*:\s*"([^"]+)"',
    re.IGNORECASE,
)

# Authorization header
_AUTH_HEADER_RE = re.compile(
    r"(Authorization:\s*Bearer\s+)(\S+)",
    re.IGNORECASE,
)

# JWT tokens (eyJ… — base64-encoded JSON headers, always start with "eyJ")
_JWT_RE = re.compile(
    r"eyJ[A-Za-z0-9_-]{10,}"
    r"(?:\.[A-Za-z0-9_=-]{4,}){0,2}"
)

# PEM private key blocks
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----"
)

# Database connection strings: proto://user:PASSWORD@host
_DB_CONNSTR_RE = re.compile(
    r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:]+:)([^@]+)(@)",
    re.IGNORECASE,
)

# Telegram bot tokens: (bot)?<digits>:<token>
_TELEGRAM_RE = re.compile(r"(bot)?(\d{8,}):([-A-Za-z0-9_]{30,})")

# URLs with query strings
_URL_WITH_QUERY_RE = re.compile(
    r"(https?|wss?|ftp)://([^\s/?#]+)([^\s?#]*)\?([^\s#]+)(#\S*)?"
)

# URLs with userinfo: https://user:pass@host
_URL_USERINFO_RE = re.compile(r"(https?|wss?|ftp)://([^/\s:@]+):([^/\s@]+)@")

# Form-urlencoded body (conservative)
_FORM_BODY_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.-]*=[^&\s]*(?:&[A-Za-z_][A-Za-z0-9_.-]*=[^&\s]*)+$"
)

# E.164 phone numbers
_PHONE_RE = re.compile(r"(\+[1-9]\d{6,14})(?![A-Za-z0-9])")

# Discord snowflake mentions
_DISCORD_RE = re.compile(r"<@!?(\d{17,20})>")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mask(token: str) -> str:
    """Short tokens are fully masked; long ones preserve prefix + suffix."""
    if len(token) < 18:
        return "***"
    return f"{token[:6]}...{token[-4:]}"


def _redact_query_string(query: str) -> str:
    parts = []
    for pair in query.split("&"):
        if "=" not in pair:
            parts.append(pair)
            continue
        key, _, value = pair.partition("=")
        parts.append(f"{key}=***" if key.lower() in _SENSITIVE_QUERY_PARAMS else pair)
    return "&".join(parts)


def _redact_url_query_params(text: str) -> str:
    def _sub(m: re.Match) -> str:
        return (
            f"{m.group(1)}://{m.group(2)}{m.group(3)}"
            f"?{_redact_query_string(m.group(4))}"
            f"{m.group(5) or ''}"
        )
    return _URL_WITH_QUERY_RE.sub(_sub, text)


def _redact_form_body(text: str) -> str:
    if not text or "\n" in text or "&" not in text:
        return text
    if not _FORM_BODY_RE.match(text.strip()):
        return text
    return _redact_query_string(text.strip())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def redact_sensitive_text(text: str) -> str:
    """Apply all redaction patterns to a block of text.

    Safe to call on any string — non-matching text passes through unchanged.
    No-ops when ESSENCE_REDACT_SECRETS=0.
    """
    if text is None:
        return text   # type: ignore[return-value]
    if not isinstance(text, str):
        text = str(text)
    if not text or not _REDACT_ENABLED:
        return text

    # 1. Known API-key prefixes (sk-, ghp_, AIza, …)
    text = _PREFIX_RE.sub(lambda m: _mask(m.group(1)), text)

    # 2. ENV assignments (OPENAI_API_KEY=…)
    def _env(m: re.Match) -> str:
        q = m.group(2)
        return f"{m.group(1)}={q}{_mask(m.group(3))}{q}"
    text = _ENV_ASSIGN_RE.sub(_env, text)

    # 3. JSON fields ("apiKey": "value")
    text = _JSON_FIELD_RE.sub(
        lambda m: f'{m.group(1)}: "{_mask(m.group(2))}"', text
    )

    # 4. Authorization headers
    text = _AUTH_HEADER_RE.sub(
        lambda m: m.group(1) + _mask(m.group(2)), text
    )

    # 5. JWT tokens (eyJ…)
    text = _JWT_RE.sub(lambda m: _mask(m.group(0)), text)

    # 6. PEM private key blocks
    text = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", text)

    # 7. Database connection strings
    text = _DB_CONNSTR_RE.sub(lambda m: f"{m.group(1)}***{m.group(3)}", text)

    # 8. Telegram bot tokens
    def _telegram(m: re.Match) -> str:
        return f"{m.group(1) or ''}{m.group(2)}:***"
    text = _TELEGRAM_RE.sub(_telegram, text)

    # 9. URL userinfo (https://user:pass@host)
    text = _URL_USERINFO_RE.sub(
        lambda m: f"{m.group(1)}://{m.group(2)}:***@", text
    )

    # 10. URL query params (?access_token=…)
    text = _redact_url_query_params(text)

    # 11. Form-urlencoded bodies
    text = _redact_form_body(text)

    # 12. E.164 phone numbers
    def _phone(m: re.Match) -> str:
        p = m.group(1)
        if len(p) <= 8:
            return f"{p[:2]}****{p[-2:]}"
        return f"{p[:4]}****{p[-4:]}"
    text = _PHONE_RE.sub(_phone, text)

    # 13. Discord snowflake mentions
    text = _DISCORD_RE.sub(
        lambda m: f"<@{'!' if '!' in m.group(0) else ''}***>", text
    )

    return text


def redact_messages(messages: list[dict]) -> list[dict]:
    """Return a new message list with all content fields scrubbed.

    Only applied when the active provider is a cloud endpoint; local Ollama
    calls skip this (handled by should_redact() in the caller).

    The original list is NOT mutated.
    """
    if not _REDACT_ENABLED or not messages:
        return messages

    result = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            scrubbed = redact_sensitive_text(content)
            if scrubbed is not content:
                msg = {**msg, "content": scrubbed}
        elif isinstance(content, list):
            # Multimodal: scrub text parts only, leave image_url parts alone
            new_parts = []
            changed = False
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    s = redact_sensitive_text(part.get("text", ""))
                    if s != part.get("text"):
                        new_parts.append({**part, "text": s})
                        changed = True
                        continue
                new_parts.append(part)
            if changed:
                msg = {**msg, "content": new_parts}
        result.append(msg)
    return result


def should_redact(provider: str, base_url: str = "") -> bool:
    """Return True when messages should be scrubbed before transmission.

    Redaction is skipped for local Ollama/LM-Studio endpoints since no
    data leaves the machine.  Everything else (cloud providers) is scrubbed.
    """
    if not _REDACT_ENABLED:
        return False
    p = (provider or "").lower()
    if p in ("ollama", "hf_local", "llamacpp", "lm-studio", "lmstudio"):
        return False
    b = (base_url or "").lower()
    local_patterns = ("localhost", "127.0.0.1", "::1", "0.0.0.0",
                      ".local", ".internal")
    if any(pat in b for pat in local_patterns):
        return False
    return True


# ---------------------------------------------------------------------------
# Logging integration
# ---------------------------------------------------------------------------

class RedactingFormatter(logging.Formatter):
    """Drop-in log formatter that scrubs secrets from every log record."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_sensitive_text(super().format(record))
