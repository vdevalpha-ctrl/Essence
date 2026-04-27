"""
title_generator.py — Session Title Auto-Generation
====================================================
Generates a short (3-7 word) descriptive title from the first user/assistant
exchange.  Runs in a background thread after the first response so it never
adds latency to the user-facing reply.

Storage
-------
  Title is stored in EpisodicMemory as an entry with entry_type="session_title".
  One per session — subsequent calls are no-ops if a title already exists.

Integration
-----------
  In kernel._handle_user_request(), after the first response:
    from server.title_generator import maybe_auto_title
    maybe_auto_title(self._episodic, text, response, self._ollama, self._model)
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import httpx

log = logging.getLogger("essence.title_generator")

_TITLE_PROMPT = (
    "Generate a short, descriptive title (3-7 words) for a conversation "
    "that starts with the following exchange. The title should capture the "
    "main topic or intent. Return ONLY the title text, nothing else — no "
    "quotes, no punctuation at the end, no prefixes like 'Title:'."
)

# --- Public helpers -----------------------------------------------------------

def generate_title(
    user_message:      str,
    assistant_response: str,
    ollama_url:        str,
    model:             str,
    timeout:           float = 20.0,
) -> Optional[str]:
    """Generate a session title via a synchronous Ollama call.

    Returns the title string (3-80 chars) or None on any failure.
    This is intentionally synchronous so it can be called from a daemon thread.
    """
    user_snippet      = (user_message or "")[:400]
    assistant_snippet = (assistant_response or "")[:400]

    prompt = (
        f"User: {user_snippet}\n\n"
        f"Assistant: {assistant_snippet}"
    )

    try:
        resp = httpx.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _TITLE_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 32},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = (resp.json().get("message") or {}).get("content", "").strip()

        # Sanitise: remove quotes / common prefixes
        raw = raw.strip('"\'')
        for prefix in ("title:", "Title:", "TITLE:"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):].strip()

        # Clip to reasonable length
        if len(raw) > 80:
            raw = raw[:77] + "…"

        return raw if raw else None

    except Exception as exc:
        log.debug("Title generation failed: %s", exc)
        return None


def get_session_title(episodic) -> Optional[str]:
    """Return the stored title for the current session, or None."""
    if episodic is None:
        return None
    try:
        import sqlite3
        with episodic._connect() as conn:
            row = conn.execute(
                """SELECT content FROM episodes
                   WHERE session_id = ? AND entry_type = 'session_title'
                   ORDER BY ts DESC LIMIT 1""",
                (episodic.session_id,),
            ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _store_title(episodic, title: str) -> None:
    """Write the title into episodic memory as a session_title entry."""
    try:
        episodic.append(
            role="system",
            content=title,
            entry_type="session_title",
            metadata={"auto_generated": True, "generated_at": int(time.time())},
        )
        log.debug("Auto-generated session title stored: %r", title)
    except Exception as exc:
        log.debug("Failed to store auto-generated title: %s", exc)


def maybe_auto_title(
    episodic,
    user_message:      str,
    assistant_response: str,
    ollama_url:        str,
    model:             str,
) -> None:
    """Fire-and-forget title generation after the first exchange.

    No-ops when:
    - episodic is None
    - a title already exists for this session
    - user_message or assistant_response is empty
    - the exchange is not the first one (>2 user turns in current session)
    """
    if not episodic or not user_message or not assistant_response:
        return

    # Only generate on the first 2 user turns
    try:
        import sqlite3
        with episodic._connect() as conn:
            user_count = conn.execute(
                """SELECT COUNT(*) FROM episodes
                   WHERE session_id = ? AND role = 'user' AND entry_type = 'turn'""",
                (episodic.session_id,),
            ).fetchone()[0]
        if user_count > 2:
            return
    except Exception:
        return

    # Skip if title already set
    if get_session_title(episodic) is not None:
        return

    # Run generation in a daemon thread — never blocks the kernel
    def _bg():
        title = generate_title(user_message, assistant_response, ollama_url, model)
        if title:
            _store_title(episodic, title)

    t = threading.Thread(target=_bg, daemon=True, name="auto-title")
    t.start()
