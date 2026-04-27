"""
episodic_memory.py — L2 Episodic Memory
=========================================
Persistent per-session turn store.  Survives restarts.  Feeds context
into every LLM call the kernel makes — conversational AND skill-driven.

Schema
------
  episodes table   — every turn (user / assistant / skill_outcome / plan_summary)
  session_summaries — compacted summary of a completed session (for cross-session recall)

Context injection strategy
---------------------------
  1. Last N turns from the *current* session  (verbatim)
  2. Last M session summaries from *previous* sessions  (one-liner each)
  Both are rendered as a structured context block, not raw message history.

Compaction
----------
  When a session exceeds MAX_RAW_TURNS, the kernel may call compact_session()
  which summarises the oldest half via the LLM and replaces them with a
  single "plan_summary" entry.  The raw turns are archived, not deleted.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

log = logging.getLogger("essence.episodic")

MAX_RAW_TURNS      = 40   # compact when a session exceeds this
COMPACT_KEEP_LAST  = 20   # keep the most-recent N turns verbatim after compaction
CROSS_SESSION_N    = 3    # number of past-session summaries to inject


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS episodes (
    id          TEXT    PRIMARY KEY,
    session_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL,   -- user | assistant | skill_outcome | plan_summary | system
    content     TEXT    NOT NULL,
    entry_type  TEXT    NOT NULL DEFAULT 'turn',  -- turn | skill_outcome | compaction
    metadata    TEXT    NOT NULL DEFAULT '{}',
    ts          INTEGER NOT NULL,
    archived    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_ep_session ON episodes(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_ep_ts      ON episodes(ts DESC);

CREATE TABLE IF NOT EXISTS session_summaries (
    session_id  TEXT PRIMARY KEY,
    summary     TEXT NOT NULL,
    turn_count  INTEGER NOT NULL DEFAULT 0,
    started_at  INTEGER NOT NULL,
    ended_at    INTEGER NOT NULL
);
"""

_MIGRATIONS: list[str] = []


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    SQLite-backed L2 episodic store.

    One instance per Essence session; shared by the kernel and readable
    by the TUI for display.
    """

    def __init__(self, db_path: Path, session_id: str | None = None) -> None:
        self._path       = db_path
        self._lock       = Lock()
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._started_at = int(time.time())
        self._embed_idx  = None   # set by init_embeddings()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_DDL)
            for stmt in _MIGRATIONS:
                try:
                    conn.execute(stmt)
                    conn.commit()
                except sqlite3.OperationalError:
                    pass

    def init_embeddings(
        self,
        ollama_url: str,
        model:      str = "",
    ) -> None:
        """
        Attach a vector index to this episodic store.
        Call once at kernel startup after the Ollama URL is known.
        Idempotent — safe to call multiple times.
        """
        try:
            from server.embeddings import EmbeddingIndex
            cache_path = self._path.parent / "embeddings.json"
            if self._embed_idx is None:
                self._embed_idx = EmbeddingIndex(
                    cache_path=cache_path,
                    ollama_url=ollama_url,
                    model=model or "nomic-embed-text",
                )
            else:
                self._embed_idx.update_config(ollama_url=ollama_url, model=model or "")
            log.debug("Episodic: embedding index attached (model=%s)", model or "nomic-embed-text")
        except Exception as exc:
            log.debug("Episodic: embedding init failed (semantic search disabled): %s", exc)

    # ── Helpers ───────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @property
    def session_id(self) -> str:
        return self._session_id

    # ── Write ─────────────────────────────────────────────────────────

    def append(
        self,
        role:       str,
        content:    str,
        entry_type: str = "turn",       # turn | skill_outcome | plan_summary | compaction
        metadata:   dict | None = None,
        session_id: str | None = None,
    ) -> str:
        """Append one episode entry.  Returns its id."""
        eid        = uuid.uuid4().hex[:16]
        session_id = session_id or self._session_id
        ts         = int(time.time() * 1000)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO episodes
                       (id, session_id, role, content, entry_type, metadata, ts)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (eid, session_id, role, content,
                     entry_type, json.dumps(metadata or {}), ts),
                )
                conn.commit()
        # Queue for semantic embedding (user + assistant turns only — skip metadata entries)
        if self._embed_idx is not None and role in ("user", "assistant") and content:
            try:
                self._embed_idx.queue(eid, content)
            except Exception:
                pass
        return eid

    def append_skill_outcome(
        self,
        intent:  str,
        skill_id: str,
        status:  str,
        summary: str,
        plan_id: str = "",
    ) -> str:
        """
        Write a compact record of a skill execution into episodic memory.
        This is what makes skill results available to future LLM calls.
        """
        content = (
            f"[Skill: {skill_id} | {status}]  Intent: {intent}\n"
            f"Result: {summary}"
        )
        return self.append(
            role="skill_outcome",
            content=content,
            entry_type="skill_outcome",
            metadata={"skill_id": skill_id, "status": status, "plan_id": plan_id},
        )

    def append_plan_summary(self, intent: str, goal: str, success: bool, summary: str) -> str:
        """
        After a multi-step plan completes, write one summary entry that
        condenses what happened.  This is the key record for future sessions.
        """
        status_icon = "✓" if success else "✗"
        content = (
            f"[Plan {status_icon}]  {goal}\n"
            f"Intent: {intent}\n"
            f"Outcome: {summary}"
        )
        return self.append(
            role="plan_summary",
            content=content,
            entry_type="plan_summary",
            metadata={"success": success, "intent": intent},
        )

    # ── Read ──────────────────────────────────────────────────────────

    def get_recent_turns(
        self,
        n: int = 12,
        session_id: str | None = None,
        include_skill_outcomes: bool = True,
    ) -> list[dict]:
        """
        Return the last N turns for the given session as role/content dicts,
        suitable for direct insertion into an Ollama messages list.
        """
        sid   = session_id or self._session_id
        types = "('turn','plan_summary','skill_outcome')" if include_skill_outcomes else "('turn')"
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT role, content, entry_type, ts FROM episodes
                    WHERE session_id = ? AND archived = 0
                      AND entry_type IN {types}
                    ORDER BY ts DESC LIMIT ?""",
                (sid, n),
            ).fetchall()
        # Return in chronological order, mapped to {role, content}
        result = []
        for r in reversed(rows):
            role = r["role"]
            # Normalise skill_outcome / plan_summary → "system" for the messages list
            if role in ("skill_outcome", "plan_summary"):
                role = "system"
            result.append({"role": role, "content": r["content"]})
        return result

    def get_past_session_summaries(self, n: int = CROSS_SESSION_N) -> list[str]:
        """Return one-liner summaries from the last N *completed* sessions."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT summary FROM session_summaries
                   WHERE session_id != ?
                   ORDER BY ended_at DESC LIMIT ?""",
                (self._session_id, n),
            ).fetchall()
        return [r["summary"] for r in rows]

    async def semantic_search(
        self,
        query:      str,
        n:          int = 6,
    ) -> list[dict]:
        """
        Return up to *n* episodic turns most semantically similar to *query*.

        Returns a list of {role, content} dicts, suitable for injecting into
        a system prompt block.  Returns [] if the embedding index is unavailable
        or the model has not been loaded.
        """
        if self._embed_idx is None or not query.strip():
            return []
        try:
            ranked = await self._embed_idx.search(query, n=n)
            if not ranked:
                return []

            ids = [eid for eid, _ in ranked]
            placeholders = ",".join("?" * len(ids))
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT id, role, content FROM episodes WHERE id IN ({placeholders})",
                    ids,
                ).fetchall()

            id_to_row = {r["id"]: r for r in rows}
            result = []
            for eid, score in ranked:
                r = id_to_row.get(eid)
                if r and score > 0.3:   # discard very low-similarity matches
                    result.append({
                        "role":    r["role"] if r["role"] in ("user", "assistant") else "system",
                        "content": r["content"],
                        "score":   round(score, 3),
                    })
            return result
        except Exception as exc:
            log.debug("Episodic.semantic_search failed: %s", exc)
            return []

    def build_context_block(
        self,
        n_turns:          int = 12,
        n_past_sessions:  int = CROSS_SESSION_N,
        include_outcomes: bool = True,
    ) -> str:
        """
        Build the full context string injected into every LLM system prompt.

        Structure
        ---------
          [Past sessions — brief]
          Previous session: <summary>
          ...

          [This session — last N turns]
          User: ...
          Assistant: ...
          [Skill: research | done]  ...
        """
        lines: list[str] = []

        # --- Cross-session recall ---
        past = self.get_past_session_summaries(n_past_sessions)
        if past:
            lines.append("## Prior session context")
            for s in past:
                lines.append(f"  {s}")
            lines.append("")

        # --- Current session turns ---
        turns = self.get_recent_turns(n=n_turns, include_skill_outcomes=include_outcomes)
        if turns:
            lines.append("## Current session")
            for t in turns:
                role_label = {"user": "User", "assistant": "Essence",
                              "system": "Context"}.get(t["role"], t["role"].title())
                # Truncate very long entries
                content = t["content"]
                if len(content) > 600:
                    content = content[:600] + "…"
                lines.append(f"{role_label}: {content}")
            lines.append("")

        return "\n".join(lines)

    def turn_count(self, session_id: str | None = None) -> int:
        sid = session_id or self._session_id
        with self._connect() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE session_id = ? AND archived = 0",
                (sid,),
            ).fetchone()[0]
        return n

    # ── Compaction ────────────────────────────────────────────────────

    async def compact_if_needed(
        self,
        ollama_url: str,
        model:      str,
    ) -> bool:
        """
        If the current session exceeds MAX_RAW_TURNS, compact the oldest
        half into a single plan_summary entry.  Returns True if compacted.
        """
        if self.turn_count() < MAX_RAW_TURNS:
            return False

        await self._compact(ollama_url, model)
        return True

    async def _compact(self, ollama_url: str, model: str) -> None:
        """Summarise the oldest (MAX_RAW_TURNS - COMPACT_KEEP_LAST) turns."""
        sid = self._session_id
        with self._connect() as conn:
            # Get all turns sorted chronologically
            rows = conn.execute(
                """SELECT id, role, content FROM episodes
                   WHERE session_id = ? AND archived = 0 AND entry_type = 'turn'
                   ORDER BY ts ASC""",
                (sid,),
            ).fetchall()

        if not rows:
            return

        # Archive the oldest half
        to_archive = rows[: max(1, len(rows) - COMPACT_KEEP_LAST)]
        archive_ids = [r["id"] for r in to_archive]
        transcript  = "\n".join(
            f"{r['role'].title()}: {r['content'][:300]}"
            for r in to_archive
        )

        # Summarise with LLM
        summary = await self._llm_summarise(transcript, ollama_url, model)

        with self._lock:
            with self._connect() as conn:
                # Archive old turns
                conn.execute(
                    f"UPDATE episodes SET archived = 1 WHERE id IN ({','.join('?'*len(archive_ids))})",
                    archive_ids,
                )
                # Insert compaction entry
                conn.execute(
                    """INSERT INTO episodes
                       (id, session_id, role, content, entry_type, metadata, ts)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        uuid.uuid4().hex[:16], sid,
                        "system", f"[Compacted context]\n{summary}",
                        "compaction", "{}", int(time.time() * 1000),
                    ),
                )
                conn.commit()

        log.debug("Episodic: compacted %d turns for session %s", len(to_archive), sid)

    @staticmethod
    async def _llm_summarise(transcript: str, ollama_url: str, model: str) -> str:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=20.0) as c:
                r = await c.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Summarise this conversation excerpt in 3-5 bullet points. "
                                    "Focus on decisions made, facts established, and tasks completed. "
                                    "Be concrete and brief."
                                ),
                            },
                            {"role": "user", "content": transcript[:4000]},
                        ],
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 256},
                    },
                )
            return r.json().get("message", {}).get("content", transcript[:400]).strip()
        except Exception as exc:
            log.warning("Episodic compact LLM failed: %s", exc)
            return transcript[:400]

    # ── Session end ───────────────────────────────────────────────────

    async def end_session(self, ollama_url: str, model: str) -> str:
        """
        Called when the TUI closes.  Generates a cross-session summary
        so the *next* session can recall what happened today.
        """
        turns = self.get_recent_turns(n=30, include_skill_outcomes=True)
        if not turns:
            return ""

        transcript = "\n".join(
            f"{t['role'].title()}: {t['content'][:200]}"
            for t in turns
        )
        try:
            import httpx
            async with httpx.AsyncClient(timeout=20.0) as c:
                r = await c.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Write a single sentence (max 25 words) summarising "
                                    "what the user accomplished in this session. "
                                    "Start with a verb. Example: 'Researched quantum computing, "
                                    "wrote a summary doc, and set a reminder for Monday.'"
                                ),
                            },
                            {"role": "user", "content": transcript[:3000]},
                        ],
                        "stream": False,
                        "options": {"temperature": 0.2, "num_predict": 60},
                    },
                )
            summary = r.json().get("message", {}).get("content", "").strip()
        except Exception:
            # Fallback: first user message
            first_user = next((t["content"][:80] for t in turns if t["role"] == "user"), "")
            summary    = f"Session started with: {first_user}" if first_user else "Session (no summary)"

        # Persist
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO session_summaries
                       (session_id, summary, turn_count, started_at, ended_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        self._session_id, summary,
                        self.turn_count(),
                        self._started_at,
                        int(time.time()),
                    ),
                )
                conn.commit()

        log.info("Episodic: session %s ended — %r", self._session_id, summary)
        return summary

    # ── Procedural memory (L4) ────────────────────────────────────────
    # Stored alongside episodic so both survive restarts.

    def save_plan(self, intent_key: str, plan_dict: dict) -> None:
        """Persist a successful plan for Pass-1 procedural memory recall."""
        with self._lock:
            with self._connect() as conn:
                # Re-use episodes table with a special session_id namespace
                conn.execute(
                    """INSERT OR REPLACE INTO episodes
                       (id, session_id, role, content, entry_type, metadata, ts)
                       VALUES (?, 'procedural', 'plan', ?, 'procedural', '{}', ?)""",
                    (
                        intent_key,
                        json.dumps(plan_dict, ensure_ascii=False),
                        int(time.time() * 1000),
                    ),
                )
                conn.commit()

    def load_plans(self) -> dict[str, dict]:
        """Load all persisted procedural plans on startup."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, content FROM episodes WHERE session_id = 'procedural'"
            ).fetchall()
        plans = {}
        for r in rows:
            try:
                plans[r["id"]] = json.loads(r["content"])
            except json.JSONDecodeError:
                pass
        return plans


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "episodic.db"
_episodic: EpisodicMemory | None = None


def init_episodic_memory(
    session_id: str | None = None,
    db_path: Path | None = None,
) -> EpisodicMemory:
    global _episodic
    _episodic = EpisodicMemory(db_path or _DEFAULT_DB, session_id)
    return _episodic


def get_episodic_memory() -> EpisodicMemory:
    global _episodic
    if _episodic is None:
        _episodic = EpisodicMemory(_DEFAULT_DB)
    return _episodic
