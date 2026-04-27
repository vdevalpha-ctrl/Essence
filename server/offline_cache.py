"""
offline_cache.py — Essence Offline Fallback Layer
===================================================
Provides resilience when the LLM backend (Ollama / remote provider) is
temporarily unreachable:

  • Session cache   — last N user/assistant turn pairs, written to disk
  • Config cache    — snapshot of agent config and tool definitions
  • Message queue   — outbound messages buffered while backend is down
  • Health probe    — lightweight async check for backend availability
  • Auto-flush      — on reconnect, drains the queue and replays to kernel

Usage
-----
    cache = get_offline_cache()

    # Queue a message when backend is down
    msg_id = cache.enqueue_message({"role": "user", "text": "hello"})

    # Flush on reconnect
    pending = cache.flush_queue()
    for item in pending:
        await kernel.handle(item["payload"])

    # Save / restore recent turns
    cache.save_turn("t-abc", "user", "What time is it?")
    cache.save_turn("t-abc", "assistant", "It is 10am.")
    turns = cache.get_session("t-abc")

    # Snapshot agent config
    cache.save_config({"model": "llama3", "persona": "concise"})
    cfg = cache.load_config()
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Awaitable

log = logging.getLogger("essence.offline_cache")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SESSION_TURNS  = 100      # turns kept per task_id
MAX_QUEUE_SIZE     = 500      # max buffered messages
PROBE_INTERVAL_S   = 15.0     # seconds between backend health probes
PROBE_TIMEOUT_S    = 3.0      # HTTP/socket probe timeout


# ---------------------------------------------------------------------------
# Session cache (file-backed)
# ---------------------------------------------------------------------------

class SessionCache:
    """
    Persists recent conversation turns to JSONL files under cache_dir/sessions/.
    One file per task_id; rotates at MAX_SESSION_TURNS.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir / "sessions"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _path(self, task_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)
        return self._dir / f"{safe}.jsonl"

    def save_turn(self, task_id: str, role: str, text: str, metadata: dict | None = None) -> None:
        record = {
            "ts":       int(time.time() * 1000),
            "task_id":  task_id,
            "role":     role,
            "text":     text,
            "metadata": metadata or {},
        }
        with self._lock:
            path = self._path(task_id)
            lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
            lines.append(json.dumps(record, ensure_ascii=False))
            # Rotate
            if len(lines) > MAX_SESSION_TURNS:
                lines = lines[-MAX_SESSION_TURNS:]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def get_session(self, task_id: str) -> list[dict]:
        path = self._path(task_id)
        if not path.exists():
            return []
        with self._lock:
            lines = path.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out

    def list_sessions(self) -> list[str]:
        return [p.stem for p in sorted(self._dir.glob("*.jsonl"))]

    def delete_session(self, task_id: str) -> bool:
        path = self._path(task_id)
        if path.exists():
            path.unlink()
            return True
        return False


# ---------------------------------------------------------------------------
# Config cache (single JSON file)
# ---------------------------------------------------------------------------

class ConfigCache:
    """Snapshots agent config and tool definitions to disk."""

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._config_path = self._dir / "agent_config.json"
        self._tools_path  = self._dir / "tool_definitions.json"
        self._lock = Lock()

    def save_config(self, config: dict) -> None:
        with self._lock:
            self._config_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    def load_config(self) -> dict:
        with self._lock:
            if not self._config_path.exists():
                return {}
            try:
                return json.loads(self._config_path.read_text(encoding="utf-8"))
            except Exception:
                return {}

    def save_tool_definitions(self, tools: list[dict]) -> None:
        with self._lock:
            self._tools_path.write_text(
                json.dumps(tools, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    def load_tool_definitions(self) -> list[dict]:
        with self._lock:
            if not self._tools_path.exists():
                return []
            try:
                return json.loads(self._tools_path.read_text(encoding="utf-8"))
            except Exception:
                return []


# ---------------------------------------------------------------------------
# Message queue (file-backed JSONL)
# ---------------------------------------------------------------------------

class MessageQueue:
    """
    Persists outbound messages that could not be delivered because the
    backend was unreachable.  Each entry is a JSONL row with a unique id,
    enqueue timestamp, and the original payload.

    flush_queue() returns and removes all pending items in FIFO order.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._path = cache_dir / "message_queue.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def enqueue(self, payload: dict) -> str:
        msg_id = uuid.uuid4().hex
        record = {
            "id":         msg_id,
            "enqueued_at": int(time.time() * 1000),
            "payload":    payload,
        }
        with self._lock:
            if self._queue_size() >= MAX_QUEUE_SIZE:
                log.warning("MessageQueue: queue full (%d), dropping oldest message", MAX_QUEUE_SIZE)
                self._drop_oldest()
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.debug("MessageQueue: enqueued %s", msg_id)
        return msg_id

    def flush(self) -> list[dict]:
        """Return all queued messages and clear the queue."""
        with self._lock:
            if not self._path.exists():
                return []
            lines = self._path.read_text(encoding="utf-8").splitlines()
            self._path.unlink(missing_ok=True)
        items = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        log.info("MessageQueue: flushed %d messages", len(items))
        return items

    def peek(self, n: int = 10) -> list[dict]:
        """Return up to n messages without removing them."""
        with self._lock:
            if not self._path.exists():
                return []
            lines = self._path.read_text(encoding="utf-8").splitlines()
        items = []
        for line in lines[:n]:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return items

    def size(self) -> int:
        with self._lock:
            return self._queue_size()

    def _queue_size(self) -> int:
        if not self._path.exists():
            return 0
        return sum(1 for line in self._path.open(encoding="utf-8") if line.strip())

    def _drop_oldest(self) -> None:
        if not self._path.exists():
            return
        lines = self._path.read_text(encoding="utf-8").splitlines()
        lines = [l for l in lines if l.strip()][1:]  # drop first
        self._path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Backend health probe
# ---------------------------------------------------------------------------

class BackendProbe:
    """
    Async health probe for the configured LLM backend.
    Runs on a background task; notifies callbacks on state change.
    """

    def __init__(
        self,
        base_url:    str,
        interval_s:  float = PROBE_INTERVAL_S,
        timeout_s:   float = PROBE_TIMEOUT_S,
    ) -> None:
        self._url       = base_url.rstrip("/") + "/api/tags"  # Ollama health endpoint
        self._interval  = interval_s
        self._timeout   = timeout_s
        self._online:   bool | None = None   # None = unknown
        self._task:     asyncio.Task | None = None
        self._on_up:    list[Callable[[], Awaitable[None]]] = []
        self._on_down:  list[Callable[[], Awaitable[None]]] = []

    @property
    def is_online(self) -> bool | None:
        return self._online

    def on_up(self, fn: Callable[[], Awaitable[None]]) -> None:
        self._on_up.append(fn)

    def on_down(self, fn: Callable[[], Awaitable[None]]) -> None:
        self._on_down.append(fn)

    def start(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._loop(), name="backend-probe")
        except RuntimeError:
            log.warning("BackendProbe.start() called outside asyncio loop")

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def probe_once(self) -> bool:
        loop = asyncio.get_event_loop()
        try:
            import urllib.request
            req = urllib.request.Request(self._url, method="GET")
            # Run blocking I/O in a thread so the event loop stays responsive
            await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=self._timeout),
            )
            return True
        except Exception:
            return False

    async def _loop(self) -> None:
        while True:
            try:
                was_online = self._online
                now_online = await self.probe_once()
                self._online = now_online

                if was_online is not True and now_online:
                    log.info("BackendProbe: backend came online (%s)", self._url)
                    for cb in self._on_up:
                        try:
                            await cb()
                        except Exception as exc:
                            log.warning("on_up callback error: %s", exc)

                elif was_online is not False and not now_online:
                    log.warning("BackendProbe: backend went offline (%s)", self._url)
                    for cb in self._on_down:
                        try:
                            await cb()
                        except Exception as exc:
                            log.warning("on_down callback error: %s", exc)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("BackendProbe loop error: %s", exc)

            await asyncio.sleep(self._interval)


# ---------------------------------------------------------------------------
# OfflineCache — top-level facade
# ---------------------------------------------------------------------------

class OfflineCache:
    """
    Unified facade for all offline-resilience subsystems.

    Typical integration in kernel._stream_response():

        if not self._offline_cache.probe.is_online:
            msg_id = self._offline_cache.enqueue_message({"text": user_text, ...})
            yield "[queued — backend offline, will retry when reconnected]"
            return
    """

    def __init__(self, cache_dir: Path, backend_url: str = "http://localhost:11434") -> None:
        self._dir    = Path(cache_dir)
        self.sessions = SessionCache(self._dir)
        self.config   = ConfigCache(self._dir)
        self.queue    = MessageQueue(self._dir)
        self.probe    = BackendProbe(backend_url)
        self._flush_callback: Callable[[list[dict]], Awaitable[None]] | None = None

    # ── Delegated shortcuts ───────────────────────────────────────────

    def save_turn(self, task_id: str, role: str, text: str, metadata: dict | None = None) -> None:
        self.sessions.save_turn(task_id, role, text, metadata)

    def get_session(self, task_id: str) -> list[dict]:
        return self.sessions.get_session(task_id)

    def save_config(self, config: dict) -> None:
        self.config.save_config(config)

    def load_config(self) -> dict:
        return self.config.load_config()

    def save_tool_definitions(self, tools: list[dict]) -> None:
        self.config.save_tool_definitions(tools)

    def load_tool_definitions(self) -> list[dict]:
        return self.config.load_tool_definitions()

    def enqueue_message(self, payload: dict) -> str:
        return self.queue.enqueue(payload)

    def flush_queue(self) -> list[dict]:
        return self.queue.flush()

    def queue_size(self) -> int:
        return self.queue.size()

    # ── Reconnect handling ────────────────────────────────────────────

    def set_flush_callback(self, fn: Callable[[list[dict]], Awaitable[None]]) -> None:
        """Register an async callback invoked with queued messages when backend reconnects."""
        self._flush_callback = fn
        self.probe.on_up(self._on_reconnect)

    async def _on_reconnect(self) -> None:
        pending = self.flush_queue()
        if not pending:
            return
        log.info("OfflineCache: replaying %d queued messages after reconnect", len(pending))
        if self._flush_callback:
            try:
                await self._flush_callback(pending)
            except Exception as exc:
                log.error("OfflineCache: flush callback error: %s", exc)
                # Re-queue on failure
                for item in pending:
                    self.queue.enqueue(item["payload"])

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        self.probe.start()

    def stop(self) -> None:
        self.probe.stop()

    # ── Diagnostics ──────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "backend_online": self.probe.is_online,
            "queue_size":     self.queue.size(),
            "sessions":       self.sessions.list_sessions(),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "offline_cache"
_cache: OfflineCache | None = None


def init_offline_cache(
    cache_dir:   Path | None = None,
    backend_url: str = "http://localhost:11434",
) -> OfflineCache:
    global _cache
    _cache = OfflineCache(cache_dir or _DEFAULT_CACHE_DIR, backend_url)
    return _cache


def get_offline_cache() -> OfflineCache:
    global _cache
    if _cache is None:
        _cache = OfflineCache(_DEFAULT_CACHE_DIR)
    return _cache
