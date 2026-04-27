"""
event_bus.py — Essence Local Event Bus
========================================
Central in-process pub/sub backbone for all Essence components.

Every cross-component message is an Envelope routed through here.
No component holds a reference to another component's internals —
they only interact via published events on named topics.

Architecture
------------
  - Persistent: every published envelope is written to the EventLog
    (SQLite) before delivery to any subscriber. Law 1 guarantee.
  - Ordered per task_id: events sharing a task_id are routed to a
    single asyncio Queue and consumed sequentially (Law 3).
  - Idempotent: command_id cache (EventLog) short-circuits duplicate
    commands (Law 2).
  - Signed: every envelope carries an HMAC-SHA256 signature over all
    required fields (practical equivalent of Ed25519 for a local-only
    Python system).

Usage
-----
    bus = get_event_bus()

    # Subscribe
    async def my_handler(env: Envelope):
        print(env.topic, env.data)
    bus.subscribe("memory.write", my_handler)

    # Publish
    await bus.publish(Envelope(
        topic="memory.write",
        source_component="agent_core",
        data={"key": "standup_time", "value": "10am"},
        task_id="t-abc123",
        command_id="cmd-xyz",
    ))
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

# ---------------------------------------------------------------------------
# Signing key — generated once per installation, stored in data/
# ---------------------------------------------------------------------------

_WS = Path(__file__).resolve().parent.parent
_KEY_FILE = _WS / "data" / "bus_signing.key"


def _load_or_create_key() -> bytes:
    import stat
    _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _KEY_FILE.exists():
        return bytes.fromhex(_KEY_FILE.read_text().strip())
    key = os.urandom(32)
    _KEY_FILE.write_text(key.hex())
    # Restrict to owner-read/write only (0o600) on Unix;
    # chmod is a no-op on Windows but harmless.
    try:
        _KEY_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass
    return key


_SIGNING_KEY: bytes | None = None


def _get_signing_key() -> bytes:
    global _SIGNING_KEY
    if _SIGNING_KEY is None:
        _SIGNING_KEY = _load_or_create_key()
    return _SIGNING_KEY


def _sign_envelope(fields: list[str]) -> str:
    """HMAC-SHA256 over sorted required fields joined with null bytes."""
    msg = "\x00".join(fields).encode("utf-8")
    return hmac.new(_get_signing_key(), msg, hashlib.sha256).hexdigest()


def _verify_envelope(fields: list[str], signature: str) -> bool:
    expected = _sign_envelope(fields)
    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

def _new_id() -> str:
    ts_hex = format(int(time.time() * 1000), "013x")
    return f"{ts_hex}-{uuid.uuid4().hex}"


@dataclass
class Envelope:
    """Standard event envelope — all fields required per spec §3.2."""
    topic: str
    source_component: str
    data: dict
    # Optional on construction — filled by bus before publish
    event_id: str = field(default_factory=_new_id)
    task_id: str = ""
    command_id: str = field(default_factory=lambda: f"cmd-{uuid.uuid4().hex[:12]}")
    metadata: dict = field(default_factory=dict)
    sequence_number: int = 0     # set by bus
    signature: str = ""           # set by bus

    def _signable_fields(self) -> list[str]:
        return [
            self.event_id,
            self.topic,
            self.source_component,
            str(self.sequence_number),
            self.task_id,
            self.command_id,
            json.dumps(self.data, sort_keys=True),
        ]

    def sign(self) -> "Envelope":
        self.signature = _sign_envelope(self._signable_fields())
        return self

    def verify(self) -> bool:
        if not self.signature:
            return False
        return _verify_envelope(self._signable_fields(), self.signature)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "topic": self.topic,
            "source_component": self.source_component,
            "sequence_number": self.sequence_number,
            "task_id": self.task_id,
            "command_id": self.command_id,
            "data": self.data,
            "metadata": self.metadata,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Envelope":
        env = cls(
            event_id=d.get("event_id", _new_id()),
            topic=d["topic"],
            source_component=d.get("source_component", "unknown"),
            data=d.get("data", {}),
            task_id=d.get("task_id", ""),
            command_id=d.get("command_id", ""),
            metadata=d.get("metadata", {}),
            sequence_number=d.get("sequence_number", 0),
            signature=d.get("signature", ""),
        )
        return env


# Handler type: async callable receiving an Envelope
Handler = Callable[[Envelope], Awaitable[None]]


# ---------------------------------------------------------------------------
# Topic actor — serialises delivery for a single task_id (Law 3)
# ---------------------------------------------------------------------------

class _TaskActor:
    """Single asyncio Queue per task_id; ensures causal ordering."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[Envelope, list[Handler]]] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def ensure_running(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="task-actor")

    async def enqueue(self, env: Envelope, handlers: list[Handler]) -> None:
        self.ensure_running()
        await self._queue.put((env, handlers))

    async def _run(self) -> None:
        while True:
            try:
                env, handlers = await asyncio.wait_for(self._queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                break  # actor idles out after 60s with no events
            for h in handlers:
                try:
                    await h(env)
                except Exception as exc:
                    import logging
                    logging.getLogger("essence.bus").warning(
                        "Handler %s raised on topic %s: %s", h.__name__, env.topic, exc
                    )
            self._queue.task_done()


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------

class EventBus:
    """
    Central in-process event bus.

    - Persists every envelope to EventLog before delivery (Law 1)
    - Routes same-task_id events through a single actor queue (Law 3)
    - Short-circuits duplicate command_ids via cache (Law 2)
    - Signs every envelope before persistence (integrity)
    """

    def __init__(self, event_log: Any) -> None:
        """
        Parameters
        ----------
        event_log : EventLog instance from event_log.py
        """
        self._log = event_log
        self._handlers: dict[str, list[Handler]] = {}
        self._actors: dict[str, _TaskActor] = {}
        self._seq: dict[str, int] = {}  # per-topic sequence numbers

    # ── Subscribe ─────────────────────────────────────────────────────

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Register *handler* to be called for every event on *topic*."""
        self._handlers.setdefault(topic, []).append(handler)

    def subscribe_many(self, topics: list[str], handler: Handler) -> None:
        for t in topics:
            self.subscribe(t, handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        if topic in self._handlers:
            try:
                self._handlers[topic].remove(handler)
            except ValueError:
                pass

    # ── Publish ───────────────────────────────────────────────────────

    async def publish(self, env: Envelope) -> Envelope:
        """
        Validate, sign, persist, and deliver an envelope.

        Returns the envelope with event_id, sequence_number, and signature set.
        Raises ValueError if any required field is missing.
        """
        self._validate(env)

        # Assign monotonic sequence number per topic
        self._seq[env.topic] = self._seq.get(env.topic, 0) + 1
        env.sequence_number = self._seq[env.topic]

        # Sign
        env.sign()

        # Law 1: persist to event log BEFORE delivery
        self._log.emit(
            topic=env.topic,
            data=env.to_dict(),
            source=env.source_component,
            task_id=env.task_id,
            command_id=env.command_id,
            sequence_number=env.sequence_number,
            signature=env.signature,
        )

        # Deliver to handlers
        handlers = list(self._handlers.get(env.topic, []))
        if handlers:
            if env.task_id:
                # Law 3: causal ordering — route through task actor
                actor = self._actors.setdefault(env.task_id, _TaskActor())
                await actor.enqueue(env, handlers)
            else:
                # No task_id → concurrent delivery
                await asyncio.gather(
                    *(h(env) for h in handlers),
                    return_exceptions=True,
                )

        return env

    def publish_sync(self, env: Envelope) -> Envelope:
        """
        Fire-and-forget synchronous wrapper — schedules publish on the
        running event loop or runs it directly if no loop is active.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.publish(env))
            return env
        except RuntimeError:
            asyncio.run(self.publish(env))
            return env

    # ── Idempotency ───────────────────────────────────────────────────

    def get_cached(self, command_id: str) -> dict | None:
        """Return cached result for command_id if not expired."""
        return self._log.get_cached(command_id)

    def cache_result(self, command_id: str, result: dict, ttl_s: float = 300) -> None:
        self._log.cache_result(command_id, result, ttl_s)

    # ── Integrity verification ────────────────────────────────────────

    def verify_log_row(self, row: dict) -> bool:
        """Verify the HMAC signature on a raw log row dict. Returns False on tamper."""
        sig = row.get("signature", "")
        if not sig:
            return False
        try:
            env = Envelope.from_dict(row)
            return env.verify()
        except Exception:
            return False

    def tail_verified(self, n: int = 50, topic: str = "") -> list[dict]:
        """Return recent log rows, emitting a warning for any with invalid signatures."""
        import logging as _logging
        _log = _logging.getLogger("essence.bus.integrity")
        rows = self._log.tail(n=n, topic=topic)
        for row in rows:
            if not self.verify_log_row(row):
                _log.warning(
                    "Integrity violation: event_id=%s topic=%s has invalid signature",
                    row.get("event_id", "?"), row.get("topic", "?"),
                )
        return rows

    # ── Internal ──────────────────────────────────────────────────────

    @staticmethod
    def _validate(env: Envelope) -> None:
        missing = []
        if not env.event_id:
            missing.append("event_id")
        if not env.topic:
            missing.append("topic")
        if not env.source_component:
            missing.append("source_component")
        if not env.command_id:
            missing.append("command_id")
        if env.data is None:
            missing.append("data")
        if missing:
            raise ValueError(f"Envelope missing required fields: {missing}")

    # ── Introspection ─────────────────────────────────────────────────

    def topics(self) -> list[str]:
        return sorted(self._handlers.keys())

    def handler_count(self, topic: str) -> int:
        return len(self._handlers.get(topic, []))


# ---------------------------------------------------------------------------
# Topic registry (spec §3.3)
# ---------------------------------------------------------------------------

TOPICS = {
    "user.request":         "Interaction → Orchestration",
    "user.response":        "Kernel → Interaction (streaming tokens)",
    "skill.execute":        "Skill router → Execution",
    "skill.result":         "Skill runner → Orchestration",
    "skill.health":         "Skill runner → Governance health view",
    "memory.write":         "Agent core / skills → Memory engine",
    "memory.read":          "Agent core → Memory engine",
    "heartbeat.tick":       "Heartbeat scheduler → Action engine",
    "trigger.fired":        "Trigger listeners → Action engine",
    "identity.write":       "Identity engine only → Governance log",
    "governance.audit":     "Audit scanner → Governance log",
    "governance.violation": "Policy engine → Governance log + Interaction",
    # ── New topics ──────────────────────────────────────────────────────
    "agent.message":        "Proactive agent → Interaction (unsolicited messages)",
    "tool.approval":        "Kernel → Interaction (human-in-the-loop tool consent)",
    "tool.approval.reply":  "Interaction → Kernel (user grant/deny decision)",
    "cron.tick":            "Cron scheduler → Action engine (scheduled job fire)",
    "model.switch":         "ModelRouter → Audit (provider/model change event)",
    "config.change":        "IdentityEngine/Settings → Audit (configuration mutation)",
}


# ---------------------------------------------------------------------------
# Tool-approval workflow helpers
# ---------------------------------------------------------------------------

class ToolApprovalTimeout(Exception):
    """Raised when a tool-approval request is not answered within the deadline."""


async def request_tool_approval(
    bus:        "EventBus",
    tool_name:  str,
    args:       dict,
    task_id:    str = "",
    timeout_s:  float = 60.0,
) -> bool:
    """
    Publish a tool.approval event and wait for a tool.approval.reply.

    The UI (TUI or web) should subscribe to tool.approval and publish
    tool.approval.reply with {"granted": true|false}.

    Returns True if the user grants the action, False if denied or timed out.
    """
    import uuid as _uuid
    approval_id = _uuid.uuid4().hex[:10]
    approved_event = asyncio.Event()
    decision: list[bool] = [False]

    async def _on_reply(env: Envelope) -> None:
        if env.data.get("approval_id") == approval_id:
            decision[0] = bool(env.data.get("granted", False))
            approved_event.set()

    bus.subscribe("tool.approval.reply", _on_reply)
    try:
        await bus.publish(Envelope(
            topic="tool.approval",
            source_component="kernel",
            task_id=task_id,
            data={
                "approval_id": approval_id,
                "tool":        tool_name,
                "args":        args,
                "timeout_s":   timeout_s,
            },
        ))
        try:
            await asyncio.wait_for(approved_event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            try:
                from server.audit_logger import get_audit_logger
                get_audit_logger().log_approval(tool_name, args, granted=False, actor="timeout")
            except Exception:
                pass
            return False
        granted = decision[0]
        try:
            from server.audit_logger import get_audit_logger
            get_audit_logger().log_approval(tool_name, args, granted=granted, actor="user")
        except Exception:
            pass
        return granted
    finally:
        bus.unsubscribe("tool.approval.reply", _on_reply)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_bus: EventBus | None = None


def init_event_bus(event_log: Any) -> EventBus:
    """Initialise the global event bus. Call once from application startup."""
    global _bus
    _bus = EventBus(event_log)
    return _bus


def get_event_bus() -> EventBus:
    """Return the global event bus instance."""
    if _bus is None:
        raise RuntimeError(
            "Event bus not initialised. Call init_event_bus(event_log) at startup."
        )
    return _bus
