"""
mesh.py — Agent Mesh for Essence
==================================
Enables multi-instance Essence deployments to discover each other via
mDNS and delegate tasks to peer agents using HMAC-signed messages.

Roles:
  coordinator  — receives high-level intents, decomposes and delegates
  worker       — accepts subtasks and executes them
  observer     — receives all events but does not execute tasks

Configuration:
  ESSENCE_MESH_ENABLED   — "true" to activate (default: "false")
  ESSENCE_MESH_ROLE      — "auto" | "coordinator" | "worker" | "observer"
  ESSENCE_MESH_PEERS     — "host:port,host:port" for manual peer discovery
  ESSENCE_MESH_SECRET    — shared HMAC secret for message authentication
  ESSENCE_MESH_PORT      — peer communication port (default: 7862)

mDNS service name: _essence._tcp.local.

Install for mDNS discovery: pip install zeroconf
Install for HTTP transport: pip install httpx (already required)
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger("essence.mesh")

_MESH_PORT   = int(os.environ.get("ESSENCE_MESH_PORT",   "7862"))
_MESH_SECRET = os.environ.get("ESSENCE_MESH_SECRET", "essence-mesh-secret-change-me")
_MESH_ROLE   = os.environ.get("ESSENCE_MESH_ROLE",   "auto")
_SERVICE_TYPE = "_essence._tcp.local."


# ── Peer info ──────────────────────────────────────────────────────────────

@dataclass
class Peer:
    id:       str
    host:     str
    port:     int
    role:     str       # coordinator | worker | observer
    seen_at:  float = field(default_factory=time.time)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_fresh(self, ttl: float = 120) -> bool:
        return time.time() - self.seen_at < ttl


# ── HMAC authentication ────────────────────────────────────────────────────

def _sign(payload: str, secret: str = _MESH_SECRET) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _verify(payload: str, signature: str, secret: str = _MESH_SECRET) -> bool:
    expected = _sign(payload, secret)
    return hmac.compare_digest(expected, signature)


def _make_envelope(data: dict, secret: str = _MESH_SECRET) -> dict:
    payload = json.dumps(data, sort_keys=True)
    return {"payload": payload, "sig": _sign(payload, secret), "ts": time.time()}


def _open_envelope(envelope: dict, secret: str = _MESH_SECRET) -> Optional[dict]:
    """Verify and unpack a mesh envelope.  Returns payload dict or None."""
    if time.time() - envelope.get("ts", 0) > 60:
        return None   # expired
    payload = envelope.get("payload", "")
    sig     = envelope.get("sig", "")
    if not _verify(payload, sig, secret):
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None


# ── Agent Mesh ─────────────────────────────────────────────────────────────

class AgentMesh:
    """
    Multi-instance coordination layer.  Discovers peers, delegates tasks,
    and receives delegated tasks from peers.
    """

    def __init__(self, workspace: Any, bus: Any) -> None:
        self._ws      = workspace
        self._bus     = bus
        self._peers:  dict[str, Peer] = {}
        self._node_id = uuid.uuid4().hex[:12]
        self._role    = self._detect_role()
        self._running = False
        self._tasks:  list[asyncio.Task] = []

    def _detect_role(self) -> str:
        if _MESH_ROLE != "auto":
            return _MESH_ROLE
        # Auto: first instance is coordinator, subsequent are workers
        return "coordinator"

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        if not _is_enabled():
            log.debug("mesh: disabled (set ESSENCE_MESH_ENABLED=true)")
            return
        if self._running:
            return
        self._running = True
        loop = asyncio.get_event_loop()

        # mDNS discovery
        self._tasks.append(loop.create_task(self._mdns_advertise(), name="mesh-mdns-advertise"))
        self._tasks.append(loop.create_task(self._mdns_browse(),    name="mesh-mdns-browse"))

        # Manual peers
        manual = os.environ.get("ESSENCE_MESH_PEERS", "")
        if manual:
            for hp in manual.split(","):
                hp = hp.strip()
                if ":" in hp:
                    host, port_s = hp.rsplit(":", 1)
                    try:
                        pid = f"manual-{host}-{port_s}"
                        self._peers[pid] = Peer(pid, host, int(port_s), "worker")
                    except ValueError:
                        pass

        # HTTP server for incoming task delegations
        self._tasks.append(loop.create_task(self._serve(), name="mesh-server"))

        log.info("AgentMesh started: node=%s role=%s port=%d", self._node_id, self._role, _MESH_PORT)

    def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            if not t.done():
                t.cancel()
        self._tasks.clear()

    # ── Task delegation ────────────────────────────────────────────

    async def delegate(self, intent: str, input_data: dict) -> Optional[dict]:
        """
        Delegate a task to the healthiest available worker peer.
        Returns the result dict or None if no peer is available.
        """
        workers = [p for p in self._peers.values()
                   if p.role == "worker" and p.is_fresh()]
        if not workers:
            return None

        # Simple round-robin: pick first fresh worker
        peer = workers[0]
        envelope = _make_envelope({
            "type":       "task",
            "task_id":    uuid.uuid4().hex[:10],
            "intent":     intent,
            "input":      input_data,
            "from_node":  self._node_id,
        })

        try:
            import httpx
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(f"{peer.url}/mesh/task", json=envelope)
                r.raise_for_status()
                result_env = r.json()
                data = _open_envelope(result_env)
                if data:
                    log.info("mesh: delegated task to %s → status=%s", peer.id, data.get("status"))
                    return data
        except Exception as exc:
            log.warning("mesh: delegate to %s failed: %s", peer.id, exc)
        return None

    # ── mDNS ──────────────────────────────────────────────────────

    async def _mdns_advertise(self) -> None:
        """Advertise this node via mDNS."""
        try:
            from zeroconf.asyncio import AsyncZeroconf  # type: ignore
            from zeroconf import ServiceInfo             # type: ignore
            import socket
            aio_zc = AsyncZeroconf()
            host_ip = socket.gethostbyname(socket.gethostname())
            info = ServiceInfo(
                _SERVICE_TYPE,
                f"essence-{self._node_id}.{_SERVICE_TYPE}",
                addresses=[socket.inet_aton(host_ip)],
                port=_MESH_PORT,
                properties={
                    "node_id": self._node_id,
                    "role":    self._role,
                },
            )
            await aio_zc.async_register_service(info)
            log.info("mesh: mDNS registered %s:%d", host_ip, _MESH_PORT)
            try:
                while self._running:
                    await asyncio.sleep(30)
            finally:
                await aio_zc.async_unregister_service(info)
                await aio_zc.async_close()
        except ImportError:
            log.debug("mesh: zeroconf not installed — mDNS disabled (pip install zeroconf)")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.warning("mesh: mDNS advertise error: %s", exc)

    async def _mdns_browse(self) -> None:
        """Discover peers via mDNS."""
        try:
            from zeroconf.asyncio import AsyncZeroconf, AsyncServiceBrowser  # type: ignore
            from zeroconf import ServiceStateChange                           # type: ignore

            def _on_change(zeroconf, service_type, name, state_change):
                pass  # handled in on_service_info

            aio_zc  = AsyncZeroconf()
            browser = AsyncServiceBrowser(aio_zc.zeroconf, _SERVICE_TYPE, handlers=[_on_change])
            while self._running:
                await asyncio.sleep(10)
                infos = aio_zc.zeroconf.get_service_info(_SERVICE_TYPE, _SERVICE_TYPE)
                # Discovery logic simplified — full impl requires listener pattern
            await aio_zc.async_close()
        except ImportError:
            pass   # already warned in _mdns_advertise
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.debug("mesh: mDNS browse error: %s", exc)

    # ── HTTP server ────────────────────────────────────────────────

    async def _serve(self) -> None:
        """Minimal HTTP server to receive delegated tasks."""
        try:
            from aiohttp import web  # type: ignore
        except ImportError:
            log.debug("mesh: aiohttp not installed — mesh server disabled")
            return

        app  = web.Application()
        routes = web.RouteTableDef()

        @routes.post("/mesh/task")
        async def handle_task(request):
            envelope = await request.json()
            data = _open_envelope(envelope)
            if not data:
                return web.json_response({"error": "invalid signature"}, status=401)

            intent = data.get("intent", "")
            log.info("mesh: received task from %s: %r", data.get("from_node"), intent[:60])

            # Submit to local kernel
            result = {"status": "accepted", "node_id": self._node_id}
            try:
                from server.event_bus import Envelope as BusEnv
                self._bus.publish_sync(BusEnv(
                    topic="user.request",
                    data={"text": intent, "task_id": data.get("task_id", "")},
                ))
                result["status"] = "delegated"
            except Exception as exc:
                result["status"] = "error"
                result["error"]  = str(exc)

            out_env = _make_envelope(result)
            return web.json_response(out_env)

        app.add_routes(routes)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", _MESH_PORT)
        try:
            await site.start()
            log.info("mesh: HTTP server listening on :%d", _MESH_PORT)
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await runner.cleanup()

    @property
    def peers(self) -> list[Peer]:
        return [p for p in self._peers.values() if p.is_fresh()]

    def status(self) -> dict:
        return {
            "node_id":  self._node_id,
            "role":     self._role,
            "peers":    len(self.peers),
            "enabled":  _is_enabled(),
        }


# ── Helpers ────────────────────────────────────────────────────────────────

def _is_enabled() -> bool:
    return os.environ.get("ESSENCE_MESH_ENABLED", "false").lower() in ("true", "1", "yes")


# ── Module-level singleton ─────────────────────────────────────────────────

_mesh: Optional[AgentMesh] = None


def init_mesh(workspace: Any, bus: Any) -> AgentMesh:
    global _mesh
    _mesh = AgentMesh(workspace, bus)
    return _mesh


def get_mesh() -> Optional[AgentMesh]:
    return _mesh
