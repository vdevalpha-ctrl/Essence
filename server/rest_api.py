"""
rest_api.py — Essence REST API
================================
Exposes Essence over HTTP/WebSocket so external tools, scripts, and
mobile apps can interact with the running instance.

Endpoints:
  POST   /chat              — send a message, get a streaming response
  GET    /memory            — query gravity memory (recent + search)
  POST   /tool              — execute a tool directly
  GET    /status            — kernel state, token counts, model info
  GET    /goals             — pending goals
  GET    /skills            — registered skills
  WS     /ws/events         — subscribe to bus events via WebSocket

Start via:
  from server.rest_api import start_rest_api
  await start_rest_api(bus, kernel, workspace, port=7860)

Or from the CLI:
  python essence.py tui    (wires automatically when REST_API_PORT is set)

Requires: pip install fastapi uvicorn
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

log = logging.getLogger("essence.rest_api")


async def start_rest_api(
    bus:       Any,
    kernel:    Any,
    workspace: Path,
    port:      int = 7860,
    host:      str = "127.0.0.1",
) -> None:
    """
    Build and launch the FastAPI app.
    Runs until cancelled — call from an asyncio task.
    """
    try:
        from fastapi import FastAPI, WebSocket, HTTPException
        from fastapi.responses import StreamingResponse, JSONResponse
        import uvicorn
    except ImportError:
        log.warning("rest_api: fastapi/uvicorn not installed — REST API disabled")
        log.warning("  Install: pip install fastapi uvicorn")
        return

    app = FastAPI(title="Essence API", version="1.0")

    # ── POST /chat ─────────────────────────────────────────────────

    @app.post("/chat")
    async def chat(body: dict):
        """
        Send a message to Essence.
        Body: {"text": "...", "stream": true}
        Response: plain text stream or JSON {"response": "..."}
        """
        text       = body.get("text", "").strip()
        do_stream  = body.get("stream", True)
        task_id    = body.get("task_id") or uuid.uuid4().hex[:10]

        if not text:
            raise HTTPException(400, detail="text is required")

        from server.event_bus import Envelope

        if not do_stream:
            # Collect full response
            tokens = []
            done_event = asyncio.Event()

            def _handler(env: Any) -> None:
                d = env.data
                if d.get("token"):
                    tokens.append(d["token"])
                if d.get("done"):
                    done_event.set()

            bus.subscribe(f"user.response.{task_id}", _handler)
            bus.subscribe("user.response", _handler)

            bus.publish_sync(Envelope(
                topic="user.request",
                task_id=task_id,
                data={"text": text, "task_id": task_id},
            ))

            try:
                await asyncio.wait_for(done_event.wait(), timeout=120)
            except asyncio.TimeoutError:
                pass
            return JSONResponse({"response": "".join(tokens), "task_id": task_id})

        # Streaming response
        async def _gen() -> AsyncGenerator[bytes, None]:
            queue: asyncio.Queue = asyncio.Queue()

            def _handler(env: Any) -> None:
                d = env.data
                token = d.get("token", "")
                done  = d.get("done", False)
                queue.put_nowait({"token": token, "done": done})

            bus.subscribe("user.response", _handler)

            bus.publish_sync(Envelope(
                topic="user.request",
                task_id=task_id,
                data={"text": text, "task_id": task_id},
            ))

            deadline = time.time() + 120
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if time.time() > deadline:
                        break
                    continue
                token = item.get("token", "")
                if token:
                    yield token.encode("utf-8")
                if item.get("done"):
                    break

        return StreamingResponse(_gen(), media_type="text/plain; charset=utf-8")

    # ── GET /memory ────────────────────────────────────────────────

    @app.get("/memory")
    async def memory_query(q: str = "", n: int = 10):
        """Query gravity memory. Pass q= for search, omit for recent."""
        try:
            from server.kernel import get_kernel
            k = get_kernel()
            ctx = k._memory.build_context_block(n=n, query=q or None)
            return JSONResponse({"context": ctx})
        except Exception as exc:
            raise HTTPException(500, detail=str(exc))

    # ── POST /tool ─────────────────────────────────────────────────

    @app.post("/tool")
    async def run_tool(body: dict):
        """
        Execute a tool directly.
        Body: {"name": "shell", "args": {"command": "ls"}}
        """
        name = body.get("name", "")
        args = body.get("args", {})
        if not name:
            raise HTTPException(400, detail="name is required")
        try:
            from server.tools_engine import execute_tool
            result = await execute_tool(name, args)
            return JSONResponse({"result": result})
        except Exception as exc:
            raise HTTPException(500, detail=str(exc))

    # ── GET /status ────────────────────────────────────────────────

    @app.get("/status")
    async def status():
        """Return kernel state, token counts, model info."""
        try:
            from server.kernel import get_kernel
            k = get_kernel()
            s = k.stats()
            s["model"]    = k._model
            s["provider"] = k._provider
            s["uptime_s"] = round(time.time() - _start_time)
            return JSONResponse(s)
        except Exception as exc:
            return JSONResponse({"error": str(exc)})

    # ── GET /goals ─────────────────────────────────────────────────

    @app.get("/goals")
    async def goals():
        try:
            from server.goal_tracker import get_goal_tracker
            pending = get_goal_tracker().get_pending()
            return JSONResponse([g.to_dict() for g in pending])
        except Exception as exc:
            raise HTTPException(500, detail=str(exc))

    # ── GET /skills ────────────────────────────────────────────────

    @app.get("/skills")
    async def skills():
        try:
            from server.kernel import get_kernel
            catalog = get_kernel()._skills.catalog_for_planner()
            return JSONResponse(catalog)
        except Exception as exc:
            raise HTTPException(500, detail=str(exc))

    # ── WS /ws/events ──────────────────────────────────────────────

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket):
        """Stream all bus events as JSON lines over WebSocket."""
        await websocket.accept()
        queue: asyncio.Queue = asyncio.Queue()

        def _handler(env: Any) -> None:
            try:
                queue.put_nowait({
                    "topic": env.topic,
                    "data":  env.data,
                    "ts":    time.time(),
                })
            except asyncio.QueueFull:
                pass

        bus.subscribe("*", _handler)

        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=30)
                    await websocket.send_text(json.dumps(item))
                except asyncio.TimeoutError:
                    # Send keepalive
                    await websocket.send_text('{"type":"ping"}')
        except Exception:
            pass
        finally:
            pass  # bus.unsubscribe not yet implemented; cleanup on GC

    # ── Launch ─────────────────────────────────────────────────────

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    log.info("REST API starting on http://%s:%d", host, port)
    try:
        await server.serve()
    except asyncio.CancelledError:
        log.info("REST API stopped")


_start_time = time.time()
