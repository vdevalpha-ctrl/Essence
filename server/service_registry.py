"""
service_registry.py — Essence Learned Service Registry
=======================================================
SQLite-backed store for every API / web service Essence has ingested.
Also manages dynamic tool registration so the kernel planner sees learned
endpoints alongside the built-in static tools.

Schema
------
services        — one row per ingested service (JSON blob for profile)

Dynamic tools
-------------
For each ingested endpoint Essence registers a tool with name:
    svc_<service_id>__<method>_<path_slug>
e.g.
    svc_petstore-api__get_v1_pets

The executor makes an authenticated HTTP call to the service using any stored
credentials and returns the response body as a string.

Credential resolution order:
  1. service_registry stored creds  (set via /services auth <id> <key> <val>)
  2. Environment variable  SERVICE_<ID_UPPER>_API_KEY / SERVICE_<ID_UPPER>_TOKEN
  3. No auth (public API)
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

log = logging.getLogger("essence.service_registry")

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "service_registry.db"


# ── helpers ───────────────────────────────────────────────────────────────────

def _path_slug(path: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", path.lower()).strip("_")[:40]


def _tool_name(service_id: str, method: str, path: str) -> str:
    return f"svc_{service_id}__{method.lower()}_{_path_slug(path)}"


# ── registry ──────────────────────────────────────────────────────────────────

class ServiceRegistry:
    """
    Persistent registry of ingested services.
    Each service stores its full ServiceProfile as a JSON blob plus
    an optional credentials dict (encrypted at rest via credstore if available).
    """

    def __init__(self, db_path: Path = _DEFAULT_DB) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = db_path
        self._init_db()
        self._tool_definitions: list[dict] = []   # dynamic additions
        self._tool_executors: dict[str, any] = {}  # name → async callable

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS services (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    base_url    TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    creds_json  TEXT NOT NULL DEFAULT '{}',
                    created_at  REAL NOT NULL,
                    updated_at  REAL NOT NULL
                );
            """)

    # ── CRUD ──────────────────────────────────────────────────────────

    def save(self, profile: "ServiceProfile") -> None:  # noqa: F821
        now = time.time()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO services (id, name, base_url, profile_json, creds_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, '{}', ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name        = excluded.name,
                    base_url    = excluded.base_url,
                    profile_json = excluded.profile_json,
                    updated_at  = excluded.updated_at
            """, (profile.id, profile.name, profile.base_url,
                  json.dumps(profile.to_dict(), ensure_ascii=False), now, now))
        log.debug("ServiceRegistry: saved %s (%s)", profile.id, profile.base_url)

    def get(self, service_id: str) -> "ServiceProfile | None":  # noqa: F821
        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile_json FROM services WHERE id = ?", (service_id,)
            ).fetchone()
        if not row:
            return None
        from server.service_ingestor import ServiceProfile
        return ServiceProfile.from_dict(json.loads(row["profile_json"]))

    def list_all(self) -> list[dict]:
        """Return lightweight summaries (no full endpoint list)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, base_url, created_at, updated_at, profile_json FROM services ORDER BY updated_at DESC"
            ).fetchall()
        result = []
        for row in rows:
            try:
                profile = json.loads(row["profile_json"])
                n_endpoints = len(profile.get("endpoints", []))
            except Exception:
                n_endpoints = 0
            result.append({
                "id":          row["id"],
                "name":        row["name"],
                "base_url":    row["base_url"],
                "endpoints":   n_endpoints,
                "created_at":  row["created_at"],
                "updated_at":  row["updated_at"],
            })
        return result

    def delete(self, service_id: str) -> bool:
        with self._connect() as conn:
            c = conn.execute("DELETE FROM services WHERE id = ?", (service_id,))
        if c.rowcount:
            # Unregister dynamic tools
            self._tool_definitions = [
                t for t in self._tool_definitions
                if not t["function"]["name"].startswith(f"svc_{service_id}__")
            ]
            self._tool_executors = {
                k: v for k, v in self._tool_executors.items()
                if not k.startswith(f"svc_{service_id}__")
            }
        return bool(c.rowcount)

    def find_by_base_url(self, base_url: str) -> "ServiceProfile | None":  # noqa: F821
        """Return the existing profile whose base_url starts with base_url (or vice versa)."""
        with self._connect() as conn:
            rows = conn.execute("SELECT id, profile_json FROM services").fetchall()
        from server.service_ingestor import ServiceProfile
        for row in rows:
            p = ServiceProfile.from_dict(json.loads(row["profile_json"]))
            if p.base_url.rstrip("/") == base_url.rstrip("/"):
                return p
            if base_url.startswith(p.base_url.rstrip("/") + "/"):
                return p
        return None

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Lightweight keyword search across service name, base_url, and endpoint descriptions.
        Returns service summaries most relevant to the query.
        """
        q = query.lower()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, base_url, profile_json FROM services"
            ).fetchall()

        scored: list[tuple[int, dict]] = []
        for row in rows:
            score = 0
            if q in row["name"].lower():
                score += 10
            if q in row["base_url"].lower():
                score += 5
            try:
                profile = json.loads(row["profile_json"])
                for ep in profile.get("endpoints", []):
                    if q in ep.get("description", "").lower() or q in ep.get("path", "").lower():
                        score += 2
            except Exception:
                pass
            if score > 0:
                scored.append((score, {
                    "id":       row["id"],
                    "name":     row["name"],
                    "base_url": row["base_url"],
                    "score":    score,
                }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    # ── credentials ───────────────────────────────────────────────────

    def set_cred(self, service_id: str, key: str, value: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT creds_json FROM services WHERE id = ?", (service_id,)
            ).fetchone()
            if not row:
                return False
            creds = json.loads(row["creds_json"] or "{}")
            creds[key] = value
            conn.execute(
                "UPDATE services SET creds_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(creds), time.time(), service_id),
            )
        # Rebuild executors so the new cred is picked up immediately
        profile = self.get(service_id)
        if profile:
            register_service_tools(profile)
        return True

    def get_creds(self, service_id: str) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT creds_json FROM services WHERE id = ?", (service_id,)
            ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["creds_json"] or "{}")
        except Exception:
            return {}

    def resolve_token(self, service_id: str) -> str:
        """Return a bearer token / api-key for the service, or empty string."""
        creds = self.get_creds(service_id)
        token = (
            creds.get("api_key") or creds.get("token") or creds.get("bearer") or ""
        )
        if not token:
            env_key = f"SERVICE_{service_id.upper().replace('-', '_')}_API_KEY"
            token   = os.environ.get(env_key, "")
        if not token:
            env_key = f"SERVICE_{service_id.upper().replace('-', '_')}_TOKEN"
            token   = os.environ.get(env_key, "")
        return token

    # ── dynamic tool access ───────────────────────────────────────────

    def get_dynamic_tool_definitions(self) -> list[dict]:
        return list(self._tool_definitions)

    def get_dynamic_executor(self, name: str):
        return self._tool_executors.get(name)

    def context_for_task(self, intent: str) -> str:
        """
        Build a short context block listing services relevant to `intent`.
        Injected into the system prompt so the planner knows what APIs exist.
        """
        results = self.search(intent, top_k=3)
        if not results:
            all_svcs = self.list_all()
            if not all_svcs:
                return ""
            results = all_svcs[:3]

        lines = ["## Learned External Services (available as tools)"]
        for svc in results:
            profile = self.get(svc["id"])
            if not profile:
                continue
            lines.append(f"\n### {profile.name}  ({profile.base_url})")
            lines.append(f"Auth: {profile.auth.type}")
            for ep in profile.endpoints[:8]:
                p_names = ", ".join(p.name for p in ep.params if p.required)
                lines.append(f"  {ep.method:<7} {ep.path:<40} {ep.description[:60]}"
                              + (f"  [requires: {p_names}]" if p_names else ""))
            if len(profile.endpoints) > 8:
                lines.append(f"  … and {len(profile.endpoints) - 8} more endpoints")

        return "\n".join(lines)


# ── dynamic tool factory ──────────────────────────────────────────────────────

def register_service_tools(profile: "ServiceProfile") -> None:  # noqa: F821
    """
    Build and register OpenAI-format tool definitions + async executors for
    every endpoint in the profile.  Existing tools for this service are
    replaced.
    """
    reg = get_service_registry()

    # Clear old tools for this service
    reg._tool_definitions = [
        t for t in reg._tool_definitions
        if not t["function"]["name"].startswith(f"svc_{profile.id}__")
    ]
    reg._tool_executors = {
        k: v for k, v in reg._tool_executors.items()
        if not k.startswith(f"svc_{profile.id}__")
    }

    for endpoint in profile.endpoints:
        tool_name = _tool_name(profile.id, endpoint.method, endpoint.path)

        # Build parameter schema
        properties: dict = {}
        required_params: list[str] = []
        for p in endpoint.params:
            properties[p.name] = {
                "type":        p.type or "string",
                "description": p.description or p.name,
            }
            if p.required:
                required_params.append(p.name)

        # Always allow headers override
        properties["_headers"] = {
            "type": "object",
            "description": "Extra HTTP headers to send (optional)",
        }
        properties["_params"] = {
            "type": "object",
            "description": "Extra query parameters (optional)",
        }

        tool_def = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": (
                    f"[{profile.name}] {endpoint.method} {endpoint.path} — "
                    f"{endpoint.description or 'API endpoint'}"
                ),
                "parameters": {
                    "type":       "object",
                    "properties": properties,
                    "required":   required_params,
                },
            },
        }
        reg._tool_definitions.append(tool_def)

        # Build executor closure
        _ep     = endpoint
        _profile = profile

        async def _make_executor(ep=_ep, prof=_profile):
            async def executor(args: dict) -> str:
                return await _call_service_endpoint(prof, ep, args)
            return executor

        import asyncio as _aio
        # We need a sync reference; store the coroutine factory and unwrap at call time
        reg._tool_executors[tool_name] = (_profile, _ep)

    log.info("ServiceRegistry: registered %d tools for service %s",
             len(profile.endpoints), profile.id)

    # Also register executors in tools_engine so execute_tool() can reach them
    try:
        from server.tools_engine import _EXECUTORS
        for tool_name, (prof, ep) in [
            (k, v) for k, v in reg._tool_executors.items()
            if k.startswith(f"svc_{profile.id}__")
        ]:
            # Closure to capture prof and ep
            def _make_exec(p=prof, e=ep):
                async def _exec(args: dict) -> str:
                    return await _call_service_endpoint(p, e, args)
                return _exec
            _EXECUTORS[tool_name] = _make_exec()
    except Exception as exc:
        log.warning("ServiceRegistry: could not register in tools_engine: %s", exc)


async def _call_service_endpoint(profile: "ServiceProfile", endpoint: "Endpoint", args: dict) -> str:  # noqa: F821
    """
    Execute a single HTTP call to a learned service endpoint.
    Handles auth, path params, query params, and JSON body.
    """
    import httpx

    # Resolve auth
    token = get_service_registry().resolve_token(profile.id)
    headers: dict = {}
    if profile.auth.type in ("bearer", "oauth2") and token:
        headers["Authorization"] = f"Bearer {token}"
    elif profile.auth.type == "api_key" and token:
        if profile.auth.param:
            # API key goes in query string
            pass   # added below with _params
        else:
            headers[profile.auth.header or "X-API-Key"] = token
    elif profile.auth.type == "basic" and token:
        headers["Authorization"] = f"Basic {token}"

    headers.update(args.pop("_headers", {}) or {})

    # Build URL — substitute path params
    path = endpoint.path
    query_params: dict  = {}
    body_params: dict   = {}
    extra_params: dict  = args.pop("_params", {}) or {}

    if profile.auth.type == "api_key" and profile.auth.param and token:
        query_params[profile.auth.param] = token

    query_params.update(extra_params)

    for p in endpoint.params:
        val = args.get(p.name)
        if val is None:
            continue
        if p.location == "path":
            path = path.replace(f"{{{p.name}}}", str(val))
        elif p.location == "query":
            query_params[p.name] = val
        elif p.location in ("body", "json"):
            body_params[p.name] = val
        elif p.location == "header":
            headers[p.name] = str(val)

    url = profile.base_url.rstrip("/") + path

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as c:
            if endpoint.method.upper() in ("POST", "PUT", "PATCH") or body_params:
                headers.setdefault("Content-Type", "application/json")
                r = await c.request(
                    endpoint.method.upper(), url,
                    headers=headers,
                    params=query_params,
                    json=body_params or None,
                )
            else:
                r = await c.request(
                    endpoint.method.upper(), url,
                    headers=headers,
                    params=query_params,
                )

        ct = r.headers.get("content-type", "")
        if "json" in ct:
            try:
                data = r.json()
                text = json.dumps(data, indent=2, ensure_ascii=False)
            except Exception:
                text = r.text
        else:
            text = r.text

        if len(text) > 8000:
            text = text[:8000] + "\n[...truncated]"

        return f"HTTP {r.status_code} {url}\n\n{text}"

    except Exception as exc:
        return f"ERROR calling {endpoint.method} {url}: {exc}"


# ── singleton ─────────────────────────────────────────────────────────────────

_registry: ServiceRegistry | None = None


def init_service_registry(db_path: Path | None = None) -> ServiceRegistry:
    global _registry
    _registry = ServiceRegistry(db_path or _DEFAULT_DB)
    # Re-register tools for all persisted services on startup
    for summary in _registry.list_all():
        profile = _registry.get(summary["id"])
        if profile:
            try:
                register_service_tools(profile)
            except Exception as exc:
                log.warning("ServiceRegistry: startup tool registration failed for %s: %s",
                            summary["id"], exc)
    log.info("ServiceRegistry: loaded %d services from %s",
             len(_registry.list_all()), db_path or _DEFAULT_DB)
    return _registry


def get_service_registry() -> ServiceRegistry:
    global _registry
    if _registry is None:
        _registry = ServiceRegistry()
        for summary in _registry.list_all():
            profile = _registry.get(summary["id"])
            if profile:
                try:
                    register_service_tools(profile)
                except Exception:
                    pass
    return _registry
