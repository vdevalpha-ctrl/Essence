"""
service_ingestor.py — Essence Dynamic Service Ingestion
========================================================
Allows Essence to learn new external APIs and web services at runtime, purely
from a URL dropped in chat or via /ingest.

Ingestion pipeline
------------------
1.  Fetch the URL (or try common suffixes: /openapi.json, /swagger.json, /docs)
2.  Detect schema type:
      • OpenAPI 3.x / Swagger 2.x  (JSON/YAML)
      • GraphQL introspection
      • HTML documentation page
      • Plain JSON with discernible endpoints
      • Generic base URL (probe well-known paths)
3.  Extract a normalised ServiceProfile:
      base_url, auth scheme, list of Endpoint(method, path, params, description)
4.  For HTML/prose docs: use LLM to extract structure from text
5.  Persist to ServiceRegistry (service_registry.py)
6.  Register dynamic callable tools in tools_engine

Usage (from kernel or TUI)
---------------------------
    from server.service_ingestor import ServiceIngestor, get_ingestor
    ingestor = get_ingestor()
    profile  = await ingestor.ingest("https://api.example.com/openapi.json")
    # profile.endpoints now available as tools
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("essence.service_ingestor")

# ── constants ─────────────────────────────────────────────────────────────────
MAX_HTML_CHARS   = 12_000   # chars sent to LLM for doc-page parsing
MAX_JSON_CHARS   = 40_000   # cap on raw JSON before we truncate
PROBE_SUFFIXES   = [
    "/openapi.json", "/swagger.json", "/api-docs",
    "/api/openapi.json", "/v1/openapi.json", "/docs/openapi.json",
]
FETCH_TIMEOUT    = 15.0
LLM_PARSE_PROMPT = """You are an API analyst. Extract a structured summary from the text below.
Return ONLY valid JSON — no markdown, no commentary — in this exact schema:
{
  "name": "short service name",
  "description": "one sentence",
  "base_url": "https://...",
  "auth": {"type": "none|bearer|api_key|basic|oauth2", "header": "Authorization", "param": ""},
  "endpoints": [
    {
      "method": "GET|POST|PUT|DELETE|PATCH",
      "path": "/path",
      "description": "what it does",
      "params": [{"name":"p","in":"query|body|path","required":false,"description":""}]
    }
  ]
}
If you cannot determine something, use null. Include up to 30 endpoints.
Text:
"""


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class EndpointParam:
    name:        str
    location:    str = "query"    # query | body | path | header
    required:    bool = False
    description: str  = ""
    type:        str  = "string"


@dataclass
class Endpoint:
    method:      str
    path:        str
    description: str = ""
    params:      list[EndpointParam] = field(default_factory=list)
    response:    str = ""


@dataclass
class AuthScheme:
    type:   str = "none"     # none | bearer | api_key | basic | oauth2
    header: str = "Authorization"
    param:  str = ""         # query-param name for api_key in query


@dataclass
class ServiceProfile:
    id:          str
    name:        str
    base_url:    str
    description: str = ""
    auth:        AuthScheme = field(default_factory=AuthScheme)
    endpoints:   list[Endpoint] = field(default_factory=list)
    source_url:  str = ""
    ingested_at: float = field(default_factory=time.time)
    raw_schema:  dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "name":        self.name,
            "base_url":    self.base_url,
            "description": self.description,
            "auth":        {
                "type":   self.auth.type,
                "header": self.auth.header,
                "param":  self.auth.param,
            },
            "endpoints": [
                {
                    "method":      e.method,
                    "path":        e.path,
                    "description": e.description,
                    "response":    e.response,
                    "params": [
                        {"name": p.name, "in": p.location,
                         "required": p.required, "description": p.description,
                         "type": p.type}
                        for p in e.params
                    ],
                }
                for e in self.endpoints
            ],
            "source_url":  self.source_url,
            "ingested_at": self.ingested_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ServiceProfile":
        auth = AuthScheme(
            type=d.get("auth", {}).get("type", "none"),
            header=d.get("auth", {}).get("header", "Authorization"),
            param=d.get("auth", {}).get("param", ""),
        )
        endpoints = [
            Endpoint(
                method=e.get("method", "GET"),
                path=e.get("path", "/"),
                description=e.get("description", ""),
                response=e.get("response", ""),
                params=[
                    EndpointParam(
                        name=p.get("name", ""),
                        location=p.get("in", "query"),
                        required=bool(p.get("required", False)),
                        description=p.get("description", ""),
                        type=p.get("type", "string"),
                    )
                    for p in e.get("params", [])
                ],
            )
            for e in d.get("endpoints", [])
        ]
        return cls(
            id=d.get("id", uuid.uuid4().hex[:8]),
            name=d.get("name", "unknown"),
            base_url=d.get("base_url", ""),
            description=d.get("description", ""),
            auth=auth,
            endpoints=endpoints,
            source_url=d.get("source_url", ""),
            ingested_at=d.get("ingested_at", time.time()),
        )


# ── parser helpers ────────────────────────────────────────────────────────────

def _make_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:32] or uuid.uuid4().hex[:8]


def _parse_openapi(data: dict, source_url: str) -> ServiceProfile:
    """Parse OpenAPI 3.x or Swagger 2.x dict into a ServiceProfile."""
    info      = data.get("info", {})
    name      = info.get("title", "API")
    desc      = info.get("description", "")
    version   = info.get("version", "")

    # Determine base URL
    servers = data.get("servers", [])
    if servers:
        base_url = servers[0].get("url", "").rstrip("/")
    elif "host" in data:
        scheme   = (data.get("schemes") or ["https"])[0]
        basepath = data.get("basePath", "")
        base_url = f"{scheme}://{data['host']}{basepath}".rstrip("/")
    else:
        base_url = source_url.rsplit("/openapi", 1)[0].rsplit("/swagger", 1)[0]

    # Auth
    security_schemes = (
        data.get("components", {}).get("securitySchemes", {})
        or data.get("securityDefinitions", {})
    )
    auth = AuthScheme()
    for _k, v in security_schemes.items():
        t = v.get("type", "").lower()
        if t in ("http",):
            scheme_type = v.get("scheme", "bearer").lower()
            auth = AuthScheme(type=scheme_type if scheme_type == "basic" else "bearer")
            break
        elif t == "apikey":
            auth = AuthScheme(type="api_key", header=v.get("name", "X-API-Key"),
                              param=v.get("name", "") if v.get("in") == "query" else "")
            break
        elif t == "oauth2":
            auth = AuthScheme(type="oauth2")
            break

    # Endpoints
    endpoints: list[Endpoint] = []
    paths = data.get("paths", {})
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method.upper() not in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
                continue
            if not isinstance(op, dict):
                continue
            op_desc = (op.get("summary") or op.get("description") or "").strip()
            params  = []
            for p in op.get("parameters", []):
                params.append(EndpointParam(
                    name=p.get("name", ""),
                    location=p.get("in", "query"),
                    required=bool(p.get("required", False)),
                    description=p.get("description", ""),
                    type=p.get("schema", {}).get("type", "") if "schema" in p else p.get("type", "string"),
                ))
            # Body params from requestBody
            rb = op.get("requestBody", {})
            if rb:
                content = rb.get("content", {})
                for ct, ct_data in content.items():
                    schema = ct_data.get("schema", {})
                    for prop, prop_data in schema.get("properties", {}).items():
                        params.append(EndpointParam(
                            name=prop, location="body",
                            required=prop in schema.get("required", []),
                            description=prop_data.get("description", ""),
                            type=prop_data.get("type", "string"),
                        ))
                    break  # first content type is enough
            endpoints.append(Endpoint(
                method=method.upper(),
                path=path,
                description=op_desc,
                params=params,
            ))

    return ServiceProfile(
        id=_make_id(name),
        name=f"{name} {version}".strip(),
        base_url=base_url,
        description=desc,
        auth=auth,
        endpoints=endpoints,
        source_url=source_url,
        raw_schema=data,
    )


def _parse_graphql_introspection(data: dict, source_url: str) -> ServiceProfile:
    """Convert a GraphQL introspection response into a ServiceProfile."""
    base_url = source_url.rsplit("/graphql", 1)[0]
    types    = data.get("data", {}).get("__schema", {}).get("types", [])
    endpoints: list[Endpoint] = []
    for t in types:
        if t.get("name") in ("Query", "Mutation"):
            method = "POST"
            for field in (t.get("fields") or []):
                fname = field.get("name", "")
                fdesc = field.get("description") or ""
                params = [
                    EndpointParam(name=a.get("name", ""), location="body",
                                  description=a.get("description", "") or "")
                    for a in (field.get("args") or [])
                ]
                endpoints.append(Endpoint(
                    method=method,
                    path=f"/graphql ({fname})",
                    description=fdesc,
                    params=params,
                ))
    return ServiceProfile(
        id=_make_id(f"{source_url}-graphql"),
        name="GraphQL API",
        base_url=base_url,
        description="GraphQL API discovered from introspection",
        auth=AuthScheme(type="bearer"),
        endpoints=endpoints[:30],
        source_url=source_url,
    )


def _html_to_text(html: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    import html as _html_mod
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = _html_mod.unescape(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _profile_from_llm_json(data: dict, source_url: str) -> ServiceProfile:
    """Build a ServiceProfile from the JSON produced by the LLM."""
    auth_raw = data.get("auth") or {}
    auth = AuthScheme(
        type=auth_raw.get("type", "none"),
        header=auth_raw.get("header", "Authorization"),
        param=auth_raw.get("param", ""),
    )
    endpoints: list[Endpoint] = []
    for e in (data.get("endpoints") or [])[:30]:
        params = [
            EndpointParam(
                name=p.get("name", ""),
                location=p.get("in", "query"),
                required=bool(p.get("required", False)),
                description=p.get("description", ""),
            )
            for p in (e.get("params") or [])
        ]
        endpoints.append(Endpoint(
            method=(e.get("method") or "GET").upper(),
            path=e.get("path") or "/",
            description=e.get("description") or "",
            params=params,
        ))
    name = data.get("name") or "Discovered API"
    return ServiceProfile(
        id=_make_id(name),
        name=name,
        base_url=data.get("base_url") or source_url,
        description=data.get("description") or "",
        auth=auth,
        endpoints=endpoints,
        source_url=source_url,
    )


# ── main ingestor ─────────────────────────────────────────────────────────────

class ServiceIngestor:
    """
    Fetches, parses, and persists service definitions from URLs.
    After ingestion the service is available as dynamic tools in the kernel.
    """

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "qwen3:4b") -> None:
        self._ollama = ollama_url.rstrip("/")
        self._model  = model

    # ── public API ────────────────────────────────────────────────────

    async def ingest(self, url: str) -> ServiceProfile:
        """
        Full ingestion pipeline.  Returns a ServiceProfile and persists it.
        Raises on unrecoverable fetch failure.
        """
        url = url.strip().rstrip("/")
        log.info("ServiceIngestor: ingesting %s", url)

        raw, content_type = await self._fetch(url)
        profile = await self._parse(raw, content_type, url)

        # Deduplicate: if same base_url already registered, overwrite
        try:
            from server.service_registry import get_service_registry
            reg = get_service_registry()
            existing = reg.find_by_base_url(profile.base_url)
            if existing:
                profile.id = existing.id   # keep same id on re-ingest
            reg.save(profile)
            log.info("ServiceIngestor: saved profile %s (%d endpoints)",
                     profile.id, len(profile.endpoints))
        except Exception as exc:
            log.warning("ServiceIngestor: could not persist profile: %s", exc)

        # Register dynamic tools
        try:
            from server.service_registry import register_service_tools
            register_service_tools(profile)
        except Exception as exc:
            log.warning("ServiceIngestor: could not register tools: %s", exc)

        return profile

    async def probe_and_ingest(self, base_url: str) -> ServiceProfile | None:
        """
        Try common OpenAPI suffixes on base_url.  Returns None if nothing found.
        """
        for suffix in PROBE_SUFFIXES:
            candidate = base_url.rstrip("/") + suffix
            try:
                raw, ct = await self._fetch(candidate)
                if raw:
                    log.info("ServiceIngestor: found schema at %s", candidate)
                    return await self.ingest(candidate)
            except Exception:
                continue
        return None

    # ── fetch ─────────────────────────────────────────────────────────

    async def _fetch(self, url: str) -> tuple[str, str]:
        """Returns (body_text, content_type)."""
        async with httpx.AsyncClient(
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "Essence-Agent/1 (+local)"},
        ) as c:
            r = await c.get(url)
            r.raise_for_status()
            ct = r.headers.get("content-type", "")
            text = r.text[:MAX_JSON_CHARS]
            return text, ct

    # ── parse dispatcher ──────────────────────────────────────────────

    async def _parse(self, raw: str, content_type: str, source_url: str) -> ServiceProfile:
        # Try JSON first regardless of content-type
        data = None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            pass

        if data is not None:
            # OpenAPI 3.x
            if "openapi" in data or "swagger" in data:
                log.debug("ServiceIngestor: detected OpenAPI spec")
                return _parse_openapi(data, source_url)
            # GraphQL introspection
            if data.get("data", {}).get("__schema"):
                log.debug("ServiceIngestor: detected GraphQL introspection")
                return _parse_graphql_introspection(data, source_url)
            # Generic JSON — let LLM interpret it
            log.debug("ServiceIngestor: generic JSON, using LLM parser")
            return await self._llm_parse(json.dumps(data)[:MAX_HTML_CHARS], source_url)

        # HTML / plain text — strip to readable text then LLM parse
        if "html" in content_type or raw.lstrip().startswith("<"):
            text = _html_to_text(raw)[:MAX_HTML_CHARS]
        else:
            text = raw[:MAX_HTML_CHARS]

        log.debug("ServiceIngestor: prose/HTML doc, using LLM parser")
        return await self._llm_parse(text, source_url)

    # ── LLM parse ─────────────────────────────────────────────────────

    async def _llm_parse(self, text: str, source_url: str) -> ServiceProfile:
        """Use the local LLM to extract a ServiceProfile from unstructured text."""
        prompt = LLM_PARSE_PROMPT + f"\nSource URL: {source_url}\n\n{text}"
        try:
            async with httpx.AsyncClient(timeout=45.0) as c:
                r = await c.post(
                    f"{self._ollama}/api/chat",
                    json={
                        "model": self._model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {"temperature": 0.0, "num_predict": 2048},
                    },
                )
            content = r.json().get("message", {}).get("content", "{}")
            # Robust JSON extraction
            m = re.search(r"\{[\s\S]*\}", content)
            if not m:
                raise ValueError("No JSON in LLM response")
            data = json.loads(m.group())
            return _profile_from_llm_json(data, source_url)
        except Exception as exc:
            log.warning("ServiceIngestor: LLM parse failed (%s), building minimal profile", exc)
            # Fallback: minimal profile so ingestion doesn't hard-fail
            return ServiceProfile(
                id=_make_id(source_url),
                name=source_url.split("//", 1)[-1].split("/")[0],
                base_url=source_url,
                description="Partially ingested (LLM parse failed)",
                source_url=source_url,
            )


# ── URL detection helper ──────────────────────────────────────────────────────

_URL_RE = re.compile(
    r"https?://[^\s\]\)\">]+"
    r"(?:/[^\s\]\)\">]*)?"
)


def extract_urls(text: str) -> list[str]:
    """Return all HTTP(S) URLs found in a chat message."""
    return _URL_RE.findall(text)


def looks_like_api_url(url: str) -> bool:
    """
    Heuristic: is this URL likely a service/API endpoint rather than
    a generic webpage to browse?
    """
    lower = url.lower()
    api_signals = [
        "/api", "/v1", "/v2", "/v3", "/openapi", "/swagger",
        "/graphql", "/rest", "/docs/api", "/api-docs",
        ".json", ".yaml", "openapi", "swagger",
    ]
    # Exclude common content sites
    skip_domains = ["github.com", "stackoverflow.com", "reddit.com",
                    "twitter.com", "youtube.com", "wikipedia.org",
                    "medium.com", "docs.google.com"]
    domain = url.split("//", 1)[-1].split("/")[0].lower()
    if any(d in domain for d in skip_domains):
        return False
    return any(sig in lower for sig in api_signals)


# ── singleton ─────────────────────────────────────────────────────────────────

_ingestor: ServiceIngestor | None = None


def init_ingestor(ollama_url: str = "http://localhost:11434", model: str = "qwen3:4b") -> ServiceIngestor:
    global _ingestor
    _ingestor = ServiceIngestor(ollama_url=ollama_url, model=model)
    return _ingestor


def get_ingestor() -> ServiceIngestor:
    global _ingestor
    if _ingestor is None:
        _ingestor = ServiceIngestor()
    return _ingestor
