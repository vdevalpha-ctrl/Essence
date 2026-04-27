"""
Essence MCP (Model Context Protocol) Server Registry.

Registered servers are persisted to  Essence/memory/mcp_servers.json.

Each entry:
{
  "id":          "filesystem",
  "label":       "Filesystem",
  "transport":   "stdio",          // stdio | http | sse
  "command":     ["npx", "@modelcontextprotocol/server-filesystem", "/workspace"],
  "url":         "",               // for http/sse transport
  "env":         {},               // extra env vars
  "enabled":     true,
  "tools":       [],               // cached tool list, populated on connect
  "last_ping":   null
}
"""
from __future__ import annotations
import json, time, logging, asyncio
from pathlib import Path
from typing import Any

log = logging.getLogger("essence.mcpstore")

_MCP_PRESETS = [
    {"id":"filesystem",  "label":"Filesystem (npx)",       "transport":"stdio","command":["npx","-y","@modelcontextprotocol/server-filesystem","."],"url":"","env":{},"enabled":False,"tools":[],"last_ping":None,"preset":True,"description":"Read/write files via MCP. Requires Node.js."},
    {"id":"memory",      "label":"Memory Store (npx)",     "transport":"stdio","command":["npx","-y","@modelcontextprotocol/server-memory"],"url":"","env":{},"enabled":False,"tools":[],"last_ping":None,"preset":True,"description":"Persistent key-value memory via MCP. Requires Node.js."},
    {"id":"brave_search","label":"Brave Search (npx)",     "transport":"stdio","command":["npx","-y","@modelcontextprotocol/server-brave-search"],"url":"","env":{"BRAVE_API_KEY":""},"enabled":False,"tools":[],"last_ping":None,"preset":True,"description":"Web search via Brave API. Requires BRAVE_API_KEY."},
    {"id":"github",      "label":"GitHub (npx)",           "transport":"stdio","command":["npx","-y","@modelcontextprotocol/server-github"],"url":"","env":{"GITHUB_PERSONAL_ACCESS_TOKEN":""},"enabled":False,"tools":[],"last_ping":None,"preset":True,"description":"GitHub API tools. Requires GITHUB_PERSONAL_ACCESS_TOKEN."},
    {"id":"sqlite",      "label":"SQLite (npx)",           "transport":"stdio","command":["npx","-y","@modelcontextprotocol/server-sqlite","./memory/db/essence.db"],"url":"","env":{},"enabled":False,"tools":[],"last_ping":None,"preset":True,"description":"SQLite database access via MCP."},
    {"id":"puppeteer",   "label":"Puppeteer / Browser",    "transport":"stdio","command":["npx","-y","@modelcontextprotocol/server-puppeteer"],"url":"","env":{},"enabled":False,"tools":[],"last_ping":None,"preset":True,"description":"Browser automation and web scraping via MCP."},
]


class MCPStore:
    def __init__(self, registry_path: Path):
        self.path = registry_path
        self._servers: dict[str, dict] = {}
        self._load()
        self._seed_presets()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._servers = {s["id"]: s for s in data.get("servers", [])}
            except Exception as e:
                log.warning("Failed to load MCP registry: %s", e)

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"servers": list(self._servers.values())}, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def _seed_presets(self) -> None:
        changed = False
        for p in _MCP_PRESETS:
            if p["id"] not in self._servers:
                self._servers[p["id"]] = dict(p)
                changed = True
        if changed:
            self._save()

    def list_all(self) -> list[dict]:
        return list(self._servers.values())

    def get(self, server_id: str) -> dict | None:
        return self._servers.get(server_id)

    def register(self, data: dict) -> dict:
        sid = data.get("id", "").strip().replace(" ", "_").lower()
        if not sid:
            sid = f"mcp_{int(time.time())}"
        entry = {
            "id":        sid,
            "label":     data.get("label", sid),
            "transport": data.get("transport", "stdio"),
            "command":   data.get("command", []),
            "url":       data.get("url", ""),
            "env":       data.get("env", {}),
            "enabled":   bool(data.get("enabled", True)),
            "tools":     [],
            "last_ping": None,
            "preset":    False,
            "description": data.get("description", ""),
        }
        self._servers[sid] = entry
        self._save()
        return entry

    def patch(self, server_id: str, patch: dict) -> dict:
        s = self._servers.get(server_id)
        if s is None:
            raise KeyError(f"MCP server '{server_id}' not found")
        allowed = {"label","transport","command","url","env","enabled","description"}
        s.update({k: v for k, v in patch.items() if k in allowed})
        self._save()
        return s

    def delete(self, server_id: str) -> bool:
        if server_id not in self._servers:
            return False
        del self._servers[server_id]
        self._save()
        return True

    async def list_tools(self, server_id: str) -> list[dict]:
        """
        Attempt to connect to the MCP server and list its tools.
        Uses stdio transport for npx-based servers.
        Returns cached tools on error.
        """
        s = self._servers.get(server_id)
        if not s:
            return []
        if s["transport"] == "http" or s["transport"] == "sse":
            return await self._list_tools_http(s)
        return await self._list_tools_stdio(s)

    async def _list_tools_stdio(self, s: dict) -> list[dict]:
        cmd = s.get("command", [])
        if not cmd:
            return s.get("tools", [])
        env = {**__import__("os").environ, **s.get("env", {})}
        try:
            # Send MCP initialize + tools/list
            init_msg  = json.dumps({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"Essence","version":"29"}}})
            tools_msg = json.dumps({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}})
            input_data = (init_msg + "\n" + tools_msg + "\n").encode()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(input_data), timeout=10)
            except asyncio.TimeoutError:
                proc.kill()
                return s.get("tools", [])
            tools = []
            for line in stdout.decode("utf-8", "replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    if msg.get("id") == 2 and "result" in msg:
                        for t in msg["result"].get("tools", []):
                            tools.append({"name": t.get("name",""), "description": t.get("description",""), "inputSchema": t.get("inputSchema",{})})
                except Exception:
                    continue
            if tools:
                s["tools"] = tools
                s["last_ping"] = time.time()
                self._save()
            return tools or s.get("tools", [])
        except FileNotFoundError:
            return s.get("tools", [])
        except Exception as e:
            log.warning("MCP stdio list_tools(%s) failed: %s", s["id"], e)
            return s.get("tools", [])

    async def _list_tools_http(self, s: dict) -> list[dict]:
        import httpx
        url = s.get("url","").rstrip("/") + "/tools/list"
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.post(url, json={"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}})
                if r.status_code == 200:
                    tools = [{"name":t.get("name",""),"description":t.get("description",""),"inputSchema":t.get("inputSchema",{})} for t in r.json().get("result",{}).get("tools",[])]
                    s["tools"] = tools
                    s["last_ping"] = time.time()
                    self._save()
                    return tools
        except Exception as e:
            log.warning("MCP http list_tools(%s) failed: %s", s["id"], e)
        return s.get("tools", [])

    async def call_tool(self, server_id: str, tool_name: str, arguments: dict) -> Any:
        """Invoke a tool on an MCP server and return its result."""
        s = self._servers.get(server_id)
        if not s:
            raise ValueError(f"MCP server '{server_id}' not found")
        if s["transport"] in ("http","sse"):
            return await self._call_tool_http(s, tool_name, arguments)
        return await self._call_tool_stdio(s, tool_name, arguments)

    async def _call_tool_stdio(self, s: dict, tool_name: str, arguments: dict) -> Any:
        cmd = s.get("command", [])
        env = {**__import__("os").environ, **s.get("env", {})}
        init_msg = json.dumps({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"Essence","version":"29"}}})
        call_msg  = json.dumps({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":tool_name,"arguments":arguments}})
        input_data = (init_msg + "\n" + call_msg + "\n").encode()
        proc = await asyncio.create_subprocess_exec(*cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(input_data), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            raise TimeoutError(f"MCP tool call timed out: {tool_name}")
        for line in stdout.decode("utf-8","replace").splitlines():
            try:
                msg = json.loads(line.strip())
                if msg.get("id") == 2 and "result" in msg:
                    return msg["result"]
                if msg.get("id") == 2 and "error" in msg:
                    raise RuntimeError(msg["error"].get("message","MCP error"))
            except json.JSONDecodeError:
                continue
        raise RuntimeError(f"No result from MCP tool: {tool_name}")

    async def _call_tool_http(self, s: dict, tool_name: str, arguments: dict) -> Any:
        import httpx
        url = s.get("url","").rstrip("/") + "/tools/call"
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(url, json={"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":tool_name,"arguments":arguments}})
            r.raise_for_status()
            d = r.json()
            if "error" in d:
                raise RuntimeError(d["error"].get("message","MCP error"))
            return d.get("result")
