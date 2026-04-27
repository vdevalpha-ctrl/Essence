"""
tools_engine.py — Essence v1 Tool Execution Engine
=================================================
Provides a registry of tools the AI can call, plus the executor that
actually runs them and returns results.

Tools are defined as plain dicts (Ollama function-calling format) and
each has a corresponding async executor function.

Safety rules
------------
• shell commands run in the workspace directory with a 30s hard timeout
• shell has a blocklist of dangerous patterns (rm -rf /, format, del /f /s C:, etc.)
• file operations are clamped to workspace root (no path traversal above it)
• http_get is GET-only, no credentials, 10s timeout, 50KB response cap
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

# ── Workspace root ────────────────────────────────────────────────────
_WORKSPACE: Path | None = None

def init_tools(workspace: Path) -> None:
    global _WORKSPACE
    _WORKSPACE = workspace

def workspace() -> Path:
    if _WORKSPACE is None:
        raise RuntimeError("tools_engine.init_tools(workspace) not called")
    return _WORKSPACE

# ── Safety ────────────────────────────────────────────────────────────
_SHELL_BLOCKLIST = [
    r"rm\s+-rf\s+/",
    r"del\s+/[fFsS].*C:\\",
    r"format\s+[Cc]:",
    r"mkfs",
    r"dd\s+if=",
    r"shutdown",
    r":(){ :|:& };:",      # fork bomb
    r">\s*/dev/sd",
    r"chmod\s+777\s+/",
]
_BLOCK_RE = [re.compile(p, re.IGNORECASE) for p in _SHELL_BLOCKLIST]

def _shell_safe(cmd: str) -> tuple[bool, str]:
    for pat in _BLOCK_RE:
        if pat.search(cmd):
            return False, f"Command blocked by safety policy: {pat.pattern}"
    return True, ""


def _safe_path(rel: str) -> Path:
    """Resolve path relative to workspace, raise ValueError if outside.

    Uses Path.relative_to() rather than startswith() to avoid the
    /home/user/essence vs /home/user/essence2 prefix collision bug.
    """
    base = workspace().resolve()
    target = (base / rel).resolve()
    try:
        target.relative_to(base)  # raises ValueError if outside
    except ValueError:
        raise ValueError(f"Path '{rel}' escapes workspace root")
    return target


# ═══════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS  (Ollama function-calling schema)
# ═══════════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": (
                "Execute a shell command in the workspace directory. "
                "Use for file system operations, running scripts, git, pip, etc. "
                "Prefer specific commands over broad ones. Output truncated at 4000 chars."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Max seconds to wait (default 30, max 120)",
                        "default": 30
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace. Path is relative to workspace root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root"
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters to return (default 8000)",
                        "default": 8000
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write or append to a file in the workspace. "
                "Creates parent directories as needed. "
                "mode='overwrite' replaces; mode='append' adds to end."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root"
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "default": "overwrite"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories in a workspace path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to workspace (default: '.')",
                        "default": "."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "http_get",
            "description": (
                "Perform an HTTP GET request. "
                "Use for fetching web pages, public APIs, or JSON feeds. "
                "Response capped at 50KB. No authentication."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to fetch"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Persist a fact or insight to long-term memory (MEMORY.md). "
                "Use to store important information learned during conversation "
                "that should be recalled in future sessions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "Fact or insight to remember (concise, factual)"
                    },
                    "section": {
                        "type": "string",
                        "description": "Memory section: 'Key facts' | 'Ongoing projects' | 'Preferences'",
                        "default": "Key facts"
                    }
                },
                "required": ["fact"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search MEMORY.md for relevant stored facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Create a tracked task that will appear in the Tasks panel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Task title (concise)"
                    },
                    "description": {
                        "type": "string",
                        "description": "What this task entails"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high"],
                        "default": "normal"
                    }
                },
                "required": ["title"]
            }
        }
    },
    # ── Git tools ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show the working-tree status of the git repository (like `git status --short`).",
            "parameters": {"type": "object", "properties": {
                "repo": {"type": "string", "description": "Optional repo path (default: workspace)"}
            }},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Show recent git commits.",
            "parameters": {"type": "object", "properties": {
                "n":      {"type": "integer", "description": "Number of commits (default: 10)"},
                "format": {"type": "string",  "enum": ["oneline", "short", "medium"], "default": "oneline"},
                "repo":   {"type": "string",  "description": "Optional repo path"},
            }},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show the git diff of the working tree or a specific ref.",
            "parameters": {"type": "object", "properties": {
                "ref":    {"type": "string",  "description": "Ref to diff against (default: HEAD)"},
                "staged": {"type": "boolean", "description": "Show staged changes only (default: false)"},
                "repo":   {"type": "string",  "description": "Optional repo path"},
            }},
        },
    },
    # ── Clipboard tools ──────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "clipboard_read",
            "description": "Read the current contents of the system clipboard.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clipboard_write",
            "description": "Write text to the system clipboard.",
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string", "description": "Text to copy to clipboard"},
            }, "required": ["text"]},
        },
    },
    # ── Wikipedia tool ───────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "wiki_search",
            "description": "Search Wikipedia and return a summary of the article.",
            "parameters": {"type": "object", "properties": {
                "query":     {"type": "string",  "description": "Article title or search query"},
                "sentences": {"type": "integer", "description": "Number of sentences to return (default: 3)"},
                "lang":      {"type": "string",  "description": "Wikipedia language code (default: 'en')"},
            }, "required": ["query"]},
        },
    },
    # ── Desktop notification tool ────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "desktop_notify",
            "description": "Send a desktop notification to the user (macOS/Windows/Linux).",
            "parameters": {"type": "object", "properties": {
                "title":   {"type": "string", "description": "Notification title"},
                "message": {"type": "string", "description": "Notification body text"},
                "urgency": {"type": "string", "enum": ["low", "normal", "critical"], "default": "normal"},
            }, "required": ["message"]},
        },
    },
]

# Merge web_tools definitions at import time (graceful if module unavailable)
try:
    from server.web_tools import WEB_TOOL_DEFINITIONS as _WEB_DEFS
    _existing_names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    for _wt in _WEB_DEFS:
        if _wt["function"]["name"] not in _existing_names:
            TOOL_DEFINITIONS.append(_wt)
except Exception:
    pass

# Merge GitHub tool definitions
try:
    from server.github_tools import GITHUB_TOOL_DEFINITIONS as _GH_DEFS
    _existing_names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    for _gt in _GH_DEFS:
        if _gt["function"]["name"] not in _existing_names:
            TOOL_DEFINITIONS.append(_gt)
except Exception:
    pass

# Merge calendar tool definitions
try:
    from server.calendar_tools import CALENDAR_TOOL_DEFINITIONS as _CAL_DEFS
    _existing_names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    for _ct in _CAL_DEFS:
        if _ct["function"]["name"] not in _existing_names:
            TOOL_DEFINITIONS.append(_ct)
except Exception:
    pass

# Tool name → definition lookup
TOOLS_BY_NAME: dict[str, dict] = {
    t["function"]["name"]: t["function"]
    for t in TOOL_DEFINITIONS
}


# ═══════════════════════════════════════════════════════════════════════
# EXECUTORS
# ═══════════════════════════════════════════════════════════════════════

async def _exec_shell(args: dict) -> str:
    cmd = args.get("command", "").strip()
    if not cmd:
        return "ERROR: empty command"
    ok, reason = _shell_safe(cmd)
    if not ok:
        return f"BLOCKED: {reason}"
    timeout = min(int(args.get("timeout", 30)), 120)
    try:
        # Use subprocess_exec via the system shell but with explicit argv so
        # we're not running user input as a raw shell string on Windows/Unix.
        # On Unix we split via /bin/sh -c; on Windows via cmd /C — this gives
        # shell conveniences (pipes, &&, globs) while still allowing the
        # blocklist check above to run against the literal command string.
        import sys as _sys
        if _sys.platform == "win32":
            argv = ["cmd", "/C", cmd]
        else:
            argv = ["/bin/sh", "-c", cmd]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(workspace()),
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            text = out.decode("utf-8", errors="replace")
            if len(text) > 4000:
                # Return last 4000 chars — errors and final output are most useful
                text = "[...truncated — showing last 4000 chars...]\n" + text[-4000:]
            return text or "(no output)"
        except asyncio.TimeoutError:
            proc.kill()
            return f"TIMEOUT after {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"


async def _exec_read_file(args: dict) -> str:
    path = args.get("path", "")
    max_chars = int(args.get("max_chars", 8000))
    try:
        p = _safe_path(path)
        if not p.exists():
            return f"ERROR: file not found: {path}"
        if p.is_dir():
            return f"ERROR: {path} is a directory, use list_dir"
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n[...truncated at {max_chars} chars]"
        return text or "(empty file)"
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: {e}"


async def _exec_write_file(args: dict) -> str:
    path = args.get("path", "")
    content = args.get("content", "")
    mode = args.get("mode", "overwrite")
    try:
        p = _safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with open(p, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            p.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} chars to {path}"
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: {e}"


async def _exec_list_dir(args: dict) -> str:
    path = args.get("path", ".")
    try:
        p = _safe_path(path)
        if not p.exists():
            return f"ERROR: path not found: {path}"
        items = []
        for entry in sorted(p.iterdir()):
            kind = "DIR " if entry.is_dir() else "FILE"
            size = ""
            if entry.is_file():
                size = f" ({entry.stat().st_size:,} bytes)"
            items.append(f"{kind}  {entry.name}{size}")
        return "\n".join(items) if items else "(empty directory)"
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: {e}"


_PRIVATE_NETS = [
    # loopback
    (0x7F000000, 0xFF000000),   # 127.0.0.0/8
    # link-local (AWS/GCP metadata)
    (0xA9FE0000, 0xFFFF0000),   # 169.254.0.0/16
    # RFC-1918
    (0x0A000000, 0xFF000000),   # 10.0.0.0/8
    (0xAC100000, 0xFFF00000),   # 172.16.0.0/12
    (0xC0A80000, 0xFFFF0000),   # 192.168.0.0/16
    # IPv6 loopback as int (::1 → 1)
]

def _is_private_ip(host: str) -> bool:
    """Return True if host resolves to a private/loopback/link-local IP."""
    import socket
    try:
        ip_str = socket.gethostbyname(host)
        parts = [int(p) for p in ip_str.split(".")]
        if len(parts) != 4:
            return False
        ip_int = (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3]
        return any((ip_int & mask) == net for net, mask in _PRIVATE_NETS)
    except Exception:
        return False   # resolution error → allow (will fail at request time)


async def _exec_http_get(args: dict) -> str:
    url = args.get("url", "").strip()
    if not url:
        return "ERROR: empty URL"
    if not url.startswith(("http://", "https://")):
        return "ERROR: only http/https URLs allowed"
    # SSRF guard: block private/loopback/link-local IPs
    try:
        from urllib.parse import urlparse as _up
        host = _up(url).hostname or ""
        if _is_private_ip(host):
            return f"BLOCKED: URL resolves to a private/loopback IP — SSRF protection"
    except Exception:
        pass
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=False) as c:
            r = await c.get(url, headers={"User-Agent": "Essence/1 (+local-agent)"})
            ct = r.headers.get("content-type", "")
            text = r.text[:50000]
            # Strip heavy HTML down to text content
            if "html" in ct:
                import html
                text = re.sub(r"<[^>]+>", " ", text)
                text = html.unescape(text)
                text = re.sub(r"\s{3,}", "\n\n", text)
                text = text[:8000]
            return f"HTTP {r.status_code} {url}\n\n{text}"
    except Exception as e:
        return f"ERROR: {e}"


async def _exec_remember(args: dict) -> str:
    fact = args.get("fact", "").strip()
    section = args.get("section", "Key facts")
    if not fact:
        return "ERROR: empty fact"
    try:
        p = workspace() / "MEMORY.md"
        content = p.read_text(encoding="utf-8") if p.exists() else ""
        # Find the section and append under it
        ts = time.strftime("%Y-%m-%d")
        entry = f"- [{ts}] {fact}"
        if f"## {section}" in content:
            content = content.replace(
                f"## {section}\n",
                f"## {section}\n{entry}\n"
            )
        else:
            content += f"\n## {section}\n{entry}\n"
        p.write_text(content, encoding="utf-8")
        return f"OK: remembered under '{section}'"
    except Exception as e:
        return f"ERROR: {e}"


async def _exec_search_memory(args: dict) -> str:
    query = args.get("query", "").lower().strip()
    if not query:
        return "ERROR: empty query"
    try:
        p = workspace() / "MEMORY.md"
        if not p.exists():
            return "MEMORY.md not found or empty."
        lines = p.read_text(encoding="utf-8").split("\n")
        matches = [l for l in lines if query in l.lower()]
        if not matches:
            return f"No memory found matching '{query}'"
        return "\n".join(matches[:30])
    except Exception as e:
        return f"ERROR: {e}"


# Task store (in-memory + persisted to tasks.json)
_TASKS: list[dict] = []

async def _exec_create_task(args: dict) -> str:
    import uuid as _uuid
    title = args.get("title", "Untitled").strip()
    desc  = args.get("description", "")
    prio  = args.get("priority", "normal")
    task  = {
        "id":          _uuid.uuid4().hex[:8],
        "label":       title,
        "description": desc,
        "priority":    prio,
        "status":      "pending",
        "provider":    "agent",
        "ts":          int(time.time() * 1000),
    }
    _TASKS.append(task)
    # Persist
    try:
        p = workspace() / "memory" / "tasks.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(p.read_text()) if p.exists() else []
        existing.append(task)
        p.write_text(json.dumps(existing, indent=2))
    except Exception:
        pass
    return f"OK: task created [{task['id']}] {title}"


# ── Git tool executors ──────────────────────────────────────────────────────

async def _exec_git_status(args: dict) -> str:
    try:
        from server.git_context import exec_git_status
        return await exec_git_status(args)
    except Exception as e:
        return f"ERROR: {e}"


async def _exec_git_log(args: dict) -> str:
    try:
        from server.git_context import exec_git_log
        return await exec_git_log(args)
    except Exception as e:
        return f"ERROR: {e}"


async def _exec_git_diff(args: dict) -> str:
    try:
        from server.git_context import exec_git_diff
        return await exec_git_diff(args)
    except Exception as e:
        return f"ERROR: {e}"


# ── Clipboard tool executors ──────────────────────────────────────────────

async def _exec_clipboard_read(args: dict) -> str:
    try:
        import platform, subprocess
        sys_name = platform.system()
        if sys_name == "Darwin":
            result = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=5)
            return result.stdout or "(clipboard empty)"
        elif sys_name == "Windows":
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() or "(clipboard empty)"
        else:
            for cmd in [["xclip", "-selection", "clipboard", "-o"],
                        ["xsel", "--clipboard", "--output"]]:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        return result.stdout or "(clipboard empty)"
                except FileNotFoundError:
                    continue
            return "ERROR: no clipboard tool found (install xclip or xsel)"
    except Exception as e:
        return f"ERROR: {e}"


async def _exec_clipboard_write(args: dict) -> str:
    text = args.get("text", "")
    try:
        import platform, subprocess
        sys_name = platform.system()
        if sys_name == "Darwin":
            p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            p.communicate(text.encode())
        elif sys_name == "Windows":
            subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"Set-Clipboard -Value @'\n{text}\n'@"],
                check=True, timeout=5,
            )
        else:
            for cmd in [["xclip", "-selection", "clipboard"],
                        ["xsel", "--clipboard", "--input"]]:
                try:
                    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                    p.communicate(text.encode())
                    return f"OK: copied {len(text)} chars to clipboard"
                except FileNotFoundError:
                    continue
            return "ERROR: no clipboard tool found (install xclip or xsel)"
        return f"OK: copied {len(text)} chars to clipboard"
    except Exception as e:
        return f"ERROR: {e}"


# ── Wikipedia tool executor ───────────────────────────────────────────────

async def _exec_wiki_search(args: dict) -> str:
    query     = args.get("query", "").strip()
    if not query:
        return "ERROR: query is required"
    lang      = args.get("lang", "en")
    sentences = min(int(args.get("sentences", 3)), 10)
    try:
        url = (f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
               + query.replace(" ", "_"))
        async with httpx.AsyncClient(timeout=10,
                                     headers={"User-Agent": "Essence/1"}) as c:
            r = await c.get(url)
            if r.status_code == 200:
                data    = r.json()
                title   = data.get("title", query)
                extract = data.get("extract", "")
                s_list  = re.split(r"(?<=[.!?])\s+", extract)
                result  = " ".join(s_list[:sentences])
                source  = (data.get("content_urls", {})
                           .get("desktop", {}).get("page", ""))
                return f"**{title}**\n{result}\nSource: {source}"
            elif r.status_code == 404:
                # Try search suggestions
                sr = await c.get(
                    f"https://{lang}.wikipedia.org/w/api.php",
                    params={"action": "query", "list": "search",
                            "srsearch": query, "format": "json", "srlimit": 3},
                )
                if sr.status_code == 200:
                    results = sr.json().get("query", {}).get("search", [])
                    if results:
                        top = results[0]["title"]
                        return (f"No exact match for '{query}'. "
                                f"Did you mean: {top}? "
                                f"Try: wiki_search query='{top}'")
                return f"Wikipedia: no article found for '{query}'"
            else:
                return f"Wikipedia HTTP {r.status_code}"
    except Exception as e:
        return f"ERROR: {e}"


# ── Desktop notification executor ─────────────────────────────────────────

async def _exec_desktop_notify(args: dict) -> str:
    title   = args.get("title", "Essence")
    message = args.get("message", "").strip()
    urgency = args.get("urgency", "normal")
    if not message:
        return "ERROR: message is required"
    try:
        import platform, subprocess
        sys_name = platform.system()
        if sys_name == "Darwin":
            safe_msg   = message.replace('"', '\\"')
            safe_title = title.replace('"', '\\"')
            script = f'display notification "{safe_msg}" with title "{safe_title}"'
            subprocess.run(["osascript", "-e", script], timeout=5)
            return "OK: notification sent"
        elif sys_name == "Windows":
            # Use BurntToast via PowerShell (graceful fallback)
            safe_msg   = message.replace("'", "`'")
            safe_title = title.replace("'", "`'")
            ps = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                f"$n = New-Object System.Windows.Forms.NotifyIcon; "
                f"$n.Icon = [System.Drawing.SystemIcons]::Information; "
                f"$n.Visible = $true; "
                f"$n.ShowBalloonTip(3000, '{safe_title}', '{safe_msg}', "
                "[System.Windows.Forms.ToolTipIcon]::Info)"
            )
            subprocess.Popen(
                ["powershell", "-NoProfile", "-Command", ps],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return "OK: notification sent"
        else:
            u = {"low": "low", "normal": "normal", "critical": "critical"}.get(urgency, "normal")
            result = subprocess.run(
                ["notify-send", "-u", u, title, message],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                return "OK: notification sent"
            return "ERROR: notify-send failed (install libnotify-bin)"
    except Exception as e:
        return f"ERROR: {e}"


def get_tasks() -> list[dict]:
    """Return current in-memory task list (for /api/tasks endpoint)."""
    return list(_TASKS)


def update_task_status(task_id: str, status: str, result: str = "") -> None:
    for t in _TASKS:
        if t["id"] == task_id:
            t["status"] = status
            if result:
                t["result"] = result
            break


# ═══════════════════════════════════════════════════════════════════════
# DISPATCH
# ═══════════════════════════════════════════════════════════════════════

_EXECUTORS = {
    "shell":           _exec_shell,
    "read_file":       _exec_read_file,
    "write_file":      _exec_write_file,
    "list_dir":        _exec_list_dir,
    "http_get":        _exec_http_get,
    "remember":        _exec_remember,
    "search_memory":   _exec_search_memory,
    "create_task":     _exec_create_task,
    # Git tools
    "git_status":      _exec_git_status,
    "git_log":         _exec_git_log,
    "git_diff":        _exec_git_diff,
    # Clipboard tools
    "clipboard_read":  _exec_clipboard_read,
    "clipboard_write": _exec_clipboard_write,
    # Wikipedia
    "wiki_search":     _exec_wiki_search,
    # Desktop notifications
    "desktop_notify":  _exec_desktop_notify,
}

# GitHub tool names (routed to github_tools module)
_GITHUB_NAMES = {
    "gh_list_repos", "gh_list_issues", "gh_create_issue",
    "gh_list_prs", "gh_get_file", "gh_search_code",
}

# Calendar tool names (routed to calendar_tools module)
_CALENDAR_NAMES = {"get_datetime", "date_math", "list_events", "add_event"}


# ═══════════════════════════════════════════════════════════════════════
# FUNCTION CALL VALIDATOR
# (ported from Hermes-Function-Calling — no external dependencies)
# ═══════════════════════════════════════════════════════════════════════

_JSON_TYPE_MAP: dict[str, type | tuple] = {
    "string":  str,
    "number":  (int, float),
    "integer": int,
    "boolean": bool,
    "array":   list,
    "object":  dict,
    "null":    type(None),
}


def _validate_enum_value(arg_name: str, arg_value: Any, enum_values: list) -> None:
    """Raise ValueError if arg_value is not in enum_values."""
    if None not in enum_values and enum_values:
        if arg_value not in enum_values:
            raise ValueError(
                f"Invalid value {arg_value!r} for '{arg_name}'. "
                f"Expected one of: {', '.join(str(v) for v in enum_values)}"
            )


def _validate_arg_type(arg_name: str, arg_value: Any, arg_schema: dict) -> None:
    """Raise TypeError / ValueError if arg_value doesn't match arg_schema."""
    arg_type = arg_schema.get("type")
    if not arg_type:
        return
    # Enum check first
    if arg_type == "string" and "enum" in arg_schema:
        _validate_enum_value(arg_name, arg_value, arg_schema["enum"])
    # Type check
    expected_py = _JSON_TYPE_MAP.get(arg_type)
    if expected_py is not None and not isinstance(arg_value, expected_py):
        raise TypeError(
            f"Type mismatch for '{arg_name}': expected {arg_type}, "
            f"got {type(arg_value).__name__}"
        )


def validate_tool_call(name: str, args: dict) -> tuple[bool, str]:
    """Validate a tool call against TOOLS_BY_NAME schema.

    Returns (ok, error_message).  error_message is "" when ok=True.

    Checks:
      1. Tool exists in registry (static + dynamic)
      2. Required arguments are present
      3. Argument types match schema (with enum enforcement)

    Does NOT raise — always returns a result tuple so callers can decide
    whether to block the call or just log and proceed.
    """
    # Resolve definition from static catalogue or dynamic service tools
    defn = TOOLS_BY_NAME.get(name)
    if defn is None:
        # Check dynamically registered service tools
        for td in TOOL_DEFINITIONS:
            if td.get("function", {}).get("name") == name:
                defn = td["function"]
                break
    if defn is None:
        return False, f"Unknown tool '{name}'"

    params = defn.get("parameters", {})
    properties = params.get("properties", {})
    required = params.get("required", [])

    # Required argument check
    missing = [r for r in required if r not in args]
    if missing:
        return False, f"Missing required argument(s): {', '.join(missing)}"

    # Per-argument type + enum validation
    for arg_name, arg_schema in properties.items():
        if arg_name not in args:
            continue
        try:
            _validate_arg_type(arg_name, args[arg_name], arg_schema)
        except (TypeError, ValueError) as exc:
            return False, str(exc)

    return True, ""


async def execute_tool(name: str, args: dict) -> str:
    """
    Execute a named tool with the given arguments dict.
    Returns a string result (always, even on error).

    Arguments are validated against the tool's schema before execution.
    Type mismatches and missing required args are returned as error strings
    rather than raising exceptions, so the LLM can recover gracefully.

    MCP tools use the naming convention  mcp__<server_id>__<tool_name>
    and are routed to MCPStore.call_tool() transparently.
    """
    # ── MCP tool routing ─────────────────────────────────────────────────
    if name.startswith("mcp__"):
        return await _execute_mcp_tool(name, args)

    # ── Web / browser tool routing ────────────────────────────────────────
    _WEB_NAMES = {"web_search", "fetch_page", "browser_goto", "browser_click",
                  "browser_fill", "browser_eval", "browser_screenshot", "browser_get_text"}
    if name in _WEB_NAMES:
        try:
            from server.web_tools import execute_web_tool as _ewt
            result = await _ewt(name, args)
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except Exception as _we:
            return f"ERROR executing web tool {name}: {_we}"

    # ── GitHub tool routing ──────────────────────────────────────────────
    if name in _GITHUB_NAMES:
        try:
            from server.github_tools import GITHUB_EXECUTORS as _ghex
            fn_gh = _ghex.get(name)
            if fn_gh:
                result = await fn_gh(args)
                return result if isinstance(result, str) else json.dumps(result)
        except Exception as _ghe:
            return f"ERROR executing GitHub tool {name}: {_ghe}"

    # ── Calendar tool routing ────────────────────────────────────────────
    if name in _CALENDAR_NAMES:
        try:
            from server.calendar_tools import CALENDAR_EXECUTORS as _calex
            fn_cal = _calex.get(name)
            if fn_cal:
                result = await fn_cal(args)
                return result if isinstance(result, str) else json.dumps(result)
        except Exception as _cale:
            return f"ERROR executing calendar tool {name}: {_cale}"

    fn = _EXECUTORS.get(name)
    if fn is None:
        return f"ERROR: unknown tool '{name}'"

    # Validate before executing
    ok, err = validate_tool_call(name, args)
    if not ok:
        import logging as _log
        _log.getLogger("essence.tools").warning("Tool call validation failed [%s]: %s", name, err)
        return f"ERROR: invalid tool call — {err}"

    try:
        result = await fn(args)
        return result if isinstance(result, str) else json.dumps(result)
    except Exception as e:
        return f"ERROR executing {name}: {e}"


# ═══════════════════════════════════════════════════════════════════════
# MCP TOOL BRIDGE
# ═══════════════════════════════════════════════════════════════════════

_MCP_STORE_PATH: Path | None = None

import logging as _tlog
_mcp_log = _tlog.getLogger("essence.tools.mcp")


def init_mcp(store_path: Path) -> None:
    """Set the MCP registry path.  Called once at startup."""
    global _MCP_STORE_PATH
    _MCP_STORE_PATH = store_path


async def _execute_mcp_tool(qualified_name: str, args: dict) -> str:
    """
    Route a call for  mcp__<server_id>__<tool_name>  to the MCPStore.
    Returns the result as a JSON string or plain text.
    """
    if _MCP_STORE_PATH is None:
        return "ERROR: MCP not initialised — call tools_engine.init_mcp()"

    parts = qualified_name.split("__", 2)   # ["mcp", server_id, tool_name]
    if len(parts) != 3:
        return f"ERROR: malformed MCP tool name '{qualified_name}'"

    _, server_id, tool_name = parts
    try:
        from server.mcpstore import MCPStore
        store  = MCPStore(_MCP_STORE_PATH)
        result = await store.call_tool(server_id, tool_name, args)
        if isinstance(result, (dict, list)):
            # Flatten content blocks from MCP response
            if isinstance(result, dict) and "content" in result:
                content = result["content"]
                if isinstance(content, list):
                    texts = []
                    for block in content:
                        if isinstance(block, dict):
                            texts.append(block.get("text", str(block)))
                        else:
                            texts.append(str(block))
                    return "\n".join(texts)
                return str(content)
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)
    except Exception as exc:
        _mcp_log.warning("MCP tool %s/%s failed: %s", server_id, tool_name, exc)
        return f"ERROR calling MCP tool {server_id}/{tool_name}: {exc}"


async def register_mcp_tools(store_path: Path) -> int:
    """
    Connect to all enabled MCP servers, list their tools, and add them
    to TOOL_DEFINITIONS so the LLM can call them natively.

    Each tool gets the name  mcp__<server_id>__<tool_name>  so it routes
    back through _execute_mcp_tool().

    Returns the number of new tools registered.
    """
    init_mcp(store_path)
    try:
        from server.mcpstore import MCPStore
    except ImportError:
        _mcp_log.warning("mcpstore not available — MCP tools disabled")
        return 0

    store   = MCPStore(store_path)
    servers = [s for s in store.list_all() if s.get("enabled")]
    if not servers:
        return 0

    registered = 0
    existing_names = {td["function"]["name"] for td in TOOL_DEFINITIONS if "function" in td}

    for s in servers:
        sid = s["id"]
        try:
            tools = await store.list_tools(sid)
        except Exception as exc:
            _mcp_log.warning("MCP server %s: tool listing failed: %s", sid, exc)
            tools = s.get("tools", [])

        for t in tools:
            qname = f"mcp__{sid}__{t['name']}"
            if qname in existing_names:
                continue   # already registered (e.g. from a prior connect call)

            # Translate MCP inputSchema → OpenAI function calling parameters
            input_schema = t.get("inputSchema") or {"type": "object", "properties": {}}
            TOOL_DEFINITIONS.append({
                "type": "function",
                "function": {
                    "name":        qname,
                    "description": f"[MCP/{sid}] {t.get('description', t['name'])}",
                    "parameters":  input_schema,
                },
            })
            existing_names.add(qname)
            registered += 1
            _mcp_log.debug("MCP tool registered: %s", qname)

    if registered:
        _mcp_log.info("MCP: registered %d tool(s) from %d server(s)", registered, len(servers))
    return registered


# ═══════════════════════════════════════════════════════════════════════
# MEMORY CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_memory_context(workspace_path: Path) -> str:
    """
    Read workspace memory files (IDENTITY.md, MEMORY.md, GOALS.md)
    and return a formatted context block to inject into system prompt.
    """
    parts = []
    for fname, label in [
        ("IDENTITY.md", "User Profile"),
        ("MEMORY.md",   "Long-term Memory"),
        ("GOALS.md",    "Active Goals"),
    ]:
        p = workspace_path / fname
        if p.exists():
            content = p.read_text(encoding="utf-8", errors="replace").strip()
            # Skip if only template/comment lines
            real_lines = [l for l in content.split("\n") if l.strip() and not l.strip().startswith("<!--")]
            if len(real_lines) > 2:
                parts.append(f"### {label}\n{content}")

    if not parts:
        return ""

    return (
        "\n\n---\n"
        "## Persistent Context (read every session)\n"
        + "\n\n".join(parts)
        + "\n---\n"
    )
