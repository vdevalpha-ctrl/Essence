#!/usr/bin/env python3
"""
Essence TUI 2.0
Full-featured terminal interface for AI agent management.
Implements the complete Essence TUI Design Specification.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

_WS = Path(__file__).resolve().parent
os.environ.setdefault("ESSENCE_WORKSPACE", str(_WS))

# ── Config ──────────────────────────────────────────────────────────────────────
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore


def _cfg() -> dict:
    p = _WS / "config.toml"
    if tomllib and p.exists():
        with open(p, "rb") as f:
            return tomllib.load(f)
    return {}


def _get_model() -> str:
    return os.environ.get("ESSENCE_MODEL", _cfg().get("inference", {}).get("model", "qwen3:4b"))


def _persist_inference(key: str, value: str) -> None:
    """Write a key=value pair under [inference] in config.toml."""
    cfg_path = _WS / "config.toml"
    if not cfg_path.exists():
        return
    try:
        lines = cfg_path.read_text(encoding="utf-8").splitlines()
        new_lines: list[str] = []
        replaced   = False
        in_section = False
        for line in lines:
            stripped = line.lstrip()
            if line.strip() == "[inference]":
                in_section = True
            elif line.strip().startswith("[") and line.strip() != "[inference]":
                in_section = False
            if (in_section and stripped.startswith(key) and
                    "=" in stripped and not stripped.startswith("#")):
                indent = line[: len(line) - len(stripped)]
                new_lines.append(f'{indent}{key:<9} = "{value}"')
                replaced = True
            else:
                new_lines.append(line)
        if not replaced:
            for i, line in enumerate(new_lines):
                if line.strip() == "[inference]":
                    new_lines.insert(i + 1, f'{key:<9} = "{value}"')
                    break
        cfg_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    except Exception:
        pass


def _persist_model(name: str) -> None:
    _persist_inference("model", name)


def _persist_backend(provider: str) -> None:
    _persist_inference("backend", provider)


def _persist_provider_key(provider: str, key: str, value: str) -> None:
    """Store a provider API key as an env-var-style line in config.toml."""
    cfg_path = _WS / "config.toml"
    try:
        text = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else ""
        section = f"[provider.{provider}]"
        key_line = f'{key} = "{value}"'
        if section in text:
            lines, in_sec = [], False
            replaced = False
            for line in text.splitlines():
                if line.strip() == section:
                    in_sec = True
                elif line.strip().startswith("[") and line.strip() != section:
                    in_sec = False
                if in_sec and line.lstrip().startswith(key) and "=" in line and not replaced:
                    lines.append(key_line)
                    replaced = True
                else:
                    lines.append(line)
            if not replaced:
                # append under section
                for i, line in enumerate(lines):
                    if line.strip() == section:
                        lines.insert(i + 1, key_line)
                        break
            cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            cfg_path.write_text(text + f"\n{section}\n{key_line}\n", encoding="utf-8")
    except Exception:
        pass


def _get_ollama() -> str:
    return os.environ.get(
        "OLLAMA_HOST", os.environ.get("ESSENCE_OLLAMA_HOST", "http://localhost:11434")
    )


# ── Paths ───────────────────────────────────────────────────────────────────────
_MEM = _WS / "memory"
_MEM.mkdir(exist_ok=True)
for _d in (_MEM / "sessions", _MEM / "snapshots"):
    _d.mkdir(exist_ok=True)

_DRAFT_FILE = _MEM / ".tui_draft"
_COST_FILE = _MEM / "cost_log.json"
_CRON_FILE = _MEM / "cron_jobs.json"
_SNAPSHOTS_DIR = _MEM / "snapshots"

# ── Themes ──────────────────────────────────────────────────────────────────────
THEMES: dict[str, dict[str, str]] = {
    "default-dark": dict(
        bg="#0B0C10", surf="#13151C", surf2="#1B1E29",
        acc="#00D4FF", acc2="#7B61FF",
        ok="#00FF88", warn="#FFB800", err="#FF4466",
        txt="#F0F2FF", txt2="#8892AA", bdr="#232640",
    ),
    "dracula": dict(
        bg="#282a36", surf="#1e1f29", surf2="#44475a",
        acc="#bd93f9", acc2="#ff79c6",
        ok="#50fa7b", warn="#f1fa8c", err="#ff5555",
        txt="#f8f8f2", txt2="#6272a4", bdr="#44475a",
    ),
    "nord": dict(
        bg="#2e3440", surf="#3b4252", surf2="#434c5e",
        acc="#88c0d0", acc2="#81a1c1",
        ok="#a3be8c", warn="#ebcb8b", err="#bf616a",
        txt="#eceff4", txt2="#d8dee9", bdr="#4c566a",
    ),
    "solarized-dark": dict(
        bg="#002b36", surf="#073642", surf2="#094a5c",
        acc="#2aa198", acc2="#268bd2",
        ok="#859900", warn="#b58900", err="#dc322f",
        txt="#fdf6e3", txt2="#93a1a1", bdr="#094a5c",
    ),
    "github-light": dict(
        bg="#ffffff", surf="#f6f8fa", surf2="#eaeef2",
        acc="#0969da", acc2="#8250df",
        ok="#1a7f37", warn="#9a6700", err="#cf222e",
        txt="#1f2328", txt2="#636c76", bdr="#d0d7de",
    ),
    "high-contrast": dict(
        bg="#000000", surf="#0a0a0a", surf2="#1a1a1a",
        acc="#ffff00", acc2="#ffffff",
        ok="#00ff00", warn="#ffaa00", err="#ff0000",
        txt="#ffffff", txt2="#cccccc", bdr="#ffffff",
    ),
}


# ── Global session state ────────────────────────────────────────────────────────
class _S:
    history: list[dict] = []
    session_id: str = uuid.uuid4().hex[:8]
    tok_in: int = 0
    tok_out: int = 0
    cost: float = 0.0
    cmd_hist: list[str] = []
    cmd_idx: int = -1
    connected: bool = False
    theme: str = "default-dark"
    model: str = _get_model()
    thinking: bool = False
    verbose: bool = False
    streaming: bool = False
    stream_start: float = 0.0
    temperature: float = 0.7
    max_tokens: int = 2048
    sys_override: str = ""        # extra system prompt injected this session
    pending_images: list = []     # images to attach on next send (populated by /image)
    json_mode: bool = False       # if True, force JSON output from the model
    reasoning_mode: str = ""      # "" | "8000" (Anthropic budget_tokens) | "high" (OpenAI)


S = _S()


def T(k: str) -> str:
    """Get current theme color."""
    return THEMES.get(S.theme, THEMES["default-dark"]).get(k, "#888888")


def _init_session() -> None:
    S.history = []
    soul = _WS / "SOUL.md"
    if soul.exists():
        S.history.append({"role": "system", "content": soul.read_text(encoding="utf-8")})


_init_session()


def _est_cost(ti: int, to: int, model: str) -> float:
    if "gpt-4o" in model:
        return ti * 5e-6 + to * 15e-6
    if "gpt-4" in model:
        return ti * 30e-6 + to * 60e-6
    if "claude" in model:
        return ti * 3e-6 + to * 15e-6
    return 0.0


def _load_cron() -> list[dict]:
    try:
        return json.loads(_CRON_FILE.read_text()) if _CRON_FILE.exists() else []
    except Exception:
        return []


def _save_cron(jobs: list[dict]) -> None:
    _CRON_FILE.write_text(json.dumps(jobs, indent=2))


def _notify(title: str, body: str) -> None:
    try:
        if sys.platform == "win32":
            script = (
                f'[System.Reflection.Assembly]::LoadWithPartialName("System.Windows.Forms");'
                f'$n=New-Object System.Windows.Forms.NotifyIcon;'
                f'$n.Icon=[System.Drawing.SystemIcons]::Information;'
                f'$n.Visible=$true;$n.ShowBalloonTip(3000,"{title}","{body}",0);'
                f"Start-Sleep -s 3;$n.Dispose()"
            )
            subprocess.Popen(
                ["powershell", "-WindowStyle", "Hidden", "-Command", script],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        elif sys.platform == "darwin":
            subprocess.Popen(
                ["osascript", "-e", f'display notification "{body}" with title "{title}"']
            )
        else:
            subprocess.Popen(["notify-send", title, body])
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════════
# Textual TUI
# ════════════════════════════════════════════════════════════════════════════════
try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, ScrollableContainer, Vertical
    from textual.reactive import reactive
    from textual.screen import ModalScreen
    from textual.widget import Widget
    from textual.widgets import DataTable, Input, Label, RichLog, Static
    from textual import work, on

    _TEXTUAL = True
except ImportError:
    _TEXTUAL = False

if _TEXTUAL:
    # ── Custom Input with command history ───────────────────────────────────────
    class HistoryInput(Input):
        def on_key(self, event: Any) -> None:
            if event.key == "up":
                if S.cmd_hist:
                    S.cmd_idx = min(S.cmd_idx + 1, len(S.cmd_hist) - 1)
                    self.value = S.cmd_hist[S.cmd_idx]
                    self.cursor_position = len(self.value)
                    event.stop()
            elif event.key == "down":
                if S.cmd_hist:
                    S.cmd_idx = max(S.cmd_idx - 1, -1)
                    self.value = S.cmd_hist[S.cmd_idx] if S.cmd_idx >= 0 else ""
                    self.cursor_position = len(self.value)
                    event.stop()

    # ── Help Screen ─────────────────────────────────────────────────────────────
    class HelpScreen(ModalScreen):
        BINDINGS = [("escape", "dismiss", "Close"), ("f1", "dismiss", "Close")]
        CSS = """
        HelpScreen { align: center middle; }
        #help-box {
            width: 80; height: auto; max-height: 42;
            background: #13151C; border: double #00D4FF; padding: 1 2;
        }
        """

        HELP_ROWS = [
            # ── Keyboard shortcuts ──────────────────────────────────────────────
            ("── Keyboard Shortcuts ──────────────────────────────────────────", ""),
            ("Ctrl+K",      "Open command palette (fuzzy-search all commands)"),
            ("Ctrl+B",      "Toggle sidebar (agent list, session info)"),
            ("Ctrl+S",      "Toggle split pane (cost / logs / resource view)"),
            ("Ctrl+D",      "Cost & token dashboard"),
            ("Ctrl+R",      "Cron job manager"),
            ("Ctrl+L",      "Model picker (list & switch Ollama models)"),
            ("Ctrl+P",      "Ping backend — check connectivity right now"),
            ("Ctrl+Z",      "Retry — resend last user message"),
            ("Ctrl+O",      "Expand tool outputs panel"),
            ("Ctrl+E",      "Export current chat (Markdown)"),
            ("Ctrl+Y",      "Copy last assistant response to clipboard"),
            ("F1",          "This help screen"),
            ("↑ / ↓",       "Scroll command history in the input box"),
            ("Esc",         "Close modal / cancel stream"),
            ("Ctrl+C × 2",  "Quit TUI (auto-saves session first)"),
            ("", ""),
            # ── Session ─────────────────────────────────────────────────────────
            ("── Session ────────────────────────────────────────────────────", ""),
            ("/clear",              "Save current session then start a fresh one"),
            ("/snapshot",           "Manually save a named snapshot  →  memory/snapshots/"),
            ("/archive",            "List all saved snapshots with timestamp & cost"),
            ("/restore <id>",       "Restore a snapshot into the active session (fuzzy match)"),
            ("/history [n]",        "Show last n turns of this session  (default 10)"),
            ("/forget [n]",         "Remove last n messages from active history  (default 1)"),
            ("/retry",              "Resend the last user message  (Ctrl+Z)"),
            ("/export md",          "Export chat as Markdown  →  chat_YYYYMMDD_HHMMSS.md"),
            ("/export json",        "Export chat as JSON  →  chat_YYYYMMDD_HHMMSS.json"),
            ("/status",             "Session info: provider, model, tokens, cost, kernel state"),
            ("", ""),
            # ── Model & Provider ─────────────────────────────────────────────────
            ("── Model & Provider ───────────────────────────────────────────", ""),
            ("/model [name]",                     "Show current model, or switch to <name> (persisted)"),
            ("/models",                           "List available Ollama models with size"),
            ("/provider list",                    "All providers: configured status & cost per 1M tokens"),
            ("/provider use <id>",                "Switch active provider  →  ollama groq openai anthropic mistral deepseek gemini"),
            ("/provider set <id> api_key <val>",  "Store API key for a provider (env + config.toml)"),
            ("/provider models [id]",             "Discover available models from a provider (or active)"),
            ("/provider status",                  "Per-provider: calls, tokens, avg latency, error count"),
            ("/temp [0.0–2.0]",                   "Show or set generation temperature  (default 0.7)"),
            ("/tokens [n]",                       "Show or set max_tokens limit  (default 2048)"),
            ("", ""),
            # ── Inference Tuning ─────────────────────────────────────────────────
            ("── Inference Tuning ───────────────────────────────────────────", ""),
            ("/sys [prompt]",       "Set an extra system prompt injected this session  (empty = clear)"),
            ("/think on|off",       "Toggle extended thinking / reasoning mode"),
            ("/think <budget>",     "Set Anthropic thinking budget in tokens  e.g. /think 16000"),
            ("/think high|medium|low", "Set OpenAI o-series reasoning effort"),
            ("/verbose on|off",     "Show token counts and latency after each response"),
            ("", ""),
            # ── Memory & Skills ─────────────────────────────────────────────────
            ("── Memory & Skills ────────────────────────────────────────────", ""),
            ("/memory list [n]",    "Show recent gravity-memory entries  (default 20)"),
            ("/memory search <q>",  "Search memory for keyword or phrase"),
            ("/memory clear",       "Wipe the in-session episodic context (not the DB)"),
            ("/skills list",        "All skills: enabled/disabled, source (builtin / store)"),
            ("/skills enable <id>", "Enable a skill by id"),
            ("/skills disable <id>","Disable a skill by id"),
            ("", ""),
            # ── System & Governance ─────────────────────────────────────────────
            ("── System & Governance ────────────────────────────────────────", ""),
            ("/config show",          "Print full config.toml"),
            ("/config get <key>",     "Read one config value  e.g. /config get inference.model"),
            ("/config set <key> <v>", "Write one config value  e.g. /config set agent.autonomy_level 2"),
            ("/kernel",               "Kernel cognitive state: queue depth, plan hits, cost"),
            ("/plugin list",          "Installed plugins with enabled status & permissions"),
            ("/plugin enable <id>",   "Enable a plugin"),
            ("/plugin disable <id>",  "Disable a plugin"),
            ("", ""),
            ("── External Services ──────────────────────────────────────────", ""),
            ("/ingest <url>",              "Ingest an API/service — learns its endpoints as live callable tools"),
            ("/services list",             "All learned services with endpoint counts"),
            ("/services show <id>",        "Full endpoint detail for a service"),
            ("/services forget <id>",      "Remove a learned service and unregister its tools"),
            ("/services auth <id> <k> <v>","Store API credentials for a service"),
            ("/services search <query>",   "Find services relevant to a keyword"),
            ("", ""),
            ("── Context Management ─────────────────────────────────────────", ""),
            ("/compress [topic]",          "Force-compress context window; optional topic focuses the summary"),
            ("/usage",                     "Token & cost summary + live rate limit state per provider"),
            ("/insights [days]",           "Session analytics: tokens, costs, models, skills, activity (default 30d)"),
            ("/title [text]",              "Show auto-generated session title, or set a custom one"),
            ("/pool list|add|reset|rotate","Manage multi-key credential pool for provider failover"),
            ("/adapt [show|reset|on|off]", "Adaptive profile: learned verbosity, style, model performance"),
            ("/goals [list|save|done|clear]","Pending goals resurfaced across sessions"),
            ("/image <path|url>",          "Attach an image to the next message (vision models)"),
            ("/json [on|off]",             "Toggle JSON output mode — model returns raw JSON"),
            ("", ""),
            ("── MCP (Model Context Protocol) ──────────────────────────────", ""),
            ("/mcp list",                  "Show all registered MCP servers and their tool counts"),
            ("/mcp enable <id>",           "Enable an MCP server so its tools are available to the model"),
            ("/mcp disable <id>",          "Disable an MCP server"),
            ("/mcp connect <url>",         "Add and connect a new MCP server (stdio or HTTP)"),
            ("/mcp refresh",               "Re-register all enabled MCP servers' tools in this session"),
            ("", ""),
            ("/audit",                "Audit trail: approvals, violations, model switches"),
            ("/ping",                 "Check backend connectivity  (Ctrl+P)"),
            ("", ""),
            # ── Interface ───────────────────────────────────────────────────────
            ("── Interface ──────────────────────────────────────────────────", ""),
            ("/theme [name]",         "Switch colour theme  →  default-dark  dracula  nord  solarized-dark  github-light  high-contrast"),
            ("/resource",             "System resource panel: CPU, RAM, disk"),
            ("/cost",                 "Cost & token dashboard (split pane)"),
            ("/logs",                 "Recent log lines (split pane)"),
            ("/cron",                 "Cron job manager"),
            ("/cron add <name> <schedule>", "Add a cron job  e.g. /cron add backup 0 2 * * *"),
            ("/attach <file>",        "Paste a file's content into the next message (max 8 KB)"),
            ("/room create|add|list", "Agent collaboration rooms"),
            ("/webhook",              "WebHook debugger"),
            ("/help",                 "This screen"),
            ("/quit",                 "Exit TUI (auto-saves session)"),
        ]

        def compose(self) -> ComposeResult:
            t = T
            with ScrollableContainer(id="help-box"):
                yield Static(f"[bold {t('acc')}]Essence TUI — Keyboard & Command Reference[/]")
                yield Static(f"[{t('txt2')}]Type any slash command in the input box. Commands marked (persisted) survive restarts.[/]")
                yield Static("")
                for k, v in self.HELP_ROWS:
                    if not k:
                        yield Static("")
                    elif v == "":
                        # Section header — starts with "──"
                        yield Static(f"\n[bold {t('acc')}]{k}[/]")
                    else:
                        yield Static(f"  [{t('acc')}]{k:<42}[/] [{t('txt')}]{v}[/]")
                yield Static("")
                yield Static(f"[{t('txt2')}]Press Esc or F1 to close  ·  /quit to exit[/]")

    # ── Command Palette ─────────────────────────────────────────────────────────
    class CommandPaletteScreen(ModalScreen):
        BINDINGS = [("escape", "dismiss", "Close")]
        CSS = """
        CommandPaletteScreen { align: center middle; }
        #palette-box {
            width: 72; height: auto; max-height: 32;
            background: #13151C; border: solid #00D4FF; padding: 1 2;
        }
        #palette-input { margin: 0 0 1 0; }
        """

        COMMANDS = [
            ("/help",                   "Keyboard & command reference"),
            ("/status",                 "Session: provider, model, tokens, cost, kernel state"),
            ("/ping",                   "Check backend connectivity right now"),
            ("/clear",                  "Save session then start a fresh one"),
            ("/retry",                  "Resend the last user message"),
            ("/forget",                 "Remove last message from history"),
            ("/history",                "Show last 10 turns"),
            ("/snapshot",               "Save a named session snapshot"),
            ("/archive",                "List all snapshots"),
            ("/export md",              "Export chat as Markdown"),
            ("/export json",            "Export chat as JSON"),
            ("/models",                 "List available Ollama models"),
            ("/provider list",          "All providers: status & cost"),
            ("/provider use groq",      "Switch to Groq (fast, free tier)"),
            ("/provider use openai",    "Switch to OpenAI"),
            ("/provider use anthropic", "Switch to Anthropic"),
            ("/provider use ollama",    "Switch back to local Ollama"),
            ("/provider models",        "Discover models from active provider"),
            ("/provider status",        "Per-provider usage stats"),
            ("/temp 0.7",               "Set generation temperature"),
            ("/tokens 2048",            "Set max tokens per response"),
            ("/sys",                    "Set extra system prompt for this session"),
            ("/memory list",            "Recent memory entries"),
            ("/memory search <query>",  "Search memory"),
            ("/config show",            "Print config.toml"),
            ("/config set <key> <val>", "Edit a config value inline"),
            ("/kernel",                 "Kernel cognitive state & stats"),
            ("/skills list",            "All skills: enabled/disabled"),
            ("/plugin list",            "Installed plugins"),
            ("/ingest <url>",           "Learn an API service from a URL"),
            ("/services list",          "All learned services"),
            ("/services show <id>",     "Service endpoint detail"),
            ("/services forget <id>",   "Remove a learned service"),
            ("/services auth <id> api_key <key>", "Store API key for a service"),
            ("/audit",                  "Audit trail: approvals, violations"),
            ("/cost",                   "Cost & token dashboard"),
            ("/logs",                   "Recent log lines"),
            ("/resource",               "System resource stats"),
            ("/cron",                   "Cron job manager"),
            ("/attach <file>",          "Paste file content into next message"),
            ("/theme default-dark",     "Default dark theme"),
            ("/theme dracula",          "Dracula theme"),
            ("/theme nord",             "Nord theme"),
            ("/theme solarized-dark",   "Solarized dark theme"),
            ("/theme high-contrast",    "High contrast theme"),
            ("/think on",               "Enable extended thinking"),
            ("/verbose on",             "Show token counts after responses"),
            ("/adapt show",             "Show adaptive profile: verbosity, style, expertise"),
            ("/adapt reset",            "Reset all learned adaptive preferences"),
            ("/goals list",             "Pending goals carried across sessions"),
            ("/goals save <text>",      "Manually save a pending goal"),
            ("/goals clear",            "Clear all pending goals"),
            ("/image <path|url>",       "Attach image to next message (vision)"),
            ("/image clear",            "Discard queued images"),
            ("/json",                   "Toggle JSON output mode (on/off)"),
            ("/routing",                "Show last routing decision (task type, provider chosen)"),
            ("/routing off",            "Disable dynamic routing (use configured provider only)"),
            ("/routing on",             "Re-enable dynamic routing"),
            ("/voice status",           "Check voice I/O availability"),
            ("/voice say <text>",       "Speak text aloud via TTS"),
            ("/voice listen",           "Capture microphone input via STT"),
            ("/quit",                   "Exit TUI (auto-saves session)"),
        ]

        def compose(self) -> ComposeResult:
            t = T
            with ScrollableContainer(id="palette-box"):
                yield Static(f"[bold {t('acc')}]Command Palette[/]  [{t('txt2')}]↑↓ navigate · Enter select · Esc close[/]")
                yield HistoryInput(placeholder="Search commands...", id="palette-input")
                for cmd, desc in self.COMMANDS:
                    yield Static(
                        f"  [{t('acc')}]{cmd:<32}[/] [{t('txt2')}]{desc}[/]",
                        classes="cmd-item",
                        id=f"pi-{cmd.strip('/').replace(' ', '-').replace('<','').replace('>','')}",
                    )

        def on_input_changed(self, ev: Input.Changed) -> None:
            q = ev.value.lower()
            for cmd, desc in self.COMMANDS:
                wid = f"pi-{cmd.strip('/').replace(' ', '-').replace('<','').replace('>','')}"
                try:
                    w = self.query_one(f"#{wid}", Static)
                    w.display = not q or q in cmd or q in desc.lower()
                except Exception:
                    pass

        def on_input_submitted(self, ev: Input.Submitted) -> None:
            # If there's a value typed that looks like a command, use it
            if ev.value.startswith("/"):
                self.dismiss(ev.value)
                return
            # Otherwise find first visible item
            for cmd, _desc in self.COMMANDS:
                wid = f"pi-{cmd.strip('/').replace(' ', '-').replace('<','').replace('>','')}"
                try:
                    w = self.query_one(f"#{wid}", Static)
                    if w.display:
                        self.dismiss(cmd)
                        return
                except Exception:
                    pass
            self.dismiss(None)

        def on_static_click(self, ev: Any) -> None:
            import re
            text = str(ev.widget.renderable)
            m = re.search(r"(/\S+(?:\s+\S+)*)", text)
            if m:
                # Extract just the command part (before the description spacing)
                parts = m.group(1).strip().split()
                # Take up to 3 parts for commands like /export md
                cmd = " ".join(parts[:3]).rstrip("]")
                self.dismiss(cmd)

    # ── App Header ──────────────────────────────────────────────────────────────
    class AppHeader(Widget):
        connected: reactive[bool] = reactive(False)
        model_name: reactive[str] = reactive("")
        cost_str: reactive[str] = reactive("$0.0000")
        streaming: reactive[bool] = reactive(False)
        DEFAULT_CSS = "AppHeader { height: 1; background: #13151C; padding: 0 1; }"

        def render(self) -> str:
            t = T
            dot = f"[{t('ok')}]●[/]" if self.connected else f"[{t('err')}]●[/]"
            state = f"[{t('warn')}]streaming[/]" if self.streaming else (
                f"[{t('ok')}]connected[/]" if self.connected else f"[{t('err')}]offline[/]"
            )
            model = self.model_name or S.model
            return (
                f" {dot} [bold {t('txt')}]Essence TUI[/] "
                f"[{t('txt2')}]·[/] agent:[{t('acc2')}]main[/] "
                f"[{t('txt2')}]·[/] model:[{t('acc')}]{model}[/] "
                f"[{t('txt2')}]·[/] cost:[{t('warn')}]{self.cost_str}[/] "
                f"[{t('txt2')}]·[/] {state}"
            )

    # ── Status Line ─────────────────────────────────────────────────────────────
    class StatusLine(Widget):
        connected: reactive[bool] = reactive(False)
        streaming: reactive[bool] = reactive(False)
        cron_count: reactive[int] = reactive(0)
        pending: reactive[int] = reactive(0)
        tok_in: reactive[int] = reactive(0)
        tok_out: reactive[int] = reactive(0)
        stream_secs: reactive[float] = reactive(0.0)
        DEFAULT_CSS = "StatusLine { height: 1; background: #13151C; padding: 0 1; }"

        def render(self) -> str:
            t = T
            dot = f"[{t('ok')}]●[/]" if self.connected else f"[{t('err')}]●[/]"
            parts = [dot]
            if self.streaming:
                parts.append(f"[{t('acc')}]streaming • {self.stream_secs:.0f}s[/]")
            if self.cron_count:
                parts.append(f"cron:{self.cron_count} active")
            if self.pending:
                parts.append(f"[{t('warn')}]approvals:{self.pending} pending[/]")
            parts.append(f"[{t('txt2')}]↑{self.tok_in} ↓{self.tok_out} tokens[/]")
            return "  ".join(parts)

    # ── Agent Sidebar ───────────────────────────────────────────────────────────
    class AgentSidebar(Widget):
        DEFAULT_CSS = """
        AgentSidebar {
            width: 22; height: 1fr;
            background: #13151C; border-right: solid #232640; padding: 0 1;
        }
        """
        active_agent: str = "main"
        active_session: str = "global"

        AGENTS = [
            {"id": "main", "label": "main", "status": "healthy"},
            {"id": "work", "label": "work", "status": "idle"},
            {"id": "test", "label": "test", "status": "idle"},
        ]
        SESSIONS = ["global", "debug", "review"]

        def compose(self) -> ComposeResult:
            yield Static(self._content(), id="sidebar-content")

        def _content(self) -> str:
            t = T
            STATUS_DOT = {
                "healthy": f"[{t('ok')}]●[/]",
                "slow": f"[{t('warn')}]●[/]",
                "error": f"[{t('err')}]●[/]",
                "idle": f"[{t('txt2')}]○[/]",
            }
            lines = [f"[bold {t('acc')}]Agents[/]", ""]
            for ag in self.AGENTS:
                dot = STATUS_DOT.get(ag["status"], f"[{t('txt2')}]○[/]")
                arrow = "► " if ag["id"] == self.active_agent else "  "
                color = t("acc") if ag["id"] == self.active_agent else t("txt")
                lines.append(f" {dot} [{color}]{arrow}{ag['label']}[/]")
            lines += ["", f"[bold {t('acc2')}]Sessions[/]", ""]
            for sess in self.SESSIONS:
                arrow = "► " if sess == self.active_session else "  "
                color = t("acc2") if sess == self.active_session else t("txt2")
                lines.append(f"  [{color}]{arrow}{sess}[/]")
            lines += ["", f"[{t('bdr')}]{'─' * 18}[/]"]
            lines += [f"  [{t('txt2')}][N] New Chat[/]"]
            return "\n".join(lines)

        def refresh_content(self) -> None:
            try:
                self.query_one("#sidebar-content", Static).update(self._content())
            except Exception:
                pass

    # ── Second / Split Pane ─────────────────────────────────────────────────────
    class SecondPane(Widget):
        DEFAULT_CSS = """
        SecondPane {
            height: 14; border-top: solid #232640;
            background: #13151C; display: none; padding: 0 1;
        }
        SecondPane.visible { display: block; }
        """

        def compose(self) -> ComposeResult:
            yield RichLog(id="sp-log", markup=True, highlight=False, wrap=True)

        def show_mode(self, mode: str) -> None:
            self.add_class("visible")
            log = self.query_one("#sp-log", RichLog)
            log.clear()
            getattr(self, f"_render_{mode}", self._render_unknown)(log)

        def _render_unknown(self, log: RichLog) -> None:
            log.write(f"[{T('txt2')}]Unknown mode[/]")

        def _render_cost(self, log: RichLog) -> None:
            t = T
            log.write(f"[bold {t('acc')}]💰  Cost Dashboard[/]  [{t('txt2')}]session {S.session_id}[/]")
            log.write("")
            log.write(f"  [{t('txt2')}]Tokens in    [/][{t('acc')}]{S.tok_in:>10,}[/]")
            log.write(f"  [{t('txt2')}]Tokens out   [/][{t('acc2')}]{S.tok_out:>10,}[/]")
            log.write(f"  [{t('txt2')}]Total        [/][{t('txt')}]{S.tok_in + S.tok_out:>10,}[/]")
            log.write(f"  [{t('txt2')}]Session cost [/][{t('warn')}]${S.cost:.6f}[/]")
            try:
                hist = json.loads(_COST_FILE.read_text()) if _COST_FILE.exists() else []
                if hist:
                    total = sum(e.get("cost", 0) for e in hist)
                    log.write(f"  [{t('txt2')}]All-time     [/][{t('warn')}]${total:.4f}[/]  [{t('txt2')}]({len(hist)} sessions)[/]")
            except Exception:
                pass

        def _render_cron(self, log: RichLog) -> None:
            t = T
            jobs = _load_cron()
            log.write(f"[bold {t('acc')}]⏰  Cron Jobs[/]  [{t('txt2')}]({len(jobs)} total)[/]")
            log.write("")
            if not jobs:
                log.write(f"  [{t('txt2')}]No jobs defined. Use /cron add <name> <schedule>[/]")
                return
            log.write(f"  [{t('txt2')}]{'NAME':<16}{'SCHEDULE':<14}{'LAST RUN':<20}STATUS[/]")
            log.write(f"  [{t('bdr')}]{'─' * 62}[/]")
            for j in jobs:
                en = t("ok") if j.get("enabled", True) else t("txt2")
                sc = {"ok": t("ok"), "error": t("err"), "running": t("warn")}.get(
                    j.get("last_status", ""), t("txt2")
                )
                log.write(
                    f"  [{en}]{j.get('name', '?'):<16}[/]"
                    f"[{t('txt2')}]{j.get('schedule', ''):<14}{j.get('last_run', 'never'):<20}[/]"
                    f"[{sc}]{j.get('last_status', 'idle')}[/]"
                )

        def _render_logs(self, log: RichLog) -> None:
            t = T
            log.write(f"[bold {t('acc')}]📋  Recent Logs[/]")
            log.write("")
            for candidate in (_WS / "server.log", _WS / "server_service.log", _WS / "essence.log"):
                if candidate.exists():
                    try:
                        lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()[-22:]
                        for line in lines:
                            c = (
                                t("err") if any(w in line for w in ("ERROR", "error", "CRITICAL"))
                                else t("warn") if any(w in line for w in ("WARN", "WARNING"))
                                else t("txt2")
                            )
                            log.write(f"[{c}]{line[:110]}[/]")
                        return
                    except Exception as e:
                        log.write(f"[{t('err')}]Read error: {e}[/]")
                        return
            log.write(f"  [{t('txt2')}]No log file found.[/]")

        def _render_resource(self, log: RichLog) -> None:
            t = T
            log.write(f"[bold {t('acc')}]💻  System Resources[/]")
            log.write("")
            try:
                import psutil  # type: ignore

                def bar(pct: float, w: int = 28) -> str:
                    n = int(pct / 100 * w)
                    c = t("err") if pct > 90 else t("warn") if pct > 70 else t("ok")
                    return f"[{c}]{'█' * n}{'░' * (w - n)}[/] {pct:5.1f}%"

                cpu = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()
                dsk = psutil.disk_usage("/")
                net = psutil.net_io_counters()
                log.write(f"  [{t('txt')}]CPU    [/]{bar(cpu)}")
                log.write(
                    f"  [{t('txt')}]Memory [/]{bar(mem.percent)}  "
                    f"[{t('txt2')}]{mem.used // 1024**3}GB / {mem.total // 1024**3}GB[/]"
                )
                log.write(
                    f"  [{t('txt')}]Disk   [/]{bar(dsk.percent)}  "
                    f"[{t('txt2')}]{dsk.used // 1024**3}GB / {dsk.total // 1024**3}GB[/]"
                )
                log.write(
                    f"  [{t('txt')}]Network[/] "
                    f"[{t('acc')}]↑{net.bytes_sent // 1024:,}KB[/]  "
                    f"[{t('acc2')}]↓{net.bytes_recv // 1024:,}KB[/]"
                )
            except ImportError:
                log.write(f"  [{t('txt2')}]psutil not installed. Run: python essence.py install[/]")
            except Exception as e:
                log.write(f"  [{t('err')}]Error: {e}[/]")

    # ── Main App ─────────────────────────────────────────────────────────────────
    class EssenceTUI(App):
        """Essence TUI 2.0 — Full-featured AI terminal interface."""

        TITLE = "Essence TUI"

        CSS = """
        Screen { background: #0B0C10; color: #F0F2FF; }

        #main-layout { height: 1fr; }

        #content { width: 1fr; height: 1fr; }

        #chat-log {
            height: 1fr;
            background: #0B0C10;
            padding: 0 1;
            border: blank;
        }

        #typing-row {
            height: 1;
            background: #0B0C10;
            padding: 0 2;
            display: none;
        }
        #typing-row.visible { display: block; }

        #composer {
            height: auto; max-height: 6;
            background: #13151C;
            border-top: solid #232640;
            padding: 0 1;
        }

        HistoryInput, Input {
            background: #0B0C10;
            border: solid #232640;
            color: #F0F2FF;
            margin: 0 0 0 0;
        }
        HistoryInput:focus, Input:focus { border: solid #00D4FF; }

        #send-hint { width: auto; color: #8892AA; padding: 0 1; content-align: center middle; }

        #footer-hints { height: 1; background: #13151C; color: #8892AA; padding: 0 1; }
        """

        BINDINGS = [
            Binding("ctrl+k", "command_palette", "Commands"),
            Binding("ctrl+b", "toggle_sidebar", "Sidebar"),
            Binding("ctrl+s", "toggle_split", "Split"),
            Binding("ctrl+d", "mode_cost", "Cost"),
            Binding("ctrl+r", "mode_cron", "Cron"),
            Binding("ctrl+l", "list_models", "Models"),
            Binding("ctrl+p", "ping_backend", "Ping"),
            Binding("ctrl+z", "retry_last", "Retry"),
            Binding("ctrl+o", "toggle_tool_output", "Tools"),
            Binding("ctrl+e", "export_chat", "Export"),
            Binding("ctrl+y", "copy_last", "Copy"),
            Binding("f1", "show_help", "Help"),
            Binding("escape", "close_pane", "Close", show=False),
        ]

        _sidebar_on: bool = True
        _split_on: bool = False
        _split_mode: str = "cost"
        _typing_worker: Any = None

        # ── Compose ─────────────────────────────────────────────────────────────
        def compose(self) -> ComposeResult:
            t = T
            yield AppHeader(id="app-header")
            with Horizontal(id="main-layout"):
                yield AgentSidebar(id="sidebar")
                with Vertical(id="content"):
                    yield RichLog(id="chat-log", markup=True, highlight=False, wrap=True)
                    yield Static("", id="typing-row")
                    yield SecondPane(id="second-pane")
                    yield StatusLine(id="status-line")
                    with Vertical(id="composer"):
                        with Horizontal():
                            yield HistoryInput(
                                placeholder=f"Message {S.model}…  (/help for commands)",
                                id="msg-input",
                            )
                            yield Static(f"[{t('txt2')}][Enter] send[/]", id="send-hint")
                        yield Static(
                            f"[{t('txt2')}][Ctrl+K] palette  [Ctrl+B] sidebar  "
                            f"[Ctrl+S] split  [Ctrl+D] cost  [Ctrl+R] cron  [F1] help[/]",
                            id="footer-hints",
                        )

        # ── Mount ────────────────────────────────────────────────────────────────
        def on_mount(self) -> None:
            draft = _DRAFT_FILE.read_text(encoding="utf-8") if _DRAFT_FILE.exists() else ""
            if draft:
                self.query_one("#msg-input", HistoryInput).value = draft

            log = self.query_one("#chat-log", RichLog)
            t = T
            log.write(f"[bold {t('acc')}]╔{'═' * 62}╗[/]")
            log.write(f"[bold {t('acc')}]║  Essence TUI 2.0 — Your Private AI{' ' * 27}║[/]")
            log.write(f"[bold {t('acc')}]╚{'═' * 62}╝[/]")
            log.write("")
            log.write(f"  [{t('txt2')}]Model   [/][{t('acc')}]{S.model}[/]")
            log.write(f"  [{t('txt2')}]Ollama  [/][{t('acc2')}]{_get_ollama()}[/]")
            log.write(f"  [{t('txt2')}]Session [/][{t('txt')}]{S.session_id}[/]")
            log.write(f"  [{t('txt2')}]Theme   [/][{t('txt')}]{S.theme}[/]")
            log.write("")
            log.write(f"  [{t('txt2')}]Type a message or /help for commands.[/]")
            log.write("")

            # Resurface pending goals from previous sessions
            try:
                from server.goal_tracker import get_goal_tracker
                _tracker = get_goal_tracker()
                _pending = _tracker.get_pending(max_resurface=3)
                if _pending:
                    _notice = _tracker.format_resurface(_pending)
                    for _line in _notice.split("\n"):
                        log.write(f"[{t('warn')}]{_line}[/]")
                    _tracker.mark_resurfaced(*[g.id for g in _pending])
            except Exception:
                pass

            self._check_connection()
            jobs = _load_cron()
            self.query_one("#status-line", StatusLine).cron_count = sum(
                1 for j in jobs if j.get("enabled", True)
            )

        # ── Connection check ─────────────────────────────────────────────────────
        @work(exclusive=False, thread=False)
        async def _check_connection(self) -> None:
            try:
                import httpx  # type: ignore

                async with httpx.AsyncClient(timeout=3) as c:
                    r = await c.get(f"{_get_ollama()}/api/tags")
                    S.connected = r.status_code == 200
            except Exception:
                S.connected = False

            hdr = self.query_one("#app-header", AppHeader)
            sl = self.query_one("#status-line", StatusLine)
            hdr.connected = S.connected
            hdr.model_name = S.model
            sl.connected = S.connected

            if not S.connected:
                log = self.query_one("#chat-log", RichLog)
                t = T
                log.write(f"  [{t('err')}]⚠  Ollama not reachable at {_get_ollama()}[/]")
                log.write(f"  [{t('txt2')}]Ensure Ollama is running and '{S.model}' is pulled.[/]")
                log.write("")

        # ── Input handling ───────────────────────────────────────────────────────
        def on_input_submitted(self, ev: Input.Submitted) -> None:
            if ev.input.id != "msg-input":
                return
            msg = ev.value.strip()
            if not msg:
                return
            ev.input.value = ""
            try:
                _DRAFT_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            S.cmd_hist.insert(0, msg)
            S.cmd_idx = -1
            if msg.startswith("/"):
                self._slash(msg)
            else:
                self._send(msg)

        def on_input_changed(self, ev: Input.Changed) -> None:
            if ev.input.id == "msg-input" and ev.value:
                try:
                    _DRAFT_FILE.write_text(ev.value, encoding="utf-8")
                except Exception:
                    pass

        # ── Slash command dispatcher ─────────────────────────────────────────────
        def _slash(self, text: str) -> None:
            parts = text.split(maxsplit=2)
            cmd = parts[0].lower()
            a1 = parts[1] if len(parts) > 1 else ""
            a2 = parts[2] if len(parts) > 2 else ""
            log = self.query_one("#chat-log", RichLog)
            t = T

            match cmd:
                case "/help" | "/h" | "/?":
                    self.action_show_help()
                case "/quit" | "/exit":
                    self.exit()
                case "/clear":
                    # Auto-save current session before wiping
                    self._autosave_session()
                    log.clear()
                    _init_session()
                    S.session_id = uuid.uuid4().hex[:8]
                    log.write(f"[{t('acc')}]New session: {S.session_id}[/]")
                case "/theme":
                    self._cmd_theme(a1, log)
                case "/model":
                    self._cmd_model(a1, log)
                case "/models":
                    self.action_list_models()
                case "/cost":
                    self.action_mode_cost()
                case "/usage":
                    self._cmd_usage(log)
                case "/cron":
                    self._cmd_cron(a1, a2, log)
                case "/logs":
                    self.action_mode_logs()
                case "/resource":
                    self.action_mode_resource()
                case "/snapshot":
                    self._cmd_snapshot(log)
                case "/archive":
                    self._cmd_archive(log)
                case "/restore":
                    self._cmd_restore(a1, log)
                case "/status":
                    self._cmd_status(log)
                case "/export":
                    self._cmd_export(a1 or "md", log)
                case "/attach":
                    self._cmd_attach(a1, log) if a1 else log.write(f"[{t('txt2')}]Usage: /attach <file>[/]")
                case "/think":
                    S.thinking = a1 != "off"
                    log.write(f"[{t('acc')}]Thinking: {'on' if S.thinking else 'off'}[/]")
                case "/verbose":
                    S.verbose = a1 != "off"
                    log.write(f"[{t('acc')}]Verbose: {'on' if S.verbose else 'off'}[/]")
                case "/room":
                    self._cmd_room(a1, a2, log)
                case "/webhook":
                    log.write(f"[bold {t('acc')}]WebHook Debugger[/]  [{t('txt2')}]No registered endpoints.[/]")
                case "/audit":
                    self._cmd_audit(log)
                case "/provider":
                    self._cmd_provider(a1, (a2.split() if a2 else []), log)
                case "/skills":
                    self._cmd_skills(a1, a2, log)
                case "/ping":
                    self._cmd_ping(log)
                case "/retry":
                    self._cmd_retry(log)
                case "/history":
                    self._cmd_history(a1, log)
                case "/forget":
                    self._cmd_forget(a1, log)
                case "/sys":
                    self._cmd_sys((" ".join(parts[1:]) if len(parts) > 1 else ""), log)
                case "/temp":
                    self._cmd_temp(a1, log)
                case "/tokens":
                    self._cmd_tokens(a1, log)
                case "/memory":
                    self._cmd_memory(a1, a2, log)
                case "/config":
                    self._cmd_config(a1, a2, log)
                case "/kernel":
                    self._cmd_kernel(log)
                case "/plugin":
                    self._cmd_plugin(a1, a2, log)
                case "/ingest":
                    self._cmd_ingest(a1, log)
                case "/services":
                    self._cmd_services(a1, (" ".join(parts[2:]) if len(parts) > 2 else ""), log)
                case "/compress":
                    self._cmd_compress((" ".join(parts[1:]) if len(parts) > 1 else ""), log)
                case "/insights":
                    self._cmd_insights(a1, log)
                case "/pool":
                    self._cmd_pool(a1, (" ".join(parts[2:]) if len(parts) > 2 else ""), log)
                case "/title":
                    self._cmd_title((" ".join(parts[1:]) if len(parts) > 1 else ""), log)
                case "/adapt":
                    self._cmd_adapt(a1, (" ".join(parts[2:]) if len(parts) > 2 else ""), log)
                case "/goals":
                    self._cmd_goals(a1, (" ".join(parts[2:]) if len(parts) > 2 else ""), log)
                case "/image":
                    self._cmd_image((" ".join(parts[1:]) if len(parts) > 1 else ""), log)
                case "/json":
                    self._cmd_json(a1, log)
                case "/think":
                    self._cmd_think(a1, log)
                case "/mcp":
                    self._cmd_mcp(a1, (" ".join(parts[2:]) if len(parts) > 2 else ""), log)
                case "/routing":
                    self._cmd_routing(a1, log)
                case "/voice":
                    self._cmd_voice(a1, (" ".join(parts[2:]) if len(parts) > 2 else ""), log)
                case "/deliver" | "/fast" | "/accessibility":
                    log.write(f"[{t('txt2')}]{cmd[1:].capitalize()} mode: {a1 or 'on'}[/]")
                case _:
                    log.write(f"[{t('warn')}]Unknown command:[/] {cmd}  [{t('txt2')}]— /help for reference[/]")

        # ── Slash command implementations ────────────────────────────────────────
        def _cmd_theme(self, name: str, log: RichLog) -> None:
            t = T
            if name in THEMES:
                S.theme = name
                log.write(f"[{T('acc')}]Theme: {name}[/]")
                try:
                    self.query_one("#sidebar", AgentSidebar).refresh_content()
                except Exception:
                    pass
            else:
                log.write(f"[{t('warn')}]Themes:[/] " + "  ".join(THEMES.keys()))

        def _cmd_model(self, name: str, log: RichLog) -> None:
            t = T
            if name:
                old_model = S.model
                S.model = name
                try:
                    self.query_one("#msg-input", HistoryInput).placeholder = f"Message {name}…"
                    self.query_one("#app-header", AppHeader).model_name = name
                except Exception:
                    pass
                # Persist model; keep the current backend (don't clobber a Groq/etc. session)
                _persist_model(name)
                try:
                    from server.kernel import get_kernel as _gk
                    _current_backend = _gk()._provider
                except Exception:
                    _current_backend = "ollama"
                _persist_backend(_current_backend)
                # Notify kernel so live session switches immediately
                try:
                    from server.kernel import get_kernel
                    get_kernel()._model = name
                except Exception:
                    pass
                # Audit log
                try:
                    from server.audit_logger import get_audit_logger
                    get_audit_logger().log_model_switch("ollama", old_model, "ollama", name, reason="user /model")
                except Exception:
                    pass
                log.write(f"[{t('acc')}]Model: {name}[/]  [{t('txt2')}](saved to config.toml)[/]")
            else:
                log.write(f"[{t('txt2')}]Current model: [/][{t('acc')}]{S.model}[/]  [{t('txt2')}](use /models to list)[/]")

        def _cmd_cron(self, sub: str, arg: str, log: RichLog) -> None:
            t = T
            if sub == "add":
                if not arg:
                    log.write(f"[{t('txt2')}]Usage: /cron add <name> <schedule>[/]  e.g. /cron add backup 0 2 * * *")
                    return
                ap = arg.split(maxsplit=1)
                name = ap[0]
                sched = ap[1] if len(ap) > 1 else "*/30 * * * *"
                jobs = _load_cron()
                jobs.append({"id": uuid.uuid4().hex[:6], "name": name, "schedule": sched,
                             "command": "", "enabled": True, "last_run": "", "last_status": "idle"})
                _save_cron(jobs)
                self.query_one("#status-line", StatusLine).cron_count = sum(
                    1 for j in jobs if j.get("enabled", True)
                )
                log.write(f"[{t('ok')}]Cron job added:[/] {name}  [{t('txt2')}][{sched}][/]")
            elif sub == "remove":
                jobs = [j for j in _load_cron() if j.get("name") != arg and j.get("id") != arg]
                _save_cron(jobs)
                log.write(f"[{t('ok')}]Removed cron job:[/] {arg}")
            else:
                self.action_mode_cron()

        def _autosave_session(self) -> None:
            """Silently persist current session to snapshots dir (no-op if empty)."""
            msgs = [m for m in S.history if m.get("role") != "system"]
            if not msgs:
                return
            try:
                snap_id = f"{S.session_id}_{int(time.time())}_auto"
                snap = {
                    "id": snap_id, "timestamp": time.time(), "model": S.model,
                    "messages": S.history, "tok_in": S.tok_in, "tok_out": S.tok_out,
                    "cost": S.cost, "auto": True,
                }
                (_SNAPSHOTS_DIR / f"{snap_id}.json").write_text(json.dumps(snap, indent=2))
            except Exception:
                pass

        def _cmd_snapshot(self, log: RichLog) -> None:
            t = T
            snap_id = f"{S.session_id}_{int(time.time())}"
            snap = {
                "id": snap_id, "timestamp": time.time(), "model": S.model,
                "messages": S.history, "tok_in": S.tok_in, "tok_out": S.tok_out, "cost": S.cost,
            }
            (_SNAPSHOTS_DIR / f"{snap_id}.json").write_text(json.dumps(snap, indent=2))
            log.write(f"[{t('ok')}]Snapshot saved:[/] {snap_id}")

        def _cmd_archive(self, log: RichLog) -> None:
            t = T
            snaps = sorted(_SNAPSHOTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not snaps:
                log.write(f"[{t('txt2')}]No snapshots. Use /snapshot to save.[/]")
                return
            log.write(f"\n[bold {t('acc')}]Session Archives ({len(snaps)})[/]")
            for p in snaps[:12]:
                try:
                    d = json.loads(p.read_text())
                    ts = datetime.fromtimestamp(d.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M")
                    msgs = len([m for m in d.get("messages", []) if m.get("role") != "system"])
                    log.write(
                        f"  [{t('acc')}]{d['id']:<32}[/] "
                        f"[{t('txt2')}]{ts}  {msgs} msgs  ${d.get('cost', 0):.4f}[/]"
                    )
                except Exception:
                    log.write(f"  [{t('txt2')}]{p.stem}[/]")
            log.write(f"  [{t('txt2')}]/restore <id> to restore[/]\n")

        def _cmd_restore(self, snap_id: str, log: RichLog) -> None:
            t = T
            path = _SNAPSHOTS_DIR / f"{snap_id}.json"
            if not path.exists():
                matches = list(_SNAPSHOTS_DIR.glob(f"*{snap_id}*.json"))
                path = matches[0] if matches else None
            if not path:
                log.write(f"[{t('err')}]Snapshot not found: {snap_id}[/]")
                return
            try:
                d = json.loads(path.read_text())
                S.history = d.get("messages", [])
                S.tok_in = d.get("tok_in", 0)
                S.tok_out = d.get("tok_out", 0)
                S.cost = d.get("cost", 0)
                log.write(f"[{t('ok')}]Restored: {d.get('id', snap_id)}[/]")
            except Exception as e:
                log.write(f"[{t('err')}]Restore failed: {e}[/]")

        def _cmd_status(self, log: RichLog) -> None:
            t = T
            log.write(f"\n[bold {t('acc')}]System Status[/]")
            log.write(f"  [{t('txt2')}]Ollama  [/][{t('ok') if S.connected else t('err')}]{'online' if S.connected else 'offline'}[/]  [{t('txt2')}]{_get_ollama()}[/]")
            log.write(f"  [{t('txt2')}]Model   [/][{t('acc')}]{S.model}[/]")
            log.write(f"  [{t('txt2')}]Session [/][{t('txt')}]{S.session_id}[/]")
            log.write(f"  [{t('txt2')}]Theme   [/][{t('txt')}]{S.theme}[/]")
            log.write(f"  [{t('txt2')}]Msgs    [/][{t('txt')}]{len([m for m in S.history if m.get('role') != 'system'])}[/]")
            log.write(f"  [{t('txt2')}]Tokens  [/][{t('txt')}]↑{S.tok_in}  ↓{S.tok_out}[/]")
            log.write(f"  [{t('txt2')}]Cost    [/][{t('warn')}]${S.cost:.6f}[/]\n")

        def _cmd_export(self, fmt: str, log: RichLog) -> None:
            t = T
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if fmt == "json":
                path = _WS / f"chat_{ts}.json"
                path.write_text(json.dumps(S.history, indent=2))
            else:
                path = _WS / f"chat_{ts}.md"
                lines = [f"# Essence Chat Export\n\nDate: {datetime.now().isoformat()}\nModel: {S.model}\n\n"]
                for m in S.history:
                    if m.get("role") == "system":
                        continue
                    role = "**You**" if m["role"] == "user" else "**Essence**"
                    lines.append(f"### {role}\n\n{m['content']}\n\n---\n\n")
                path.write_text("".join(lines))
            log.write(f"[{t('ok')}]Exported → {path.name}[/]")

        def _cmd_attach(self, path_str: str, log: RichLog) -> None:
            t = T
            p = Path(path_str).expanduser()
            if not p.exists():
                log.write(f"[{t('err')}]Not found: {path_str}[/]")
                return
            try:
                content = p.read_text(encoding="utf-8", errors="replace")[:8192]
                inp = self.query_one("#msg-input", HistoryInput)
                inp.value = f"[File: {p.name}]\n\n{content}"
                log.write(f"[{t('ok')}]Attached: {p.name} ({p.stat().st_size:,} bytes)[/]")
            except Exception as e:
                log.write(f"[{t('err')}]Attach failed: {e}[/]")

        def _cmd_room(self, sub: str, arg: str, log: RichLog) -> None:
            t = T
            if sub == "create":
                log.write(f"[{t('acc')}]Room created: {arg or 'unnamed'}[/]  [{t('txt2')}]Use /room add @agent to invite[/]")
            elif sub == "add":
                log.write(f"[{t('acc')}]Added {arg} to room[/]")
            elif sub == "list":
                log.write(f"[{t('txt2')}]No active rooms. /room create <name>[/]")
            else:
                log.write(f"[{t('txt2')}]Usage: /room create|add|list[/]")

        def _cmd_audit(self, log: RichLog) -> None:
            t = T
            log.write(f"\n[bold {t('acc')}]Audit Trail[/]")
            log.write(f"  [{t('txt2')}]Session:     {S.session_id}[/]")
            log.write(f"  [{t('txt2')}]Messages:    {len([m for m in S.history if m.get('role') == 'user'])} sent[/]")
            try:
                from server.audit_logger import get_audit_logger
                st = get_audit_logger().stats()
                log.write(f"  [{t('txt2')}]Approvals:   {st.get('denials', 0)} denied  |  {st.get('total', 0)} total events[/]")
                log.write(f"  [{t('txt2')}]Violations:  {st.get('violations', 0)}[/]")
            except Exception:
                log.write(f"  [{t('txt2')}]Approvals:   (audit logger unavailable)[/]")
            log.write(f"  [{t('txt2')}]Cost:        ${S.cost:.6f}[/]\n")

        def _cmd_provider(self, sub: str, args: list[str], log: RichLog) -> None:
            t = T
            from server.model_router import PROVIDER_BASE_URL, PROVIDER_COST_PER_1M
            all_providers = list(PROVIDER_BASE_URL.keys())

            if not sub or sub == "list":
                log.write(f"\n[bold {t('acc')}]Providers[/]")
                try:
                    from server.kernel import get_kernel
                    _k = get_kernel()
                    active = _k._provider
                    router = _k._router
                    for pid in all_providers:
                        cur = f"[{t('ok')}]◄ active[/]" if pid == active else ""
                        cfg_ok = router._provider_configured(pid)
                        status = f"[{t('ok')}]configured[/]" if cfg_ok else f"[{t('txt2')}]no key[/]"
                        cost_p, cost_c = PROVIDER_COST_PER_1M.get(pid, (0, 0))
                        cost_str = "free" if cost_p == 0 else f"${cost_p:.3f}/${cost_c:.3f}/1M"
                        log.write(f"  [{t('acc')}]{pid:<14}[/] {status:<22} [{t('txt2')}]{cost_str:<18}[/] {cur}")
                except Exception:
                    for pid in all_providers:
                        cost_p, cost_c = PROVIDER_COST_PER_1M.get(pid, (0, 0))
                        cost_str = "free" if cost_p == 0 else f"${cost_p:.3f}/${cost_c:.3f}/1M"
                        log.write(f"  [{t('acc')}]{pid:<14}[/] [{t('txt2')}]{cost_str}[/]")
                log.write(f"\n  [{t('txt2')}]/provider use <id>              — switch active provider[/]")
                log.write(f"  [{t('txt2')}]/provider models [id]           — discover available models[/]")
                log.write(f"  [{t('txt2')}]/provider set <id> api_key <key> — store API key[/]")
                log.write(f"  [{t('txt2')}]/provider status                — token & latency stats[/]\n")

            elif sub == "use":
                pid = args[0] if args else ""
                if pid not in all_providers:
                    log.write(f"[{t('err')}]Unknown provider: {pid}[/]  [{t('txt2')}]choices: {', '.join(all_providers)}[/]")
                    return
                try:
                    from server.kernel import get_kernel
                    _k = get_kernel()
                    old_provider = _k._provider
                    old_model    = _k._model
                    _k._provider = pid
                    # Switch to provider's default model if current model won't work there
                    if pid != "ollama":
                        new_model = _k._router._default_model(pid)
                        _k._model = new_model
                        S.model   = new_model
                        _persist_model(new_model)
                        _persist_backend(pid)
                        log.write(f"[{t('ok')}]Provider → {pid}[/]  [{t('txt2')}]model → {new_model}[/]  [{t('txt2')}](saved)[/]")
                    else:
                        _persist_backend("ollama")
                        log.write(f"[{t('ok')}]Provider → {pid}[/]  [{t('txt2')}]model → {_k._model}[/]  [{t('txt2')}](saved)[/]")
                    try:
                        from server.audit_logger import get_audit_logger
                        get_audit_logger().log_model_switch(
                            old_provider, old_model, pid, _k._model, reason="user /provider use"
                        )
                    except Exception:
                        pass
                except Exception as e:
                    log.write(f"[{t('err')}]Provider switch failed: {e}[/]")

            elif sub == "set":
                # /provider set <id> <key> <value>
                if len(args) < 3:
                    log.write(f"[{t('txt2')}]Usage: /provider set <provider> <key> <value>[/]")
                    log.write(f"  [{t('txt2')}]e.g.  /provider set groq api_key gsk_...[/]")
                    return
                pid, key, value = args[0], args[1], args[2]
                if pid not in all_providers:
                    log.write(f"[{t('err')}]Unknown provider: {pid}[/]")
                    return
                # Store in env so current process picks it up immediately
                import os as _os
                env_key = f"{pid.upper()}_API_KEY" if key == "api_key" else f"ESSENCE_{pid.upper()}_{key.upper()}"
                _os.environ[env_key] = value
                # Persist to config.toml
                _persist_provider_key(pid, key, value)
                # Refresh router's provider_cfg
                try:
                    from server.kernel import get_kernel
                    get_kernel()._router._provider_cfg.setdefault(pid, {})[key] = value
                except Exception:
                    pass
                masked = value[:4] + "..." + value[-4:] if len(value) > 12 else "***"
                log.write(f"[{t('ok')}]{pid}.{key} = {masked}[/]  [{t('txt2')}](saved to config.toml)[/]")

            elif sub == "models":
                # /provider models [id]  — discover available models from a provider
                pid = args[0] if args else None
                self._cmd_provider_models(pid, log)

            elif sub == "status":
                try:
                    from server.kernel import get_kernel
                    router = get_kernel()._router
                    log.write(f"\n[bold {t('acc')}]Provider Usage[/]")
                    for pid, u in router._usage.items():
                        avail = f"[{t('ok')}]ok[/]" if u.is_available() else f"[{t('err')}]limited[/]"
                        log.write(
                            f"  [{t('acc')}]{pid:<14}[/] {avail}  "
                            f"[{t('txt2')}]calls={u.call_count}  "
                            f"tokens={u.token_count}  "
                            f"avg={u.avg_latency_s:.2f}s  "
                            f"errors={u.error_count}[/]"
                        )
                    log.write("")
                except Exception as e:
                    log.write(f"[{t('err')}]Status unavailable: {e}[/]")
            else:
                log.write(f"[{t('txt2')}]Usage: /provider [list|use|set|status|models [id]][/]")

        def _cmd_provider_models(self, pid: str | None, log: RichLog) -> None:
            """Discover available models from a provider (or all configured providers)."""
            t = T
            from server.model_router import PROVIDER_BASE_URL, PROVIDER_COST_PER_1M

            def _api_key_for(provider: str) -> str:
                try:
                    from server.kernel import get_kernel
                    cfg = get_kernel()._router._provider_cfg.get(provider, {})
                    key = cfg.get("api_key", "")
                except Exception:
                    key = ""
                if not key:
                    import os as _os
                    key = _os.environ.get(
                        f"ESSENCE_{provider.upper()}_KEY",
                        _os.environ.get(f"{provider.upper()}_API_KEY", ""),
                    )
                return key

            async def _fetch_models(provider: str) -> list[str]:
                import httpx as _hx
                base = PROVIDER_BASE_URL.get(provider, "")
                if not base:
                    return []
                key = _api_key_for(provider)
                headers: dict = {}
                if key:
                    headers["Authorization"] = f"Bearer {key}"
                if provider == "anthropic":
                    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
                    url = "https://api.anthropic.com/v1/models"
                else:
                    url = base.rstrip("/") + "/models"
                try:
                    async with _hx.AsyncClient(timeout=10.0) as c:
                        r = await c.get(url, headers=headers)
                        if r.status_code != 200:
                            return []
                        data = r.json()
                        # OpenAI-compatible: {"data": [{"id": "...", ...}]}
                        if "data" in data:
                            return sorted(m.get("id", m.get("name", "")) for m in data["data"])
                        # Ollama: {"models": [{"name": "..."}]}
                        if "models" in data:
                            return sorted(m.get("name", m.get("id", "")) for m in data["models"])
                        # Anthropic: {"data": [...]} (same as OAI)
                        return []
                except Exception:
                    return []

            async def _discover():
                targets = [pid] if pid else [p for p in PROVIDER_BASE_URL if p != "ollama"] + ["ollama"]
                for provider in targets:
                    key = _api_key_for(provider)
                    configured = bool(key) or provider == "ollama"
                    if not configured:
                        log.write(f"  [{t('txt2')}]{provider:<14}[/] [{t('warn')}]no api_key — skipping[/]")
                        continue
                    log.write(f"  [{t('txt2')}]Fetching {provider}…[/]")
                    models = await _fetch_models(provider)
                    if not models:
                        log.write(f"  [{t('acc')}]{provider:<14}[/] [{t('err')}]unreachable or empty[/]")
                    else:
                        log.write(f"  [bold {t('acc')}]{provider}[/]  [{t('txt2')}]({len(models)} models)[/]")
                        for m in models:
                            log.write(f"    [{t('txt')}]{m}[/]")
                log.write("")

            log.write(f"\n[bold {t('acc')}]Provider Model Discovery[/]"
                      + (f"  — {pid}" if pid else "  — all configured"))
            import asyncio as _aio
            try:
                loop = _aio.get_event_loop()
                if loop.is_running():
                    loop.create_task(_discover())
                else:
                    loop.run_until_complete(_discover())
            except Exception as e:
                log.write(f"[{t('err')}]Discovery error: {e}[/]")

        def _cmd_skills(self, sub: str, arg: str, log: RichLog) -> None:
            t = T
            from pathlib import Path as _Path
            import sys as _sys
            _ws = _Path(__file__).resolve().parent
            _sys.path.insert(0, str(_ws))
            try:
                from server.skillstore import SkillStore
                from server.skill_orchestrator import BUILTIN_SKILLS
                ss = SkillStore(_ws / "memory" / "skills")
                store_skills = ss.list_all()
                store_ids = {s["id"] for s in store_skills}
                all_skills = list(store_skills)
                for sd in BUILTIN_SKILLS:
                    if sd.id not in store_ids:
                        all_skills.append({
                            "id": sd.id, "label": sd.name, "category": sd.category,
                            "enabled": sd.enabled, "_source": "in-process",
                        })
                if not sub or sub == "list":
                    log.write(f"\n[bold {t('acc')}]Skills[/]  [{t('txt2')}]{len(all_skills)} total · {sum(1 for s in all_skills if s.get('enabled'))} enabled[/]")
                    for s in sorted(all_skills, key=lambda x: x["id"]):
                        en_tag = f"[{t('ok')}]ON[/]" if s.get("enabled") else f"[{t('warn')}]OFF[/]"
                        src = s.get("_source", "builtin" if s.get("builtin") else "json_store")
                        log.write(f"  {en_tag}  [{t('acc')}]{s['id']:<20}[/] [{t('txt2')}]{s.get('label', s['id']):<28}[/] {s.get('category',''):<14} [{t('txt2')}]{src}[/]")
                    log.write(f"  [{t('txt2')}]Use: /skills enable|disable <id>[/]\n")
                elif sub in ("enable", "disable") and arg:
                    enabled = sub == "enable"
                    try:
                        ss.patch(arg, {"enabled": enabled})
                        log.write(f"  [{t('ok') if enabled else t('warn')}]{arg}: {'enabled' if enabled else 'disabled'}[/]")
                    except KeyError:
                        log.write(f"  [{t('err')}]Skill not found: {arg}[/]")
                else:
                    log.write(f"  [{t('txt2')}]Usage: /skills [list|enable|disable <id>][/]")
            except Exception as e:
                log.write(f"  [{t('err')}]Skills error: {e}[/]")

        # ── New commands ─────────────────────────────────────────────────────────

        def _cmd_ping(self, log: RichLog) -> None:
            t = T
            import httpx as _hx
            base = _get_ollama()
            log.write(f"[{t('txt2')}]Pinging {base} …[/]")
            try:
                r = _hx.get(f"{base}/api/tags", timeout=4)
                ms = int(r.elapsed.total_seconds() * 1000)
                models = len(r.json().get("models", []))
                S.connected = True
                log.write(f"  [{t('ok')}]online[/]  [{t('txt2')}]{ms}ms  {models} model(s)[/]")
            except Exception as e:
                S.connected = False
                log.write(f"  [{t('err')}]offline:[/] {e}")
            # Also check active cloud provider if not ollama
            try:
                from server.kernel import get_kernel
                prov = get_kernel()._provider
                if prov != "ollama":
                    from server.model_router import PROVIDER_BASE_URL
                    url = PROVIDER_BASE_URL.get(prov, "")
                    if url:
                        r2 = _hx.get(url.rstrip("/v1") or url, timeout=4)
                        log.write(f"  [{t('ok')}]{prov}[/]  [{t('txt2')}]{int(r2.elapsed.total_seconds()*1000)}ms[/]")
            except Exception:
                pass

        def _cmd_retry(self, log: RichLog) -> None:
            t = T
            user_msgs = [m for m in S.history if m.get("role") == "user"]
            if not user_msgs:
                log.write(f"[{t('warn')}]No messages to retry.[/]")
                return
            last_msg = user_msgs[-1]["content"]
            # Remove the last user + assistant pair from history before resending
            while S.history and S.history[-1].get("role") == "assistant":
                S.history.pop()
            if S.history and S.history[-1].get("role") == "user":
                S.history.pop()
            log.write(f"[{t('txt2')}]Retrying: {last_msg[:80]}{'…' if len(last_msg) > 80 else ''}[/]")
            self._send(last_msg)

        def _cmd_history(self, n_str: str, log: RichLog) -> None:
            t = T
            n = int(n_str) if n_str.isdigit() else 10
            turns = [m for m in S.history if m.get("role") in ("user", "assistant")]
            turns = turns[-n * 2:]
            if not turns:
                log.write(f"[{t('txt2')}]No history yet.[/]")
                return
            log.write(f"\n[bold {t('acc')}]Last {min(n, len(turns)//2 + 1)} turns[/]")
            for m in turns:
                role_tag = t("acc2") if m["role"] == "user" else t("ok")
                label    = "You" if m["role"] == "user" else "AI"
                body     = m["content"][:200].replace("\n", " ")
                suffix   = "…" if len(m["content"]) > 200 else ""
                log.write(f"  [{role_tag}]{label}:[/] [{t('txt')}]{body}{suffix}[/]")
            log.write("")

        def _cmd_forget(self, n_str: str, log: RichLog) -> None:
            t = T
            n = int(n_str) if n_str.isdigit() else 1
            removed = 0
            for _ in range(n):
                # Remove one user+assistant pair from the tail
                while S.history and S.history[-1].get("role") == "assistant":
                    S.history.pop(); removed += 1
                if S.history and S.history[-1].get("role") == "user":
                    S.history.pop(); removed += 1
            log.write(f"[{t('ok')}]Removed {removed} message(s) from history. {len([m for m in S.history if m.get('role') != 'system'])} remain.[/]")

        def _cmd_sys(self, prompt: str, log: RichLog) -> None:
            t = T
            S.sys_override = prompt.strip()
            if S.sys_override:
                # Inject into history as a system message (replace any prior override)
                S.history = [m for m in S.history if m.get("_sys_override") is not True]
                S.history.insert(0, {"role": "system", "content": S.sys_override, "_sys_override": True})
                log.write(f"[{t('ok')}]System prompt set:[/] {S.sys_override[:120]}{'…' if len(S.sys_override) > 120 else ''}")
            else:
                S.history = [m for m in S.history if m.get("_sys_override") is not True]
                log.write(f"[{t('txt2')}]System prompt cleared.[/]")

        def _cmd_temp(self, val: str, log: RichLog) -> None:
            t = T
            if val:
                try:
                    v = float(val)
                    if not 0.0 <= v <= 2.0:
                        raise ValueError
                    S.temperature = v
                    try:
                        from server.kernel import get_kernel
                        get_kernel()._router  # just verify kernel is up
                    except Exception:
                        pass
                    log.write(f"[{t('ok')}]Temperature → {v}[/]  [{t('txt2')}](affects next message)[/]")
                except ValueError:
                    log.write(f"[{t('err')}]Invalid value. Use a float between 0.0 and 2.0[/]")
            else:
                log.write(f"[{t('txt2')}]Temperature: [/][{t('acc')}]{S.temperature}[/]  [{t('txt2')}](0.0 = deterministic · 1.0 = default · 2.0 = creative)[/]")

        def _cmd_tokens(self, val: str, log: RichLog) -> None:
            t = T
            if val:
                try:
                    n = int(val)
                    if n < 64 or n > 32768:
                        raise ValueError
                    S.max_tokens = n
                    log.write(f"[{t('ok')}]max_tokens → {n}[/]")
                except ValueError:
                    log.write(f"[{t('err')}]Invalid value. Use an integer between 64 and 32768[/]")
            else:
                log.write(f"[{t('txt2')}]max_tokens: [/][{t('acc')}]{S.max_tokens}[/]")

        def _cmd_memory(self, sub: str, arg: str, log: RichLog) -> None:
            t = T
            try:
                from server.gravity_memory import GravityMemory
                from pathlib import Path as _P
                mem = GravityMemory(_P(__file__).resolve().parent / "data" / "gravity_memory.db")
            except Exception as e:
                log.write(f"[{t('err')}]Memory unavailable: {e}[/]")
                return
            if not sub or sub == "list":
                n = int(arg) if arg.isdigit() else 20
                try:
                    entries = mem.list_recent(n)
                except Exception:
                    entries = []
                log.write(f"\n[bold {t('acc')}]Memory ({len(entries)} entries)[/]")
                for e in entries:
                    key   = e.get("key", "?")
                    val   = str(e.get("value", ""))[:80]
                    score = e.get("gravity", 0)
                    log.write(f"  [{t('acc')}]{key:<28}[/] [{t('txt')}]{val}[/] [{t('txt2')}]g={score:.2f}[/]")
                log.write("")
            elif sub == "search":
                if not arg:
                    log.write(f"[{t('txt2')}]Usage: /memory search <query>[/]"); return
                try:
                    results = mem.search(arg, top_k=10)
                except Exception:
                    results = []
                log.write(f"\n[bold {t('acc')}]Memory search: {arg}[/]")
                for e in results:
                    log.write(f"  [{t('acc')}]{e.get('key','?'):<28}[/] [{t('txt')}]{str(e.get('value',''))[:80]}[/]")
                log.write("")
            elif sub == "clear":
                # Clear only the episodic context injected into this session, not the DB
                S.history = [m for m in S.history if m.get("role") == "system"]
                log.write(f"[{t('ok')}]Episodic context cleared from this session.[/]")
            else:
                log.write(f"[{t('txt2')}]Usage: /memory [list [n] | search <q> | clear][/]")

        def _cmd_config(self, sub: str, arg: str, log: RichLog) -> None:
            t = T
            cfg_path = _WS / "config.toml"
            if not cfg_path.exists():
                log.write(f"[{t('err')}]config.toml not found at {cfg_path}[/]"); return
            if not sub or sub == "show":
                log.write(f"\n[bold {t('acc')}]config.toml[/]")
                for line in cfg_path.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if stripped.startswith("["):
                        log.write(f"[bold {t('acc')}]{line}[/]")
                    elif stripped.startswith("#") or not stripped:
                        log.write(f"[{t('txt2')}]{line}[/]")
                    else:
                        k, _, v = line.partition("=")
                        log.write(f"  [{t('txt2')}]{k.strip():<22}[/]= [{t('txt')}]{v.strip()}[/]")
                log.write("")
            elif sub == "get":
                if not arg:
                    log.write(f"[{t('txt2')}]Usage: /config get <section.key>  e.g. inference.model[/]"); return
                parts = arg.split(".", 1)
                section = parts[0] if len(parts) == 2 else ""
                key     = parts[1] if len(parts) == 2 else parts[0]
                try:
                    import tomllib as _tl
                    with open(cfg_path, "rb") as f:
                        data = _tl.load(f)
                    val = data.get(section, {}).get(key) if section else data.get(key)
                    if val is None:
                        log.write(f"[{t('warn')}]Key not found: {arg}[/]")
                    else:
                        log.write(f"[{t('acc')}]{arg}[/] = [{t('txt')}]{val}[/]")
                except Exception as e:
                    log.write(f"[{t('err')}]Read error: {e}[/]")
            elif sub == "set":
                parts = arg.split(maxsplit=1)
                if len(parts) < 2:
                    log.write(f"[{t('txt2')}]Usage: /config set <section.key> <value>[/]"); return
                dotkey, value = parts
                sec_parts = dotkey.split(".", 1)
                section   = sec_parts[0] if len(sec_parts) == 2 else ""
                key       = sec_parts[1] if len(sec_parts) == 2 else sec_parts[0]
                _persist_inference(key, value) if section == "inference" else _persist_provider_key(section, key, value)
                log.write(f"[{t('ok')}]{dotkey} = {value}[/]  [{t('txt2')}](saved)[/]")
            else:
                log.write(f"[{t('txt2')}]Usage: /config [show | get <key> | set <key> <value>][/]")

        def _cmd_kernel(self, log: RichLog) -> None:
            t = T
            try:
                from server.kernel import get_kernel
                k = get_kernel()
                st = k.stats()
                log.write(f"\n[bold {t('acc')}]Kernel[/]")
                log.write(f"  [{t('txt2')}]State       [/][{t('ok') if st['state']=='idle' else t('warn')}]{st['state']}[/]")
                log.write(f"  [{t('txt2')}]Provider    [/][{t('acc')}]{k._provider}[/]  [{t('txt2')}]model:[/] [{t('acc')}]{k._model}[/]")
                log.write(f"  [{t('txt2')}]Queue depth [/][{t('txt')}]{st['queue_depth']}[/]")
                log.write(f"  [{t('txt2')}]Tokens in   [/][{t('txt')}]{st['tokens_in']}[/]  [{t('txt2')}]out:[/] [{t('txt')}]{st['tokens_out']}[/]")
                log.write(f"  [{t('txt2')}]Cost        [/][{t('warn')}]${st['cost']:.6f}[/]")
                log.write(f"  [{t('txt2')}]Proc. hits  [/][{t('txt')}]{st['procedural_hits']}[/]")
                try:
                    from server.offline_cache import get_offline_cache
                    oc = get_offline_cache()
                    ost = oc.status()
                    log.write(f"  [{t('txt2')}]Backend     [/][{t('ok') if ost['backend_online'] else t('err')}]{'online' if ost['backend_online'] else 'offline'}[/]")
                    log.write(f"  [{t('txt2')}]Queue       [/][{t('txt')}]{ost['queue_size']} msg(s)[/]")
                except Exception:
                    pass
                log.write("")
            except Exception as e:
                log.write(f"[{t('err')}]Kernel unavailable: {e}[/]")

        def _cmd_plugin(self, sub: str, arg: str, log: RichLog) -> None:
            t = T
            try:
                from server.plugin_sdk import PluginStore
                store = PluginStore(_WS / "memory" / "plugins")
                plugins = store.list_all()
            except Exception as e:
                log.write(f"[{t('err')}]Plugin SDK unavailable: {e}[/]"); return
            if not sub or sub == "list":
                if not plugins:
                    log.write(f"[{t('txt2')}]No plugins installed. Place plugin directories in memory/plugins/[/]")
                    return
                log.write(f"\n[bold {t('acc')}]Plugins ({len(plugins)})[/]")
                for p in plugins:
                    en = f"[{t('ok')}]ON[/]" if p.enabled else f"[{t('warn')}]OFF[/]"
                    perms = ", ".join(p.permissions) or "none"
                    log.write(f"  {en}  [{t('acc')}]{p.id:<22}[/] [{t('txt')}]v{p.version}[/]  [{t('txt2')}]perms: {perms}[/]")
                log.write(f"  [{t('txt2')}]/plugin enable|disable <id>[/]\n")
            elif sub in ("enable", "disable") and arg:
                ok = store.set_enabled(arg, sub == "enable")
                if ok:
                    log.write(f"[{t('ok') if sub=='enable' else t('warn')}]{arg}: {sub}d[/]")
                else:
                    log.write(f"[{t('err')}]Plugin not found: {arg}[/]")
            else:
                log.write(f"[{t('txt2')}]Usage: /plugin [list | enable <id> | disable <id>][/]")

        # ── Service ingestion commands ────────────────────────────────────────────

        def _cmd_ingest(self, url: str, log: RichLog) -> None:
            t = T
            if not url:
                log.write(f"[{t('txt2')}]Usage: /ingest <url>[/]")
                log.write(f"  [{t('txt2')}]e.g. /ingest https://petstore3.swagger.io/api/v3/openapi.json[/]")
                return
            log.write(f"[{t('txt2')}]Ingesting {url} …[/]")

            async def _do():
                try:
                    import sys as _sys, pathlib as _pl
                    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
                    from server.service_ingestor import get_ingestor
                    profile = await get_ingestor().ingest(url)
                    log.write(
                        f"[{t('ok')}]Learned: {profile.name}[/]  "
                        f"[{t('txt2')}]{len(profile.endpoints)} endpoints · id: {profile.id}[/]"
                    )
                    for ep in profile.endpoints[:6]:
                        log.write(f"  [{t('acc')}]{ep.method:<7}[/] [{t('txt')}]{ep.path}[/]  [{t('txt2')}]{ep.description[:60]}[/]")
                    if len(profile.endpoints) > 6:
                        log.write(f"  [{t('txt2')}]… and {len(profile.endpoints)-6} more — /services show {profile.id}[/]")
                    log.write(f"\n  [{t('txt2')}]Tool prefix: svc_{profile.id}__*  ·  /services auth {profile.id} api_key <key>[/]\n")
                except Exception as e:
                    log.write(f"[{t('err')}]Ingestion failed: {e}[/]")

            import asyncio as _aio
            loop = _aio.get_event_loop()
            if loop.is_running():
                loop.create_task(_do())
            else:
                loop.run_until_complete(_do())

        def _cmd_usage(self, log: RichLog) -> None:
            """Show rate limit state + cost summary from this session."""
            t = T
            try:
                import sys as _sys, pathlib as _pl
                _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

                # Cost / token summary
                from server.usage_pricing import get_usage_tracker
                summary = get_usage_tracker().summary()
                log.write(f"\n[{t('acc')}]── Usage Summary ──────────────────────────────[/]")
                for line in summary.splitlines():
                    if line.strip():
                        log.write(f"[{t('txt')}]{line}[/]")
                    else:
                        log.write("")

                # Rate limit state per provider
                from server.rate_limit_tracker import get_tracker as _rl_tracker, format_rate_limit_display
                states = _rl_tracker().all_states()
                if states:
                    log.write(f"\n[{t('acc')}]── Rate Limits ────────────────────────────────[/]")
                    for _prov, state in states.items():
                        display = format_rate_limit_display(state)
                        for line in display.splitlines():
                            log.write(f"[{t('txt2')}]{line}[/]")
                else:
                    log.write(f"[{t('txt2')}]No rate limit data yet (make an API request first)[/]")
            except Exception as exc:
                log.write(f"[{t('err')}]/usage error: {exc}[/]")

        def _cmd_compress(self, topic: str, log: RichLog) -> None:
            """Force context compression of the current conversation window."""
            t = T
            try:
                import sys as _sys, pathlib as _pl
                _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
                from server.context_compressor import get_compressor
                comp = get_compressor()
                if comp is None:
                    log.write(f"[{t('warn')}]Context compressor not initialised.[/]")
                    return
                n_before = comp.compression_count
                log.write(f"[{t('txt2')}]Compressing context window{' (focus: ' + topic + ')' if topic else ''} …[/]")
                # The actual compression happens inside _stream_response.
                # Here we just report the compressor's current state and
                # confirm the next request will trigger a forced compress.
                log.write(
                    f"[{t('ok')}]Ready.[/]  [{t('txt2')}]"
                    f"Compression #{n_before + 1} will fire on your next message. "
                    f"{'Focus: ' + topic + '  ' if topic else ''}"
                    f"Anti-thrash count: {comp._ineffective_count}[/]"
                )
                # Store the focus topic so the next _stream_response picks it up
                S._compress_topic = topic
                S._compress_forced = True
            except Exception as exc:
                log.write(f"[{t('err')}]/compress error: {exc}[/]")

        def _cmd_insights(self, arg: str, log: RichLog) -> None:
            """Display session analytics for the past N days (/insights [days])."""
            t = T
            try:
                import sys as _sys, pathlib as _pl
                _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
                from server.session_insights import get_insights_engine
                days = int(arg.strip()) if (arg or "").strip().isdigit() else 30
                log.write(f"[{t('txt2')}]Generating insights for last {days} days …[/]")
                engine = get_insights_engine()
                report = engine.generate(days=days)
                rendered = engine.format_terminal(report)
                for line in rendered.split("\n"):
                    log.write(f"[{t('txt')}]{line}[/]")
            except Exception as exc:
                log.write(f"[{t('err')}]/insights error: {exc}[/]")

        def _cmd_pool(self, sub: str, arg: str, log: RichLog) -> None:
            """Credential pool management (/pool list|add|reset [provider] [key])."""
            t = T
            try:
                import sys as _sys, pathlib as _pl
                _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
                from server.credential_pool import get_pool, rotate_credential

                sub = (sub or "").strip().lower()
                provider = S.provider if hasattr(S, "provider") else "openai"

                if sub == "list" or not sub:
                    log.write(f"\n[bold {t('acc')}]Credential Pools[/]")
                    # Show pool state for current provider
                    pool = get_pool(provider)
                    for line in pool.summary_lines():
                        log.write(f"[{t('txt2')}]{line}[/]")
                    log.write(f"\n[{t('txt2')}]/pool add <key> [label]   /pool reset   /pool rotate[/]\n")

                elif sub == "add":
                    parts2 = (arg or "").strip().split(None, 1)
                    api_key = parts2[0] if parts2 else ""
                    label   = parts2[1] if len(parts2) > 1 else ""
                    if not api_key:
                        log.write(f"[{t('warn')}]Usage: /pool add <api_key> [label][/]")
                        return
                    pool  = get_pool(provider)
                    entry = pool.add(api_key, label=label or f"key-{len(pool.entries())+1}")
                    log.write(f"[{t('ok')}]Added credential {entry.id[:6]} ({entry.label}) to pool[{provider}][/]")

                elif sub == "reset":
                    pool  = get_pool(provider)
                    count = pool.reset_all()
                    log.write(f"[{t('ok')}]Reset {count} exhausted credential(s) in pool[{provider}][/]")

                elif sub == "rotate":
                    pool = get_pool(provider)
                    if len(pool.entries()) < 2:
                        log.write(f"[{t('warn')}]Pool has fewer than 2 credentials — nothing to rotate to.[/]")
                        return
                    nxt = pool.rotate()
                    if nxt:
                        log.write(f"[{t('ok')}]Rotated to: {nxt.short_label()} [{nxt.id[:6]}][/]")
                    else:
                        log.write(f"[{t('warn')}]All credentials exhausted — no rotation available.[/]")

                else:
                    log.write(f"[{t('warn')}]Usage: /pool list|add|reset|rotate[/]")

            except Exception as exc:
                log.write(f"[{t('err')}]/pool error: {exc}[/]")

        def _cmd_title(self, title_arg: str, log: RichLog) -> None:
            """Show or set the current session title (/title [new title])."""
            t = T
            try:
                import sys as _sys, pathlib as _pl
                _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
                from server.title_generator import get_session_title, _store_title

                # Need access to episodic memory
                try:
                    from server.episodic_memory import get_episodic_memory
                    episodic = get_episodic_memory()
                except Exception:
                    log.write(f"[{t('warn')}]Episodic memory unavailable.[/]")
                    return

                title_arg = (title_arg or "").strip()
                if title_arg:
                    # Set a custom title
                    _store_title(episodic, title_arg)
                    log.write(f"[{t('ok')}]Session title set: [bold]{title_arg}[/][/]")
                else:
                    # Show current title
                    current = get_session_title(episodic)
                    if current:
                        log.write(f"[{t('acc')}]Session title:[/] [bold {t('txt')}]{current}[/]")
                    else:
                        log.write(
                            f"[{t('txt2')}]No title yet. "
                            f"Auto-generation fires after the first exchange, "
                            f"or set one with /title <text>[/]"
                        )
            except Exception as exc:
                log.write(f"[{t('err')}]/title error: {exc}[/]")

        def _cmd_adapt(self, sub: str, arg: str, log: RichLog) -> None:
            """Adaptive behavior profile (/adapt [show|reset [dimension]])."""
            t = T
            try:
                from server.adaptive import get_adaptive_engine
                engine = get_adaptive_engine()
                sub = (sub or "show").strip().lower()

                if sub in ("show", "status", ""):
                    report = engine.format_report()
                    for line in report.split("\n"):
                        log.write(f"[{t('txt')}]{line}[/]")

                elif sub == "reset":
                    dimension = arg.strip().lower() or None
                    msg = engine.reset(dimension)
                    log.write(f"[{t('ok')}]{msg}[/]")

                elif sub == "off":
                    import os; os.environ["ESSENCE_ADAPTIVE"] = "0"
                    log.write(f"[{t('warn')}]Adaptive calibration paused this session "
                              f"(set ESSENCE_ADAPTIVE=0 in env to persist).[/]")

                elif sub == "on":
                    import os; os.environ["ESSENCE_ADAPTIVE"] = "1"
                    log.write(f"[{t('ok')}]Adaptive calibration resumed.[/]")

                else:
                    log.write(f"[{t('txt2')}]Usage: /adapt [show|reset [dimension]|on|off][/]")

            except Exception as exc:
                log.write(f"[{t('err')}]/adapt error: {exc}[/]")

        def _cmd_goals(self, sub: str, arg: str, log: RichLog) -> None:
            """Pending goal management (/goals [list|save|done|clear])."""
            t = T
            try:
                from server.goal_tracker import get_goal_tracker
                tracker = get_goal_tracker()
                sub = (sub or "list").strip().lower()

                if sub in ("list", "show", ""):
                    lines = tracker.format_list()
                    for line in lines.split("\n"):
                        log.write(f"[{t('txt')}]{line}[/]")

                elif sub in ("save", "add"):
                    text = arg.strip()
                    if not text:
                        log.write(f"[{t('txt2')}]Usage: /goals save <description>[/]")
                        return
                    goal = tracker.add_explicit(text)
                    log.write(f"[{t('ok')}]Goal saved [{goal.id}]: {text[:80]}[/]")

                elif sub in ("done", "complete", "del", "remove"):
                    gid = arg.strip()
                    if not gid:
                        log.write(f"[{t('txt2')}]Usage: /goals done <goal-id>[/]")
                        return
                    ok = tracker.complete(gid)
                    log.write(
                        f"[{t('ok')}]Goal {gid} marked done.[/]" if ok
                        else f"[{t('warn')}]Goal {gid} not found.[/]"
                    )

                elif sub == "clear":
                    n = tracker.clear_all()
                    log.write(f"[{t('ok')}]Cleared {n} pending goal(s).[/]")

                else:
                    log.write(f"[{t('txt2')}]Usage: /goals [list|save <text>|done <id>|clear][/]")

            except Exception as exc:
                log.write(f"[{t('err')}]/goals error: {exc}[/]")

        def _cmd_image(self, source: str, log: RichLog) -> None:
            """Attach an image to the next message (/image <path|url>)."""
            t = T
            if not source:
                if S.pending_images:
                    log.write(f"[{t('acc')}]Queued images ({len(S.pending_images)}):[/]")
                    for img in S.pending_images:
                        label = img.get("path") or img.get("url") or "(base64)"
                        log.write(f"  [{t('txt2')}]{label}[/]")
                    log.write(f"  [{t('txt2')}]Send your next message to include them, or /image clear to discard.[/]")
                else:
                    log.write(f"[{t('txt2')}]Usage: /image <path | https://url>[/]\n  [{t('txt2')}]Attaches an image to your next message.[/]")
                return

            if source.strip().lower() == "clear":
                S.pending_images = []
                log.write(f"[{t('ok')}]Image queue cleared.[/]")
                return

            try:
                from server.multimodal import load_image_to_data_uri, provider_supports_vision
                src = source.strip()
                if src.startswith("http://") or src.startswith("https://"):
                    entry = {"url": src}
                    label = src
                else:
                    # Local path — validate it exists
                    p = Path(src).expanduser().resolve()
                    if not p.exists():
                        log.write(f"[{t('err')}]File not found: {p}[/]")
                        return
                    # Pre-load to catch encoding errors early
                    data_uri, mt = load_image_to_data_uri(str(p))
                    b64 = data_uri.split(";base64,", 1)[1] if ";base64," in data_uri else ""
                    entry = {"base64": b64, "media_type": mt, "path": str(p)}
                    label = f"{p.name} ({mt})"

                S.pending_images.append(entry)
                log.write(
                    f"[{t('ok')}]Image queued:[/] {label}\n"
                    f"  [{t('txt2')}]{len(S.pending_images)} image(s) will be sent with your next message.[/]"
                )
            except Exception as exc:
                log.write(f"[{t('err')}]/image error: {exc}[/]")

        def _cmd_json(self, arg: str, log: RichLog) -> None:
            """Toggle JSON output mode (/json [on|off])."""
            t = T
            arg = (arg or "").strip().lower()
            if arg in ("on", "1", "true"):
                S.json_mode = True
            elif arg in ("off", "0", "false"):
                S.json_mode = False
            else:
                S.json_mode = not S.json_mode

            state = "ON" if S.json_mode else "OFF"
            colour = t("ok") if S.json_mode else t("txt2")
            log.write(
                f"[{colour}]JSON mode {state}[/]"
                + (" — model will respond with raw JSON only." if S.json_mode
                   else " — model responds in natural language.")
            )

        def _cmd_think(self, arg: str, log: RichLog) -> None:
            """Toggle extended reasoning/thinking mode (/think [on|off|<budget>])."""
            t = T
            arg = (arg or "").strip().lower()
            if arg in ("off", "0", "false", "no"):
                S.reasoning_mode = ""
            elif arg in ("on", "yes", "true", ""):
                # Default: on with 8000 token budget (Anthropic) or "high" effort (OpenAI)
                S.reasoning_mode = "8000"
            elif arg in ("high", "medium", "low"):
                S.reasoning_mode = arg   # OpenAI o-series effort level
            elif arg.isdigit():
                S.reasoning_mode = arg   # Anthropic budget_tokens
            else:
                # Toggle
                S.reasoning_mode = "" if S.reasoning_mode else "8000"

            if S.reasoning_mode:
                if S.reasoning_mode.isdigit():
                    log.write(
                        f"[{t('ok')}]Extended thinking ON[/] — budget: {S.reasoning_mode} tokens "
                        f"(Anthropic) or 'high' effort (OpenAI o-series).\n"
                        f"  [{t('txt2')}]Responses will be slower but significantly more thorough.[/]"
                    )
                else:
                    log.write(
                        f"[{t('ok')}]Extended reasoning ON[/] — effort: {S.reasoning_mode} "
                        f"(OpenAI o-series)."
                    )
            else:
                log.write(f"[{t('txt2')}]Extended thinking OFF.[/]")

        def _cmd_mcp(self, sub: str, arg: str, log: RichLog) -> None:
            """Manage MCP (Model Context Protocol) servers.

            /mcp list              — show all registered servers
            /mcp enable  <id>      — enable a server
            /mcp disable <id>      — disable a server
            /mcp connect <url>     — add + connect a new stdio/HTTP server
            /mcp refresh           — re-register all enabled servers' tools
            """
            t = T
            sub = (sub or "list").strip().lower()
            try:
                import sys as _sys, pathlib as _pl
                _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
                from server.mcpstore import MCPStore
                _store_path = _pl.Path(__file__).resolve().parent / "memory" / "ledger" / "mcp_servers.json"
                store = MCPStore(_store_path)
            except Exception as e:
                log.write(f"[{t('err')}]MCPStore unavailable: {e}[/]"); return

            if sub == "list":
                servers = store.list_all()
                log.write(f"\n[bold {t('acc')}]MCP Servers ({len(servers)})[/]")
                if not servers:
                    log.write(
                        f"  [{t('txt2')}]No MCP servers registered yet.\n"
                        f"  Use  /mcp connect <url>  to add one.[/]"
                    )
                    return
                for s in servers:
                    status = f"[{t('ok')}]enabled[/]" if s.get("enabled") else f"[{t('warn')}]disabled[/]"
                    tools_n = len(s.get("tools", []))
                    log.write(
                        f"  [{t('acc')}]{s['id']:<22}[/] {status}  "
                        f"[{t('txt2')}]{tools_n} tool(s)  {s.get('url') or s.get('command','?')}[/]"
                    )
                log.write(f"\n  [{t('txt2')}]/mcp enable <id>   /mcp disable <id>   /mcp refresh[/]\n")

            elif sub == "enable":
                if not arg:
                    log.write(f"[{t('warn')}]Usage: /mcp enable <server-id>[/]"); return
                try:
                    store.patch(arg, {"enabled": True})
                    log.write(f"[{t('ok')}]MCP server '{arg}' enabled.  Use /mcp refresh to load its tools.[/]")
                except Exception as e:
                    log.write(f"[{t('err')}]Enable failed: {e}[/]")

            elif sub == "disable":
                if not arg:
                    log.write(f"[{t('warn')}]Usage: /mcp disable <server-id>[/]"); return
                try:
                    store.patch(arg, {"enabled": False})
                    log.write(f"[{t('ok')}]MCP server '{arg}' disabled.[/]")
                except Exception as e:
                    log.write(f"[{t('err')}]Disable failed: {e}[/]")

            elif sub == "connect":
                if not arg:
                    log.write(f"[{t('warn')}]Usage: /mcp connect <url-or-command>[/]"); return
                try:
                    # Detect transport from arg
                    if arg.startswith("http://") or arg.startswith("https://"):
                        entry = store.register({"url": arg, "transport": "http", "enabled": True})
                    else:
                        # treat as command string
                        entry = store.register({"command": arg.split(), "transport": "stdio", "enabled": True})
                    log.write(
                        f"[{t('ok')}]MCP server registered (id={entry['id']}).[/]\n"
                        f"  [{t('txt2')}]Use /mcp refresh to load its tools into the current session.[/]"
                    )
                except Exception as e:
                    log.write(f"[{t('err')}]Connect failed: {e}[/]")

            elif sub == "refresh":
                async def _do_refresh() -> None:
                    try:
                        from server.tools_engine import register_mcp_tools
                        _store_path2 = _pl.Path(__file__).resolve().parent / "memory" / "ledger" / "mcp_servers.json"
                        n = await register_mcp_tools(_store_path2)
                        log.write(f"[{t('ok')}]MCP refresh: {n} tool(s) registered.[/]")
                    except Exception as e:
                        log.write(f"[{t('err')}]MCP refresh failed: {e}[/]")
                import asyncio as _aio
                _aio.get_event_loop().create_task(_do_refresh())

            else:
                log.write(
                    f"[{t('warn')}]Unknown /mcp subcommand '{sub}'.[/]\n"
                    f"  [{t('txt2')}]Subcommands: list  enable  disable  connect  refresh[/]"
                )

        def _cmd_services(self, sub: str, arg: str, log: RichLog) -> None:
            t = T
            try:
                import sys as _sys, pathlib as _pl
                _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
                from server.service_registry import get_service_registry
                reg = get_service_registry()
            except Exception as e:
                log.write(f"[{t('err')}]Service registry unavailable: {e}[/]"); return

            if not sub or sub == "list":
                svcs = reg.list_all()
                log.write(f"\n[bold {t('acc')}]Learned Services ({len(svcs)})[/]")
                if not svcs:
                    log.write(f"  [{t('txt2')}]None yet. Use /ingest <url> or drop an API URL in chat.[/]")
                    return
                for s in svcs:
                    log.write(
                        f"  [{t('acc')}]{s['id']:<28}[/] "
                        f"[{t('txt')}]{s['name']:<24}[/] "
                        f"[{t('txt2')}]{s['endpoints']:>4} endpoints  {s['base_url'][:38]}[/]"
                    )
                log.write(f"\n  [{t('txt2')}]/services show <id>   /services forget <id>   /services auth <id> <key> <val>[/]\n")

            elif sub == "show":
                sid = arg.strip().split()[0] if arg.strip() else ""
                if not sid:
                    log.write(f"[{t('txt2')}]Usage: /services show <id>[/]"); return
                profile = reg.get(sid)
                if not profile:
                    log.write(f"[{t('err')}]Service not found: {sid}[/]"); return
                log.write(f"\n[bold {t('acc')}]{profile.name}[/]  [{t('txt2')}]{profile.id}[/]")
                log.write(f"  [{t('txt2')}]Base URL:    [/][{t('txt')}]{profile.base_url}[/]")
                log.write(f"  [{t('txt2')}]Auth:        [/][{t('txt')}]{profile.auth.type}[/]")
                log.write(f"  [{t('txt2')}]Description: [/][{t('txt')}]{profile.description[:100]}[/]")
                log.write(f"  [{t('txt2')}]Endpoints ({len(profile.endpoints)}):[/]")
                for ep in profile.endpoints:
                    req = [p.name for p in ep.params if p.required]
                    req_str = f"  [required: {', '.join(req)}]" if req else ""
                    log.write(
                        f"    [{t('acc')}]{ep.method:<7}[/] [{t('txt')}]{ep.path:<40}[/] "
                        f"[{t('txt2')}]{ep.description[:50]}{req_str}[/]"
                    )
                log.write("")

            elif sub == "forget":
                sid = arg.strip().split()[0] if arg.strip() else ""
                if not sid:
                    log.write(f"[{t('txt2')}]Usage: /services forget <id>[/]"); return
                ok = reg.delete(sid)
                log.write(
                    f"[{t('ok')}]Removed: {sid}[/]" if ok
                    else f"[{t('err')}]Not found: {sid}[/]"
                )

            elif sub == "auth":
                # /services auth <id> <key> <value>
                parts = arg.strip().split(maxsplit=2)
                if len(parts) < 3:
                    log.write(f"[{t('txt2')}]Usage: /services auth <id> <key> <value>[/]")
                    log.write(f"  [{t('txt2')}]e.g. /services auth my-api api_key sk-...[/]"); return
                sid, key, value = parts
                ok = reg.set_cred(sid, key, value)
                if ok:
                    masked = value[:4] + "..." + value[-4:] if len(value) > 12 else "***"
                    log.write(f"[{t('ok')}]{sid}.{key} = {masked}[/]  [{t('txt2')}](saved)[/]")
                else:
                    log.write(f"[{t('err')}]Service not found: {sid}[/]")

            elif sub == "search":
                if not arg.strip():
                    log.write(f"[{t('txt2')}]Usage: /services search <query>[/]"); return
                results = reg.search(arg.strip())
                log.write(f"\n[bold {t('acc')}]Service search: {arg.strip()}[/]")
                if not results:
                    log.write(f"  [{t('txt2')}]No matches.[/]")
                for r in results:
                    log.write(f"  [{t('acc')}]{r['id']:<28}[/] [{t('txt')}]{r['name']}[/]  [{t('txt2')}]{r['base_url']}[/]")
                log.write("")

            else:
                log.write(f"[{t('txt2')}]Usage: /services [list | show <id> | forget <id> | auth <id> <key> <val> | search <q>][/]")

        def _cmd_routing(self, sub: str, log: RichLog) -> None:
            """Show / toggle dynamic routing."""
            t = T
            try:
                from server.kernel import get_kernel as _gk
                k = _gk()
            except Exception as e:
                log.write(f"[{t('err')}]Kernel unavailable: {e}[/]"); return

            if sub == "off":
                k._dynamic_routing = False
                log.write(f"[{t('warn')}]Dynamic routing DISABLED[/]  [{t('txt2')}]all requests use {k._provider}/{k._model}[/]")
                return
            if sub == "on":
                k._dynamic_routing = True
                log.write(f"[{t('ok')}]Dynamic routing ENABLED[/]  [{t('txt2')}]requests routed per task type[/]")
                return

            # Status display
            enabled = getattr(k, "_dynamic_routing", True)
            log.write(f"\n[bold {t('acc')}]Dynamic Routing[/]  [{'bold green' if enabled else t('err')}]{'ON' if enabled else 'OFF'}[/]")
            log.write(f"  [{t('txt2')}]Configured provider:  [/][{t('acc')}]{k._provider}[/]  [{t('txt2')}]model:[/] [{t('acc')}]{k._model}[/]")

            try:
                dr   = k._dynamic_router
                last = dr.last_route
                if last:
                    task    = last.get("task", "?")
                    chosen  = last.get("top_choice", ("?", "?"))
                    n_chain = last.get("chain_len", 0)
                    sigs    = last.get("signals", {})
                    log.write(f"\n  [{t('acc2')}]Last request[/]")
                    log.write(f"  [{t('txt2')}]Task type:  [/][{t('acc')}]{task}[/]")
                    log.write(f"  [{t('txt2')}]Routed to:  [/][{t('acc')}]{chosen[0]}/{chosen[1]}[/]")
                    log.write(f"  [{t('txt2')}]Fallbacks:  [/]{n_chain - 1}")
                    if sigs:
                        ts = sigs.get("task_scores", {})
                        if ts:
                            top_scores = sorted(ts.items(), key=lambda x: -x[1])[:3]
                            log.write(f"  [{t('txt2')}]Signals:    [/]" +
                                      "  ".join(f"{k}={v}" for k, v in top_scores))
                else:
                    log.write(f"  [{t('txt2')}]No routing decision yet (send a message first).[/]")

                # Show available providers
                log.write(f"\n  [{t('acc2')}]Available providers[/]")
                from server.dynamic_router import _PROVIDER_BEST_MODEL
                for pid in _PROVIDER_BEST_MODEL:
                    avail = dr._is_available(pid)
                    h     = dr._health.get(pid, {})
                    err   = h.get("error_rate", 0.0)
                    lim   = h.get("is_limited", False)
                    status = (f"[{t('err')}]rate-limited[/]" if lim else
                              f"[{t('err')}]error rate {err:.0%}[/]" if err > 0.1 else
                              f"[bold green]ok[/]" if avail else
                              f"[{t('txt2')}]no key[/]")
                    log.write(f"  [{t('acc')}]{pid:<14}[/] {status}")
            except Exception as e:
                log.write(f"[{t('err')}]Router error: {e}[/]")

            log.write(f"\n  [{t('txt2')}]/routing on | off   — toggle dynamic routing[/]\n")

        def _cmd_voice(self, sub: str, arg: str, log: RichLog) -> None:
            """Voice I/O: status | say <text> | listen"""
            t = T
            try:
                import sys as _sys, pathlib as _pl
                _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
                from server.voice_io import get_voice_io
                vio = get_voice_io()
            except Exception as e:
                log.write(f"[{t('err')}]Voice I/O unavailable: {e}[/]")
                log.write(f"  [{t('txt2')}]Install: pip install faster-whisper kokoro-onnx sounddevice[/]")
                return

            if not sub or sub == "status":
                stt = vio.is_stt_available()
                tts = vio.is_tts_available()
                mic = vio.is_mic_available()
                log.write(f"\n[bold {t('acc')}]Voice I/O Status[/]")
                log.write(f"  [{t('txt2')}]STT (listen):  [/][{'bold green' if stt else t('err')}]{'✓ available' if stt else '✗ unavailable'}[/]")
                log.write(f"  [{t('txt2')}]TTS (speak):   [/][{'bold green' if tts else t('err')}]{'✓ available' if tts else '✗ unavailable'}[/]")
                log.write(f"  [{t('txt2')}]Microphone:    [/][{'bold green' if mic else t('err')}]{'✓ available' if mic else '✗ unavailable'}[/]")
                if not stt or not tts:
                    log.write(f"  [{t('txt2')}]Install:  pip install faster-whisper kokoro-onnx sounddevice[/]")
                log.write("")

            elif sub == "say":
                if not arg.strip():
                    log.write(f"[{t('txt2')}]Usage: /voice say <text>[/]")
                    return
                if not vio.is_tts_available():
                    log.write(f"[{t('err')}]TTS not available. Install: pip install kokoro-onnx sounddevice[/]")
                    return
                import asyncio
                log.write(f"[{t('txt2')}]Speaking…[/]")
                async def _speak():
                    await vio.speak(arg.strip())
                asyncio.get_event_loop().create_task(_speak())
                log.write(f"[{t('ok')}]▶ {arg.strip()[:80]}[/]")

            elif sub == "listen":
                if not vio.is_stt_available():
                    log.write(f"[{t('err')}]STT not available. Install: pip install faster-whisper sounddevice[/]")
                    return
                if not vio.is_mic_available():
                    log.write(f"[{t('err')}]No microphone detected.[/]")
                    return
                import asyncio
                log.write(f"[{t('txt2')}]Listening… (speak now)[/]")
                async def _listen():
                    text = await vio.listen()
                    if text:
                        log.write(f"[{t('ok')}]Heard:[/] {text}")
                        # Inject as a new user message
                        input_box = self.query_one("#msg-input")
                        input_box.value = text
                        input_box.focus()
                    else:
                        log.write(f"[{t('warn')}]Nothing transcribed.[/]")
                asyncio.get_event_loop().create_task(_listen())

            else:
                log.write(f"[{t('txt2')}]Usage: /voice [status | say <text> | listen][/]")

        # ── Binding actions for new keyboard shortcuts ────────────────────────────
        def action_ping_backend(self) -> None:
            log = self.query_one("#chat-log", RichLog)
            self._cmd_ping(log)

        def action_retry_last(self) -> None:
            log = self.query_one("#chat-log", RichLog)
            self._cmd_retry(log)

        # ── Chat send & stream (kernel-routed) ──────────────────────────────────
        def _send(self, msg: str) -> None:
            """Publish user.request to the kernel bus — never call Ollama directly."""
            log = self.query_one("#chat-log", RichLog)
            t = T
            ts = datetime.now().strftime("%H:%M")
            log.write(f"\n[bold {t('acc2')}]You[/]  [{t('txt2')}]{ts}[/]")
            log.write(f"  {msg}")
            log.write("")
            self._stream_chat(msg)

        @work(exclusive=True, thread=False)
        async def _stream_chat(self, msg: str) -> None:
            """
            Route the message through the kernel via the event bus.
            Subscribes to user.response for this task_id and renders tokens.
            Falls back to direct Ollama call if kernel is not initialised.
            """
            log_w = self.query_one("#chat-log", RichLog)
            hdr   = self.query_one("#app-header", AppHeader)
            sl    = self.query_one("#status-line", StatusLine)
            typing = self.query_one("#typing-row", Static)
            t = T

            S.streaming    = True
            S.stream_start = time.time()
            hdr.streaming  = True
            sl.streaming   = True
            typing.update(f"  [{t('txt2')}]🤖 Essence  ◉ ◉ ◉[/]")
            typing.add_class("visible")

            buf          = ""
            display_buf  = ""
            last_flush   = time.time()
            wrote_header = False
            task_id      = uuid.uuid4().hex[:10]

            # ── Attempt kernel-routed path ────────────────────────────────────────
            kernel_available = False
            try:
                from server.kernel import get_kernel
                from server.event_bus import Envelope, get_event_bus
                _kernel = get_kernel()
                _bus    = get_event_bus()
                kernel_available = True
            except Exception:
                pass

            if kernel_available:
                # Queue for tokens arriving on user.response
                token_queue: asyncio.Queue = asyncio.Queue()
                done_event  = asyncio.Event()

                async def _on_response(env: Any) -> None:
                    if env.data.get("task_id") == task_id or env.task_id == task_id:
                        await token_queue.put(env.data)

                _bus.subscribe("user.response", _on_response)

                # Publish user.request — kernel takes it from here
                _req_data: dict = {
                    "text": msg, "session_id": S.session_id,
                    "temperature": S.temperature, "max_tokens": S.max_tokens,
                }
                # Attach any pending images then clear the queue
                if getattr(S, "pending_images", []):
                    _req_data["images"] = list(S.pending_images)
                    S.pending_images = []
                # Pass JSON mode flag
                if getattr(S, "json_mode", False):
                    _req_data["json_mode"] = True
                # Pass reasoning mode
                if getattr(S, "reasoning_mode", ""):
                    _req_data["reasoning_mode"] = S.reasoning_mode
                # Pass forced-compress state (set by /compress command) then clear it
                if getattr(S, "_compress_forced", False):
                    _req_data["compress_forced"] = True
                    _req_data["compress_topic"] = getattr(S, "_compress_topic", "")
                    S._compress_forced = False
                    S._compress_topic = ""
                env = Envelope(
                    topic="user.request",
                    source_component="tui",
                    task_id=task_id,
                    data=_req_data,
                )
                await _bus.publish(env)

                # Drain token_queue until done=True or timeout
                deadline = time.time() + 180.0
                try:
                    while time.time() < deadline:
                        try:
                            data = await asyncio.wait_for(token_queue.get(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue

                        token = data.get("token", "")
                        is_done = data.get("done", False)

                        if token:
                            if not wrote_header:
                                typing.remove_class("visible")
                                ts2 = datetime.now().strftime("%H:%M")
                                state = getattr(_kernel, "state", "")
                                state_tag = (f"[{t('warn')}]{state}[/] " if state not in ("", "idle") else "")
                                log_w.write(f"[bold {t('ok')}]🤖 Essence[/]  {state_tag}[{t('txt2')}]{ts2}[/]")
                                wrote_header = True
                            buf         += token
                            display_buf += token

                        now = time.time()
                        if display_buf and ("\n" in display_buf or now - last_flush > 0.08):
                            segs = display_buf.split("\n")
                            for seg in segs[:-1]:
                                log_w.write(f"  {seg}" if seg.strip() else "")
                            display_buf = segs[-1]
                            last_flush  = now

                        if is_done:
                            S.tok_in  += data.get("tokens_in",  0)
                            S.tok_out += data.get("tokens_out", 0)
                            S.cost    += _est_cost(S.tok_in, S.tok_out, S.model)
                            sl.stream_secs = time.time() - S.stream_start
                            break

                finally:
                    _bus.unsubscribe("user.response", _on_response)

            else:
                # ── Direct Ollama fallback (kernel not running) ───────────────────
                try:
                    import httpx  # type: ignore
                except ImportError:
                    typing.remove_class("visible")
                    log_w.write(f"[{t('err')}]httpx not installed. Run: python essence.py install[/]")
                    S.streaming = False
                    return

                S.history.append({"role": "user", "content": msg})
                try:
                    async with httpx.AsyncClient(timeout=120) as c:
                        async with c.stream(
                            "POST", f"{_get_ollama()}/api/chat",
                            json={"model": S.model, "messages": S.history, "stream": True},
                        ) as r:
                            r.raise_for_status()
                            async for line in r.aiter_lines():
                                if not line:
                                    continue
                                try:
                                    d = json.loads(line)
                                except Exception:
                                    continue
                                token = d.get("message", {}).get("content", "")
                                if token:
                                    if not wrote_header:
                                        typing.remove_class("visible")
                                        ts2 = datetime.now().strftime("%H:%M")
                                        log_w.write(f"[bold {t('ok')}]🤖 Essence[/]  [{t('txt2')}]{ts2}[/]")
                                        wrote_header = True
                                    buf         += token
                                    display_buf += token
                                now = time.time()
                                if display_buf and ("\n" in display_buf or now - last_flush > 0.08):
                                    segs = display_buf.split("\n")
                                    for seg in segs[:-1]:
                                        log_w.write(f"  {seg}" if seg.strip() else "")
                                    display_buf = segs[-1]
                                    last_flush  = now
                                if d.get("done"):
                                    S.tok_in  += d.get("prompt_eval_count", 0)
                                    S.tok_out += d.get("eval_count", 0)
                                    S.cost    += _est_cost(S.tok_in, S.tok_out, S.model)
                                    break
                                sl.stream_secs = time.time() - S.stream_start
                except Exception as e:
                    typing.remove_class("visible")
                    log_w.write(f"\n  [{t('err')}]⚠  {e}[/]")
                    if S.history and S.history[-1].get("role") == "user":
                        S.history.pop()
                    S.streaming = False
                    hdr.streaming  = False
                    sl.streaming   = False
                    return

            # ── Finalise display ─────────────────────────────────────────────────
            if display_buf.strip():
                log_w.write(f"  {display_buf}")
            typing.remove_class("visible")
            log_w.write("")
            if buf:
                S.history.append({"role": "assistant", "content": buf})
                if S.verbose:
                    elapsed = time.time() - S.stream_start
                    log_w.write(
                        f"  [{t('txt2')}]↑{S.tok_in}t ↓{S.tok_out}t  "
                        f"{elapsed:.1f}s  ${S.cost:.6f}[/]"
                    )
            S.streaming    = False
            hdr.streaming  = False
            hdr.cost_str   = f"${S.cost:.4f}"
            sl.streaming   = False
            sl.tok_in      = S.tok_in
            sl.tok_out     = S.tok_out

        # ── Actions ──────────────────────────────────────────────────────────────
        def action_show_help(self) -> None:
            self.push_screen(HelpScreen())

        async def action_command_palette(self) -> None:
            result = await self.push_screen_wait(CommandPaletteScreen())
            if result:
                inp = self.query_one("#msg-input", HistoryInput)
                inp.value = result
                inp.focus()

        def action_toggle_sidebar(self) -> None:
            self._sidebar_on = not self._sidebar_on
            self.query_one("#sidebar", AgentSidebar).display = self._sidebar_on

        def action_toggle_split(self) -> None:
            self._split_on = not self._split_on
            sp = self.query_one("#second-pane", SecondPane)
            if self._split_on:
                sp.show_mode(self._split_mode)
            else:
                sp.remove_class("visible")

        def action_mode_cost(self) -> None:
            self._split_mode = "cost"
            self._split_on = True
            self.query_one("#second-pane", SecondPane).show_mode("cost")

        def action_mode_cron(self) -> None:
            self._split_mode = "cron"
            self._split_on = True
            self.query_one("#second-pane", SecondPane).show_mode("cron")

        def action_mode_logs(self) -> None:
            self._split_mode = "logs"
            self._split_on = True
            self.query_one("#second-pane", SecondPane).show_mode("logs")

        def action_mode_resource(self) -> None:
            self._split_mode = "resource"
            self._split_on = True
            self.query_one("#second-pane", SecondPane).show_mode("resource")

        @work(exclusive=False, thread=False)
        async def action_list_models(self) -> None:
            log = self.query_one("#chat-log", RichLog)
            t = T
            try:
                import httpx  # type: ignore

                async with httpx.AsyncClient(timeout=5) as c:
                    r = await c.get(f"{_get_ollama()}/api/tags")
                    models = r.json().get("models", [])
                log.write(f"\n[bold {t('acc')}]Available Models ({len(models)})[/]")
                for m in models:
                    name = m.get("name", "?")
                    size_gb = m.get("size", 0) / 1024**3
                    cur = f"[{t('ok')}]◄ current[/]" if name == S.model else ""
                    log.write(f"  [{t('acc')}]{name:<34}[/] [{t('txt2')}]{size_gb:.1f}GB[/]  {cur}")
                log.write(f"  [{t('txt2')}]Use /model <name> to switch[/]\n")
            except Exception as e:
                log.write(f"[{t('err')}]Cannot list models: {e}[/]")

        def action_toggle_tool_output(self) -> None:
            log = self.query_one("#chat-log", RichLog)
            log.write(f"[{T('txt2')}]Tool output: toggled (applies to next tool call)[/]")

        def action_export_chat(self) -> None:
            self._cmd_export("md", self.query_one("#chat-log", RichLog))

        def action_copy_last(self) -> None:
            last = next((m for m in reversed(S.history) if m.get("role") == "assistant"), None)
            if not last:
                return
            txt = last["content"]
            try:
                if sys.platform == "win32":
                    subprocess.run(["clip"], input=txt.encode("utf-16"), check=True, capture_output=True)
                elif sys.platform == "darwin":
                    subprocess.run(["pbcopy"], input=txt.encode(), check=True)
                else:
                    subprocess.run(["xclip", "-selection", "clipboard"], input=txt.encode(), check=True)
                self.query_one("#chat-log", RichLog).write(f"[{T('ok')}]Copied to clipboard.[/]")
            except Exception:
                self.query_one("#chat-log", RichLog).write(f"[{T('warn')}]Clipboard unavailable.[/]")

        def action_close_pane(self) -> None:
            if self._split_on:
                self._split_on = False
                sp = self.query_one("#second-pane", SecondPane)
                sp.remove_class("visible")

        # ── Cleanup ──────────────────────────────────────────────────────────────
        def on_unmount(self) -> None:
            # Auto-save session history so /restore works after restart
            self._autosave_session()
            try:
                hist = json.loads(_COST_FILE.read_text()) if _COST_FILE.exists() else []
                hist.append({
                    "session": S.session_id, "timestamp": time.time(),
                    "model": S.model, "tok_in": S.tok_in,
                    "tok_out": S.tok_out, "cost": S.cost,
                })
                _COST_FILE.write_text(json.dumps(hist[-200:], indent=2))
            except Exception:
                pass


# ── Fallback simple REPL ────────────────────────────────────────────────────────
def _simple_chat() -> None:
    try:
        import httpx
    except ImportError:
        print("Run: python essence.py install")
        sys.exit(1)

    model, base = S.model, _get_ollama()
    print(f"\nEssence Chat — {model} — /exit to quit\n")
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg or msg.lower() in ("/exit", "/quit"):
            break
        S.history.append({"role": "user", "content": msg})
        print("AI: ", end="", flush=True)
        buf = ""
        try:
            with httpx.Client(timeout=120) as c:
                with c.stream(
                    "POST",
                    f"{base}/api/chat",
                    json={"model": model, "messages": S.history, "stream": True},
                ) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                        except Exception:
                            continue
                        t = d.get("message", {}).get("content", "")
                        print(t, end="", flush=True)
                        buf += t
                        if d.get("done"):
                            break
        except Exception as e:
            print(f"\n  Error: {e}")
            S.history.pop()
            continue
        print()
        S.history.append({"role": "assistant", "content": buf})


# ── Entry point ─────────────────────────────────────────────────────────────────
def main() -> None:
    if _TEXTUAL:
        EssenceTUI().run()
    else:
        _simple_chat()


if __name__ == "__main__":
    main()
