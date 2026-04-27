# Essence — Local-First Personal AI Agent

> **v28.0.0** — Privacy-first. Runs entirely on your machine. No data leaves without your permission.

Essence is a self-contained AI agent system built around a cognitive kernel, multi-provider LLM routing, a rich terminal UI, and a layered memory architecture. Designed for personal and power-user workloads: autonomous task execution, skill composition, proactive automation, and persistent episodic memory — all under your control.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [CLI Reference — `essence.py`](#cli-reference--essencepy)
6. [TUI Reference — `tui_app.py`](#tui-reference--tui_apppy)
   - [Keyboard Shortcuts](#keyboard-shortcuts)
   - [Slash Commands](#slash-commands)
7. [Server Components](#server-components)
8. [Provider & Model System](#provider--model-system)
9. [Skill System](#skill-system)
10. [Plugin System](#plugin-system)
11. [Memory Architecture](#memory-architecture)
12. [Offline Resilience](#offline-resilience)
13. [Governance & Audit](#governance--audit)
14. [Automation & Cron](#automation--cron)
15. [Environment Variables](#environment-variables)
16. [Directory Structure](#directory-structure)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Interface Layer                                                 │
│  ┌──────────────────┐   ┌──────────────────────────────────┐    │
│  │  TUI  (Textual)  │   │  CLI  (essence.py)               │    │
│  │  tui_app.py      │   │  probe / chat / models / skills  │    │
│  └────────┬─────────┘   └──────────────────────────────────┘    │
└───────────┼─────────────────────────────────────────────────────┘
            │  user.request / user.response  (EventBus)
┌───────────▼─────────────────────────────────────────────────────┐
│  Cognitive Kernel  (server/kernel.py)                           │
│  3-pass planner → skill executor → reflector → streamer         │
│  CognitiveState: IDLE → PLANNING → EXECUTING → REFLECTING       │
└───┬──────────────┬───────────────┬──────────────────┬───────────┘
    │              │               │                  │
┌───▼────────┐ ┌───▼────────┐ ┌───▼────────┐ ┌──────▼──────────┐
│ModelRouter │ │SkillAgent  │ │ EventBus   │ │GovernanceEngine │
│multi-prov  │ │sandboxed   │ │HMAC-signed │ │policy / trust   │
│fallbacks   │ │subprocess  │ │SQLite log  │ │tool approval    │
└───┬────────┘ └────────────┘ └────────────┘ └─────────────────┘
    │
┌───▼────────────────────────────────────────────────────────────┐
│  Memory Layer                                                  │
│  EpisodicMemory · GravityMemory · OfflineCache · AuditLogger  │
└────────────────────────────────────────────────────────────────┘
```

**Data flow for every user message:**

1. TUI publishes a signed `user.request` envelope onto the EventBus
2. The Kernel dequeues it and runs 3-pass planning (procedural → capability → LLM)
3. If a skill is needed, SkillAgent executes it in a subprocess sandbox
4. Otherwise, ModelRouter selects the best available provider and streams tokens
5. Tokens arrive on `user.response` and are rendered in the TUI in real-time
6. The full response is written to EpisodicMemory and the offline session cache

---

## Requirements

| Dependency | Minimum | Notes |
|---|---|---|
| Python | 3.11+ | 3.12+ recommended |
| Ollama | 0.1.24+ | Local LLM runtime |
| RAM | 8 GB | 16 GB+ for 7–14B models |

**Python packages** (installed via `python essence.py install`):

```
httpx  psutil  python-json-logger  openai  tiktoken  textual
huggingface_hub  pandas  numpy  scikit-learn  scipy  matplotlib
joblib  pypdf  python-docx  beautifulsoup4  vaderSentiment
cryptography  keyring
```

---

## Installation

### 1. Install Ollama

Download from **https://ollama.com/download** and run the installer.

### 2. Pull a language model

```bash
ollama pull qwen3:4b        # default — fast, ~2.5 GB
ollama pull qwen3:8b        # stronger — ~5 GB
ollama pull gemma4:e2b      # vision-capable
```

### 3. Install Python dependencies

```bash
cd /path/to/essence
python essence.py install
```

### 4. Verify the installation

```bash
python essence.py probe
```

### 5. Launch the TUI

```bash
python essence.py tui
```

---

## Configuration

All settings live in **`config.toml`** at the workspace root. Most inference settings also apply immediately via TUI commands without restarting.

```toml
[core]
version   = "28.0.0"
workspace = "/path/to/workspace"       # auto-detected at startup

[inference]
backend   = "ollama"                   # active provider
model     = "qwen3:4b"                 # active model

[agent]
thinking        = false                # chain-of-thought reasoning
budget          = 1024                 # max thinking tokens
critic          = true                 # CriticGate validation on every response
max_steps       = 12                   # max plan steps per task
memory_window   = 10                   # turns kept before episodic distillation
allow_outside   = false                # sandbox: block file access outside workspace
autonomy_level  = 1                    # 0=confirm all tools  1=confirm destructive  2=auto

[rag]
enabled    = true
chunk_size = 512                       # characters per chunk
overlap    = 64
top_k      = 6                         # passages injected per turn

[memory]
backend = "auto"                       # auto | json | sqlite_vec | faiss | qdrant

[heartbeat]
enabled  = true
interval = "30m"                       # cron/automation tick interval

[server]
host = "0.0.0.0"
port = 7860

[ml]
plots_dir       = "…/plots"
models_dir      = "…/models"
experiments_dir = "…/experiments"
default_hpo     = false                # Optuna HPO on train_model

[voice]
enabled = false                        # pip install pyaudio faster-whisper kokoro-onnx

[mesh]
enabled        = false
role           = "auto"                # auto | coordinator | worker
peer_discovery = "mdns"

[search]
# BRAVE_API_KEY  — 2,000 free web search calls/month
# SEARXNG_URL    — self-hosted SearXNG instance

[channels]
# TELEGRAM_BOT_TOKEN
# DISCORD_WEBHOOK_URL
```

Provider API keys go in named sections:

```toml
[provider.groq]
api_key = "gsk_…"

[provider.openai]
api_key = "sk-…"

[provider.anthropic]
api_key = "sk-ant-…"
```

---

## CLI Reference — `essence.py`

Run any command with `python essence.py <command> [subcommand] [args]`.

---

### Core

| Command | Description |
|---|---|
| `python essence.py` | System probe — Ollama status, model availability, workspace info |
| `python essence.py probe` | Same as above (explicit) |
| `python essence.py tui` | Launch the full Textual TUI (recommended) |
| `python essence.py chat` | Minimal streaming REPL without the TUI |
| `python essence.py install` | Install / upgrade all Python dependencies |
| `python essence.py setup` | Interactive first-run setup wizard |
| `python essence.py version` | Print version and build info |
| `python essence.py health` | System health check — Python, Ollama, model, disk, RAM |

---

### Models

| Command | Description |
|---|---|
| `python essence.py pull <model>` | Pull an Ollama model (e.g. `qwen3:8b`) |
| `python essence.py models` | List Ollama models with size and modification date |
| `python essence.py models list` | Same — explicit subcommand |
| `python essence.py models pull <name>` | Pull a specific model |
| `python essence.py models rm <name>` | Remove an Ollama model |
| `python essence.py models select` | Interactive best-model picker based on available RAM |
| `python essence.py models hf-list` | List locally downloaded GGUF files |
| `python essence.py models hf-pull` | Auto-select and pull best GGUF for your RAM tier |
| `python essence.py models hf-pull <repo> <file>` | Pull a specific GGUF from HuggingFace |
| `python essence.py models hf-run [<file>]` | Chat with a local GGUF via llama-cpp-python |

---

### Skills

| Command | Description |
|---|---|
| `python essence.py skills` | List all skills (built-in + JSON store) with enabled status |
| `python essence.py skills list` | Same — explicit |
| `python essence.py skills add --name "My Skill" [--category tool] [--desc "…"] [--prompt "…"]` | Create a skill |
| `python essence.py skills enable <skill_id>` | Enable a skill |
| `python essence.py skills disable <skill_id>` | Disable a skill |
| `python essence.py skills remove <skill_id>` | Delete a skill |
| `python essence.py skills show <skill_id>` | Print full skill JSON |
| `python essence.py skills from-md <file.md>` | Import skill from Markdown spec |
| `python essence.py skills from-json <file.json>` | Bulk-import skills from JSON |

---

### Plugins

| Command | Description |
|---|---|
| `python essence.py plugin list` | List installed plugin files |
| `python essence.py plugin install <url\|path>` | Install plugin from URL or local `.py` file |
| `python essence.py plugin uninstall <name>` | Remove an installed plugin |
| `python essence.py plugin update <name>` | Re-install plugin from its original source |

---

### Automation

| Command | Description |
|---|---|
| `python essence.py automate list` | List all proactive automation patterns |
| `python essence.py automate add <name> [--trigger schedule\|event\|condition] [--schedule "…"] [--prompt "…"]` | Create a pattern |
| `python essence.py automate rm <id>` | Remove a pattern |
| `python essence.py automate activate <id>` | Enable a pattern |
| `python essence.py automate run <id>` | Fire a pattern immediately |

---

### Cron Jobs

| Command | Description |
|---|---|
| `python essence.py cron list` | List all cron jobs with schedule and status |
| `python essence.py cron add <name> <schedule> [command]` | Add a job (crontab syntax e.g. `"0 8 * * *"`) |
| `python essence.py cron enable <name\|id>` | Enable a job |
| `python essence.py cron disable <name\|id>` | Disable a job |
| `python essence.py cron rm <name\|id>` | Remove a job |

---

### Memory & Data

| Command | Description |
|---|---|
| `python essence.py memory list` | List recent memory entries with gravity scores |
| `python essence.py memory stats` | Memory database statistics |
| `python essence.py memory clear` | Clear all episodic memory entries |
| `python essence.py snapshot` | List saved session snapshots |
| `python essence.py snapshot list` | Same — explicit |
| `python essence.py snapshot save [id]` | Save current session as a named snapshot |
| `python essence.py snapshot rm <id>` | Delete a snapshot |
| `python essence.py cost` | Token and cost summary across all sessions |

---

### Integrations (MCP)

| Command | Description |
|---|---|
| `python essence.py mcp list` | List registered MCP servers |
| `python essence.py mcp add <name> <command>` | Register an MCP server |
| `python essence.py mcp remove <name>` | Remove an MCP server |
| `python essence.py mcp enable <name>` | Enable an MCP server |
| `python essence.py mcp disable <name>` | Disable an MCP server |

---

### System

| Command | Description |
|---|---|
| `python essence.py logs` | Show last 50 lines from the active log file |
| `python essence.py logs --tail <n>` | Show last N lines |
| `python essence.py config show` | Print full config.toml |
| `python essence.py config get <key>` | Read a value (e.g. `inference.model`) |
| `python essence.py clean` | Housekeeping: vacuum SQLite DBs, prune episodic, archive logs |
| `python essence.py clean --dry-run` | Preview what would be cleaned without deleting |

---

## TUI Reference — `tui_app.py`

Launch with `python essence.py tui`.

---

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `F1` | Open the help screen |
| `Ctrl+K` | Command palette — fuzzy-search all slash commands |
| `Ctrl+B` | Toggle sidebar |
| `Ctrl+S` | Toggle split pane |
| `Ctrl+D` | Cost & token dashboard |
| `Ctrl+R` | Cron job manager |
| `Ctrl+L` | List available Ollama models |
| `Ctrl+P` | Ping backend — check Ollama + active provider |
| `Ctrl+Z` | Retry — resend the last user message |
| `Ctrl+O` | Toggle tool output panel |
| `Ctrl+E` | Export current chat to Markdown |
| `Ctrl+Y` | Copy last AI response to clipboard |
| `Escape` | Close active pane / dismiss modal |

---

### Slash Commands

Type any slash command directly in the message input. All commands are also reachable via `Ctrl+K`.

#### Session

| Command | Description |
|---|---|
| `/help` or `/?` | Open the full keyboard and command reference |
| `/clear` | Auto-save then start a fresh session |
| `/status` | Session summary: provider, model, token count, cost, kernel state |
| `/history [n]` | Show last `n` conversation turns (default 10) |
| `/forget [n]` | Remove last `n` user+assistant pairs from history (default 1) |
| `/retry` | Resend the last user message — also `Ctrl+Z` |
| `/snapshot` | Save the current session as a named snapshot |
| `/archive` | List all saved snapshots |
| `/restore <id>` | Restore a snapshot into the active session (fuzzy match on id) |
| `/export md` | Export chat to Markdown → `chat_YYYYMMDD_HHMMSS.md` |
| `/export json` | Export chat to JSON |
| `/attach <file>` | Inject a file's contents as context into the next message |

---

#### Model & Provider

| Command | Description |
|---|---|
| `/model [name]` | Show active model, or switch to `<name>` (persists to config) |
| `/models` | List all available Ollama models with size |
| `/provider list` | All providers: configured status and cost per 1M tokens |
| `/provider use <id>` | Switch active provider and persist — `ollama groq openai anthropic mistral deepseek gemini` |
| `/provider models [id]` | Discover available models from `<id>`, or the active provider if omitted |
| `/provider set <id> api_key <value>` | Store an API key for a provider (env + config.toml) |
| `/provider status` | Per-provider usage: calls, tokens, avg latency, error count |

---

#### Inference Tuning

| Command | Description |
|---|---|
| `/temp [0.0–2.0]` | Show or set generation temperature (default `0.7`) |
| `/tokens [n]` | Show or set max tokens per response — range `64–32768` (default `2048`) |
| `/sys [prompt]` | Set an extra system prompt for this session; omit to clear |
| `/think [off]` | Toggle chain-of-thought reasoning mode |
| `/verbose [off]` | Toggle verbose tool output |

---

#### Memory & Skills

| Command | Description |
|---|---|
| `/memory list [n]` | Show last `n` memory entries with gravity scores (default 20) |
| `/memory search <query>` | Semantic search across long-term memory |
| `/memory clear` | Clear episodic context injected into this session |
| `/skills` or `/skills list` | List all skills with enabled/disabled status and source |
| `/skills enable <id>` | Enable a skill |
| `/skills disable <id>` | Disable a skill |
| `/plugin list` | Installed plugins with permissions |
| `/plugin enable <id>` | Enable a plugin |
| `/plugin disable <id>` | Disable a plugin |

---

#### System & Governance

| Command | Description |
|---|---|
| `/config show` | Print full `config.toml` with syntax highlighting |
| `/config get <section.key>` | Read one value, e.g. `/config get inference.model` |
| `/config set <section.key> <value>` | Write one value, e.g. `/config set agent.autonomy_level 2` |
| `/kernel` | Kernel cognitive state: queue depth, tokens in/out, cost, offline queue |
| `/ping` | Check Ollama and active cloud provider connectivity — also `Ctrl+P` |
| `/audit` | Audit trail: approvals, violations, model switches |
| `/cost` | Token and cost dashboard |
| `/cron` | Cron job manager |
| `/logs` | Recent log output |

---

#### Interface

| Command | Description |
|---|---|
| `/theme [name]` | Switch colour theme: `default-dark` `dracula` `nord` `solarized` `high-contrast` |
| `/room <id>` | Switch conversation room / context scope |
| `/webhook` | WebHook debugger panel |
| `/deliver` / `/fast` | Mode toggles |
| `/quit` or `/exit` | Exit the TUI |

---

## Server Components

All server modules live in `server/`. They communicate exclusively through the EventBus — no component calls another directly.

| Module | Role |
|---|---|
| `kernel.py` | Cognitive core — 3-pass planner, skill executor, response streamer, reflector |
| `event_bus.py` | HMAC-SHA256-signed pub/sub broker; JSONL-persisted event log with integrity verification |
| `model_router.py` | Multi-provider LLM router — tier selection, fallback chain, OpenAI normalisation, cost/latency routing |
| `offline_cache.py` | Offline resilience — per-session JSONL cache, message queue, async health probe, auto-flush on reconnect |
| `audit_logger.py` | Compliance audit trail — SQLite `audit_logs` table; JSON and CSV export |
| `governance.py` | Policy enforcement — autonomy levels, idempotency guard, trust ledger, HITL tool approval |
| `identity_engine.py` | Identity and persona — reads `identity.toml`, logs every field change to audit trail |
| `episodic_memory.py` | Short-term conversation memory — turn storage, LLM-driven distillation, semantic compaction |
| `gravity_memory.py` | Long-term memory — SQLite-backed with gravity scoring, semantic search |
| `skill_orchestrator.py` | Built-in skill registry — defines `BUILTIN_SKILLS` with category, description, system prompt |
| `skill_agent.py` | Skill execution harness — runs a skill given a plan step and context dict |
| `skillstore.py` | Persistent skill store — CRUD over JSON files in `memory/skills/` |
| `plugin_sdk.py` | External plugin system — manifest validation, sandboxed subprocess runner, concurrent hook dispatch |
| `action_engine.py` | Proactive automation — pattern registry, trigger evaluation (schedule / event / condition) |
| `tools_engine.py` | Tool registry and executor — maps tool names to callable Python functions |
| `web_tools.py` | Web tools — Brave search, SearXNG, page fetch |
| `credstore.py` | Credential store — keyring-backed secret management |
| `mcpstore.py` | MCP server registry — CRUD for Model Context Protocol endpoints |
| `proactive_agent.py` | Background agent — evaluates patterns on every heartbeat tick |
| `materialized_views.py` | Pre-computed memory summaries for fast context injection |
| `event_log.py` | Structured event log — in-memory ring buffer + SQLite persistence |

---

## Provider & Model System

Essence supports **7 provider backends** through a unified OpenAI-compatible request normaliser. All providers expose the same streaming interface to the kernel; format differences (Anthropic headers, Ollama native JSON) are handled transparently.

### Supported Providers

| Provider ID | Endpoint | Key Source |
|---|---|---|
| `ollama` | `http://localhost:11434/v1` | No key required |
| `groq` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `anthropic` | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` |
| `mistral` | `https://api.mistral.ai/v1` | `MISTRAL_API_KEY` |
| `deepseek` | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` |
| `gemini` | `https://generativelanguage.googleapis.com/v1beta/openai` | `GEMINI_API_KEY` |

### Switching Providers

**TUI (live, persists):**
```
/provider use groq
/provider use openai
/provider use ollama
```

**Config file (on next start):**
```toml
[inference]
backend = "groq"
model   = "llama-3.3-70b-versatile"
```

**Environment variable (one-off):**
```bash
ESSENCE_BACKEND=groq ESSENCE_MODEL=llama-3.3-70b-versatile python essence.py tui
```

### Automatic Fallback

If the active provider fails (network error, rate limit, 5xx), the kernel tries each configured provider in priority order before falling back to Ollama as a last resort. A brief notice is emitted into the TUI when a fallback triggers.

### Model Discovery

```
/provider models              # discover from active provider
/provider models groq         # discover from Groq specifically
/provider models openai       # discover from OpenAI
```

### Storing API Keys

```
/provider set groq api_key gsk_...
/provider set openai api_key sk-...
/provider set anthropic api_key sk-ant-...
```

Keys are stored in the running environment (active immediately) and persisted to `config.toml` for the next session.

---

## Skill System

Skills are named, categorised agent behaviours with their own system prompts and enabled flags. Built-in in-process skills are merged with user-defined JSON skills from `memory/skills/`.

### Skill Categories

| Category | Description |
|---|---|
| `tool` | File reader, code executor, web search |
| `intelligence` | Summariser, critic, planner |
| `memory` | Memory retriever, context injector |
| `ledger` | Cost tracker, trust evaluator |
| `cie` | Interrupt budget, cognitive load scorer |
| `custom` | User-defined skills |

### Managing Skills

**CLI:**
```bash
python essence.py skills list
python essence.py skills add --name "Daily Briefing" --category tool --prompt "Summarise today's tasks"
python essence.py skills enable daily-briefing
python essence.py skills disable daily-briefing
python essence.py skills remove daily-briefing
python essence.py skills from-json skills_export.json
```

**TUI:**
```
/skills list
/skills enable <id>
/skills disable <id>
```

---

## Plugin System

Plugins are external Python files placed in `memory/plugins/`. Each plugin directory must include a `manifest.json` declaring its permissions and hooks.

### Plugin Manifest (`manifest.json`)

```json
{
  "id": "my-plugin",
  "name": "My Plugin",
  "version": "1.0.0",
  "permissions": ["read_files", "web_search"],
  "hooks": ["on_message", "on_turn_end"],
  "entry_point": "main.py"
}
```

**Valid permissions:** `read_files` `write_files` `web_search` `run_code` `network` `shell`

**Valid hooks:** `on_message` `on_turn_start` `on_turn_end` `on_skill_result` `on_session_start` `on_session_end`

### Managing Plugins

**CLI:**
```bash
python essence.py plugin list
python essence.py plugin install https://example.com/my_plugin.py
python essence.py plugin install ./my_plugin.py
python essence.py plugin uninstall my_plugin
```

**TUI:**
```
/plugin list
/plugin enable <id>
/plugin disable <id>
```

Plugin invocations run as isolated subprocesses (10-second timeout). Input and output are exchanged as JSON over stdin/stdout.

---

## Memory Architecture

Essence uses four layered memory stores:

### 1. Episodic Memory (`server/episodic_memory.py`)

Short-term working memory. Every user and assistant turn is appended. When the window exceeds `memory_window` turns, the LLM summarises the oldest turns and replaces them with a compact digest. The most recent turns are always injected into the system prompt.

### 2. Gravity Memory (`server/gravity_memory.py`)

Long-term persistent memory backed by SQLite. Each entry carries a **gravity score** representing its importance — higher-gravity facts are retained longer and surfaced more often. Supports semantic search. Retrieval results are injected into context on every turn.

```
/memory list 20
/memory search "deployment pipeline"
```

### 3. Offline Session Cache (`server/offline_cache.py`)

JSONL files written after every turn, one file per session under `data/offline_cache/sessions/`. Survives process restarts and provides turn-by-turn recovery. Also snapshots agent config and tool definitions.

### 4. Snapshots

Full session exports saved as JSON to `memory/snapshots/`. Include all message history, cost data, and model info. Restorable via `/restore <id>` in the TUI.

```bash
python essence.py snapshot save my-session
python essence.py snapshot list
```

---

## Offline Resilience

When the LLM backend becomes unreachable, Essence degrades gracefully without data loss:

| Layer | Behaviour |
|---|---|
| **Health probe** | Async HTTP check every 15 s — does not block the event loop |
| **Message queue** | Outbound messages serialised to `data/offline_cache/message_queue.jsonl` (up to 500 entries) |
| **Offline notice** | TUI shows `[offline — message queued; will retry on reconnect]` |
| **Auto-flush** | On reconnect, queue drains and messages replay through the kernel in order |
| **Re-queue on error** | If flush callback fails, messages are re-queued to prevent data loss |
| **Status** | `/kernel` shows live backend online/offline and queue depth |

---

## Governance & Audit

### Governance Engine (`server/governance.py`)

Controls how much autonomy the agent exercises:

| Level | Behaviour |
|---|---|
| `0` | Confirm every tool call before execution |
| `1` | Confirm only destructive / irreversible operations (default) |
| `2` | Fully autonomous — no confirmations |

Set in config or live:
```bash
# config.toml
autonomy_level = 1

# TUI
/config set agent.autonomy_level 2
```

Additional protections:
- **Trust ledger** — 3-axis trust (COMPETENCE / VALUES / JUDGMENT) per domain, built from observed outcomes
- **Idempotency guard** — duplicate command IDs replay cached responses rather than re-executing
- **HITL tool approval** — at level 0, a `tool.approval` event is published and execution waits for a reply

### Audit Logger (`server/audit_logger.py`)

Every significant action is appended to `data/audit_log.db`:

| Event Type | Logged When |
|---|---|
| `approval` | Tool call approved or denied |
| `model_switch` | Active provider or model changed |
| `config_change` | Any configuration key updated |
| `violation` | Governance policy breach detected |

**TUI:**
```
/audit
```

**Programmatic export:**
```python
from server.audit_logger import get_audit_logger
al = get_audit_logger()
al.export_compliance_report(fmt="csv", path="audit.csv")
al.export_compliance_report(fmt="json", path="audit.json")
```

---

## Automation & Cron

### Cron Jobs

Stored in `memory/cron_jobs.json`. Uses standard crontab syntax.

```bash
# Add a daily 8am briefing
python essence.py cron add "morning-brief" "0 8 * * *" "brief me"

python essence.py cron list
python essence.py cron enable morning-brief
python essence.py cron disable morning-brief
python essence.py cron rm morning-brief
```

**TUI manager:** `Ctrl+R`

### Proactive Patterns (`server/action_engine.py`)

Patterns fire on schedule, external events, or conditions. The proactive agent evaluates all active patterns on every heartbeat tick.

```bash
python essence.py automate list
python essence.py automate add "file-watcher" \
  --trigger condition \
  --prompt "alert if new files appear in ~/Downloads"
python essence.py automate run <id>     # fire immediately
python essence.py automate activate <id>
```

### Heartbeat

Background tick that drives proactive evaluation. Configurable in `config.toml`:

```toml
[heartbeat]
enabled  = true
interval = "30m"
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ESSENCE_MODEL` | _(config.toml)_ | Override active model name |
| `ESSENCE_BACKEND` | `ollama` | Override active provider |
| `ESSENCE_WORKSPACE` | _(script directory)_ | Workspace root path |
| `ESSENCE_OLLAMA_HOST` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_HOST` | `http://localhost:11434` | Alternative Ollama URL |
| `GROQ_API_KEY` | — | Groq Cloud API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `MISTRAL_API_KEY` | — | Mistral API key |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `ESSENCE_HF_TOKEN` | — | HuggingFace token for GGUF downloads |
| `HF_TOKEN` | — | Alternative HuggingFace token |
| `BRAVE_API_KEY` | — | Brave web search (2,000 free calls/month) |
| `SEARXNG_URL` | — | Self-hosted SearXNG instance URL |
| `TELEGRAM_BOT_TOKEN` | — | Telegram bot integration |
| `DISCORD_WEBHOOK_URL` | — | Discord webhook URL |
| `MLFLOW_TRACKING_URI` | — | MLflow experiment tracking server |
| `WANDB_API_KEY` | — | Weights & Biases API key |

---

## Directory Structure

```
essence/
├── essence.py                  # CLI entry point — all commands
├── tui_app.py                  # Textual TUI — full terminal interface
├── config.toml                 # Primary configuration
├── identity.toml               # Agent identity and persona settings
│
├── server/                     # Backend components
│   ├── kernel.py               # Cognitive kernel
│   ├── event_bus.py            # HMAC-signed pub/sub broker
│   ├── model_router.py         # Multi-provider LLM router
│   ├── offline_cache.py        # Offline resilience layer
│   ├── audit_logger.py         # Compliance audit trail (SQLite)
│   ├── governance.py           # Policy enforcement and trust
│   ├── identity_engine.py      # Identity state manager
│   ├── episodic_memory.py      # Short-term session memory
│   ├── gravity_memory.py       # Long-term memory with gravity scoring
│   ├── skill_orchestrator.py   # Built-in skill definitions
│   ├── skill_agent.py          # Skill execution harness
│   ├── skillstore.py           # JSON skill persistence
│   ├── plugin_sdk.py           # External plugin system
│   ├── action_engine.py        # Proactive automation engine
│   ├── tools_engine.py         # Tool registry and executor
│   ├── web_tools.py            # Web search and page fetch
│   ├── credstore.py            # Keyring-backed credential store
│   ├── mcpstore.py             # MCP server registry
│   ├── proactive_agent.py      # Background proactive agent
│   ├── materialized_views.py   # Pre-computed memory summaries
│   └── event_log.py            # Structured event log
│
├── memory/                     # Runtime data (auto-created)
│   ├── skills/                 # User-defined skill JSON files
│   ├── sessions/               # Session state files
│   ├── snapshots/              # Saved session snapshots
│   ├── ledger/                 # Trust ledger, MCP registry
│   ├── cost_log.json           # Per-session token and cost log
│   ├── cron_jobs.json          # Cron job definitions
│   └── schedule.json           # Automation schedule state
│
├── data/                       # Persistent databases (auto-created)
│   ├── audit_log.db            # SQLite audit trail
│   ├── gravity_memory.db       # SQLite long-term memory
│   └── offline_cache/          # Offline resilience store
│       ├── sessions/           # Per-session JSONL turn files
│       ├── agent_config.json   # Last-known agent config snapshot
│       ├── tool_definitions.json
│       └── message_queue.jsonl # Outbound queue when backend is offline
│
├── skills/                     # Built-in skill scripts
├── packages/                   # Optional packages (essence-core, etc.)
│
├── SOUL.md                     # Agent identity, values, work style
├── IDENTITY.md                 # Persona definition (read by agent)
├── GOALS.md                    # User-editable goals (read every turn)
├── LEARNED.md                  # Agent-maintained learned facts
├── MEMORY.md                   # Memory summary (auto-maintained)
├── AGENTS.md                   # Multi-agent configuration
├── PROJECTS.md                 # Active project context
├── TOOLS.md                    # Registered tool definitions
└── HEARTBEAT.md                # Heartbeat and automation config
```

---

## Design Principles

- **Privacy first** — data never leaves the machine unless you explicitly request it
- **Transparency** — every decision is logged; the audit trail is always accessible
- **Safety** — irreversible operations require confirmation at autonomy level 0 or 1
- **Resilience** — graceful degradation when backends are unavailable; no silent data loss
- **Composability** — skills, plugins, providers, and memory backends are independently configurable and replaceable
