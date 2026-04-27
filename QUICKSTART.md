# Essence v29 — First-Run Guide
**Unified Adaptive Intelligence System — Local AI that runs entirely on your machine.**

---

## What is Essence?

Essence is a personal AI agent that runs locally using [Ollama](https://ollama.com) for language models.
Everything stays on your computer — no data leaves your machine.

**What's new in v29:**
- **Living Intent Fabric (LIF)** — tracks your goals across 4 time horizons (now / this week / this month / life)
- **Theory of Mind (ToMM)** — models your cognitive load, focus, and stress so Essence knows when not to interrupt
- **Restraint Engine** — Essence defaults to *not* acting unless conditions are right (ABSTAIN-first)
- **Quiet Ledger** — every action AND every deliberate abstention is recorded — full accountability
- **Trust Ledger** — 3-axis trust (COMPETENCE / VALUES / JUDGMENT) builds over time per domain
- **CIE Scorer** — daily interrupt budget (10/day) prevents notification fatigue
- **V2 UI** — Svelte-based dark cyberpunk interface with 8 nav views
- **Multi-Provider Router** — switch between Ollama, OpenAI, Anthropic, Groq, Gemini, Mistral, Claude Code, LM Studio, llama.cpp, OpenRouter, Together AI, DeepSeek using `provider:model` syntax
- **GGUF Support** — scan local GGUF files, import into Ollama, or route to LM Studio / llama.cpp server
- **Skill System** — view and toggle 15 intelligence/CIE/ledger/tool skills from the UI
- **3-Pane Resizable Workspace** — primary output, preview renderer (markdown/JSON/code), context panel
- **Preview Renderer** — raw/rendered toggle per message, markdown, syntax-highlighted code, JSON
- **Debug Console** — real-time virtualized SSE log with type filters, search, JSON export
- **Session Sidebar** — full session management with search, export (Markdown), delete
- **File Ingestor** — drag-drop files onto the title bar to inject context into any conversation
- **Reasoning Dropdown** — top-right panel showing live TGS, intent, ToMM snapshot, trust axes
- **Notification Stack** — bottom-right auto-dismiss toasts for system events
- **Tier-aware** — auto-detects Lite/Standard/Mem-Opt/GPU-Acc based on available models

---

## Step 1 — Install Ollama (one-time, 2 minutes)

1. Go to **https://ollama.com/download** and download the Windows installer
2. Run the installer — it adds `ollama` to your PATH automatically
3. Open a terminal and verify: `ollama --version`

---

## Step 2 — Pull a language model (one-time, ~5 minutes)

Open a terminal and run:

```
ollama pull qwen3:4b
```

> **qwen3:4b** is the default — fast, fits on most machines (2.5 GB RAM).
> For a stronger model: `ollama pull qwen3:8b` (5 GB) or `ollama pull qwen3:14b` (9 GB)

Verify the model downloaded: `ollama list`

---

## Step 3 — Install Python dependencies (one-time, ~1 minute)

Open a terminal **in the Essence folder**:

```
cd C:\Users\MODAdministrator\AppData\Roaming\Essence
python essence.py install
```

This installs FastAPI, uvicorn, httpx, and all other required packages.

---

## Step 4 — Build the UI (one-time, ~2 minutes)

You need [Node.js](https://nodejs.org) (v18+) installed.

```
cd C:\Users\MODAdministrator\AppData\Roaming\Essence\ui
npm install
npm run build
```

This compiles the Svelte UI into `ui/dist/`. You only need to do this once (or after UI updates).

> **Skip this step** if you just want to use the API or TUI — the server still works without it.
> The legacy HTML UI at `server/index.html` will be used as fallback.

---

## Step 5 — Start Essence

```
cd C:\Users\MODAdministrator\AppData\Roaming\Essence
python essence.py up
```

You should see:
```
╔════════════════════════════════════════════════════════════╗
║  Essence  --  Essence — Your Private AI                 ║
╚════════════════════════════════════════════════════════════╝
  Server → http://0.0.0.0:7860
```

Open your browser: **http://localhost:7860**

---

## Step 6 — First conversation

1. The dark cyberpunk UI loads — you'll see the Essence interface
2. Type a message in the chat box and press **Enter**
3. Essence streams the response token by token

**Try these slash commands:**
| Command | What it does |
|---------|-------------|
| `/brief` | Generate your Morning Brief (top intents, system status) |
| `/status` | Show Ollama connection and model health |
| `/trust` | Show trust ledger for the 'general' domain |
| `/quiet` | Show the 24-hour Quiet Ledger summary |
| `/files` | List currently ingested file context |
| `/clear` | Clear chat history and start a new session |
| `/help` | Show all slash commands |

**Multi-provider model syntax:**
```
openai:gpt-4o
anthropic:claude-3-5-sonnet-20241022
groq:llama-3.3-70b-versatile
gemini:gemini-1.5-pro
mistral:mistral-large-latest
lmstudio:local-model
llamacpp:local-model
claude_code:
openrouter:meta-llama/llama-3.1-70b-instruct
deepseek:deepseek-chat
ollama:qwen3:4b  (or just qwen3:4b)
```

---

## Understanding the UI

```
┌──────────────────────────────────────────────────────────────────────────┐
│ [▶] U Essence v29  [standard]       ⬆ Drop files      qwen3:4b  REASON▼ ●live │  ← Titlebar (36px)
├────┬──────┬─────────────────────────────────────────────┬────────────────┤
│Sess│      │ INTENT: — | TGS 0.00 | RESTRAINT ABSTAIN    │                │  ← LIF Bar
│Side│  💬  │                                             │ Status  Log    │
│bar │  📁  │                                             │ Abstain Trust  │
│    │  ⬡   │   Chat / Sessions / Workspace / Console     │                │
│    │  🔧  │   Providers / Skills / Brief / Config       │  System info   │
│    │  🔌  │                                             │  Trust axes    │
│    │  ⚡  │                                             │  Log stream    │
│    │  ☀   │                                             │                │
│    │  ⚙   │                                             │                │
├────┴──────┴─────────────────────────────────────────────┴────────────────┤
│ TGS 0.00 · RESTRAINT ABSTAIN · TRUST T0/T0/T0 · CIE 10 left · TIER std  │  ← Status bar
└──────────────────────────────────────────────────────────────────────────┘
```

**[▶] button:** Toggle session sidebar (session list with search, export, delete)
**Title bar drag zone:** Drop files onto the title bar to inject them as context
**REASON▼ button:** Opens reasoning dropdown (TGS, intent, ToMM snapshot, trust axes)
**Activity Rail (8 icons):** Chat | Sessions | Workspace | Console | Providers | Skills | Brief | Config
**LIF bar:** Live intent tracking — TGS (Temporal Gravity Score) shows urgency pressure
**ToMM dropdown:** Click "ToMM▼" to see Essence's model of your current cognitive state
**Right panel:** Status (system health + CIE), Log (filterable SSE stream), Abstentions, Trust axes
**Workspace view:** Resizable 3-pane layout — primary output, preview renderer, context/files
**Console view:** Virtualized debug console with type filters, search, JSON export
**Providers view:** Configure all 12+ providers including local GGUF, cloud APIs, Claude Code CLI
**Skills view:** Toggle all 15 intelligence/CIE/ledger/tool skills

---

## Understanding the Trust Ledger

Essence tracks trust in three dimensions per domain:

| Axis | Meaning |
|------|---------|
| **COMP** (Competence) | Can Essence do this reliably? |
| **VALS** (Values) | Does Essence align with your values here? |
| **JUDG** (Judgment) | Do you agree with Essence's decisions here? |

Trust builds automatically through endorsements (`+0.08`) and erodes through overrides (`-0.10`).
Tiers advance at score ≥ 0.75 and regress at ≤ 0.25.

---

## Understanding Restraint (ABSTAIN-first)

Essence defaults to **not acting** until all conditions are met:
- Daily CIE budget not depleted (10 interrupts/day max)
- No active anomaly detected
- You're not in deep focus (ToMM: is_deep_focus = false)
- Trust tier is not T0 (blocked)
- Endorsement score ≥ 0.65
- Temporal Gravity Score sufficient for the interruption

Abstentions are recorded in the **Quiet Ledger** — you can always see what Essence chose not to do and why.

---

## Changing the Model

**In the UI:** Click ⚙ Config → select a different model from the dropdown

**Via environment variable:**
```
set ESSENCE_MODEL=qwen3:8b
python essence.py up
```

**Via config.toml:**
```toml
[inference]
model = "qwen3:8b"
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Ollama offline" in UI | Run `ollama serve` in a separate terminal |
| Model not found | Run `ollama pull qwen3:4b` |
| Port 7860 in use | Set `[server] port = 7861` in config.toml |
| UI shows fallback HTML | Run `cd ui && npm run build` |
| Slow responses | Switch to a smaller model (`qwen3:4b`) |
| Intelligence endpoints 503 | Check packages installed: `python essence.py install` |

---

## Directory Structure

```
Essence/
├── Essence.py              ← Main CLI (python essence.py up/chat/probe)
├── config.toml          ← All settings — edit freely
├── SOUL.md              ← Essence's identity and values (edit to personalise)
├── IDENTITY.md          ← Your profile (fill in for better responses)
├── server/
│   └── app.py           ← FastAPI server with all API endpoints
├── ui/
│   ├── src/             ← Svelte source (edit to customise UI)
│   └── dist/            ← Compiled UI (served by FastAPI)
├── packages/
│   ├── essence-intelligence/   ← LIF, ToMM, Restraint, FIS, MorningBrief
│   ├── essence-cie/            ← CIE Scorer, Routine Model, Anomaly Detector
│   └── essence-ledger/         ← QuietLedger, TrustLedger
├── memory/
│   └── ledger/          ← SQLite databases (trust.db, quiet.db, routine.json)
└── QUICKSTART.md        ← This file
```

---

## API Quick Reference

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Server + Ollama health check |
| `POST /api/chat` | Chat (streaming NDJSON) |
| `GET /api/intelligence/lif` | Living Intent Fabric state |
| `GET /api/intelligence/tomm` | Theory of Mind state |
| `GET /api/cie/status` | CIE budget + routine warmth |
| `GET /api/trust/general` | Trust axes for 'general' domain |
| `GET /api/quiet-ledger/summary` | 24h action/abstention summary |
| `GET /api/brief` | Generate Morning Brief |
| `GET /api/events` | SSE stream for UI real-time updates |
| `GET /api/docs` | Interactive Swagger API docs |

Full API: **http://localhost:7860/api/docs**

---

*Essence is privacy-first. Nothing leaves your machine unless you configure a cloud provider.*
