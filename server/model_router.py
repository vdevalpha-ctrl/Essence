"""
model_router.py — Essence Model Router
=====================================
Intelligent model selection with tier-awareness and automatic fallback:

  1. Selects the best available model for the requested task type
  2. Tracks token usage and error rates per provider
  3. Falls back to the next best provider when limits are hit
  4. Downloads and runs HuggingFace models locally when all cloud
     providers are exhausted or unavailable
  5. Respects the performance tier (lite / standard / mem-opt / gpu-acc)
     so large models are only attempted on capable hardware

Tier definitions
----------------
  lite      – CPU only, <8 GB RAM  → quantised 1-4B models only
  standard  – CPU/iGPU, 8-16 GB   → up to 8B models
  mem-opt   – 14-32 GB RAM        → up to 14B models
  gpu-acc   – dedicated GPU       → 30B+ models permitted

HuggingFace pull
----------------
When `pull_from_hf(repo_id, filename)` is called, the model is downloaded
via `huggingface_hub` into the workspace's `models/` directory and served
via llama-cpp-python if available, or ollama pull if not.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import json
import platform
import sqlite3
import subprocess
from threading import Lock

log = logging.getLogger("essence.model_router")

# ---------------------------------------------------------------------------
# Tier → max parameter size allowed
# ---------------------------------------------------------------------------
TIER_MAX_PARAMS: dict[str, int] = {
    "lite":     4,    # billion params
    "standard": 8,
    "mem-opt":  14,
    "gpu-acc":  72,
}

# ---------------------------------------------------------------------------
# Provider priority order for each task type
# ---------------------------------------------------------------------------
TASK_PROVIDER_ORDER: dict[str, list[str]] = {
    "chat":     ["ollama", "groq", "openai", "anthropic", "mistral", "together", "hf_local"],
    "code":     ["ollama", "groq", "openai", "anthropic", "hf_local"],
    "research": ["anthropic", "openai", "ollama", "groq", "hf_local"],
    "fast":     ["groq", "ollama", "openai", "hf_local"],
    "long":     ["anthropic", "openai", "ollama", "hf_local"],
    "default":  ["ollama", "groq", "openai", "anthropic", "hf_local"],
}

# ---------------------------------------------------------------------------
# HuggingFace model suggestions per tier for automatic pull
# ---------------------------------------------------------------------------
HF_MODELS_BY_TIER: dict[str, list[dict]] = {
    "lite": [
        {"repo": "TheBloke/phi-2-GGUF",          "file": "phi-2.Q4_K_M.gguf",       "params": 3},
        {"repo": "TheBloke/TinyLlama-1.1B-GGUF",  "file": "tinyllama-1.1b.Q4_K_M.gguf", "params": 1},
    ],
    "standard": [
        {"repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf", "params": 7},
        {"repo": "TheBloke/Llama-3-8B-Instruct-GGUF",      "file": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf", "params": 8},
    ],
    "mem-opt": [
        {"repo": "TheBloke/Mistral-Nemo-Instruct-2407-GGUF", "file": "mistral-nemo-instruct-2407.Q4_K_M.gguf", "params": 12},
        {"repo": "TheBloke/Qwen2-14B-Instruct-GGUF",         "file": "qwen2-14b-instruct.Q4_K_M.gguf",        "params": 14},
    ],
    "gpu-acc": [
        {"repo": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF", "file": "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf", "params": 47},
        {"repo": "TheBloke/Qwen2-72B-Instruct-GGUF",          "file": "qwen2-72b-instruct.Q4_K_M.gguf",         "params": 72},
    ],
}


# ---------------------------------------------------------------------------
# Cost table (USD per 1M tokens: [prompt, completion])
# ---------------------------------------------------------------------------
PROVIDER_COST_PER_1M: dict[str, tuple[float, float]] = {
    "ollama":     (0.00,   0.00),    # local — free
    "hf_local":   (0.00,   0.00),
    "groq":       (0.05,   0.08),
    "openai":     (0.15,   0.60),    # gpt-4o-mini
    "anthropic":  (0.25,   1.25),    # claude-3-5-haiku
    "mistral":    (0.10,   0.30),
    "together":   (0.10,   0.30),
    "openrouter": (0.10,   0.40),
    "deepseek":   (0.14,   0.28),
    "gemini":     (0.075,  0.30),
}

# ---------------------------------------------------------------------------
# OpenAI-compatible endpoint table
# ---------------------------------------------------------------------------
PROVIDER_BASE_URL: dict[str, str] = {
    "ollama":     "http://localhost:11434/v1",
    "groq":       "https://api.groq.com/openai/v1",
    "openai":     "https://api.openai.com/v1",
    "anthropic":  "https://api.anthropic.com/v1",
    "mistral":    "https://api.mistral.ai/v1",
    "together":   "https://api.together.xyz/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek":   "https://api.deepseek.com/v1",
    "gemini":     "https://generativelanguage.googleapis.com/v1beta/openai",
}


# ---------------------------------------------------------------------------
# Per-provider usage tracking
# ---------------------------------------------------------------------------
@dataclass
class ProviderUsage:
    provider_id:   str
    token_count:   int   = 0
    error_count:   int   = 0
    last_error:    float = 0.0
    last_success:  float = 0.0
    rate_limited:  bool  = False
    rl_until:      float = 0.0    # epoch when rate limit expires
    total_latency: float = 0.0    # cumulative seconds
    call_count:    int   = 0

    @property
    def avg_latency_s(self) -> float:
        return round(self.total_latency / self.call_count, 3) if self.call_count else 0.0

    @property
    def cost_per_1m_prompt(self) -> float:
        return PROVIDER_COST_PER_1M.get(self.provider_id, (0.10, 0.30))[0]

    @property
    def cost_per_1m_completion(self) -> float:
        return PROVIDER_COST_PER_1M.get(self.provider_id, (0.10, 0.30))[1]

    def is_available(self) -> bool:
        if self.rate_limited and time.time() < self.rl_until:
            return False
        if self.rate_limited and time.time() >= self.rl_until:
            self.rate_limited = False   # auto-reset
        return True

    def record_success(self, tokens: int = 0, latency_s: float = 0.0) -> None:
        self.last_success   = time.time()
        self.token_count   += tokens
        self.call_count    += 1
        self.total_latency += latency_s
        self.error_count    = max(0, self.error_count - 1)   # decay errors

    def record_error(self, is_rate_limit: bool = False, retry_after: float = 60.0) -> None:
        self.error_count  += 1
        self.last_error    = time.time()
        if is_rate_limit:
            self.rate_limited = True
            self.rl_until     = time.time() + retry_after


# ---------------------------------------------------------------------------
# Model Router
# ---------------------------------------------------------------------------
class ModelRouter:
    """
    Central model routing engine for Essence.

    Usage
    -----
        router = ModelRouter(workspace, ollama_url, performance_tier)
        provider, model = router.select("chat", preferred_model="qwen3:8b")
    """

    def __init__(
        self,
        workspace:    Path,
        ollama_url:   str  = "http://localhost:11434",
        perf_tier:    str  = "standard",
        provider_cfg: dict | None = None,
    ) -> None:
        self._workspace    = workspace
        self._ollama_url   = ollama_url
        self._perf_tier    = perf_tier
        self._provider_cfg = provider_cfg or {}
        self._usage: dict[str, ProviderUsage] = {}
        self._hf_available: bool | None = None   # lazy-checked
        self._local_models: list[dict]  = []     # HF models pulled locally
        self._last_provider: str = ""            # tracks switches for audit log
        self._last_model:    str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        task_type: str = "default",
        preferred_model: str = "",
        preferred_provider: str = "",
    ) -> tuple[str, str]:
        """
        Return (provider_id, model_name) for the given task.
        Falls back through the priority list until an available provider is found.
        """
        order = TASK_PROVIDER_ORDER.get(task_type, TASK_PROVIDER_ORDER["default"])

        # Honour explicit preference if that provider is healthy
        if preferred_provider and preferred_provider in order:
            usage = self._get_usage(preferred_provider)
            if usage.is_available():
                model = preferred_model or self._default_model(preferred_provider)
                return preferred_provider, model

        # Try providers in priority order
        for pid in order:
            if pid == "hf_local":
                if self._has_local_models():
                    m = self._best_local_model()
                    return "hf_local", m["file"] if m else "local"
                continue
            usage = self._get_usage(pid)
            if not usage.is_available():
                continue
            # Check provider is actually configured (has key or is local)
            if not self._provider_configured(pid):
                continue
            model = preferred_model if preferred_provider == pid else self._default_model(pid)
            return self._track_and_return(pid, model)

        # Last resort: return ollama with whatever model is set
        log.warning("ModelRouter: all providers exhausted — falling back to ollama")
        return self._track_and_return("ollama", preferred_model or "qwen3:4b")

    def _track_and_return(self, provider: str, model: str) -> tuple[str, str]:
        """Record a model switch to the audit log if provider/model changed."""
        if self._last_provider and (self._last_provider != provider or self._last_model != model):
            try:
                from server.audit_logger import get_audit_logger
                get_audit_logger().log_model_switch(
                    self._last_provider, self._last_model,
                    provider, model,
                    reason="router selection",
                )
            except Exception:
                pass
        self._last_provider = provider
        self._last_model    = model
        return provider, model

    def record_success(self, provider_id: str, tokens: int = 0, latency_s: float = 0.0) -> None:
        self._get_usage(provider_id).record_success(tokens, latency_s)

    def record_error(
        self,
        provider_id: str,
        is_rate_limit: bool = False,
        retry_after: float = 60.0,
    ) -> None:
        u = self._get_usage(provider_id)
        u.record_error(is_rate_limit, retry_after)
        if is_rate_limit:
            log.warning(
                "ModelRouter: %s rate-limited for %.0fs", provider_id, retry_after
            )

    def normalize_to_openai(
        self,
        provider_id:     str,
        messages:        list[dict],
        model:           str,
        tools:           list[dict] | None = None,
        stream:          bool = True,
        temperature:     float = 0.7,
        max_tokens:      int = 2048,
        response_format: str = "",   # "" | "json_object" | "json_schema"
        reasoning_mode:  str = "",   # "" | budget_tokens (int str) for Anthropic;
                                     # "high"|"medium"|"low" for OpenAI o-series
    ) -> tuple[str, dict, dict]:
        """
        Return (endpoint_url, request_body, headers) for the target provider.

        All callers pass messages in standard OpenAI format.  This method
        handles every provider-specific translation so no other code needs
        to know about provider wire-format differences:

          • Anthropic: system prompt extracted to top-level field, messages
            normalised, Anthropic-format tools injected, x-api-key header
          • OpenAI-compatible providers: pass-through with correct auth header
          • Providers without native tools: tool defs injected as system prompt
        """
        # ── Resolve base URL ──────────────────────────────────────────
        cfg  = self._provider_cfg.get(provider_id, {})
        base = (
            cfg.get("base_url")                                   # runtime override
            or PROVIDER_BASE_URL.get(provider_id, PROVIDER_BASE_URL["ollama"])
        )
        if provider_id == "ollama":
            base = f"{self._ollama_url}/v1"   # always use configured Ollama URL

        url     = f"{base}/chat/completions"
        headers = {"Content-Type": "application/json"}

        # ── Resolve API key ───────────────────────────────────────────
        api_key = cfg.get("api_key") or os.environ.get(
            f"ESSENCE_{provider_id.upper()}_KEY",
            os.environ.get(f"{provider_id.upper()}_API_KEY", ""),
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # ── Provider-specific header adjustments ───────────────────────
        if provider_id == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
            if api_key:
                headers["x-api-key"] = api_key
            headers.pop("Authorization", None)
        elif provider_id == "openrouter":
            # Configurable via env so forks / private deployments don't leak the original repo URL
            _or_referer = os.environ.get("OPENROUTER_REFERER", "https://github.com/olatunjih/Essence")
            _or_title   = os.environ.get("OPENROUTER_TITLE", "Essence")
            headers["HTTP-Referer"] = _or_referer
            headers["X-Title"]      = _or_title

        # ── Message normalisation (provider-agnostic bridge) ──────────
        from server.tool_bridge import get_tool_bridge
        bridge = get_tool_bridge()

        # Normalise message shapes (tool role → provider-correct format, etc.)
        norm_messages = bridge.normalise_messages(provider_id, list(messages))

        # Anthropic requires system prompt in a dedicated top-level field
        if provider_id == "anthropic":
            system_text, norm_messages = bridge.extract_anthropic_system(norm_messages)
        else:
            system_text = ""

        # ── Build base body ───────────────────────────────────────────
        body: dict = {
            "model":       model,
            "messages":    norm_messages,
            "stream":      stream,
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }
        if system_text and provider_id == "anthropic":
            body["system"] = system_text

        # ── Tool injection (provider-agnostic) ────────────────────────
        # The bridge handles OpenAI format, Anthropic format, and the
        # system-prompt fallback for providers without native tool support.
        if tools:
            bridge.adapt_request(provider_id, body, tools)

        # ── Extended reasoning mode ───────────────────────────────────
        if reasoning_mode:
            if provider_id == "anthropic":
                # Anthropic extended thinking: budget_tokens required; temperature=1
                try:
                    budget = int(reasoning_mode)
                except ValueError:
                    budget = 8000   # default when non-numeric string given
                body["thinking"]    = {"type": "enabled", "budget_tokens": budget}
                body["temperature"] = 1  # Anthropic requires exactly 1 for thinking
                headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"
            elif provider_id in ("openai",):
                # OpenAI o-series: reasoning_effort parameter
                effort = reasoning_mode if reasoning_mode in ("high", "medium", "low") else "high"
                body["reasoning_effort"] = effort
                # o-series don't support temperature or max_tokens (use max_completion_tokens)
                body.pop("temperature", None)
                mc = body.pop("max_tokens", max_tokens)
                body["max_completion_tokens"] = mc
            # Other providers: ignore — extended thinking not supported

        # ── Structured output / JSON mode ─────────────────────────────
        if response_format:
            if provider_id == "anthropic":
                # Anthropic has no response_format field — use a system-prompt prefix
                _json_hint = (
                    "Respond with raw JSON only. No markdown fences, no explanation — "
                    "output a single valid JSON object or array.\n\n"
                )
                if "system" in body:
                    body["system"] = _json_hint + body["system"]
                else:
                    # Inject into system message in messages list
                    for _m in body.get("messages", []):
                        if _m.get("role") == "system":
                            _m["content"] = _json_hint + _m.get("content", "")
                            break
            elif provider_id == "ollama":
                # Ollama uses a top-level "format" key
                body["format"] = "json"
            else:
                # All other OpenAI-compatible providers support response_format
                body["response_format"] = {"type": response_format}

        return url, body, headers

    def select_by_cost_latency(
        self,
        task_type:      str   = "default",
        max_cost_usd:   float = 0.01,    # per request budget (prompt + completion)
        max_latency_s:  float = 10.0,
        est_tokens:     int   = 1000,
    ) -> tuple[str, str]:
        """
        Pick the cheapest available provider that fits within cost and latency budgets.
        Falls back to ModelRouter.select() if no provider qualifies.
        """
        order = TASK_PROVIDER_ORDER.get(task_type, TASK_PROVIDER_ORDER["default"])
        best_provider, best_model, best_cost = "", "", float("inf")

        for pid in order:
            usage = self._get_usage(pid)
            if not usage.is_available():
                continue
            if not self._provider_configured(pid):
                continue
            # Estimate cost for est_tokens (50% prompt / 50% completion split assumed)
            p_cost  = usage.cost_per_1m_prompt    * (est_tokens * 0.5) / 1_000_000
            c_cost  = usage.cost_per_1m_completion * (est_tokens * 0.5) / 1_000_000
            est_cost = p_cost + c_cost
            if est_cost > max_cost_usd:
                continue
            # Latency check (only meaningful after first call)
            if usage.call_count > 0 and usage.avg_latency_s > max_latency_s:
                continue
            if est_cost < best_cost:
                best_cost     = est_cost
                best_provider = pid
                best_model    = self._default_model(pid)

        if best_provider:
            log.info(
                "ModelRouter.select_by_cost_latency: picked %s (est cost $%.5f)",
                best_provider, best_cost,
            )
            return best_provider, best_model

        log.debug("ModelRouter.select_by_cost_latency: no provider met thresholds — falling back")
        return self.select(task_type)

    def fallback_chain(
        self,
        task_type:       str = "default",
        exclude_provider: str = "",
    ) -> list[tuple[str, str]]:
        """
        Return an ordered list of (provider_id, model) candidates to try.

        The active/preferred provider is excluded (it already failed).
        Only includes providers that are configured and currently available.
        Ollama is always appended as the last resort even without a key.
        """
        order = TASK_PROVIDER_ORDER.get(task_type, TASK_PROVIDER_ORDER["default"])
        chain: list[tuple[str, str]] = []
        seen: set[str] = set()

        for pid in order:
            if pid == exclude_provider or pid in seen:
                continue
            seen.add(pid)
            if pid == "hf_local":
                if self._has_local_models():
                    m = self._best_local_model()
                    if m:
                        chain.append(("hf_local", m["file"]))
                continue
            if not self._provider_configured(pid):
                continue
            if not self._get_usage(pid).is_available():
                continue
            chain.append((pid, self._default_model(pid)))

        # Ollama is always the last resort (works even without a key)
        if exclude_provider != "ollama" and "ollama" not in seen:
            chain.append(("ollama", self._default_model("ollama")))

        return chain

    def set_provider_key(self, provider_id: str, api_key: str, base_url: str = "") -> None:
        """Hot-swap the API key (and optionally base URL) for a provider.

        Called by the kernel after a credential pool rotation so that the
        next normalize_to_openai() call uses the newly selected key.
        """
        if provider_id not in self._provider_cfg:
            self._provider_cfg[provider_id] = {}
        if api_key:
            self._provider_cfg[provider_id]["api_key"] = api_key
        if base_url:
            self._provider_cfg[provider_id]["base_url"] = base_url
        log.debug("ModelRouter: updated key for provider %s", provider_id)

    def set_tier(self, tier: str) -> None:
        self._perf_tier = tier

    def usage_summary(self) -> dict:
        return {
            pid: {
                "tokens":       u.token_count,
                "errors":       u.error_count,
                "available":    u.is_available(),
                "rate_limited": u.rate_limited,
            }
            for pid, u in self._usage.items()
        }

    # ------------------------------------------------------------------
    # HuggingFace pull
    # ------------------------------------------------------------------

    async def pull_from_hf(
        self,
        repo_id:  str,
        filename: str,
        progress_cb=None,
        hf_token: str = "",
    ) -> Path | None:
        """
        Download a GGUF model from HuggingFace Hub into workspace/models/.
        Returns the local path, or None on failure.
        Requires: pip install huggingface_hub

        Authentication:
          - Pass hf_token explicitly, OR
          - Set env var ESSENCE_HF_TOKEN or HF_TOKEN (checked automatically)
          - Required for gated / private models on HuggingFace
        """
        models_dir = self._workspace / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        dest = models_dir / filename
        if dest.exists():
            log.info("ModelRouter: HF model already local: %s", dest)
            self._register_local_model(repo_id, filename, dest)
            return dest

        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except ImportError:
            log.error("ModelRouter: huggingface_hub not installed — run: pip install huggingface_hub")
            return None

        # Resolve HF token: explicit arg > env ESSENCE_HF_TOKEN > env HF_TOKEN
        token = (
            hf_token
            or os.environ.get("ESSENCE_HF_TOKEN", "")
            or os.environ.get("HF_TOKEN", "")
            or None
        )
        if token:
            log.info("ModelRouter: using HF token for pull")
        else:
            log.info("ModelRouter: no HF token — pulling public model")

        log.info("ModelRouter: pulling %s/%s from HuggingFace…", repo_id, filename)
        try:
            loop = asyncio.get_event_loop()
            local = await loop.run_in_executor(
                None,
                lambda: hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(models_dir),
                    local_dir_use_symlinks=False,
                    token=token,
                ),
            )
            dest = Path(local)
            self._register_local_model(repo_id, filename, dest)
            log.info("ModelRouter: HF model saved → %s", dest)
            if progress_cb:
                progress_cb({"status": "done", "path": str(dest)})
            return dest
        except Exception as e:
            log.error("ModelRouter: HF pull failed: %s", e)
            if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
                log.error(
                    "ModelRouter: This is a gated/private model. "
                    "Set ESSENCE_HF_TOKEN env var with your HuggingFace token. "
                    "Get one at: https://huggingface.co/settings/tokens"
                )
            return None

    async def suggest_and_pull_hf(self) -> dict | None:
        """
        Pick the best HF model for the current tier and pull it.
        Returns info dict or None.
        """
        candidates = HF_MODELS_BY_TIER.get(self._perf_tier, HF_MODELS_BY_TIER["standard"])
        max_p = TIER_MAX_PARAMS.get(self._perf_tier, 8)
        for c in candidates:
            if c["params"] <= max_p:
                path = await self.pull_from_hf(c["repo"], c["file"])
                if path:
                    return {**c, "local_path": str(path)}
        return None

    # ------------------------------------------------------------------
    # Load local models already on disk
    # ------------------------------------------------------------------

    def scan_local_models(self) -> list[dict]:
        """Scan workspace/models/ for GGUF files and register them."""
        models_dir = self._workspace / "models"
        if not models_dir.exists():
            return []
        self._local_models = [
            {"file": f.name, "path": str(f), "size_gb": round(f.stat().st_size / 1e9, 2)}
            for f in sorted(models_dir.glob("*.gguf"))
        ]
        if self._local_models:
            log.info("ModelRouter: found %d local GGUF models", len(self._local_models))
        return self._local_models

    # ------------------------------------------------------------------
    # Ollama model availability check
    # ------------------------------------------------------------------

    async def get_ollama_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{self._ollama_url}/api/tags")
                if r.status_code == 200:
                    return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # Smart model selection (async, with optional HITL)
    # ------------------------------------------------------------------

    def _score_ollama_model(self, name: str) -> tuple[bool, int]:
        """
        Parse param count from an Ollama model tag, e.g. 'qwen3:4b' → (True, 4).
        Returns (fits_in_tier, param_count_billions).
        Models whose inferred param count exceeds the tier ceiling are marked False.
        Models with unknown size are optimistically included with count=0.
        """
        max_p = TIER_MAX_PARAMS.get(self._perf_tier, 8)
        m = re.search(r"[:\-_v](\d+(?:\.\d+)?)b", name.lower())
        if m:
            params = float(m.group(1))
            return params <= max_p, int(params)
        return True, 0   # unknown size — include

    async def select_best_available(
        self,
        task_type:  str  = "default",
        hitl:       bool = False,
        n_suggest:  int  = 8,
    ) -> tuple[str, str]:
        """
        Intelligently pick the best available model, in order:

          1. Query Ollama for pulled models, score by tier compatibility.
          2. If hitl=True and multiple candidates exist → interactive picker.
          3. If Ollama has no models → scan local GGUF files.
          4. If no local GGUFs → suggest + pull from HuggingFace (smallest tier fit).
          5. Absolute last resort → 'ollama', 'qwen3:4b'.

        Returns (provider_id, model_name).
        """
        models = await self.get_ollama_models()

        if not models:
            log.info("ModelRouter: Ollama empty — scanning local GGUF cache")
            self.scan_local_models()
            if self._local_models:
                best = self._best_local_model()
                if best:
                    log.info("ModelRouter: using local GGUF %s", best["file"])
                    return "hf_local", best["file"]

            log.info("ModelRouter: no local models; attempting HF suggestion-pull")
            pulled = await self.suggest_and_pull_hf()
            if pulled:
                return "hf_local", pulled["file"]

            log.warning("ModelRouter: all sources exhausted — falling back to qwen3:4b")
            return "ollama", "qwen3:4b"

        # ── Score Ollama models ──────────────────────────────────
        scored: list[tuple[int, int, str]] = []
        excluded: list[str] = []
        for name in models:
            fits, params = self._score_ollama_model(name)
            if not fits:
                excluded.append(name)
                continue
            # Boost instruct / chat tuned variants
            quality_bonus = 1 if any(
                kw in name.lower()
                for kw in ("instruct", "chat", "-it", "q4_k", "q8_0")
            ) else 0
            scored.append((params + quality_bonus, params, name))

        if excluded:
            log.debug(
                "ModelRouter: skipped %d models exceeding tier '%s' limit: %s",
                len(excluded), self._perf_tier, excluded,
            )

        if not scored:
            # No model fits the tier — fall back to smallest available
            log.warning(
                "ModelRouter: no models fit tier '%s'; using smallest available",
                self._perf_tier,
            )
            scored = [(0, 0, name) for name in models]

        # Best first (largest params that fit)
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        if hitl and len(scored) > 1:
            selected = self._hitl_select(scored, n_suggest)
            return "ollama", selected

        _, _, best = scored[0]
        log.info("ModelRouter: auto-selected '%s' (tier: %s)", best, self._perf_tier)
        return "ollama", best

    def _hitl_select(
        self,
        scored:    list[tuple[int, int, str]],
        n_suggest: int = 8,
    ) -> str:
        """
        Present a numbered model menu and return the user's choice.
        Falls back to the top-scored model on any input error.
        """
        choices  = [name for _, _, name in scored[:n_suggest]]
        max_p    = TIER_MAX_PARAMS.get(self._perf_tier, 8)

        print(f"\n  ┌─ Model Selection  (tier: {self._perf_tier} · max {max_p}B params) ──────")
        for i, name in enumerate(choices, 1):
            _, params, _ = scored[i - 1]
            size_label   = f"{params}B" if params else "?B"
            note         = "  ◄ recommended" if i == 1 else ""
            print(f"  │  [{i}] {name:<40} {size_label}{note}")
        print(f"  │  [0] Enter model name manually")
        print(f"  └" + "─" * 54)

        try:
            raw = input(f"  Select [1–{len(choices)}]  (↵ = {choices[0]}): ").strip()
            if not raw:
                return choices[0]
            idx = int(raw)
            if idx == 0:
                custom = input("  Model name: ").strip()
                return custom or choices[0]
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
        except (ValueError, IndexError, EOFError, KeyboardInterrupt):
            pass
        return choices[0]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_usage(self, pid: str) -> ProviderUsage:
        if pid not in self._usage:
            self._usage[pid] = ProviderUsage(provider_id=pid)
        return self._usage[pid]

    def _provider_configured(self, pid: str) -> bool:
        if pid == "ollama":
            return True   # always try ollama
        cfg = self._provider_cfg.get(pid, {})
        return bool(cfg.get("api_key") or cfg.get("base_url"))

    def _default_model(self, pid: str) -> str:
        defaults = {
            "ollama":     "qwen3:4b",
            "groq":       "llama3-70b-8192",
            "openai":     "gpt-4o-mini",
            "anthropic":  "claude-3-5-haiku-20241022",
            "mistral":    "mistral-small-latest",
            "together":   "meta-llama/Llama-3-8b-chat-hf",
            "openrouter": "mistralai/mistral-7b-instruct",
            "deepseek":   "deepseek-chat",
            "gemini":     "gemini-1.5-flash",
        }
        return defaults.get(pid, "default")

    def _has_local_models(self) -> bool:
        if not self._local_models:
            self.scan_local_models()
        return bool(self._local_models)

    def _best_local_model(self) -> dict | None:
        if not self._local_models:
            return None
        max_p = TIER_MAX_PARAMS.get(self._perf_tier, 8)
        # Prefer largest model that fits in tier
        for m in reversed(self._local_models):
            return m   # just return the last one for now
        return None

    def _register_local_model(self, repo_id: str, filename: str, path: Path) -> None:
        entry = {"repo": repo_id, "file": filename, "path": str(path),
                 "size_gb": round(path.stat().st_size / 1e9, 2)}
        if not any(m["file"] == filename for m in self._local_models):
            self._local_models.append(entry)


# ---------------------------------------------------------------------------
# HFInferenceBackend — llama-cpp-python direct inference
# ---------------------------------------------------------------------------

class HFInferenceBackend:
    """
    Direct llama-cpp-python inference backend.

    Used when Ollama is unavailable but a local GGUF model is on disk.
    Requires:  pip install llama-cpp-python

    GPU support (optional):
      pip install llama-cpp-python --extra-index-url \
          https://abetlen.github.io/llama-cpp-python/whl/cu121

    HITL pull flow
    ---------------
      router = ModelRouter(workspace)
      _, model_file = await router.select_best_available(hitl=True)
      backend = HFInferenceBackend(workspace / "models" / model_file)
      reply   = backend.chat([{"role": "user", "content": "hello"}])
    """

    def __init__(
        self,
        model_path:    "str | Path",
        n_ctx:         int = 4096,
        n_gpu_layers:  int = 0,    # set > 0 to offload layers to GPU
        verbose:       bool = False,
    ) -> None:
        self._path         = str(model_path)
        self._n_ctx        = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._verbose      = verbose
        self._llm          = None   # lazy-loaded on first call

    # ── Availability check ────────────────────────────────────────────

    @staticmethod
    def llama_cpp_available() -> bool:
        """True iff llama-cpp-python is importable."""
        try:
            import llama_cpp  # type: ignore  # noqa: F401
            return True
        except ImportError:
            return False

    def is_ready(self) -> bool:
        """True iff the model file exists and llama-cpp-python is installed."""
        return Path(self._path).exists() and self.llama_cpp_available()

    # ── Lazy model load ───────────────────────────────────────────────

    def _load(self) -> None:
        if self._llm is not None:
            return
        if not Path(self._path).exists():
            raise FileNotFoundError(f"GGUF model not found: {self._path}")
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed.\n"
                "Run:  pip install llama-cpp-python\n"
                "For GPU:  pip install llama-cpp-python "
                "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
            )
        log.info("HFInferenceBackend: loading %s (ctx=%d, gpu_layers=%d)",
                 self._path, self._n_ctx, self._n_gpu_layers)
        self._llm = Llama(
            model_path=self._path,
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            verbose=self._verbose,
        )
        log.info("HFInferenceBackend: model ready")

    # ── Inference ─────────────────────────────────────────────────────

    def chat(
        self,
        messages:    list[dict],
        max_tokens:  int   = 512,
        temperature: float = 0.7,
        stream:      bool  = False,
    ) -> "str | Any":
        """
        Synchronous chat completion.

        Parameters
        ----------
        messages    : OpenAI-style list of {role, content} dicts.
        max_tokens  : Maximum tokens to generate.
        temperature : Sampling temperature.
        stream      : If True, returns a generator of token strings.
                      If False, returns the complete response string.
        """
        self._load()
        result = self._llm.create_chat_completion(   # type: ignore[union-attr]
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )
        if stream:
            # Yield token strings from the streaming generator
            def _gen():
                for chunk in result:
                    delta = chunk["choices"][0].get("delta", {})
                    tok   = delta.get("content", "")
                    if tok:
                        yield tok
            return _gen()
        return result["choices"][0]["message"]["content"]

    async def achat(
        self,
        messages:    list[dict],
        max_tokens:  int   = 512,
        temperature: float = 0.7,
    ) -> str:
        """Async wrapper — runs blocking inference in a thread pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.chat(messages, max_tokens, temperature, stream=False),
        )

    def unload(self) -> None:
        """Release the loaded model and free memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            log.info("HFInferenceBackend: model unloaded")


# ---------------------------------------------------------------------------
# DeviceProfile — hardware detection for smart tier selection
# ---------------------------------------------------------------------------

@dataclass
class DeviceProfile:
    """
    Hardware snapshot used to derive the performance tier and recommend models.

    Tier derivation
    ---------------
      lite     : effective_gb <  8  (CPU only, 1-4B models)
      standard : effective_gb <  16 (iGPU / CPU, up to 8B)
      mem-opt  : effective_gb <  32 (up to 14B)
      gpu-acc  : effective_gb >= 32 OR dedicated GPU with ≥16 GB VRAM
    where effective_gb = VRAM (if GPU present and VRAM > 0) else RAM.
    """
    cpu_cores:  int   = 0
    ram_gb:     float = 0.0
    has_gpu:    bool  = False
    gpu_name:   str   = ""
    vram_gb:    float = 0.0
    has_npu:    bool  = False
    platform:   str   = ""
    arch:       str   = ""
    tier:       str   = "standard"

    @classmethod
    def detect(cls) -> "DeviceProfile":
        """Probe available hardware and return a DeviceProfile."""
        p = cls()
        import sys as _sys
        import platform as _plt
        p.platform = _sys.platform
        p.arch = _plt.machine().lower()

        try:
            import psutil  # type: ignore
            p.ram_gb    = psutil.virtual_memory().total / 1024 ** 3
            p.cpu_cores = psutil.cpu_count(logical=False) or 1
        except ImportError:
            p.cpu_cores = __import__("os").cpu_count() or 1
            p.ram_gb    = 8.0

        # NVIDIA GPU
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0 and r.stdout.strip():
                parts     = r.stdout.strip().split(",")
                p.has_gpu  = True
                p.gpu_name = parts[0].strip()
                p.vram_gb  = float(parts[1].strip()) / 1024 if len(parts) > 1 else 0.0
        except Exception:
            pass

        # AMD GPU (rocm-smi)
        if not p.has_gpu:
            try:
                r = subprocess.run(
                    ["rocm-smi", "--showproductname"],
                    capture_output=True, text=True, timeout=3,
                )
                if r.returncode == 0 and "GPU" in r.stdout:
                    p.has_gpu  = True
                    p.gpu_name = "AMD GPU (rocm)"
            except Exception:
                pass

        # NPU / Neural Engine
        if p.arch in ("arm64", "aarch64") and p.platform == "darwin":
            p.has_npu = True   # Apple Silicon Neural Engine

        # Derive tier
        eff = p.vram_gb if (p.has_gpu and p.vram_gb > 0) else p.ram_gb
        if eff >= 32 or (p.has_gpu and p.vram_gb >= 16):
            p.tier = "gpu-acc"
        elif eff >= 16:
            p.tier = "mem-opt"
        elif eff >= 8:
            p.tier = "standard"
        else:
            p.tier = "lite"

        return p

    def summary(self) -> str:
        parts = [f"CPU {self.cpu_cores}c  RAM {self.ram_gb:.1f} GB"]
        if self.has_gpu:
            parts.append(f"GPU {self.gpu_name} {self.vram_gb:.1f} GB VRAM")
        if self.has_npu:
            parts.append("NPU")
        return "  |  ".join(parts) + f"   tier: {self.tier}"


# ---------------------------------------------------------------------------
# Provider registry (SQLite-backed, multi-cloud)
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider (local or cloud)."""
    id:            str
    name:          str
    api_key:       str   = ""
    base_url:      str   = ""
    priority:      int   = 50      # lower = higher priority
    enabled:       bool  = True
    default_model: str   = ""
    extra:         dict  = field(default_factory=dict)


_PROVIDER_DDL = """
CREATE TABLE IF NOT EXISTS providers (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    api_key       TEXT NOT NULL DEFAULT '',
    base_url      TEXT NOT NULL DEFAULT '',
    priority      INTEGER NOT NULL DEFAULT 50,
    enabled       INTEGER NOT NULL DEFAULT 1,
    default_model TEXT NOT NULL DEFAULT '',
    extra         TEXT NOT NULL DEFAULT '{}'
);
"""

_BUILTIN_PROVIDERS: list[dict] = [
    {"id": "ollama",     "name": "Ollama (local)",          "base_url": "http://localhost:11434",                  "priority": 1,  "enabled": True,  "default_model": ""},
    {"id": "hf_local",  "name": "HuggingFace Local (GGUF)", "base_url": "",                                       "priority": 5,  "enabled": True,  "default_model": ""},
    {"id": "groq",       "name": "Groq",                     "base_url": "https://api.groq.com/openai/v1",         "priority": 10, "enabled": False, "default_model": "llama-3.1-8b-instant"},
    {"id": "openai",     "name": "OpenAI",                   "base_url": "https://api.openai.com/v1",              "priority": 15, "enabled": False, "default_model": "gpt-4o-mini"},
    {"id": "anthropic",  "name": "Anthropic",                "base_url": "https://api.anthropic.com",              "priority": 20, "enabled": False, "default_model": "claude-3-5-haiku-20241022"},
    {"id": "bedrock",    "name": "AWS Bedrock",              "base_url": "",                                       "priority": 25, "enabled": False, "default_model": "amazon.titan-text-express-v1"},
    {"id": "gemini",     "name": "Google Gemini",            "base_url": "https://generativelanguage.googleapis.com/v1beta", "priority": 28, "enabled": False, "default_model": "gemini-1.5-flash"},
    {"id": "mistral",    "name": "Mistral AI",               "base_url": "https://api.mistral.ai/v1",              "priority": 30, "enabled": False, "default_model": "mistral-small-latest"},
    {"id": "deepseek",   "name": "DeepSeek",                 "base_url": "https://api.deepseek.com/v1",            "priority": 32, "enabled": False, "default_model": "deepseek-chat"},
    {"id": "together",   "name": "Together AI",              "base_url": "https://api.together.xyz/v1",            "priority": 35, "enabled": False, "default_model": "meta-llama/Llama-3-8b-chat-hf"},
    {"id": "openrouter", "name": "OpenRouter",               "base_url": "https://openrouter.ai/api/v1",           "priority": 40, "enabled": False, "default_model": "mistralai/mistral-7b-instruct"},
    {"id": "lmstudio",   "name": "LM Studio (local)",        "base_url": "http://localhost:1234/v1",               "priority":  6, "enabled": False, "default_model": ""},
]


class ProviderRegistry:
    """SQLite-backed CRUD store for provider configurations."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._lock = Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_PROVIDER_DDL)
            conn.commit()
        self._seed_builtins()

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self._path), check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _to_cfg(self, row: sqlite3.Row) -> ProviderConfig:
        d = dict(row)
        return ProviderConfig(
            id=d["id"], name=d["name"], api_key=d["api_key"],
            base_url=d["base_url"], priority=d["priority"],
            enabled=bool(d["enabled"]), default_model=d["default_model"],
            extra=json.loads(d["extra"] or "{}"),
        )

    def _seed_builtins(self) -> None:
        for b in _BUILTIN_PROVIDERS:
            with self._connect() as conn:
                exists = conn.execute("SELECT 1 FROM providers WHERE id=?", (b["id"],)).fetchone()
                if not exists:
                    conn.execute(
                        "INSERT INTO providers (id,name,base_url,priority,enabled,default_model,extra) VALUES (?,?,?,?,?,?,'{}')",
                        (b["id"], b["name"], b["base_url"], b["priority"], int(b.get("enabled", False)), b.get("default_model", "")),
                    )
                    conn.commit()

    def upsert(self, cfg: ProviderConfig) -> ProviderConfig:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO providers (id,name,api_key,base_url,priority,enabled,default_model,extra)
                       VALUES (?,?,?,?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET
                         name=excluded.name, api_key=excluded.api_key, base_url=excluded.base_url,
                         priority=excluded.priority, enabled=excluded.enabled,
                         default_model=excluded.default_model, extra=excluded.extra""",
                    (cfg.id, cfg.name, cfg.api_key, cfg.base_url, cfg.priority,
                     int(cfg.enabled), cfg.default_model, json.dumps(cfg.extra)),
                )
                conn.commit()
        return cfg

    def get(self, pid: str) -> ProviderConfig | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM providers WHERE id=?", (pid,)).fetchone()
        return self._to_cfg(row) if row else None

    def list_all(self) -> list[ProviderConfig]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM providers ORDER BY priority ASC").fetchall()
        return [self._to_cfg(r) for r in rows]

    def list_enabled(self) -> list[ProviderConfig]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM providers WHERE enabled=1 ORDER BY priority ASC").fetchall()
        return [self._to_cfg(r) for r in rows]

    def patch(self, pid: str, updates: dict) -> ProviderConfig | None:
        cfg = self.get(pid)
        if not cfg:
            return None
        for k, v in updates.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return self.upsert(cfg)

    def delete(self, pid: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                n = conn.execute("DELETE FROM providers WHERE id=?", (pid,)).rowcount
                conn.commit()
        return n > 0


# ---------------------------------------------------------------------------
# Model catalog with favourites (SQLite-backed)
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    """Canonical model record across all providers."""
    id:             str                         # "{provider}:{model_name}"
    provider:       str
    model_name:     str
    display_name:   str  = ""
    params_b:       int  = 0                    # param count in billions (0 = unknown)
    context_window: int  = 4096
    capabilities:   list = field(default_factory=list)   # chat|code|vision|tools|embedding|reasoning|long_context
    input_price:    float = 0.0                 # per 1M tokens USD (0 = free/local)
    output_price:   float = 0.0
    is_favourite:   bool  = False
    tier_min:       str   = "lite"              # minimum device tier
    tags:           list  = field(default_factory=list)
    notes:          str   = ""


_CATALOG_DDL = """
CREATE TABLE IF NOT EXISTS models (
    id             TEXT PRIMARY KEY,
    provider       TEXT NOT NULL,
    model_name     TEXT NOT NULL,
    display_name   TEXT NOT NULL DEFAULT '',
    params_b       INTEGER NOT NULL DEFAULT 0,
    context_window INTEGER NOT NULL DEFAULT 4096,
    capabilities   TEXT NOT NULL DEFAULT '[]',
    input_price    REAL NOT NULL DEFAULT 0.0,
    output_price   REAL NOT NULL DEFAULT 0.0,
    is_favourite   INTEGER NOT NULL DEFAULT 0,
    tier_min       TEXT NOT NULL DEFAULT 'lite',
    tags           TEXT NOT NULL DEFAULT '[]',
    notes          TEXT NOT NULL DEFAULT '',
    discovered_at  INTEGER NOT NULL DEFAULT 0,
    last_used      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_models_provider ON models(provider);
CREATE INDEX IF NOT EXISTS idx_models_fav      ON models(is_favourite);
"""

# Curated known-good models used when live discovery fails
_KNOWN_MODELS: dict[str, list[dict]] = {
    "openai": [
        {"model_name":"gpt-4o",                 "params_b":0,  "context_window":128000, "capabilities":["chat","vision","tools","code"],            "input_price":5.0,   "output_price":15.0,  "tier_min":"lite"},
        {"model_name":"gpt-4o-mini",             "params_b":0,  "context_window":128000, "capabilities":["chat","tools","code"],                    "input_price":0.15,  "output_price":0.6,   "tier_min":"lite"},
        {"model_name":"o1-mini",                 "params_b":0,  "context_window":65536,  "capabilities":["chat","reasoning"],                        "input_price":3.0,   "output_price":12.0,  "tier_min":"lite"},
        {"model_name":"gpt-3.5-turbo",           "params_b":20, "context_window":16384,  "capabilities":["chat"],                                    "input_price":0.5,   "output_price":1.5,   "tier_min":"lite"},
    ],
    "anthropic": [
        {"model_name":"claude-opus-4-5",         "params_b":0,  "context_window":200000, "capabilities":["chat","vision","tools","code","long_context"], "input_price":15.0,  "output_price":75.0,  "tier_min":"lite"},
        {"model_name":"claude-sonnet-4-5",       "params_b":0,  "context_window":200000, "capabilities":["chat","vision","tools","code","long_context"], "input_price":3.0,   "output_price":15.0,  "tier_min":"lite"},
        {"model_name":"claude-3-5-haiku-20241022","params_b":0, "context_window":200000, "capabilities":["chat","tools","code"],                     "input_price":0.25,  "output_price":1.25,  "tier_min":"lite"},
    ],
    "groq": [
        {"model_name":"llama-3.1-70b-versatile", "params_b":70, "context_window":131072, "capabilities":["chat","tools"],    "input_price":0.59,  "output_price":0.79,  "tier_min":"lite"},
        {"model_name":"llama-3.1-8b-instant",    "params_b":8,  "context_window":131072, "capabilities":["chat"],            "input_price":0.05,  "output_price":0.08,  "tier_min":"lite"},
        {"model_name":"mixtral-8x7b-32768",      "params_b":47, "context_window":32768,  "capabilities":["chat"],            "input_price":0.24,  "output_price":0.24,  "tier_min":"lite"},
        {"model_name":"gemma2-9b-it",            "params_b":9,  "context_window":8192,   "capabilities":["chat"],            "input_price":0.20,  "output_price":0.20,  "tier_min":"lite"},
    ],
    "mistral": [
        {"model_name":"mistral-large-latest",    "params_b":0,  "context_window":131072, "capabilities":["chat","tools","code"],  "input_price":2.0,   "output_price":6.0,   "tier_min":"lite"},
        {"model_name":"mistral-small-latest",    "params_b":0,  "context_window":131072, "capabilities":["chat","tools"],        "input_price":0.1,   "output_price":0.3,   "tier_min":"lite"},
        {"model_name":"codestral-latest",        "params_b":0,  "context_window":32768,  "capabilities":["code","chat"],         "input_price":0.2,   "output_price":0.6,   "tier_min":"lite"},
    ],
    "together": [
        {"model_name":"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  "params_b":8,  "context_window":131072, "capabilities":["chat"],       "input_price":0.18, "output_price":0.18, "tier_min":"lite"},
        {"model_name":"meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "params_b":70, "context_window":131072, "capabilities":["chat"],       "input_price":0.88, "output_price":0.88, "tier_min":"lite"},
        {"model_name":"Qwen/Qwen2.5-Coder-32B-Instruct",              "params_b":32, "context_window":32768,  "capabilities":["code","chat"], "input_price":0.8,  "output_price":0.8,  "tier_min":"lite"},
    ],
    "deepseek": [
        {"model_name":"deepseek-chat",           "params_b":0,  "context_window":65536,  "capabilities":["chat","tools"],        "input_price":0.14, "output_price":0.28,  "tier_min":"lite"},
        {"model_name":"deepseek-reasoner",       "params_b":0,  "context_window":65536,  "capabilities":["chat","reasoning"],    "input_price":0.55, "output_price":2.19,  "tier_min":"lite"},
    ],
    "gemini": [
        {"model_name":"gemini-2.0-flash",        "params_b":0,  "context_window":1000000,"capabilities":["chat","vision","tools","long_context"], "input_price":0.0,   "output_price":0.0,   "tier_min":"lite"},
        {"model_name":"gemini-1.5-flash",        "params_b":0,  "context_window":1000000,"capabilities":["chat","vision","tools","long_context"], "input_price":0.075, "output_price":0.3,   "tier_min":"lite"},
        {"model_name":"gemini-1.5-pro",          "params_b":0,  "context_window":2000000,"capabilities":["chat","vision","tools","long_context"], "input_price":3.5,   "output_price":10.5,  "tier_min":"lite"},
    ],
}


class ModelCatalog:
    """
    SQLite-backed model catalog.

    Stores discovered + manually-pinned models from all providers.
    Supports favourites (user-curated shortlist), search, per-provider listing.
    """

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._lock = Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_CATALOG_DDL)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self._path), check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _to_entry(self, row: sqlite3.Row) -> ModelEntry:
        d = dict(row)
        return ModelEntry(
            id=d["id"], provider=d["provider"], model_name=d["model_name"],
            display_name=d["display_name"], params_b=d["params_b"],
            context_window=d["context_window"],
            capabilities=json.loads(d["capabilities"] or "[]"),
            input_price=d["input_price"], output_price=d["output_price"],
            is_favourite=bool(d["is_favourite"]), tier_min=d["tier_min"],
            tags=json.loads(d["tags"] or "[]"), notes=d["notes"],
        )

    def upsert(self, entry: ModelEntry) -> ModelEntry:
        now = int(time.time())
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO models
                       (id,provider,model_name,display_name,params_b,context_window,
                        capabilities,input_price,output_price,is_favourite,tier_min,tags,notes,discovered_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET
                         display_name=excluded.display_name, params_b=excluded.params_b,
                         context_window=excluded.context_window,
                         capabilities=excluded.capabilities,
                         input_price=excluded.input_price, output_price=excluded.output_price,
                         tier_min=excluded.tier_min, tags=excluded.tags, notes=excluded.notes""",
                    (entry.id, entry.provider, entry.model_name, entry.display_name,
                     entry.params_b, entry.context_window,
                     json.dumps(entry.capabilities), entry.input_price, entry.output_price,
                     int(entry.is_favourite), entry.tier_min,
                     json.dumps(entry.tags), entry.notes, now),
                )
                conn.commit()
        return entry

    def get(self, model_id: str) -> ModelEntry | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
        return self._to_entry(row) if row else None

    def list_by_provider(self, provider: str) -> list[ModelEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM models WHERE provider=? ORDER BY is_favourite DESC, params_b DESC",
                (provider,)
            ).fetchall()
        return [self._to_entry(r) for r in rows]

    def list_favourites(self) -> list[ModelEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM models WHERE is_favourite=1 ORDER BY provider, model_name"
            ).fetchall()
        return [self._to_entry(r) for r in rows]

    def set_favourite(self, model_id: str, is_fav: bool = True) -> bool:
        with self._lock:
            with self._connect() as conn:
                n = conn.execute(
                    "UPDATE models SET is_favourite=? WHERE id=?",
                    (int(is_fav), model_id)
                ).rowcount
                conn.commit()
        return n > 0

    def search(
        self,
        query:      str  = "",
        provider:   str  = "",
        capability: str  = "",
        tier_max:   str  = "",
        favs_only:  bool = False,
        limit:      int  = 50,
    ) -> list[ModelEntry]:
        clauses, params = [], []
        if query:
            clauses.append("(model_name LIKE ? OR display_name LIKE ? OR tags LIKE ?)")
            q = f"%{query}%"; params += [q, q, q]
        if provider:
            clauses.append("provider=?"); params.append(provider)
        if capability:
            clauses.append("capabilities LIKE ?"); params.append(f"%{capability}%")
        if favs_only:
            clauses.append("is_favourite=1")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM models {where} ORDER BY is_favourite DESC, params_b DESC LIMIT ?",
                params + [limit],
            ).fetchall()
        entries = [self._to_entry(r) for r in rows]
        if tier_max:
            _order = ["lite", "standard", "mem-opt", "gpu-acc"]
            max_idx = _order.index(tier_max) if tier_max in _order else 3
            entries = [e for e in entries if _order.index(e.tier_min) <= max_idx]
        return entries

    def mark_used(self, model_id: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute("UPDATE models SET last_used=? WHERE id=?", (int(time.time()), model_id))
                conn.commit()

    def delete(self, model_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                n = conn.execute("DELETE FROM models WHERE id=?", (model_id,)).rowcount
                conn.commit()
        return n > 0

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]

    def count_favourites(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM models WHERE is_favourite=1").fetchone()[0]


# ---------------------------------------------------------------------------
# Provider discovery — live model listing per provider
# ---------------------------------------------------------------------------

class ProviderDiscovery:
    """
    Queries each provider's model-list endpoint and returns ModelEntry objects.
    Falls back to the curated _KNOWN_MODELS table when the live API fails.
    """

    def _from_known(self, provider_id: str) -> list[ModelEntry]:
        return [
            ModelEntry(
                id=f"{provider_id}:{m['model_name']}",
                provider=provider_id,
                display_name=m["model_name"],
                **{k: v for k, v in m.items()},
            )
            for m in _KNOWN_MODELS.get(provider_id, [])
        ]

    async def discover_openai_compatible(self, cfg: ProviderConfig) -> list[ModelEntry]:
        """Works for OpenAI, Groq, Together, DeepSeek, Mistral, OpenRouter, LM Studio."""
        headers = ({"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {})
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(f"{cfg.base_url}/models", headers=headers)
                if r.status_code == 200:
                    data  = r.json()
                    items = data.get("data", data.get("models", []))
                    entries = []
                    for m in items:
                        mid = m.get("id", m.get("name", ""))
                        if not mid:
                            continue
                        entries.append(ModelEntry(
                            id=f"{cfg.id}:{mid}",
                            provider=cfg.id,
                            model_name=mid,
                            display_name=mid,
                            capabilities=["chat"],
                        ))
                    if entries:
                        return entries
        except Exception as exc:
            log.debug("%s discovery failed: %s", cfg.id, exc)
        return self._from_known(cfg.id)

    async def discover_anthropic(self, cfg: ProviderConfig) -> list[ModelEntry]:
        if cfg.api_key:
            try:
                async with httpx.AsyncClient(timeout=10.0) as c:
                    r = await c.get(
                        "https://api.anthropic.com/v1/models",
                        headers={"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01"},
                    )
                    if r.status_code == 200:
                        items = r.json().get("data", [])
                        if items:
                            return [ModelEntry(
                                id=f"anthropic:{m['id']}",
                                provider="anthropic",
                                model_name=m["id"],
                                display_name=m.get("display_name", m["id"]),
                                capabilities=["chat", "vision", "tools", "code"],
                                tier_min="lite",
                            ) for m in items]
            except Exception as exc:
                log.debug("Anthropic discovery failed: %s", exc)
        return self._from_known("anthropic")

    async def discover_ollama(self, cfg: ProviderConfig) -> list[ModelEntry]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{cfg.base_url}/api/tags")
                if r.status_code == 200:
                    return [ModelEntry(
                        id=f"ollama:{m['name']}",
                        provider="ollama",
                        model_name=m["name"],
                        display_name=m["name"],
                        capabilities=["chat"],
                        tier_min="lite",
                    ) for m in r.json().get("models", [])]
        except Exception:
            pass
        return []

    async def discover(self, cfg: ProviderConfig) -> list[ModelEntry]:
        if not cfg.enabled:
            return []
        if cfg.id == "ollama":
            return await self.discover_ollama(cfg)
        if cfg.id == "anthropic":
            return await self.discover_anthropic(cfg)
        if cfg.id == "hf_local":
            return []   # scanned separately via ModelRouter.scan_local_models()
        return await self.discover_openai_compatible(cfg)

    async def discover_all(
        self,
        registry: ProviderRegistry,
        catalog:  ModelCatalog,
    ) -> int:
        """
        Discover models from all enabled providers and upsert into catalog.
        Returns total number of models stored.
        """
        total = 0
        tasks = [self.discover(cfg) for cfg in registry.list_enabled()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for entries in results:
            if isinstance(entries, Exception):
                log.warning("Discovery error: %s", entries)
                continue
            for entry in entries:
                catalog.upsert(entry)
                total += 1
        log.info("ProviderDiscovery: %d models indexed across all providers", total)
        return total


# ---------------------------------------------------------------------------
# SmartModelSelector — unified ranked selection across all providers
# ---------------------------------------------------------------------------

_TASK_CAPS: dict[str, list[str]] = {
    "chat":     ["chat"],
    "code":     ["code", "chat"],
    "vision":   ["vision", "chat"],
    "tools":    ["tools", "function_calling", "chat"],
    "research": ["chat", "long_context"],
    "fast":     ["chat"],
    "embed":    ["embedding"],
    "reason":   ["reasoning", "chat"],
    "default":  ["chat"],
}


class SmartModelSelector:
    """
    Selects the best available model across all providers.

    Selection priority
    ------------------
      1. Favourites (user-curated shortlist)
      2. Device-compatible (params fit within device tier)
      3. Task capability match
      4. Provider priority order (lower number = preferred)
      5. Free / local preferred over paid cloud
    """

    def __init__(
        self,
        catalog:    ModelCatalog,
        providers:  ProviderRegistry,
        device:     DeviceProfile | None = None,
    ) -> None:
        self._catalog   = catalog
        self._providers = providers
        self._device    = device or DeviceProfile.detect()

    def select(
        self,
        task_type:       str        = "default",
        prefer_favourite: bool      = True,
        prefer_local:    bool       = True,
        context_needed:  int        = 0,
        allow_providers: list | None = None,
    ) -> tuple[str, str] | None:
        """
        Return (provider_id, model_name) or None if nothing available.
        """
        enabled = {p.id: p for p in self._providers.list_enabled()}
        if allow_providers:
            enabled = {k: v for k, v in enabled.items() if k in allow_providers}

        _order = ["lite", "standard", "mem-opt", "gpu-acc"]
        tier_idx = _order.index(self._device.tier) if self._device.tier in _order else 1
        cap_wanted = _TASK_CAPS.get(task_type, ["chat"])

        all_models = self._catalog.search(limit=200)
        scored: list[tuple[float, ModelEntry]] = []

        for m in all_models:
            if m.provider not in enabled:
                continue
            m_tier_idx = _order.index(m.tier_min) if m.tier_min in _order else 0
            if m_tier_idx > tier_idx:
                continue   # too large for device
            if context_needed and m.context_window < context_needed:
                continue

            score = 0.0
            if m.is_favourite and prefer_favourite:
                score += 200.0
            if prefer_local and m.provider in ("ollama", "hf_local"):
                score += 80.0
            prov = enabled.get(m.provider)
            if prov:
                score += max(0, 100 - prov.priority)
            for cap in cap_wanted:
                if cap in m.capabilities:
                    score += 25.0
            score += min(m.params_b, 70) * 0.5
            if m.input_price == 0.0:
                score += 15.0   # free/local bonus

            scored.append((score, m))

        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        self._catalog.mark_used(best.id)
        return best.provider, best.model_name

    def ranked_candidates(self, task_type: str = "default", n: int = 10) -> list[ModelEntry]:
        enabled = {p.id for p in self._providers.list_enabled()}
        return [m for m in self._catalog.search(limit=n * 3) if m.provider in enabled][:n]


# ---------------------------------------------------------------------------
# First-run GGUF bootstrap
# ---------------------------------------------------------------------------

FIRST_RUN_GGUF: dict[str, list[dict]] = {
    "lite": [
        {
            "name": "Gemma-2-2B-Instruct (Q4_K_M)",
            "repo": "bartowski/gemma-2-2b-it-GGUF",
            "file": "gemma-2-2b-it-Q4_K_M.gguf",
            "params": 2, "size_gb": 1.5,
            "desc": "Google Gemma 2 2B — best-in-class 2B model, fast on CPU",
        },
        {
            "name": "Phi-3.5-Mini-Instruct (Q4_K_M)",
            "repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
            "file": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
            "params": 4, "size_gb": 2.4,
            "desc": "Microsoft Phi-3.5 Mini — strong reasoning, great for low RAM",
        },
    ],
    "standard": [
        {
            "name": "Llama-3.2-3B-Instruct (Q4_K_M)",
            "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "params": 3, "size_gb": 1.9,
            "desc": "Meta Llama 3.2 3B — Meta's latest small model",
        },
        {
            "name": "Mistral-7B-Instruct-v0.3 (Q4_K_M)",
            "repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
            "file": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
            "params": 7, "size_gb": 4.1,
            "desc": "Mistral 7B v0.3 — proven, excellent all-round instruction model",
        },
    ],
    "mem-opt": [
        {
            "name": "Llama-3.1-8B-Instruct (Q4_K_M)",
            "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "params": 8, "size_gb": 4.9,
            "desc": "Meta Llama 3.1 8B — 128K context, excellent quality",
        },
        {
            "name": "Qwen2.5-14B-Instruct (Q4_K_M)",
            "repo": "bartowski/Qwen2.5-14B-Instruct-GGUF",
            "file": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
            "params": 14, "size_gb": 8.9,
            "desc": "Alibaba Qwen 2.5 14B — multilingual, strong at coding",
        },
    ],
    "gpu-acc": [
        {
            "name": "Qwen2.5-32B-Instruct (Q4_K_M)",
            "repo": "bartowski/Qwen2.5-32B-Instruct-GGUF",
            "file": "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
            "params": 32, "size_gb": 19.8,
            "desc": "Alibaba Qwen 2.5 32B — top multilingual + coding model",
        },
        {
            "name": "Llama-3.1-70B-Instruct (Q4_K_M)",
            "repo": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
            "file": "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
            "params": 70, "size_gb": 40.0,
            "desc": "Meta Llama 3.1 70B — near GPT-4 class, requires GPU",
        },
    ],
}


class FirstRunBootstrap:
    """
    Guide the user through model setup on first run:
      1. Detect device tier
      2. Present curated GGUF options for that tier
      3. Ask user to pick one (or skip to use Ollama / cloud)
      4. Download via ModelRouter.pull_from_hf()
    """

    def __init__(self, workspace: Path, router: "ModelRouter") -> None:
        self._workspace = workspace
        self._router    = router

    async def is_needed(self) -> bool:
        """True if no models are available from any source."""
        if list((self._workspace / "models").glob("*.gguf")) if (self._workspace / "models").exists() else []:
            return False
        return len(await self._router.get_ollama_models()) == 0

    async def run_interactive(self, device: DeviceProfile) -> dict | None:
        """
        Show tier-appropriate model menu and download the user's choice.
        Returns the pulled model info dict, or None if skipped.
        """
        candidates = FIRST_RUN_GGUF.get(device.tier, FIRST_RUN_GGUF["standard"])

        print("\n  First Run — Model Setup")
        print(f"  {'='*60}")
        print(f"  No AI models detected.  Essence needs a base model to run.")
        print(f"  Device: {device.summary()}")
        print(f"\n  Recommended models for your device:\n")
        for i, m in enumerate(candidates, 1):
            print(f"  [{i}] {m['name']:<44} ~{m['size_gb']:.1f} GB")
            print(f"       {m['desc']}")
        print(f"\n  [0] Skip — I will configure Ollama or a cloud provider manually")
        print()

        try:
            raw = input("  Download model [1]: ").strip() or "1"
            idx = int(raw)
            if idx == 0:
                print("  Skipping. Run: python essence.py setup  to configure a provider.")
                return None
            if 1 <= idx <= len(candidates):
                chosen = candidates[idx - 1]
                print(f"\n  Downloading {chosen['name']}  (~{chosen['size_gb']:.1f} GB)...")
                path = await self._router.pull_from_hf(chosen["repo"], chosen["file"])
                if path:
                    print(f"  Ready: {path}\n")
                    return {**chosen, "local_path": str(path)}
                else:
                    print("  Download failed. Check your connection.")
                    return None
        except (ValueError, KeyboardInterrupt, EOFError):
            return None
        return None
