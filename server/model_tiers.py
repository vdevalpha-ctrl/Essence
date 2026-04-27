"""
model_tiers.py — Three-Tier Model Configuration
=================================================
Inspired by SwarmForge's per-role backend assignment and agents-cli's
local→cloud deployment gradient.

Tier hierarchy
--------------
  PRIMARY   Best available model for normal operation.
            Can be cloud-hosted (Groq, OpenAI, Anthropic, etc.) or local.
            Configured by the user; dynamic router may route AROUND it when
            the task type is better served by a specialist.

  FALLBACK  Used when primary is rate-limited, erroring, or unavailable.
            Should always be a reliable, lower-cost option.
            Defaults to the local Ollama model (always present if Ollama runs).

  EMERGENCY Last resort — must always be reachable and produce SOME response.
            Defaults to the lightest local model available.
            Never skipped; if this fails the system emits an error token.

Per-task-type overrides
-----------------------
  You can override the PRIMARY for specific task types, e.g.:
    code     → deepseek/deepseek-coder (better at code)
    research → groq/llama-3.3-70b      (fast + large context)
    vision   → openai/gpt-4o           (multimodal)

  Fallback and emergency stay the same regardless of override.

Configuration
-------------
  config.toml (preferred):

    [models.primary]
    provider = "groq"
    model    = "llama-3.3-70b-versatile"

    [models.fallback]
    provider = "ollama"
    model    = "qwen3:4b"

    [models.emergency]
    provider = "ollama"
    model    = "tinyllama"

    [models.overrides.code]
    provider = "deepseek"
    model    = "deepseek-coder"

    [models.overrides.vision]
    provider = "openai"
    model    = "gpt-4o"

  Environment variables (override config.toml):
    ESSENCE_PRIMARY_PROVIDER / ESSENCE_PRIMARY_MODEL
    ESSENCE_FALLBACK_PROVIDER / ESSENCE_FALLBACK_MODEL
    ESSENCE_EMERGENCY_PROVIDER / ESSENCE_EMERGENCY_MODEL

  Runtime API (from TUI):
    /tiers set primary groq llama-3.3-70b-versatile
    /tiers set fallback ollama qwen3:4b
    /tiers set emergency ollama tinyllama
    /tiers override code deepseek deepseek-coder
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("essence.model_tiers")

# ---------------------------------------------------------------------------
# Tier dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelTier:
    name:        str    # "primary" | "fallback" | "emergency"
    provider:    str
    model:       str
    description: str = ""

    def as_tuple(self) -> tuple[str, str]:
        return (self.provider, self.model)

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


# ---------------------------------------------------------------------------
# Tier registry
# ---------------------------------------------------------------------------

# Sensible built-in defaults (all pointing to local Ollama so the system
# works out-of-the-box without any API keys)
_DEFAULT_TIERS: dict[str, dict] = {
    "primary": {
        "provider":    "ollama",
        "model":       "qwen3:4b",
        "description": "Best model for normal operation",
    },
    "fallback": {
        "provider":    "ollama",
        "model":       "qwen3:1.7b",
        "description": "Used when primary is unavailable or rate-limited",
    },
    "emergency": {
        "provider":    "ollama",
        "model":       "tinyllama",
        "description": "Last resort — must always produce a response",
    },
}

# Per-task-type provider→model that should override PRIMARY for that task
_DEFAULT_OVERRIDES: dict[str, dict] = {
    # code tasks benefit from a code-specialist model when available
    "code":     {"provider": "deepseek", "model": "deepseek-coder"},
    # research tasks need large context + fast responses
    "research": {"provider": "groq",     "model": "llama-3.3-70b-versatile"},
    # vision tasks need multimodal capability
    "vision":   {"provider": "openai",   "model": "gpt-4o-mini"},
    # complex analysis benefits from best reasoning model
    "analysis": {"provider": "anthropic","model": "claude-3-5-sonnet-20241022"},
}


class ModelTierRegistry:
    """
    Manages primary/fallback/emergency model configuration and
    per-task-type overrides.

    Thread-safe reads; writes are infrequent (config changes).
    """

    def __init__(self, workspace: Path | None = None) -> None:
        self._workspace = workspace
        self._tiers: dict[str, ModelTier] = {}
        self._overrides: dict[str, ModelTier] = {}   # task_type → ModelTier
        self._tier_history: list[dict] = []           # [{ts, from, to, reason}]
        self._load()

    # ── Public API ────────────────────────────────────────────────

    @property
    def primary(self) -> ModelTier:
        return self._tiers["primary"]

    @property
    def fallback(self) -> ModelTier:
        return self._tiers["fallback"]

    @property
    def emergency(self) -> ModelTier:
        return self._tiers["emergency"]

    def get_override(self, task_type: str) -> ModelTier | None:
        """Return a task-specific primary override, or None."""
        return self._overrides.get(task_type)

    def set_tier(self, name: str, provider: str, model: str) -> None:
        """Set a tier at runtime (also persists to config.toml)."""
        if name not in ("primary", "fallback", "emergency"):
            raise ValueError(f"Unknown tier: {name!r}")
        old = str(self._tiers[name])
        self._tiers[name] = ModelTier(
            name=name, provider=provider, model=model,
            description=self._tiers[name].description,
        )
        self._tier_history.append({
            "ts":     time.time(),
            "tier":   name,
            "from":   old,
            "to":     f"{provider}/{model}",
            "source": "runtime",
        })
        log.info("ModelTiers: %s → %s/%s", name, provider, model)
        self._persist_tier(name, provider, model)

    def set_override(self, task_type: str, provider: str, model: str) -> None:
        """Add or update a per-task-type primary override."""
        self._overrides[task_type] = ModelTier(
            name=f"override.{task_type}", provider=provider, model=model,
        )
        log.info("ModelTiers: override[%s] → %s/%s", task_type, provider, model)
        self._persist_override(task_type, provider, model)

    def remove_override(self, task_type: str) -> bool:
        if task_type in self._overrides:
            del self._overrides[task_type]
            return True
        return False

    def build_chain(
        self,
        task_type: str = "",
        exclude_providers: set | None = None,
        dynamic_suggestions: list[tuple[str, str]] | None = None,
    ) -> list[tuple[str, str]]:
        """
        Build the full ordered fallback chain for a request.

        Chain construction logic (SwarmForge-inspired role-priority ordering):

          1. Task-type override (if defined and not excluded)
          2. Dynamic router suggestions (capability-matched, scored)
          3. PRIMARY tier
          4. FALLBACK tier
          5. EMERGENCY tier  ← always last, always included

        Duplicates are removed while preserving order.
        """
        exclude = exclude_providers or set()
        chain: list[tuple[str, str]] = []
        seen:  set[str] = set()   # "provider/model" dedup keys

        def _add(provider: str, model: str) -> None:
            key = f"{provider}/{model}"
            if key not in seen and provider not in exclude:
                seen.add(key)
                chain.append((provider, model))

        # 1. Task-type override
        ov = self.get_override(task_type) if task_type else None
        if ov:
            _add(ov.provider, ov.model)

        # 2. Dynamic router suggestions (already scored/ordered)
        for p, m in (dynamic_suggestions or []):
            _add(p, m)

        # 3-5. Tier order
        for tier in (self._tiers["primary"], self._tiers["fallback"], self._tiers["emergency"]):
            _add(tier.provider, tier.model)

        # Emergency is always last and always present (re-add if excluded)
        em = self._tiers["emergency"]
        em_key = f"{em.provider}/{em.model}"
        if em_key not in seen:
            chain.append((em.provider, em.model))

        return chain

    def status(self) -> dict:
        return {
            "primary":   str(self.primary),
            "fallback":  str(self.fallback),
            "emergency": str(self.emergency),
            "overrides": {k: str(v) for k, v in self._overrides.items()},
            "history":   self._tier_history[-10:],
        }

    # ── Loading ───────────────────────────────────────────────────

    def _load(self) -> None:
        """Load tiers from env, config.toml, then built-in defaults (priority order)."""
        cfg = self._read_config()

        for tier_name, defaults in _DEFAULT_TIERS.items():
            provider = (
                os.environ.get(f"ESSENCE_{tier_name.upper()}_PROVIDER")
                or cfg.get(tier_name, {}).get("provider")
                or defaults["provider"]
            )
            model = (
                os.environ.get(f"ESSENCE_{tier_name.upper()}_MODEL")
                or cfg.get(tier_name, {}).get("model")
                or defaults["model"]
            )
            self._tiers[tier_name] = ModelTier(
                name=tier_name, provider=provider, model=model,
                description=defaults["description"],
            )
            log.debug("ModelTiers: loaded %s = %s/%s", tier_name, provider, model)

        # Load per-task overrides from config
        raw_overrides = cfg.get("overrides", {})
        for task_type, ovr in raw_overrides.items():
            p = ovr.get("provider", "")
            m = ovr.get("model", "")
            if p and m:
                self._overrides[task_type] = ModelTier(
                    name=f"override.{task_type}", provider=p, model=m,
                )

        # Merge in default overrides only for tasks NOT already in config
        for task_type, ovr in _DEFAULT_OVERRIDES.items():
            if task_type not in self._overrides:
                # Default overrides are SUGGESTIONS — only activate if the
                # provider actually has a key.  We don't hard-set them; the
                # dynamic router will still try them in its scored chain.
                pass   # don't auto-add, let dynamic router handle it

    def _read_config(self) -> dict:
        """Read [models] section from config.toml."""
        if not self._workspace:
            return {}
        try:
            import tomllib
            p = self._workspace / "config.toml"
            if p.exists():
                with open(p, "rb") as f:
                    full = tomllib.load(f)
                return full.get("models", {})
        except Exception:
            pass
        return {}

    def _persist_tier(self, name: str, provider: str, model: str) -> None:
        if not self._workspace:
            return
        try:
            p = self._workspace / "config.toml"
            text = p.read_text(encoding="utf-8") if p.exists() else ""
            section = f"[models.{name}]"
            new_lines = [f'provider = "{provider}"', f'model    = "{model}"']
            if section in text:
                lines, in_sec, written = [], False, False
                for line in text.splitlines():
                    if line.strip() == section:
                        in_sec = True
                        lines.append(line)
                        continue
                    if in_sec and line.strip().startswith("["):
                        if not written:
                            lines.extend(new_lines)
                            written = True
                        in_sec = False
                    if not in_sec or not (
                        line.lstrip().startswith("provider") or
                        line.lstrip().startswith("model")
                    ):
                        lines.append(line)
                if in_sec and not written:
                    lines.extend(new_lines)
                p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            else:
                p.write_text(
                    text.rstrip() + f"\n\n{section}\n" + "\n".join(new_lines) + "\n",
                    encoding="utf-8",
                )
        except Exception as e:
            log.debug("ModelTiers: persist failed: %s", e)

    def _persist_override(self, task_type: str, provider: str, model: str) -> None:
        if not self._workspace:
            return
        try:
            p = self._workspace / "config.toml"
            text = p.read_text(encoding="utf-8") if p.exists() else ""
            section = f"[models.overrides.{task_type}]"
            new_lines = [f'provider = "{provider}"', f'model    = "{model}"']
            if section in text:
                lines, in_sec, written = [], False, False
                for line in text.splitlines():
                    if line.strip() == section:
                        in_sec = True
                        lines.append(line)
                        continue
                    if in_sec and line.strip().startswith("["):
                        if not written:
                            lines.extend(new_lines)
                            written = True
                        in_sec = False
                    if not in_sec or not (
                        line.lstrip().startswith("provider") or
                        line.lstrip().startswith("model")
                    ):
                        lines.append(line)
                if in_sec and not written:
                    lines.extend(new_lines)
                p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            else:
                p.write_text(
                    text.rstrip() + f"\n\n{section}\n" + "\n".join(new_lines) + "\n",
                    encoding="utf-8",
                )
        except Exception as e:
            log.debug("ModelTiers: persist override failed: %s", e)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_registry: ModelTierRegistry | None = None


def get_tier_registry(workspace: Path | None = None) -> ModelTierRegistry:
    global _registry
    if _registry is None:
        _registry = ModelTierRegistry(workspace)
    return _registry


def init_tier_registry(workspace: Path) -> ModelTierRegistry:
    global _registry
    _registry = ModelTierRegistry(workspace)
    return _registry
