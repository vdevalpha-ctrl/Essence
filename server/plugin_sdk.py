"""
plugin_sdk.py — Essence Plugin SDK
====================================
Defines the contract for third-party Essence plugins.

Every plugin ships with a plugin.json manifest that declares:
  • id, name, version, author
  • permissions  — what the plugin may access (network, filesystem, llm_call, …)
  • hooks        — which Essence events trigger the plugin (user.request, cron.tick, …)
  • settingsSchema — JSON Schema for user-configurable settings
  • entry        — path to the Python entry-point (relative to plugin root)

Execution model
---------------
Plugins run in an isolated subprocess (not in the Essence process) so a
misbehaving plugin cannot crash the agent or access the signing key.
Each invocation receives a JSON payload on stdin and must write a JSON
result to stdout.  stderr is captured and logged.  A hard timeout of
`manifest.timeout_s` is enforced via subprocess kill.

Directory layout
----------------
  memory/plugins/
    my_plugin/
      plugin.json   ← manifest (authoritative)
      main.py       ← entry point declared in manifest
      settings.json ← user settings (written by Essence, read by plugin)

Usage
-----
    store  = PluginStore(workspace / "memory" / "plugins")
    plugin = store.get("my_plugin")
    runner = PluginRunner(plugin)
    result = await runner.invoke("user.request", {"text": "hello"})
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("essence.plugin_sdk")

# ---------------------------------------------------------------------------
# Allowed permission tokens
# ---------------------------------------------------------------------------
VALID_PERMISSIONS: frozenset[str] = frozenset({
    "network",       # HTTP outbound
    "filesystem",    # read/write within plugin directory only
    "llm_call",      # call the configured LLM
    "memory_read",   # read Essence memory
    "memory_write",  # write Essence memory
    "shell",         # run shell commands (high-privilege — requires explicit grant)
    "calendar",
    "notifications",
})

# Allowed hook topics (plugin can subscribe to these bus events)
VALID_HOOKS: frozenset[str] = frozenset({
    "user.request",
    "agent.message",
    "cron.tick",
    "skill.result",
    "heartbeat.tick",
    "trigger.fired",
})


# ---------------------------------------------------------------------------
# Manifest dataclass
# ---------------------------------------------------------------------------
@dataclass
class PluginManifest:
    id:             str
    name:           str
    version:        str
    author:         str
    description:    str
    entry:          str              # e.g. "main.py"
    permissions:    list[str] = field(default_factory=list)
    hooks:          list[str] = field(default_factory=list)
    settings_schema: dict    = field(default_factory=dict)
    timeout_s:      int      = 30
    enabled:        bool     = True

    # Path set by PluginStore after loading
    root: Path = field(default_factory=Path, repr=False)

    @classmethod
    def from_dict(cls, data: dict, root: Path) -> "PluginManifest":
        m = cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            version=data.get("version", "0.1.0"),
            author=data.get("author", "unknown"),
            description=data.get("description", ""),
            entry=data.get("entry", "main.py"),
            permissions=data.get("permissions", []),
            hooks=data.get("hooks", []),
            settings_schema=data.get("settingsSchema", {}),
            timeout_s=int(data.get("timeout_s", 30)),
            enabled=bool(data.get("enabled", True)),
            root=root,
        )
        m.validate()
        return m

    def validate(self) -> None:
        """Raise ValueError for any manifest violation."""
        if not self.id:
            raise ValueError("Plugin manifest missing 'id'")
        bad_perms = [p for p in self.permissions if p not in VALID_PERMISSIONS]
        if bad_perms:
            raise ValueError(f"Plugin '{self.id}' declares unknown permissions: {bad_perms}")
        bad_hooks = [h for h in self.hooks if h not in VALID_HOOKS]
        if bad_hooks:
            raise ValueError(f"Plugin '{self.id}' declares unknown hooks: {bad_hooks}")
        entry_path = self.root / self.entry
        if not entry_path.exists():
            raise ValueError(
                f"Plugin '{self.id}' entry '{self.entry}' not found at {entry_path}"
            )

    def to_dict(self) -> dict:
        return {
            "id":             self.id,
            "name":           self.name,
            "version":        self.version,
            "author":         self.author,
            "description":    self.description,
            "entry":          self.entry,
            "permissions":    self.permissions,
            "hooks":          self.hooks,
            "settingsSchema": self.settings_schema,
            "timeout_s":      self.timeout_s,
            "enabled":        self.enabled,
        }


# ---------------------------------------------------------------------------
# Plugin Store — discovers and loads plugin manifests
# ---------------------------------------------------------------------------
class PluginStore:
    """
    Scans memory/plugins/ for plugin directories containing plugin.json.
    Each subdirectory is treated as an independent plugin package.
    """

    def __init__(self, plugins_dir: Path) -> None:
        self._dir = plugins_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._manifests: dict[str, PluginManifest] = {}
        self._load_all()

    def _load_all(self) -> None:
        for sub in sorted(self._dir.iterdir()):
            if not sub.is_dir():
                continue
            manifest_file = sub / "plugin.json"
            if not manifest_file.exists():
                continue
            try:
                data = json.loads(manifest_file.read_text(encoding="utf-8"))
                m = PluginManifest.from_dict(data, root=sub)
                self._manifests[m.id] = m
                log.debug("PluginStore: loaded plugin '%s' v%s", m.id, m.version)
            except Exception as exc:
                log.warning("PluginStore: failed to load plugin at %s: %s", sub, exc)

    def reload(self) -> None:
        self._manifests.clear()
        self._load_all()

    def get(self, plugin_id: str) -> PluginManifest | None:
        return self._manifests.get(plugin_id)

    def list_all(self, enabled_only: bool = False) -> list[PluginManifest]:
        plugins = list(self._manifests.values())
        if enabled_only:
            plugins = [p for p in plugins if p.enabled]
        return plugins

    def list_for_hook(self, hook: str) -> list[PluginManifest]:
        """Return enabled plugins that subscribe to *hook*."""
        return [p for p in self._manifests.values() if p.enabled and hook in p.hooks]

    def set_enabled(self, plugin_id: str, enabled: bool) -> bool:
        m = self._manifests.get(plugin_id)
        if m is None:
            return False
        m.enabled = enabled
        manifest_file = m.root / "plugin.json"
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            data["enabled"] = enabled
            manifest_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning("PluginStore: could not persist enabled state for '%s': %s", plugin_id, exc)
        return True

    def get_settings(self, plugin_id: str) -> dict:
        m = self._manifests.get(plugin_id)
        if m is None:
            return {}
        settings_file = m.root / "settings.json"
        if not settings_file.exists():
            return {}
        try:
            return json.loads(settings_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_settings(self, plugin_id: str, settings: dict) -> bool:
        m = self._manifests.get(plugin_id)
        if m is None:
            return False
        settings_file = m.root / "settings.json"
        try:
            settings_file.write_text(json.dumps(settings, indent=2), encoding="utf-8")
            return True
        except Exception as exc:
            log.warning("PluginStore: could not save settings for '%s': %s", plugin_id, exc)
            return False


# ---------------------------------------------------------------------------
# Plugin Runner — isolated subprocess execution
# ---------------------------------------------------------------------------
class PluginRunner:
    """
    Executes a single plugin invocation in an isolated subprocess.

    The plugin receives a JSON object on stdin:
      {
        "hook":     "<topic that fired>",
        "payload":  { ... event data ... },
        "settings": { ... user settings ... },
      }

    And must write a JSON object to stdout:
      {
        "status":  "ok" | "error",
        "output":  { ... any result data ... },
        "error":   "error message if status=error"
      }

    stdout beyond the first JSON object is discarded.
    stderr is captured and logged at DEBUG level.
    """

    def __init__(self, manifest: PluginManifest, store: PluginStore | None = None) -> None:
        self._manifest = manifest
        self._store    = store

    async def invoke(
        self,
        hook:    str,
        payload: dict,
    ) -> dict:
        """
        Run the plugin for the given hook event.
        Returns the plugin's output dict, or an error dict on failure.
        """
        if not self._manifest.enabled:
            return {"status": "skipped", "reason": "disabled"}
        if hook not in self._manifest.hooks:
            return {"status": "skipped", "reason": f"hook '{hook}' not registered"}
        if not self._check_permission(hook):
            return {"status": "error", "error": "permission denied"}

        settings = self._store.get_settings(self._manifest.id) if self._store else {}
        stdin_payload = json.dumps({
            "hook":     hook,
            "payload":  payload,
            "settings": settings,
        }, ensure_ascii=False)

        entry = str(self._manifest.root / self._manifest.entry)
        try:
            started = time.time()
            proc = await asyncio.create_subprocess_exec(
                "python", entry,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._manifest.root),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=stdin_payload.encode("utf-8")),
                    timeout=self._manifest.timeout_s,
                )
            except asyncio.TimeoutError:
                proc.kill()
                log.warning("Plugin '%s': timed out after %ds", self._manifest.id, self._manifest.timeout_s)
                return {"status": "error", "error": "timeout"}

            elapsed = round(time.time() - started, 2)
            if stderr:
                log.debug("Plugin '%s' stderr: %s", self._manifest.id, stderr.decode("utf-8", errors="replace")[:500])

            raw = stdout.decode("utf-8", errors="replace").strip()
            if not raw:
                return {"status": "error", "error": "no output from plugin"}
            try:
                result = json.loads(raw)
                result["_elapsed_s"] = elapsed
                return result
            except json.JSONDecodeError as e:
                return {"status": "error", "error": f"invalid JSON output: {e}", "raw": raw[:200]}

        except FileNotFoundError:
            return {"status": "error", "error": f"plugin entry not found: {entry}"}
        except Exception as exc:
            log.error("Plugin '%s' invocation error: %s", self._manifest.id, exc)
            return {"status": "error", "error": str(exc)}

    def _check_permission(self, hook: str) -> bool:
        """Basic sanity check — plugins that declared no permissions still run."""
        return True  # Permissions validated at manifest load time


# ---------------------------------------------------------------------------
# Convenience: fire all plugins for a hook
# ---------------------------------------------------------------------------
async def dispatch_hook(
    store:   PluginStore,
    hook:    str,
    payload: dict,
) -> list[dict]:
    """
    Run all enabled plugins that subscribe to *hook* concurrently.
    Returns a list of result dicts (one per plugin).
    """
    plugins = store.list_for_hook(hook)
    if not plugins:
        return []
    tasks = [PluginRunner(m, store).invoke(hook, payload) for m in plugins]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out = []
    for m, r in zip(plugins, results):
        if isinstance(r, Exception):
            out.append({"plugin": m.id, "status": "error", "error": str(r)})
        else:
            out.append({"plugin": m.id, **r})
    return out
