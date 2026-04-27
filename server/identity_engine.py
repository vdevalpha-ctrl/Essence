"""
identity_engine.py — Essence Identity Engine
Manages persona, user profile, and voice settings for the Essence personal AI agent.
"""

from __future__ import annotations

import logging
import tomllib
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("essence.identity")

# ---------------------------------------------------------------------------
# Default identity structure
# ---------------------------------------------------------------------------

DEFAULT_IDENTITY: dict[str, Any] = {
    "persona": {
        "name": "Essence",
        "tone": "concise",
        "verbosity": "medium",
        "expertise": ["productivity", "engineering"],
        "language": "en",
        "schema_ver": "1.0",
    },
    "user": {
        "name": "",
        "role": "",
        "timezone": "UTC",
        "preferences": [],
    },
    "voice": {
        "style": "peer",
        "formality": 0.4,
    },
    "history": {
        "last_updated": "",
    },
}

IDENTITY_FILENAME = "identity.toml"
HISTORY_DIR = "identity.history"


# ---------------------------------------------------------------------------
# TOML serialiser (stdlib only — no tomli-w dependency)
# ---------------------------------------------------------------------------

def _toml_value(val: Any) -> str:
    """Serialise a single Python value to its TOML representation."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(val, list):
        items = ", ".join(_toml_value(v) for v in val)
        return f"[{items}]"
    # Fallback: stringify
    escaped = str(val).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _dict_to_toml(data: dict[str, Any]) -> str:
    """Convert a nested dict (one level of TOML tables) to a TOML string."""
    lines: list[str] = []
    # Top-level scalar keys first (uncommon but handle gracefully)
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}

    for key, val in scalars.items():
        lines.append(f"{key} = {_toml_value(val)}")

    if scalars and tables:
        lines.append("")

    for section, fields in tables.items():
        lines.append(f"[{section}]")
        for key, val in fields.items():
            lines.append(f"{key} = {_toml_value(val)}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------

def build_system_prompt(identity: dict[str, Any]) -> str:
    """
    Pure function — derive a system prompt string from an identity dict.
    All fields are read with safe defaults so partial dicts never raise.
    """
    persona = identity.get("persona", {})
    user = identity.get("user", {})
    voice = identity.get("voice", {})

    name: str = persona.get("name", "Essence")
    tone: str = persona.get("tone", "concise")
    verbosity: str = persona.get("verbosity", "medium")
    expertise: list[str] = persona.get("expertise", [])
    language: str = persona.get("language", "en")

    user_name: str = user.get("name", "")
    user_role: str = user.get("role", "")
    preferences: list[str] = user.get("preferences", [])

    style: str = voice.get("style", "peer")
    formality: float = float(voice.get("formality", 0.4))

    # Build expertise clause
    expertise_clause = (
        f"Your areas of expertise include: {', '.join(expertise)}."
        if expertise
        else ""
    )

    # Build user clause
    user_parts: list[str] = []
    if user_name:
        user_parts.append(f"You are speaking with {user_name}")
        if user_role:
            user_parts.append(f"who works as {user_role}")
        user_parts[-1] += "."
    elif user_role:
        user_parts.append(f"The user's role is {user_role}.")

    preferences_clause = (
        f"Keep in mind the following user preferences: {', '.join(preferences)}."
        if preferences
        else ""
    )

    # Formality descriptor
    if formality < 0.25:
        formality_desc = "very informal"
    elif formality < 0.5:
        formality_desc = "informal"
    elif formality < 0.75:
        formality_desc = "moderately formal"
    else:
        formality_desc = "formal"

    parts = [
        f"You are {name}, a personal AI agent.",
        f"Communicate in a {tone} manner with {verbosity} verbosity.",
        f"Adopt a {style} voice that is {formality_desc}.",
        f"Respond in language code: {language}.",
    ]

    if expertise_clause:
        parts.append(expertise_clause)

    parts.extend(p for p in user_parts if p)

    if preferences_clause:
        parts.append(preferences_clause)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Deep merge helper
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *patch* into a copy of *base*."""
    result = {k: (v.copy() if isinstance(v, dict) else v) for k, v in base.items()}
    for key, val in patch.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# IdentityEngine
# ---------------------------------------------------------------------------

class IdentityEngine:
    """Manages loading, saving, and querying the Essence identity profile."""

    def __init__(self, workspace: Path, event_bus: Any = None) -> None:
        self.workspace = Path(workspace)
        self._identity_path = self.workspace / IDENTITY_FILENAME
        self._history_dir = self.workspace / HISTORY_DIR
        self._cache: dict[str, Any] | None = None
        self._bus = event_bus  # optional EventBus — set after bus init

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _ensure_history_dir(self) -> None:
        self._history_dir.mkdir(parents=True, exist_ok=True)

    def _snapshot_path(self, day: date | None = None) -> Path:
        day = day or date.today()
        return self._history_dir / f"{day.isoformat()}.toml"

    def _write_toml(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_dict_to_toml(data), encoding="utf-8")

    def _read_toml(self, path: Path) -> dict[str, Any]:
        with open(path, "rb") as fh:
            return tomllib.load(fh)

    def _stamp_updated(self, data: dict[str, Any]) -> dict[str, Any]:
        data.setdefault("history", {})["last_updated"] = (
            datetime.now(tz=timezone.utc).isoformat()
        )
        return data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> dict[str, Any]:
        """
        Load identity.toml from the workspace root.
        Creates a default file if none exists.
        """
        if not self._identity_path.exists():
            default = _deep_merge(DEFAULT_IDENTITY, {})
            self._write_toml(self._identity_path, default)
            self._cache = default
            return dict(default)

        try:
            raw = self._read_toml(self._identity_path)
        except Exception:
            raw = {}

        # Merge loaded data on top of defaults so new fields are always present
        merged = _deep_merge(DEFAULT_IDENTITY, raw)
        self._cache = merged
        return dict(merged)

    def save(self, data: dict[str, Any]) -> None:
        """
        Write *data* to identity.toml and create a versioned snapshot.
        """
        self._stamp_updated(data)
        self._write_toml(self._identity_path, data)
        self._cache = data

        # Versioned snapshot
        self._ensure_history_dir()
        self._write_toml(self._snapshot_path(), data)

    def update(self, fields: dict[str, Any]) -> dict[str, Any]:
        """
        Protected write path (spec §4.4): emits identity.write to the event
        bus BEFORE touching the file.  The event log records the intent first;
        then the actual write proceeds.  Direct calls to save() bypass this
        guard — internal code should always go through update().
        """
        current = self._cache if self._cache is not None else self.load()
        updated = _deep_merge(current, fields)

        # Emit governance event before writing (Law 1 compliance)
        self._emit_identity_write(fields, updated)

        # Audit log each changed key
        try:
            from server.audit_logger import get_audit_logger
            _al = get_audit_logger()
            for k, v in fields.items():
                old_v = current.get(k, "")
                if old_v != v:
                    _al.log_config_change(f"identity.{k}", str(old_v), str(v))
        except Exception:
            pass

        self.save(updated)
        return dict(updated)

    def _emit_identity_write(
        self, patch: dict[str, Any], full: dict[str, Any]
    ) -> None:
        """Publish identity.write to the event bus if one is wired up."""
        if self._bus is None:
            return
        try:
            from server.event_bus import Envelope
            env = Envelope(
                topic="identity.write",
                source_component="identity_engine",
                data={"patch_keys": list(patch.keys()), "schema_ver": full.get("persona", {}).get("schema_ver", "")},
            )
            self._bus.publish_sync(env)
        except Exception as exc:
            log.warning("identity.write bus publish failed: %s", exc)

    def get_system_prompt(self) -> str:
        """Return a system prompt derived from the current identity."""
        identity = self._cache if self._cache is not None else self.load()
        return build_system_prompt(identity)

    def diff(self) -> list[dict[str, Any]]:
        """
        Return metadata for all snapshots in identity.history/, newest first.
        Each entry: {"date": "YYYY-MM-DD", "path": str}
        """
        if not self._history_dir.exists():
            return []

        snapshots: list[dict[str, Any]] = []
        for path in sorted(self._history_dir.glob("*.toml"), reverse=True):
            snapshots.append({"date": path.stem, "path": str(path)})
        return snapshots

    def export(self) -> bytes:
        """Return the raw bytes of the current identity.toml."""
        if not self._identity_path.exists():
            self.load()  # creates default
        return self._identity_path.read_bytes()

    def import_identity(self, data: bytes) -> dict[str, Any]:
        """
        Parse *data* as TOML, merge with defaults, save, and return the
        resulting identity dict.
        """
        try:
            parsed = tomllib.loads(data.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid TOML in import payload: {exc}") from exc

        merged = _deep_merge(DEFAULT_IDENTITY, parsed)
        self.save(merged)
        return dict(merged)

    def reset(self) -> dict[str, Any]:
        """Reset identity to defaults, saving a snapshot of the previous state."""
        if self._identity_path.exists():
            # Snapshot current state before wiping
            try:
                old = self._read_toml(self._identity_path)
                self._ensure_history_dir()
                now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H%M%S")
                backup_path = self._history_dir / f"pre-reset-{now_str}.toml"
                self._write_toml(backup_path, old)
            except Exception:
                pass  # Best-effort backup

        default = _deep_merge(DEFAULT_IDENTITY, {})
        self.save(default)
        return dict(default)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_DEFAULT_WS = Path(__file__).resolve().parent.parent
_engine: IdentityEngine | None = None


def init_identity_engine(workspace: Path, event_bus: Any = None) -> IdentityEngine:
    """Initialise the global identity engine. Call once from application startup."""
    global _engine
    _engine = IdentityEngine(workspace, event_bus=event_bus)
    return _engine


def get_identity_engine() -> IdentityEngine:
    """Return the global engine, lazily creating one at the default workspace."""
    global _engine
    if _engine is None:
        _engine = IdentityEngine(_DEFAULT_WS)
    return _engine
