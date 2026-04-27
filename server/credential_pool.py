"""
credential_pool.py — Multi-Credential Pool for Essence
========================================================
Enables multiple API keys per provider so Essence can rotate automatically
when one key hits its rate limit, billing cap, or auth failure.

Supports three selection strategies:
  fill_first   — use key #1 until exhausted, then #2, etc. (default)
  round_robin  — rotate through keys in order on each request
  least_used   — prefer the key with the fewest lifetime requests

Pool state is persisted to <workspace>/data/credential_pool.json so
exhaustion cooldowns survive restarts.

Integration
-----------
  kernel._stream_response() calls rotate_credential(provider) when
  classified.should_rotate_credential is True.

  essence.py cmd_pool() handles /pool subcommands (list, add, reset).

Usage
-----
  from server.credential_pool import get_pool, rotate_credential

  pool = get_pool("openai")
  entry = pool.select()          # pick active credential
  pool.mark_exhausted(entry.id, status_code=429)
  next_entry = pool.rotate(entry.id, status_code=429)

  # From kernel: rotate and get the new key
  new_cred = rotate_credential("openai", status_code=429)
  if new_cred:
      headers["Authorization"] = f"Bearer {new_cred.api_key}"
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("essence.credential_pool")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATUS_OK        = "ok"
STATUS_EXHAUSTED = "exhausted"

STRATEGY_FILL_FIRST  = "fill_first"
STRATEGY_ROUND_ROBIN = "round_robin"
STRATEGY_LEAST_USED  = "least_used"

# Cooldown before retrying an exhausted credential
EXHAUSTED_TTL_429 = 3600   # 1 hour for rate-limit
EXHAUSTED_TTL_402 = 3600   # 1 hour for billing
EXHAUSTED_TTL_DEFAULT = 1800  # 30 min otherwise

_POOL_FILE_NAME = "credential_pool.json"


# ---------------------------------------------------------------------------
# PoolEntry
# ---------------------------------------------------------------------------

@dataclass
class PoolEntry:
    """One credential within a provider pool."""
    id:            str
    provider:      str
    label:         str
    api_key:       str
    base_url:      str   = ""
    priority:      int   = 0
    status:        str   = STATUS_OK
    status_at:     Optional[float] = None
    error_code:    Optional[int]   = None
    reset_at:      Optional[float] = None   # provider-supplied reset timestamp
    request_count: int   = 0

    # ── Serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "PoolEntry":
        return PoolEntry(
            id            = d.get("id", uuid.uuid4().hex[:8]),
            provider      = d.get("provider", ""),
            label         = d.get("label", ""),
            api_key       = d.get("api_key", ""),
            base_url      = d.get("base_url", ""),
            priority      = int(d.get("priority", 0)),
            status        = d.get("status", STATUS_OK),
            status_at     = d.get("status_at"),
            error_code    = d.get("error_code"),
            reset_at      = d.get("reset_at"),
            request_count = int(d.get("request_count", 0)),
        )

    # ── Status helpers ────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """True if not currently in exhaustion cooldown."""
        if self.status != STATUS_EXHAUSTED:
            return True
        # Check if cooldown has expired
        until = self._exhausted_until
        return until is None or time.time() >= until

    @property
    def _exhausted_until(self) -> Optional[float]:
        if self.status != STATUS_EXHAUSTED:
            return None
        # Provider-supplied reset timestamp takes priority
        if self.reset_at and self.reset_at > time.time():
            return self.reset_at
        # Fall back to default TTL based on error code
        if self.status_at:
            ttl = {429: EXHAUSTED_TTL_429, 402: EXHAUSTED_TTL_402}.get(
                self.error_code or 0, EXHAUSTED_TTL_DEFAULT
            )
            return self.status_at + ttl
        return None

    @property
    def cooldown_remaining(self) -> float:
        """Seconds remaining in exhaustion cooldown (0 if available)."""
        until = self._exhausted_until
        if until is None:
            return 0.0
        return max(0.0, until - time.time())

    def short_label(self) -> str:
        return self.label or self.id[:8]


# ---------------------------------------------------------------------------
# CredentialPool
# ---------------------------------------------------------------------------

class CredentialPool:
    """Thread-safe pool of credentials for one provider."""

    def __init__(
        self,
        provider:  str,
        entries:   List[PoolEntry],
        data_file: Path,
        strategy:  str = STRATEGY_FILL_FIRST,
    ) -> None:
        self.provider   = provider
        self._entries   = sorted(entries, key=lambda e: e.priority)
        self._data_file = data_file
        self._strategy  = strategy
        self._lock      = threading.Lock()
        self._current_id: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────

    def entries(self) -> List[PoolEntry]:
        """Return a snapshot of all entries."""
        with self._lock:
            return list(self._entries)

    def available_entries(self) -> List[PoolEntry]:
        """Return entries not currently in exhaustion cooldown."""
        with self._lock:
            return self._available_unlocked(clear_expired=True)

    def select(self) -> Optional[PoolEntry]:
        """Pick the next credential based on strategy."""
        with self._lock:
            return self._select_unlocked()

    def current(self) -> Optional[PoolEntry]:
        """Return the currently selected entry (if any)."""
        with self._lock:
            if not self._current_id:
                return None
            return next((e for e in self._entries if e.id == self._current_id), None)

    def mark_exhausted(
        self,
        entry_id:   str,
        *,
        status_code: Optional[int] = None,
        reset_at:    Optional[float] = None,
    ) -> None:
        """Mark a specific credential as exhausted."""
        with self._lock:
            for i, entry in enumerate(self._entries):
                if entry.id == entry_id:
                    self._entries[i] = PoolEntry(
                        **{**entry.to_dict(),
                           "status": STATUS_EXHAUSTED,
                           "status_at": time.time(),
                           "error_code": status_code,
                           "reset_at": reset_at,
                           }
                    )
                    log.info(
                        "pool[%s]: marked %s exhausted (http=%s cooldown=%.0fs)",
                        self.provider, entry.short_label(),
                        status_code, self._entries[i].cooldown_remaining,
                    )
                    break
            self._persist_unlocked()

    def rotate(
        self,
        current_id:  Optional[str] = None,
        *,
        status_code: Optional[int] = None,
        reset_at:    Optional[float] = None,
    ) -> Optional[PoolEntry]:
        """Mark current as exhausted, then select the next available."""
        with self._lock:
            eid = current_id or self._current_id
            if eid:
                for i, entry in enumerate(self._entries):
                    if entry.id == eid:
                        self._entries[i] = PoolEntry(
                            **{**entry.to_dict(),
                               "status": STATUS_EXHAUSTED,
                               "status_at": time.time(),
                               "error_code": status_code,
                               "reset_at": reset_at,
                               }
                        )
                        log.info(
                            "pool[%s]: rotating away from %s",
                            self.provider, entry.short_label(),
                        )
                        break
            self._current_id = None
            next_entry = self._select_unlocked()
            if next_entry:
                log.info("pool[%s]: rotated to %s", self.provider, next_entry.short_label())
            else:
                log.warning("pool[%s]: all credentials exhausted", self.provider)
            self._persist_unlocked()
            return next_entry

    def reset_all(self) -> int:
        """Clear exhaustion status on all entries.  Returns reset count."""
        with self._lock:
            count = 0
            for i, entry in enumerate(self._entries):
                if entry.status == STATUS_EXHAUSTED:
                    self._entries[i] = PoolEntry(
                        **{**entry.to_dict(),
                           "status": STATUS_OK,
                           "status_at": None,
                           "error_code": None,
                           "reset_at": None,
                           }
                    )
                    count += 1
            if count:
                self._persist_unlocked()
            return count

    def add(
        self,
        api_key:  str,
        *,
        base_url: str = "",
        label:    str = "",
    ) -> PoolEntry:
        """Add a new credential to the pool."""
        with self._lock:
            priority = max((e.priority for e in self._entries), default=-1) + 1
            entry = PoolEntry(
                id       = uuid.uuid4().hex[:8],
                provider = self.provider,
                label    = label or f"key-{priority + 1}",
                api_key  = api_key,
                base_url = base_url,
                priority = priority,
            )
            self._entries.append(entry)
            self._persist_unlocked()
            log.info("pool[%s]: added credential %s", self.provider, entry.short_label())
            return entry

    def remove(self, entry_id: str) -> bool:
        """Remove a credential by id.  Returns True if found and removed."""
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.id != entry_id]
            if len(self._entries) < before:
                if self._current_id == entry_id:
                    self._current_id = None
                self._persist_unlocked()
                return True
            return False

    def summary_lines(self) -> List[str]:
        """Return human-readable status lines for /pool list."""
        with self._lock:
            if not self._entries:
                return [f"  pool[{self.provider}]  (empty — using single configured key)"]
            lines = [f"  pool[{self.provider}]  strategy={self._strategy}"]
            for e in sorted(self._entries, key=lambda x: x.priority):
                star = "►" if e.id == self._current_id else " "
                if e.status == STATUS_EXHAUSTED:
                    cd = e.cooldown_remaining
                    status_str = f"exhausted  cooldown={cd:.0f}s"
                else:
                    status_str = f"ok  req={e.request_count}"
                label = e.short_label()[:30]
                lines.append(f"  {star} #{e.priority+1} [{e.id[:6]}] {label:<30} {status_str}")
            return lines

    # ── Internal ──────────────────────────────────────────────────────

    def _available_unlocked(self, *, clear_expired: bool = False) -> List[PoolEntry]:
        """Return available entries; optionally clear expired exhaustion."""
        available: List[PoolEntry] = []
        cleared = False
        for i, entry in enumerate(self._entries):
            if entry.status == STATUS_EXHAUSTED:
                if entry.is_available:  # cooldown expired
                    if clear_expired:
                        self._entries[i] = PoolEntry(
                            **{**entry.to_dict(),
                               "status": STATUS_OK,
                               "status_at": None,
                               "error_code": None,
                               "reset_at": None,
                               }
                        )
                        cleared = True
                        available.append(self._entries[i])
                    # else: still exhausted — skip
                # else: still in cooldown — skip
            else:
                available.append(entry)
        if cleared:
            self._persist_unlocked()
        return available

    def _select_unlocked(self) -> Optional[PoolEntry]:
        available = self._available_unlocked(clear_expired=True)
        if not available:
            self._current_id = None
            return None

        if self._strategy == STRATEGY_LEAST_USED:
            entry = min(available, key=lambda e: e.request_count)
        elif self._strategy == STRATEGY_ROUND_ROBIN and self._current_id:
            # Find the entry after current in the priority-sorted list
            ids = [e.id for e in available]
            try:
                idx = ids.index(self._current_id)
                entry = available[(idx + 1) % len(available)]
            except ValueError:
                entry = available[0]
        else:
            # fill_first: first by priority
            entry = available[0]

        # Increment request_count
        for i, e in enumerate(self._entries):
            if e.id == entry.id:
                self._entries[i] = PoolEntry(
                    **{**e.to_dict(), "request_count": e.request_count + 1}
                )
                entry = self._entries[i]
                break

        self._current_id = entry.id
        self._persist_unlocked()
        return entry

    def _persist_unlocked(self) -> None:
        """Write pool state to JSON file (called with lock held)."""
        try:
            self._data_file.parent.mkdir(parents=True, exist_ok=True)
            raw: dict = {}
            try:
                raw = json.loads(self._data_file.read_text()) if self._data_file.exists() else {}
            except Exception:
                pass
            raw[self.provider] = {
                "strategy": self._strategy,
                "entries":  [e.to_dict() for e in self._entries],
            }
            self._data_file.write_text(json.dumps(raw, indent=2))
        except Exception as exc:
            log.debug("pool persist failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton registry + helpers
# ---------------------------------------------------------------------------

_pools:     Dict[str, CredentialPool] = {}
_pools_lock = threading.Lock()
_data_file: Optional[Path] = None


def _get_data_file() -> Path:
    global _data_file
    if _data_file is None:
        ws = Path(__file__).resolve().parent.parent
        _data_file = ws / "data" / _POOL_FILE_NAME
    return _data_file


def _load_pool_entries(provider: str, data_file: Path) -> Tuple[List[PoolEntry], str]:
    """Load pool entries and strategy from the JSON sidecar."""
    try:
        if not data_file.exists():
            return [], STRATEGY_FILL_FIRST
        raw = json.loads(data_file.read_text())
        pdata = raw.get(provider, {})
        strategy = pdata.get("strategy", STRATEGY_FILL_FIRST)
        entries  = [PoolEntry.from_dict(e) for e in pdata.get("entries", [])]
        return entries, strategy
    except Exception as exc:
        log.debug("Failed to load pool for %s: %s", provider, exc)
        return [], STRATEGY_FILL_FIRST


def get_pool(provider: str) -> CredentialPool:
    """Return (creating if needed) the CredentialPool singleton for provider."""
    with _pools_lock:
        if provider not in _pools:
            df = _get_data_file()
            entries, strategy = _load_pool_entries(provider, df)
            _pools[provider] = CredentialPool(provider, entries, df, strategy)
        return _pools[provider]


def rotate_credential(
    provider:    str,
    *,
    status_code: Optional[int] = None,
    reset_at:    Optional[float] = None,
) -> Optional[PoolEntry]:
    """
    Rotate away from the current credential for provider.
    Returns the next available PoolEntry or None if pool is empty/exhausted.

    Called from kernel when classify_api_error returns should_rotate_credential=True.
    If the pool has only 0-1 entries, returns None (single-key mode).
    """
    pool = get_pool(provider)
    if len(pool.entries()) < 2:
        # No pool configured — nothing to rotate to
        return None
    current = pool.current()
    return pool.rotate(
        current.id if current else None,
        status_code=status_code,
        reset_at=reset_at,
    )


def pool_summary(provider: str) -> List[str]:
    """Return summary lines for TUI /pool list display."""
    return get_pool(provider).summary_lines()
