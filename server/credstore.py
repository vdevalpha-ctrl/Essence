"""
Essence Credential Store — Windows Credential Manager (DPAPI) backend.

All provider API keys and custom base URLs are stored in Windows Credential
Vault under the service name "Essence-v29". Nothing is written to disk in plain
text. The keyring library delegates to WinVaultKeyring on Windows, which uses
the same DPAPI encryption as the OS itself.

Service layout
--------------
  service : "Essence-v29"
  username: "key:{provider_id}"      → API key string
  username: "url:{provider_id}"      → custom base URL string (optional)

Usage
-----
  from server.credstore import cred_get, cred_set, cred_delete, cred_load_all

  cred_set("openai", api_key="sk-…")          # save
  key = cred_get("openai")                    # retrieve
  cred_delete("openai")                       # remove
  mapping = cred_load_all(provider_ids)       # bulk load → dict {pid: {"api_key":…, "base_url":…}}
"""
from __future__ import annotations
import logging

log = logging.getLogger("essence.credstore")

_SERVICE = "Essence-v29"

try:
    import keyring as _kr
    _AVAILABLE = True
except ImportError:
    _kr = None      # type: ignore
    _AVAILABLE = False
    log.warning("keyring not installed — API keys will NOT be persisted. Run: pip install keyring")


def _key_name(pid: str) -> str:
    return f"key:{pid}"

def _url_name(pid: str) -> str:
    return f"url:{pid}"


def cred_set(pid: str, *, api_key: str = "", base_url: str = "") -> None:
    """Store (or update) credentials for provider `pid` in Windows Credential Manager."""
    if not _AVAILABLE:
        return
    if api_key:
        _kr.set_password(_SERVICE, _key_name(pid), api_key)
        log.debug("cred_set: stored key for %s", pid)
    if base_url:
        _kr.set_password(_SERVICE, _url_name(pid), base_url)
        log.debug("cred_set: stored url for %s", pid)


def cred_get(pid: str) -> str:
    """Return the stored API key for `pid`, or empty string."""
    if not _AVAILABLE:
        return ""
    try:
        return _kr.get_password(_SERVICE, _key_name(pid)) or ""
    except Exception as e:
        log.warning("cred_get(%s) failed: %s", pid, e)
        return ""


def cred_get_url(pid: str) -> str:
    """Return the stored custom base URL for `pid`, or empty string."""
    if not _AVAILABLE:
        return ""
    try:
        return _kr.get_password(_SERVICE, _url_name(pid)) or ""
    except Exception as e:
        log.warning("cred_get_url(%s) failed: %s", pid, e)
        return ""


def cred_delete(pid: str) -> None:
    """Delete stored credentials for `pid` from Windows Credential Manager."""
    if not _AVAILABLE:
        return
    for username in (_key_name(pid), _url_name(pid)):
        try:
            _kr.delete_password(_SERVICE, username)
            log.debug("cred_delete: removed %s", username)
        except Exception:
            pass   # not stored — ignore


def cred_has(pid: str) -> bool:
    """Return True if an API key exists for `pid`."""
    return bool(cred_get(pid))


def cred_load_all(provider_ids: list[str]) -> dict[str, dict]:
    """
    Bulk-load all credentials for the given provider IDs.
    Returns {pid: {"api_key": "…", "base_url": "…"}} for any that have stored data.
    """
    result: dict[str, dict] = {}
    for pid in provider_ids:
        key = cred_get(pid)
        url = cred_get_url(pid)
        if key or url:
            result[pid] = {}
            if key: result[pid]["api_key"]  = key
            if url: result[pid]["base_url"] = url
    return result
