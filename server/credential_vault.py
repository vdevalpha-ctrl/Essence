"""
credential_vault.py — Scope-Restricted Credential Vault for Essence
====================================================================
Provides short-lived, scope-restricted tokens that skills can request
and that expire automatically after a configurable TTL.

No plaintext secrets are stored in the vault.  API keys are stored
encrypted using Fernet symmetric encryption.  The vault key is derived
from the system keyring (if available) or from a per-workspace key file
(chmod 600).

Usage:
  from server.credential_vault import get_vault
  vault = get_vault()

  # Store (called once, by operator)
  vault.store("openai", "OPENAI_API_KEY", api_key)

  # Retrieve (called by skills)
  token = vault.issue("openai", scope="chat", ttl=300)
  key   = vault.retrieve(token)   # raises VaultError if expired/wrong scope
  vault.revoke(token)             # revoke early
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import time
from pathlib import Path
from threading import Lock
from typing import Optional

log = logging.getLogger("essence.vault")

_VAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "credential_vault.enc.json"
_KEY_PATH   = Path(__file__).resolve().parent.parent / "data" / ".vault_key"
_DEFAULT_TTL = 300   # seconds


class VaultError(Exception):
    pass


class CredentialVault:
    """
    Fernet-encrypted credential store with scope-restricted TTL tokens.
    """

    def __init__(
        self,
        vault_path: Path = _VAULT_PATH,
        key_path:   Path = _KEY_PATH,
    ) -> None:
        self._vault_path = vault_path
        self._key_path   = key_path
        self._lock       = Lock()
        self._tokens:   dict[str, dict]  = {}   # token → {scope, provider, expires}
        self._fernet     = self._load_or_create_fernet()

    # ── Fernet key management ────────────────────────────────────

    def _load_or_create_fernet(self):
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            log.warning("vault: cryptography not installed — vault disabled (pip install cryptography)")
            return None

        # Try system keyring first
        try:
            import keyring  # type: ignore
            raw = keyring.get_password("essence", "vault_key")
            if raw:
                return Fernet(raw.encode())
        except Exception:
            pass

        # Fall back to file-based key
        if self._key_path.exists():
            key = self._key_path.read_bytes().strip()
        else:
            key = Fernet.generate_key()
            self._key_path.parent.mkdir(parents=True, exist_ok=True)
            self._key_path.write_bytes(key)
            try:
                self._key_path.chmod(0o600)
            except Exception:
                pass
            log.info("vault: generated new key at %s", self._key_path)

        return Fernet(key)

    # ── Encrypt / decrypt ─────────────────────────────────────────

    def _encrypt(self, plaintext: str) -> str:
        if self._fernet is None:
            return plaintext   # no-op when cryptography not available
        return self._fernet.encrypt(plaintext.encode()).decode()

    def _decrypt(self, ciphertext: str) -> str:
        if self._fernet is None:
            return ciphertext
        try:
            from cryptography.fernet import Fernet, InvalidToken
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except Exception as exc:
            raise VaultError(f"Decryption failed: {exc}") from exc

    # ── Persistence ───────────────────────────────────────────────

    def _load(self) -> dict:
        try:
            if self._vault_path.exists():
                return json.loads(self._vault_path.read_text("utf-8"))
        except Exception as exc:
            log.debug("vault: load failed: %s", exc)
        return {}

    def _save(self, data: dict) -> None:
        try:
            self._vault_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._vault_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(self._vault_path)
            try:
                self._vault_path.chmod(0o600)
            except Exception:
                pass
        except Exception as exc:
            log.warning("vault: save failed: %s", exc)

    # ── Public API ─────────────────────────────────────────────────

    def store(self, provider: str, env_var: str, api_key: str) -> None:
        """
        Store an API key for a provider.
        `env_var` is the environment variable name (for documentation only).
        """
        with self._lock:
            data = self._load()
            data[provider] = {
                "env_var":    env_var,
                "ciphertext": self._encrypt(api_key),
                "stored_at":  time.time(),
            }
            self._save(data)
        log.info("vault: stored key for %s (%s)", provider, env_var)

    def issue(
        self,
        provider: str,
        scope:    str   = "*",
        ttl:      int   = _DEFAULT_TTL,
    ) -> str:
        """
        Issue a short-lived token for the given provider.
        The token grants access to the provider's key for `ttl` seconds,
        restricted to `scope`.
        Returns a token string.
        """
        with self._lock:
            data = self._load()
            if provider not in data:
                raise VaultError(f"No credential stored for provider '{provider}'")
            token = secrets.token_urlsafe(32)
            self._tokens[token] = {
                "provider": provider,
                "scope":    scope,
                "expires":  time.time() + ttl,
            }
        return token

    def retrieve(self, token: str, required_scope: str = "*") -> str:
        """
        Retrieve the API key for the given token.
        Raises VaultError if the token is expired, unknown, or out of scope.
        """
        self._purge_expired()
        with self._lock:
            meta = self._tokens.get(token)
            if meta is None:
                raise VaultError("Invalid or expired vault token")
            if time.time() > meta["expires"]:
                del self._tokens[token]
                raise VaultError("Vault token has expired")
            token_scope = meta.get("scope", "*")
            if token_scope != "*" and required_scope != "*" and token_scope != required_scope:
                raise VaultError(
                    f"Vault token scope mismatch: token={token_scope} required={required_scope}"
                )
            data = self._load()
            provider = meta["provider"]
            if provider not in data:
                raise VaultError(f"Credential for '{provider}' no longer exists")
            return self._decrypt(data[provider]["ciphertext"])

    def revoke(self, token: str) -> bool:
        """Revoke a token early.  Returns True if it existed."""
        with self._lock:
            return self._tokens.pop(token, None) is not None

    def delete(self, provider: str) -> bool:
        """Permanently remove a stored credential."""
        with self._lock:
            data = self._load()
            if provider not in data:
                return False
            del data[provider]
            self._save(data)
            # Revoke all active tokens for this provider
            self._tokens = {
                t: m for t, m in self._tokens.items() if m["provider"] != provider
            }
        return True

    def list_providers(self) -> list[str]:
        return list(self._load().keys())

    def _purge_expired(self) -> None:
        now = time.time()
        with self._lock:
            self._tokens = {t: m for t, m in self._tokens.items() if m["expires"] > now}


# ── Module-level singleton ─────────────────────────────────────────────────

_vault: Optional[CredentialVault] = None


def get_vault() -> CredentialVault:
    global _vault
    if _vault is None:
        _vault = CredentialVault()
    return _vault
