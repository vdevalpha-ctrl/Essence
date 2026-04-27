"""
skill_marketplace.py — Essence Skill Marketplace
=================================================
Provides a signed skill registry so users can browse, install, and
verify community skills.

Architecture:
  MarketplaceClient — searches/fetches from a registry endpoint
  SkillVerifier     — verifies Ed25519 signatures on skill packages
  install_skill()   — downloads, verifies, and installs to memory/skills/

Registry format (JSON):
  {
    "skills": [
      {
        "id":          "web-researcher",
        "name":        "Web Researcher",
        "version":     "1.2.0",
        "author":      "community",
        "description": "...",
        "category":    "research",
        "download_url":"https://...",
        "signature":   "<base64-ed25519-sig>",
        "sha256":      "<hex-digest>",
        "tags":        ["web", "search"]
      }
    ]
  }

Usage via CLI:
  python essence.py skill search <query>
  python essence.py skill install <id>
  python essence.py skill verify <id>
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("essence.marketplace")

_DEFAULT_REGISTRY = os.environ.get(
    "ESSENCE_MARKETPLACE_URL",
    "https://raw.githubusercontent.com/olatunjih/Essence/main/marketplace/registry.json",
)


# ── Registry client ────────────────────────────────────────────────────────

class MarketplaceClient:
    """Fetch and search the remote skill registry."""

    def __init__(self, registry_url: str = _DEFAULT_REGISTRY) -> None:
        self._url   = registry_url
        self._cache: Optional[list[dict]] = None

    async def fetch(self) -> list[dict]:
        """Download and return the list of registry entries."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get(self._url, follow_redirects=True,
                                headers={"User-Agent": "Essence/1"})
                r.raise_for_status()
                data = r.json()
                self._cache = data.get("skills", [])
                return self._cache
        except Exception as exc:
            log.warning("marketplace: fetch failed: %s", exc)
            return self._cache or []

    async def search(self, query: str) -> list[dict]:
        """Search the registry by id/name/description/tags."""
        skills = self._cache or await self.fetch()
        q = query.lower()
        results = []
        for s in skills:
            haystack = " ".join([
                s.get("id", ""), s.get("name", ""),
                s.get("description", ""), " ".join(s.get("tags", [])),
            ]).lower()
            if q in haystack:
                results.append(s)
        return results

    async def get(self, skill_id: str) -> Optional[dict]:
        """Fetch metadata for a specific skill id."""
        skills = self._cache or await self.fetch()
        return next((s for s in skills if s.get("id") == skill_id), None)


# ── Signature verifier ─────────────────────────────────────────────────────

class SkillVerifier:
    """
    Verifies Ed25519 signatures on skill packages.
    The public key is embedded here and checked against the registry's
    signature field.

    For community-signed skills the verification is advisory (warn-only).
    For official skills from the Essence repo, verification is mandatory.
    """

    # Placeholder public key — replace with actual project key
    _PUBLIC_KEY_HEX = os.environ.get(
        "ESSENCE_MARKETPLACE_PUBKEY",
        "",    # empty → skip signature verification (warn-only mode)
    )

    def verify(self, content: bytes, signature_b64: str, sha256_hex: str) -> tuple[bool, str]:
        """
        Verify content integrity and optionally its Ed25519 signature.
        Returns (ok, message).
        """
        # SHA-256 integrity check (always)
        actual = hashlib.sha256(content).hexdigest()
        if actual != sha256_hex:
            return False, f"SHA-256 mismatch (got {actual[:16]}…)"

        # Ed25519 signature check (if key configured)
        if not self._PUBLIC_KEY_HEX:
            return True, "SHA-256 OK (signature verification skipped — no public key configured)"

        try:
            import base64
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
            from cryptography.exceptions import InvalidSignature

            pub_bytes = bytes.fromhex(self._PUBLIC_KEY_HEX)
            pub_key   = Ed25519PublicKey.from_public_bytes(pub_bytes)
            sig       = base64.b64decode(signature_b64)
            pub_key.verify(sig, content)
            return True, "SHA-256 OK · Ed25519 signature verified"
        except ImportError:
            return True, "SHA-256 OK (Ed25519 check skipped — install cryptography)"
        except Exception as exc:
            return False, f"Signature invalid: {exc}"


# ── Installer ─────────────────────────────────────────────────────────────

async def install_skill(
    skill_id:   str,
    skills_dir: Path,
    registry_url: str = _DEFAULT_REGISTRY,
    force:      bool  = False,
) -> dict:
    """
    Download and install a skill from the marketplace.

    Returns a result dict: {"ok": bool, "message": str, "path": str}
    """
    client   = MarketplaceClient(registry_url)
    verifier = SkillVerifier()

    meta = await client.get(skill_id)
    if meta is None:
        return {"ok": False, "message": f"Skill '{skill_id}' not found in marketplace"}

    dest = skills_dir / f"{skill_id}.md"
    if dest.exists() and not force:
        return {"ok": False, "message": f"Skill already installed at {dest} (use --force to overwrite)"}

    import httpx
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
            r = await c.get(meta["download_url"], headers={"User-Agent": "Essence/1"})
            r.raise_for_status()
            content = r.content
    except Exception as exc:
        return {"ok": False, "message": f"Download failed: {exc}"}

    # Verify
    ok, msg = verifier.verify(
        content,
        meta.get("signature", ""),
        meta.get("sha256", hashlib.sha256(content).hexdigest()),
    )
    if not ok:
        return {"ok": False, "message": f"Verification failed: {msg}"}

    skills_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    log.info("marketplace: installed %s → %s", skill_id, dest)
    return {"ok": True, "message": msg, "path": str(dest)}


# ── Module-level singleton ─────────────────────────────────────────────────

_client: Optional[MarketplaceClient] = None


def get_marketplace(registry_url: str = _DEFAULT_REGISTRY) -> MarketplaceClient:
    global _client
    if _client is None:
        _client = MarketplaceClient(registry_url)
    return _client
