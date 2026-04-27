"""
memory_sync.py — Remote Memory Backup / Sync for Essence
=========================================================
Backs up critical SQLite databases to an S3-compatible object store
(S3, R2, MinIO, Backblaze B2) on graceful shutdown and optionally
restores them on first boot.

Configuration (environment variables):
  ESSENCE_SYNC_BUCKET       — S3 bucket name (required)
  ESSENCE_SYNC_PREFIX       — object key prefix (default: "essence/memory/")
  AWS_ACCESS_KEY_ID         — AWS/R2/MinIO access key
  AWS_SECRET_ACCESS_KEY     — AWS/R2/MinIO secret key
  AWS_DEFAULT_REGION        — region (default: "auto" for R2)
  ESSENCE_SYNC_ENDPOINT     — custom endpoint URL (for R2/MinIO/B2)
  ESSENCE_SYNC_ENABLED      — "true" / "false" (default: auto-detect)

Without boto3 installed the module loads fine but sync is a no-op
(logs a debug message instead of failing noisily).
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

log = logging.getLogger("essence.memory_sync")

# Files to back up (relative to workspace/data/)
_SYNC_FILES = [
    "episodic.db",
    "gravity_memory.db",
    "event_log.db",
    "trust_ledger.json",
    "cie_budget.json",
    "service_registry.db",
    "audit_log.db",
]


def _is_enabled() -> bool:
    val = os.environ.get("ESSENCE_SYNC_ENABLED", "").lower()
    if val in ("false", "0", "off"):
        return False
    if val in ("true", "1", "on"):
        return True
    # Auto-detect: enable if bucket is configured
    return bool(os.environ.get("ESSENCE_SYNC_BUCKET", ""))


def _get_s3():
    """Return a boto3 S3 client or None if boto3 unavailable / not configured."""
    if not _is_enabled():
        return None
    try:
        import boto3  # type: ignore
    except ImportError:
        log.debug("memory_sync: boto3 not installed — sync disabled")
        return None

    bucket = os.environ.get("ESSENCE_SYNC_BUCKET")
    if not bucket:
        return None

    kwargs: dict = {
        "aws_access_key_id":     os.environ.get("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "region_name":           os.environ.get("AWS_DEFAULT_REGION", "auto"),
    }
    endpoint = os.environ.get("ESSENCE_SYNC_ENDPOINT")
    if endpoint:
        kwargs["endpoint_url"] = endpoint

    return boto3.client("s3", **{k: v for k, v in kwargs.items() if v})


async def backup(workspace: Path) -> dict:
    """
    Upload modified database files to S3.
    Returns a summary dict: {"uploaded": [...], "skipped": [...], "errors": [...]}.
    """
    s3 = _get_s3()
    if s3 is None:
        log.debug("memory_sync: backup skipped (not configured)")
        return {"uploaded": [], "skipped": [], "errors": []}

    bucket = os.environ.get("ESSENCE_SYNC_BUCKET", "")
    prefix = os.environ.get("ESSENCE_SYNC_PREFIX", "essence/memory/")
    data_dir = workspace / "data"

    uploaded: list[str] = []
    skipped:  list[str] = []
    errors:   list[str] = []

    for fname in _SYNC_FILES:
        p = data_dir / fname
        if not p.exists():
            skipped.append(fname)
            continue
        key = prefix + fname
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda p=p, key=key: s3.upload_file(str(p), bucket, key)
            )
            uploaded.append(fname)
            log.info("memory_sync: uploaded %s → s3://%s/%s", fname, bucket, key)
        except Exception as exc:
            errors.append(f"{fname}: {exc}")
            log.warning("memory_sync: upload failed for %s: %s", fname, exc)

    return {"uploaded": uploaded, "skipped": skipped, "errors": errors}


async def restore(workspace: Path) -> dict:
    """
    Download database files from S3 (only if local file does not exist).
    Returns a summary dict.
    """
    s3 = _get_s3()
    if s3 is None:
        log.debug("memory_sync: restore skipped (not configured)")
        return {"restored": [], "skipped": [], "errors": []}

    bucket   = os.environ.get("ESSENCE_SYNC_BUCKET", "")
    prefix   = os.environ.get("ESSENCE_SYNC_PREFIX", "essence/memory/")
    data_dir = workspace / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    restored: list[str] = []
    skipped:  list[str] = []
    errors:   list[str] = []

    for fname in _SYNC_FILES:
        p = data_dir / fname
        if p.exists():
            skipped.append(fname)
            continue
        key = prefix + fname
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda p=p, key=key: s3.download_file(bucket, key, str(p))
            )
            restored.append(fname)
            log.info("memory_sync: restored %s from s3://%s/%s", fname, bucket, key)
        except Exception as exc:
            # File might not exist in bucket (new install) — that's fine
            log.debug("memory_sync: restore skipped for %s: %s", fname, exc)

    return {"restored": restored, "skipped": skipped, "errors": errors}


def sync_status() -> str:
    """Return human-readable sync status string."""
    if not _is_enabled():
        return "memory_sync: disabled (set ESSENCE_SYNC_BUCKET to enable)"
    bucket   = os.environ.get("ESSENCE_SYNC_BUCKET", "?")
    prefix   = os.environ.get("ESSENCE_SYNC_PREFIX", "essence/memory/")
    endpoint = os.environ.get("ESSENCE_SYNC_ENDPOINT", "AWS S3")
    return f"memory_sync: enabled  bucket={bucket}  prefix={prefix}  endpoint={endpoint}"
