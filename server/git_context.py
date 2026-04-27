"""
git_context.py — Git Repository Context Injection
==================================================
Auto-detects a git repo at or above the workspace root and injects
branch/commit/diff context into the LLM system prompt.

Also provides async executor functions for the git_status, git_log,
and git_diff tools that are registered in tools_engine.py.
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("essence.git_context")

_MAX_DIFF_CHARS = 3000
_MAX_LOG_LINES  = 12
_CACHE_TTL      = 30.0   # seconds

# ── Repo detection ─────────────────────────────────────────────────────────

def detect_git_repo(workspace: Path) -> Optional[Path]:
    """Walk up from workspace to find a .git directory.  Returns repo root or None."""
    p = workspace.resolve()
    for ancestor in [p, *p.parents]:
        if (ancestor / ".git").exists():
            return ancestor
    return None


# ── Context builder ────────────────────────────────────────────────────────

def get_git_context(repo: Path, max_chars: int = 1500) -> str:
    """
    Return a compact context string: branch, HEAD, recent commits, short diff.
    Returns "" if not in a git repo or git is unavailable.
    """
    parts: list[str] = []

    try:
        branch = _git(repo, ["rev-parse", "--abbrev-ref", "HEAD"]).strip()
        head   = _git(repo, ["log", "-1", "--format=%h %s"]).strip()
        parts.append(f"Git: branch={branch}  HEAD: {head}")
    except Exception:
        return ""

    try:
        log_out = _git(repo, [
            "log", f"-{_MAX_LOG_LINES}", "--oneline", "--no-merges",
        ]).strip()
        if log_out:
            parts.append("Recent commits:\n" + log_out)
    except Exception:
        pass

    try:
        stat = _git(repo, ["diff", "--stat", "HEAD"]).strip()
        if stat:
            diff = _git(repo, ["diff", "HEAD"]).strip()
            if len(diff) > _MAX_DIFF_CHARS:
                diff = diff[:_MAX_DIFF_CHARS] + "\n[...diff truncated]"
            parts.append("Working-tree diff:\n" + diff)
    except Exception:
        pass

    ctx = "\n\n".join(parts)
    return ctx[:max_chars]


# ── Tool executors ─────────────────────────────────────────────────────────

async def exec_git_status(args: dict) -> str:
    repo = _resolve_repo(args)
    if repo is None:
        return "ERROR: not in a git repository"
    try:
        out = _git(repo, ["status", "--short"]).strip()
        return out or "(clean working tree)"
    except Exception as e:
        return f"ERROR: {e}"


async def exec_git_log(args: dict) -> str:
    repo = _resolve_repo(args)
    if repo is None:
        return "ERROR: not in a git repository"
    n   = min(int(args.get("n", 10)), 50)
    fmt = args.get("format", "oneline")
    git_fmt = {"oneline": "--oneline", "short": "--format=short",
               "medium": "--format=medium"}.get(fmt, "--oneline")
    try:
        out = _git(repo, ["log", f"-{n}", git_fmt]).strip()
        return out or "(no commits)"
    except Exception as e:
        return f"ERROR: {e}"


async def exec_git_diff(args: dict) -> str:
    repo = _resolve_repo(args)
    if repo is None:
        return "ERROR: not in a git repository"
    ref    = args.get("ref", "HEAD")
    staged = bool(args.get("staged", False))
    cmd    = ["diff", "--cached"] if staged else ["diff", ref]
    try:
        out = _git(repo, cmd)
        if len(out) > 6000:
            out = out[:6000] + "\n[...truncated]"
        return out or "(no diff)"
    except Exception as e:
        return f"ERROR: {e}"


# ── Cached context ─────────────────────────────────────────────────────────

_cached_ctx:  str            = ""
_cached_at:   float          = 0.0
_cached_repo: Optional[Path] = None


def get_cached_git_context(workspace: Path, max_chars: int = 1500) -> str:
    """Return a cached git context string, refreshed every CACHE_TTL seconds."""
    global _cached_ctx, _cached_at, _cached_repo

    now = time.monotonic()
    if _cached_repo is None:
        _cached_repo = detect_git_repo(workspace)
    if _cached_repo is None:
        return ""

    if now - _cached_at < _CACHE_TTL and _cached_ctx:
        return _cached_ctx

    try:
        _cached_ctx = get_git_context(_cached_repo, max_chars)
        _cached_at  = now
    except Exception as exc:
        log.debug("git_context: refresh failed: %s", exc)

    return _cached_ctx


# ── Internals ──────────────────────────────────────────────────────────────

def _git(repo: Path, args: list[str], timeout: int = 5) -> str:
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True, text=True, cwd=str(repo), timeout=timeout,
        )
        return result.stdout + result.stderr
    except FileNotFoundError:
        raise RuntimeError("git not found in PATH")


def _resolve_repo(args: dict) -> Optional[Path]:
    """Resolve repo path from args, falling back to workspace."""
    try:
        from server.tools_engine import workspace as _ws
        ws = _ws()
    except Exception:
        ws = Path.cwd()

    raw = args.get("repo", "")
    if raw:
        p = Path(raw)
        if (p / ".git").exists():
            return p
    return detect_git_repo(ws)
