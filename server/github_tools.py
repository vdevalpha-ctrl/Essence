"""
github_tools.py — GitHub Integration Tools for Essence
=======================================================
Provides GitHub-aware tools that the LLM can call natively.

Authentication: set GITHUB_TOKEN environment variable (classic PAT or
fine-grained token).  Without a token, public repo queries still work
but rate limits are tighter (60 req/hour vs 5000).

Tools:
  gh_list_repos   — list repos for user / org
  gh_list_issues  — list open issues on a repo
  gh_create_issue — create a new issue
  gh_list_prs     — list open pull requests
  gh_get_file     — get a file from a repo
  gh_search_code  — search code across GitHub
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

log = logging.getLogger("essence.github")

_BASE = "https://api.github.com"
_TIMEOUT = 15


def _headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN", "")
    h: dict = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Essence/1",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


async def _get(path: str, params: dict | None = None) -> Any:
    url = _BASE + path
    async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_headers()) as c:
        r = await c.get(url, params=params)
        if r.status_code == 401:
            return {"error": "Unauthorized — set GITHUB_TOKEN environment variable"}
        if r.status_code == 404:
            return {"error": f"Not found: {url}"}
        r.raise_for_status()
        return r.json()


async def _post(path: str, body: dict) -> Any:
    url = _BASE + path
    async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_headers()) as c:
        r = await c.post(url, json=body)
        if r.status_code in (401, 403):
            return {"error": "Unauthorized — check GITHUB_TOKEN permissions"}
        r.raise_for_status()
        return r.json()


# ── Tool executors ─────────────────────────────────────────────────────────

async def exec_gh_list_repos(args: dict) -> str:
    owner = args.get("owner", "")
    if not owner:
        return "ERROR: owner (user/org name) is required"
    try:
        data = await _get(f"/users/{owner}/repos",
                          params={"sort": "updated", "per_page": 20})
        if isinstance(data, dict) and "error" in data:
            return f"ERROR: {data['error']}"
        lines = [f"{r['full_name']}  {'⭐'+str(r['stargazers_count']):<8}  {r.get('description','')[:60]}"
                 for r in data]
        return "\n".join(lines) or "(no repos)"
    except Exception as e:
        return f"ERROR: {e}"


async def exec_gh_list_issues(args: dict) -> str:
    repo  = args.get("repo", "")
    state = args.get("state", "open")
    if not repo or "/" not in repo:
        return "ERROR: repo must be in owner/name format"
    try:
        data = await _get(f"/repos/{repo}/issues",
                          params={"state": state, "per_page": 20})
        if isinstance(data, dict) and "error" in data:
            return f"ERROR: {data['error']}"
        lines = [f"#{i['number']}  [{i['state']}]  {i['title'][:80]}  ({i.get('user',{}).get('login','')})"
                 for i in data if "pull_request" not in i]
        return "\n".join(lines) or f"(no {state} issues)"
    except Exception as e:
        return f"ERROR: {e}"


async def exec_gh_create_issue(args: dict) -> str:
    repo  = args.get("repo", "")
    title = args.get("title", "")
    body  = args.get("body", "")
    if not repo or not title:
        return "ERROR: repo and title are required"
    try:
        data = await _post(f"/repos/{repo}/issues", {"title": title, "body": body})
        if isinstance(data, dict) and "error" in data:
            return f"ERROR: {data['error']}"
        return f"Created issue #{data['number']}: {data['html_url']}"
    except Exception as e:
        return f"ERROR: {e}"


async def exec_gh_list_prs(args: dict) -> str:
    repo  = args.get("repo", "")
    state = args.get("state", "open")
    if not repo or "/" not in repo:
        return "ERROR: repo must be in owner/name format"
    try:
        data = await _get(f"/repos/{repo}/pulls",
                          params={"state": state, "per_page": 20})
        if isinstance(data, dict) and "error" in data:
            return f"ERROR: {data['error']}"
        lines = [f"#{p['number']}  {p['title'][:70]}  ← {p['head']['ref']}  ({p['user']['login']})"
                 for p in data]
        return "\n".join(lines) or f"(no {state} PRs)"
    except Exception as e:
        return f"ERROR: {e}"


async def exec_gh_get_file(args: dict) -> str:
    repo   = args.get("repo", "")
    path   = args.get("path", "")
    ref    = args.get("ref", "main")
    if not repo or not path:
        return "ERROR: repo and path are required"
    try:
        import base64
        data = await _get(f"/repos/{repo}/contents/{path}", params={"ref": ref})
        if isinstance(data, dict) and "error" in data:
            return f"ERROR: {data['error']}"
        if isinstance(data, dict) and data.get("encoding") == "base64":
            content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            if len(content) > 8000:
                content = content[:8000] + "\n[...truncated]"
            return content
        return json.dumps(data, indent=2)[:4000]
    except Exception as e:
        return f"ERROR: {e}"


async def exec_gh_search_code(args: dict) -> str:
    query = args.get("query", "")
    if not query:
        return "ERROR: query is required"
    try:
        data = await _get("/search/code", params={"q": query, "per_page": 10})
        if isinstance(data, dict) and "error" in data:
            return f"ERROR: {data['error']}"
        items = data.get("items", [])
        lines = [
            f"{i['repository']['full_name']}/{i['path']}  ({i.get('html_url','')})"
            for i in items
        ]
        total = data.get("total_count", len(items))
        return f"{total} results\n" + "\n".join(lines)
    except Exception as e:
        return f"ERROR: {e}"


# ── Tool definitions ───────────────────────────────────────────────────────

GITHUB_TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "gh_list_repos",
            "description": "List GitHub repositories for a user or organization.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "GitHub username or org name"}
                },
                "required": ["owner"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gh_list_issues",
            "description": "List GitHub issues for a repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo":  {"type": "string", "description": "owner/repo"},
                    "state": {"type": "string", "enum": ["open", "closed", "all"], "default": "open"},
                },
                "required": ["repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gh_create_issue",
            "description": "Create a new GitHub issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo":  {"type": "string", "description": "owner/repo"},
                    "title": {"type": "string"},
                    "body":  {"type": "string", "description": "Issue body (markdown)"},
                },
                "required": ["repo", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gh_list_prs",
            "description": "List pull requests for a GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo":  {"type": "string", "description": "owner/repo"},
                    "state": {"type": "string", "enum": ["open", "closed", "all"], "default": "open"},
                },
                "required": ["repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gh_get_file",
            "description": "Fetch the contents of a file from a GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "owner/repo"},
                    "path": {"type": "string", "description": "file path within repo"},
                    "ref":  {"type": "string", "description": "branch/tag/commit", "default": "main"},
                },
                "required": ["repo", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gh_search_code",
            "description": "Search code across GitHub repositories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "GitHub code search query"}
                },
                "required": ["query"],
            },
        },
    },
]

GITHUB_EXECUTORS: dict = {
    "gh_list_repos":   exec_gh_list_repos,
    "gh_list_issues":  exec_gh_list_issues,
    "gh_create_issue": exec_gh_create_issue,
    "gh_list_prs":     exec_gh_list_prs,
    "gh_get_file":     exec_gh_get_file,
    "gh_search_code":  exec_gh_search_code,
}
