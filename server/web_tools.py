"""
web_tools.py — Essence Web Tools
=================================
Provides web search, page fetching, and browser automation to Essence.

Tools available:
  web_search(query, max_results)   — DuckDuckGo search (no API key needed)
  fetch_page(url, extract)         — HTTP fetch + clean text via BeautifulSoup
  browser_goto(url)                — Playwright browser navigation
  browser_click(selector)          — Click element
  browser_fill(selector, value)    — Fill input field
  browser_eval(js_code)            — Execute JavaScript
  browser_screenshot()             — Capture screenshot as base64
  browser_get_text()               — Get visible page text

All browser tools gracefully return an error dict if playwright is not installed.
Install: pip install playwright && playwright install chromium
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Any

log = logging.getLogger("essence.web")

# ── Optional imports ─────────────────────────────────────────────────────

try:
    import httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    _BS4 = False

try:
    from duckduckgo_search import DDGS
    _DDG = True
except ImportError:
    _DDG = False

try:
    from playwright.async_api import async_playwright, Browser, Page
    _PW = True
except ImportError:
    _PW = False

# ── Shared browser state (lazy init) ────────────────────────────────────

_pw_instance = None
_pw_browser: "Browser | None" = None
_pw_page: "Page | None" = None


async def _get_page() -> "Page":
    global _pw_instance, _pw_browser, _pw_page
    if not _PW:
        raise RuntimeError(
            "playwright not installed. Run: pip install playwright && playwright install chromium"
        )
    if _pw_page is None:
        _pw_instance = await async_playwright().start()
        _pw_browser  = await _pw_instance.chromium.launch(headless=True)
        _pw_page     = await _pw_browser.new_page()
    return _pw_page


async def _close_browser() -> None:
    global _pw_instance, _pw_browser, _pw_page
    if _pw_page:
        await _pw_page.close()
    if _pw_browser:
        await _pw_browser.close()
    if _pw_instance:
        await _pw_instance.stop()
    _pw_page = _pw_browser = _pw_instance = None


# ── Tool 1: web_search ───────────────────────────────────────────────────

async def _perplexity_search(query: str, max_results: int) -> list[dict] | None:
    """
    Search via Perplexity Sonar API.  Returns results or None if unavailable.
    Requires PERPLEXITY_API_KEY environment variable.
    """
    import os
    api_key = os.environ.get("PERPLEXITY_API_KEY", "")
    if not api_key or not _HTTPX:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "sonar",
                    "messages": [{"role": "user", "content": query}],
                    "max_tokens": 1024,
                    "return_citations": True,
                },
            )
            if r.status_code != 200:
                return None
            data    = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            sources = data.get("citations", [])
            results = [{"title": "Perplexity answer", "url": "", "snippet": content[:600]}]
            for src in sources[:max_results - 1]:
                results.append({"title": src.get("title", ""), "url": src.get("url", ""), "snippet": ""})
            return results
    except Exception as exc:
        log.debug("Perplexity search failed: %s", exc)
        return None


async def web_search(query: str, max_results: int = 6) -> list[dict]:
    """
    Search the web.  Provider priority:
      1. Perplexity Sonar (if PERPLEXITY_API_KEY is set)
      2. DuckDuckGo (duckduckgo_search package)
      3. DuckDuckGo HTML fallback (httpx scrape)
    Returns list of {title, url, snippet}.
    """
    # Priority 1: Perplexity
    px_results = await _perplexity_search(query, max_results)
    if px_results:
        return px_results

    if _DDG:
        try:
            with DDGS() as ddg:
                results = list(ddg.text(query, max_results=max_results))
            return [
                {"title": r.get("title",""), "url": r.get("href",""), "snippet": r.get("body","")}
                for r in results
            ]
        except Exception as e:
            log.warning("DDG search failed: %s — falling back to HTML scrape", e)

    # Fallback: scrape DuckDuckGo HTML
    if not _HTTPX:
        return [{"error": "httpx not installed. Run: pip install httpx"}]
    try:
        url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        async with httpx.AsyncClient(timeout=10, follow_redirects=True,
                                     headers={"User-Agent": "Mozilla/5.0 Essence/1"}) as c:
            r = await c.get(url)
        if not _BS4:
            return [{"error": "beautifulsoup4 not installed for HTML scrape"}]
        soup = BeautifulSoup(r.text, "html.parser")
        out  = []
        for result in soup.select(".result")[:max_results]:
            title_el   = result.select_one(".result__title")
            snippet_el = result.select_one(".result__snippet")
            link_el    = result.select_one(".result__url")
            out.append({
                "title":   title_el.get_text(strip=True) if title_el else "",
                "url":     link_el.get_text(strip=True) if link_el else "",
                "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
            })
        return out
    except Exception as e:
        return [{"error": str(e)}]


# ── Tool 2: fetch_page ───────────────────────────────────────────────────

async def fetch_page(url: str, extract: str = "text") -> dict:
    """
    Fetch a URL and return clean content.
    extract: "text" | "markdown" | "links" | "structured"
    """
    if not _HTTPX:
        return {"error": "httpx not installed. Run: pip install httpx"}

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(
            timeout=15, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 Essence/1"}
        ) as c:
            r = await c.get(url)
        r.raise_for_status()
        fetch_ms = round((time.monotonic() - t0) * 1000)
    except Exception as e:
        return {"url": url, "error": str(e)}

    if not _BS4:
        return {"url": url, "content": r.text[:4000], "fetch_time_ms": fetch_ms}

    soup = BeautifulSoup(r.text, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "iframe", "noscript",
                     "aside", "header", "form", "button"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""

    if extract == "links":
        links = [{"text": a.get_text(strip=True), "href": a.get("href", "")}
                 for a in soup.find_all("a", href=True)]
        return {"url": url, "title": title, "links": links[:80], "fetch_time_ms": fetch_ms}

    if extract == "structured":
        headings = [h.get_text(strip=True) for h in soup.find_all(["h1","h2","h3"])]
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]
        return {
            "url":        url,
            "title":      title,
            "headings":   headings[:20],
            "paragraphs": paragraphs[:15],
            "fetch_time_ms": fetch_ms,
        }

    # Default: clean text
    text = soup.get_text(separator="\n", strip=True)
    # Collapse blank lines
    lines = [l for l in text.splitlines() if l.strip()]
    text  = "\n".join(lines)

    if extract == "markdown":
        # Very basic markdown conversion
        for h in soup.find_all(["h1","h2","h3"]):
            level = int(h.name[1])
            h.replace_with("#" * level + " " + h.get_text(strip=True) + "\n")
        text = soup.get_text(separator="\n", strip=True)

    return {
        "url":           url,
        "title":         title,
        "content":       text[:8000],
        "length":        len(text),
        "fetch_time_ms": fetch_ms,
    }


# ── Browser tools ────────────────────────────────────────────────────────

async def browser_goto(url: str) -> dict:
    try:
        page = await _get_page()
        await page.goto(url, timeout=15000)
        return {"ok": True, "url": page.url, "title": await page.title()}
    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)[:200]}


async def browser_click(selector: str) -> dict:
    try:
        page = await _get_page()
        await page.click(selector, timeout=5000)
        return {"ok": True, "selector": selector}
    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)[:200]}


async def browser_fill(selector: str, value: str) -> dict:
    try:
        page = await _get_page()
        await page.fill(selector, value, timeout=5000)
        return {"ok": True, "selector": selector, "value": value}
    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)[:200]}


async def browser_eval(js_code: str) -> dict:
    try:
        page   = await _get_page()
        result = await page.evaluate(js_code)
        return {"ok": True, "result": result}
    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)[:200]}


async def browser_screenshot() -> dict:
    try:
        page  = await _get_page()
        data  = await page.screenshot(type="png")
        b64   = base64.b64encode(data).decode()
        return {"ok": True, "image_base64": b64, "mime": "image/png"}
    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)[:200]}


async def browser_get_text() -> dict:
    try:
        page = await _get_page()
        text = await page.inner_text("body")
        return {"ok": True, "text": text[:8000], "length": len(text)}
    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)[:200]}


# ── Unified executor ─────────────────────────────────────────────────────

async def execute_web_tool(name: str, args: dict) -> dict:
    """Dispatch a web tool call by name."""
    if name == "web_search":
        return {"results": await web_search(args.get("query",""), args.get("max_results", 6))}
    if name == "fetch_page":
        return await fetch_page(args.get("url",""), args.get("extract", "text"))
    if name == "browser_goto":
        return await browser_goto(args.get("url",""))
    if name == "browser_click":
        return await browser_click(args.get("selector",""))
    if name == "browser_fill":
        return await browser_fill(args.get("selector",""), args.get("value",""))
    if name == "browser_eval":
        return await browser_eval(args.get("js_code",""))
    if name == "browser_screenshot":
        return await browser_screenshot()
    if name == "browser_get_text":
        return await browser_get_text()
    return {"error": f"Unknown web tool: {name}"}


# ── Ollama function-calling schemas ──────────────────────────────────────

WEB_TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name":        "web_search",
            "description": "Search the web for current information. Returns titles, URLs, and snippets.",
            "parameters":  {
                "type":       "object",
                "properties": {
                    "query":       {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 6, "description": "Number of results to return"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "fetch_page",
            "description": "Fetch a URL and extract its content as clean text, links, or structured data.",
            "parameters":  {
                "type":       "object",
                "properties": {
                    "url":     {"type": "string", "description": "URL to fetch"},
                    "extract": {
                        "type":   "string",
                        "enum":   ["text", "markdown", "links", "structured"],
                        "default": "text",
                        "description": "Extraction format",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "browser_goto",
            "description": "Navigate the browser to a URL. Use before other browser_* tools.",
            "parameters":  {
                "type":       "object",
                "properties": {"url": {"type": "string", "description": "URL to navigate to"}},
                "required":   ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "browser_click",
            "description": "Click an element in the browser using a CSS selector.",
            "parameters":  {
                "type":       "object",
                "properties": {"selector": {"type": "string", "description": "CSS selector"}},
                "required":   ["selector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "browser_fill",
            "description": "Fill an input field in the browser.",
            "parameters":  {
                "type":       "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector for the input"},
                    "value":    {"type": "string", "description": "Value to fill"},
                },
                "required": ["selector", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "browser_eval",
            "description": "Execute JavaScript in the browser and return the result.",
            "parameters":  {
                "type":       "object",
                "properties": {"js_code": {"type": "string", "description": "JavaScript to execute"}},
                "required":   ["js_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "browser_screenshot",
            "description": "Take a screenshot of the current browser page. Returns base64 PNG.",
            "parameters":  {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "browser_get_text",
            "description": "Get all visible text from the current browser page.",
            "parameters":  {"type": "object", "properties": {}},
        },
    },
]


# ── FastAPI router ────────────────────────────────────────────────────────

try:
    from fastapi import APIRouter, Request

    web_router = APIRouter(prefix="/api/web", tags=["web"])

    @web_router.post("/search")
    async def api_web_search(req: Request):
        body = await req.json()
        return {"results": await web_search(body.get("query",""), body.get("max_results", 6))}

    @web_router.post("/fetch")
    async def api_fetch_page(req: Request):
        body = await req.json()
        return await fetch_page(body.get("url",""), body.get("extract","text"))

    @web_router.post("/browser")
    async def api_browser(req: Request):
        body   = await req.json()
        action = body.get("action","")
        return await execute_web_tool(
            f"browser_{action}" if not action.startswith("browser_") else action,
            body,
        )

except ImportError:
    web_router = None  # type: ignore
