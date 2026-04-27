"""
trigger_network.py — Real-Time Event Trigger Network for Essence
================================================================
Monitors external event sources and fires `trigger.fired` bus events
when conditions are met.

Supported watchers (all optional / gracefully degraded):
  • Filesystem watcher  — watches workspace dirs for file changes
  • Webhook receiver    — lightweight HTTP endpoint (/api/trigger)
  • RSS/Atom feed check — polls configured feeds for new items
  • IMAP inbox check    — polls for new emails

Configuration:
  ESSENCE_WATCH_DIRS    — comma-separated workspace-relative paths
  ESSENCE_WEBHOOK_PORT  — port for webhook receiver (default: 7861)
  ESSENCE_RSS_FEEDS     — comma-separated RSS/Atom URLs to poll
  ESSENCE_IMAP_HOST     — IMAP server host
  ESSENCE_IMAP_USER     — IMAP username
  ESSENCE_IMAP_PASSWORD — IMAP password
  ESSENCE_IMAP_FOLDER   — folder to watch (default: INBOX)

All watchers run as asyncio background tasks and shut down cleanly.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("essence.trigger_network")


class TriggerNetwork:
    """
    Manages all trigger sources and converts them to bus events.
    """

    def __init__(self, workspace: Path, bus: Any) -> None:
        self._ws    = workspace
        self._bus   = bus
        self._tasks: list[asyncio.Task] = []
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        loop = asyncio.get_event_loop()

        # Filesystem watcher
        watch_dirs = os.environ.get("ESSENCE_WATCH_DIRS", "")
        if watch_dirs:
            for d in watch_dirs.split(","):
                d = d.strip()
                path = self._ws / d
                if path.exists():
                    t = loop.create_task(
                        self._fs_watcher(path), name=f"trigger-fs-{d}"
                    )
                    self._tasks.append(t)

        # RSS feed poller
        rss_feeds = os.environ.get("ESSENCE_RSS_FEEDS", "")
        if rss_feeds:
            urls = [u.strip() for u in rss_feeds.split(",") if u.strip()]
            if urls:
                t = loop.create_task(
                    self._rss_poller(urls), name="trigger-rss"
                )
                self._tasks.append(t)

        # IMAP inbox poller
        if os.environ.get("ESSENCE_IMAP_HOST"):
            t = loop.create_task(self._imap_poller(), name="trigger-imap")
            self._tasks.append(t)

        if self._tasks:
            log.info("TriggerNetwork: started %d watcher(s)", len(self._tasks))

    def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            if not t.done():
                t.cancel()
        self._tasks.clear()

    def fire(self, source: str, data: dict, skill_id: str = "") -> None:
        """Publish a trigger.fired event to the bus."""
        try:
            from server.event_bus import Envelope
            self._bus.publish_sync(Envelope(
                topic="trigger.fired",
                data={
                    "source":   source,
                    "skill_id": skill_id,
                    "intent":   data.get("intent", ""),
                    "payload":  data,
                    "ts":       time.time(),
                },
            ))
            log.info("trigger: fired from %s — %s", source, str(data)[:80])
        except Exception as exc:
            log.warning("trigger: fire failed: %s", exc)

    # ── Filesystem watcher ──────────────────────────────────────────

    async def _fs_watcher(self, watch_path: Path) -> None:
        """Poll a directory for changes every 5s."""
        log.info("trigger: watching fs: %s", watch_path)
        snapshots: dict[str, float] = {}

        # Initial snapshot
        for f in watch_path.rglob("*"):
            if f.is_file():
                snapshots[str(f)] = f.stat().st_mtime

        while self._running:
            await asyncio.sleep(5)
            try:
                for f in watch_path.rglob("*"):
                    if not f.is_file():
                        continue
                    key = str(f)
                    mtime = f.stat().st_mtime
                    prev  = snapshots.get(key, 0)
                    if mtime > prev:
                        is_new = key not in snapshots
                        snapshots[key] = mtime
                        rel = f.relative_to(self._ws)
                        self.fire("filesystem", {
                            "event":   "created" if is_new else "modified",
                            "path":    str(rel),
                            "intent":  f"File {'created' if is_new else 'modified'}: {rel}",
                        })
            except Exception as exc:
                log.debug("trigger: fs_watcher error: %s", exc)

    # ── RSS/Atom feed poller ────────────────────────────────────────

    async def _rss_poller(self, urls: list[str], interval: int = 300) -> None:
        """Poll RSS/Atom feeds every `interval` seconds for new items."""
        seen_ids: set[str] = set()

        # Prime with current items (don't fire on startup)
        for url in urls:
            for entry_id in await self._fetch_feed_ids(url):
                seen_ids.add(entry_id)

        while self._running:
            await asyncio.sleep(interval)
            for url in urls:
                try:
                    new_entries = []
                    for entry_id, entry_title in await self._fetch_feed_entries(url):
                        if entry_id not in seen_ids:
                            seen_ids.add(entry_id)
                            new_entries.append(entry_title)
                    for title in new_entries:
                        self.fire("rss", {
                            "feed":   url,
                            "title":  title,
                            "intent": f"New RSS item: {title}",
                        })
                except Exception as exc:
                    log.debug("trigger: rss_poller error for %s: %s", url, exc)

    async def _fetch_feed_ids(self, url: str) -> list[str]:
        entries = await self._fetch_feed_entries(url)
        return [eid for eid, _ in entries]

    async def _fetch_feed_entries(self, url: str) -> list[tuple[str, str]]:
        """Return list of (id, title) from an RSS/Atom feed."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(url, follow_redirects=True,
                                headers={"User-Agent": "Essence/1"})
                text = r.text

            # Minimal XML extraction (no feedparser dependency)
            import re
            entries = []
            for block in re.findall(r'<item>(.*?)</item>', text, re.DOTALL) + \
                         re.findall(r'<entry>(.*?)</entry>', text, re.DOTALL):
                eid = re.search(r'<guid[^>]*>(.*?)</guid>', block)
                if not eid:
                    eid = re.search(r'<id>(.*?)</id>', block)
                title_m = re.search(r'<title[^>]*>(.*?)</title>', block)
                eid_val   = eid.group(1) if eid else ""
                title_val = title_m.group(1) if title_m else ""
                if eid_val:
                    entries.append((eid_val, title_val))
            return entries
        except Exception:
            return []

    # ── IMAP poller ─────────────────────────────────────────────────

    async def _imap_poller(self, interval: int = 120) -> None:
        """Poll IMAP inbox for new messages every `interval` seconds."""
        host     = os.environ.get("ESSENCE_IMAP_HOST", "")
        user     = os.environ.get("ESSENCE_IMAP_USER", "")
        password = os.environ.get("ESSENCE_IMAP_PASSWORD", "")
        folder   = os.environ.get("ESSENCE_IMAP_FOLDER", "INBOX")

        if not (host and user and password):
            return

        log.info("trigger: IMAP polling %s@%s/%s", user, host, folder)
        seen_uids: set[str] = set()

        while self._running:
            await asyncio.sleep(interval)
            try:
                loop = asyncio.get_event_loop()
                new_msgs = await loop.run_in_executor(
                    None, self._imap_check, host, user, password, folder, seen_uids
                )
                for msg in new_msgs:
                    seen_uids.add(msg["uid"])
                    self.fire("imap", {
                        "from":    msg.get("from", ""),
                        "subject": msg.get("subject", ""),
                        "intent":  f"New email from {msg.get('from','?')}: {msg.get('subject','')}",
                    })
            except Exception as exc:
                log.debug("trigger: imap_poller error: %s", exc)

    def _imap_check(
        self, host: str, user: str, password: str, folder: str, seen: set
    ) -> list[dict]:
        import imaplib
        import email as _email
        msgs = []
        try:
            with imaplib.IMAP4_SSL(host) as imap:
                imap.login(user, password)
                imap.select(folder, readonly=True)
                _, data = imap.search(None, "UNSEEN")
                uids = data[0].split()
                for uid in uids[-20:]:   # cap to last 20 unseen
                    uid_str = uid.decode()
                    if uid_str in seen:
                        continue
                    _, msg_data = imap.fetch(uid, "(RFC822.SIZE BODY[HEADER.FIELDS (FROM SUBJECT)])")
                    for part in msg_data:
                        if isinstance(part, tuple):
                            msg = _email.message_from_bytes(part[1])
                            msgs.append({
                                "uid":     uid_str,
                                "from":    msg.get("From", ""),
                                "subject": msg.get("Subject", ""),
                            })
        except Exception as exc:
            log.debug("trigger: imap_check failed: %s", exc)
        return msgs


# ── Module-level singleton ─────────────────────────────────────────────────

_trigger_net: Optional[TriggerNetwork] = None


def init_trigger_network(workspace: Path, bus: Any) -> TriggerNetwork:
    global _trigger_net
    _trigger_net = TriggerNetwork(workspace, bus)
    return _trigger_net


def get_trigger_network() -> Optional[TriggerNetwork]:
    return _trigger_net
