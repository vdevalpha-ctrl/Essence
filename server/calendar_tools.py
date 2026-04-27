"""
calendar_tools.py — Calendar Awareness Tools for Essence
=========================================================
Provides date/time and calendar tools the LLM can call natively.

Tools:
  get_datetime    — current date, time, timezone, day-of-week
  list_events     — list events from a local .ics file (if present)
  add_event       — append an event to the local calendar file
  date_math       — add/subtract days/weeks from a date

No external API keys required for basic date/time tools.
For Google Calendar or Outlook: set GOOGLE_CALENDAR_ID or
OUTLOOK_CALENDAR_URL environment variables (future extension point).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("essence.calendar")


# ── Tool executors ─────────────────────────────────────────────────────────

async def exec_get_datetime(args: dict) -> str:
    """Return current date, time, weekday, and timezone."""
    tz_name = time.tzname[0]
    now     = datetime.now()
    utc_now = datetime.now(timezone.utc)
    return (
        f"Date:     {now.strftime('%Y-%m-%d')} ({now.strftime('%A')})\n"
        f"Time:     {now.strftime('%H:%M:%S')} {tz_name}\n"
        f"UTC:      {utc_now.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        f"Week:     {now.isocalendar()[1]} of {now.year}\n"
        f"Quarter:  Q{(now.month - 1) // 3 + 1} {now.year}"
    )


async def exec_date_math(args: dict) -> str:
    """Add or subtract days/weeks from a date."""
    date_str = args.get("date", datetime.now().strftime("%Y-%m-%d"))
    delta    = int(args.get("delta", 0))
    unit     = args.get("unit", "days")  # days | weeks

    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return f"ERROR: invalid date '{date_str}' — use YYYY-MM-DD format"

    if unit == "weeks":
        result = dt + timedelta(weeks=delta)
    else:
        result = dt + timedelta(days=delta)

    return (
        f"Input:   {dt.strftime('%Y-%m-%d (%A)')}\n"
        f"Delta:   {'+' if delta >= 0 else ''}{delta} {unit}\n"
        f"Result:  {result.strftime('%Y-%m-%d (%A)')}"
    )


async def exec_list_events(args: dict) -> str:
    """
    List events from a local calendar file (ICS format).
    Falls back to memory/calendar/events.json if no .ics is available.
    """
    try:
        from server.tools_engine import workspace as _ws
        ws = _ws()
    except Exception:
        ws = Path.cwd()

    # Try JSON store first
    json_path = ws / "memory" / "calendar" / "events.json"
    if json_path.exists():
        try:
            events = json.loads(json_path.read_text("utf-8"))
            # Filter by date range
            after  = args.get("after",  datetime.now().strftime("%Y-%m-%d"))
            before = args.get("before", (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"))
            filtered = [e for e in events
                        if after <= e.get("date", "9999") <= before]
            if not filtered:
                return f"No events between {after} and {before}"
            lines = [f"{e.get('date','')} {e.get('time',''):<8} {e.get('title','')}"
                     for e in sorted(filtered, key=lambda x: x.get("date", ""))]
            return "\n".join(lines)
        except Exception as exc:
            log.debug("calendar: JSON load failed: %s", exc)

    # Try ICS file
    ics_candidates = list((ws / "memory" / "calendar").glob("*.ics")) if (ws / "memory" / "calendar").exists() else []
    home_ics = Path.home() / "calendar.ics"
    if home_ics.exists():
        ics_candidates.insert(0, home_ics)

    if not ics_candidates:
        return "No calendar file found. Add events with add_event or place a .ics file in memory/calendar/"

    # Basic ICS parser (no external deps)
    ics_path = ics_candidates[0]
    try:
        text   = ics_path.read_text(encoding="utf-8", errors="replace")
        events = _parse_ics(text)
        after  = args.get("after",  datetime.now().strftime("%Y-%m-%d"))
        before = args.get("before", (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"))
        filtered = [e for e in events if after <= e.get("date", "9999") <= before]
        if not filtered:
            return f"No events between {after} and {before}"
        lines = [f"{e.get('date','')} {e.get('time','    ')[:5]}  {e.get('title','')}"
                 for e in sorted(filtered, key=lambda x: x.get("date", ""))]
        return "\n".join(lines)
    except Exception as exc:
        return f"ERROR: {exc}"


async def exec_add_event(args: dict) -> str:
    """Add an event to the local JSON calendar store."""
    title = args.get("title", "").strip()
    date  = args.get("date", "").strip()
    if not title or not date:
        return "ERROR: title and date are required"

    # Validate date
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return f"ERROR: invalid date '{date}' — use YYYY-MM-DD"

    try:
        from server.tools_engine import workspace as _ws
        ws = _ws()
    except Exception:
        ws = Path.cwd()

    cal_dir = ws / "memory" / "calendar"
    cal_dir.mkdir(parents=True, exist_ok=True)
    json_path = cal_dir / "events.json"

    events: list[dict] = []
    if json_path.exists():
        try:
            events = json.loads(json_path.read_text("utf-8"))
        except Exception:
            pass

    event = {
        "title":    title,
        "date":     date,
        "time":     args.get("time", ""),
        "duration": args.get("duration", ""),
        "notes":    args.get("notes", ""),
        "added_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    events.append(event)
    json_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
    return f"OK: added event '{title}' on {date}"


# ── ICS parser helper ──────────────────────────────────────────────────────

def _parse_ics(text: str) -> list[dict]:
    """Minimal ICS parser — extracts VEVENT blocks."""
    events = []
    in_event = False
    current: dict = {}
    for line in text.splitlines():
        line = line.rstrip()
        if line == "BEGIN:VEVENT":
            in_event = True
            current = {}
        elif line == "END:VEVENT":
            if current:
                events.append(current)
            in_event = False
        elif in_event:
            if line.startswith("SUMMARY:"):
                current["title"] = line[8:]
            elif line.startswith("DTSTART"):
                # DTSTART;TZID=...:20241015T090000  or  DTSTART:20241015
                raw = line.split(":")[-1]
                if len(raw) >= 8:
                    current["date"] = f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"
                if len(raw) > 8:
                    current["time"] = f"{raw[9:11]}:{raw[11:13]}"
            elif line.startswith("DESCRIPTION:"):
                current["notes"] = line[12:]
    return events


# ── Tool definitions ───────────────────────────────────────────────────────

CALENDAR_TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Get the current date, time, weekday, week number, and timezone.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "date_math",
            "description": "Add or subtract days or weeks from a date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date":  {"type": "string", "description": "Start date YYYY-MM-DD (default: today)"},
                    "delta": {"type": "integer", "description": "Number to add (negative to subtract)"},
                    "unit":  {"type": "string",  "enum": ["days", "weeks"], "default": "days"},
                },
                "required": ["delta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_events",
            "description": "List calendar events from the local calendar store.",
            "parameters": {
                "type": "object",
                "properties": {
                    "after":  {"type": "string", "description": "Start date YYYY-MM-DD (default: today)"},
                    "before": {"type": "string", "description": "End date YYYY-MM-DD (default: +30 days)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_event",
            "description": "Add an event to the local calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title":    {"type": "string", "description": "Event title"},
                    "date":     {"type": "string", "description": "Date YYYY-MM-DD"},
                    "time":     {"type": "string", "description": "Time HH:MM (optional)"},
                    "duration": {"type": "string", "description": "Duration e.g. '1h', '30m' (optional)"},
                    "notes":    {"type": "string", "description": "Additional notes (optional)"},
                },
                "required": ["title", "date"],
            },
        },
    },
]

CALENDAR_EXECUTORS: dict = {
    "get_datetime": exec_get_datetime,
    "date_math":    exec_date_math,
    "list_events":  exec_list_events,
    "add_event":    exec_add_event,
}
