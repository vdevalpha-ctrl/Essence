"""
cron_executor.py — Cron Job Executor for Essence
=================================================
Reads memory/cron_jobs.json and fires jobs whose cron schedule
matches the current minute.  Called on heartbeat.tick events from
the bus.

Cron syntax: 5-field standard (min hour day month weekday).
Supports: *, */n, n-m, comma-separated values.

Job record format (in cron_jobs.json):
  {
    "id":          "abc123",
    "name":        "Daily summary",
    "schedule":    "0 9 * * *",      # 09:00 every day
    "command":     "give me a daily summary",  # intent sent to kernel
    "enabled":     true,
    "last_run":    "",
    "last_status": "idle"
  }
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("essence.cron")


# ---------------------------------------------------------------------------
# Cron expression parser
# ---------------------------------------------------------------------------

def _cron_matches(expr: str, t: "time.struct_time") -> bool:
    """Return True if 5-field cron expression matches the given localtime."""
    fields = expr.strip().split()
    if len(fields) != 5:
        return False
    minute, hour, day, month, weekday = fields
    checks = [
        (minute,  t.tm_min),
        (hour,    t.tm_hour),
        (day,     t.tm_mday),
        (month,   t.tm_mon),
        (weekday, t.tm_wday),
    ]
    return all(_field_matches(f, v) for f, v in checks)


def _field_matches(field: str, value: int) -> bool:
    if field == "*":
        return True
    for part in field.split(","):
        if "/" in part:
            base, step_str = part.split("/", 1)
            step  = int(step_str)
            start = 0 if base == "*" else int(base.split("-")[0])
            if value >= start and (value - start) % step == 0:
                return True
        elif "-" in part:
            lo, hi = part.split("-", 1)
            if int(lo) <= value <= int(hi):
                return True
        else:
            try:
                if int(part) == value:
                    return True
            except ValueError:
                pass
    return False


# ---------------------------------------------------------------------------
# CronExecutor
# ---------------------------------------------------------------------------

class CronExecutor:
    """
    Fires cron jobs whose schedule matches the current minute.
    Call `on_tick()` once per heartbeat.tick event.
    """

    def __init__(self, workspace: Path, bus: Any) -> None:
        self._ws   = workspace
        self._bus  = bus
        self._path = workspace / "memory" / "cron_jobs.json"
        self._last_minute: int = -1

    # ── Public API ─────────────────────────────────────────────────────

    async def on_tick(self, ts: float | None = None) -> None:
        """Check and fire matching cron jobs for the current minute."""
        now = ts or time.time()
        t   = time.localtime(now)
        cur_minute = t.tm_hour * 60 + t.tm_min

        if cur_minute == self._last_minute:
            return          # already fired this minute
        self._last_minute = cur_minute

        jobs    = self._load_jobs()
        changed = False

        for job in jobs:
            if not job.get("enabled", True):
                continue
            sched = job.get("schedule", "")
            if not sched:
                continue
            try:
                matches = _cron_matches(sched, t)
            except Exception as exc:
                log.debug("cron: invalid schedule %r: %s", sched, exc)
                continue
            if not matches:
                continue

            job_id   = job.get("id", "?")
            job_name = job.get("name", "unnamed")
            command  = job.get("command", "")
            log.info("cron: firing job [%s] %s", job_id, job_name)

            try:
                from server.event_bus import Envelope
                self._bus.publish_sync(Envelope(
                    topic="cron.tick",
                    data={
                        "job_id":   job_id,
                        "job_name": job_name,
                        "command":  command,
                        "ts":       now,
                    },
                ))
                job["last_run"]    = time.strftime("%Y-%m-%d %H:%M", t)
                job["last_status"] = "fired"
                changed = True
            except Exception as exc:
                log.warning("cron: job [%s] fire failed: %s", job_id, exc)
                job["last_status"] = f"error: {exc!s:.80}"
                changed = True

        if changed:
            self._save_jobs(jobs)

    # ── Persistence ────────────────────────────────────────────────────

    def _load_jobs(self) -> list[dict]:
        try:
            if self._path.exists():
                return json.loads(self._path.read_text("utf-8"))
        except Exception as exc:
            log.debug("cron: load failed: %s", exc)
        return []

    def _save_jobs(self, jobs: list[dict]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
        except Exception as exc:
            log.debug("cron: save failed: %s", exc)
