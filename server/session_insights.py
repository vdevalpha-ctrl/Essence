"""
session_insights.py — Session Insights Engine
===============================================
Analyses Essence's event_log and episodic_memory databases to produce
comprehensive usage reports: token consumption, cost estimates, tool usage
patterns, model/provider breakdowns, and activity trends.

Exposed via the /insights [days] TUI command.

Usage
-----
  from server.session_insights import InsightsEngine, get_insights_engine
  engine = get_insights_engine()
  report = engine.generate(days=30)
  print(engine.format_terminal(report))
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("essence.insights")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    """Format seconds into a compact human-readable string."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s}s" if s else f"{m}m"
    h, rem = divmod(seconds, 3600)
    m = rem // 60
    return f"{h}h {m}m" if m else f"{h}h"


def _bar(value: int, peak: int, width: int = 16) -> str:
    if peak == 0 or value == 0:
        return ""
    filled = max(1, int(value / peak * width))
    return "█" * filled


# ---------------------------------------------------------------------------
# InsightsEngine
# ---------------------------------------------------------------------------

class InsightsEngine:
    """
    Derives usage insights from Essence's event_log and episodic_memory SQLite
    databases.  Both paths are resolved from the workspace root at construction.
    """

    def __init__(self, workspace: Path) -> None:
        self._ws             = workspace
        self._event_log_path = workspace / "data" / "event_log.db"
        self._episodic_path  = workspace / "data" / "episodic.db"

    # ── Internal DB helpers ────────────────────────────────────────────

    def _el_conn(self) -> Optional[sqlite3.Connection]:
        if not self._event_log_path.exists():
            return None
        conn = sqlite3.connect(str(self._event_log_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ep_conn(self) -> Optional[sqlite3.Connection]:
        if not self._episodic_path.exists():
            return None
        conn = sqlite3.connect(str(self._episodic_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Data gathering from event_log ─────────────────────────────────

    def _get_response_events(self, cutoff_ms: int) -> List[dict]:
        """Collect done=True user.response events (token counts + provider/model)."""
        conn = self._el_conn()
        if conn is None:
            return []
        try:
            rows = conn.execute(
                """SELECT data_json, created_at FROM event_log
                   WHERE topic = 'user.response'
                     AND created_at >= ?
                   ORDER BY created_at""",
                (cutoff_ms,),
            ).fetchall()
            results = []
            for row in rows:
                try:
                    d = json.loads(row["data_json"])
                    if d.get("done") and (d.get("tokens_in") or d.get("tokens_out")):
                        results.append({
                            "tokens_in":  int(d.get("tokens_in", 0)),
                            "tokens_out": int(d.get("tokens_out", 0)),
                            "total_in":   int(d.get("total_in", 0)),
                            "total_out":  int(d.get("total_out", 0)),
                            "provider":   d.get("provider", "unknown"),
                            "model":      d.get("model", "unknown"),
                            "created_at": row["created_at"],
                        })
                except Exception:
                    continue
            return results
        finally:
            conn.close()

    def _get_request_events(self, cutoff_ms: int) -> List[dict]:
        """Collect user.request events (timestamps, session detection)."""
        conn = self._el_conn()
        if conn is None:
            return []
        try:
            rows = conn.execute(
                """SELECT data_json, created_at FROM event_log
                   WHERE topic = 'user.request'
                     AND created_at >= ?
                   ORDER BY created_at""",
                (cutoff_ms,),
            ).fetchall()
            results = []
            for row in rows:
                try:
                    d = json.loads(row["data_json"])
                    results.append({
                        "session_id": d.get("session_id", ""),
                        "created_at": row["created_at"],
                    })
                except Exception:
                    continue
            return results
        finally:
            conn.close()

    def _get_skill_events(self, cutoff_ms: int) -> List[dict]:
        """Collect skill.execute events for tool usage breakdown."""
        conn = self._el_conn()
        if conn is None:
            return []
        try:
            rows = conn.execute(
                """SELECT data_json, created_at FROM event_log
                   WHERE topic = 'skill.execute'
                     AND created_at >= ?
                   ORDER BY created_at""",
                (cutoff_ms,),
            ).fetchall()
            results = []
            for row in rows:
                try:
                    d = json.loads(row["data_json"])
                    skill_id = d.get("skill_id", "")
                    if skill_id:
                        results.append({"skill_id": skill_id, "created_at": row["created_at"]})
                except Exception:
                    continue
            return results
        finally:
            conn.close()

    def _get_plan_events(self, cutoff_ms: int) -> List[dict]:
        """Collect governance.audit events for plan success/failure tracking."""
        conn = self._el_conn()
        if conn is None:
            return []
        try:
            rows = conn.execute(
                """SELECT data_json, created_at FROM event_log
                   WHERE topic = 'governance.audit'
                     AND created_at >= ?
                   ORDER BY created_at""",
                (cutoff_ms,),
            ).fetchall()
            results = []
            for row in rows:
                try:
                    d = json.loads(row["data_json"])
                    results.append({
                        "success":    bool(d.get("success", False)),
                        "plan_source": d.get("plan_source", ""),
                        "step_count": int(d.get("step_count", 0)),
                        "created_at": row["created_at"],
                    })
                except Exception:
                    continue
            return results
        finally:
            conn.close()

    def _get_session_summaries(self, cutoff_ms: int) -> List[dict]:
        """Collect cross-session summary records from episodic memory."""
        conn = self._ep_conn()
        if conn is None:
            return []
        cutoff_s = cutoff_ms / 1000
        try:
            rows = conn.execute(
                """SELECT session_id, turn_count, started_at, ended_at
                   FROM session_summaries
                   WHERE started_at >= ?
                   ORDER BY started_at DESC""",
                (cutoff_s,),
            ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            return []
        finally:
            conn.close()

    def _get_current_session_turns(self, cutoff_ms: int) -> int:
        """Count user turns in current session from episodic memory."""
        conn = self._ep_conn()
        if conn is None:
            return 0
        cutoff_s = cutoff_ms / 1000
        try:
            row = conn.execute(
                """SELECT COUNT(*) FROM episodes
                   WHERE role = 'user' AND entry_type = 'turn'
                     AND ts >= ?""",
                (cutoff_s * 1000,),  # ts is milliseconds
            ).fetchone()
            return row[0] if row else 0
        except Exception:
            return 0
        finally:
            conn.close()

    # ── Computation ────────────────────────────────────────────────────

    def _compute_overview(
        self,
        requests: List[dict],
        responses: List[dict],
        plans: List[dict],
        session_summaries: List[dict],
        current_turns: int,
    ) -> Dict[str, Any]:
        total_in  = sum(r["tokens_in"]  for r in responses)
        total_out = sum(r["tokens_out"] for r in responses)
        total_tok = total_in + total_out
        n_calls   = len(responses)
        n_requests = len(requests)

        # Session counting: from session_summaries + current (incomplete) session
        past_sessions   = len(session_summaries)
        current_session = 1 if current_turns > 0 else 0
        total_sessions  = past_sessions + current_session

        # Plan stats
        n_plans     = len(plans)
        n_plans_ok  = sum(1 for p in plans if p["success"])
        plan_pct    = (n_plans_ok / n_plans * 100) if n_plans else 0.0

        # Unique skill calls
        return {
            "total_sessions":  total_sessions,
            "total_requests":  n_requests,
            "total_llm_calls": n_calls,
            "total_in":        total_in,
            "total_out":       total_out,
            "total_tokens":    total_tok,
            "avg_in_per_call": total_in  / max(n_calls, 1),
            "avg_out_per_call": total_out / max(n_calls, 1),
            "total_plans":     n_plans,
            "plans_succeeded": n_plans_ok,
            "plan_success_pct": plan_pct,
        }

    def _compute_model_breakdown(self, responses: List[dict]) -> List[Dict]:
        model_data: Dict[str, dict] = defaultdict(lambda: {
            "calls": 0, "tokens_in": 0, "tokens_out": 0,
        })
        for r in responses:
            key = f"{r['provider']}/{r['model']}"
            d   = model_data[key]
            d["calls"]      += 1
            d["tokens_in"]  += r["tokens_in"]
            d["tokens_out"] += r["tokens_out"]

        # Estimate cost using usage_pricing table
        result = []
        try:
            from server.usage_pricing import CanonicalUsage, estimate_cost, format_cost
            from decimal import Decimal
            for key, d in model_data.items():
                provider, model = key.split("/", 1) if "/" in key else ("", key)
                usage  = CanonicalUsage(
                    input_tokens=d["tokens_in"],
                    output_tokens=d["tokens_out"],
                )
                cost   = estimate_cost(usage, model=model, provider=provider)
                result.append({
                    "model":      key,
                    "calls":      d["calls"],
                    "tokens_in":  d["tokens_in"],
                    "tokens_out": d["tokens_out"],
                    "total_tok":  d["tokens_in"] + d["tokens_out"],
                    "cost_str":   format_cost(cost),
                    "cost_val":   float(cost) if cost is not None else 0.0,
                })
        except Exception:
            for key, d in model_data.items():
                result.append({
                    "model":      key,
                    "calls":      d["calls"],
                    "tokens_in":  d["tokens_in"],
                    "tokens_out": d["tokens_out"],
                    "total_tok":  d["tokens_in"] + d["tokens_out"],
                    "cost_str":   "unknown",
                    "cost_val":   0.0,
                })

        result.sort(key=lambda x: x["total_tok"], reverse=True)
        return result

    def _compute_skill_breakdown(self, skill_events: List[dict]) -> List[Dict]:
        counts: Counter = Counter(s["skill_id"] for s in skill_events)
        total   = sum(counts.values())
        result  = []
        for skill_id, count in counts.most_common():
            result.append({
                "skill":   skill_id,
                "count":   count,
                "pct":     count / total * 100 if total else 0.0,
            })
        return result

    def _compute_activity_patterns(self, requests: List[dict]) -> Dict:
        """Analyse activity by day-of-week and hour."""
        day_counts  = Counter()   # 0=Monday
        hour_counts = Counter()

        for r in requests:
            dt = datetime.fromtimestamp(r["created_at"] / 1000)
            day_counts[dt.weekday()] += 1
            hour_counts[dt.hour]     += 1

        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        by_day  = [{"day": day_names[i], "count": day_counts.get(i, 0)} for i in range(7)]
        by_hour = [{"hour": i,            "count": hour_counts.get(i, 0)} for i in range(24)]

        busiest_day  = max(by_day,  key=lambda x: x["count"]) if by_day  else None
        busiest_hour = max(by_hour, key=lambda x: x["count"]) if by_hour else None

        return {
            "by_day":       by_day,
            "by_hour":      by_hour,
            "busiest_day":  busiest_day,
            "busiest_hour": busiest_hour,
        }

    # ── Public API ─────────────────────────────────────────────────────

    def generate(self, days: int = 30) -> Dict[str, Any]:
        """Generate a complete insights report for the past N days."""
        cutoff_s  = time.time() - (days * 86400)
        cutoff_ms = int(cutoff_s * 1000)

        responses  = self._get_response_events(cutoff_ms)
        requests   = self._get_request_events(cutoff_ms)
        skills     = self._get_skill_events(cutoff_ms)
        plans      = self._get_plan_events(cutoff_ms)
        summaries  = self._get_session_summaries(cutoff_ms)
        cur_turns  = self._get_current_session_turns(cutoff_ms)

        if not responses and not requests:
            return {"days": days, "empty": True}

        overview = self._compute_overview(requests, responses, plans, summaries, cur_turns)
        models   = self._compute_model_breakdown(responses)
        skills_b = self._compute_skill_breakdown(skills)
        activity = self._compute_activity_patterns(requests)

        # Total cost
        total_cost = sum(m["cost_val"] for m in models)
        try:
            from server.usage_pricing import format_cost
            from decimal import Decimal
            total_cost_str = format_cost(Decimal(str(round(total_cost, 6))))
        except Exception:
            total_cost_str = f"${total_cost:.4f}"

        return {
            "days":           days,
            "empty":          False,
            "generated_at":   time.time(),
            "overview":       overview,
            "models":         models,
            "skills":         skills_b,
            "activity":       activity,
            "total_cost_str": total_cost_str,
        }

    def format_terminal(self, report: Dict) -> str:
        """Format the insights report for terminal display."""
        if report.get("empty"):
            days = report.get("days", 30)
            return f"\n  No activity found in the last {days} days."

        lines: List[str] = []
        o    = report["overview"]
        days = report["days"]

        # ── Header ────────────────────────────────────────────────────
        lines.append("")
        lines.append("  ╔══════════════════════════════════════════════════════════╗")
        lines.append("  ║                   📊 Essence Insights                    ║")
        period = f"Last {days} day{'s' if days != 1 else ''}"
        pad    = 56 - len(period)
        lines.append(f"  ║{' ' * (pad // 2)} {period} {' ' * (pad - pad // 2)}║")
        lines.append("  ╚══════════════════════════════════════════════════════════╝")
        lines.append("")

        # ── Overview ──────────────────────────────────────────────────
        lines.append("  📋  Overview")
        lines.append("  " + "─" * 56)
        lines.append(f"  Sessions:        {o['total_sessions']:<14}  "
                     f"Requests:     {o['total_requests']:,}")
        lines.append(f"  LLM calls:       {o['total_llm_calls']:<14,}  "
                     f"Input tokens: {o['total_in']:,}")
        lines.append(f"  Output tokens:   {o['total_out']:<14,}  "
                     f"Total tokens: {o['total_tokens']:,}")
        if o["total_llm_calls"]:
            lines.append(
                f"  Avg in/call:     {o['avg_in_per_call']:<14,.0f}  "
                f"Avg out/call: {o['avg_out_per_call']:,.0f}"
            )
        if o["total_plans"]:
            lines.append(
                f"  Plans run:       {o['total_plans']:<14,}  "
                f"Success:      {o['plan_success_pct']:.0f}%"
            )
        lines.append(f"  Est. cost:       {report['total_cost_str']}")
        lines.append("")

        # ── Model breakdown ───────────────────────────────────────────
        if report["models"]:
            lines.append("  🤖  Models / Providers")
            lines.append("  " + "─" * 56)
            lines.append(f"  {'Model':<36}  {'Calls':>5}  {'Tokens':>10}  {'Cost':>10}")
            for m in report["models"]:
                name = m["model"][:34]
                lines.append(
                    f"  {name:<36}  {m['calls']:>5,}  "
                    f"{m['total_tok']:>10,}  {m['cost_str']:>10}"
                )
            lines.append("")

        # ── Skill usage ───────────────────────────────────────────────
        if report["skills"]:
            lines.append("  🔧  Top Skills Used")
            lines.append("  " + "─" * 56)
            lines.append(f"  {'Skill':<36}  {'Calls':>6}  {'%':>6}")
            for s in report["skills"][:15]:
                name = s["skill"][:34]
                lines.append(f"  {name:<36}  {s['count']:>6,}  {s['pct']:>5.1f}%")
            if len(report["skills"]) > 15:
                lines.append(f"  … and {len(report['skills']) - 15} more skills")
            lines.append("")

        # ── Activity pattern ──────────────────────────────────────────
        act = report.get("activity", {})
        if act.get("by_day"):
            lines.append("  📅  Activity by Day")
            lines.append("  " + "─" * 56)
            peak_day = max(d["count"] for d in act["by_day"])
            for d in act["by_day"]:
                bar = _bar(d["count"], peak_day, 20)
                lines.append(f"  {d['day']}  {bar:<20}  {d['count']}")
            lines.append("")

            busy_hours = sorted(
                [h for h in act["by_hour"] if h["count"] > 0],
                key=lambda h: h["count"], reverse=True
            )[:5]
            if busy_hours:
                hour_strs = []
                for h in busy_hours:
                    hr   = h["hour"]
                    ampm = "AM" if hr < 12 else "PM"
                    disp = hr % 12 or 12
                    hour_strs.append(f"{disp}{ampm}({h['count']})")
                lines.append(f"  Peak hours: {', '.join(hour_strs)}")
                lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine: Optional[InsightsEngine] = None


def get_insights_engine(workspace: Optional[Path] = None) -> InsightsEngine:
    global _engine
    if _engine is None:
        if workspace is None:
            workspace = Path(__file__).resolve().parent.parent
        _engine = InsightsEngine(workspace)
    return _engine
