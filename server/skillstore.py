"""
Essence Skill Store — file-backed dynamic skill registry.

Skills live in  Essence/skills/*.json  (one file per skill).
The built-in skills are seeded on first run and can be overridden.

Each skill JSON schema
----------------------
{
  "id":                  "my_skill",
  "label":               "My Skill",
  "category":            "tool",          // intelligence | cie | ledger | tool | mcp | custom
  "description":         "What it does",
  "enabled":             true,
  "builtin":             false,           // built-ins survive reset; user skills can be deleted
  "system_prompt":       "",             // injected into system prompt when enabled
  "mcp_server":          "",             // MCP server id this skill delegates to (optional)
  "tags":                [],
  "created":             1234567890.0,
  "updated":             1234567890.0
}
"""
from __future__ import annotations
import json, time, logging
from pathlib import Path

log = logging.getLogger("essence.skillstore")

_BUILTIN_SKILLS = [
    {"id":"lif",         "label":"Living Intent Fabric",    "category":"intelligence","description":"4-horizon temporal intent tracking with TGS scoring","enabled":True, "builtin":True,"system_prompt":"","mcp_server":"","tags":["core","intelligence"]},
    {"id":"tomm",        "label":"Theory of Mind",          "category":"intelligence","description":"6-dimension cognitive state modeling (load, valence, focus, stress, clarity)","enabled":True, "builtin":True,"system_prompt":"","mcp_server":"","tags":["core","intelligence"]},
    {"id":"restraint",   "label":"Restraint Engine",        "category":"intelligence","description":"ABSTAIN-first decision gating — 8 pre-conditions evaluated before every action","enabled":True, "builtin":True,"system_prompt":"","mcp_server":"","tags":["core","intelligence"]},
    {"id":"endorsement", "label":"Endorsement Projector",   "category":"intelligence","description":"Weighted goal/value/historical/constraint scoring (threshold 0.65)","enabled":True, "builtin":True,"system_prompt":"","mcp_server":"","tags":["core","intelligence"]},
    {"id":"fis",         "label":"FIS Classifier",          "category":"intelligence","description":"7-axis intent classification, 500ms latency target","enabled":True, "builtin":True,"system_prompt":"","mcp_server":"","tags":["core","intelligence"]},
    {"id":"morning_brief","label":"Morning Brief Engine",   "category":"intelligence","description":"Fatigue-adaptive daily brief (HEALTHY/WATCH/FATIGUED)","enabled":True, "builtin":True,"system_prompt":"","mcp_server":"","tags":["core","intelligence"]},
    {"id":"cie",         "label":"CIE Scorer",              "category":"cie",         "description":"Daily interrupt budget (10/day), midnight reset","enabled":True, "builtin":True,"system_prompt":"","mcp_server":"","tags":["core","cie"]},
    {"id":"routine",     "label":"Routine Model",           "category":"cie",         "description":"Markov α=0.3 behavioral pattern tracker, cold-start < 50 obs","enabled":True, "builtin":True,"system_prompt":"","mcp_server":"","tags":["core","cie"]},
    {"id":"anomaly",     "label":"Anomaly Detector",        "category":"cie",         "description":"Moving baseline ring buffers for context/focus/proposal drift","enabled":True, "builtin":True,"system_prompt":"","mcp_server":"","tags":["core","cie"]},
    {"id":"quiet_ledger","label":"Quiet Ledger",            "category":"ledger",      "description":"Accountable silence — every action and abstention recorded with rollback tiers","enabled":True, "builtin":True,"system_prompt":"","mcp_server":"","tags":["core","ledger"]},
    {"id":"trust_ledger","label":"Trust Ledger",            "category":"ledger",      "description":"3-axis trust (COMP/VALS/JUDG) per domain, never_graduate lock on SEVERE_VIOLATION","enabled":True, "builtin":True,"system_prompt":"","mcp_server":"","tags":["core","ledger"]},
    {"id":"web_search",  "label":"Web Search",              "category":"tool",        "description":"Live web search integration — inject search results into context","enabled":False,"builtin":True,"system_prompt":"You have access to web search. When the user asks about current events or facts, search for them.","mcp_server":"","tags":["tool"]},
    {"id":"code_exec",   "label":"Code Executor",           "category":"tool",        "description":"Sandboxed Python/JS execution for code output verification","enabled":False,"builtin":True,"system_prompt":"You can execute code when asked. Show the result of execution.","mcp_server":"","tags":["tool"]},
    {"id":"file_watcher","label":"File Watcher",            "category":"tool",        "description":"Watch workspace files and inject changes into context automatically","enabled":False,"builtin":True,"system_prompt":"","mcp_server":"","tags":["tool"]},
    {"id":"voice_input", "label":"Voice Input",             "category":"tool",        "description":"Browser microphone → Whisper transcription","enabled":False,"builtin":True,"system_prompt":"","mcp_server":"","tags":["tool"]},
]


class SkillStore:
    def __init__(self, skills_dir: Path):
        self.dir = skills_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._seed_builtins()

    def _skill_path(self, skill_id: str) -> Path:
        return self.dir / f"{skill_id}.json"

    def _seed_builtins(self) -> None:
        """Write built-in skills to disk if they don't exist yet."""
        for s in _BUILTIN_SKILLS:
            p = self._skill_path(s["id"])
            if not p.exists():
                self._write(s)
        log.debug("SkillStore seeded %d built-in skills", len(_BUILTIN_SKILLS))

    def _write(self, skill: dict) -> None:
        p = self._skill_path(skill["id"])
        p.write_text(json.dumps(skill, indent=2), encoding="utf-8")

    def _read(self, p: Path) -> dict | None:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Failed to read skill %s: %s", p, e)
            return None

    def list_all(self) -> list[dict]:
        skills = []
        for p in sorted(self.dir.glob("*.json")):
            s = self._read(p)
            if s:
                skills.append(s)
        return skills

    def get(self, skill_id: str) -> dict | None:
        p = self._skill_path(skill_id)
        return self._read(p) if p.exists() else None

    def create(self, data: dict) -> dict:
        sid = data.get("id", "").strip().replace(" ", "_").lower()
        if not sid:
            import re
            sid = re.sub(r"[^a-z0-9_]", "_", data.get("label", "skill").lower())
        if not sid:
            sid = f"skill_{int(time.time())}"
        # Don't overwrite built-ins
        existing = self.get(sid)
        if existing and existing.get("builtin"):
            raise ValueError(f"Cannot overwrite built-in skill '{sid}'")
        now = time.time()
        skill = {
            "id":           sid,
            "label":        data.get("label", sid),
            "category":     data.get("category", "custom"),
            "description":  data.get("description", ""),
            "enabled":      bool(data.get("enabled", True)),
            "builtin":      False,
            "system_prompt": data.get("system_prompt", ""),
            "mcp_server":   data.get("mcp_server", ""),
            "tags":         data.get("tags", []),
            "created":      existing.get("created", now) if existing else now,
            "updated":      now,
        }
        self._write(skill)
        return skill

    def patch(self, skill_id: str, patch: dict) -> dict:
        s = self.get(skill_id)
        if s is None:
            raise KeyError(f"Skill '{skill_id}' not found")
        allowed = {"label","description","enabled","category","system_prompt","mcp_server","tags"}
        s.update({k: v for k, v in patch.items() if k in allowed})
        s["updated"] = time.time()
        self._write(s)
        return s

    def delete(self, skill_id: str) -> bool:
        s = self.get(skill_id)
        if s is None:
            return False
        if s.get("builtin"):
            raise ValueError(f"Cannot delete built-in skill '{skill_id}'. Disable it instead.")
        p = self._skill_path(skill_id)
        p.unlink(missing_ok=True)
        return True

    def active_system_prompts(self) -> list[str]:
        """Return system_prompt strings for all enabled skills that have one."""
        prompts = []
        for s in self.list_all():
            if s.get("enabled") and s.get("system_prompt", "").strip():
                prompts.append(s["system_prompt"].strip())
        return prompts

    @staticmethod
    def from_markdown(md_text: str) -> dict:
        """
        Parse a Markdown skill definition:
          # Skill Name
          **Category:** tool
          **Description:** What it does
          ---
          (system prompt text below the hr)
        """
        import re
        lines = md_text.strip().splitlines()
        data: dict = {"label": "", "category": "custom", "description": "", "system_prompt": ""}
        hr_idx = None
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("# "):
                data["label"] = line[2:].strip()
            elif line.lower().startswith("**category:**"):
                data["category"] = line.split(":", 1)[1].strip().strip("**").lower()
            elif line.lower().startswith("**description:**"):
                data["description"] = line.split(":", 1)[1].strip().strip("**")
            elif line.startswith("---") and hr_idx is None:
                hr_idx = i
        if hr_idx is not None:
            data["system_prompt"] = "\n".join(lines[hr_idx+1:]).strip()
        return data
