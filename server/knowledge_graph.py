"""
knowledge_graph.py — Entity/Relation Knowledge Graph for Essence
=================================================================
Stores extracted entities and their relationships in a SQLite-backed
graph. At query time the ego-subgraph around relevant entities is
retrieved and injected into the system prompt as structured context.

Storage: data/knowledge_graph.db

Entity extraction is done heuristically (name patterns, NER keywords)
plus via LLM on important turns (optional, controlled by ESSENCE_KG_LLM).

Usage:
  from server.knowledge_graph import get_knowledge_graph
  kg = get_knowledge_graph()
  kg.add_entity("Python", "language", {"version": "3.12"})
  kg.add_relation("Essence", "uses", "Python")
  ctx = kg.ego_context("Python", depth=2)
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("essence.knowledge_graph")

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "knowledge_graph.db"


# ── Schema ─────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    name       TEXT PRIMARY KEY,
    type       TEXT NOT NULL DEFAULT 'entity',
    props      TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS relations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    subject    TEXT NOT NULL,
    predicate  TEXT NOT NULL,
    object     TEXT NOT NULL,
    weight     REAL NOT NULL DEFAULT 1.0,
    source     TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    UNIQUE(subject, predicate, object)
);
CREATE INDEX IF NOT EXISTS idx_rel_subject ON relations(subject);
CREATE INDEX IF NOT EXISTS idx_rel_object  ON relations(object);
"""


class KnowledgeGraph:
    """SQLite-backed entity/relation graph."""

    def __init__(self, db_path: Path = _DEFAULT_DB) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self._path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ── Write ────────────────────────────────────────────────────────

    def add_entity(
        self, name: str, entity_type: str = "entity", props: dict | None = None
    ) -> None:
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO entities (name, type, props, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET
                     type=excluded.type,
                     props=excluded.props,
                     updated_at=excluded.updated_at""",
                (name, entity_type, json.dumps(props or {}), now, now),
            )

    def add_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        weight: float = 1.0,
        source: str = "",
    ) -> None:
        # Auto-create entities if missing
        self.add_entity(subject)
        self.add_entity(obj)
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO relations
                   (subject, predicate, object, weight, source, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (subject, predicate, obj, weight, source, now),
            )

    def update_weight(self, subject: str, predicate: str, obj: str, delta: float = 0.1) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE relations SET weight = MIN(weight + ?, 5.0)
                   WHERE subject=? AND predicate=? AND object=?""",
                (delta, subject, predicate, obj),
            )

    # ── Read ─────────────────────────────────────────────────────────

    def get_entity(self, name: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM entities WHERE name=?", (name,)
            ).fetchone()
            if row:
                d = dict(row)
                d["props"] = json.loads(d.get("props", "{}"))
                return d
        return None

    def neighbors(self, name: str) -> list[dict]:
        """Return all direct neighbors (outbound + inbound relations)."""
        with self._conn() as conn:
            out = conn.execute(
                "SELECT subject, predicate, object, weight FROM relations WHERE subject=?",
                (name,),
            ).fetchall()
            inn = conn.execute(
                "SELECT subject, predicate, object, weight FROM relations WHERE object=?",
                (name,),
            ).fetchall()
        result = []
        for row in out:
            result.append({"from": row["subject"], "rel": row["predicate"],
                           "to": row["object"], "w": row["weight"]})
        for row in inn:
            result.append({"from": row["subject"], "rel": row["predicate"],
                           "to": row["object"], "w": row["weight"]})
        return result

    def ego_context(self, name: str, depth: int = 2, max_edges: int = 30) -> str:
        """
        Return a text summary of the ego-subgraph centered on `name`.
        Suitable for injection into the LLM system prompt.
        """
        visited: set[str] = set()
        frontier = [name]
        edges:   list[str] = []

        for _ in range(depth):
            next_frontier: list[str] = []
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                for rel in self.neighbors(node):
                    edge = f"{rel['from']} --[{rel['rel']}]--> {rel['to']}"
                    if edge not in edges:
                        edges.append(edge)
                        other = rel["to"] if rel["from"] == node else rel["from"]
                        if other not in visited:
                            next_frontier.append(other)
                if len(edges) >= max_edges:
                    break
            frontier = next_frontier
            if not frontier:
                break

        if not edges:
            return ""
        return "## Knowledge Graph Context\n" + "\n".join(edges[:max_edges])

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Simple substring search on entity names."""
        q = f"%{query}%"
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT name, type FROM entities WHERE name LIKE ? LIMIT ?",
                (q, limit),
            ).fetchall()
        return [{"name": r["name"], "type": r["type"]} for r in rows]

    # ── Auto-extract ──────────────────────────────────────────────────

    def extract_from_text(self, text: str, source: str = "") -> int:
        """
        Heuristic entity + relation extraction from plain text.
        Returns count of new triples added.

        Patterns detected:
          "<Entity> is a <type>"       → entity + isA relation
          "<Entity> uses <Entity>"     → uses relation
          "<X> and <Y>"               → related relation
        """
        count = 0

        # Proper-noun detection (capitalized words in the middle of text)
        pn = re.findall(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b', text)
        entities = list(set(pn))[:20]  # cap

        # Pattern: "X is a Y"
        for m in re.finditer(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+a[n]?\s+([a-z][a-z\s]+)',
            text,
        ):
            subj, obj = m.group(1), m.group(2).strip().split()[0]
            self.add_relation(subj, "isA", obj, source=source)
            count += 1

        # Pattern: "X uses Y"
        for m in re.finditer(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+uses?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            text,
        ):
            self.add_relation(m.group(1), "uses", m.group(2), source=source)
            count += 1

        # Ensure all detected entities exist
        for e in entities:
            if len(e) > 2:
                self.add_entity(e)

        return count

    def stats(self) -> dict:
        with self._conn() as conn:
            n_ent = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            n_rel = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        return {"entities": n_ent, "relations": n_rel}


# ── Module-level singleton ─────────────────────────────────────────────────

_kg: Optional[KnowledgeGraph] = None


def get_knowledge_graph(db_path: Path | None = None) -> KnowledgeGraph:
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph(db_path or _DEFAULT_DB)
    return _kg
