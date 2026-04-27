"""
embeddings.py — Local Vector Embeddings for Semantic Memory Retrieval
======================================================================
Generates and caches text embeddings via Ollama's /api/embeddings endpoint,
then does cosine-similarity search over stored episodes so that episodic
memory retrieval is semantically accurate rather than purely recency-based.

Architecture
------------
  EmbeddingIndex
    • In-memory dict: episode_id → float[] vector
    • Disk cache:     data/embeddings.json  (survives restarts)
    • Pending queue:  texts waiting for their first embedding
    • flush_pending() — async, batches Ollama /api/embeddings calls
    • search(query, n) — async, returns ranked list of (id, score) tuples

Integration
-----------
  EpisodicMemory.append() calls embedding_index.queue(id, text).
  EpisodicMemory.semantic_search(query, n, ollama_url, model) calls
    flush_pending() + search() and returns matching episode dicts.
  kernel._stream_response() calls semantic_search() with the current
    user message as the query, replacing the plain recency-based
    build_context_block() call for retrieval (recency-based is still
    used for the raw message history injected into messages[]).

Fallback
--------
  When the embedding model is unavailable or Ollama is offline, all
  methods fail silently and return empty results.  The kernel always
  has the recency-based fallback in place, so degradation is invisible.

Model default
-------------
  "nomic-embed-text" — free, fast, local, 768-dim, ~274 MB
  Can be changed via ESSENCE_EMBED_MODEL env var or init_embedding_index().
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("essence.embeddings")

_DEFAULT_EMBED_MODEL = os.environ.get("ESSENCE_EMBED_MODEL", "nomic-embed-text")
_CACHE_VERSION       = 1
_BATCH_SIZE          = 16    # max concurrent embedding calls
_EMBED_TIMEOUT       = 10.0  # seconds per embedding request


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) or 1e-9


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity ∈ [−1, 1]."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return _dot(a, b) / (_norm(a) * _norm(b))


# ---------------------------------------------------------------------------
# EmbeddingIndex
# ---------------------------------------------------------------------------

class EmbeddingIndex:
    """
    Lightweight vector store backed by a JSON file.

    Not meant for millions of documents — episodic memory typically has
    a few hundred entries per session, so a brute-force cosine scan is fast.
    """

    def __init__(
        self,
        cache_path: "Path | None" = None,
        ollama_url: str = "http://localhost:11434",
        model:      str = _DEFAULT_EMBED_MODEL,
    ) -> None:
        if cache_path is None:
            cache_path = Path(__file__).resolve().parent.parent / "data" / "embeddings.json"
        self._cache_path  = cache_path
        self._ollama_url  = ollama_url
        self._model       = model
        self._vectors:    dict[str, list[float]] = {}   # id → vector
        self._texts:      dict[str, str]          = {}   # id → text (for re-embed)
        self._pending:    dict[str, str]          = {}   # id → text (not yet embedded)
        self._available:  bool | None             = None  # None = untested
        self._load_cache()

    # ── Cache I/O ─────────────────────────────────────────────────────

    def _load_cache(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            if data.get("version") != _CACHE_VERSION:
                return
            self._vectors = data.get("vectors", {})
            self._texts   = data.get("texts",   {})
            log.debug("embeddings: loaded %d cached vectors", len(self._vectors))
        except Exception as exc:
            log.debug("embeddings: cache load failed: %s", exc)

    def _save_cache(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {"version": _CACHE_VERSION,
                     "vectors": self._vectors,
                     "texts":   self._texts},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            tmp.replace(self._cache_path)
        except Exception as exc:
            log.debug("embeddings: cache save failed: %s", exc)

    # ── Write ──────────────────────────────────────────────────────────

    def queue(self, episode_id: str, text: str) -> None:
        """
        Schedule *text* for embedding under *episode_id*.
        Fast — does not hit the network.  Call flush_pending() to embed.
        """
        if not text or not text.strip():
            return
        clean = text.strip()
        self._texts[episode_id] = clean
        if episode_id not in self._vectors:
            self._pending[episode_id] = clean

    def remove(self, episode_id: str) -> None:
        """Remove an episode from the index."""
        self._vectors.pop(episode_id, None)
        self._texts.pop(episode_id, None)
        self._pending.pop(episode_id, None)

    # ── Flush pending ─────────────────────────────────────────────────

    async def flush_pending(self, limit: int = 0) -> int:
        """
        Generate embeddings for all queued texts.  Processes in batches.
        Returns the number of new vectors generated.
        limit: max items to process per call (0 = all pending)
        """
        if not self._pending:
            return 0

        items = list(self._pending.items())
        if limit:
            items = items[:limit]

        generated = 0
        for i in range(0, len(items), _BATCH_SIZE):
            batch = items[i : i + _BATCH_SIZE]
            for eid, text in batch:
                vec = await self._embed_one(text)
                if vec:
                    self._vectors[eid] = vec
                    self._pending.pop(eid, None)
                    generated += 1

        if generated:
            self._save_cache()
            log.debug("embeddings: generated %d new vectors (%d pending)", generated, len(self._pending))

        return generated

    async def _embed_one(self, text: str) -> list[float] | None:
        """Call Ollama /api/embeddings for a single text. Returns None on failure."""
        if self._available is False:
            return None  # known-down, skip

        try:
            async with httpx.AsyncClient(timeout=_EMBED_TIMEOUT) as c:
                r = await c.post(
                    f"{self._ollama_url}/api/embeddings",
                    json={"model": self._model, "prompt": text[:2000]},
                )
                r.raise_for_status()
                vec = r.json().get("embedding", [])
                if vec:
                    self._available = True
                    return vec
                log.debug("embeddings: empty embedding returned for model %s", self._model)
                return None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                log.info(
                    "embeddings: model '%s' not found in Ollama — "
                    "run: ollama pull %s",
                    self._model, self._model,
                )
                self._available = False
            else:
                log.debug("embeddings: HTTP %d from Ollama", exc.response.status_code)
            return None
        except Exception as exc:
            if self._available is None:
                log.debug("embeddings: Ollama not reachable (%s) — semantic search disabled", exc)
                self._available = False
            return None

    # ── Search ────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        n: int = 8,
    ) -> list[tuple[str, float]]:
        """
        Return top-N (episode_id, score) pairs by cosine similarity.
        Flushes pending embeddings first; skips items not yet embedded.
        """
        if not query.strip() or not self._vectors:
            return []

        await self.flush_pending(limit=_BATCH_SIZE)

        q_vec = await self._embed_one(query.strip()[:1000])
        if not q_vec:
            return []

        scored: list[tuple[str, float]] = []
        for eid, vec in self._vectors.items():
            score = cosine_similarity(q_vec, vec)
            scored.append((eid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    # ── Stats ─────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._vectors)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def is_available(self) -> bool:
        """True if embedding model was reachable in the last attempt."""
        return self._available is True

    def update_config(self, ollama_url: str = "", model: str = "") -> None:
        """Hot-update Ollama URL and/or embedding model."""
        if ollama_url:
            self._ollama_url = ollama_url
            self._available  = None  # re-probe
        if model and model != self._model:
            self._model     = model
            self._available = None
            # Invalidate all cached vectors (different model = different dim)
            self._vectors.clear()
            self._pending   = dict(self._texts)  # re-queue everything
            self._save_cache()


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_index: EmbeddingIndex | None = None


def init_embedding_index(
    cache_path: Path,
    ollama_url: str = "http://localhost:11434",
    model: str = _DEFAULT_EMBED_MODEL,
) -> EmbeddingIndex:
    global _index
    _index = EmbeddingIndex(cache_path, ollama_url, model)
    return _index


def get_embedding_index() -> EmbeddingIndex | None:
    return _index
