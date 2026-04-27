"""essence_cie.routine — Markov Routine Predictor (ported from Essence v1.0)."""
from __future__ import annotations
import json, math, os, threading
from pathlib import Path
from typing import Any

_ALPHA            = 0.3
_MIN_OBSERVATIONS = 50
_DRIFT_ALERT      = 0.40
_PRUNE_THRESHOLD  = 0.05


class RoutineModel:
    """Thread-safe first-order Markov habit predictor. Persisted to JSON."""

    def __init__(self, cache_path: str | Path = "") -> None:
        self._cache_path  = str(cache_path)
        self._transitions: dict[str, dict[str, float]] = {}
        self._entropy     = 0.0
        self._drift_score = 0.0
        self._observations = 0
        self._lock        = threading.RLock()
        self._load()

    def predict_next(self, fingerprint: str) -> tuple[str, float]:
        with self._lock:
            if self._observations < _MIN_OBSERVATIONS:
                return "", 0.0
            probs = self._transitions.get(fingerprint)
            if not probs:
                return "", 0.0
            best_skill, best_prob = max(probs.items(), key=lambda kv: kv[1])
            adjusted = best_prob * (1.0 - self._drift_score)
            return best_skill, round(adjusted, 4)

    def observe(self, fingerprint: str, skill: str) -> None:
        with self._lock:
            if fingerprint not in self._transitions:
                self._transitions[fingerprint] = {}
            probs = self._transitions[fingerprint]
            for k in list(probs):
                probs[k] = (_ALPHA if k == skill else 0.0) + (1 - _ALPHA) * probs[k]
                if k != skill:
                    probs[k] = (1 - _ALPHA) * probs[k]
            if skill not in probs:
                probs[skill] = _ALPHA
            # Re-normalise
            total = sum(probs.values())
            if total > 0:
                for k in probs:
                    probs[k] /= total
            self._observations += 1
            self._prune(fingerprint)
            self._update_entropy()
            self._update_drift(fingerprint, skill)
            self._save()

    def fingerprint(self, hour: int | None = None, app: str = "", day: int | None = None) -> str:
        parts = []
        if hour is not None: parts.append(f"h{hour // 3}")
        if day  is not None: parts.append("wknd" if day >= 5 else "wkdy")
        if app:
            norm = "".join(c for c in app.lower() if c.isalnum() or c in "._-")[:20]
            parts.append(norm)
        return ":".join(parts) or "default"

    @property
    def is_warm(self) -> bool: return self._observations >= _MIN_OBSERVATIONS
    @property
    def drift_score(self) -> float: return self._drift_score
    @property
    def entropy(self) -> float: return self._entropy
    @property
    def observations(self) -> int: return self._observations

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"observations": self._observations, "is_warm": self.is_warm,
                    "entropy": round(self._entropy, 4), "drift_score": round(self._drift_score, 4),
                    "fingerprints": len(self._transitions)}

    def _prune(self, fp: str) -> None:
        probs = self._transitions.get(fp, {})
        for k in [k for k, p in probs.items() if p < _PRUNE_THRESHOLD]:
            del probs[k]

    def _update_entropy(self) -> None:
        total = count = 0.0
        for probs in self._transitions.values():
            e = sum(-p * math.log2(p) for p in probs.values() if p > 0)
            total += e; count += 1
        self._entropy = total / count if count else 0.0

    def _update_drift(self, fp: str, observed: str) -> None:
        probs = self._transitions.get(fp, {})
        pred  = max(probs, key=probs.get) if probs else ""
        miss  = 0.0 if pred == observed else 1.0
        self._drift_score = 0.1 * miss + 0.9 * self._drift_score

    def _save(self) -> None:
        if not self._cache_path: return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._cache_path)), exist_ok=True)
            data = {"transitions": self._transitions, "entropy": self._entropy,
                    "drift_score": self._drift_score, "observations": self._observations}
            tmp = self._cache_path + ".tmp"
            with open(tmp, "w") as f: json.dump(data, f)
            os.replace(tmp, self._cache_path)
        except Exception: pass

    def _load(self) -> None:
        if not self._cache_path or not os.path.exists(self._cache_path): return
        try:
            with open(self._cache_path) as f: data = json.load(f)
            self._transitions  = data.get("transitions", {})
            self._entropy      = data.get("entropy", 0.0)
            self._drift_score  = data.get("drift_score", 0.0)
            self._observations = data.get("observations", 0)
        except Exception: pass
