"""
wake_word.py — Always-On Wake-Word Detection for Essence
=========================================================
Listens for a configurable wake phrase on the system microphone
and emits a `trigger.fired` event when detected so the voice_io
module can record and transcribe the follow-up command.

Wake-word matching is done locally using:
  1. Whisper (faster-whisper) — offline, accurate, ~200ms latency
     Audio is captured in 1-second chunks and matched against the
     wake phrase using simple string similarity.

Configuration:
  ESSENCE_WAKE_WORD      — phrase to listen for (default: "hey essence")
  ESSENCE_WAKE_THRESHOLD — match confidence 0-1 (default: 0.75)
  ESSENCE_WAKE_ENABLED   — "true" / "false" (default: "false")

Install: pip install faster-whisper sounddevice numpy

Usage:
  from server.wake_word import WakeWordDetector
  detector = WakeWordDetector(bus)
  detector.start()   # background task
  detector.stop()
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("essence.wake_word")

_DEFAULT_WAKE_WORD  = os.environ.get("ESSENCE_WAKE_WORD", "hey essence").lower()
_DEFAULT_THRESHOLD  = float(os.environ.get("ESSENCE_WAKE_THRESHOLD", "0.75"))
_CHUNK_SECS         = 1.5   # seconds per detection chunk
_SAMPLE_RATE        = 16000


class WakeWordDetector:
    """
    Continuously records audio in short chunks and looks for the wake word.
    Fires trigger.fired on the bus when the wake phrase is detected.
    """

    def __init__(
        self,
        bus:       Any,
        wake_word: str   = _DEFAULT_WAKE_WORD,
        threshold: float = _DEFAULT_THRESHOLD,
        model_size: str  = "tiny",
    ) -> None:
        self._bus        = bus
        self._wake_word  = wake_word.lower()
        self._threshold  = threshold
        self._model_size = model_size
        self._running    = False
        self._task: Optional[asyncio.Task] = None
        self._model      = None    # lazy

    # ── Public API ─────────────────────────────────────────────────

    def start(self) -> None:
        if not _is_enabled():
            log.debug("wake_word: disabled (set ESSENCE_WAKE_ENABLED=true to enable)")
            return
        if self._running:
            return
        self._running = True
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._loop(), name="wake-word-detector")
        log.info("WakeWord: listening for %r (threshold=%.2f)", self._wake_word, self._threshold)

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    # ── Detection loop ─────────────────────────────────────────────

    async def _loop(self) -> None:
        while self._running:
            try:
                loop = asyncio.get_event_loop()
                audio = await loop.run_in_executor(None, self._record_chunk)
                if audio is None:
                    await asyncio.sleep(0.5)
                    continue

                transcript = await loop.run_in_executor(None, self._transcribe, audio)
                if transcript:
                    score = self._similarity(transcript.lower(), self._wake_word)
                    if score >= self._threshold:
                        log.info("WakeWord: detected %r (score=%.2f)", transcript[:60], score)
                        self._fire()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.debug("wake_word: loop error: %s", exc)
                await asyncio.sleep(1.0)

    def _record_chunk(self):
        try:
            import numpy as np
            import sounddevice as sd
            data = sd.rec(
                int(_CHUNK_SECS * _SAMPLE_RATE),
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            return data.flatten()
        except ImportError:
            log.warning("wake_word: sounddevice not installed — wake word disabled")
            self._running = False
            return None
        except Exception as exc:
            log.debug("wake_word: record_chunk: %s", exc)
            return None

    def _transcribe(self, audio) -> str:
        try:
            if self._model is None:
                from faster_whisper import WhisperModel  # type: ignore
                self._model = WhisperModel(
                    self._model_size, device="cpu", compute_type="int8"
                )
            segments, _ = self._model.transcribe(audio, beam_size=1)
            return " ".join(s.text for s in segments).strip()
        except ImportError:
            log.warning("wake_word: faster-whisper not installed — wake word disabled")
            self._running = False
            return ""
        except Exception as exc:
            log.debug("wake_word: transcribe: %s", exc)
            return ""

    def _similarity(self, transcript: str, wake_word: str) -> float:
        """Simple token-overlap similarity."""
        t_tokens = set(transcript.split())
        w_tokens = set(wake_word.split())
        if not w_tokens:
            return 0.0
        overlap = t_tokens & w_tokens
        return len(overlap) / len(w_tokens)

    def _fire(self) -> None:
        try:
            from server.event_bus import Envelope
            self._bus.publish_sync(Envelope(
                topic="trigger.fired",
                data={
                    "source":   "wake_word",
                    "intent":   "voice_command",
                    "payload":  {"wake_word": self._wake_word},
                },
            ))
        except Exception as exc:
            log.warning("wake_word: fire failed: %s", exc)


# ── Helpers ────────────────────────────────────────────────────────────────

def _is_enabled() -> bool:
    val = os.environ.get("ESSENCE_WAKE_ENABLED", "false").lower()
    return val in ("true", "1", "yes")


# ── Module-level singleton ─────────────────────────────────────────────────

_detector: Optional[WakeWordDetector] = None


def get_wake_detector(bus: Any) -> WakeWordDetector:
    global _detector
    if _detector is None:
        _detector = WakeWordDetector(bus)
    return _detector
