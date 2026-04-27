"""
voice_io.py — Voice I/O for Essence
=====================================
STT via faster-whisper  (offline, no API key)
TTS via kokoro-onnx     (offline, no API key)

Install:
  pip install faster-whisper kokoro-onnx pyaudio sounddevice numpy

Both engines are lazy-loaded so the module imports cleanly even if
the packages are absent.  Use `is_stt_available()` / `is_tts_available()`
to check before calling.

Usage:
  from server.voice_io import VoiceIO
  vio = VoiceIO()
  text = await vio.listen()        # record microphone → text
  await vio.speak("Hello world")   # text → audio playback
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("essence.voice_io")

# ── Availability helpers ───────────────────────────────────────────────────

def is_stt_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def is_tts_available() -> bool:
    try:
        import kokoro  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def is_mic_available() -> bool:
    try:
        import sounddevice  # noqa: F401
        return True
    except ImportError:
        try:
            import pyaudio  # noqa: F401
            return True
        except ImportError:
            return False


# ── VoiceIO class ──────────────────────────────────────────────────────────

class VoiceIO:
    """
    Manages STT and TTS for Essence.

    Parameters
    ----------
    model_size  : faster-whisper model ('tiny', 'base', 'small', 'medium', 'large-v3')
    tts_voice   : kokoro voice id (e.g. 'af_sky', 'am_adam')
    record_secs : max recording duration in seconds
    sample_rate : audio sample rate (default 16000 Hz for Whisper)
    """

    def __init__(
        self,
        model_size: str = "base",
        tts_voice:  str = "af_sky",
        record_secs: float = 10.0,
        sample_rate: int   = 16000,
    ) -> None:
        self._model_size  = model_size
        self._tts_voice   = tts_voice
        self._record_secs = record_secs
        self._sample_rate = sample_rate
        self._stt_model   = None   # lazy-loaded
        self._tts_pipeline = None  # lazy-loaded
        self._running     = False

    # ── STT ────────────────────────────────────────────────────────────

    def _load_stt(self) -> None:
        """Lazy-load faster-whisper model."""
        if self._stt_model is not None:
            return
        try:
            from faster_whisper import WhisperModel  # type: ignore
            device = "cuda" if _has_cuda() else "cpu"
            compute = "float16" if device == "cuda" else "int8"
            log.info("voice: loading Whisper %s on %s/%s", self._model_size, device, compute)
            self._stt_model = WhisperModel(
                self._model_size,
                device=device,
                compute_type=compute,
            )
        except ImportError:
            raise RuntimeError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )

    async def listen(self, timeout: float | None = None) -> str:
        """
        Record audio from the microphone and transcribe to text.
        Returns the transcribed string (empty string on error/silence).
        """
        if not is_mic_available():
            return "[ERROR: no audio input device found]"

        secs = timeout or self._record_secs
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(None, self._record_audio, secs)
        if audio_data is None:
            return ""

        return await loop.run_in_executor(None, self._transcribe, audio_data)

    def _record_audio(self, secs: float):
        """Record audio synchronously — runs in thread pool."""
        try:
            import numpy as np
            try:
                import sounddevice as sd
                log.info("voice: recording %.1fs via sounddevice…", secs)
                data = sd.rec(
                    int(secs * self._sample_rate),
                    samplerate=self._sample_rate,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()
                return data.flatten()
            except ImportError:
                pass

            # Fallback: pyaudio
            import pyaudio
            import wave
            pa   = pyaudio.PyAudio()
            fmt  = pyaudio.paInt16
            stream = pa.open(format=fmt, channels=1, rate=self._sample_rate, input=True,
                             frames_per_buffer=1024)
            log.info("voice: recording %.1fs via pyaudio…", secs)
            frames = []
            for _ in range(int(self._sample_rate / 1024 * secs)):
                frames.append(stream.read(1024, exception_on_overflow=False))
            stream.stop_stream()
            stream.close()
            pa.terminate()
            raw = b"".join(frames)
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            return arr
        except Exception as exc:
            log.warning("voice: record_audio failed: %s", exc)
            return None

    def _transcribe(self, audio) -> str:
        """Transcribe audio ndarray synchronously."""
        try:
            self._load_stt()
            import numpy as np
            if not isinstance(audio, np.ndarray):
                return ""
            segments, _info = self._stt_model.transcribe(audio, beam_size=5)
            text = " ".join(seg.text for seg in segments).strip()
            log.info("voice: STT result: %r", text[:80])
            return text
        except Exception as exc:
            log.warning("voice: transcription failed: %s", exc)
            return ""

    # ── TTS ────────────────────────────────────────────────────────────

    def _load_tts(self) -> None:
        """Lazy-load kokoro TTS pipeline."""
        if self._tts_pipeline is not None:
            return
        try:
            from kokoro import KPipeline  # type: ignore
            log.info("voice: loading Kokoro TTS (voice=%s)", self._tts_voice)
            self._tts_pipeline = KPipeline(lang_code="a")
        except ImportError:
            raise RuntimeError(
                "kokoro-onnx not installed. Run: pip install kokoro-onnx"
            )

    async def speak(self, text: str, blocking: bool = True) -> None:
        """
        Synthesise text to speech and play through the default audio output.
        If blocking=False, fire-and-forget in a thread.
        """
        if not text.strip():
            return
        loop = asyncio.get_event_loop()
        if blocking:
            await loop.run_in_executor(None, self._tts_and_play, text)
        else:
            loop.run_in_executor(None, self._tts_and_play, text)

    def _tts_and_play(self, text: str) -> None:
        """Synthesise and play synchronously — runs in thread pool."""
        try:
            self._load_tts()
            import numpy as np
            import soundfile as sf  # type: ignore

            audio_chunks = []
            for _, _, audio in self._tts_pipeline(text, voice=self._tts_voice):
                audio_chunks.append(audio)

            if not audio_chunks:
                return

            audio = np.concatenate(audio_chunks)

            # Play via sounddevice if available
            try:
                import sounddevice as sd
                sd.play(audio, samplerate=24000)
                sd.wait()
                return
            except ImportError:
                pass

            # Fallback: write WAV to temp file, play with system player
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
            sf.write(tmp, audio, 24000)
            self._play_wav(tmp)
            try:
                Path(tmp).unlink()
            except Exception:
                pass

        except Exception as exc:
            log.warning("voice: TTS playback failed: %s", exc)

    def _play_wav(self, path: str) -> None:
        """Platform-specific WAV playback fallback."""
        import subprocess, sys
        if sys.platform == "darwin":
            subprocess.run(["afplay", path], timeout=60)
        elif sys.platform == "win32":
            subprocess.run(
                ["powershell", "-c", f"(New-Object Media.SoundPlayer '{path}').PlaySync()"],
                timeout=60,
            )
        else:
            for cmd in [["aplay", path], ["paplay", path], ["ffplay", "-nodisp", "-autoexit", path]]:
                try:
                    subprocess.run(cmd, timeout=60, stderr=subprocess.DEVNULL)
                    return
                except FileNotFoundError:
                    continue


# ── Module-level singleton ─────────────────────────────────────────────────

_voice_io: Optional[VoiceIO] = None


def get_voice_io(
    model_size: str = "base",
    tts_voice:  str = "af_sky",
) -> VoiceIO:
    """Return the process-level VoiceIO singleton."""
    global _voice_io
    if _voice_io is None:
        _voice_io = VoiceIO(model_size=model_size, tts_voice=tts_voice)
    return _voice_io


def _has_cuda() -> bool:
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except ImportError:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, timeout=3
            )
            return result.returncode == 0
        except Exception:
            return False
