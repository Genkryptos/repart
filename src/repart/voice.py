"""Spoken descriptions → text with Muse Voice Transcribe (file endpoint).

The browser records WebM/Opus or MP4/AAC; the API wants mono 16-bit WAV, so audio is
converted with the ffmpeg binary that ships with imageio-ffmpeg (no system install).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg
import requests

from . import ledger
from .config import VOICE_FILE_URL, VOICE_MODEL, api_key

MAX_SECONDS = 90
# Words a repair description leans on, so they aren't misheard.
KEYWORDS = ["millimetre", "mm", "shaft", "knob", "D-shaped", "flat", "clip", "bracket", "coupler",
            "spline", "teeth", "hole", "slot", "thread", "mixie", "regulator", "cooler", "socket"]


class VoiceError(RuntimeError):
    pass


def to_wav(src: Path, dst: Path, rate: int = 24000) -> float:
    """Convert any browser recording to mono 16-bit WAV; returns duration in seconds."""
    r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
                        "-ac", "1", "-ar", str(rate), "-sample_fmt", "s16", "-t", str(MAX_SECONDS), str(dst)],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise VoiceError("Couldn't read that recording; try again.")
    with wave.open(str(dst)) as w:
        return w.getnframes() / w.getframerate()


def transcribe(audio: Path, languages: list[str] | None = None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "speech.wav"
        seconds = to_wav(audio, wav)
        if seconds < 0.3:
            raise VoiceError("That recording was too short.")
        price = seconds * ledger.PRICES["muse-voice-transcribe-1.0:second"]
        ledger.ensure(price, "voice transcription")
        req = {"mode": "PUSH_TO_TALK", "model": VOICE_MODEL, "audioEncoding": "WAV", "keywords": KEYWORDS}
        if languages:
            req["languageBias"] = languages
        with wav.open("rb") as f:
            r = requests.post(VOICE_FILE_URL, headers={"Authorization": f"Bearer {api_key()}"},
                              files={"request": (None, json.dumps(req), "application/json"),
                                     "audio": ("speech.wav", f)}, timeout=120)
        if r.status_code != 200:
            raise VoiceError(f"Transcription failed ({r.status_code}).")
        ledger.record(VOICE_MODEL, price, seconds=round(seconds, 1))
        out = r.json()
        return {"text": (out.get("transcript") or "").strip(), "seconds": round(seconds, 1)}
