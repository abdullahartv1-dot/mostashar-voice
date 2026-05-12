"""OpenAI STT — drop-in alternative to local Whisper.

Sends a 16 kHz PCM16 LE audio buffer to OpenAI's transcription endpoint
and returns the Arabic transcript. Way more accurate than local Whisper
on the dialects and acoustic conditions our users hit, at the cost of
~$0.003-0.006 per minute of audio.

Three OpenAI transcription models exposed via env:
  • gpt-4o-mini-transcribe (default — cheapest, best Arabic dialects)
  • gpt-4o-transcribe       (slightly better, ~2x cost)
  • whisper-1               (legacy, similar quality to local Whisper)

Switching strategy: set MV_STT_BACKEND=openai on the pod. The server's
audio handler routes through this module instead of the local
_whisper_transcribe_sync. When OPENAI key is missing, falls back to
local Whisper automatically so the system never silently breaks.
"""

from __future__ import annotations

import io
import logging
import os
import struct
import wave
from typing import Optional

import httpx
import numpy as np


log = logging.getLogger("voice_openai_stt")

OPENAI_TRANSCRIBE_URL = os.environ.get(
    "MV_OPENAI_STT_URL",
    "https://api.openai.com/v1/audio/transcriptions",
)
OPENAI_STT_MODEL = os.environ.get(
    "MV_OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"
)
OPENAI_STT_LANGUAGE = os.environ.get("MV_OPENAI_STT_LANGUAGE", "ar")
OPENAI_STT_TIMEOUT_S = float(os.environ.get("MV_OPENAI_STT_TIMEOUT_S", "20"))


def _pcm16_to_wav_bytes(audio_f32: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 audio in [-1, 1] to WAV bytes (PCM16 mono).

    Done in-memory to avoid touching the overflowed container disk.
    """
    # Clip and convert to int16
    clipped = np.clip(audio_f32, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


async def openai_transcribe(
    audio_f32_16k: np.ndarray,
    *,
    prompt: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """Transcribe a single 16 kHz mono audio clip via OpenAI's API.

    Args:
        audio_f32_16k: float32 mono audio @ 16 kHz, values in [-1, 1].
        prompt: optional Arabic prompt that biases vocabulary/style.
        language: override the language (default from env: "ar").

    Returns:
        Transcribed Arabic text, or "" on any failure / silent input.

    Raises:
        RuntimeError if MV_OPENAI_KEY is missing — caller should catch
        and fall back to local Whisper.
    """
    api_key = os.environ.get("MV_OPENAI_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MV_OPENAI_KEY env var not set")

    if audio_f32_16k.size < int(0.2 * 16000):
        # Too short to bother sending — match local Whisper's gate.
        log.info("[openai-stt] skipped (too short: %.2fs)",
                 audio_f32_16k.size / 16000)
        return ""

    wav_bytes = _pcm16_to_wav_bytes(audio_f32_16k, sample_rate=16000)

    # multipart/form-data — model + language + file + optional prompt.
    files = {
        "file": ("audio.wav", wav_bytes, "audio/wav"),
    }
    data = {
        "model": OPENAI_STT_MODEL,
        "language": language or OPENAI_STT_LANGUAGE,
        "response_format": "json",
        # Lower temperature → more faithful to actual audio, less
        # creative interpretation. Helps with dialect Arabic.
        "temperature": "0",
    }
    if prompt:
        data["prompt"] = prompt[:240]  # API caps prompt at ~250 tokens

    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=OPENAI_STT_TIMEOUT_S) as client:
            resp = await client.post(
                OPENAI_TRANSCRIBE_URL, headers=headers,
                files=files, data=data,
            )
    except httpx.TimeoutException:
        log.warning("[openai-stt] timeout after %.0fs", OPENAI_STT_TIMEOUT_S)
        return ""
    except Exception as e:  # noqa: BLE001
        log.warning("[openai-stt] network error: %s", e)
        return ""

    if resp.status_code >= 400:
        log.warning("[openai-stt] status %d: %s",
                    resp.status_code, resp.text[:300])
        return ""

    try:
        body = resp.json()
    except ValueError:
        log.warning("[openai-stt] non-JSON response: %s", resp.text[:200])
        return ""

    text = (body.get("text") or "").strip()
    if text:
        log.info("[openai-stt] %d chars from %.2fs audio (%s)",
                 len(text), audio_f32_16k.size / 16000, OPENAI_STT_MODEL)
    return text
