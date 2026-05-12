"""ElevenLabs TTS — high-quality streaming Arabic voice generation.

Why this exists: VibeVoice produces good Arabic but needs careful
prompting (periods not commas, proper orthography) and still drifts
into Persian/Chinese-sounding tokens on long inputs. ElevenLabs's
multilingual model handles Arabic dialects natively, ships clean
streaming PCM, and produces noticeably more natural prosody.

API: https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream

We request `pcm_24000` output so the bytes are a drop-in replacement
for VibeVoice's stream — both are int16 LE 24 kHz mono raw PCM, which
is exactly what the frontend PCMPlayer expects.

Env:
  MV_ELEVENLABS_KEY        — required, the xi-api-key
  MV_ELEVENLABS_VOICE_ID   — default 21m00Tcm4TlvDq8ikWAM (Rachel)
  MV_ELEVENLABS_MODEL_ID   — default eleven_multilingual_v2
  MV_ELEVENLABS_OUTPUT_FORMAT — default pcm_24000
"""

from __future__ import annotations

import os
from typing import AsyncGenerator, Dict, List, Optional

import httpx


ELEVENLABS_BASE = os.environ.get(
    "MV_ELEVENLABS_BASE", "https://api.elevenlabs.io"
)
DEFAULT_VOICE_ID = os.environ.get(
    "MV_ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"  # Rachel (multilingual)
)
DEFAULT_MODEL_ID = os.environ.get(
    "MV_ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"
)
DEFAULT_OUTPUT_FORMAT = os.environ.get(
    "MV_ELEVENLABS_OUTPUT_FORMAT", "pcm_24000"
)
# Reading-style settings. Higher stability = more consistent voice across
# the response; higher similarity boost = closer to the source voice's
# acoustic fingerprint. style is 0..1 for emotional expressiveness.
STABILITY = float(os.environ.get("MV_ELEVENLABS_STABILITY", "0.5"))
SIMILARITY_BOOST = float(os.environ.get("MV_ELEVENLABS_SIMILARITY", "0.75"))
STYLE = float(os.environ.get("MV_ELEVENLABS_STYLE", "0.0"))
USE_SPEAKER_BOOST = (
    os.environ.get("MV_ELEVENLABS_SPEAKER_BOOST", "1") not in ("0", "false", "False", "")
)
REQUEST_TIMEOUT_S = float(os.environ.get("MV_ELEVENLABS_TIMEOUT_S", "30"))


async def elevenlabs_stream(
    text: str,
    voice_id: Optional[str] = None,
    *,
    model_id: Optional[str] = None,
    output_format: Optional[str] = None,
    optimize_streaming_latency: int = 2,
) -> AsyncGenerator[bytes, None]:
    """Stream PCM16 LE bytes from ElevenLabs as the model produces them.

    Each yielded chunk is a slice of the raw audio stream — usually
    1024-4096 bytes — that we forward straight to the user's WebSocket.
    No header, no encoding wrapper. PCM at 24 kHz so the frontend's
    pre-existing 24 kHz worklet plays it without resampling.

    optimize_streaming_latency:
      0 — best quality, slowest first byte
      1-4 — progressively faster TTFA, slightly lower quality
      We default to 2 which is a good balance.

    Raises RuntimeError if MV_ELEVENLABS_KEY is unset. Caller should
    catch and fall back to local VibeVoice in that case.
    """
    api_key = os.environ.get("MV_ELEVENLABS_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MV_ELEVENLABS_KEY env var not set")

    vid = voice_id or DEFAULT_VOICE_ID
    mid = model_id or DEFAULT_MODEL_ID
    fmt = output_format or DEFAULT_OUTPUT_FORMAT

    url = f"{ELEVENLABS_BASE}/v1/text-to-speech/{vid}/stream"
    params = {
        "output_format": fmt,
        "optimize_streaming_latency": str(optimize_streaming_latency),
    }
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg" if fmt.startswith("mp3") else "audio/pcm",
    }
    payload = {
        "text": text,
        "model_id": mid,
        "voice_settings": {
            "stability": STABILITY,
            "similarity_boost": SIMILARITY_BOOST,
            "style": STYLE,
            "use_speaker_boost": USE_SPEAKER_BOOST,
        },
    }

    print(
        f"[elevenlabs] streaming {len(text)} chars to {vid} ({mid}, {fmt})",
        flush=True,
    )

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
            async with client.stream(
                "POST", url, params=params, headers=headers, json=payload,
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    print(
                        f"[elevenlabs] HTTP {resp.status_code}: {body[:300]!r}",
                        flush=True,
                    )
                    return
                bytes_total = 0
                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    if chunk:
                        bytes_total += len(chunk)
                        yield chunk
                print(
                    f"[elevenlabs] streamed {bytes_total} bytes "
                    f"({bytes_total / (24000 * 2):.1f}s of audio)",
                    flush=True,
                )
    except httpx.TimeoutException:
        print(f"[elevenlabs] timeout after {REQUEST_TIMEOUT_S:.0f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[elevenlabs] streaming error: {type(e).__name__}: {e}", flush=True)


async def list_voices() -> List[Dict[str, str]]:
    """Fetch the user's available voices from ElevenLabs.

    Returns list of {voice_id, name, labels} so the frontend can offer
    a voice picker that matches their actual account. Used by /v1/voices
    when MV_TTS_BACKEND=elevenlabs.
    """
    api_key = os.environ.get("MV_ELEVENLABS_KEY", "").strip()
    if not api_key:
        return []
    headers = {"xi-api-key": api_key}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{ELEVENLABS_BASE}/v1/voices", headers=headers)
        if resp.status_code != 200:
            print(f"[elevenlabs] list_voices HTTP {resp.status_code}", flush=True)
            return []
        body = resp.json()
        out: List[Dict[str, str]] = []
        for v in body.get("voices", []):
            out.append({
                "voice_id": v.get("voice_id", ""),
                "name": v.get("name", ""),
                "labels": v.get("labels", {}),
            })
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[elevenlabs] list_voices error: {e}", flush=True)
        return []
