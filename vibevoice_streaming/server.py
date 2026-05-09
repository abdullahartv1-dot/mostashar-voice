"""Streaming TTS server using VibeVoice-Large with sub-sentence chunking.

Architecture:
- Load VibeVoice-Large once at startup with flash_attention_2
- Cache the voice prefill per voice_id (acoustic_tokenizer + LM prefill is the slow part)
- On WebSocket: receive text, split into chunks, generate each, stream WAV bytes
- Client receives binary WAV chunks via WebSocket, plays gapless via Web Audio API

Sub-sentence chunking strategy:
- Split on Arabic period . and ! ? ؟
- Optionally split further on ، , (configurable via chunk_strategy)
- Each chunk goes through model.generate() with same cached voice
"""
import os
import re
import io
import time
import json
import struct
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---- config ----
MODEL_NAME = os.environ.get("VV_MODEL", "aoi-ot/VibeVoice-Large")
DIFFUSION_STEPS = int(os.environ.get("VV_DIFF_STEPS", "60"))
CFG_SCALE = float(os.environ.get("VV_CFG", "1.8"))
CHUNK_STRATEGY = os.environ.get("VV_CHUNK", "sentence")  # "sentence" or "sub_sentence"
SAMPLE_RATE = 24000
REF_AUDIO_PATH = os.environ.get("VV_REF", "/workspace/refs/01.mp3")

print(f"=== streaming server config ===")
print(f"  MODEL: {MODEL_NAME}")
print(f"  DIFFUSION_STEPS: {DIFFUSION_STEPS}")
print(f"  CFG_SCALE: {CFG_SCALE}")
print(f"  CHUNK_STRATEGY: {CHUNK_STRATEGY}")
print(f"  REF_AUDIO: {REF_AUDIO_PATH}")

# ---- model singleton (loaded on startup) ----
processor = None
model = None
voice_cache = {}  # voice_id -> reference_audio np.array (loaded once)


def split_text(text: str, strategy: str = "sentence") -> list[str]:
    """Split Arabic text into chunks at natural boundaries."""
    text = text.strip()
    if strategy == "sentence":
        # Split on . ! ? ؟ keeping the punctuation with the chunk
        parts = re.split(r'(?<=[\.\!\?\؟])\s+', text)
    elif strategy == "sub_sentence":
        # Split on . ! ? ؟ ، , (more aggressive)
        parts = re.split(r'(?<=[\.\!\?\؟\،\,])\s+', text)
    else:
        parts = [text]
    return [p.strip() for p in parts if p.strip()]


def get_voice(voice_id: str) -> np.ndarray:
    """Get cached reference audio. Loads from disk on first call."""
    if voice_id in voice_cache:
        return voice_cache[voice_id]
    # For now, voice_id "default" maps to REF_AUDIO_PATH
    if voice_id == "default":
        path = REF_AUDIO_PATH
    else:
        path = f"/workspace/refs/{voice_id}.mp3"
    if not os.path.exists(path):
        raise ValueError(f"voice not found: {voice_id} (looked at {path})")
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    audio = audio[:30 * SAMPLE_RATE]  # cap at 30s
    voice_cache[voice_id] = audio
    print(f"  cached voice '{voice_id}' from {path} ({len(audio)/SAMPLE_RATE:.1f}s)")
    return audio


def wav_bytes(audio_np: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """Convert float32 audio to 16-bit PCM WAV bytes."""
    buf = io.BytesIO()
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    sf.write(buf, audio_int16, sr, format='WAV', subtype='PCM_16')
    return buf.getvalue()


def generate_chunk(text: str, ref_audio: np.ndarray) -> tuple[np.ndarray, float]:
    """Generate one audio chunk for the given text using cached voice. Returns (audio, gen_time_s)."""
    speaker_text = "Speaker 1: " + text
    inputs = processor(
        text=[speaker_text],
        voice_samples=[[ref_audio]],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: (v.to("cuda") if hasattr(v, 'to') else v) for k, v in inputs.items()}
    torch.manual_seed(42)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=CFG_SCALE,
            tokenizer=processor.tokenizer,
            generation_config={'do_sample': False, 'num_beams': 1},
            verbose=False,
        )
    elapsed = time.time() - t0
    audio = out.speech_outputs[0].cpu().float().numpy().flatten()
    return audio, elapsed


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, free on shutdown."""
    global processor, model
    print(f"\n[startup] loading {MODEL_NAME} with flash_attention_2...")
    t0 = time.time()
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    processor = VibeVoiceProcessor.from_pretrained(MODEL_NAME)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=DIFFUSION_STEPS)
    print(f"[startup] loaded in {time.time()-t0:.1f}s, GPU mem={torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Pre-warm with default voice
    try:
        get_voice("default")
        print(f"[startup] pre-warmed default voice")
    except Exception as e:
        print(f"[startup] warning: could not pre-warm default voice: {e}")

    yield

    print("[shutdown] cleaning up")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    """Serve test client."""
    here = Path(__file__).parent
    client_path = here / "client.html"
    if client_path.exists():
        return FileResponse(client_path)
    return HTMLResponse("<h1>VibeVoice Streaming Server</h1><p>client.html not found</p>")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "diffusion_steps": DIFFUSION_STEPS,
        "cfg_scale": CFG_SCALE,
        "chunk_strategy": CHUNK_STRATEGY,
        "voices_cached": list(voice_cache.keys()),
        "gpu_mem_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
    }


@app.websocket("/tts/stream")
async def tts_stream(ws: WebSocket):
    """Stream TTS audio to client via WebSocket.

    Client sends: {"text": "...", "voice_id": "default", "chunk_strategy": "sentence"}
    Server sends:
      - text frame: {"type": "meta", "n_chunks": N, "sample_rate": 24000}
      - text frame: {"type": "chunk_meta", "i": 0, "text": "...", "gen_time": 1.23, "duration": 4.5}
      - binary frame: WAV bytes for chunk 0
      - text frame: {"type": "chunk_meta", "i": 1, ...}
      - binary frame: WAV bytes for chunk 1
      - ...
      - text frame: {"type": "done"}
    """
    await ws.accept()
    request_start = time.time()
    log_prefix = f"[ws {id(ws) % 10000}]"

    try:
        data = await ws.receive_json()
        text = data["text"]
        voice_id = data.get("voice_id", "default")
        chunk_strategy = data.get("chunk_strategy", CHUNK_STRATEGY)
        print(f"{log_prefix} received: text='{text[:60]}...' voice={voice_id} strategy={chunk_strategy}")

        ref_audio = get_voice(voice_id)
        chunks = split_text(text, chunk_strategy)
        await ws.send_json({
            "type": "meta",
            "n_chunks": len(chunks),
            "sample_rate": SAMPLE_RATE,
            "request_start_ms": 0,
        })
        print(f"{log_prefix} split into {len(chunks)} chunks")

        cumulative_audio_dur = 0.0
        for i, chunk_text in enumerate(chunks):
            audio, gen_time = generate_chunk(chunk_text, ref_audio)
            audio_dur = len(audio) / SAMPLE_RATE
            cumulative_audio_dur += audio_dur

            ms_since_request = (time.time() - request_start) * 1000
            ttfa_marker = f" TTFA={ms_since_request:.0f}ms" if i == 0 else ""
            print(f"{log_prefix}   chunk {i}: '{chunk_text[:40]}...' gen={gen_time:.2f}s audio={audio_dur:.2f}s{ttfa_marker}")

            await ws.send_json({
                "type": "chunk_meta",
                "i": i,
                "text": chunk_text,
                "gen_time_s": round(gen_time, 3),
                "audio_duration_s": round(audio_dur, 3),
                "ms_since_request": round(ms_since_request, 0),
            })
            await ws.send_bytes(wav_bytes(audio))

        total_time = (time.time() - request_start) * 1000
        await ws.send_json({
            "type": "done",
            "total_chunks": len(chunks),
            "total_audio_s": round(cumulative_audio_dur, 2),
            "total_wall_ms": round(total_time, 0),
        })
        print(f"{log_prefix} done: {len(chunks)} chunks, {cumulative_audio_dur:.2f}s audio, {total_time:.0f}ms wall")

    except WebSocketDisconnect:
        print(f"{log_prefix} client disconnected")
    except Exception as e:
        import traceback; traceback.print_exc()
        try:
            await ws.send_json({"type": "error", "message": str(e)[:500]})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("VV_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
