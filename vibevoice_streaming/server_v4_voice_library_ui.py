"""STREAMING TTS server v4 — ElevenLabs-style Voice Library UI.

New in v4:
- POST /voices/clone   accepts: voice_id, name, language, audio file, start_s, end_s
- GET  /voices         returns list with name, language, preview_url
- GET  /voices/{id}/preview  serves the trimmed reference audio
- DELETE /voices/{id}
- WS   /tts/stream     uses cached voice (no re-cloning per request)

Voice profiles are stored on disk:
  /workspace/refs/voices/{voice_id}.wav   trimmed reference audio
  /workspace/refs/voices/{voice_id}.json  metadata (name, language, dur)
"""
import os, io, time, json, asyncio, threading, uuid, re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import librosa
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response

MODEL_NAME = os.environ.get("VV_MODEL", "aoi-ot/VibeVoice-Large")
DIFFUSION_STEPS = int(os.environ.get("VV_DIFF_STEPS", "15"))
CFG_SCALE = float(os.environ.get("VV_CFG", "1.8"))
SAMPLE_RATE = 24000
DEFAULT_REF = os.environ.get("VV_REF", "/workspace/refs/01.mp3")
VOICES_DIR = os.environ.get("VV_VOICES_DIR", "/workspace/refs/voices")
os.makedirs(VOICES_DIR, exist_ok=True)

processor = None
model = None
voice_profiles = {}  # voice_id -> {ref_audio_np, name, language, dur_s}


def _meta_path(voice_id):    return os.path.join(VOICES_DIR, f"{voice_id}.json")
def _wav_path(voice_id):     return os.path.join(VOICES_DIR, f"{voice_id}.wav")


def _load_voice_from_disk(voice_id: str):
    wp = _wav_path(voice_id)
    mp = _meta_path(voice_id)
    if not os.path.exists(wp):
        return None
    audio_np, _ = librosa.load(wp, sr=SAMPLE_RATE)
    meta = {"voice_id": voice_id, "name": voice_id, "language": "ar"}
    if os.path.exists(mp):
        try:
            meta.update(json.load(open(mp, encoding="utf-8")))
        except Exception:
            pass
    profile = {
        "voice_id": voice_id,
        "ref_audio_np": audio_np,
        "ref_audio_dur_s": len(audio_np) / SAMPLE_RATE,
        "name": meta.get("name", voice_id),
        "language": meta.get("language", "ar"),
        "created_at": meta.get("created_at"),
    }
    voice_profiles[voice_id] = profile
    return profile


def get_voice(voice_id: str):
    if voice_id in voice_profiles:
        return voice_profiles[voice_id]
    p = _load_voice_from_disk(voice_id)
    if p is not None:
        return p
    raise HTTPException(404, f"voice not found: {voice_id}")


def save_voice(voice_id: str, audio_np: np.ndarray, name: str, language: str):
    """Trim to <=30s, save WAV + JSON metadata."""
    if len(audio_np) > 30 * SAMPLE_RATE:
        audio_np = audio_np[:30 * SAMPLE_RATE]
    sf.write(_wav_path(voice_id), audio_np, SAMPLE_RATE)
    meta = {
        "voice_id": voice_id,
        "name": name,
        "language": language,
        "created_at": int(time.time()),
        "dur_s": len(audio_np) / SAMPLE_RATE,
    }
    with open(_meta_path(voice_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    profile = {
        "voice_id": voice_id,
        "ref_audio_np": audio_np,
        "ref_audio_dur_s": meta["dur_s"],
        "name": name,
        "language": language,
        "created_at": meta["created_at"],
    }
    voice_profiles[voice_id] = profile
    return profile


def chunk_to_pcm_bytes(audio_tensor):
    audio_np = audio_tensor.detach().cpu().float().numpy().flatten()
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    return audio_int16.tobytes()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model
    print(f"[startup] loading {MODEL_NAME}...")
    t0 = time.time()
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    processor = VibeVoiceProcessor.from_pretrained(MODEL_NAME)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=DIFFUSION_STEPS)
    print(f"[startup] loaded in {time.time()-t0:.1f}s")

    # Auto-discover voices on disk
    for f in os.listdir(VOICES_DIR):
        if f.endswith(".wav"):
            vid = f[:-4]
            try:
                _load_voice_from_disk(vid)
            except Exception as e:
                print(f"  failed to load {vid}: {e}")

    # Ensure default voice exists
    if "default" not in voice_profiles and os.path.exists(DEFAULT_REF):
        audio_np, _ = librosa.load(DEFAULT_REF, sr=SAMPLE_RATE)
        save_voice("default", audio_np, "الصوت الافتراضي", "ar")

    print(f"[startup] voices: {[(p['voice_id'], p['name']) for p in voice_profiles.values()]}")
    # Warmup with default voice
    if "default" in voice_profiles:
        try:
            ref = voice_profiles["default"]["ref_audio_np"]
            inputs = processor(text=["Speaker 1: مرحبا"], voice_samples=[[ref]], return_tensors="pt", padding=True)
            inputs = {k: (v.to("cuda") if hasattr(v, 'to') else v) for k, v in inputs.items()}
            torch.manual_seed(42)
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=None, cfg_scale=CFG_SCALE,
                    tokenizer=processor.tokenizer,
                    generation_config={'do_sample': False, 'num_beams': 1},
                    verbose=False, show_progress_bar=False)
            print("[startup] warmup done")
        except Exception as e:
            print(f"[startup] warmup error: {e}")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    here = Path(__file__).parent
    p = here / "client_v4.html"
    if p.exists():
        return FileResponse(p)
    return HTMLResponse("<h1>v4</h1>")


@app.get("/health")
async def health():
    return {"version": "v4", "model": MODEL_NAME, "diff_steps": DIFFUSION_STEPS,
            "voices": len(voice_profiles), "gpu_mem_gb": torch.cuda.memory_allocated()/1e9}


@app.get("/voices")
async def list_voices():
    voices = []
    for vid, p in voice_profiles.items():
        voices.append({
            "voice_id": vid,
            "name": p.get("name", vid),
            "language": p.get("language", "ar"),
            "dur_s": round(p.get("ref_audio_dur_s", 0), 2),
            "preview_url": f"/voices/{vid}/preview",
            "created_at": p.get("created_at"),
        })
    voices.sort(key=lambda v: -(v["created_at"] or 0))
    return {"voices": voices}


@app.get("/voices/{voice_id}/preview")
async def voice_preview(voice_id: str):
    if voice_id not in voice_profiles:
        _load_voice_from_disk(voice_id)
    if voice_id not in voice_profiles:
        raise HTTPException(404)
    return FileResponse(_wav_path(voice_id), media_type="audio/wav")


def _slugify(s: str) -> str:
    s = re.sub(r'[^\w؀-ۿ-]+', '-', s.strip(), flags=re.UNICODE)
    s = re.sub(r'-+', '-', s).strip('-')
    return s[:40] or "voice"


@app.post("/voices/clone")
async def clone_voice(
    name: str = Form(...),
    language: str = Form("ar"),
    start_s: float = Form(0.0),
    end_s: float = Form(30.0),
    audio: UploadFile = File(...),
):
    """Clone a voice. Trims audio to [start_s, end_s] (max 30s).
    Returns voice_id + redirect URL."""
    if not name.strip():
        raise HTTPException(400, "name required")
    audio_bytes = await audio.read()
    try:
        buf = io.BytesIO(audio_bytes)
        audio_np, _ = librosa.load(buf, sr=SAMPLE_RATE)
    except Exception as e:
        raise HTTPException(400, f"failed to decode audio: {e}")

    # Trim to selected region
    s0 = max(0, int(start_s * SAMPLE_RATE))
    s1 = min(len(audio_np), int(end_s * SAMPLE_RATE))
    if s1 - s0 < 3 * SAMPLE_RATE:
        raise HTTPException(400, "selection must be at least 3 seconds")
    audio_np = audio_np[s0:s1]
    if len(audio_np) > 30 * SAMPLE_RATE:
        audio_np = audio_np[:30 * SAMPLE_RATE]

    voice_id = f"{_slugify(name)}-{uuid.uuid4().hex[:6]}"
    profile = save_voice(voice_id, audio_np, name, language)
    print(f"[clone] saved {voice_id} '{name}' ({language}) {profile['ref_audio_dur_s']:.1f}s")
    return {
        "voice_id": voice_id,
        "name": name,
        "language": language,
        "dur_s": profile["ref_audio_dur_s"],
        "redirect": f"/#library?selected={voice_id}",
    }


@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    if voice_id == "default":
        raise HTTPException(403, "cannot delete default")
    if voice_id in voice_profiles:
        del voice_profiles[voice_id]
    for path in (_wav_path(voice_id), _meta_path(voice_id)):
        if os.path.exists(path):
            os.remove(path)
    return {"ok": True}


@app.websocket("/tts/stream")
async def tts_stream(ws: WebSocket):
    await ws.accept()
    request_start = time.time()
    log_prefix = f"[ws {id(ws) % 10000}]"
    try:
        data = await ws.receive_json()
        text = data["text"]
        voice_id = data.get("voice_id", "default")
        try:
            profile = get_voice(voice_id)
        except HTTPException:
            await ws.send_json({"type": "error", "message": f"voice not found: {voice_id}"})
            return

        ref_audio = profile["ref_audio_np"]
        speaker_text = "Speaker 1: " + text
        inputs = processor(text=[speaker_text], voice_samples=[[ref_audio]], return_tensors="pt", padding=True)
        inputs = {k: (v.to("cuda") if hasattr(v, 'to') else v) for k, v in inputs.items()}

        await ws.send_json({"type": "meta", "sample_rate": SAMPLE_RATE, "voice_name": profile.get("name")})

        from vibevoice.modular.streamer import AudioStreamer
        streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=30.0)

        def run_generate():
            try:
                torch.manual_seed(42)
                with torch.no_grad():
                    model.generate(
                        **inputs, max_new_tokens=None, cfg_scale=CFG_SCALE,
                        tokenizer=processor.tokenizer,
                        generation_config={'do_sample': False, 'num_beams': 1},
                        verbose=False, show_progress_bar=False,
                        audio_streamer=streamer,
                    )
            except Exception as e:
                import traceback; traceback.print_exc()
                streamer.end()

        gen_thread = threading.Thread(target=run_generate, daemon=True)
        gen_thread.start()

        first_chunk_time = None
        chunk_idx = 0
        cumulative = 0
        loop = asyncio.get_event_loop()
        while True:
            try:
                chunk = await loop.run_in_executor(None,
                    lambda: streamer.audio_queues[0].get(timeout=30.0))
            except Exception:
                break
            if chunk is None or chunk is streamer.stop_signal:
                break
            if first_chunk_time is None:
                first_chunk_time = time.time()
                ttfa = (first_chunk_time - request_start) * 1000
                print(f"{log_prefix} TTFA={ttfa:.0f}ms voice={voice_id}")
                await ws.send_json({"type": "ttfa", "ms": round(ttfa, 0)})
            await ws.send_bytes(chunk_to_pcm_bytes(chunk))
            cumulative += len(chunk.flatten())
            chunk_idx += 1

        gen_thread.join(timeout=60)
        await ws.send_json({
            "type": "done",
            "total_chunks": chunk_idx,
            "total_audio_s": round(cumulative / SAMPLE_RATE, 2),
            "total_wall_ms": round((time.time() - request_start) * 1000, 0),
        })
    except WebSocketDisconnect:
        print(f"{log_prefix} disconnected")
    except Exception as e:
        import traceback; traceback.print_exc()
        try: await ws.send_json({"type": "error", "message": str(e)[:500]})
        except Exception: pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("VV_PORT", "8080")), log_level="info")
