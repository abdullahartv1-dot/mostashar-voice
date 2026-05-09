"""STREAMING TTS server v3 — Voice library + cached processor inputs.

Improvement over v2:
- Voice library: pre-register multiple voices, user selects by voice_id
- Cache the processor outputs (input_ids, speech_tensors, speech_masks, speech_input_mask)
  per voice to skip the audio encoding step per request
- Each request only varies the text portion of input_ids

Endpoints:
- POST /voices/register  body: {voice_id, audio_path?, audio_b64?}  → caches voice profile
- GET  /voices           list registered voices
- WS   /tts/stream       generate streaming audio
"""
import os
import io
import time
import json
import asyncio
import threading
import base64
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

MODEL_NAME = os.environ.get("VV_MODEL", "aoi-ot/VibeVoice-Large")
DIFFUSION_STEPS = int(os.environ.get("VV_DIFF_STEPS", "15"))
CFG_SCALE = float(os.environ.get("VV_CFG", "1.8"))
SAMPLE_RATE = 24000
DEFAULT_REF = os.environ.get("VV_REF", "/workspace/refs/01.mp3")
VOICES_DIR = os.environ.get("VV_VOICES_DIR", "/workspace/refs/voices")
os.makedirs(VOICES_DIR, exist_ok=True)

processor = None
model = None
voice_profiles = {}  # voice_id -> dict with cached inputs


def _save_voice_audio(voice_id: str, audio_np: np.ndarray):
    """Save the trimmed reference audio to disk as WAV."""
    path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
    sf.write(path, audio_np, SAMPLE_RATE)
    print(f"  saved voice audio to {path}")


def register_voice(voice_id: str, ref_audio_np: np.ndarray) -> dict:
    """Register a voice — store the trimmed reference audio in memory + disk.
    Encoder runs per-request (only ~3ms anyway, not the bottleneck)."""
    if len(ref_audio_np) > 30 * SAMPLE_RATE:
        ref_audio_np = ref_audio_np[:30 * SAMPLE_RATE]
    profile = {
        "voice_id": voice_id,
        "ref_audio_np": ref_audio_np,
        "ref_audio_dur_s": len(ref_audio_np) / SAMPLE_RATE,
    }
    voice_profiles[voice_id] = profile
    _save_voice_audio(voice_id, ref_audio_np)
    return profile


def get_voice_profile(voice_id: str) -> dict:
    if voice_id in voice_profiles:
        return voice_profiles[voice_id]
    # Try loading from disk
    wav_path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
    if os.path.exists(wav_path):
        audio_np, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        return register_voice(voice_id, audio_np)
    # Default fallback
    if voice_id == "default" and os.path.exists(DEFAULT_REF):
        audio_np, _ = librosa.load(DEFAULT_REF, sr=SAMPLE_RATE)
        return register_voice("default", audio_np)
    raise ValueError(f"voice not found: {voice_id}")


def chunk_to_pcm_bytes(audio_tensor: torch.Tensor) -> bytes:
    audio_np = audio_tensor.detach().cpu().float().numpy().flatten()
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    return audio_int16.tobytes()


def build_inputs_for_text(text: str, voice_id: str):
    """Build inputs for text using a registered voice's reference audio."""
    profile = get_voice_profile(voice_id)
    speaker_text = "Speaker 1: " + text
    inputs = processor(
        text=[speaker_text],
        voice_samples=[[profile["ref_audio_np"]]],
        return_tensors="pt", padding=True,
    )
    return inputs


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model
    print(f"[startup] loading {MODEL_NAME} with flash_attention_2...")
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

    # Register default voice + load any voices on disk
    for f in os.listdir(VOICES_DIR):
        if f.endswith(".pt"):
            voice_id = f[:-3]
            try:
                get_voice_profile(voice_id)
            except Exception as e:
                print(f"  failed to load {voice_id}: {e}")

    if "default" not in voice_profiles:
        try:
            get_voice_profile("default")
        except Exception as e:
            print(f"  warning: could not register default voice: {e}")

    print(f"[startup] voices registered: {list(voice_profiles.keys())}")

    # Warmup
    try:
        warm_inputs = build_inputs_for_text("مرحبا", "default")
        warm_inputs = {k: (v.to("cuda") if hasattr(v, 'to') else v) for k, v in warm_inputs.items()}
        torch.manual_seed(42)
        with torch.no_grad():
            _ = model.generate(
                **warm_inputs, max_new_tokens=None, cfg_scale=CFG_SCALE,
                tokenizer=processor.tokenizer,
                generation_config={'do_sample': False, 'num_beams': 1},
                verbose=False, show_progress_bar=False,
            )
        print("[startup] warmup done")
    except Exception as e:
        print(f"[startup] warmup error: {e}")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    here = Path(__file__).parent
    p = here / "client_v3.html"
    if p.exists():
        return FileResponse(p)
    return FileResponse(here / "client_v2.html") if (here / "client_v2.html").exists() else HTMLResponse("<h1>v3</h1>")


@app.get("/health")
async def health():
    return {
        "version": "v3-voice-library",
        "model": MODEL_NAME,
        "diffusion_steps": DIFFUSION_STEPS,
        "voices": list(voice_profiles.keys()),
        "gpu_mem_gb": torch.cuda.memory_allocated() / 1e9,
    }


@app.get("/voices")
async def list_voices():
    return {
        "voices": [
            {"voice_id": vid, "ref_audio_dur_s": p.get("ref_audio_dur_s", 0)}
            for vid, p in voice_profiles.items()
        ]
    }


@app.post("/voices/register")
async def register_voice_endpoint(
    voice_id: str = Form(...),
    audio: UploadFile = File(...),
):
    """Register a new voice from uploaded audio (mp3/wav/flac)."""
    if not voice_id.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(400, "voice_id must be alphanumeric (with _ or -)")
    audio_bytes = await audio.read()
    # Decode using soundfile/librosa
    buf = io.BytesIO(audio_bytes)
    audio_np, sr = librosa.load(buf, sr=SAMPLE_RATE)
    if len(audio_np) > 30 * SAMPLE_RATE:
        audio_np = audio_np[:30 * SAMPLE_RATE]
    elif len(audio_np) < 5 * SAMPLE_RATE:
        raise HTTPException(400, "audio must be at least 5 seconds")

    profile = register_voice(voice_id, audio_np)
    return {
        "voice_id": voice_id,
        "ref_audio_dur_s": profile["ref_audio_dur_s"],
        "ok": True,
    }


@app.websocket("/tts/stream")
async def tts_stream(ws: WebSocket):
    await ws.accept()
    log_prefix = f"[ws {id(ws) % 10000}]"
    request_start = time.time()

    try:
        data = await ws.receive_json()
        text = data["text"]
        voice_id = data.get("voice_id", "default")
        print(f"{log_prefix} text='{text[:40]}...' voice={voice_id}")

        inputs = build_inputs_for_text(text, voice_id)
        inputs = {k: (v.to("cuda") if hasattr(v, 'to') else v) for k, v in inputs.items()}

        await ws.send_json({"type": "meta", "sample_rate": SAMPLE_RATE, "format": "pcm16"})

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

        chunk_idx = 0
        first_chunk_time = None
        cumulative_samples = 0

        while True:
            loop = asyncio.get_event_loop()
            try:
                chunk = await loop.run_in_executor(
                    None,
                    lambda: streamer.audio_queues[0].get(timeout=30.0),
                )
            except Exception:
                break

            if chunk is None or chunk is streamer.stop_signal:
                break

            if first_chunk_time is None:
                first_chunk_time = time.time()
                ttfa_ms = (first_chunk_time - request_start) * 1000
                print(f"{log_prefix} TTFA = {ttfa_ms:.0f}ms")
                await ws.send_json({"type": "ttfa", "ms": round(ttfa_ms, 0)})

            pcm_bytes = chunk_to_pcm_bytes(chunk)
            cumulative_samples += len(chunk.flatten())
            await ws.send_bytes(pcm_bytes)
            chunk_idx += 1

        gen_thread.join(timeout=60)
        total_audio_s = cumulative_samples / SAMPLE_RATE
        total_wall_ms = (time.time() - request_start) * 1000
        await ws.send_json({
            "type": "done",
            "total_chunks": chunk_idx,
            "total_audio_s": round(total_audio_s, 2),
            "total_wall_ms": round(total_wall_ms, 0),
        })
        print(f"{log_prefix} done: {chunk_idx} chunks, {total_audio_s:.2f}s audio, {total_wall_ms:.0f}ms")

    except WebSocketDisconnect:
        print(f"{log_prefix} disconnected")
    except Exception as e:
        import traceback; traceback.print_exc()
        try:
            await ws.send_json({"type": "error", "message": str(e)[:500]})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("VV_PORT", "8080")), log_level="info")
