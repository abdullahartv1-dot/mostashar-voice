"""STREAMING TTS server v2 — uses AudioStreamer to emit chunks AS THE MODEL PRODUCES THEM.

Discovery (2026-05-09): VibeVoice-Large's generate() ALREADY supports audio_streamer.put()
inside its diffusion loop (line 657 of modeling_vibevoice_inference.py). Each diffusion
step emits a small audio chunk (~140ms). We pipe these directly to WebSocket.

Expected TTFA: ~150-300ms (first diffusion chunk after model setup).
"""
import os
import io
import time
import json
import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Queue

import numpy as np
import soundfile as sf
import librosa
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse

MODEL_NAME = os.environ.get("VV_MODEL", "aoi-ot/VibeVoice-Large")
DIFFUSION_STEPS = int(os.environ.get("VV_DIFF_STEPS", "15"))  # sweet spot from exp1
CFG_SCALE = float(os.environ.get("VV_CFG", "1.8"))
SAMPLE_RATE = 24000
REF_AUDIO_PATH = os.environ.get("VV_REF", "/workspace/refs/01.mp3")

print(f"=== streaming server v2 (AudioStreamer) ===")
print(f"  MODEL: {MODEL_NAME}")
print(f"  DIFFUSION_STEPS: {DIFFUSION_STEPS}")
print(f"  CFG_SCALE: {CFG_SCALE}")

processor = None
model = None
voice_cache = {}


def get_voice(voice_id: str) -> np.ndarray:
    if voice_id in voice_cache:
        return voice_cache[voice_id]
    path = REF_AUDIO_PATH if voice_id == "default" else f"/workspace/refs/{voice_id}.mp3"
    if not os.path.exists(path):
        raise ValueError(f"voice not found: {voice_id}")
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    audio = audio[:30 * SAMPLE_RATE]
    voice_cache[voice_id] = audio
    print(f"  cached voice '{voice_id}' from {path} ({len(audio)/SAMPLE_RATE:.1f}s)")
    return audio


def chunk_to_wav_bytes(audio_tensor: torch.Tensor, sr: int = SAMPLE_RATE) -> bytes:
    """Convert a single audio chunk tensor → 16-bit PCM WAV bytes (with header)."""
    audio_np = audio_tensor.detach().cpu().float().numpy().flatten()
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, audio_int16, sr, format='WAV', subtype='PCM_16')
    return buf.getvalue()


def chunk_to_pcm_bytes(audio_tensor: torch.Tensor) -> bytes:
    """Convert audio chunk → raw PCM 16-bit (no header). For continuous streaming."""
    audio_np = audio_tensor.detach().cpu().float().numpy().flatten()
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    return audio_int16.tobytes()


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
    print(f"[startup] loaded in {time.time()-t0:.1f}s, GPU mem={torch.cuda.memory_allocated()/1e9:.1f}GB")
    try:
        get_voice("default")
        # Pre-warm with a single short generation
        print("[startup] warming up...")
        ref = get_voice("default")
        inputs = processor(text=["Speaker 1: مرحبا"], voice_samples=[[ref]], return_tensors="pt", padding=True)
        inputs = {k: (v.to("cuda") if hasattr(v, 'to') else v) for k, v in inputs.items()}
        torch.manual_seed(42)
        with torch.no_grad():
            _ = model.generate(
                **inputs, max_new_tokens=None, cfg_scale=CFG_SCALE,
                tokenizer=processor.tokenizer,
                generation_config={'do_sample': False, 'num_beams': 1},
                verbose=False, show_progress_bar=False,
            )
        print(f"[startup] warmup done")
    except Exception as e:
        print(f"[startup] warmup error: {e}")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    here = Path(__file__).parent
    p = here / "client_v2.html"
    if p.exists():
        return FileResponse(p)
    p2 = here / "client.html"
    if p2.exists():
        return FileResponse(p2)
    return HTMLResponse("<h1>v2 streaming server</h1>")


@app.get("/health")
async def health():
    return {
        "version": "v2-streaming",
        "model": MODEL_NAME,
        "diffusion_steps": DIFFUSION_STEPS,
        "cfg_scale": CFG_SCALE,
        "voices": list(voice_cache.keys()),
        "gpu_mem_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
    }


@app.websocket("/tts/stream")
async def tts_stream(ws: WebSocket):
    """v2 streaming endpoint — emits chunks AS the model generates them."""
    await ws.accept()
    log_prefix = f"[ws {id(ws) % 10000}]"
    request_start = time.time()

    try:
        data = await ws.receive_json()
        text = data["text"]
        voice_id = data.get("voice_id", "default")
        print(f"{log_prefix} text='{text[:60]}...' voice={voice_id}")

        ref_audio = get_voice(voice_id)
        speaker_text = "Speaker 1: " + text
        inputs = processor(
            text=[speaker_text], voice_samples=[[ref_audio]],
            return_tensors="pt", padding=True,
        )
        inputs = {k: (v.to("cuda") if hasattr(v, 'to') else v) for k, v in inputs.items()}

        # Send meta first
        await ws.send_json({"type": "meta", "sample_rate": SAMPLE_RATE, "format": "pcm16"})

        # Set up streamer (sync version since model.generate runs in thread)
        from vibevoice.modular.streamer import AudioStreamer
        streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=30.0)

        # Run generation in a thread
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

        # Async-pull from sync queue
        chunk_idx = 0
        first_chunk_time = None
        cumulative_audio_samples = 0

        while True:
            # Run blocking queue.get in executor to not block event loop
            loop = asyncio.get_event_loop()
            try:
                chunk = await loop.run_in_executor(
                    None,
                    lambda: streamer.audio_queues[0].get(timeout=30.0),
                )
            except Exception as e:
                print(f"{log_prefix} queue error: {e}")
                break

            if chunk is None or chunk is streamer.stop_signal:
                break

            if first_chunk_time is None:
                first_chunk_time = time.time()
                ttfa_ms = (first_chunk_time - request_start) * 1000
                print(f"{log_prefix} TTFA = {ttfa_ms:.0f}ms (first chunk)")
                await ws.send_json({"type": "ttfa", "ms": round(ttfa_ms, 0)})

            pcm_bytes = chunk_to_pcm_bytes(chunk)
            cumulative_audio_samples += len(chunk.flatten())
            await ws.send_bytes(pcm_bytes)

            if chunk_idx < 5 or chunk_idx % 20 == 0:
                ms = (time.time() - request_start) * 1000
                samples = len(chunk.flatten())
                print(f"{log_prefix}  chunk {chunk_idx}: {samples} samples ({samples/SAMPLE_RATE*1000:.0f}ms audio) at t+{ms:.0f}ms")
            chunk_idx += 1

        gen_thread.join(timeout=60)
        total_audio_s = cumulative_audio_samples / SAMPLE_RATE
        total_wall_ms = (time.time() - request_start) * 1000
        await ws.send_json({
            "type": "done",
            "total_chunks": chunk_idx,
            "total_audio_s": round(total_audio_s, 2),
            "total_wall_ms": round(total_wall_ms, 0),
        })
        print(f"{log_prefix} done: {chunk_idx} chunks, {total_audio_s:.2f}s audio, {total_wall_ms:.0f}ms wall")

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
