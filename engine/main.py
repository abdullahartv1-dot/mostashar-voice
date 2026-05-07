"""FastAPI engine entrypoint."""
import logging
import shutil
import uuid
from pathlib import Path
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from . import config
from .schemas import HealthResponse
from .model_manager import ModelManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("vs2.engine")

app = FastAPI(title="Voice Studio v2 Engine")

# Single global model manager
mm = ModelManager(vram_budget_gb=config.VRAM_TOTAL_GB * config.VRAM_BUDGET_PCT)

# Serve generated audio files
config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(config.JOBS_DIR)), name="files")


def require_api_key(x_api_key: str = Header(None)) -> None:
    if x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health", response_model=HealthResponse)
def health():
    """Returns engine status — used by tunnel/proxy to verify aliveness."""
    gpu_used_gb = 0.0
    gpu_total_gb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_used_gb = torch.cuda.memory_allocated() / 1024 ** 3
            gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception as e:
        logger.warning(f"GPU info unavailable: {e}")
    return HealthResponse(
        status="ok",
        gpu_mem_used_gb=round(gpu_used_gb, 2),
        gpu_mem_total_gb=round(gpu_total_gb, 2),
        models_loaded=list(mm.loaded.keys()),
    )


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), _=Depends(require_api_key)):
    """Save uploaded audio to /workspace/voice-studio-v2/jobs/{uuid}/orig.{ext}."""
    job_id = uuid.uuid4().hex[:8]
    job_dir = config.JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix or ".bin"
    dest = job_dir / f"orig{ext}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    # Convert to 16k mono WAV for downstream services
    import subprocess, imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    wav_path = job_dir / "audio_16k.wav"
    subprocess.run(
        [ffmpeg, "-y", "-i", str(dest), "-ac", "1", "-ar", "16000", str(wav_path)],
        capture_output=True, check=True,
    )
    import soundfile as sf
    info = sf.info(str(wav_path))
    if info.duration > config.MAX_AUDIO_DURATION_SEC:
        raise HTTPException(status_code=413, detail=f"Audio exceeds {config.MAX_AUDIO_DURATION_SEC}s limit")

    return {
        "job_id": job_id,
        "audio_path": str(wav_path),
        "duration": info.duration,
        "size_bytes": dest.stat().st_size,
    }


import time as _time
from .schemas import ProcessRequest, ProcessResponse
from .services import STT_HANDLERS, DIAR_HANDLERS
from .cost_tracker import calc_cost
from . import jobs_store


def _extract_speaker_samples(wav_path, segments, job_dir):
    import soundfile as sf, numpy as np
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    by_spk = {}
    for s in segments:
        by_spk.setdefault(s.get("speaker", "SPEAKER_0"), []).append(s)
    silence = np.zeros(int(0.2 * sr))
    samples = {}
    for sp, segs in by_spk.items():
        chunks, total = [], 0.0
        for s in segs[:50]:
            if total >= 30: break
            si, ei = int(s["start"] * sr), int(s["end"] * sr)
            chunks.append(audio[si:ei])
            chunks.append(silence)
            total += s["end"] - s["start"]
        samp = np.concatenate(chunks) if chunks else np.zeros(int(0.5 * sr))
        path = job_dir / f"{sp}_sample.wav"
        sf.write(path, samp, sr)
        samples[sp] = {"url": f"/files/{job_dir.name}/{path.name}", "duration": round(len(samp)/sr, 2)}
    return samples


@app.post("/api/process", response_model=ProcessResponse)
def api_process(req: ProcessRequest, _=Depends(require_api_key)):
    if req.stt_tool not in STT_HANDLERS:
        raise HTTPException(400, f"Unknown stt_tool: {req.stt_tool}")

    audio_path = req.audio_path
    if not Path(audio_path).exists():
        raise HTTPException(404, f"Audio not found: {audio_path}")

    job_dir = Path(audio_path).parent
    job_id = job_dir.name

    timings = {}

    # STT
    t0 = _time.time()
    stt_result = STT_HANDLERS[req.stt_tool](audio_path)
    timings["stt_sec"] = round(_time.time() - t0, 2)
    segments = stt_result["segments"]

    # Diarization (skip if VibeVoice ASR — already provides speakers)
    diar_used = req.diar_tool
    if req.stt_tool != "vibevoice-asr":
        if not req.diar_tool or req.diar_tool not in DIAR_HANDLERS:
            raise HTTPException(400, f"diar_tool required for stt {req.stt_tool}")
        t0 = _time.time()
        diar_result = DIAR_HANDLERS[req.diar_tool](audio_path, segments)
        timings["diar_sec"] = round(_time.time() - t0, 2)
        segments = diar_result["segments"]
    else:
        timings["diar_sec"] = 0.0
        diar_used = "built-in (vibevoice-asr)"

    # Sample extraction
    t0 = _time.time()
    samples = _extract_speaker_samples(audio_path, segments, job_dir)
    timings["extract_sec"] = round(_time.time() - t0, 2)

    # Cost
    rate = config.GPU_RATE_PER_HOUR_USD
    cost = {
        "stt_usd": calc_cost(timings["stt_sec"], rate),
        "diar_usd": calc_cost(timings["diar_sec"], rate),
        "extract_usd": calc_cost(timings["extract_sec"], rate),
    }
    cost["total_usd"] = round(sum(cost.values()), 4)

    duration = stt_result["duration"]
    job_data = {
        "id": job_id,
        "audio_path": audio_path,
        "segments": segments,
        "samples": samples,
        "timings": timings,
        "cost": cost,
        "tools_used": {"stt": req.stt_tool, "diar": diar_used},
        "duration": duration,
    }
    if jobs_store.get_job(job_id):
        jobs_store.update_job(job_id, **job_data)
    else:
        jobs_store._JOBS.update({job_id: job_data})

    return ProcessResponse(
        job_id=job_id,
        duration=duration,
        segments=segments,
        samples=samples,
        timings=timings,
        cost=cost,
        tools_used={"stt": req.stt_tool, "diar": diar_used},
    )


from .schemas import CloneRequest, CloneResponse
from .services import TTS_HANDLERS


@app.post("/api/clone", response_model=CloneResponse)
def api_clone(req: CloneRequest, _=Depends(require_api_key)):
    if req.tts_tool not in TTS_HANDLERS:
        raise HTTPException(400, f"Unknown tts_tool: {req.tts_tool}")

    job = jobs_store.get_job(req.job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {req.job_id}")

    samples = job.get("samples", {})
    if req.speaker_id not in samples:
        raise HTTPException(400, f"Speaker {req.speaker_id} not in job samples")

    sample_url = samples[req.speaker_id]["url"]  # /files/{job}/{name}
    sample_path = config.JOBS_DIR / sample_url.split("/files/", 1)[1]

    if len(req.text) > config.MAX_TEXT_CHARS:
        raise HTTPException(400, f"Text exceeds {config.MAX_TEXT_CHARS} chars")

    t0 = _time.time()
    handler = TTS_HANDLERS[req.tts_tool]
    if req.tts_tool.startswith("vibevoice"):
        out_path_str = handler(
            text=req.text,
            ref=str(sample_path),
            diffusion_steps=req.diffusion_steps,
            cfg_scale=req.cfg_scale,
            seed=req.seed,
        )
    else:
        out_path_str = handler(
            text=req.text,
            ref=str(sample_path),
        )
    elapsed = _time.time() - t0

    out_path = Path(out_path_str)
    import soundfile as sf
    out_duration = sf.info(str(out_path)).duration

    final = config.JOBS_DIR / req.job_id / f"clone_{int(_time.time())}.wav"
    final.parent.mkdir(parents=True, exist_ok=True)
    if str(out_path) != str(final):
        shutil.move(str(out_path), str(final))

    cost = calc_cost(elapsed, config.GPU_RATE_PER_HOUR_USD)
    return CloneResponse(
        audio_url=f"/files/{req.job_id}/{final.name}",
        duration=round(out_duration, 2),
        elapsed=round(elapsed, 2),
        rtf=round(elapsed / out_duration, 2) if out_duration > 0 else 0,
        cost_usd=cost,
        tool=req.tts_tool,
    )


@app.get("/api/jobs")
def api_jobs(_=Depends(require_api_key)):
    return [
        {
            "id": j["id"],
            "duration": j.get("duration"),
            "tools_used": j.get("tools_used"),
            "cost": j.get("cost"),
        }
        for j in jobs_store.all_jobs().values()
    ]


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str, _=Depends(require_api_key)):
    j = jobs_store.get_job(job_id)
    if not j:
        raise HTTPException(404, f"Job not found: {job_id}")
    return j


@app.get("/api/costs")
def api_costs(_=Depends(require_api_key)):
    total = 0.0
    for j in jobs_store.all_jobs().values():
        c = j.get("cost", {})
        total += c.get("total_usd", 0)
    return {
        "total_usd": round(total, 4),
        "rate_per_hour": config.GPU_RATE_PER_HOUR_USD,
        "jobs_count": len(jobs_store.all_jobs()),
    }
