"""
OpenVoice v2 sidecar — reference-enhancement service.

Runs in its own venv (PyTorch + OpenVoice) to avoid polluting the main
VibeVoice server's environment. The main server calls this over HTTP at
localhost:8083 once per `/v1/voices/add` upload to produce a *clarity-
enhanced* reference audio that VibeVoice then clones from.

Why we run this only at clone time (not per-conversation):
  - VibeVoice generates ~24 kHz audio at ~1× realtime; adding a 200-300 ms
    voice-conversion step on every reply would visibly hurt UX.
  - The user is happy to wait 5-15 s ONCE during voice creation in
    exchange for permanent quality on every future reply.
  - The enhanced WAV is just a *better reference*; nothing about the
    realtime serving path changes.

Pipeline (per request):
  1. user_upload.wav (~10 s)             — voice they want VibeVoice to mimic
  2. donor.wav (e.g. hamed_saudi.wav)    — clarity-of-articulation donor
  3. extract user tone-color embedding   — captures timbre, pitch range
  4. apply user tone-color to donor      — donor's words/timing, user's voice
  5. enhanced.wav                        — VibeVoice's new reference

Endpoints:
  GET  /health                  → liveness + cached donors + VRAM
  GET  /v1/donors               → list pre-cached clarity donors
  POST /v1/enhance-reference    → multipart {target: wav, donor_id|donor_path}
                                  returns the enhanced WAV (binary)
  POST /v1/quality-compare      → multipart {before: wav, after: wav}
                                  returns spectral metrics for both

Env vars:
  OPENVOICE_PORT       (default: 8083)
  OPENVOICE_CKPT_DIR   (default: /workspace/openvoice/checkpoints_v2)
  OPENVOICE_DONORS_DIR (default: /workspace/refs/voices)
                       — directory of pre-cached donor WAVs (hamed_saudi.wav,
                         default.wav, …). Donor SE is computed on first use
                         and cached in memory for the process lifetime.
  OPENVOICE_DEFAULT_DONOR (default: hamed_saudi)
                       — the donor used when the caller doesn't pick one.

Run:
  /workspace/openvoice-venv/bin/python /workspace/x/vibevoice_streaming/openvoice_server.py
"""
from __future__ import annotations

import base64
import io
import os
import time
import asyncio
import tempfile
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import librosa
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, FileResponse, Response

# ---------- config ----------
PORT = int(os.environ.get("OPENVOICE_PORT", "8083"))
CKPT_DIR = os.environ.get(
    "OPENVOICE_CKPT_DIR", "/workspace/openvoice/checkpoints_v2"
)
DONORS_DIR = os.environ.get(
    "OPENVOICE_DONORS_DIR", "/workspace/refs/voices"
)
DEFAULT_DONOR = os.environ.get("OPENVOICE_DEFAULT_DONOR", "hamed_saudi")

# OpenVoice's tone-color converter operates at 22.05 kHz internally; we
# resample everything to a single rate so that what we hand back to the
# main server can be stored verbatim as a VibeVoice reference (24 kHz).
OUTPUT_SAMPLE_RATE = 24_000

# ---------- shared state ----------
converter = None                              # ToneColorConverter
_donor_cache: dict[str, "torch.Tensor"] = {}  # donor_id → cached SE tensor
_lock: Optional[asyncio.Lock] = None
_warmup_ms = 0


# ---------- lifecycle ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global converter, _lock, _warmup_ms
    _lock = asyncio.Lock()

    print(f"[openvoice] loading ToneColorConverter from {CKPT_DIR} …")
    t0 = time.time()
    try:
        from openvoice.api import ToneColorConverter
        converter = ToneColorConverter(
            os.path.join(CKPT_DIR, "converter", "config.json"),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        converter.load_ckpt(os.path.join(CKPT_DIR, "converter", "checkpoint.pth"))
    except Exception as e:
        print(f"[openvoice] FATAL: failed to load converter: {e}")
        traceback.print_exc()
        raise

    # Pre-cache the default donor's tone-color embedding so the FIRST clone
    # request doesn't pay the SE-extraction tax (~1-2 s on cold path).
    try:
        _ = await _get_donor_se(DEFAULT_DONOR)
        print(f"[openvoice] pre-cached donor: {DEFAULT_DONOR}")
    except Exception as e:
        # Non-fatal — sidecar still runs, just a slower first request.
        print(f"[openvoice] WARN: could not pre-cache donor {DEFAULT_DONOR}: {e}")

    _warmup_ms = int((time.time() - t0) * 1000)
    print(f"[openvoice] ready in {_warmup_ms} ms, "
          f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB"
          if torch.cuda.is_available() else
          f"[openvoice] ready in {_warmup_ms} ms (CPU)")
    yield


app = FastAPI(title="OpenVoice v2 reference-enhancement sidecar",
              lifespan=lifespan)


# ---------- helpers ----------
def _donor_path(donor_id: str) -> str:
    """Resolve a donor_id (e.g. 'hamed_saudi') to an absolute WAV path.

    Allows callers to pass either:
      - a bare donor_id we look up in DONORS_DIR
      - an absolute path (escape hatch for ad-hoc donors)
    """
    if os.path.isabs(donor_id) and os.path.exists(donor_id):
        return donor_id
    p = os.path.join(DONORS_DIR, f"{donor_id}.wav")
    if not os.path.exists(p):
        raise HTTPException(404, f"donor not found: {donor_id} (looked at {p})")
    return p


async def _get_donor_se(donor_id: str):
    """Return the cached speaker-embedding tensor for a donor.

    SE extraction runs the converter's encoder + a VAD pass over the donor
    audio. It's deterministic for a given file, so we cache by donor_id
    (file path resolved) for the process lifetime.
    """
    if donor_id in _donor_cache:
        return _donor_cache[donor_id]

    path = _donor_path(donor_id)
    assert _lock is not None
    async with _lock:
        # Re-check inside the lock to avoid duplicate extraction in a race.
        if donor_id in _donor_cache:
            return _donor_cache[donor_id]
        loop = asyncio.get_event_loop()
        se = await loop.run_in_executor(None, _extract_se_sync, path)
        _donor_cache[donor_id] = se
        return se


def _extract_se_sync(audio_path: str):
    """Run the OpenVoice SE extractor synchronously (heavy work)."""
    from openvoice import se_extractor
    se, _audio_name = se_extractor.get_se(
        audio_path,
        converter,
        vad=True,
    )
    return se


def _spectral_report(audio_path: str) -> dict:
    """Compute the same gold-reference metrics our main server uses,
    so the caller can show a before/after table to the user.

    Mirrors `vibevoice_streaming/_diag_noise.py` but inline so this
    sidecar has no dependency on the main package.
    """
    audio, sr = librosa.load(audio_path, sr=24_000, mono=True)
    if len(audio) < sr // 2:
        return {"error": "audio too short for spectral analysis"}

    # Speech-band energy (80 Hz - 4 kHz).
    stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    sp_band = (freqs >= 80) & (freqs <= 4000)
    sib_band = (freqs >= 4000) & (freqs <= 8000)
    hf_band = freqs >= 8000

    sp_e = float(stft[sp_band].mean())
    sib_e = float(stft[sib_band].mean()) if sib_band.any() else 0.0
    hf_e = float(stft[hf_band].mean()) if hf_band.any() else 0.0

    # Spectral flatness — Wiener entropy. Lower = more tonal (= cleaner).
    sf_arr = librosa.feature.spectral_flatness(y=audio).flatten()

    return {
        "duration_s": round(len(audio) / sr, 2),
        "peak": round(float(np.abs(audio).max()), 3),
        "spectral_flatness": round(float(sf_arr.mean()), 4),
        "sib_to_speech": round(sib_e / sp_e if sp_e > 0 else 0, 4),
        "hf_to_speech": round(hf_e / sp_e if sp_e > 0 else 0, 4),
    }


# ---------- endpoints ----------
@app.get("/health")
async def health():
    return {
        "status": "ok" if converter is not None else "loading",
        "loaded": converter is not None,
        "warmup_ms": _warmup_ms,
        "cached_donors": list(_donor_cache.keys()),
        "default_donor": DEFAULT_DONOR,
        "vram_gb": (
            round(torch.cuda.memory_allocated() / 1e9, 2)
            if torch.cuda.is_available() else 0
        ),
    }


@app.get("/v1/donors")
async def list_donors():
    """Discover available clarity donors in DONORS_DIR. The caller picks
    one (or accepts the default) when invoking /v1/enhance-reference."""
    if not os.path.isdir(DONORS_DIR):
        return {"donors_dir": DONORS_DIR, "donors": []}
    donors = []
    for fname in sorted(os.listdir(DONORS_DIR)):
        if not fname.endswith(".wav"):
            continue
        donor_id = fname[:-4]
        donors.append({
            "id": donor_id,
            "path": os.path.join(DONORS_DIR, fname),
            "cached": donor_id in _donor_cache,
            "is_default": donor_id == DEFAULT_DONOR,
        })
    return {
        "donors_dir": DONORS_DIR,
        "default_donor": DEFAULT_DONOR,
        "donors": donors,
    }


@app.post("/v1/enhance-reference")
async def enhance_reference(
    target: UploadFile = File(
        ...,
        description="The user's uploaded clone audio (provides voice color).",
    ),
    donor_id: str = Form(
        DEFAULT_DONOR,
        description="Clarity donor id (filename without .wav in DONORS_DIR), "
                    "or absolute path. Defaults to OPENVOICE_DEFAULT_DONOR.",
    ),
    return_metrics: bool = Form(
        True,
        description="If true, the response is JSON with a base64-ish struct. "
                    "If false, the body IS the WAV bytes (audio/wav).",
    ),
):
    """Apply the *target* user's tone color onto the *donor* clarity audio.

    The output WAV will sound like the user is reading what the donor said
    — same words, same timing, same articulation, but the user's vocal
    timbre. That output is then used as VibeVoice's clone reference.
    """
    if converter is None:
        raise HTTPException(503, "converter not loaded yet")

    raw = await target.read()
    if len(raw) < 8_000:
        raise HTTPException(400, "target audio too short (need >250 ms)")

    # 1. Stage the target to disk — OpenVoice's SE extractor wants paths.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tt:
        try:
            audio_np, _sr = librosa.load(io.BytesIO(raw), sr=22_050, mono=True)
        except Exception as e:
            os.unlink(tt.name)
            raise HTTPException(400, f"could not decode target audio: {e}")
        if len(audio_np) < 22_050:  # < 1 s
            os.unlink(tt.name)
            raise HTTPException(400, "target audio < 1 second after decode")
        sf.write(tt.name, audio_np, 22_050)
        target_path = tt.name

    out_path = tempfile.mktemp(suffix=".enhanced.wav")
    t0 = time.time()
    try:
        donor_path = _donor_path(donor_id)
        target_se = await _extract_target_se(target_path)
        donor_se = await _get_donor_se(donor_id)
        await _convert(donor_path, donor_se, target_se, out_path)
        gen_ms = int((time.time() - t0) * 1000)

        # Resample to 24 kHz mono so the main server can drop it straight
        # into its voice library without further conversion.
        out_audio, _ = librosa.load(out_path, sr=OUTPUT_SAMPLE_RATE, mono=True)
        sf.write(out_path, out_audio, OUTPUT_SAMPLE_RATE)

        if return_metrics:
            metrics = {
                "donor": _spectral_report(donor_path),
                "target": _spectral_report(target_path),
                "enhanced": _spectral_report(out_path),
            }
            with open(out_path, "rb") as f:
                wav_bytes = f.read()
            # Base64 encoding inflates by ~33% (cheaper than the ~3×
            # cost of a JSON list-of-ints) while keeping a single
            # round-trip alongside the metrics.
            return JSONResponse({
                "donor_id": donor_id,
                "gen_ms": gen_ms,
                "output_sample_rate": OUTPUT_SAMPLE_RATE,
                "metrics": metrics,
                "wav_b64_len": len(wav_bytes),
                "wav_b64": base64.b64encode(wav_bytes).decode("ascii"),
            })

        # Binary mode: just stream the WAV.
        return FileResponse(
            out_path,
            media_type="audio/wav",
            filename="enhanced.wav",
            headers={"X-Generation-Ms": str(gen_ms),
                     "X-Donor-Id": donor_id},
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"enhancement failed: {e}")
    finally:
        # Clean up temps. Note: in binary-mode we let FileResponse stream
        # then the OS will reap when the temp dir cycles — acceptable.
        try:
            os.unlink(target_path)
        except Exception:
            pass


async def _extract_target_se(target_path: str):
    """SE extraction for a one-shot target — NOT cached (every clone is
    different). Held under the lock so we don't OOM the GPU."""
    assert _lock is not None
    async with _lock:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _extract_se_sync, target_path)


async def _convert(donor_path, donor_se, target_se, out_path):
    """The actual tone-color conversion call. Synchronous under the
    GPU lock so concurrent /enhance requests don't overlap."""
    assert _lock is not None and converter is not None
    async with _lock:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, _convert_sync, donor_path, donor_se, target_se, out_path
        )


def _convert_sync(donor_path, donor_se, target_se, out_path):
    """Wrap converter.convert with a fixed seed for reproducibility."""
    converter.convert(  # type: ignore[union-attr]
        audio_src_path=donor_path,
        src_se=donor_se,
        tgt_se=target_se,
        output_path=out_path,
        # Tau controls how strongly the target's tone overrides the
        # donor's. 0.3 is OpenVoice's default and a good middle ground;
        # smaller values keep more of the donor's identity (== more like
        # hamed). We picked 0.3 to maximise the user's identity.
        tau=0.3,
    )


@app.post("/v1/quality-compare")
async def quality_compare(
    before: UploadFile = File(..., description="Original (raw) reference."),
    after: UploadFile = File(..., description="Enhanced reference."),
):
    """Diagnostic — compute spectral metrics for both files. Useful for
    showing a before/after table on the clone-confirmation UI."""
    paths = []
    for f in (before, after):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
            t.write(await f.read())
            paths.append(t.name)
    try:
        return {
            "before": _spectral_report(paths[0]),
            "after": _spectral_report(paths[1]),
        }
    finally:
        for p in paths:
            try: os.unlink(p)
            except Exception: pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=PORT,
        log_level="info",
    )
