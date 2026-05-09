"""
Gemma 4 E4B sidecar — direct audio-to-text-response service.

Runs in its own venv (transformers 5.8.0+) to avoid version conflicts
with the main VibeVoice server (transformers 4.51.3). The main server
calls this over HTTP at localhost:8081 for the conversation pipeline,
collapsing the previous ASR → LLM two-step into one call.

Why this design:
  - Gemma 4 understands Arabic audio natively (no ASR transcript needed).
  - One forward pass instead of two = ~60% lower per-turn latency.
  - 16 GB VRAM vs 31 GB for VibeVoice-ASR + Qwen2.5-7B.
  - Cleaner separation: audio model in its own process, easy to restart
    or replace independently of the main TTS server.

Endpoints:
  GET  /health              → liveness
  POST /v1/audio-chat       → multipart {file: wav, json: {system, history}}
                              returns {text, total_ms, gen_ms, audio_dur_s}

Env vars:
  GEMMA4_MODEL    (default: google/gemma-4-E4B-it)
  GEMMA4_PORT     (default: 8081)
  HF_HOME         (where to look for the model snapshot)

Run:
  /workspace/gemma4-venv/bin/python /workspace/gemma4_server.py
"""
from __future__ import annotations

import io
import os
import time
import asyncio
import json
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import librosa
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MODEL_ID = os.environ.get("GEMMA4_MODEL", "google/gemma-4-E4B-it")
PORT = int(os.environ.get("GEMMA4_PORT", "8081"))

processor = None
model = None
_lock: Optional[asyncio.Lock] = None

DEFAULT_SYSTEM_PROMPT = (
    "أنت مساعد صوتي ذكي لمنصة مستشار. استمع لكلام المستخدم بعناية ثم أجبه "
    "بالعربية الفصحى الواضحة، بنبرة ودودة ومحترفة. اجعل ردك جملة أو "
    "جملتين قصيرتين فقط لأن الرد سيُحوَّل إلى صوت ويسمعه المستخدم مباشرة. "
    "لا تستخدم أي لغة غير العربية. لا تستخدم Markdown أو قوائم. "
    "إذا لم تفهم الكلام بوضوح، اطلب من المستخدم إعادة سؤاله بإيجاز."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, _lock
    _lock = asyncio.Lock()
    print(f"[gemma4] loading {MODEL_ID}…")
    t0 = time.time()
    try:
        from transformers import AutoProcessor, AutoModelForMultimodalLM as ModelClass
        ModelName = "AutoModelForMultimodalLM"
    except ImportError:
        from transformers import AutoProcessor, AutoModelForImageTextToText as ModelClass
        ModelName = "AutoModelForImageTextToText"
    print(f"[gemma4] using {ModelName}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = ModelClass.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    print(f"[gemma4] loaded in {time.time()-t0:.1f}s, "
          f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    # Warmup so the first real request isn't 13 s.
    try:
        _silent = np.zeros(16000, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, _silent, 16000)
            await _generate(tmp.name, DEFAULT_SYSTEM_PROMPT, [], 32)
        print("[gemma4] warmup done")
    except Exception as e:
        print(f"[gemma4] warmup error: {e}")
    yield


app = FastAPI(title="Gemma 4 audio-chat sidecar", lifespan=lifespan)


def _build_messages(system: str, history: list[dict], audio_path: str) -> list[dict]:
    """Build a chat-formatted message list with prior history + new audio turn."""
    messages: list[dict] = []
    if system:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system}],
        })
    # Replay prior conversation as text-only turns (history is text→text).
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({
                "role": role,
                "content": [{"type": "text", "text": content}],
            })
    # New user turn carries the audio.
    messages.append({
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": "استمع لكلامي ثم أجبني."},
        ],
    })
    return messages


async def _generate(audio_path: str, system: str, history: list[dict], max_tokens: int) -> dict:
    """Sync generate, awaitable. Holds the lock so concurrent calls serialize."""
    assert _lock is not None and processor is not None and model is not None
    async with _lock:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, _generate_sync, audio_path, system, history, max_tokens
        )


def _generate_sync(audio_path: str, system: str, history: list[dict], max_tokens: int) -> dict:
    messages = _build_messages(system, history, audio_path)
    t0 = time.time()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    prep_ms = (time.time() - t0) * 1000

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    gen_ms = (time.time() - t0) * 1000
    new_ids = outputs[0, inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "prep_ms": round(prep_ms),
        "gen_ms": round(gen_ms),
        "total_ms": round(prep_ms + gen_ms),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "loaded": model is not None,
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if model else 0,
    }


class HistoryEntry(BaseModel):
    role: str
    content: str


@app.post("/v1/audio-chat")
async def audio_chat(
    file: UploadFile = File(..., description="WAV (16 kHz mono PCM16LE)"),
    system: Optional[str] = Form(None, description="Override system prompt"),
    history: Optional[str] = Form(
        None, description='JSON list: [{"role":"user","content":"..."}, ...]'
    ),
    max_tokens: int = Form(160),
):
    """One conversation turn: audio in → assistant text out (no TTS here)."""
    audio_bytes = await file.read()
    # Decode + resample to 16 kHz mono — Gemma's audio encoder expects 16 kHz.
    try:
        buf = io.BytesIO(audio_bytes)
        audio_np, _sr = librosa.load(buf, sr=16000, mono=True)
    except Exception as e:
        raise HTTPException(400, f"failed to decode audio: {e}")
    if len(audio_np) < 800:  # < 50 ms
        raise HTTPException(400, "audio too short")
    if len(audio_np) > 30 * 16000:  # > 30 s — Gemma's audio limit
        audio_np = audio_np[: 30 * 16000]
    audio_dur_s = len(audio_np) / 16000

    # Gemma's audio loader wants a file path (not raw array). Stage to disk.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_np, 16000)
        tmp_path = tmp.name
    try:
        hist: list[dict] = []
        if history:
            try:
                hist = json.loads(history)
            except Exception:
                pass
        sys_prompt = system or DEFAULT_SYSTEM_PROMPT
        result = await _generate(tmp_path, sys_prompt, hist, max_tokens)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return JSONResponse({
        **result,
        "audio_dur_s": round(audio_dur_s, 2),
        "model": MODEL_ID,
    })


def _build_text_messages(system: str, history: list[dict], user_text: str) -> list[dict]:
    """Pure-text variant of `_build_messages` — used by /v1/text-chat
    so the long-audio conversation path can pipe Whisper-transcribed
    text through Gemma 4 (better Arabic / dialect coverage than the
    main server's Qwen fallback)."""
    messages: list[dict] = []
    if system:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system}],
        })
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({
                "role": role,
                "content": [{"type": "text", "text": content}],
            })
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": user_text}],
    })
    return messages


def _generate_text_sync(system: str, history: list[dict], user_text: str, max_tokens: int) -> dict:
    """Same model + tokenizer as audio-chat, but with no audio modality.
    Mirrors `_generate_sync` so the timing breakdown is comparable."""
    messages = _build_text_messages(system, history, user_text)
    t0 = time.time()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    prep_ms = (time.time() - t0) * 1000

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    gen_ms = (time.time() - t0) * 1000
    new_ids = outputs[0, inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "prep_ms": round(prep_ms),
        "gen_ms": round(gen_ms),
        "total_ms": round(prep_ms + gen_ms),
    }


async def _generate_text(system: str, history: list[dict], user_text: str, max_tokens: int) -> dict:
    """Awaitable wrapper for `_generate_text_sync` — holds the same lock
    as audio-chat so concurrent calls serialise on the GPU."""
    assert _lock is not None and processor is not None and model is not None
    async with _lock:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, _generate_text_sync, system, history, user_text, max_tokens
        )


@app.post("/v1/text-chat")
async def text_chat(
    text: str = Form(..., description="The user's message (already a string — not audio)."),
    system: Optional[str] = Form(None, description="Override system prompt"),
    history: Optional[str] = Form(
        None, description='JSON list: [{"role":"user","content":"..."}, ...]'
    ),
    max_tokens: int = Form(160),
):
    """Text-only chat turn. Same Gemma 4 model as /v1/audio-chat but
    without the audio modality — useful when the caller already has the
    user's text (e.g. from Whisper transcription of a long clip).

    Why: keeps response generation consistent across short-audio
    (Gemma audio) and long-audio (Whisper → Gemma text) paths instead
    of falling through to a different LM (Qwen) that has weaker
    Arabic + dialect coverage.
    """
    user_text = (text or "").strip()
    if not user_text:
        raise HTTPException(400, "text is empty")
    hist: list[dict] = []
    if history:
        try:
            hist = json.loads(history)
        except Exception:
            pass
    sys_prompt = system or DEFAULT_SYSTEM_PROMPT
    result = await _generate_text(sys_prompt, hist, user_text, max_tokens)
    return JSONResponse({**result, "model": MODEL_ID})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
