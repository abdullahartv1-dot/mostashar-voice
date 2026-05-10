"""Apples-to-apples comparison of TTS engines on Arabic — commercial-safe only.

Tests: VibeVoice-Large (MIT), Chatterbox MTL (MIT), Higgs Audio V2 (Apache 2.0).
XTTS-v2 dropped per user feedback. Fish/IndexTTS dropped (non-commercial license).

Same text + same reference audio for all.
Outputs: WAV files + table of (model, gen_time, rtf, whisper_accuracy).

Run on H100 Pod:
    python -m engine.compare_tts_engines
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path

sys.path.insert(0, "/workspace/voice-studio-v2")

# ---- shared inputs ----
REF_AUDIO = "/workspace/voice-studio-v2/jobs/950cb7a5/SPEAKER_0_sample.wav"
TEXT = (
    "مرحبا بكم في تجربة استنساخ الصوت العربي من منصة مُسْتَشَار. "
    "هذا النص الذي تسمعونه الآن مُوَلَّد بالكامل بواسطة الذكاء الاصطناعي. "
    "لكنه يستخدم بصمة صوت المتحدث الأصلي."
)
OUT_DIR = Path("/workspace/voice-studio-v2/jobs/tts_compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)

results = []


def time_it(name, fn):
    """Run fn(); record gen_time, write metadata, return out_path."""
    print(f"\n{'='*60}\n[{name}] generating…\n{'='*60}", flush=True)
    t0 = time.time()
    out_path, extra = fn()
    elapsed = time.time() - t0
    import soundfile as sf
    info = sf.info(str(out_path))
    rtf = elapsed / info.duration if info.duration > 0 else 0
    rec = {
        "engine": name,
        "elapsed_sec": round(elapsed, 2),
        "out_duration_sec": round(info.duration, 2),
        "rtf": round(rtf, 3),
        "out_path": str(out_path),
        **extra,
    }
    results.append(rec)
    print(f"  → {out_path.name}: {info.duration:.2f}s in {elapsed:.2f}s (RTF {rtf:.2f}x)")
    return out_path


# ---- engine 1: VibeVoice-Large ----
def vibevoice_large():
    from engine.services.tts_vibevoice import clone_vibevoice
    out = clone_vibevoice(
        text=TEXT, reference_audio=REF_AUDIO,
        model="VibeVoice-Large", diffusion_steps=60, cfg_scale=1.8, seed=42,
    )
    dst = OUT_DIR / "vibevoice_large.wav"
    import shutil; shutil.copy(out, dst)
    return dst, {"params": "diff=60 cfg=1.8"}


# ---- engine 2: Chatterbox Multilingual (Resemble AI, MIT) ----
def chatterbox():
    import torchaudio as ta
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    if not hasattr(chatterbox, "_model"):
        chatterbox._model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    model = chatterbox._model
    wav = model.generate(TEXT, language_id="ar", audio_prompt_path=REF_AUDIO)
    dst = OUT_DIR / "chatterbox.wav"
    ta.save(str(dst), wav, model.sr)
    return dst, {"params": "language_id=ar"}


# ---- engine 3: Higgs Audio V2 (Boson AI, Apache 2.0) ----
def higgs_audio():
    """Zero-shot voice cloning via reference. 24GB+ VRAM on H100 — no problem."""
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
    if not hasattr(higgs_audio, "_engine"):
        higgs_audio._engine = HiggsAudioServeEngine(
            model_name_or_path="bosonai/higgs-audio-v2-generation-3B-base",
            audio_tokenizer_name_or_path="bosonai/higgs-audio-v2-tokenizer",
        )
    engine = higgs_audio._engine
    messages = [
        Message(role="system", content="Generate audio following instruction."),
        Message(role="user", content=[AudioContent(audio_url=REF_AUDIO)]),
        Message(role="assistant", content="Acknowledged. I'll clone this voice."),
        Message(role="user", content=TEXT),
    ]
    output = engine.generate(chat_ml_sample=ChatMLSample(messages=messages))
    import soundfile as sf
    dst = OUT_DIR / "higgs_audio.wav"
    sf.write(str(dst), output.audio, output.sampling_rate)
    return dst, {"params": "v2-generation-3B"}


# ---- run all ----
# Higgs-Audio-V2 disabled: needs transformers <5.x but Chatterbox installs 5.2.0.
# Would need a separate venv. Skipping for now.
engines = [
    ("VibeVoice-Large", vibevoice_large),
    ("Chatterbox",      chatterbox),
]
for name, fn in engines:
    try:
        time_it(name, fn)
    except Exception as e:
        import traceback; traceback.print_exc()
        results.append({"engine": name, "error": str(e)[:200]})


# ---- whisper accuracy on each output ----
print(f"\n{'='*60}\n[whisper accuracy check]\n{'='*60}", flush=True)
from engine.services.stt_whisper import transcribe_whisper
for r in results:
    if "out_path" not in r:
        continue
    try:
        tx = transcribe_whisper(r["out_path"], "large-v3")
        text_out = " ".join(s["text"].strip() for s in tx["segments"])
        r["transcribed"] = text_out
        # crude similarity: % of input words that appear in output
        in_words = set(TEXT.replace(",", "").replace(".", "").split())
        out_words = set(text_out.replace(",", "").replace(".", "").split())
        match = len(in_words & out_words) / max(len(in_words), 1)
        r["word_match_pct"] = round(match * 100, 1)
    except Exception as e:
        r["whisper_error"] = str(e)[:200]


# ---- final report ----
print(f"\n{'='*60}\n[FINAL COMPARISON]\n{'='*60}", flush=True)
for r in results:
    print(json.dumps(r, ensure_ascii=False, indent=2))
    print()

(OUT_DIR / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(f"\nResults saved: {OUT_DIR}/results.json")
print(f"WAV files in:  {OUT_DIR}/")
