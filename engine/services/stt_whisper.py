"""Whisper STT (large-v3 and large-v3-turbo)."""
import logging
import time
from typing import Dict, Any
from faster_whisper import WhisperModel
import soundfile as sf

logger = logging.getLogger(__name__)

# Module-level cache — keyed by model size so we don't reload
_models: Dict[str, WhisperModel] = {}


def _get_model(model_size: str) -> WhisperModel:
    """Get a cached Whisper model, loading it if necessary."""
    if model_size not in _models:
        logger.info(f"Loading Whisper {model_size} on GPU…")
        _models[model_size] = WhisperModel(
            model_size, device="cuda", compute_type="float16"
        )
    return _models[model_size]


def transcribe_whisper(audio_path: str, model_size: str = "large-v3") -> Dict[str, Any]:
    """Transcribe with Whisper. Returns dict with duration, segments, timings."""
    model = _get_model(model_size)
    duration = sf.info(audio_path).duration

    t0 = time.time()
    seg_iter, info = model.transcribe(
        audio_path,
        language="ar",
        beam_size=5,
        vad_filter=True,
    )
    segments = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in seg_iter
    ]
    elapsed = time.time() - t0

    return {
        "duration": duration,
        "segments": segments,
        "lang_prob": info.language_probability,
        "elapsed_sec": round(elapsed, 2),
        "speedup": round(duration / elapsed, 2) if elapsed > 0 else 0,
    }
