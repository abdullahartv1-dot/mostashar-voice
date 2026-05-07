"""NVIDIA NeMo Canary-1B STT service."""
import logging
import time
from typing import Dict, Any, Optional
import soundfile as sf

logger = logging.getLogger(__name__)

_model = None  # nemo.collections.asr.models.EncDecMultiTaskModel


def _load():
    global _model
    if _model is not None:
        return
    logger.info("Loading NVIDIA Canary-1B…")
    from nemo.collections.asr.models import EncDecMultiTaskModel
    _model = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b")
    _model.eval()


def transcribe_canary(audio_path: str, source_lang: str = "en") -> Dict[str, Any]:
    """Transcribe with NVIDIA Canary-1B.

    Note: Canary-1B officially supports en/de/es/fr only. For Arabic input,
    the model will run but output is not meaningful — used here only to
    verify the service contract (shape).
    """
    _load()
    duration = sf.info(audio_path).duration

    t0 = time.time()
    # Canary expects a list of paths; returns list of strings
    transcripts = _model.transcribe(
        audio=[audio_path],
        batch_size=1,
        source_lang=source_lang,
        target_lang=source_lang,
        task="asr",
        pnc="yes",
    )
    elapsed = time.time() - t0

    text = transcripts[0] if transcripts else ""
    return {
        "duration": duration,
        "segments": [{"start": 0, "end": duration, "text": text}],
        "elapsed_sec": round(elapsed, 2),
        "speedup": round(duration / elapsed, 2) if elapsed > 0 else 0,
    }
