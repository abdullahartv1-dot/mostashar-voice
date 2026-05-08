"""F5-TTS voice cloning."""
import logging
import time
from pathlib import Path
from typing import Optional
import torch

# Disable cuDNN at import time — Pod has cuDNN init issue
torch.backends.cudnn.enabled = False

logger = logging.getLogger(__name__)

_model = None


def _load():
    global _model
    if _model is not None:
        return
    logger.info("Loading F5-TTS model…")
    from f5_tts.api import F5TTS
    _model = F5TTS()
    logger.info("F5-TTS loaded.")


def _autotranscribe_ref(reference_audio: str) -> str:
    """Transcribe reference using our existing faster-whisper turbo to avoid
    F5-TTS's internal HF-pipeline Whisper download (which fails due to disk/quota
    or model file mismatch in transformers pipeline). Returns short-text string.
    """
    try:
        from .stt_whisper import transcribe_whisper
        out = transcribe_whisper(reference_audio, model_size="large-v3-turbo")
        text = " ".join(s["text"].strip() for s in out.get("segments", []))
        text = text.strip() or "ref"
        # Cap reference text length — F5 only needs first chunk
        return text[:300]
    except Exception as e:
        logger.warning(f"Auto-transcribe ref failed: {e}; using placeholder")
        return "reference audio."


def clone_f5(
    text: str,
    reference_audio: str,
    reference_text: str = "",
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    seed: int = 42,
) -> str:
    """Generate cloned audio. Returns path to output WAV."""
    _load()

    out_dir = Path("/workspace/voice-studio-v2/jobs/tts_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"f5_{int(time.time())}.wav"

    # F5-TTS will internally try to load openai/whisper-large-v3-turbo via
    # transformers.pipeline if ref_text is empty. That model is not in HF cache
    # in the right format, so we always pre-transcribe via faster-whisper.
    if not reference_text:
        reference_text = _autotranscribe_ref(reference_audio)
        logger.info(f"F5-TTS auto-transcribed ref: {reference_text[:80]}…")

    t0 = time.time()
    _model.infer(
        ref_file=reference_audio,
        ref_text=reference_text,
        gen_text=text,
        file_wave=str(out_path),
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        seed=seed,
    )
    logger.info(f"F5-TTS done in {time.time()-t0:.1f}s")
    return str(out_path)
