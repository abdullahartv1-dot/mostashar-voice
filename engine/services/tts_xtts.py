"""Coqui XTTS-v2 voice cloning."""
import logging
import os
import time
from pathlib import Path
from typing import Optional
import torch

# Disable cuDNN — Pod has cuDNN init issue
torch.backends.cudnn.enabled = False

logger = logging.getLogger(__name__)
os.environ["COQUI_TOS_AGREED"] = "1"

_tts = None


def _load():
    global _tts
    if _tts is not None:
        return
    from TTS.api import TTS
    logger.info("Loading XTTS-v2…")
    _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)


def clone_xtts(
    text: str,
    reference_audio: str,
    language: str = "ar",
) -> str:
    _load()
    out_dir = Path("/workspace/voice-studio-v2/jobs/tts_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"xtts_{int(time.time())}.wav"

    t0 = time.time()
    _tts.tts_to_file(
        text=text,
        speaker_wav=reference_audio,
        language=language,
        file_path=str(out_path),
    )
    logger.info(f"XTTS-v2 done in {time.time()-t0:.1f}s")
    return str(out_path)
