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
