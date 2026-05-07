"""VibeVoice TTS — wraps the HF Space inference code (proven working).

Uses the vendored `vibe-voice-custom-voices` HF Space which provides a
ComfyUI-style node wrapper around VibeVoice. We mock `comfy` and
`folder_paths` (ComfyUI internals) so the wrapper imports cleanly outside
ComfyUI.

Note: VibeVoice-Large (8GB) is intentionally disabled by disk-quota plan;
re-enable in `_ensure_node` once /workspace headroom increases.
"""
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# Pod cuDNN init is broken (CUDNN_STATUS_NOT_INITIALIZED on conv ops).
# Disable cuDNN — falls back to native CUDA kernels (slower but works).
torch.backends.cudnn.enabled = False

logger = logging.getLogger(__name__)

# Add VibeVoice vendor + comfy mock to path
VENDOR = Path("/workspace/voice-studio-v2/engine/vendor")
SPACE_DIR = VENDOR / "vibe-voice-custom-voices"
COMFY_MOCK = VENDOR / "comfy_mock"
for p in (str(SPACE_DIR), str(COMFY_MOCK)):
    if p not in sys.path:
        sys.path.insert(0, p)


# folder_paths mock (ComfyUI shim). The Space looks up "checkpoints" then
# joins to a sibling "vibevoice" dir for the model cache.
class _MockFolderPaths:
    def get_folder_paths(self, name: str):
        if name == "checkpoints":
            d = SPACE_DIR / "models"
            d.mkdir(parents=True, exist_ok=True)
            return [str(d)]
        return []


if "folder_paths" not in sys.modules:
    sys.modules["folder_paths"] = _MockFolderPaths()


_node = None  # VibeVoiceSingleSpeakerNode (cached)
_loaded_model = None  # str: which model is currently loaded


def _ensure_node(model: str = "VibeVoice-1.5B") -> object:
    global _node, _loaded_model
    if _node is not None and _loaded_model == model:
        return _node

    from nodes.single_speaker_node import VibeVoiceSingleSpeakerNode
    _node = VibeVoiceSingleSpeakerNode()
    model_paths = {
        "VibeVoice-1.5B": "microsoft/VibeVoice-1.5B",
        "VibeVoice-Large": "aoi-ot/VibeVoice-Large",  # disabled by disk quota
    }
    logger.info(f"Loading VibeVoice TTS model: {model}…")
    _node.load_model(
        model_name=model,
        model_path=model_paths[model],
        attention_type="auto",
    )
    _loaded_model = model
    return _node


def clone_vibevoice(
    text: str,
    reference_audio: str,
    model: str = "VibeVoice-1.5B",
    diffusion_steps: int = 20,
    cfg_scale: float = 1.3,
    seed: int = 42,
) -> str:
    """Generate cloned audio. Returns path to output WAV."""
    node = _ensure_node(model)

    import soundfile as sf
    import numpy as np

    audio, sr = sf.read(reference_audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Space expects (batch, channels, samples) 3D format
    waveform_in = np.expand_dims(np.expand_dims(audio.astype(np.float32), 0), 0)
    voice_dict = {"waveform": torch.from_numpy(waveform_in), "sample_rate": sr}

    t0 = time.time()
    (audio_dict,) = node.generate_speech(
        text=text,
        model=model,
        attention_type="auto",
        free_memory_after_generate=False,
        diffusion_steps=diffusion_steps,
        seed=seed,
        cfg_scale=cfg_scale,
        use_sampling=False,
        voice_to_clone=voice_dict,
    )
    elapsed = time.time() - t0

    # Save output to /workspace/voice-studio-v2/jobs/
    out_dir = Path("/workspace/voice-studio-v2/jobs/tts_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vibevoice_{int(time.time())}.wav"

    waveform = audio_dict["waveform"]
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu().float().numpy()
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 2:
        waveform = waveform[0]
    sf.write(out_path, waveform, audio_dict["sample_rate"])
    logger.info(f"VibeVoice TTS done in {elapsed:.1f}s → {out_path}")
    return str(out_path)
