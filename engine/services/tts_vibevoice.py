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

# cuDNN was broken on the previous A5000 Pod (CUDNN_STATUS_NOT_INITIALIZED).
# H100 has working cuDNN — keep it enabled for max throughput. If a future Pod
# has the same init bug, set VIBEVOICE_DISABLE_CUDNN=1 in the env.
if os.environ.get("VIBEVOICE_DISABLE_CUDNN") == "1":
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

    # On older A5000 Pod, VibeVoice-Large was blocked because 8GB / 10 shards triggered
    # HF CDN throttling and partial downloads filled the 20GB workspace quota. The H100
    # Pod has 50GB workspace and downloads complete in ~50s — Large is fully supported here.

    from nodes.single_speaker_node import VibeVoiceSingleSpeakerNode
    _node = VibeVoiceSingleSpeakerNode()
    model_paths = {
        "VibeVoice-1.5B": "microsoft/VibeVoice-1.5B",
        "VibeVoice-Large": "aoi-ot/VibeVoice-Large",  # disabled by disk quota
    }
    # On A5000 without flash-attn, "auto" picks sdpa which has different
    # bf16 numerics from the HF Space's flash_attention_2 path → garbled output.
    # "eager" is the reference impl and matches the Space's quality on Arabic.
    attn = os.environ.get("VIBEVOICE_ATTN", "eager")
    logger.info(f"Loading VibeVoice TTS model: {model} (attention={attn})…")
    _node.load_model(
        model_name=model,
        model_path=model_paths[model],
        attention_type=attn,
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

    # Match HF Space exactly: librosa @ sr=24000 mono, then (1,1,N) tensor.
    # Going through soundfile + base_vibevoice's resampler path produces
    # different output (gibberish on Arabic) even when SR matches.
    import librosa
    waveform, _ = librosa.load(reference_audio, sr=24000, mono=True)
    waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0)
    voice_dict = {"waveform": waveform_tensor, "sample_rate": 24000}

    attn = os.environ.get("VIBEVOICE_ATTN", "eager")
    t0 = time.time()
    (audio_dict,) = node.generate_speech(
        text=text,
        model=model,
        attention_type=attn,
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
    import soundfile as sf
    sf.write(out_path, waveform, audio_dict["sample_rate"])
    logger.info(f"VibeVoice TTS done in {elapsed:.1f}s → {out_path}")
    return str(out_path)
