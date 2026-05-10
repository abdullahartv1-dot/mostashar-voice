"""VibeVoice ASR — combined STT + speaker diarization in one model.

KNOWN LIMITATION: This model needs 8 safetensor shards (~5GB) plus Qwen-7B (~15GB)
that get downloaded from HuggingFace at first call. The Pod's HF CDN connections
get throttled/dropped mid-download, leaving partial files that fill the disk
quota without ever yielding a usable model. Until that's fixed (e.g., by
pre-downloading via a separate one-shot install script + verifying all 8 shards
exist), this handler refuses to start the download.
"""
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

logger = logging.getLogger(__name__)

_model: Optional[VibeVoiceASRForConditionalGeneration] = None
_processor: Optional[VibeVoiceASRProcessor] = None


_CACHE_ROOT = "/workspace/voice-studio-v2/engine/vendor/vibe-voice-custom-voices/vibevoice"


def _check_complete_cache() -> Optional[str]:
    """Return error string if VibeVoice ASR shards aren't all locally cached."""
    snap_root = Path(_CACHE_ROOT) / "models--microsoft--VibeVoice-ASR-HF" / "snapshots"
    if not snap_root.exists():
        return f"VibeVoice-ASR cache not found at {snap_root}"
    snaps = list(snap_root.iterdir())
    if not snaps:
        return f"No snapshots in {snap_root}"
    snap = snaps[0]
    needed = [f"model-0000{i}-of-00008.safetensors" for i in range(1, 9)]
    missing = [n for n in needed if not (snap / n).exists()]
    if missing:
        return (
            f"VibeVoice-ASR is incomplete in cache (missing {len(missing)}/8 shards: "
            f"{missing[0]}…). Run snapshot_download to fetch all shards."
        )
    return None


def _load() -> None:
    global _model, _processor
    if _model is not None:
        return
    err = _check_complete_cache()
    if err:
        raise RuntimeError(err)
    logger.info("Loading VibeVoice ASR-HF (~5GB model + Qwen 7B = ~20GB VRAM)…")
    os.environ["HF_HUB_CACHE"] = _CACHE_ROOT
    _processor = VibeVoiceASRProcessor.from_pretrained(
        "microsoft/VibeVoice-ASR-HF",
        language_model_pretrained_name="Qwen/Qwen2.5-7B",
        cache_dir=_CACHE_ROOT,
    )
    _model = (
        VibeVoiceASRForConditionalGeneration.from_pretrained(
            "microsoft/VibeVoice-ASR-HF",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
            cache_dir=_CACHE_ROOT,
        )
        .to("cuda")
        .eval()
    )


def transcribe_vibevoice(
    audio_path: str,
    context_info: str = "Arabic conversation",
    max_new_tokens: int = 16384,
) -> Dict[str, Any]:
    _load()
    import soundfile as sf

    duration = sf.info(audio_path).duration

    t0 = time.time()
    inputs = _processor(
        audio=audio_path,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        context_info=context_info,
    )
    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=_processor.pad_id,
            eos_token_id=_processor.tokenizer.eos_token_id,
            do_sample=False,
        )
    input_length = inputs["input_ids"].shape[1]
    generated_text = _processor.decode(output_ids[0, input_length:], skip_special_tokens=True)
    elapsed = time.time() - t0

    try:
        segments = _processor.post_process_transcription(generated_text)
    except Exception as e:
        logger.warning(f"Post-process failed: {e}")
        segments = [{"text": generated_text, "speaker": "unknown", "start": 0, "end": duration}]

    return {
        "duration": duration,
        "segments": segments,
        "raw_text": generated_text,
        "elapsed_sec": round(elapsed, 2),
        "speedup": round(duration / elapsed, 2) if elapsed > 0 else 0,
    }
