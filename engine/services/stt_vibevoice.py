"""VibeVoice ASR — combined STT + speaker diarization in one model."""
import logging
import time
from typing import Dict, Any, Optional
import torch
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

logger = logging.getLogger(__name__)

_model: Optional[VibeVoiceASRForConditionalGeneration] = None
_processor: Optional[VibeVoiceASRProcessor] = None


def _load() -> None:
    global _model, _processor
    if _model is not None:
        return
    logger.info("Loading VibeVoice ASR-HF (~5GB model + Qwen 7B = ~20GB VRAM)…")
    _processor = VibeVoiceASRProcessor.from_pretrained(
        "microsoft/VibeVoice-ASR-HF",
        language_model_pretrained_name="Qwen/Qwen2.5-7B",
    )
    _model = (
        VibeVoiceASRForConditionalGeneration.from_pretrained(
            "microsoft/VibeVoice-ASR-HF",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
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
        audio_path=audio_path,
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
