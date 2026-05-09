"""
Quick smoke test for VibeVoice-ASR on our Arabic voice samples.

Verifies:
- Loads VibeVoice-ASR with flash_attention_2
- Transcribes existing Arabic refs (default, hamed_saudi, salma_egypt, ...)
- Reports raw text + segments + per-sample latency

Usage on the pod:
    HF_HOME=/workspace/hf-cache python /workspace/test_asr_arabic.py
"""
import os
import time
import json
import torch
import transformers
transformers.logging.set_verbosity_error()

# Monkey-patch json to handle torch.dtype (transformers config has dtype keys
# that crash logger.info(f"...{config}") during from_pretrained).
_orig_dumps = json.dumps
def _safe_dumps(obj, **kw):
    def _default(o):
        if isinstance(o, torch.dtype):
            return str(o).replace("torch.", "")
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    if "default" not in kw:
        kw["default"] = _default
    return _orig_dumps(obj, **kw)
json.dumps = _safe_dumps

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

MODEL = os.environ.get("ASR_MODEL", "microsoft/VibeVoice-ASR")
VOICES_DIR = os.environ.get("MV_VOICES_DIR", "/workspace/refs/voices")

# Reference text used to seed premade voices — for comparing accuracy
KNOWN_TEXT = (
    "السلام عليكم ورحمة الله وبركاته. مرحباً بكم في منصة مستشار. "
    "نحن نقدم خدمات الاستشارات والتواصل الصوتي بأعلى جودة. "
    "أتمنى أن تكون تجربتكم معنا مميزة ومفيدة. شكراً لكم على ثقتكم بنا."
)

# Sample set — premade voices (known text) + default + a user clone
SAMPLES = [
    "default.wav",
    "hamed_saudi.wav",
    "salma_egypt.wav",
    "rami_lebanon.wav",
    "fatima_uae.wav",
    "noura_kuwait.wav",
]


def main():
    print(f"=== Loading {MODEL} (bf16 + flash_attention_2) ===")
    t0 = time.time()
    processor = VibeVoiceASRProcessor.from_pretrained(
        MODEL, language_model_pretrained_name="Qwen/Qwen2.5-7B"
    )
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model = model.to(torch.bfloat16).to("cuda")
    # encode_speech reads config.torch_dtype to cast inputs; if unset it falls
    # back to fp32 and crashes the conv1d (weights are bf16). Set it explicitly.
    model.config.torch_dtype = torch.bfloat16
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB allocated / "
          f"{torch.cuda.memory_reserved()/1e9:.1f} GB reserved")
    print()

    print(f"=== Reference text (used to seed premade voices) ===")
    print(f"  {KNOWN_TEXT}")
    print()

    audio_paths = [
        os.path.join(VOICES_DIR, s) for s in SAMPLES if os.path.exists(os.path.join(VOICES_DIR, s))
    ]
    if not audio_paths:
        print(f"!! No samples found in {VOICES_DIR}")
        return

    print(f"=== Transcribing {len(audio_paths)} samples ===")
    inputs = processor(
        audio=audio_paths,
        sampling_rate=None,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
    )
    # Cast float tensors (speech_tensors) to bf16 to match model dtype.
    def _to_dev(v):
        if isinstance(v, torch.Tensor):
            v = v.to("cuda")
            if v.dtype == torch.float32:
                v = v.to(torch.bfloat16)
        return v
    inputs = {k: _to_dev(v) for k, v in inputs.items()}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    # Also verify the acoustic_tokenizer first conv weight dtype
    print(f"  acoustic_tokenizer.encoder dtype check: "
          f"{next(model.model.acoustic_tokenizer.encoder.parameters()).dtype}")
    print(f"  Input IDs: {inputs['input_ids'].shape}")
    print(f"  Speech tensors: {inputs['speech_tensors'].shape}")
    print()

    gen_cfg = {
        "max_new_tokens": 512,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": processor.pad_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_cfg)
    wall = time.time() - t0
    print(f"  Generation: {wall:.2f}s ({wall/len(audio_paths):.2f}s/sample avg)")
    print()

    in_len = inputs["input_ids"].shape[1]
    for i, path in enumerate(audio_paths):
        gen_ids = out[i, in_len:]
        eos_pos = (gen_ids == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            gen_ids = gen_ids[: eos_pos[0] + 1]
        text = processor.decode(gen_ids, skip_special_tokens=True)
        try:
            segs = processor.post_process_transcription(text)
        except Exception as e:
            segs = []

        print(f"--- {os.path.basename(path)} ---")
        print(f"  Raw: {text[:400]}")
        if segs:
            print(f"  Segments ({len(segs)}):")
            for s in segs[:5]:
                print(f"    {json.dumps(s, ensure_ascii=False)}")
        print()


if __name__ == "__main__":
    main()
