"""Experiment 2: Profile WHERE the 1.1s fixed overhead is.
Then identify what to cache.
"""
import os, time
os.environ['HF_HOME'] = '/workspace/hf-cache'

import torch
import soundfile as sf
import librosa
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

MODEL = "aoi-ot/VibeVoice-Large"
REF = "/workspace/refs/01.mp3"

print("loading...")
t0 = time.time()
processor = VibeVoiceProcessor.from_pretrained(MODEL)
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="flash_attention_2",
)
model.eval()
model.set_ddpm_inference_steps(num_steps=15)  # use the sweet spot
print(f"loaded in {time.time()-t0:.1f}s")

ref_audio, _ = librosa.load(REF, sr=24000)
ref_audio = ref_audio[:30*24000]

TEXT = "مرحبا بكم في منصة مستشار."

def profile_one(label):
    print(f"\n=== {label} ===")
    t_start = time.time()

    # Step 1: processor (text + audio encoding)
    t0 = time.time()
    speaker_text = "Speaker 1: " + TEXT
    inputs = processor(
        text=[speaker_text], voice_samples=[[ref_audio]],
        return_tensors="pt", padding=True,
    )
    inputs = {k: (v.to("cuda") if hasattr(v, 'to') else v) for k, v in inputs.items()}
    t_processor = time.time() - t0
    print(f"  processor:    {t_processor*1000:6.0f}ms")

    # Print sizes for understanding
    print(f"    input_ids shape: {inputs['input_ids'].shape}")
    if 'speech_tensors' in inputs:
        print(f"    speech_tensors shape: {inputs['speech_tensors'].shape}")

    # Step 2: model.generate() — full inference
    torch.manual_seed(42)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=None, cfg_scale=1.8,
            tokenizer=processor.tokenizer,
            generation_config={'do_sample': False, 'num_beams': 1},
            verbose=False,
        )
    t_generate = time.time() - t0
    print(f"  generate:     {t_generate*1000:6.0f}ms")

    audio = out.speech_outputs[0].cpu().float().numpy().flatten()
    audio_dur = len(audio) / 24000
    total = time.time() - t_start
    print(f"  total:        {total*1000:6.0f}ms")
    print(f"  audio_dur:    {audio_dur:.2f}s")
    return total, t_processor, t_generate, audio_dur


# Warmup
profile_one("WARMUP")

# Run 3 times to see if there's any per-run caching
profile_one("RUN 1")
profile_one("RUN 2")
profile_one("RUN 3")

# Test with DIFFERENT text but SAME audio — does encoder re-run?
TEXT2 = "هذا اختبار للأداء."
print(f"\n=== Different text, same voice ===")
TEXT = TEXT2
profile_one("RUN 4 (new text, same voice)")
