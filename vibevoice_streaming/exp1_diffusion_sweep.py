"""Experiment 1: Sweep diffusion_steps for VibeVoice-Large to find quality/speed sweet spot.

Hypothesis: Realtime uses 5 steps, Large uses 60. Maybe 15-30 still gives acceptable quality
with much faster generation.

Output: For each step count, generate same Arabic text + measure gen_time + save WAV.
"""
import os, time, json
os.environ['HF_HOME'] = '/workspace/hf-cache'

import torch
import soundfile as sf
import librosa
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

MODEL = "aoi-ot/VibeVoice-Large"
REF = "/workspace/refs/01.mp3"
OUT_DIR = "/workspace/refs/exp1"
os.makedirs(OUT_DIR, exist_ok=True)

# Test text — short phrase to focus on TTFA, plus a longer one for quality
SHORT_TEXT = "مرحبا بكم في منصة مستشار."
LONG_TEXT = "مرحبا بكم في منصة مستشار. هذا اختبار لجودة الصوت بعد تقليل خطوات التنقية. نريد أن نختبر هل لا تزال الجودة مقبولة."

STEPS_TO_TEST = [60, 40, 30, 20, 15, 10, 5]

print(f"loading {MODEL}...", flush=True)
t0 = time.time()
processor = VibeVoiceProcessor.from_pretrained(MODEL)
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
)
model.eval()
print(f"  loaded in {time.time()-t0:.1f}s, GPU mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

ref_audio, _ = librosa.load(REF, sr=24000)
ref_audio = ref_audio[:30*24000]
print(f"  ref: {len(ref_audio)/24000:.1f}s")


def gen(text, steps, label):
    """Generate audio with given diffusion_steps."""
    model.set_ddpm_inference_steps(num_steps=steps)
    speaker_text = "Speaker 1: " + text
    inputs = processor(
        text=[speaker_text],
        voice_samples=[[ref_audio]],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: (v.to("cuda") if hasattr(v, 'to') else v) for k, v in inputs.items()}
    torch.manual_seed(42)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=None, cfg_scale=1.8,
            tokenizer=processor.tokenizer,
            generation_config={'do_sample': False, 'num_beams': 1},
            verbose=False,
        )
    elapsed = time.time() - t0
    audio = out.speech_outputs[0].cpu().float().numpy().flatten()
    dur = len(audio) / 24000
    path = f"{OUT_DIR}/{label}_steps{steps}.wav"
    sf.write(path, audio, 24000)
    return {"steps": steps, "label": label, "gen_s": round(elapsed, 2),
            "dur_s": round(dur, 2), "rtf": round(elapsed/dur, 2), "path": path}


# Warmup
print("\n[warmup]")
gen(SHORT_TEXT, 30, "warmup")

results = []
print(f"\n=== short text test (TTFA-relevant) ===")
print(f"text: {SHORT_TEXT}")
for steps in STEPS_TO_TEST:
    r = gen(SHORT_TEXT, steps, "short")
    results.append(r)
    print(f"  steps={r['steps']:3d}: gen={r['gen_s']:5.2f}s, audio={r['dur_s']:.2f}s, RTF={r['rtf']:.2f} ← {r['path']}")

print(f"\n=== long text test (quality-relevant) ===")
for steps in [60, 30, 15, 10]:
    r = gen(LONG_TEXT, steps, "long")
    results.append(r)
    print(f"  steps={r['steps']:3d}: gen={r['gen_s']:5.2f}s, audio={r['dur_s']:.2f}s, RTF={r['rtf']:.2f} ← {r['path']}")

with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n=== ✅ DONE === results.json + WAVs in {OUT_DIR}")
