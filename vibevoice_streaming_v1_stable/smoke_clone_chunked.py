"""Test 1: Clone Arabic voice with VibeVoice-1.5B (community fork) — non-streaming baseline.
Then test SENTENCE CHUNKING to simulate streaming: split text on periods, generate each segment,
measure TTFA (time to first chunk ready)."""
import os, sys, time, re
os.environ['HF_HOME'] = '/workspace/hf-cache'

import torch
import soundfile as sf
import librosa
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

MODEL = "microsoft/VibeVoice-1.5B"
REF = "/workspace/refs/01.mp3"
OUT_DIR = "/workspace/refs/out"
os.makedirs(OUT_DIR, exist_ok=True)

TEXT = "مرحبا بكم في منصة مستشار. هذا اختبار لاستنساخ الصوت بالعربية. نريد أن نختبر جودة النطق."

print(f"[1/4] loading processor + model from {MODEL}...", flush=True)
t0 = time.time()
processor = VibeVoiceProcessor.from_pretrained(MODEL)
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
)
model.eval()
model.set_ddpm_inference_steps(num_steps=10)  # fewer steps = faster
print(f"  loaded in {time.time()-t0:.1f}s, GPU mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

print(f"\n[2/4] loading reference audio {REF}...")
ref_audio, _ = librosa.load(REF, sr=24000)
print(f"  ref duration: {len(ref_audio)/24000:.1f}s")
# trim to 30s for cloning
ref_audio = ref_audio[:30*24000]


def gen_segment(text, idx, voice_samples):
    """Generate one segment using cached voice. Returns (path, gen_time, audio_dur)."""
    speaker_text = "Speaker 1: " + text
    inputs = processor(
        text=[speaker_text],
        voice_samples=[voice_samples],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: (v.to("cuda") if hasattr(v, 'to') else v) for k, v in inputs.items()}
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=1.3,
            tokenizer=processor.tokenizer,
            generation_config={'do_sample': False, 'num_beams': 1},
            verbose=False,
        )
    elapsed = time.time() - t0
    audio = out.speech_outputs[0].cpu().float().numpy().flatten()
    sr = 24000
    path = f"{OUT_DIR}/seg_{idx}.wav"
    sf.write(path, audio, sr)
    return path, elapsed, len(audio) / sr


print(f"\n[3/4] === full text in one shot (baseline) ===")
t0 = time.time()
path, gen_time, dur = gen_segment(TEXT, 99, [ref_audio])
total_wall = time.time() - t0
print(f"  one-shot: gen_time={gen_time:.2f}s, audio_dur={dur:.2f}s, total_wall={total_wall:.2f}s, RTF={gen_time/dur:.2f}")

print(f"\n[4/4] === sentence chunking simulation ===")
sentences = [s.strip() for s in re.split(r'(?<=[\.\!\؟\?])\s+', TEXT.strip()) if s.strip()]
print(f"  split into {len(sentences)} sentences:")
for i, s in enumerate(sentences):
    print(f"    [{i}] {s}")

print("\n  generating each sentence (TTFA = time until chunk 0 is ready):")
t_start = time.time()
for i, s in enumerate(sentences):
    path, gen_time, dur = gen_segment(s, i, [ref_audio])
    ready_at = time.time() - t_start
    marker = " <-- TTFA" if i == 0 else ""
    print(f"    chunk {i}: gen={gen_time:.2f}s, audio={dur:.2f}s, ready_at={ready_at:.2f}s{marker}")

print(f"\n=== ✅ DONE === outputs in {OUT_DIR}/")
