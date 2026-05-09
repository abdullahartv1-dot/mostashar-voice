"""Smoke test: load VibeVoice-1.5B and time a single generation.
Then time SENTENCE-BY-SENTENCE generation to validate chunking strategy."""
import os, sys, time, re
os.environ['HF_HOME'] = '/workspace/vv/models'

import soundfile as sf
import torch
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

MODEL_PATH = "microsoft/VibeVoice-1.5B"
TEXT = "مرحبا بكم في منصة مُسْتَشَار. هذا اختبار للبث الصوتي اللحظي. نحن نختبر تقنية التقطيع على مستوى الجملة."
REF_AUDIO = None  # use random voice for first test; will plug in cloned voice next

print(f"loading processor + model from {MODEL_PATH}...", flush=True)
t0 = time.time()
processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

def split_sentences(text):
    """Split Arabic text on sentence boundaries (. ! ? ؟)"""
    parts = re.split(r'(?<=[\.\!\؟\?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def gen_one(text, idx):
    """Generate one segment, return (path, gen_time, audio_dur)."""
    speaker = "Speaker 1: " + text
    inputs = processor(
        text=[speaker],
        voice_samples=[None],
        return_tensors="pt",
        padding=True,
    ).to("cuda")
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
    path = f"/tmp/seg_{idx}.wav"
    sf.write(path, audio, sr)
    return path, elapsed, len(audio) / sr

print("\n=== TEST 1: full text in one shot ===")
t0 = time.time()
path, gen_time, dur = gen_one(TEXT, 999)
total = time.time() - t0
print(f"full text: gen={gen_time:.2f}s, dur={dur:.2f}s, total wall={total:.2f}s")

print("\n=== TEST 2: sentence-by-sentence (simulating streaming) ===")
sentences = split_sentences(TEXT)
print(f"split into {len(sentences)} sentences:")
for i, s in enumerate(sentences):
    print(f"  [{i}] {s}")

print("\ngenerating one by one (TTFA = time until first chunk ready):")
overall_start = time.time()
for i, s in enumerate(sentences):
    path, gen_time, dur = gen_one(s, i)
    elapsed_since_request = time.time() - overall_start
    print(f"  chunk {i}: gen={gen_time:.2f}s, dur={dur:.2f}s, ready_at={elapsed_since_request:.2f}s — {path}")

print("\n=== ✅ DONE ===")
print(f"baseline (one-shot TTFA): {total:.2f}s")
print(f"streaming TTFA (first chunk ready): see chunk 0 ready_at above")
