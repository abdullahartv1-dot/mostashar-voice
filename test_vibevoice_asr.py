"""Run on RunPod: VibeVoice ASR benchmark"""
import json, time, sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
sys.path.insert(0, '/workspace/voice-studio/VibeVoice/demo')

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

INPUT = sys.argv[1] if len(sys.argv) > 1 else '/workspace/voice-studio/03.mp3'
MODEL_PATH = 'microsoft/VibeVoice-ASR'

print(f'Loading VibeVoice-ASR on GPU (downloads ~20GB on first run)...')
t0 = time.time()
processor = VibeVoiceASRProcessor.from_pretrained(
    MODEL_PATH,
    language_model_pretrained_name='Qwen/Qwen2.5-7B'
)
model = VibeVoiceASRForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    attn_implementation='sdpa',
    trust_remote_code=True,
).to('cuda').eval()
load_time = time.time() - t0
print(f'  Loaded in {load_time:.1f}s')

# Audio duration
import subprocess, re, imageio_ffmpeg
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
r = subprocess.run([ffmpeg, '-i', INPUT], capture_output=True, text=True)
m = re.search(r'Duration: (\d+):(\d+):([\d.]+)', r.stderr)
duration = (int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3))) if m else 0

print(f'Audio: {INPUT} | Duration: {duration:.1f}s')
print('Transcribing (this may take 5-15 min for long audio)...')

t0 = time.time()
inputs = processor(
    audio_path=INPUT,
    return_tensors='pt',
    padding=True,
    add_generation_prompt=True,
    context_info='Arabic interview Saudi dialect',
)
inputs = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=16384,
        pad_token_id=processor.pad_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        do_sample=False,
    )

input_length = inputs['input_ids'].shape[1]
generated_ids = output_ids[0, input_length:]
generated_text = processor.decode(generated_ids, skip_special_tokens=True)
elapsed = time.time() - t0

try:
    segments = processor.post_process_transcription(generated_text)
except Exception as e:
    print(f'Parse error: {e}')
    segments = []

print(f'\nDone in {elapsed:.1f}s ({duration/elapsed:.2f}x realtime)')
print(f'Segments: {len(segments)}')

# Save
out = {
    'audio_file': INPUT, 'duration': duration,
    'load_time': round(load_time, 2),
    'transcribe_time': round(elapsed, 2),
    'speedup': round(duration/elapsed, 2),
    'cost_per_hour': 0.27,
    'cost': round((load_time + elapsed) / 3600 * 0.27, 4),
    'raw_text': generated_text,
    'segments': segments,
}
out_path = INPUT.replace('.mp3', '_vibevoice_asr.json').replace('.mp4', '_vibevoice_asr.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f'\nSaved: {out_path}')

# Print first 20 segments
if segments:
    for s in segments[:20]:
        st = s.get('start_time', s.get('start', 0))
        et = s.get('end_time', s.get('end', 0))
        sp = s.get('speaker_id', s.get('speaker', '?'))
        tx = s.get('text', '')[:100]
        print(f"[{st} - {et}] Spk{sp}: {tx}")
