"""Run on RunPod: Test Whisper Large-v3 with GPU"""
import json, time, sys, os
sys.stdout.reconfigure(encoding='utf-8')
from faster_whisper import WhisperModel

INPUT = sys.argv[1] if len(sys.argv) > 1 else '/workspace/voice-studio/01.mp3'
MODEL_SIZE = sys.argv[2] if len(sys.argv) > 2 else 'large-v3'

print(f'Loading Whisper {MODEL_SIZE} on GPU (this may take ~30s first time)...')
t0 = time.time()
model = WhisperModel(MODEL_SIZE, device='cuda', compute_type='float16')
load_time = time.time() - t0
print(f'  Loaded in {load_time:.1f}s')

# Get audio duration
import soundfile as sf
import subprocess
import imageio_ffmpeg
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
import re
result = subprocess.run([ffmpeg, '-i', INPUT], capture_output=True, text=True)
m = re.search(r'Duration: (\d+):(\d+):([\d.]+)', result.stderr)
if m:
    duration = int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3))
else:
    duration = 0

print(f'Audio: {INPUT} | Duration: {duration:.1f}s')
print(f'Transcribing...')

t0 = time.time()
segments_iter, info = model.transcribe(
    INPUT, language='ar', vad_filter=True, beam_size=5,
)

segments = []
for s in segments_iter:
    segments.append({'start': round(s.start,2), 'end': round(s.end,2), 'text': s.text.strip()})

elapsed = time.time() - t0
print(f'\nTranscription done in {elapsed:.1f}s ({duration/elapsed:.1f}x realtime)')
print(f'Segments: {len(segments)}')
print(f'\n=== Transcript ===')
for s in segments:
    print(f"[{s['start']:6.2f}s - {s['end']:6.2f}s] {s['text']}")

# Save results
out = {
    'input': INPUT,
    'model': MODEL_SIZE,
    'duration': duration,
    'load_time': round(load_time, 2),
    'transcribe_time': round(elapsed, 2),
    'speedup': round(duration/elapsed, 2),
    'segments': segments,
}
out_path = INPUT.replace('.mp3', '_whisper_large.json').replace('.mp4', '_whisper_large.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f'\nSaved: {out_path}')
