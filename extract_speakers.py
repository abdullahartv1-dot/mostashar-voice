import json
import sys
import subprocess
import imageio_ffmpeg
from pathlib import Path
import soundfile as sf
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR = Path('C:/Users/Admin/Desktop/project/x')
AUDIO_WAV = str(OUT_DIR / 'audio_16k.wav')

# Load diarized transcript
with open(OUT_DIR / 'diarized.json', encoding='utf-8') as f:
    segments = json.load(f)

# Load audio
audio, sr = sf.read(AUDIO_WAV)
print(f'Loaded audio: {len(audio)/sr:.2f}s @ {sr}Hz')

# Group by speaker
speakers = {}
for seg in segments:
    sp = seg['speaker']
    if sp not in speakers:
        speakers[sp] = []
    speakers[sp].append(seg)

# For each speaker, build a contiguous sample (concat all their segments + small silence between)
silence = np.zeros(int(0.2 * sr))  # 200ms silence between concatenated segments
for sp, segs in speakers.items():
    chunks = []
    total_dur = 0
    for s in segs:
        s_idx = int(s['start'] * sr)
        e_idx = int(s['end'] * sr)
        chunks.append(audio[s_idx:e_idx])
        chunks.append(silence)
        total_dur += s['end'] - s['start']

    sample = np.concatenate(chunks)
    out_path = OUT_DIR / f'{sp}_sample.wav'
    sf.write(out_path, sample, sr)
    print(f'  {sp}: saved {len(sample)/sr:.2f}s sample (raw speech: {total_dur:.2f}s) -> {out_path.name}')

    # Also save the longest single segment as a "clean" sample for cloning
    longest = max(segs, key=lambda s: s['end'] - s['start'])
    s_idx = int(longest['start'] * sr)
    e_idx = int(longest['end'] * sr)
    clean = audio[s_idx:e_idx]
    out_path_clean = OUT_DIR / f'{sp}_clean.wav'
    sf.write(out_path_clean, clean, sr)
    print(f'  {sp}: longest segment {longest["end"]-longest["start"]:.2f}s -> {out_path_clean.name}: "{longest["text"]}"')

print('\nDone.')
