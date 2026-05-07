import json
import sys
import time
import warnings
import numpy as np
import torch
import soundfile as sf
import subprocess
import imageio_ffmpeg
from pathlib import Path

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR = Path('C:/Users/Admin/Desktop/project/x')
AUDIO_MP3 = 'C:/Users/Admin/Downloads/01.mp3'
AUDIO_WAV = str(OUT_DIR / 'audio_16k.wav')

# 1) Convert MP3 -> mono 16kHz WAV (for speechbrain & VibeVoice compatibility)
print('[1/5] Converting MP3 -> 16kHz mono WAV...')
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run(
    [ffmpeg, '-y', '-i', AUDIO_MP3, '-ac', '1', '-ar', '16000', AUDIO_WAV],
    capture_output=True, check=True
)
audio, sr = sf.read(AUDIO_WAV)
print(f'  Loaded: {len(audio)/sr:.2f}s @ {sr}Hz')

# 2) Load transcript with timestamps
print('[2/5] Loading transcript...')
with open(OUT_DIR / 'transcript.json', encoding='utf-8') as f:
    segments = json.load(f)
print(f'  {len(segments)} segments')

# 3) Load ECAPA-TDNN speaker embedding model
print('[3/5] Loading ECAPA-TDNN speaker embedding model...')
t0 = time.time()
from speechbrain.inference.speaker import EncoderClassifier
classifier = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    savedir=str(OUT_DIR / 'pretrained_models' / 'spkrec-ecapa'),
    run_opts={'device': 'cpu'},
)
print(f'  Loaded in {time.time()-t0:.1f}s')

# 4) Embed each segment
print('[4/5] Computing speaker embeddings per segment...')
embeddings = []
valid_segs = []
for seg in segments:
    s = int(seg['start'] * sr)
    e = int(seg['end'] * sr)
    if e - s < int(0.5 * sr):  # skip < 0.5s (too short for reliable embedding)
        continue
    chunk = torch.tensor(audio[s:e]).unsqueeze(0).float()
    with torch.no_grad():
        emb = classifier.encode_batch(chunk)
    emb = emb.squeeze().cpu().numpy()
    embeddings.append(emb)
    valid_segs.append(seg)

print(f'  Got {len(embeddings)} embeddings')

# 5) Cluster (we know there are likely 2 speakers; let model decide via threshold)
print('[5/5] Clustering speakers...')
from sklearn.cluster import AgglomerativeClustering
X = np.stack(embeddings)
# Try 2 clusters first since the conversation has interview pattern
clustering = AgglomerativeClustering(n_clusters=2, metric='cosine', linkage='average')
labels = clustering.fit_predict(X)

# Assign back to original segments (skipped ones get prev label)
final = []
emb_idx = 0
last_label = 0
for seg in segments:
    if emb_idx < len(valid_segs) and seg is valid_segs[emb_idx]:
        last_label = int(labels[emb_idx])
        emb_idx += 1
    final.append({
        'start': seg['start'],
        'end': seg['end'],
        'text': seg['text'],
        'speaker': f'SPEAKER_{last_label}',
    })

print('\n========= DIARIZATION RESULT =========')
for s in final:
    print(f"[{s['start']:6.2f}s - {s['end']:6.2f}s] {s['speaker']}: {s['text']}")

# Save
with open(OUT_DIR / 'diarized.json', 'w', encoding='utf-8') as f:
    json.dump(final, f, ensure_ascii=False, indent=2)
print('\nSaved: diarized.json')

# Summary
n_per_speaker = {}
for s in final:
    n_per_speaker[s['speaker']] = n_per_speaker.get(s['speaker'], 0) + (s['end'] - s['start'])
print('\nSpeaker totals:')
for sp, dur in n_per_speaker.items():
    print(f'  {sp}: {dur:.2f}s')
