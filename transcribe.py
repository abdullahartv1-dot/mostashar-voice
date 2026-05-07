import json
import sys
import time
from faster_whisper import WhisperModel

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

print('Loading Whisper small model...')
t0 = time.time()
model = WhisperModel('small', device='cpu', compute_type='int8')
print(f'Loaded in {time.time()-t0:.1f}s')

print('Transcribing Arabic audio...')
t0 = time.time()
segments, info = model.transcribe(
    'C:/Users/Admin/Downloads/01.mp3',
    language='ar',
    word_timestamps=True,
    vad_filter=True,
)
print(f'Detected language: {info.language} (prob {info.language_probability:.2f})')

result = []
for s in segments:
    seg = {
        'start': round(s.start, 2),
        'end': round(s.end, 2),
        'text': s.text.strip(),
        'words': [{'w': w.word, 's': round(w.start, 2), 'e': round(w.end, 2)} for w in (s.words or [])],
    }
    result.append(seg)
    print(f"[{seg['start']:6.2f}s - {seg['end']:6.2f}s] {seg['text']}")

print(f'\nTotal transcribe time: {time.time()-t0:.1f}s')

# Save JSON for later use
with open('C:/Users/Admin/Desktop/project/x/transcript.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print('Saved: transcript.json')
