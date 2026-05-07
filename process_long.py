"""
Process a long audio file (1+ hour) with progress logging.
Run: python process_long.py <input.mp3>
Outputs to: long_results/
"""
import json
import os
import sys
import time
import warnings
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
PROGRESS_FILE = 'C:/Users/Admin/Desktop/project/x/long_progress.json'


def log_progress(stage, percent, message):
    data = {'stage': stage, 'percent': percent, 'message': message, 't': time.time()}
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print(f'[{stage}] {percent:>5.1f}% - {message}', flush=True)


def main(input_path):
    out_dir = Path('C:/Users/Admin/Desktop/project/x/long_results')
    out_dir.mkdir(exist_ok=True)

    log_progress('init', 0, f'Starting on {Path(input_path).name}')

    # 1. Convert to 16kHz mono WAV
    wav_path = out_dir / 'audio_16k.wav'
    log_progress('convert', 5, 'Converting MP3 -> 16kHz mono WAV...')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    t0 = time.time()
    subprocess.run(
        [ffmpeg, '-y', '-i', input_path, '-ac', '1', '-ar', '16000', str(wav_path)],
        capture_output=True, check=True,
    )
    info = sf.info(str(wav_path))
    duration = info.duration
    log_progress('convert', 8, f'Done ({time.time()-t0:.1f}s). Audio: {duration:.0f}s ({int(duration//60)}:{int(duration%60):02d})')

    # 2. Whisper - use BASE model (2x faster than small)
    log_progress('whisper_load', 10, 'Loading Whisper base model...')
    model = WhisperModel('base', device='cpu', compute_type='int8')

    log_progress('whisper', 12, 'Transcribing (will take ~15-25 min)...')
    t0 = time.time()
    segments_iter, info = model.transcribe(
        str(wav_path), language='ar', word_timestamps=False,
        vad_filter=True, vad_parameters={'min_silence_duration_ms': 500},
        beam_size=1,  # Faster
    )

    segments = []
    last_log = time.time()
    for s in segments_iter:
        segments.append({
            'start': round(s.start, 2),
            'end': round(s.end, 2),
            'text': s.text.strip(),
        })
        # Log progress every 30s of wall-clock time
        if time.time() - last_log > 30:
            pct = 12 + (s.end / duration) * 50  # whisper is 50% of total work
            log_progress('whisper', pct,
                         f'{len(segments)} segments | audio pos {s.end:.0f}s / {duration:.0f}s | wall {time.time()-t0:.0f}s')
            last_log = time.time()

    elapsed = time.time() - t0
    log_progress('whisper', 62, f'Done in {elapsed/60:.1f} min. {len(segments)} segments.')

    # Save transcript
    with open(out_dir / 'transcript.json', 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    # 3. ECAPA-TDNN diarization
    log_progress('diar_load', 65, 'Loading ECAPA-TDNN model...')
    spk = EncoderClassifier.from_hparams(
        source='speechbrain/spkrec-ecapa-voxceleb',
        savedir='C:/Users/Admin/Desktop/project/x/pretrained_models/spkrec-ecapa',
        run_opts={'device': 'cpu'},
    )

    log_progress('diar', 67, f'Computing embeddings for {len(segments)} segments...')
    audio, sr = sf.read(str(wav_path))
    embeddings = []
    valid_idx = []
    t0 = time.time()
    for i, seg in enumerate(segments):
        s = int(seg['start'] * sr)
        e = int(seg['end'] * sr)
        if e - s < int(0.5 * sr):
            continue
        chunk = torch.tensor(audio[s:e]).unsqueeze(0).float()
        with torch.no_grad():
            emb = spk.encode_batch(chunk).squeeze().cpu().numpy()
        embeddings.append(emb)
        valid_idx.append(i)
        if i % 50 == 0:
            pct = 67 + (i / len(segments)) * 23
            log_progress('diar', pct, f'{i}/{len(segments)} embeddings, {time.time()-t0:.0f}s')

    log_progress('diar', 90, f'Clustering {len(embeddings)} embeddings into 2 speakers...')
    if len(embeddings) >= 2:
        labels = AgglomerativeClustering(
            n_clusters=2, metric='cosine', linkage='average'
        ).fit_predict(np.stack(embeddings))
    else:
        labels = [0] * len(embeddings)

    # Assign back
    last_label = 0
    li = 0
    for i, seg in enumerate(segments):
        if li < len(valid_idx) and valid_idx[li] == i:
            last_label = int(labels[li])
            li += 1
        seg['speaker'] = f'SPEAKER_{last_label}'

    # 4. Extract speaker samples (longest contiguous block per speaker, max 30s)
    log_progress('extract', 93, 'Extracting per-speaker samples...')
    by_spk = {}
    for s in segments:
        by_spk.setdefault(s['speaker'], []).append(s)

    samples = {}
    for sp, segs in by_spk.items():
        # Find best 30s window (longest contiguous, near beginning is usually clearest)
        chunks = []
        total = 0
        for s in segs[:50]:  # Use first 50 segments per speaker for sample
            if total >= 30:
                break
            si = int(s['start'] * sr)
            ei = int(s['end'] * sr)
            chunks.append(audio[si:ei])
            chunks.append(np.zeros(int(0.15 * sr)))  # short silence
            total += s['end'] - s['start']
        sample = np.concatenate(chunks)
        path = out_dir / f'{sp}_sample.wav'
        sf.write(path, sample, sr)
        # Total stats
        all_dur = sum(s['end'] - s['start'] for s in segs)
        samples[sp] = {
            'path': str(path),
            'sample_duration': round(len(sample) / sr, 2),
            'total_speech_duration': round(all_dur, 2),
            'segment_count': len(segs),
        }
        log_progress('extract', 95,
                     f'{sp}: {len(segs)} segments, {all_dur:.0f}s total speech, sample {len(sample)/sr:.1f}s')

    # 5. Save final results
    final = {
        'duration': duration,
        'segments': segments,
        'samples': samples,
        'lang_prob': 1.0,
    }
    with open(out_dir / 'diarized.json', 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    log_progress('done', 100, f'COMPLETE! Total {len(segments)} segments, {len(samples)} speakers.')
    print('Files saved to:', out_dir)


if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/Admin/Downloads/03.mp3'
    main(inp)
