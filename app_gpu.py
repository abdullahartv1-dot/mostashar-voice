"""
Voice Studio GPU Edition - runs entirely on RunPod GPU
- Whisper Large-v3 (GPU) for transcription
- ECAPA-TDNN + clustering for diarization (auto-detects speakers)
- HF Spaces VibeVoice (still uses quota - to be replaced later with local)
- Real-time cost tracking ($0.27/hr RTX A5000)
"""
import json, os, shutil, sys, time, uuid, warnings
from pathlib import Path
import imageio_ffmpeg
import numpy as np
import soundfile as sf
import subprocess
import torch
from faster_whisper import WhisperModel
from flask import Flask, jsonify, request, render_template, send_from_directory
from gradio_client import Client, handle_file
from sklearn.cluster import AgglomerativeClustering, estimate_bandwidth
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).parent
STATIC = ROOT / 'static'
AUDIO_DIR = STATIC / 'audio'
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
JOBS = {}

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
GPU_HOURLY_COST = 0.27  # RTX A5000 on RunPod
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'[boot] Device: {DEVICE}')
print(f'[boot] Loading Whisper Large-v3...')
WHISPER = WhisperModel('large-v3', device=DEVICE, compute_type='float16' if DEVICE == 'cuda' else 'int8')
print(f'[boot] Loading ECAPA-TDNN...')
SPK = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    savedir=str(ROOT / 'pretrained_models' / 'spkrec-ecapa'),
    run_opts={'device': DEVICE},
)
print('[boot] Models ready.')

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024


def to_wav_16k_mono(src, dst):
    subprocess.run([FFMPEG, '-y', '-i', src, '-ac', '1', '-ar', '16000', dst],
                   capture_output=True, check=True)


def transcribe(wav_path):
    segs, info = WHISPER.transcribe(wav_path, language='ar', vad_filter=True, beam_size=5)
    out = [{'start': round(s.start, 2), 'end': round(s.end, 2), 'text': s.text.strip()} for s in segs]
    return out, info.language_probability


def auto_diarize(wav_path, segments, max_speakers=8):
    """Auto-detect number of speakers using silhouette score"""
    audio, sr = sf.read(wav_path)
    embeddings, valid_idx = [], []
    for i, seg in enumerate(segments):
        s, e = int(seg['start'] * sr), int(seg['end'] * sr)
        if e - s < int(0.5 * sr): continue
        chunk = torch.tensor(audio[s:e]).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            emb = SPK.encode_batch(chunk).squeeze().cpu().numpy()
        embeddings.append(emb)
        valid_idx.append(i)

    if len(embeddings) < 2:
        for s in segments: s['speaker'] = 'SPEAKER_0'
        return segments, 1

    X = np.stack(embeddings)
    # Try 2-max_speakers and pick best silhouette
    best_score, best_k, best_labels = -1, 2, None
    for k in range(2, min(max_speakers + 1, len(embeddings))):
        try:
            labels = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average').fit_predict(X)
            score = silhouette_score(X, labels, metric='cosine')
            if score > best_score:
                best_score, best_k, best_labels = score, k, labels
        except Exception:
            continue

    if best_labels is None:
        best_labels = AgglomerativeClustering(n_clusters=2, metric='cosine', linkage='average').fit_predict(X)
        best_k = 2

    last_label, li = 0, 0
    for i, seg in enumerate(segments):
        if li < len(valid_idx) and valid_idx[li] == i:
            last_label = int(best_labels[li])
            li += 1
        seg['speaker'] = f'SPEAKER_{last_label}'
    return segments, best_k


def extract_samples(wav_path, segments, job_dir):
    audio, sr = sf.read(wav_path)
    by_spk = {}
    for s in segments:
        by_spk.setdefault(s['speaker'], []).append(s)
    silence = np.zeros(int(0.2 * sr))
    samples = {}
    for sp, segs in by_spk.items():
        chunks = []
        total = 0
        for s in segs[:50]:
            if total >= 30: break
            si, ei = int(s['start'] * sr), int(s['end'] * sr)
            chunks.append(audio[si:ei])
            chunks.append(silence)
            total += s['end'] - s['start']
        sample = np.concatenate(chunks)
        path = job_dir / f'{sp}_sample.wav'
        sf.write(path, sample, sr)
        samples[sp] = {
            'path': str(path),
            'rel': f'/static/audio/{job_dir.name}/{path.name}',
            'duration': round(len(sample) / sr, 2),
        }
    return samples


def calc_cost(seconds):
    return round(seconds / 3600 * GPU_HOURLY_COST, 4)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def api_process():
    f = request.files.get('audio')
    if not f:
        return jsonify({'error': 'no file'}), 400
    auto = request.form.get('auto_speakers', 'true').lower() == 'true'
    n_speakers_hint = int(request.form.get('n_speakers', 0))

    job_id = uuid.uuid4().hex[:8]
    job_dir = AUDIO_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    timings, t_overall = {}, time.time()

    t0 = time.time()
    src_path = job_dir / f.filename
    f.save(src_path)
    file_size_mb = src_path.stat().st_size / 1024 / 1024
    timings['upload'] = round(time.time() - t0, 2)

    t0 = time.time()
    wav_path = job_dir / 'audio_16k.wav'
    to_wav_16k_mono(str(src_path), str(wav_path))
    duration = sf.info(str(wav_path)).duration
    timings['convert'] = round(time.time() - t0, 2)

    t0 = time.time()
    segs, lang_prob = transcribe(str(wav_path))
    timings['transcribe'] = round(time.time() - t0, 2)

    t0 = time.time()
    segs, n_detected = auto_diarize(str(wav_path), segs)
    timings['diarize'] = round(time.time() - t0, 2)

    t0 = time.time()
    samples = extract_samples(str(wav_path), segs, job_dir)
    timings['extract_samples'] = round(time.time() - t0, 2)

    timings['total'] = round(time.time() - t_overall, 2)
    ratios = {
        'transcribe_x_realtime': round(duration / timings['transcribe'], 2) if timings['transcribe'] > 0 else 0,
        'diarize_x_realtime': round(duration / timings['diarize'], 2) if timings['diarize'] > 0 else 0,
        'total_x_realtime': round(duration / timings['total'], 2) if timings['total'] > 0 else 0,
    }
    cost = {
        'mode': f'runpod_gpu' if DEVICE == 'cuda' else 'local_cpu',
        'gpu_cost_per_hour': GPU_HOURLY_COST if DEVICE == 'cuda' else 0,
        'estimated_cost_for_this_job': calc_cost(timings['total']) if DEVICE == 'cuda' else 0,
        'cost_per_minute_of_audio': round(calc_cost(timings['total']) / (duration / 60), 6) if DEVICE == 'cuda' and duration > 0 else 0,
        'breakdown': {
            'transcribe': calc_cost(timings['transcribe']) if DEVICE == 'cuda' else 0,
            'diarize': calc_cost(timings['diarize']) if DEVICE == 'cuda' else 0,
        },
    }

    JOBS[job_id] = {
        'wav_path': str(wav_path), 'segments': segs, 'samples': samples,
        'duration': duration, 'timings': timings, 'cost': cost,
    }

    return jsonify({
        'job_id': job_id,
        'audio_url': f'/static/audio/{job_id}/audio_16k.wav',
        'duration': duration, 'lang_prob': lang_prob,
        'segments': segs,
        'samples': {sp: {'url': samples[sp]['rel'], 'duration': samples[sp]['duration']} for sp in samples},
        'timings': timings, 'ratios': ratios, 'cost': cost,
        'file_size_mb': round(file_size_mb, 2),
        'segment_count': len(segs),
        'speakers_detected': n_detected,
        'auto_speakers': auto,
    })


@app.route('/api/clone', methods=['POST'])
def api_clone():
    data = request.json or {}
    job_id, text = data.get('job_id'), data.get('text', '').strip()
    speaker = data.get('speaker', 'SPEAKER_0')
    diff_steps = int(data.get('diffusion_steps', 40))
    cfg_scale = float(data.get('cfg_scale', 1.5))

    if not text or job_id not in JOBS:
        return jsonify({'error': 'invalid job or empty text'}), 400
    job = JOBS[job_id]
    if speaker not in job['samples']:
        return jsonify({'error': f'speaker not found: {speaker}'}), 400

    sp_path = job['samples'][speaker]['path']
    tagged = f'[1]: {text}' if not text.startswith('[') else text

    try:
        t0 = time.time()
        client = Client('vibingvoice/vibe-voice-custom-voices')
        connect_time = time.time() - t0
        t0 = time.time()
        result = client.predict(
            text=tagged, speaker1_audio_path=handle_file(sp_path),
            speaker2_audio_path=handle_file(sp_path),
            speaker3_audio_path=handle_file(sp_path),
            speaker4_audio_path=handle_file(sp_path),
            seed=42, diffusion_steps=diff_steps, cfg_scale=cfg_scale,
            use_sampling=False, temperature=0.95, top_p=0.95,
            max_words_per_chunk=250, api_name='/generate_speech_gradio',
        )
        gen_time = time.time() - t0
        out_name = f'cloned_{int(time.time())}.wav'
        out_path = AUDIO_DIR / job_id / out_name
        shutil.copy(result, out_path)
        out_duration = sf.info(str(out_path)).duration

        return jsonify({
            'audio_url': f'/static/audio/{job_id}/{out_name}',
            'elapsed': round(gen_time, 2), 'connect_time': round(connect_time, 2),
            'output_duration': round(out_duration, 2),
            'rtf': round(gen_time / out_duration, 2) if out_duration > 0 else 0,
            'chars': len(text), 'words': len(text.split()),
            'time_per_char_ms': round(gen_time / len(text) * 1000, 2),
            'cost_estimate': calc_cost(gen_time) if DEVICE == 'cuda' else 0,
            'mode': 'hf_spaces_zerogpu',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
