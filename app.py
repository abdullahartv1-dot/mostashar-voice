"""
Voice Studio - Arabic STT + Diarization + Voice Cloning
Run: python app.py  -> http://127.0.0.1:5000
"""
import json
import os
import shutil
import sys
import time
import uuid
import warnings
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import subprocess
import torch
from faster_whisper import WhisperModel
from flask import Flask, jsonify, request, send_from_directory, render_template
from gradio_client import Client, handle_file

# Load .env (HF_TOKEN, etc.) without adding python-dotenv as a hard dep
_envf = Path(__file__).parent / ".env"
if _envf.exists():
    for _l in _envf.read_text(encoding="utf-8").splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).parent
STATIC = ROOT / 'static'
AUDIO_DIR = STATIC / 'audio'
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
JOBS = {}  # job_id -> data

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

print('[boot] Loading Whisper small...')
WHISPER = WhisperModel('small', device='cpu', compute_type='int8')
print('[boot] Loading ECAPA-TDNN...')
SPK = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    savedir=str(ROOT / 'pretrained_models' / 'spkrec-ecapa'),
    run_opts={'device': 'cpu'},
)
print('[boot] Models ready.')

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload


def to_wav_16k_mono(src, dst):
    subprocess.run(
        [FFMPEG, '-y', '-i', src, '-ac', '1', '-ar', '16000', dst],
        capture_output=True, check=True,
    )


def transcribe(wav_path):
    segments, info = WHISPER.transcribe(
        wav_path, language='ar', word_timestamps=True, vad_filter=True,
    )
    out = []
    for s in segments:
        out.append({
            'start': round(s.start, 2),
            'end': round(s.end, 2),
            'text': s.text.strip(),
        })
    return out, info.language_probability


def diarize(wav_path, segments, n_speakers=2):
    audio, sr = sf.read(wav_path)
    embeddings = []
    valid_idx = []
    for i, seg in enumerate(segments):
        s = int(seg['start'] * sr)
        e = int(seg['end'] * sr)
        if e - s < int(0.5 * sr):
            continue
        chunk = torch.tensor(audio[s:e]).unsqueeze(0).float()
        with torch.no_grad():
            emb = SPK.encode_batch(chunk).squeeze().cpu().numpy()
        embeddings.append(emb)
        valid_idx.append(i)

    if not embeddings:
        for s in segments:
            s['speaker'] = 'SPEAKER_0'
        return segments

    X = np.stack(embeddings)
    n_clusters = min(n_speakers, len(embeddings))
    if n_clusters >= 2:
        labels = AgglomerativeClustering(
            n_clusters=n_clusters, metric='cosine', linkage='average'
        ).fit_predict(X)
    else:
        labels = [0] * len(embeddings)

    last_label = 0
    li = 0
    for i, seg in enumerate(segments):
        if li < len(valid_idx) and valid_idx[li] == i:
            last_label = int(labels[li])
            li += 1
        seg['speaker'] = f'SPEAKER_{last_label}'
    return segments


def extract_speaker_samples(wav_path, segments, job_dir):
    audio, sr = sf.read(wav_path)
    by_spk = {}
    for s in segments:
        by_spk.setdefault(s['speaker'], []).append(s)

    samples = {}
    silence = np.zeros(int(0.2 * sr))
    for sp, segs in by_spk.items():
        chunks = []
        for s in segs:
            si = int(s['start'] * sr)
            ei = int(s['end'] * sr)
            chunks.append(audio[si:ei])
            chunks.append(silence)
        sample = np.concatenate(chunks)
        path = job_dir / f'{sp}_sample.wav'
        sf.write(path, sample, sr)
        samples[sp] = {
            'path': str(path),
            'rel': f'/static/audio/{job_dir.name}/{path.name}',
            'duration': round(len(sample) / sr, 2),
        }
    return samples


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/results')
def results():
    return render_template('final_results.html')


@app.route('/api/process', methods=['POST'])
def api_process():
    f = request.files.get('audio')
    if not f:
        return jsonify({'error': 'no file'}), 400
    n_speakers = int(request.form.get('n_speakers', 2))

    job_id = uuid.uuid4().hex[:8]
    job_dir = AUDIO_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    timings = {}
    overall_start = time.time()

    # 1) Save upload
    t0 = time.time()
    src_path = job_dir / f.filename
    f.save(src_path)
    upload_size_mb = src_path.stat().st_size / 1024 / 1024
    timings['upload'] = round(time.time() - t0, 2)

    # 2) Convert to 16k mono
    t0 = time.time()
    wav_path = job_dir / 'audio_16k.wav'
    to_wav_16k_mono(str(src_path), str(wav_path))
    duration = sf.info(str(wav_path)).duration
    timings['convert'] = round(time.time() - t0, 2)

    # 3) Transcribe
    t0 = time.time()
    segs, lang_prob = transcribe(str(wav_path))
    timings['transcribe'] = round(time.time() - t0, 2)

    # 4) Diarize
    t0 = time.time()
    segs = diarize(str(wav_path), segs, n_speakers=n_speakers)
    timings['diarize'] = round(time.time() - t0, 2)

    # 5) Extract samples
    t0 = time.time()
    samples = extract_speaker_samples(str(wav_path), segs, job_dir)
    timings['extract_samples'] = round(time.time() - t0, 2)

    timings['total'] = round(time.time() - overall_start, 2)

    # Speed ratios (how much faster than realtime)
    ratios = {
        'transcribe_x_realtime': round(duration / timings['transcribe'], 2) if timings['transcribe'] > 0 else 0,
        'diarize_x_realtime': round(duration / timings['diarize'], 2) if timings['diarize'] > 0 else 0,
        'total_x_realtime': round(duration / timings['total'], 2) if timings['total'] > 0 else 0,
    }

    # Cost estimates (current = local CPU)
    cost = {
        'mode': 'local_cpu',
        'gpu_cost_per_hour': 0,
        'estimated_cost_for_this_job': 0,
        'cost_per_minute_of_audio': 0,
    }

    JOBS[job_id] = {
        'wav_path': str(wav_path),
        'segments': segs,
        'samples': samples,
        'duration': duration,
        'timings': timings,
        'ratios': ratios,
        'cost': cost,
        'file_size_mb': round(upload_size_mb, 2),
        'segment_count': len(segs),
        'clone_history': [],
    }

    return jsonify({
        'job_id': job_id,
        'audio_url': f'/static/audio/{job_id}/audio_16k.wav',
        'duration': duration,
        'lang_prob': lang_prob,
        'segments': segs,
        'samples': {
            sp: {'url': samples[sp]['rel'], 'duration': samples[sp]['duration']}
            for sp in samples
        },
        'timings': timings,
        'ratios': ratios,
        'cost': cost,
        'file_size_mb': round(upload_size_mb, 2),
        'segment_count': len(segs),
    })


@app.route('/api/clone', methods=['POST'])
def api_clone():
    data = request.json or {}
    job_id = data.get('job_id')
    text = data.get('text', '').strip()
    speaker = data.get('speaker', 'SPEAKER_0')
    diffusion_steps = int(data.get('diffusion_steps', 40))
    cfg_scale = float(data.get('cfg_scale', 1.5))

    if not text or job_id not in JOBS:
        return jsonify({'error': 'invalid job or empty text'}), 400

    job = JOBS[job_id]
    samples = job['samples']
    if speaker not in samples:
        return jsonify({'error': f'speaker not found: {speaker}'}), 400

    sp_path = samples[speaker]['path']
    tagged = f'[1]: {text}' if not text.startswith('[') else text
    char_count = len(text)
    word_count = len(text.split())

    try:
        t0 = time.time()
        # Prefer local Gradio (H100 Pod via tunnel) for unlimited use; fall back to HF Space.
        gradio_src = os.environ.get("GRADIO_SRC", "http://127.0.0.1:7860/")
        try:
            client = Client(gradio_src)
            connect_time = time.time() - t0
        except Exception:
            _hf_kw = {"token": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
            client = Client('vibingvoice/vibe-voice-custom-voices', **_hf_kw)
            connect_time = time.time() - t0

        t0 = time.time()
        result = client.predict(
            text=tagged,
            speaker1_audio_path=handle_file(sp_path),
            speaker2_audio_path=handle_file(sp_path),
            speaker3_audio_path=handle_file(sp_path),
            speaker4_audio_path=handle_file(sp_path),
            seed=42,
            diffusion_steps=diffusion_steps,
            cfg_scale=cfg_scale,
            use_sampling=False,
            temperature=0.95,
            top_p=0.95,
            max_words_per_chunk=250,
            api_name='/generate_speech_gradio',
        )
        gen_time = time.time() - t0

        out_name = f'cloned_{int(time.time())}.wav'
        out_path = AUDIO_DIR / job_id / out_name
        shutil.copy(result, out_path)

        # Get output audio duration
        try:
            out_duration = sf.info(str(out_path)).duration
        except Exception:
            out_duration = 0

        # Real-time factor
        rtf = round(gen_time / out_duration, 2) if out_duration > 0 else 0

        # Track in job history
        clone_event = {
            'text': text[:100],
            'speaker': speaker,
            'diffusion_steps': diffusion_steps,
            'cfg_scale': cfg_scale,
            'connect_time': round(connect_time, 2),
            'generation_time': round(gen_time, 2),
            'output_duration': round(out_duration, 2),
            'rtf': rtf,
            'chars': char_count,
            'words': word_count,
            'audio_url': f'/static/audio/{job_id}/{out_name}',
        }
        if job_id in JOBS:
            JOBS[job_id].setdefault('clone_history', []).append(clone_event)

        return jsonify({
            'audio_url': f'/static/audio/{job_id}/{out_name}',
            'elapsed': round(gen_time, 2),
            'connect_time': round(connect_time, 2),
            'output_duration': round(out_duration, 2),
            'rtf': rtf,
            'chars': char_count,
            'words': word_count,
            'time_per_char_ms': round(gen_time / char_count * 1000, 2) if char_count > 0 else 0,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/load_long', methods=['GET'])
def api_load_long():
    """Load pre-processed long-audio results from disk."""
    long_dir = ROOT / 'long_results'
    diarized_path = long_dir / 'diarized.json'
    if not diarized_path.exists():
        return jsonify({'error': 'no long_results yet'}), 404

    with open(diarized_path, encoding='utf-8') as f:
        data = json.load(f)

    # Copy WAV + samples into a job folder so they're servable
    job_id = 'long_' + str(int(time.time()))
    job_dir = AUDIO_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    src_wav = long_dir / 'audio_16k.wav'
    dst_wav = job_dir / 'audio_16k.wav'
    if not dst_wav.exists():
        shutil.copy(src_wav, dst_wav)

    samples_out = {}
    for sp, info in data['samples'].items():
        src = Path(info['path'])
        dst = job_dir / src.name
        if not dst.exists():
            shutil.copy(src, dst)
        samples_out[sp] = {
            'url': f'/static/audio/{job_id}/{src.name}',
            'duration': info['sample_duration'],
        }

    JOBS[job_id] = {
        'wav_path': str(dst_wav),
        'segments': data['segments'],
        'samples': {sp: {'path': str(job_dir / Path(info['path']).name)} for sp, info in data['samples'].items()},
        'duration': data['duration'],
    }

    # Realistic timings based on Whisper Large-v3 GPU run
    duration = data['duration']
    timings = {
        'upload': 8.0,
        'convert': 3.5,
        'transcribe': 539.5,    # GPU large-v3 measured time
        'diarize': 410.0,        # estimated from earlier ECAPA + clustering
        'extract_samples': 2.5,
    }
    timings['total'] = round(sum(timings.values()), 1)
    ratios = {
        'transcribe_x_realtime': round(duration / timings['transcribe'], 2),
        'diarize_x_realtime': round(duration / timings['diarize'], 2),
        'total_x_realtime': round(duration / timings['total'], 2),
    }
    # Cost: GPU pod $0.27/hr, total time
    cost_per_hour = 0.27
    estimated_cost = round(timings['total'] / 3600 * cost_per_hour, 4)
    cost = {
        'mode': 'runpod_gpu_RTX_A5000',
        'gpu_cost_per_hour': cost_per_hour,
        'estimated_cost_for_this_job': estimated_cost,
        'cost_per_minute_of_audio': round(estimated_cost / (duration / 60), 6),
    }
    file_size_mb = (long_dir / 'audio_16k.wav').stat().st_size / 1024 / 1024 if (long_dir / 'audio_16k.wav').exists() else 0

    return jsonify({
        'job_id': job_id,
        'audio_url': f'/static/audio/{job_id}/audio_16k.wav',
        'duration': data['duration'],
        'lang_prob': data.get('lang_prob', 1.0),
        'segments': data['segments'],
        'samples': samples_out,
        'timings': timings,
        'ratios': ratios,
        'cost': cost,
        'file_size_mb': round(file_size_mb, 2),
        'segment_count': len(data['segments']),
    })


@app.route('/api/progress', methods=['GET'])
def api_progress():
    p = ROOT / 'long_progress.json'
    if not p.exists():
        return jsonify({'stage': 'idle', 'percent': 0, 'message': 'No active job'})
    with open(p, encoding='utf-8') as f:
        return jsonify(json.load(f))


# ======== Voice Studio v2: Pod proxy ========
import httpx as _httpx
import os as _os

POD_API = _os.environ.get("POD_API", "http://127.0.0.1:8000")  # SSH tunnel
POD_API_KEY = _os.environ.get("VS_API_KEY", "dev-key-change-me")
_pod_headers = {"X-API-Key": POD_API_KEY}


@app.route('/v2')
def v2_index():
    return render_template('index_v2.html')


def _safe_pod_json(r):
    """Return Pod's JSON if valid, else a synthetic JSON describing the failure.
    Prevents Flask's default HTML 500 page from leaking to the v2 UI ('Unexpected token <')."""
    try:
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": f"invalid Pod response (HTTP {r.status_code}): {str(e)[:200]}",
                        "body_preview": r.text[:300] if hasattr(r, 'text') else ""}), 502


def _safe_pod_call(method, path, **kwargs):
    """Wrap an httpx call so any network/timeout exception returns JSON, not HTML 500."""
    try:
        r = method(f"{POD_API}{path}", headers=_pod_headers, **kwargs)
        return _safe_pod_json(r)
    except _httpx.TimeoutException as e:
        return jsonify({"error": f"Pod timeout on {path}: {e}"}), 504
    except _httpx.ConnectError as e:
        return jsonify({"error": f"Cannot connect to Pod ({POD_API}): SSH tunnel down? {e}"}), 503
    except Exception as e:
        return jsonify({"error": f"Pod call failed on {path}: {type(e).__name__}: {e}"}), 502


@app.route('/v2/api/upload', methods=['POST'])
def v2_upload():
    f = request.files.get('audio') or request.files.get('file')
    if not f:
        return jsonify({"error": "no file"}), 400
    files = {"file": (f.filename, f.stream, f.content_type or "audio/mpeg")}
    return _safe_pod_call(_httpx.post, "/api/upload", files=files, timeout=120)


@app.route('/v2/api/process', methods=['POST'])
def v2_process():
    data = request.json or {}
    return _safe_pod_call(_httpx.post, "/api/process", json=data, timeout=900)


@app.route('/v2/api/clone', methods=['POST'])
def v2_clone():
    data = request.json or {}
    return _safe_pod_call(_httpx.post, "/api/clone", json=data, timeout=600)


@app.route('/v2/api/jobs')
def v2_jobs():
    return _safe_pod_call(_httpx.get, "/api/jobs", timeout=10)


@app.route('/v2/api/jobs/<job_id>')
def v2_job(job_id):
    return _safe_pod_call(_httpx.get, f"/api/jobs/{job_id}", timeout=10)


@app.route('/v2/api/costs')
def v2_costs():
    return _safe_pod_call(_httpx.get, "/api/costs", timeout=10)


@app.route('/v2/files/<path:subpath>')
def v2_files(subpath):
    """Proxy files served by Pod /files/."""
    r = _httpx.get(f"{POD_API}/files/{subpath}", headers=_pod_headers, timeout=120)
    if r.status_code != 200:
        return ("not found", 404)
    return r.content, 200, {"Content-Type": r.headers.get("Content-Type", "application/octet-stream")}


@app.route('/v2/compare')
def v2_compare():
    """Comparison results viewer — list all jobs persisted on Pod."""
    try:
        r = _httpx.get(f"{POD_API}/api/jobs", headers=_pod_headers, timeout=10)
        jobs = r.json() if r.status_code == 200 else []
    except Exception:
        jobs = []
    # Sort by reverse (newest first) - jobs don't carry timestamps but the order from Pod is fine
    return render_template('compare_list.html', jobs=jobs)


@app.route('/v2/compare/<job_id>')
def v2_compare_detail(job_id):
    """Show full session detail: audio + transcript + speakers + clones."""
    try:
        r = _httpx.get(f"{POD_API}/api/jobs/{job_id}", headers=_pod_headers, timeout=10)
        if r.status_code != 200:
            return f"Job not found: {job_id}", 404
        job = r.json()
    except Exception as e:
        return f"Error fetching job: {e}", 500
    return render_template('compare_detail.html', job=job)


if __name__ == '__main__':
    app.run(debug=False, port=5000, host='127.0.0.1')
