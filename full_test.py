"""
Full pipeline test:
1. STT: Whisper Large-v3 (already done on 03.mp3)
2. Diarization: ECAPA-TDNN (already done on 03.mp3)
3. Voice Cloning: VibeVoice TTS via Pod GPU (port 7860)

Saves all results to results_full.json for the comparison page.
"""
import json, sys, time, shutil
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from gradio_client import Client, handle_file

ROOT = Path('C:/Users/Admin/Desktop/project/x')

# Load existing results
print('Loading existing Whisper Large-v3 + Diarization results...')
with open(ROOT / 'long_results' / 'diarized.json', encoding='utf-8') as f:
    diarized = json.load(f)
with open(ROOT / '03_whisper_large.json', encoding='utf-8') as f:
    whisper_large = json.load(f)

print(f"  Audio: 93:29 minutes ({diarized['duration']:.0f}s)")
print(f"  Whisper Large-v3: {whisper_large['transcribe_time']}s ({whisper_large['speedup']}x realtime), {len(whisper_large['segments'])} segments")
print(f"  Diarization speakers: {len(diarized['samples'])}")

# Run VibeVoice TTS test on GPU Pod via SSH tunnel
print('\nConnecting to VibeVoice TTS on Pod GPU (port 7860)...')
client = Client('http://127.0.0.1:7860/')

# Use SPEAKER_0 sample (35 seconds reference)
sample_path = str(ROOT / 'long_results' / 'SPEAKER_0_sample.wav')

# Test text
test_text = '[1]: مرحباً بكم في تجربة استنساخ الصوت العربي على بطاقة الرسوميات. هذا النص الذي تسمعونه الآن مولّد بالكامل بواسطة الذكاء الاصطناعي على الخادم السحابي.'

print(f'  Speaker reference: SPEAKER_0_sample.wav (~35s from 03.mp3)')
print(f'  Test text: {test_text[:80]}...')

# Test 1: Default settings (40 steps, 1.5 cfg)
print('\nTest 1: Default quality (40 steps, 1.5 cfg)')
t0 = time.time()
result1 = client.predict(
    text=test_text,
    speaker1_audio_path=handle_file(sample_path),
    speaker2_audio_path=handle_file(sample_path),
    speaker3_audio_path=handle_file(sample_path),
    speaker4_audio_path=handle_file(sample_path),
    seed=42, diffusion_steps=40, cfg_scale=1.5,
    use_sampling=False, temperature=0.95, top_p=0.95,
    max_words_per_chunk=250,
    api_name='/generate_speech_gradio',
)
gen1_time = time.time() - t0
out1 = ROOT / 'gpu_clone_default.wav'
shutil.copy(result1, out1)
import soundfile as sf
out1_dur = sf.info(str(out1)).duration
print(f'  Done in {gen1_time:.1f}s, output: {out1_dur:.1f}s, RTF: {gen1_time/out1_dur:.2f}x')

# Test 2: High quality (80 steps, 2.0 cfg)
print('\nTest 2: High quality (80 steps, 2.0 cfg)')
t0 = time.time()
result2 = client.predict(
    text=test_text,
    speaker1_audio_path=handle_file(sample_path),
    speaker2_audio_path=handle_file(sample_path),
    speaker3_audio_path=handle_file(sample_path),
    speaker4_audio_path=handle_file(sample_path),
    seed=42, diffusion_steps=80, cfg_scale=2.0,
    use_sampling=False, temperature=0.95, top_p=0.95,
    max_words_per_chunk=250,
    api_name='/generate_speech_gradio',
)
gen2_time = time.time() - t0
out2 = ROOT / 'gpu_clone_high.wav'
shutil.copy(result2, out2)
out2_dur = sf.info(str(out2)).duration
print(f'  Done in {gen2_time:.1f}s, output: {out2_dur:.1f}s, RTF: {gen2_time/out2_dur:.2f}x')

# Cost calculation
GPU_HOURLY = 0.27
total_seconds = (whisper_large['transcribe_time'] +  # transcribe
                 410 +  # diarization (estimated from earlier)
                 gen1_time + gen2_time)  # TTS

results = {
    'audio_file': '03.mp3',
    'audio_duration_sec': diarized['duration'],
    'audio_duration_min': diarized['duration'] / 60,
    'gpu_hourly_cost': GPU_HOURLY,

    'pipeline': {
        'transcribe': {
            'model': 'Whisper Large-v3',
            'device': 'GPU RTX A5000',
            'time_sec': whisper_large['transcribe_time'],
            'speedup': whisper_large['speedup'],
            'segments': len(whisper_large['segments']),
            'cost_usd': round(whisper_large['transcribe_time'] / 3600 * GPU_HOURLY, 4),
        },
        'diarize': {
            'model': 'ECAPA-TDNN + Agglomerative Clustering',
            'device': 'GPU RTX A5000',
            'time_sec': 410,  # estimated
            'speakers_detected': len(diarized['samples']),
            'cost_usd': round(410 / 3600 * GPU_HOURLY, 4),
        },
        'voice_clone_default': {
            'model': 'VibeVoice-Large',
            'device': 'GPU RTX A5000',
            'time_sec': round(gen1_time, 2),
            'output_duration': round(out1_dur, 2),
            'rtf': round(gen1_time / out1_dur, 2),
            'diffusion_steps': 40,
            'cfg_scale': 1.5,
            'cost_usd': round(gen1_time / 3600 * GPU_HOURLY, 4),
            'output_file': 'gpu_clone_default.wav',
        },
        'voice_clone_high': {
            'model': 'VibeVoice-Large',
            'device': 'GPU RTX A5000',
            'time_sec': round(gen2_time, 2),
            'output_duration': round(out2_dur, 2),
            'rtf': round(gen2_time / out2_dur, 2),
            'diffusion_steps': 80,
            'cfg_scale': 2.0,
            'cost_usd': round(gen2_time / 3600 * GPU_HOURLY, 4),
            'output_file': 'gpu_clone_high.wav',
        },
    },

    'totals': {
        'time_sec': total_seconds,
        'cost_usd': round(total_seconds / 3600 * GPU_HOURLY, 4),
        'cost_per_minute_audio': round(total_seconds / 3600 * GPU_HOURLY / (diarized['duration'] / 60), 6),
    },

    'test_text': test_text,
    'reference_speaker': 'SPEAKER_0',
    'timestamp': time.time(),
}

with open(ROOT / 'results_full.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print('\n=== FINAL SUMMARY ===')
print(f"Audio:           93:29 ({diarized['duration']:.0f}s)")
print(f"Transcribe:      {whisper_large['transcribe_time']}s ({whisper_large['speedup']}x realtime) = ${results['pipeline']['transcribe']['cost_usd']}")
print(f"Diarize:         410s = ${results['pipeline']['diarize']['cost_usd']}")
print(f"Clone default:   {gen1_time:.1f}s ({out1_dur:.1f}s output) = ${results['pipeline']['voice_clone_default']['cost_usd']}")
print(f"Clone high:      {gen2_time:.1f}s ({out2_dur:.1f}s output) = ${results['pipeline']['voice_clone_high']['cost_usd']}")
print(f"TOTAL:           {total_seconds:.0f}s = ${results['totals']['cost_usd']}")
print(f"\nSaved: results_full.json")
print(f"Audio outputs: gpu_clone_default.wav, gpu_clone_high.wav")
