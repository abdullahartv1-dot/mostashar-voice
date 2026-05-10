#!/bin/bash
# Full bootstrap from scratch on a fresh H100 Pod (volume + container both empty).
set -euo pipefail

VS=/workspace/voice-studio-v2

echo "=== [1/8] apt: ffmpeg + git ==="
apt-get update -qq && apt-get install -y -qq ffmpeg git build-essential

echo "=== [2/8] pip core deps (transformers 4.51.3 for VibeVoice ABI) ==="
pip install --quiet --ignore-installed blinker
pip install --quiet --ignore-installed \
  fastapi uvicorn pydantic 'transformers==4.51.3' accelerate \
  diffusers ml-collections av peft 'huggingface_hub>=0.25.1' \
  faster-whisper soundfile librosa 'numpy>=1.20' scipy \
  imageio-ffmpeg python-multipart speechbrain scikit-learn \
  'gradio>=4.0' spaces

echo "=== [3/8] clone VibeVoice GitHub package + install ==="
[ -d /workspace/vv-pkg ] || git clone --depth 1 https://github.com/microsoft/VibeVoice /workspace/vv-pkg
pip install --quiet -e /workspace/vv-pkg

echo "=== [4/8] patch transformers JSON dtype serialization ==="
python <<'PYEOF'
from pathlib import Path
import transformers, re
p = Path(transformers.__file__).parent / 'configuration_utils.py'
s = p.read_text()
new_s = re.sub(r'json\.dumps\(([^)]+)\)',
               lambda m: 'json.dumps(' + m.group(1) + ', default=str)' if 'default=' not in m.group(1) else m.group(0),
               s)
if new_s != s:
    p.write_text(new_s); print('PATCHED transformers')
else:
    print('SKIP (already patched)')
PYEOF

echo "=== [5/8] clone HF Space (vendor) + apply patches ==="
mkdir -p $VS/engine/vendor
cd $VS/engine/vendor
[ -d vibe-voice-custom-voices ] || git clone --depth 1 https://huggingface.co/spaces/vibingvoice/vibe-voice-custom-voices

python <<'PYEOF'
from pathlib import Path
# Patch 1: librosa resampler instead of np.interp
p = Path('/workspace/voice-studio-v2/engine/vendor/vibe-voice-custom-voices/nodes/base_vibevoice.py')
s = p.read_text()
old = """            # Resample if needed
            if input_sample_rate != target_sample_rate:
                target_length = int(len(audio_np) * target_sample_rate / input_sample_rate)
                audio_np = np.interp(np.linspace(0, len(audio_np), target_length),
                                   np.arange(len(audio_np)), audio_np)"""
new = """            # Resample if needed (high-quality via librosa)
            if input_sample_rate != target_sample_rate:
                import librosa as _librosa
                audio_np = _librosa.resample(audio_np.astype(np.float32),
                                             orig_sr=int(input_sample_rate),
                                             target_sr=int(target_sample_rate),
                                             res_type="soxr_hq")"""
if old in s and new not in s:
    p.write_text(s.replace(old, new)); print('PATCH1 librosa.resample applied')
else:
    print('PATCH1 SKIP')

# Patch 2: Gradio launch on 0.0.0.0:7860 + comfy_mock path + use 1.5B/Large via env
p2 = Path('/workspace/voice-studio-v2/engine/vendor/vibe-voice-custom-voices/app.py')
s2 = p2.read_text()
old2 = '''if project_root not in sys.path:
    sys.path.insert(0, project_root)'''
new2 = '''if project_root not in sys.path:
    sys.path.insert(0, project_root)

_comfy_mock = '/workspace/voice-studio-v2/engine/vendor/comfy_mock'
if _comfy_mock not in sys.path:
    sys.path.insert(0, _comfy_mock)'''
if old2 in s2 and 'comfy_mock' not in s2:
    s2 = s2.replace(old2, new2)
s2 = s2.replace('demo.launch()', 'demo.launch(server_name="0.0.0.0", server_port=7860, share=False)')
s2 = s2.replace("attention_type='auto'", "attention_type='eager'")
p2.write_text(s2)
print('PATCH2 Gradio launch + eager + comfy_mock applied')
PYEOF

echo "=== [6/8] create comfy_mock module ==="
mkdir -p $VS/engine/vendor/comfy_mock/comfy
echo '' > $VS/engine/vendor/comfy_mock/comfy/__init__.py
cat > $VS/engine/vendor/comfy_mock/comfy/model_management.py <<'EOF'
class InterruptProcessingException(Exception):
    pass
def throw_exception_if_processing_interrupted():
    pass
EOF

echo "=== [7/8] download models (VibeVoice-Large + 1.5B + ASR-HF + Qwen 1.5B/7B) ==="
HF_HOME=$VS/engine/vendor/vibe-voice-custom-voices/vibevoice python <<'PYEOF'
import os
os.environ['HF_HOME']='/workspace/voice-studio-v2/engine/vendor/vibe-voice-custom-voices/vibevoice'
from huggingface_hub import snapshot_download
cache='/workspace/voice-studio-v2/engine/vendor/vibe-voice-custom-voices/vibevoice'
print('1/5 VibeVoice-Large...')
snapshot_download('aoi-ot/VibeVoice-Large', cache_dir=cache)
print('2/5 VibeVoice-1.5B...')
snapshot_download('microsoft/VibeVoice-1.5B', cache_dir=cache)
print('3/5 VibeVoice-ASR-HF...')
snapshot_download('microsoft/VibeVoice-ASR-HF', cache_dir=cache)
print('4/5 Qwen2.5-1.5B (tokenizer)...')
snapshot_download('Qwen/Qwen2.5-1.5B', cache_dir=cache, allow_patterns=['tokenizer*','vocab*','merges*','config.json','special_tokens*'])
print('5/5 Qwen2.5-7B (full, for ASR-HF LM)...')
snapshot_download('Qwen/Qwen2.5-7B', cache_dir=cache)
print('all models done')
PYEOF

echo "=== [8/8] verify ==="
df -h /workspace
python -c "
import torch, transformers, librosa, gradio, vibevoice
print(f'torch {torch.__version__} cuda:{torch.cuda.is_available()}')
print(f'transformers {transformers.__version__}')
print(f'gradio {gradio.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo "=== ✅ full setup complete ==="
