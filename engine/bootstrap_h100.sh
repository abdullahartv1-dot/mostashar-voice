#!/bin/bash
# H100 Pod bootstrap — installs deps, downloads VibeVoice 1.5B, starts engine.
# Usage on Pod: bash /workspace/bootstrap_h100.sh
set -euo pipefail

VS_HOME=/workspace/voice-studio-v2
mkdir -p "$VS_HOME"

echo "=== [1/6] System packages ==="
apt-get update -qq && apt-get install -y -qq ffmpeg git build-essential

echo "=== [2/6] Python deps ==="
pip install --quiet --upgrade pip
pip install --quiet \
  fastapi uvicorn pydantic 'transformers>=4.51.3' accelerate \
  diffusers ml-collections av peft 'huggingface_hub>=0.25.1' \
  faster-whisper soundfile librosa 'numpy>=1.20' scipy \
  imageio-ffmpeg python-multipart speechbrain scikit-learn \
  TTS  # XTTS-v2

echo "=== [3/6] flash-attention 2.6.3 (cp311 prebuilt) ==="
pip install --quiet \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

echo "=== [4/6] Verify flash-attn ==="
python -c "import flash_attn; print('flash_attn:', flash_attn.__version__)"

echo "=== [5/6] Clone vendored VibeVoice Space ==="
mkdir -p "$VS_HOME/engine/vendor"
cd "$VS_HOME/engine/vendor"
if [ ! -d vibe-voice-custom-voices ]; then
  git clone https://huggingface.co/spaces/vibingvoice/vibe-voice-custom-voices
fi
# Apply librosa-resample patch (replace np.interp with high-quality resampler)
python <<'PYEOF'
from pathlib import Path
p = Path("vibe-voice-custom-voices/nodes/base_vibevoice.py")
src = p.read_text()
old = """            # Resample if needed
            if input_sample_rate != target_sample_rate:
                target_length = int(len(audio_np) * target_sample_rate / input_sample_rate)
                audio_np = np.interp(np.linspace(0, len(audio_np), target_length),
                                   np.arange(len(audio_np)), audio_np)"""
new = """            # Resample if needed (high-quality via librosa instead of np.interp linear)
            if input_sample_rate != target_sample_rate:
                import librosa as _librosa
                audio_np = _librosa.resample(audio_np.astype(np.float32),
                                             orig_sr=int(input_sample_rate),
                                             target_sr=int(target_sample_rate),
                                             res_type="soxr_hq")"""
if old in src:
    p.write_text(src.replace(old, new))
    print("librosa.resample patch applied")
else:
    print("librosa.resample patch already applied (or anchor moved)")
PYEOF

echo "=== [6/6] Pre-download VibeVoice 1.5B + Qwen tokenizer ==="
mkdir -p "$VS_HOME/engine/vendor/vibe-voice-custom-voices/vibevoice"
HF_HOME="$VS_HOME/hf-cache" python <<'PYEOF'
import os
os.environ["HF_HOME"] = "/workspace/voice-studio-v2/hf-cache"
from huggingface_hub import snapshot_download
print("Downloading microsoft/VibeVoice-1.5B...")
snapshot_download("microsoft/VibeVoice-1.5B", cache_dir="/workspace/voice-studio-v2/engine/vendor/vibe-voice-custom-voices/vibevoice")
print("Downloading Qwen/Qwen2.5-1.5B (tokenizer only)...")
snapshot_download("Qwen/Qwen2.5-1.5B", cache_dir="/workspace/voice-studio-v2/engine/vendor/vibe-voice-custom-voices/vibevoice", allow_patterns=["tokenizer*", "vocab*", "merges*", "config.json"])
print("done")
PYEOF

echo
echo "=== ✅ Bootstrap complete. Engine code will be rsynced separately. ==="
