#!/bin/bash
# Continuation of H100 bootstrap (run after pip install for main deps completes).
set -euo pipefail

VS_HOME=/workspace/voice-studio-v2

echo "=== flash-attention 2.6.3 (cp311 prebuilt) ==="
pip install --quiet --ignore-installed \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
python -c "import flash_attn; print('flash_attn:', flash_attn.__version__)"

echo
echo "=== Clone vendored VibeVoice Space ==="
mkdir -p "$VS_HOME/engine/vendor"
cd "$VS_HOME/engine/vendor"
if [ ! -d vibe-voice-custom-voices ]; then
  git clone --depth 1 https://huggingface.co/spaces/vibingvoice/vibe-voice-custom-voices
fi

echo
echo "=== Apply librosa-resample patch ==="
python <<'PYEOF'
from pathlib import Path
p = Path("/workspace/voice-studio-v2/engine/vendor/vibe-voice-custom-voices/nodes/base_vibevoice.py")
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
    print("librosa.resample patch already applied or anchor moved")
PYEOF

echo
echo "=== comfy_mock module (needed by VibeVoice node code) ==="
mkdir -p "$VS_HOME/engine/vendor/comfy_mock/comfy"
cat > "$VS_HOME/engine/vendor/comfy_mock/comfy/__init__.py" <<'EOF'
EOF
cat > "$VS_HOME/engine/vendor/comfy_mock/comfy/model_management.py" <<'EOF'
"""Minimal ComfyUI model_management shim for running VibeVoice node code outside ComfyUI."""
class InterruptProcessingException(Exception):
    pass

def throw_exception_if_processing_interrupted():
    pass
EOF

echo
echo "=== Pre-download VibeVoice 1.5B + Qwen tokenizer ==="
HF_HOME="$VS_HOME/hf-cache" python <<'PYEOF'
import os
os.environ["HF_HOME"] = "/workspace/voice-studio-v2/hf-cache"
from huggingface_hub import snapshot_download
print("Downloading microsoft/VibeVoice-1.5B...")
p = snapshot_download(
    "microsoft/VibeVoice-1.5B",
    cache_dir="/workspace/voice-studio-v2/engine/vendor/vibe-voice-custom-voices/vibevoice",
)
print(" ->", p)
print("Downloading Qwen/Qwen2.5-1.5B (tokenizer + config only)...")
p = snapshot_download(
    "Qwen/Qwen2.5-1.5B",
    cache_dir="/workspace/voice-studio-v2/engine/vendor/vibe-voice-custom-voices/vibevoice",
    allow_patterns=["tokenizer*", "vocab*", "merges*", "config.json", "special_tokens*"],
)
print(" ->", p)
print("done")
PYEOF

echo
echo "=== Verify imports ==="
python -c "
import torch
import flash_attn
import transformers
import librosa
print(f'torch {torch.__version__} cuda:{torch.cuda.is_available()} cudnn:{torch.backends.cudnn.enabled}')
print(f'flash_attn {flash_attn.__version__}')
print(f'transformers {transformers.__version__}')
print(f'librosa {librosa.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')
"

echo
echo "=== ✅ Bootstrap p2 complete ==="
