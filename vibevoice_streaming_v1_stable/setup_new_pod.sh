#!/bin/bash
# Setup script for fresh RTX PRO 6000 Pod (torch 2.8 + cu128 + py3.12 + Ubuntu 24.04)
# Use --break-system-packages because of PEP 668 in Python 3.12+
set -euo pipefail

PIP="pip install --break-system-packages --quiet"

echo "=== [1/5] system info ==="
python --version
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"
df -h /workspace | tail -1

echo ""
echo "=== [2/5] install python deps ==="
$PIP --upgrade pip 2>&1 | tail -2
$PIP \
    'transformers>=4.51,<5.0' \
    accelerate \
    soundfile \
    librosa \
    fastapi \
    'uvicorn[standard]' \
    edge-tts \
    huggingface_hub 2>&1 | tail -3

echo ""
echo "=== [3/5] try install flash-attn for torch 2.8 + cu128 + py3.12 ==="
# Try latest flash-attn version (will pick wheel for current env)
$PIP flash-attn --no-build-isolation 2>&1 | tail -5 || echo "  flash-attn install may have failed - will fall back to eager"
python -c "import flash_attn; print('flash_attn:', flash_attn.__version__)" 2>&1 | tail -3 || echo "flash_attn import failed"

echo ""
echo "=== [4/5] clone VibeVoice community fork ==="
[ -d /workspace/vv-community ] || git clone --depth 1 https://github.com/vibevoice-community/VibeVoice /workspace/vv-community
cd /workspace/vv-community && $PIP -e . 2>&1 | tail -3

echo ""
echo "=== [5/5] verify VibeVoice import ==="
python -c "
import vibevoice
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
print('vibevoice imports OK')
" 2>&1 | tail -3

echo ""
echo "=== ✅ setup complete ==="
echo "Next: download VibeVoice-Large and reference audio"
