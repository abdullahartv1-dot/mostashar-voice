#!/bin/bash
set -euo pipefail

echo "=== [1/6] system + workspace setup ==="
mkdir -p /workspace/vv && cd /workspace/vv

echo "=== [2/6] python deps ==="
pip install --quiet --upgrade pip
pip install --quiet \
    'transformers>=4.51,<5.0' \
    accelerate \
    soundfile \
    librosa \
    fastapi \
    'uvicorn[standard]' \
    edge-tts \
    huggingface_hub

echo "=== [3/6] clone vibevoice ==="
[ -d /workspace/vv-pkg ] || git clone --depth 1 https://github.com/microsoft/VibeVoice /workspace/vv-pkg
cd /workspace/vv-pkg
pip install --quiet -e .

echo "=== [4/6] vendor patches (from prior sessions) ==="
PKG=/workspace/vv-pkg/vibevoice

# Patch 1: modeling_utils import (transformers 4.51 path)
sed -i 's|^from transformers import modeling_utils$|import transformers.modeling_utils as modeling_utils|' \
    "$PKG/modular/modeling_vibevoice.py" \
    "$PKG/modular/modeling_vibevoice_inference.py" \
    "$PKG/modular/modeling_vibevoice_streaming_inference.py" 2>/dev/null || true

# Patch 2: tie_weights signature
sed -i 's|def tie_weights(self):|def tie_weights(self, **_kwargs):|' \
    "$PKG/modular/modeling_vibevoice.py" 2>/dev/null || true

# Patch 3: AutoModel.register exist_ok
find "$PKG" -name '*.py' -exec sed -i \
    's|AutoModel.register(\([^,]*\), \([^,)]*\))|AutoModel.register(\1, \2, exist_ok=True)|g' {} \;

# Patch 4: Qwen2TokenizerFast import
sed -i 's|from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast|from transformers import Qwen2TokenizerFast|' \
    $(grep -rl 'tokenization_qwen2_fast' "$PKG" 2>/dev/null) 2>/dev/null || true

echo "=== [5/6] preflight: import test ==="
python -c "
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
print('imports OK')
" 2>&1 | tail -5

echo "=== [6/6] download VibeVoice-1.5B (smaller, faster for streaming test) ==="
python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('microsoft/VibeVoice-1.5B', cache_dir='/workspace/vv/models')
print('model at:', p)
" 2>&1 | tail -3

echo
echo "=== ✅ install complete ==="
echo "next: bash run_test.sh"
