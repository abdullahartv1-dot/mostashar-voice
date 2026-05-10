#!/bin/bash
# Quick reinstall after Pod restart (preserves /workspace, loses container packages).
set -euo pipefail

echo "=== reinstalling pip packages ==="
pip install --quiet --ignore-installed blinker
pip install --quiet --ignore-installed \
  fastapi uvicorn pydantic 'transformers==4.51.3' accelerate \
  diffusers ml-collections av peft 'huggingface_hub>=0.25.1' \
  faster-whisper soundfile librosa 'numpy>=1.20' scipy \
  imageio-ffmpeg python-multipart speechbrain scikit-learn \
  'gradio>=4.0' spaces

echo "=== reinstall torchaudio + torchvision matching torch 2.11+cu130 ==="
pip uninstall -y torchaudio torchvision 2>&1 | tail -2
pip install --quiet torchaudio==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu130

echo "=== install vibevoice from /workspace/vv-pkg ==="
pip install --quiet -e /workspace/vv-pkg

echo "=== re-apply transformers JSON dtype patch ==="
python <<'PYEOF'
from pathlib import Path
import transformers, re
p = Path(transformers.__file__).parent / 'configuration_utils.py'
s = p.read_text()
new_s = re.sub(r'json\.dumps\(([^)]+)\)',
               lambda m: 'json.dumps(' + m.group(1) + ', default=str)' if 'default=' not in m.group(1) else m.group(0),
               s)
if new_s != s:
    p.write_text(new_s)
    print('PATCHED transformers')
else:
    print('SKIP (already patched)')
PYEOF

echo "=== verify imports ==="
python -c "
import torch, transformers, librosa, gradio, vibevoice
print(f'torch {torch.__version__} cuda:{torch.cuda.is_available()}')
print(f'transformers {transformers.__version__}')
print(f'gradio {gradio.__version__}')
print('vibevoice:', vibevoice.__file__)
"

echo
echo "=== ✅ reinstall done. Now run engine + gradio ==="
