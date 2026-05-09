#!/bin/bash
# Deploy streaming server to a freshly-restarted Pod.
# Run AFTER:
# 1. Pod is started (Container disk gets wiped on restart, /workspace persists)
# 2. SSH key re-added to /root/.ssh/authorized_keys via web terminal
# 3. POD_PORT and POD_HOST exported (or hardcoded below)

set -euo pipefail
POD_HOST=${POD_HOST:-103.207.149.153}
POD_PORT=${POD_PORT:?set POD_PORT to the new SSH port from RunPod console}

echo "=== [1/5] verify SSH ==="
ssh -i ~/.ssh/id_ed25519 -p $POD_PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 root@$POD_HOST \
    "echo CONNECTED; nvidia-smi --query-gpu=name --format=csv,noheader; ls /workspace/ | head -10"

echo
echo "=== [2/5] reinstall pip deps (wiped on Pod stop) ==="
ssh -i ~/.ssh/id_ed25519 -p $POD_PORT root@$POD_HOST "
pip install --quiet --upgrade pip 2>&1 | tail -3
pip install --quiet 'transformers>=4.51,<5.0' accelerate soundfile librosa fastapi 'uvicorn[standard]' 2>&1 | tail -3
pip install --quiet flash-attn==2.6.3 --no-build-isolation 2>&1 | tail -3
echo
echo '=== reinstall community vv-pkg ==='
cd /workspace/vv-community && pip install --quiet -e . 2>&1 | tail -3
echo
python -c 'import torch, vibevoice, flash_attn; print(\"torch:\", torch.__version__, \"flash_attn:\", flash_attn.__version__)'
"

echo
echo "=== [3/5] upload server + client ==="
scp -i ~/.ssh/id_ed25519 -P $POD_PORT \
    server.py client.html \
    root@$POD_HOST:/workspace/

echo
echo "=== [4/5] kill old server, start new ==="
ssh -i ~/.ssh/id_ed25519 -p $POD_PORT root@$POD_HOST "
pkill -9 -f 'server.py|uvicorn' 2>/dev/null || true
sleep 2
cd /workspace && nohup env VV_MODEL='aoi-ot/VibeVoice-Large' VV_DIFF_STEPS=60 VV_CFG=1.8 VV_REF=/workspace/refs/01.mp3 VV_PORT=8080 python server.py > /tmp/server.log 2>&1 &
echo PID=\$!
"

echo
echo "=== [5/5] wait for /health to come up (model load takes ~10-30s) ==="
echo "  the SSH tunnel should be up locally before this returns"
echo "  next: bash tunnel.sh (with POD_PORT exported) then open http://127.0.0.1:8080/"
