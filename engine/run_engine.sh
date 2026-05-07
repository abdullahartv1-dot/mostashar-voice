#!/bin/bash
# Launches engine as a true daemon. Survives SSH disconnects.
set -euo pipefail

cd /workspace/voice-studio-v2

# Stop any existing engine
pkill -9 -f "uvicorn engine.main" 2>/dev/null || true
sleep 1

# Source env
export VS_HOME=/workspace/voice-studio-v2
export HF_HOME=$VS_HOME/.cache/huggingface
export TMPDIR=$VS_HOME/tmp
export GRADIO_TEMP_DIR=$VS_HOME/tmp
export PYTHONUNBUFFERED=1

# Double-fork to fully detach
(
  setsid bash -c "
    exec python -m uvicorn engine.main:app \
      --host 0.0.0.0 \
      --port ${VS_PORT:-8000} \
      --log-level info \
      > $VS_HOME/engine.log 2>&1
  " </dev/null >/dev/null 2>&1 &
)

sleep 3
if pgrep -f "uvicorn engine.main" > /dev/null; then
  echo "Engine started. Log: $VS_HOME/engine.log"
  curl -s http://127.0.0.1:${VS_PORT:-8000}/health || echo "(not yet responding)"
else
  echo "ERROR: engine failed to start"
  tail -20 $VS_HOME/engine.log
  exit 1
fi
