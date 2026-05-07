#!/bin/bash
# Launch app.py as a true daemon
cd /workspace/voice-studio/vibe-voice-custom-voices

# Double-fork via setsid + nohup
(
  setsid bash -c "
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/comfy_mock \
    HF_HOME=/workspace/.cache/huggingface \
    TMPDIR=/workspace/tmp \
    GRADIO_TEMP_DIR=/workspace/tmp \
    python -u app.py
  " </dev/null >/workspace/voice-studio/vv_daemon.log 2>&1 &
)
echo "Started"
sleep 1
ps aux | grep -E "python.*app.py" | grep -v grep | head -3
