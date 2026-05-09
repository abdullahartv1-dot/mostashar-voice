#!/bin/bash
# start_openvoice.sh — one-shot bootstrap + run for the OpenVoice v2 sidecar.
#
# Idempotent: safe to re-run.
#   - Creates the venv if it doesn't exist
#   - Installs/updates deps if requirements.txt is newer than venv stamp
#   - Downloads checkpoints if missing
#   - Restarts the sidecar process
#
# Usage:
#   bash /workspace/x/vibevoice_streaming/start_openvoice.sh
#
# Logs:
#   /tmp/openvoice.log     (stdout + stderr)
#   /tmp/openvoice.pid     (running PID, written on each start)

set -euo pipefail

VENV_DIR="${OPENVOICE_VENV:-/workspace/openvoice-venv}"
CKPT_DIR="${OPENVOICE_CKPT_DIR:-/workspace/openvoice/checkpoints_v2}"
DONORS_DIR="${OPENVOICE_DONORS_DIR:-/workspace/refs/voices}"
DEFAULT_DONOR="${OPENVOICE_DEFAULT_DONOR:-hamed_saudi}"
PORT="${OPENVOICE_PORT:-8083}"
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/openvoice_server.py"

echo "=== [1/5] kill any previous instance ==="
if [ -f /tmp/openvoice.pid ]; then
    OLD_PID="$(cat /tmp/openvoice.pid 2>/dev/null || true)"
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "killing pid $OLD_PID"
        kill -TERM "$OLD_PID" || true
        sleep 1
        kill -KILL "$OLD_PID" 2>/dev/null || true
    fi
    rm -f /tmp/openvoice.pid
fi
# Belt-and-braces: stale processes without a pid file
pkill -f openvoice_server.py 2>/dev/null || true
sleep 1

echo
echo "=== [2/5] venv ($VENV_DIR) ==="
if [ ! -d "$VENV_DIR" ]; then
    echo "creating venv…"
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python --version

echo
echo "=== [3/5] dependencies ==="
STAMP="$VENV_DIR/.deps-stamp"
# Re-install if stamp missing or older than this script.
if [ ! -f "$STAMP" ] || [ "$SCRIPT" -nt "$STAMP" ]; then
    echo "(re)installing deps…"
    pip install --quiet --upgrade pip wheel
    pip install --quiet \
        'torch>=2.0' 'torchaudio>=2.0' \
        librosa soundfile numpy \
        fastapi 'uvicorn[standard]' httpx \
        pydantic
    pip install --quiet git+https://github.com/myshell-ai/OpenVoice@main || {
        echo "WARN: OpenVoice pip install failed; you may need to install manually"
        echo "      pip install git+https://github.com/myshell-ai/OpenVoice@main"
    }
    touch "$STAMP"
else
    echo "deps up-to-date (skipped)"
fi

echo
echo "=== [4/5] checkpoints ($CKPT_DIR) ==="
if [ ! -f "$CKPT_DIR/converter/checkpoint.pth" ]; then
    echo "downloading OpenVoice v2 checkpoints…"
    mkdir -p "$CKPT_DIR"
    cd "$CKPT_DIR"
    if command -v wget >/dev/null; then
        wget -q https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip
    else
        curl -sSL -o checkpoints_v2_0417.zip \
            https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip
    fi
    unzip -q checkpoints_v2_0417.zip
    rm -f checkpoints_v2_0417.zip
    cd - >/dev/null
else
    echo "checkpoints already present"
fi
ls -la "$CKPT_DIR/converter/" | head -5

echo
echo "=== [5/5] start sidecar (port $PORT) ==="
export OPENVOICE_PORT="$PORT"
export OPENVOICE_CKPT_DIR="$CKPT_DIR"
export OPENVOICE_DONORS_DIR="$DONORS_DIR"
export OPENVOICE_DEFAULT_DONOR="$DEFAULT_DONOR"
nohup python "$SCRIPT" > /tmp/openvoice.log 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > /tmp/openvoice.pid
echo "started pid=$NEW_PID, logs=/tmp/openvoice.log"

echo
echo "waiting for /health to come up…"
for i in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "✓ sidecar is up"
        curl -s "http://127.0.0.1:$PORT/health" | python -m json.tool || true
        exit 0
    fi
    sleep 2
done
echo "✗ sidecar did not become ready in 60s — check /tmp/openvoice.log"
tail -30 /tmp/openvoice.log
exit 1
