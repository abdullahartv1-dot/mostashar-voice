#!/bin/bash
# start_gemma4.sh — launch the Gemma 4 sidecar.
#
# Gemma 4 lives in its own Python venv because it pins
# transformers >= 5.x for its multimodal class, while the main server
# is on transformers 4.51 (required by the patched VibeVoice fork).
# Mixing them in one venv would force a downgrade somewhere.
#
# This script assumes /workspace/gemma4-venv was created during
# bootstrap. If it's missing, see infra/bootstrap_new_pod.sh which
# can recreate it.

set -euo pipefail

VENV="/workspace/gemma4-venv"
PORT="${GEMMA4_PORT:-8082}"   # matches MV_GEMMA4_URL in start_server.sh
SERVER_PY="${GEMMA4_SERVER_PY:-/workspace/gemma4_server.py}"
[ -f "${SERVER_PY}" ] || SERVER_PY=/workspace/x/vibevoice_streaming/gemma4_server.py
LOG="${GEMMA4_LOG:-/workspace/gemma4.log}"

# Sanity-check the venv exists.
if [ ! -x "${VENV}/bin/python" ]; then
    echo "ERROR: ${VENV} not found. Run bootstrap_new_pod.sh first to create it."
    exit 1
fi

# Kill old instance — use lsof against the port (NOT pgrep -f, see
# start_server.sh for why).
PIDS=$(lsof -ti :"${PORT}" 2>/dev/null || true)
if [ -n "${PIDS}" ]; then
    echo "killing existing Gemma PID(s) on :${PORT}: ${PIDS}"
    kill -9 ${PIDS} 2>/dev/null || true
    sleep 3
fi

: > "${LOG}"

setsid bash -c "exec env \
    HF_HOME='${HF_HOME:-/workspace/hf-cache}' \
    GEMMA4_PORT='${PORT}' \
    GEMMA4_MODEL='${GEMMA4_MODEL:-google/gemma-4-E4B-it}' \
    '${VENV}/bin/python' '${SERVER_PY}' > '${LOG}' 2>&1" </dev/null &
disown
sleep 3

NEW_PID=$(lsof -ti :"${PORT}" 2>/dev/null || true)
echo "Gemma 4 sidecar started on :${PORT}, PID=${NEW_PID:-(loading model)}"
echo "log: ${LOG} (model warmup ~60 s)"
