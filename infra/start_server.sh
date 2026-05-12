#!/bin/bash
# start_server.sh — launch the main vibevoice server with the canonical
# environment variables. Idempotent: safely kills any previous instance
# before starting a new one.
#
# Used by:
#   - bootstrap_new_pod.sh (manual first launch)
#   - GitHub Actions deploy-pod.yml (automated relaunch after scp)
#   - the operator doing a manual restart
#
# Make sure bootstrap_new_pod.sh has been run at least once (so the
# models + Python deps are present).
#
# Env overrides — set these BEFORE calling, e.g.
#   MV_PORT=9080 bash start_server.sh
#
# Defaults are aligned with what the GitHub Actions workflow uses, so
# behaviour is identical whether you launch manually or via CI.

set -euo pipefail

PORT="${MV_PORT:-8080}"
LOG="${SERVER_LOG:-/workspace/server_v5.log}"
SERVER_PY="${SERVER_PY:-/workspace/server_v5_api.py}"
[ -f "${SERVER_PY}" ] || SERVER_PY=/workspace/x/vibevoice_streaming/server_v5_api.py

# 1. Kill any process listening on the port. We use lsof here instead
#    of pgrep -f, because pgrep -f matches THIS shell script's argv too
#    (server_v5_api.py appears in our command line) — using lsof
#    targets only what's actually bound to the port.
PIDS=$(lsof -ti :"${PORT}" 2>/dev/null || true)
if [ -n "${PIDS}" ]; then
    echo "killing existing PID(s) on :${PORT}: ${PIDS}"
    kill -9 ${PIDS} 2>/dev/null || true
    sleep 4
fi

# 2. Truncate the log so the next session is easy to read.
: > "${LOG}"

# 3. Launch fully detached in a new session so the server survives
#    SSH disconnect / shell exit. setsid + disown is the cleanest
#    incantation for "background, immune to SIGHUP, no zombie".
setsid bash -c "cd /workspace && exec env \
    HF_HOME='${HF_HOME:-/workspace/hf-cache}' \
    MV_MODEL='${MV_MODEL:-aoi-ot/VibeVoice-Large}' \
    MV_ASR_MODEL='${MV_ASR_MODEL:-microsoft/VibeVoice-ASR}' \
    MV_LLM_MODEL='${MV_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}' \
    MV_GEMMA4_URL='${MV_GEMMA4_URL:-http://127.0.0.1:8082}' \
    MV_OPENVOICE_URL='${MV_OPENVOICE_URL:-http://127.0.0.1:8083}' \
    MV_DIFF_STEPS='${MV_DIFF_STEPS:-15}' \
    MV_PORT='${PORT}' \
    MV_API_KEY='${MV_API_KEY:-mostashar-dev-key}' \
    MV_VOICES_DIR='${MV_VOICES_DIR:-/workspace/refs/voices}' \
    MV_WHISPER_MODEL='${MV_WHISPER_MODEL:-openai/whisper-large-v3}' \
    MV_WHISPER_LANGUAGE='${MV_WHISPER_LANGUAGE:-ar}' \
    MV_LLM_BACKEND='${MV_LLM_BACKEND:-gemma}' \
    MV_OPENAI_KEY='${MV_OPENAI_KEY:-}' \
    python '${SERVER_PY}' > '${LOG}' 2>&1" </dev/null &
disown
sleep 3

NEW_PID=$(lsof -ti :"${PORT}" 2>/dev/null || true)
echo "started — listening on :${PORT}, PID=${NEW_PID:-(starting)}"
echo "log: ${LOG}"
echo
echo "Wait for ready:"
echo "  for i in \$(seq 1 18); do sleep 10; \\"
echo "    curl -fsS http://127.0.0.1:${PORT}/v1/health 2>/dev/null && break; \\"
echo "  done"
