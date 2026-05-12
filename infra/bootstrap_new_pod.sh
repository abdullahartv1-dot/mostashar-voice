#!/bin/bash
# bootstrap_new_pod.sh — one-shot setup for a fresh RunPod pod.
#
# Run this ONCE on a freshly-rented pod (or after wiping /workspace).
# The script is idempotent: re-running it re-syncs deps + models
# without re-downloading what's already in the cache.
#
# Total time on a healthy pod with fast HF mirror: ~10-15 minutes.
# Most of that is downloading the ~46 GB of model weights.
#
# Usage:
#
#   ssh root@<pod>
#   curl -fsSL https://raw.githubusercontent.com/abdullahartv1-dot/mostashar-voice/feat/voice-studio-v2/infra/bootstrap_new_pod.sh \
#       | bash
#   # ...wait...
#   bash /workspace/x/infra/start_server.sh
#
# Or with no curl:
#
#   git clone https://github.com/abdullahartv1-dot/mostashar-voice.git /workspace/x
#   bash /workspace/x/infra/bootstrap_new_pod.sh
#
# Environment variables (override defaults):
#   REPO_URL    git remote (default: the official repo)
#   REPO_BRANCH branch to check out (default: feat/voice-studio-v2)
#   HF_HOME     model cache root (default: /workspace/hf-cache)

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/abdullahartv1-dot/mostashar-voice.git}"
REPO_BRANCH="${REPO_BRANCH:-feat/voice-studio-v2}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
WORKDIR="/workspace"
REPO_DIR="${WORKDIR}/x"
VOICES_DIR="${WORKDIR}/refs/voices"

step() { printf "\n\033[1;36m=== %s ===\033[0m\n" "$1"; }

step "1/6 — system info"
uname -a
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || echo "(no GPU)"
df -h /workspace | tail -1

step "2/6 — clone or update repo"
if [ -d "${REPO_DIR}/.git" ]; then
    echo "repo exists at ${REPO_DIR}, pulling latest"
    git -C "${REPO_DIR}" fetch origin
    git -C "${REPO_DIR}" checkout "${REPO_BRANCH}"
    git -C "${REPO_DIR}" reset --hard "origin/${REPO_BRANCH}"
else
    git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${REPO_DIR}"
fi
git -C "${REPO_DIR}" log --oneline -3

step "3/6 — Python dependencies (main server)"
pip install --quiet --upgrade pip wheel
pip install --quiet \
    'torch>=2.4' 'torchaudio>=2.4' \
    'transformers>=4.51,<5.0' 'accelerate>=0.30' \
    'fastapi>=0.110' 'uvicorn[standard]>=0.27' \
    'pydantic>=2.0' \
    'numpy<2' 'soundfile>=0.12' 'librosa>=0.10' \
    'httpx>=0.25'
# flash-attention is built against the installed torch; pin to a
# version known to work with our 8B model on H100. Set
# SKIP_FLASH_ATTN=1 to skip if the build is failing on this pod.
if [ "${SKIP_FLASH_ATTN:-0}" != "1" ]; then
    pip install --quiet flash-attn==2.6.3 --no-build-isolation 2>&1 | tail -3 || \
        echo "WARN: flash-attn install failed. Server can run without it but will be slower."
fi

step "4/6 — VibeVoice community fork (needed for the model class)"
# Note: the original myshell-ai/VibeVoice-Community repo was deleted in
# early 2026. The community fork now lives at vibevoice-community/VibeVoice
# (same Python package layout — drop-in replacement). We also disable
# git's interactive auth prompt so bootstrap doesn't hang on a non-TTY
# pod with no GitHub credentials.
if [ ! -d /workspace/vv-community ]; then
    GIT_TERMINAL_PROMPT=0 git clone https://github.com/vibevoice-community/VibeVoice.git /workspace/vv-community
fi
cd /workspace/vv-community && pip install --quiet -e . && cd "${WORKDIR}"

step "5/6 — pre-fetch model weights to ${HF_HOME}"
mkdir -p "${HF_HOME}"
python3 - <<'PYEOF'
import os, time
os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
from huggingface_hub import snapshot_download

# (repo_id, allow_patterns) — None means "everything".
# We use allow_patterns on Whisper to avoid the multi-format weight
# duplication (PyTorch + TF + Flax + ONNX = 4× the size).
MODELS = [
    ("aoi-ot/VibeVoice-Large",       None),
    ("microsoft/VibeVoice-ASR",      None),
    ("Qwen/Qwen2.5-7B-Instruct",     None),
    ("openai/whisper-large-v3",      ["*.json", "*.txt", "model.safetensors", "tokenizer.json"]),
    # Gemma 4 lives in its own sidecar venv — see start_gemma4.sh.
]
for repo, patterns in MODELS:
    t0 = time.time()
    print(f"  fetching {repo} ...", flush=True)
    p = snapshot_download(repo, cache_dir="/workspace/hf-cache/hub",
                          allow_patterns=patterns)
    print(f"    -> {p}  ({time.time()-t0:.0f}s)")
print("model fetch done")
PYEOF

step "6/6 — voice references"
mkdir -p "${VOICES_DIR}"
if [ -z "$(ls -A "${VOICES_DIR}" 2>/dev/null)" ]; then
    echo "No voices found. To migrate the library from another pod:"
    echo "  scp -r OLD_POD:/workspace/refs/voices/ ${VOICES_DIR}/"
    echo "Then rerun: bash ${REPO_DIR}/infra/start_server.sh"
else
    echo "$(ls "${VOICES_DIR}" | wc -l) voice files already present"
fi

step "DONE"
echo "Next step:  bash ${REPO_DIR}/infra/start_server.sh"
echo
echo "To set up GitHub Actions auto-deploy from your laptop:"
echo "  1. cat ~/.ssh/id_ed25519        (full private key contents)"
echo "  2. Copy it into a GitHub repo secret named POD_SSH_PRIVATE_KEY"
echo "  3. Add POD_HOST + POD_PORT secrets too (see Connect tab in"
echo "     RunPod console for the direct-TCP host:port)"
