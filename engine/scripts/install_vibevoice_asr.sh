#!/usr/bin/env bash
# install_vibevoice_asr.sh
# -------------------------------------------------------------------
# Installs Microsoft VibeVoice ASR (microsoft/VibeVoice-ASR-HF).
#
# IMPORTANT — earlier docs / install attempts on this project tried
# to install the GitHub `microsoft/VibeVoice` package and patch it.
# DO NOT do that for the ASR checkpoint:
#   - The GitHub vibevoice package is the original TTS-focused codebase.
#     Its `vibevoice.modular.modeling_vibevoice_asr` uses a DIFFERENT
#     internal architecture / parameter naming than the HF checkpoint
#     `microsoft/VibeVoice-ASR-HF`. Loading the checkpoint via that
#     class produces hundreds of "newly initialized" weights (i.e.
#     random init, NOT loaded from disk) — silently broken.
#   - The official install path, per the HF model card, is via
#     `transformers >= 5.3.0`, which ships native
#     `transformers.models.vibevoice_asr` — every weight in the
#     checkpoint maps cleanly. We verified this end-to-end on CPU
#     (Hetzner i5-13500, 62 GB RAM, no GPU): processor.apply_transcription_request
#     + model.generate ran successfully on synthetic audio and
#     emitted the JSON-style assistant prefix.
#
# Tested in Docker image:
#   pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime  (Python 3.11.9)
#
# Usage:
#   ./install_vibevoice_asr.sh [WORKDIR] [HF_CACHE]
# Defaults:
#   WORKDIR  = /workspace
#   HF_CACHE = $WORKDIR/hf-cache
#
# Disk requirement: ~16 GB for the model snapshot, plus ~2 GB for
# the Python deps. Plan for at least 18-20 GB free in WORKDIR.
# -------------------------------------------------------------------
set -euo pipefail

WORKDIR="${1:-/workspace}"
HF_CACHE="${2:-${WORKDIR}/hf-cache}"

echo "==> WORKDIR=${WORKDIR}"
echo "==> HF_CACHE=${HF_CACHE}"

mkdir -p "${WORKDIR}" "${HF_CACHE}"

# -------------------------------------------------------------------
# 1. OS deps (assumes Debian/Ubuntu base; harmless on others)
# -------------------------------------------------------------------
if command -v apt-get >/dev/null 2>&1; then
    echo "==> Installing OS deps (git/curl/wget)..."
    apt-get update -qq
    apt-get install -y -q git curl wget
fi

# -------------------------------------------------------------------
# 2. Python deps (PINNED to versions we tested end-to-end)
#
#    transformers==5.3.0 is the FIRST release that ships a native
#    `vibevoice_asr` model — earlier versions (e.g. 4.51.x) do NOT
#    have `VibeVoiceAsrForConditionalGeneration` and the GitHub
#    vibevoice package's lookalike class loads the HF checkpoint
#    incorrectly (silent random-init).
#
#    accelerate is required for `device_map="auto"` (used on GPU).
#
#    soundfile / librosa / imageio-ffmpeg are needed by the
#    processor's audio loading path.
# -------------------------------------------------------------------
echo "==> Installing pinned Python deps..."
pip install --no-input \
    "transformers==5.3.0" \
    "accelerate>=0.34.2" \
    "soundfile==0.12.1" \
    "imageio-ffmpeg==0.5.1" \
    "librosa==0.10.2"

# -------------------------------------------------------------------
# 3. Download the model into HF_CACHE
#
#    HF_HUB_DISABLE_XET=1: works around a GIL crash in some Linux
#    builds of huggingface_hub's xet backend during concurrent
#    shard downloads. Falls back to plain HTTPS, which is
#    completely reliable in our tests.
#
#    Downloads are idempotent on huggingface_hub >=0.27 — resuming
#    interrupted/partial shards is automatic, no flag needed.
# -------------------------------------------------------------------
echo "==> Downloading microsoft/VibeVoice-ASR-HF (~16 GB)..."
HF_HOME="${HF_CACHE}" HF_HUB_DISABLE_XET=1 python - <<PYEOF
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    "microsoft/VibeVoice-ASR-HF",
    cache_dir="${HF_CACHE}",
)
total = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, dn, fns in os.walk(p) for f in fns
)
print(f"    snapshot at: {p}")
print(f"    total size:  {total / 1024**3:.2f} GB")
PYEOF

# -------------------------------------------------------------------
# 4. Verify imports + model load (CPU-only smoke test)
#
#    We deliberately load on CPU in bfloat16 — works on any host
#    with ~16 GB free RAM, even without a GPU. On GPU just pass
#    `device_map="auto"`.
# -------------------------------------------------------------------
echo "==> Verifying imports + model load on CPU (bfloat16)..."
HF_HOME="${HF_CACHE}" python - <<PYEOF
import torch
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

processor = AutoProcessor.from_pretrained(
    "microsoft/VibeVoice-ASR-HF", cache_dir="${HF_CACHE}",
)
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
    "microsoft/VibeVoice-ASR-HF",
    cache_dir="${HF_CACHE}",
    torch_dtype=torch.bfloat16,
)
n = sum(p.numel() for p in model.parameters())
print(f"    OK — model loaded.")
print(f"    Param count: {n / 1e9:.2f} B  (expect ~8.33 B)")
print(f"    Processor type: {type(processor).__name__}")
print(f"    Model dtype: {next(model.parameters()).dtype}")
PYEOF

# -------------------------------------------------------------------
# 5. Disk usage report
# -------------------------------------------------------------------
echo ""
echo "==> Install complete. Disk usage:"
du -sh "${HF_CACHE}" 2>/dev/null || true
du -sh "$(python -c 'import sys; print(sys.prefix)')" 2>/dev/null || true
echo ""
echo "==> NEXT STEP — run inference. On a GPU machine:"
echo ""
echo "    HF_HOME=${HF_CACHE} python -c '"
echo "        import torch"
echo "        from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration"
echo "        proc = AutoProcessor.from_pretrained(\"microsoft/VibeVoice-ASR-HF\")"
echo "        model = VibeVoiceAsrForConditionalGeneration.from_pretrained("
echo "            \"microsoft/VibeVoice-ASR-HF\","
echo "            torch_dtype=torch.bfloat16,"
echo "            device_map=\"auto\","
echo "        )"
echo "        inputs = proc.apply_transcription_request(audio=\"path/to/audio.wav\")"
echo "        inputs = inputs.to(model.device, model.dtype)"
echo "        out = model.generate(**inputs)"
echo "        gen = out[:, inputs[\"input_ids\"].shape[1]:]"
echo "        print(proc.decode(gen, return_format=\"parsed\")[0])"
echo "    '"
