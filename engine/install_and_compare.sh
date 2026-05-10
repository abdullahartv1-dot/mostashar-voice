#!/bin/bash
# Run AFTER full_setup_h100.sh completes. Installs the 2 extra engines and runs comparison.
set -euo pipefail

VS=/workspace/voice-studio-v2

echo "=== [1/4] sync engine code ==="
cd $VS && tar xzf /tmp/engine_v2.tgz --no-same-owner
cp /tmp/SPEAKER_0_sample.wav $VS/jobs/_ref/SPEAKER_0_sample.wav 2>/dev/null || mkdir -p $VS/jobs/_ref && cp /tmp/SPEAKER_0_sample.wav $VS/jobs/_ref/

echo "=== [2/4] install chatterbox-tts (Resemble AI, MIT) ==="
pip install --quiet chatterbox-tts 2>&1 | tail -3
python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; print('chatterbox import OK')" 2>&1 | tail -2

echo "=== [3/4] install higgs-audio (Boson AI, Apache 2.0) ==="
[ -d /workspace/higgs-audio ] || git clone --depth 1 https://github.com/boson-ai/higgs-audio /workspace/higgs-audio
cd /workspace/higgs-audio && pip install --quiet -e . 2>&1 | tail -3
python -c "from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine; print('higgs import OK')" 2>&1 | tail -2

echo "=== [4/4] run comparison ==="
# update reference path in compare script
sed -i "s|/workspace/voice-studio-v2/jobs/950cb7a5/SPEAKER_0_sample.wav|/workspace/voice-studio-v2/jobs/_ref/SPEAKER_0_sample.wav|" $VS/engine/compare_tts_engines.py
cd $VS && python -m engine.compare_tts_engines 2>&1 | tail -100

echo
echo "=== ✅ comparison complete. WAV files in $VS/jobs/tts_compare/ ==="
ls $VS/jobs/tts_compare/
