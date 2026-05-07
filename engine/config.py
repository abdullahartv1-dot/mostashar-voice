"""Engine configuration — single source of truth for paths and constants."""
import os
from pathlib import Path

# Base paths
VS_HOME = Path(os.environ.get("VS_HOME", "/workspace/voice-studio-v2"))
MODELS_DIR = VS_HOME / "models"
JOBS_DIR = VS_HOME / "jobs"
TMP_DIR = VS_HOME / "tmp"

# GPU economics
GPU_RATE_PER_HOUR_USD = 0.27  # RTX A5000 on RunPod
VRAM_TOTAL_GB = 24
VRAM_BUDGET_PCT = 0.80  # leave 20% headroom

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
API_KEY = os.environ.get("VS_API_KEY", "dev-key-change-me")

# Limits (defense against runaway cost)
MAX_AUDIO_DURATION_SEC = 6 * 3600  # 6 hours
MAX_TEXT_CHARS = 2000
INFERENCE_TIMEOUT_SEC = 600  # 10 minutes

# Tool registry — tool name → (stage, vram_gb_estimate)
TOOLS = {
    # STT
    "whisper-large-v3": ("stt", 3),
    "whisper-turbo": ("stt", 1.5),
    "vibevoice-asr": ("stt", 18),  # Qwen 7B inside
    "nemo-canary": ("stt", 4),
    # Diarization
    "pyannote-3.1": ("diar", 1),
    "ecapa-tdnn": ("diar", 0.5),
    # TTS
    "vibevoice-large": ("tts", 8),
    "vibevoice-1.5b": ("tts", 3),
    "f5-tts": ("tts", 2),
    "xtts-v2": ("tts", 2),
    "fish-speech-1.5": ("tts", 4),
}
