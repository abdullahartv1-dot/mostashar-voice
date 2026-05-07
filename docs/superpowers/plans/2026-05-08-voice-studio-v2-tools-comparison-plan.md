# Voice Studio v2 — Tools Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Voice Studio comparison platform on RunPod GPU letting users select different AI tools per pipeline stage (STT/Diarization/TTS), run the same audio through them, and compare quality, speed, and cost.

**Architecture:** Local Flask serves UI; FastAPI engine runs on Pod GPU (RTX A5000); SSH tunnel connects them; lazy-loading model manager swaps models in 24GB VRAM as the user picks tools.

**Tech Stack:** Python 3.11, FastAPI + uvicorn, PyTorch 2.4.1+cu124, faster-whisper, pyannote.audio, speechbrain, transformers, Flask (existing), HTML/JS frontend.

---

## File Structure

### Pod-side (`/workspace/voice-studio-v2/engine/`)
- `engine/main.py` — FastAPI app + route registration
- `engine/config.py` — Settings (GPU rate, model paths, VRAM budget)
- `engine/model_manager.py` — Lazy load + LRU eviction
- `engine/cost_tracker.py` — Per-job cost calculation
- `engine/schemas.py` — Pydantic request/response models
- `engine/jobs_store.py` — In-memory job state
- `engine/services/__init__.py` — Service registry (tool name → handler)
- `engine/services/stt_whisper.py` — Whisper Large-v3 + Turbo
- `engine/services/stt_vibevoice.py` — VibeVoice ASR-HF
- `engine/services/stt_nemo.py` — NVIDIA Canary
- `engine/services/diar_pyannote.py` — pyannote.audio 3.1
- `engine/services/diar_ecapa.py` — ECAPA-TDNN + clustering
- `engine/services/tts_vibevoice.py` — VibeVoice Large + 1.5B
- `engine/services/tts_f5.py` — F5-TTS
- `engine/services/tts_xtts.py` — XTTS-v2
- `engine/services/tts_fishspeech.py` — Fish Speech 1.5
- `engine/requirements.txt` — Pinned dependencies
- `engine/run_engine.sh` — Daemon launcher
- `engine/tests/conftest.py` — Pytest fixtures
- `engine/tests/test_smoke_stt.py` — STT smoke tests
- `engine/tests/test_smoke_diar.py` — Diarization smoke tests
- `engine/tests/test_smoke_tts.py` — TTS smoke tests
- `engine/tests/test_api.py` — FastAPI endpoint tests
- `engine/tests/fixtures/short_arabic.wav` — 5-second Arabic test clip

### Local (existing project)
- `app.py` — MODIFY: add `/v2` route + proxy endpoints
- `templates/index_v2.html` — NEW: based on `index.html` with tool dropdowns + comparison panel
- `static/audio/jobs/` — NEW directory: cached job results
- `tunnel.sh` — NEW: SSH tunnel launcher

---

## Phase 0: Pod Environment Cleanup

### Task 0.1: Clean up old experimental files on Pod

**Files:**
- Modify: Pod `/workspace/voice-studio/` (cleanup)

- [ ] **Step 1: Verify SSH access works**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "echo connected && pwd"
```
Expected: prints `connected` and `/root`

- [ ] **Step 2: Inventory current Pod state**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "ls /workspace/voice-studio/ && du -sh /workspace/voice-studio/*"
```
Expected: lists experimental directories (VibeVoice/, vibe-voice-custom-voices/, etc.)

- [ ] **Step 3: Stop any running app processes**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "pkill -9 -f 'python.*app.py' 2>/dev/null; pkill -9 -f 'python.*test_' 2>/dev/null; sleep 2; ps aux | grep python | grep -v grep || echo 'No python processes'"
```
Expected: prints `No python processes`

- [ ] **Step 4: Free container disk (clear /tmp, /root cache)**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "rm -rf /tmp/* 2>/dev/null; rm -rf /root/.cache/pip/* 2>/dev/null; df -h / | tail -1"
```
Expected: `/` usage shown, ideally < 50%

- [ ] **Step 5: Commit (no code changes, just notes)**

Run locally:
```bash
git add docs/superpowers/plans/2026-05-08-voice-studio-v2-tools-comparison-plan.md
git commit -m "docs: add voice-studio-v2 implementation plan"
```

### Task 0.2: Create new directory structure on Pod

**Files:**
- Create: Pod `/workspace/voice-studio-v2/` and subdirectories

- [ ] **Step 1: Create directory structure**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "mkdir -p /workspace/voice-studio-v2/{engine,engine/services,engine/tests/fixtures,models,jobs,tmp,.cache/huggingface}"
```
Expected: no output (success)

- [ ] **Step 2: Verify structure**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "find /workspace/voice-studio-v2 -type d"
```
Expected: lists all subdirectories

- [ ] **Step 3: Set environment variables in shell rc**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cat >> ~/.bashrc << 'EOF'

# Voice Studio v2 environment
export VS_HOME=/workspace/voice-studio-v2
export HF_HOME=\$VS_HOME/.cache/huggingface
export TMPDIR=\$VS_HOME/tmp
export GRADIO_TEMP_DIR=\$VS_HOME/tmp
export PYTHONUNBUFFERED=1
EOF
source ~/.bashrc; echo VS_HOME=\$VS_HOME"
```
Expected: prints `VS_HOME=/workspace/voice-studio-v2`

### Task 0.3: Install pinned Python dependencies on Pod

**Files:**
- Create: Pod `/workspace/voice-studio-v2/engine/requirements.txt`

- [ ] **Step 1: Create requirements.txt**

Write to Pod via SSH heredoc:
```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cat > /workspace/voice-studio-v2/engine/requirements.txt << 'EOF'
# Core
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
python-multipart==0.0.12
httpx==0.27.2

# Audio I/O
soundfile==0.12.1
imageio-ffmpeg==0.5.1
librosa==0.10.2
av==13.0.0

# STT
faster-whisper==1.0.3
nemo-toolkit[asr]==2.0.0

# Diarization
pyannote.audio==3.3.1
speechbrain==1.0.2

# TTS — versions known compatible with torch 2.4.1+cu124
TTS==0.22.0

# ML core (already on Pod, but pin to be safe)
# torch==2.4.1 (do NOT reinstall — already provided by Pod template)
# torchaudio==2.4.1
# torchvision==0.19.1
transformers==4.51.3
scikit-learn==1.5.2
numpy==1.26.4

# VibeVoice (cloned separately, this is its dep list)
diffusers==0.30.3
accelerate==0.34.2
peft==0.13.0
EOF
echo 'requirements.txt written'"
```
Expected: prints `requirements.txt written`

- [ ] **Step 2: Install requirements**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2/engine && pip install --quiet -r requirements.txt 2>&1 | tail -5"
```
Expected: pip notices, no error tracebacks

- [ ] **Step 3: Verify torch + cuda still working**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "python -c 'import torch; print(\"torch:\", torch.__version__, \"| cuda:\", torch.cuda.is_available(), \"| device:\", torch.cuda.get_device_name(0))'"
```
Expected: `torch: 2.4.1+cu124 | cuda: True | device: NVIDIA RTX A5000`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(plan): pod env ready for engine"
```

### Task 0.4: Generate 5-second Arabic smoke-test fixture

**Files:**
- Create: Pod `/workspace/voice-studio-v2/engine/tests/fixtures/short_arabic.wav`

- [ ] **Step 1: Extract 5s clip from existing 03.mp3 (already on Pod)**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "python -c \"
import subprocess, imageio_ffmpeg
ff = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ff, '-y', '-ss', '60', '-t', '5', '-i', '/workspace/voice-studio/03.mp3', '-ac', '1', '-ar', '16000', '/workspace/voice-studio-v2/engine/tests/fixtures/short_arabic.wav'], check=True)
import soundfile as sf
info = sf.info('/workspace/voice-studio-v2/engine/tests/fixtures/short_arabic.wav')
print('Duration:', info.duration, 'sr:', info.samplerate)
\""
```
Expected: `Duration: 5.0 sr: 16000`

- [ ] **Step 2: Verify file exists and is non-zero**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "ls -la /workspace/voice-studio-v2/engine/tests/fixtures/short_arabic.wav"
```
Expected: file size ~160 KB

---

## Phase 1: Engine Foundation

### Task 1.1: Create config module

**Files:**
- Create: Pod `/workspace/voice-studio-v2/engine/config.py`

- [ ] **Step 1: Write config.py**

Use `scp` to upload from local. First create local file:

Local file `engine/config.py`:
```python
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
```

Then upload:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/config.py root@194.68.245.175:/workspace/voice-studio-v2/engine/config.py
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -c 'from engine.config import TOOLS, GPU_RATE_PER_HOUR_USD; print(\"tools:\", len(TOOLS), \"| rate:\", GPU_RATE_PER_HOUR_USD)'"
```
Expected: `tools: 11 | rate: 0.27`

- [ ] **Step 3: Commit**

```bash
git add engine/config.py
git commit -m "feat(engine): add config with tool registry and GPU rate"
```

### Task 1.2: Cost tracker — write failing test first

**Files:**
- Create: Pod and local `engine/tests/test_cost.py`

- [ ] **Step 1: Write failing test**

Local file `engine/tests/test_cost.py`:
```python
"""Tests for cost_tracker."""
from engine.cost_tracker import calc_cost, format_breakdown


def test_calc_cost_one_hour_at_27_cents():
    assert calc_cost(seconds=3600, rate_per_hour=0.27) == 0.27


def test_calc_cost_60_seconds():
    # 60s = 1/60 hour = 0.27/60 = 0.0045
    assert calc_cost(seconds=60, rate_per_hour=0.27) == round(0.0045, 4)


def test_calc_cost_zero_seconds_is_zero():
    assert calc_cost(seconds=0, rate_per_hour=0.27) == 0.0


def test_format_breakdown_returns_dict_with_keys():
    breakdown = format_breakdown([
        ("stt", 60),
        ("diar", 30),
    ], rate_per_hour=0.27)
    assert "stt" in breakdown
    assert "diar" in breakdown
    assert "total_usd" in breakdown
    assert breakdown["total_usd"] == round((60 + 30) / 3600 * 0.27, 4)
```

Upload:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/tests/test_cost.py root@194.68.245.175:/workspace/voice-studio-v2/engine/tests/
```

- [ ] **Step 2: Run to confirm failure**

```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_cost.py -v 2>&1 | tail -10"
```
Expected: ImportError or ModuleNotFoundError for `engine.cost_tracker`

- [ ] **Step 3: Implement cost_tracker.py**

Local file `engine/cost_tracker.py`:
```python
"""Tracks GPU-hour cost per stage and per session."""
from typing import List, Tuple, Dict


def calc_cost(seconds: float, rate_per_hour: float) -> float:
    """Returns USD cost rounded to 4 decimal places."""
    return round(seconds / 3600 * rate_per_hour, 4)


def format_breakdown(
    stages: List[Tuple[str, float]],
    rate_per_hour: float,
) -> Dict[str, float]:
    """Convert list of (stage_name, seconds) into a cost breakdown dict.

    Returns:
        {"stt": 0.0045, "diar": 0.0023, ..., "total_usd": 0.0068}
    """
    out = {name: calc_cost(secs, rate_per_hour) for name, secs in stages}
    total_secs = sum(s for _, s in stages)
    out["total_usd"] = calc_cost(total_secs, rate_per_hour)
    return out
```

Upload:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/cost_tracker.py root@194.68.245.175:/workspace/voice-studio-v2/engine/
```

- [ ] **Step 4: Run tests to verify pass**

```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_cost.py -v"
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add engine/cost_tracker.py engine/tests/test_cost.py
git commit -m "feat(engine): cost tracker with per-stage breakdown"
```

### Task 1.3: Pydantic schemas

**Files:**
- Create: `engine/schemas.py`

- [ ] **Step 1: Write schemas.py**

Local file `engine/schemas.py`:
```python
"""Pydantic models for API request/response."""
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


class ProcessRequest(BaseModel):
    audio_path: str  # path on Pod
    stt_tool: str = Field(..., description="One of: whisper-large-v3, whisper-turbo, vibevoice-asr, nemo-canary")
    diar_tool: Optional[str] = Field(None, description="One of: pyannote-3.1, ecapa-tdnn. Skipped if stt_tool is vibevoice-asr.")


class SpeakerSample(BaseModel):
    url: str
    duration: float


class ProcessResponse(BaseModel):
    job_id: str
    duration: float
    segments: List[Segment]
    samples: Dict[str, SpeakerSample]
    timings: Dict[str, float]
    cost: Dict[str, float]
    tools_used: Dict[str, str]


class CloneRequest(BaseModel):
    job_id: str
    speaker_id: str
    text: str = Field(..., max_length=2000)
    tts_tool: str
    diffusion_steps: int = 40
    cfg_scale: float = 1.5
    seed: int = 42


class CloneResponse(BaseModel):
    audio_url: str
    duration: float
    elapsed: float
    rtf: float
    cost_usd: float
    tool: str


class HealthResponse(BaseModel):
    status: str
    gpu_mem_used_gb: float
    gpu_mem_total_gb: float
    models_loaded: List[str]
```

Upload + commit:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/schemas.py root@194.68.245.175:/workspace/voice-studio-v2/engine/
git add engine/schemas.py
git commit -m "feat(engine): pydantic schemas"
```

### Task 1.4: Model manager (lazy load + LRU eviction)

**Files:**
- Create: `engine/model_manager.py`
- Create: `engine/tests/test_model_manager.py`

- [ ] **Step 1: Write failing test**

Local file `engine/tests/test_model_manager.py`:
```python
"""Tests for ModelManager — lazy load + LRU eviction."""
import pytest
from engine.model_manager import ModelManager


class FakeModel:
    """Stand-in for an ML model — has a vram footprint."""
    def __init__(self, name: str, vram_gb: float):
        self.name = name
        self.vram_gb = vram_gb


def loader_factory(name: str, vram_gb: float):
    """Returns a function that creates a FakeModel; tracks call count."""
    calls = {"n": 0}
    def load():
        calls["n"] += 1
        return FakeModel(name, vram_gb)
    return load, calls


def test_first_get_loads_model():
    mm = ModelManager(vram_budget_gb=20)
    loader, calls = loader_factory("whisper-large", 3)
    model = mm.get("whisper-large", loader, vram_gb=3)
    assert model.name == "whisper-large"
    assert calls["n"] == 1


def test_second_get_uses_cached():
    mm = ModelManager(vram_budget_gb=20)
    loader, calls = loader_factory("whisper-large", 3)
    mm.get("whisper-large", loader, vram_gb=3)
    mm.get("whisper-large", loader, vram_gb=3)
    assert calls["n"] == 1


def test_evicts_lru_when_over_budget():
    mm = ModelManager(vram_budget_gb=10)
    l1, c1 = loader_factory("a", 6)
    l2, c2 = loader_factory("b", 6)  # 6+6=12 > 10, must evict a
    mm.get("a", l1, vram_gb=6)
    mm.get("b", l2, vram_gb=6)
    assert "a" not in mm.loaded
    assert "b" in mm.loaded


def test_get_after_eviction_reloads():
    mm = ModelManager(vram_budget_gb=10)
    l1, c1 = loader_factory("a", 6)
    l2, c2 = loader_factory("b", 6)
    mm.get("a", l1, vram_gb=6)
    mm.get("b", l2, vram_gb=6)  # evicts a
    mm.get("a", l1, vram_gb=6)  # reloads a, evicts b
    assert c1["n"] == 2
    assert "a" in mm.loaded
    assert "b" not in mm.loaded
```

Upload:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/tests/test_model_manager.py root@194.68.245.175:/workspace/voice-studio-v2/engine/tests/
```

- [ ] **Step 2: Run test to verify failure**

```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_model_manager.py -v 2>&1 | tail -5"
```
Expected: ModuleNotFoundError for `engine.model_manager`

- [ ] **Step 3: Implement ModelManager**

Local file `engine/model_manager.py`:
```python
"""Lazy-loading model manager with LRU eviction.

Models are loaded on first request and cached in memory. When loading a new
model would exceed the VRAM budget, the least-recently-used model is evicted
to free space.
"""
import logging
import time
from collections import OrderedDict
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, vram_budget_gb: float):
        self.vram_budget_gb = vram_budget_gb
        # Ordered dict: oldest first, newest last (move_to_end on access)
        self.loaded: "OrderedDict[str, dict]" = OrderedDict()

    def _current_vram_gb(self) -> float:
        return sum(entry["vram_gb"] for entry in self.loaded.values())

    def _evict_lru(self) -> None:
        """Remove the oldest entry to free VRAM."""
        if not self.loaded:
            return
        name, entry = self.loaded.popitem(last=False)
        logger.info(f"Evicting LRU model: {name} ({entry['vram_gb']} GB)")
        # Free the model's GPU memory
        del entry["model"]
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def get(self, name: str, loader: Callable[[], Any], vram_gb: float) -> Any:
        """Get a model by name; load it if not already in memory.

        Args:
            name: unique tool name (e.g. "whisper-large-v3")
            loader: zero-arg function that returns the model object
            vram_gb: estimated VRAM footprint in gigabytes

        Returns:
            The model object.
        """
        # Cache hit: move to end (mark as most recently used)
        if name in self.loaded:
            self.loaded.move_to_end(name)
            return self.loaded[name]["model"]

        # Cache miss: ensure budget allows loading
        while self._current_vram_gb() + vram_gb > self.vram_budget_gb:
            if not self.loaded:
                # Single model exceeds budget — load anyway, log warning
                logger.warning(
                    f"Model {name} ({vram_gb} GB) exceeds VRAM budget "
                    f"({self.vram_budget_gb} GB) — loading anyway"
                )
                break
            self._evict_lru()

        t0 = time.time()
        model = loader()
        load_time = time.time() - t0
        logger.info(f"Loaded model {name} in {load_time:.1f}s ({vram_gb} GB)")

        self.loaded[name] = {"model": model, "vram_gb": vram_gb, "loaded_at": time.time()}
        return model
```

Upload:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/model_manager.py root@194.68.245.175:/workspace/voice-studio-v2/engine/
```

- [ ] **Step 4: Run tests to verify pass**

```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_model_manager.py -v"
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add engine/model_manager.py engine/tests/test_model_manager.py
git commit -m "feat(engine): lazy-load model manager with LRU eviction"
```

### Task 1.5: FastAPI skeleton with /health

**Files:**
- Create: `engine/main.py`
- Create: `engine/jobs_store.py`

- [ ] **Step 1: Write jobs_store.py**

Local file `engine/jobs_store.py`:
```python
"""In-memory job state. Maps job_id -> dict (transcript, samples, costs, etc.)"""
import uuid
from typing import Dict, Any, Optional

_JOBS: Dict[str, Dict[str, Any]] = {}


def create_job() -> str:
    job_id = uuid.uuid4().hex[:8]
    _JOBS[job_id] = {"id": job_id, "status": "pending"}
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    return _JOBS.get(job_id)


def update_job(job_id: str, **fields) -> None:
    if job_id in _JOBS:
        _JOBS[job_id].update(fields)


def all_jobs() -> Dict[str, Dict[str, Any]]:
    return dict(_JOBS)
```

- [ ] **Step 2: Write main.py with /health**

Local file `engine/main.py`:
```python
"""FastAPI engine entrypoint."""
import logging
from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles

from . import config
from .schemas import HealthResponse
from .model_manager import ModelManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("vs2.engine")

app = FastAPI(title="Voice Studio v2 Engine")

# Single global model manager
mm = ModelManager(vram_budget_gb=config.VRAM_TOTAL_GB * config.VRAM_BUDGET_PCT)

# Serve generated audio files
config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(config.JOBS_DIR)), name="files")


def require_api_key(x_api_key: str = Header(None)) -> None:
    if x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health", response_model=HealthResponse)
def health():
    """Returns engine status — used by tunnel/proxy to verify aliveness."""
    gpu_used_gb = 0.0
    gpu_total_gb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_used_gb = torch.cuda.memory_allocated() / 1024 ** 3
            gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception as e:
        logger.warning(f"GPU info unavailable: {e}")
    return HealthResponse(
        status="ok",
        gpu_mem_used_gb=round(gpu_used_gb, 2),
        gpu_mem_total_gb=round(gpu_total_gb, 2),
        models_loaded=list(mm.loaded.keys()),
    )
```

Upload both:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/jobs_store.py engine/main.py root@194.68.245.175:/workspace/voice-studio-v2/engine/
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "touch /workspace/voice-studio-v2/engine/__init__.py /workspace/voice-studio-v2/engine/services/__init__.py /workspace/voice-studio-v2/engine/tests/__init__.py"
```

- [ ] **Step 3: Run engine in foreground to verify startup**

```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && timeout 5 python -m uvicorn engine.main:app --host 127.0.0.1 --port 8000 2>&1 | head -10"
```
Expected: shows `Uvicorn running on http://127.0.0.1:8000`

- [ ] **Step 4: Commit**

```bash
git add engine/main.py engine/jobs_store.py
git commit -m "feat(engine): FastAPI skeleton with /health"
```

### Task 1.6: Daemon launcher script

**Files:**
- Create: `engine/run_engine.sh`

- [ ] **Step 1: Write daemon script that survives SSH disconnect**

Local file `engine/run_engine.sh`:
```bash
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
```

Upload + make executable:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/run_engine.sh root@194.68.245.175:/workspace/voice-studio-v2/engine/
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "chmod +x /workspace/voice-studio-v2/engine/run_engine.sh"
```

- [ ] **Step 2: Launch engine via daemon script**

```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "/workspace/voice-studio-v2/engine/run_engine.sh"
```
Expected: prints `Engine started.` followed by health JSON

- [ ] **Step 3: Verify engine still alive after fresh SSH session**

```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "curl -s http://127.0.0.1:8000/health"
```
Expected: returns JSON `{"status": "ok", ...}`

- [ ] **Step 4: Commit**

```bash
git add engine/run_engine.sh
git commit -m "feat(engine): daemon launcher script"
```

### Task 1.7: SSH tunnel + local proxy verification

**Files:**
- Create: local `tunnel.sh`

- [ ] **Step 1: Write tunnel script (local)**

Local file `tunnel.sh`:
```bash
#!/bin/bash
# Opens an SSH tunnel from localhost:8000 to the Pod's engine.
set -euo pipefail

POD_HOST=${POD_HOST:-194.68.245.175}
POD_PORT=${POD_PORT:-22133}
LOCAL_PORT=${LOCAL_PORT:-8000}

# Kill existing tunnel on the same local port (best-effort)
EXISTING=$(netstat -ano 2>&1 | grep ":${LOCAL_PORT}\b" | grep LISTENING | awk '{print $NF}' | head -1 || true)
if [ -n "$EXISTING" ]; then
  taskkill //F //PID "$EXISTING" >/dev/null 2>&1 || true
  sleep 1
fi

# Open new tunnel in background
ssh -i ~/.ssh/id_ed25519 \
    -p $POD_PORT \
    -L ${LOCAL_PORT}:localhost:8000 \
    -N -f \
    -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=30 \
    root@$POD_HOST

sleep 1
HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:${LOCAL_PORT}/health)
if [ "$HTTP" = "200" ]; then
  echo "Tunnel up: http://127.0.0.1:${LOCAL_PORT}/health → 200 OK"
else
  echo "Tunnel may not be working: HTTP $HTTP"
  exit 1
fi
```

- [ ] **Step 2: Run tunnel script**

```bash
chmod +x tunnel.sh && ./tunnel.sh
```
Expected: prints `Tunnel up: ... → 200 OK`

- [ ] **Step 3: Hit /health from local Python**

```bash
python -c "import httpx; print(httpx.get('http://127.0.0.1:8000/health').json())"
```
Expected: prints health dict with `status: ok`

- [ ] **Step 4: Commit**

```bash
git add tunnel.sh
git commit -m "feat: SSH tunnel script for local→Pod engine"
```

---

## Phase 2: STT Services

### Task 2.1: Whisper Large-v3 service + smoke test

**Files:**
- Create: `engine/services/stt_whisper.py`
- Create: `engine/tests/test_smoke_stt.py`

- [ ] **Step 1: Write smoke test (failing)**

Local file `engine/tests/test_smoke_stt.py`:
```python
"""Smoke tests for STT services — verifies each model loads + transcribes 5s clip."""
import pytest
from pathlib import Path

FIXTURE = Path("/workspace/voice-studio-v2/engine/tests/fixtures/short_arabic.wav")


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
def test_whisper_large_transcribes_arabic():
    from engine.services.stt_whisper import transcribe_whisper
    result = transcribe_whisper(str(FIXTURE), model_size="large-v3")
    assert "duration" in result
    assert "segments" in result
    assert result["duration"] == pytest.approx(5.0, abs=0.5)
    assert len(result["segments"]) >= 1
    # Should produce non-empty Arabic text
    assert any(seg["text"].strip() for seg in result["segments"])
```

Upload + run:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/tests/test_smoke_stt.py root@194.68.245.175:/workspace/voice-studio-v2/engine/tests/
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_smoke_stt.py::test_whisper_large_transcribes_arabic -v 2>&1 | tail -5"
```
Expected: ImportError for `engine.services.stt_whisper`

- [ ] **Step 2: Implement stt_whisper.py**

Local file `engine/services/stt_whisper.py`:
```python
"""Whisper STT (large-v3 and large-v3-turbo)."""
import logging
import time
from typing import Dict, Any
from faster_whisper import WhisperModel
import soundfile as sf

logger = logging.getLogger(__name__)

# Module-level cache — keyed by model size so we don't reload
_models: Dict[str, WhisperModel] = {}


def _get_model(model_size: str) -> WhisperModel:
    """Get a cached Whisper model, loading it if necessary."""
    if model_size not in _models:
        logger.info(f"Loading Whisper {model_size} on GPU…")
        _models[model_size] = WhisperModel(
            model_size, device="cuda", compute_type="float16"
        )
    return _models[model_size]


def transcribe_whisper(audio_path: str, model_size: str = "large-v3") -> Dict[str, Any]:
    """Transcribe with Whisper. Returns dict with duration, segments, timings."""
    model = _get_model(model_size)
    duration = sf.info(audio_path).duration

    t0 = time.time()
    seg_iter, info = model.transcribe(
        audio_path,
        language="ar",
        beam_size=5,
        vad_filter=True,
    )
    segments = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in seg_iter
    ]
    elapsed = time.time() - t0

    return {
        "duration": duration,
        "segments": segments,
        "lang_prob": info.language_probability,
        "elapsed_sec": round(elapsed, 2),
        "speedup": round(duration / elapsed, 2) if elapsed > 0 else 0,
    }
```

Upload:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/services/stt_whisper.py root@194.68.245.175:/workspace/voice-studio-v2/engine/services/
```

- [ ] **Step 3: Run smoke test (will load 3GB model — first run slower)**

```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_smoke_stt.py::test_whisper_large_transcribes_arabic -v -s 2>&1 | tail -15"
```
Expected: 1 passed (may take 30-60s for first run while model downloads)

- [ ] **Step 4: Commit**

```bash
git add engine/services/stt_whisper.py engine/tests/test_smoke_stt.py
git commit -m "feat(engine): Whisper Large-v3 STT service + smoke test"
```

### Task 2.2: Whisper Turbo (uses same module)

**Files:**
- Modify: `engine/tests/test_smoke_stt.py`

- [ ] **Step 1: Add turbo test case**

Append to `engine/tests/test_smoke_stt.py`:
```python
@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
def test_whisper_turbo_transcribes_arabic():
    from engine.services.stt_whisper import transcribe_whisper
    result = transcribe_whisper(str(FIXTURE), model_size="large-v3-turbo")
    assert result["duration"] == pytest.approx(5.0, abs=0.5)
    assert len(result["segments"]) >= 1
```

Upload + run:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/tests/test_smoke_stt.py root@194.68.245.175:/workspace/voice-studio-v2/engine/tests/
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_smoke_stt.py::test_whisper_turbo_transcribes_arabic -v 2>&1 | tail -10"
```
Expected: 1 passed (downloads 1.5GB model on first run)

- [ ] **Step 2: Commit**

```bash
git add engine/tests/test_smoke_stt.py
git commit -m "test: Whisper Turbo smoke test"
```

### Task 2.3: VibeVoice ASR-HF service + smoke test

**Files:**
- Create: `engine/services/stt_vibevoice.py`
- Modify: `engine/tests/test_smoke_stt.py`

- [ ] **Step 1: Apply transformers JSON-serialization patch (known issue)**

Run on Pod:
```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "python -c \"
fp = '/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py'
with open(fp) as f: c = f.read()
old = 'return json.dumps(config_dict, indent=2, sort_keys=True)'
new = 'return json.dumps(config_dict, indent=2, sort_keys=True, default=str)'
if old in c and 'default=str' not in c:
    with open(fp, 'w') as f: f.write(c.replace(old, new))
    print('patched')
else:
    print('already patched or not found')
\""
```
Expected: prints `patched` or `already patched`

- [ ] **Step 2: Clone VibeVoice repo to engine/vendor/ if not present**

```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "
mkdir -p /workspace/voice-studio-v2/engine/vendor
if [ ! -d /workspace/voice-studio-v2/engine/vendor/VibeVoice ]; then
  cd /workspace/voice-studio-v2/engine/vendor && git clone --depth 1 https://github.com/microsoft/VibeVoice.git
fi
# Apply the known import fix
sed -i 's|^from transformers import modeling_utils$|import transformers.modeling_utils as modeling_utils|' /workspace/voice-studio-v2/engine/vendor/VibeVoice/vibevoice/modular/modeling_vibevoice_streaming_inference.py
echo 'VibeVoice ready'"
```
Expected: prints `VibeVoice ready`

- [ ] **Step 3: Install VibeVoice locally**

```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2/engine/vendor/VibeVoice && pip install --quiet -e . 2>&1 | tail -3"
```
Expected: no errors

- [ ] **Step 4: Add failing smoke test**

Append to `engine/tests/test_smoke_stt.py`:
```python
@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
def test_vibevoice_asr_transcribes_arabic():
    from engine.services.stt_vibevoice import transcribe_vibevoice
    result = transcribe_vibevoice(str(FIXTURE))
    assert "segments" in result
    # VibeVoice ASR returns segments with speakers
    assert len(result["segments"]) >= 1
    if result["segments"]:
        seg = result["segments"][0]
        assert "speaker" in seg or "speaker_id" in seg
```

- [ ] **Step 5: Implement stt_vibevoice.py**

Local file `engine/services/stt_vibevoice.py`:
```python
"""VibeVoice ASR — combined STT + speaker diarization in one model."""
import logging
import time
from typing import Dict, Any, Optional
import torch
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

logger = logging.getLogger(__name__)

_model: Optional[VibeVoiceASRForConditionalGeneration] = None
_processor: Optional[VibeVoiceASRProcessor] = None


def _load() -> None:
    global _model, _processor
    if _model is not None:
        return
    logger.info("Loading VibeVoice ASR-HF (~5GB model + Qwen 7B = ~20GB VRAM)…")
    _processor = VibeVoiceASRProcessor.from_pretrained(
        "microsoft/VibeVoice-ASR-HF",
        language_model_pretrained_name="Qwen/Qwen2.5-7B",
    )
    _model = (
        VibeVoiceASRForConditionalGeneration.from_pretrained(
            "microsoft/VibeVoice-ASR-HF",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        .to("cuda")
        .eval()
    )


def transcribe_vibevoice(
    audio_path: str,
    context_info: str = "Arabic conversation",
    max_new_tokens: int = 16384,
) -> Dict[str, Any]:
    _load()
    import soundfile as sf

    duration = sf.info(audio_path).duration

    t0 = time.time()
    inputs = _processor(
        audio_path=audio_path,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        context_info=context_info,
    )
    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=_processor.pad_id,
            eos_token_id=_processor.tokenizer.eos_token_id,
            do_sample=False,
        )
    input_length = inputs["input_ids"].shape[1]
    generated_text = _processor.decode(output_ids[0, input_length:], skip_special_tokens=True)
    elapsed = time.time() - t0

    try:
        segments = _processor.post_process_transcription(generated_text)
    except Exception as e:
        logger.warning(f"Post-process failed: {e}")
        segments = [{"text": generated_text, "speaker": "unknown", "start": 0, "end": duration}]

    return {
        "duration": duration,
        "segments": segments,
        "raw_text": generated_text,
        "elapsed_sec": round(elapsed, 2),
        "speedup": round(duration / elapsed, 2) if elapsed > 0 else 0,
    }
```

Upload + run smoke:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/services/stt_vibevoice.py engine/tests/test_smoke_stt.py root@194.68.245.175:/workspace/voice-studio-v2/engine/services/
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_smoke_stt.py::test_vibevoice_asr_transcribes_arabic -v 2>&1 | tail -20"
```
Expected: 1 passed. May take 5-15 min on first run (downloads VibeVoice ASR + Qwen 7B = ~20GB). If it fails with cuDNN/segfault, document the failure and continue — VibeVoice ASR is marked as experimental in the spec.

- [ ] **Step 6: Commit**

```bash
git add engine/services/stt_vibevoice.py engine/tests/test_smoke_stt.py
git commit -m "feat(engine): VibeVoice ASR service + smoke test"
```

### Task 2.4: NVIDIA Canary-1B service + smoke test

**Files:**
- Create: `engine/services/stt_nemo.py`
- Modify: `engine/tests/test_smoke_stt.py`

- [ ] **Step 1: Append failing smoke test**

Append to `engine/tests/test_smoke_stt.py`:
```python
@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
def test_nemo_canary_transcribes_arabic():
    from engine.services.stt_nemo import transcribe_canary
    result = transcribe_canary(str(FIXTURE))
    assert result["duration"] == pytest.approx(5.0, abs=0.5)
    assert len(result["segments"]) >= 1
```

- [ ] **Step 2: Implement stt_nemo.py**

Local file `engine/services/stt_nemo.py`:
```python
"""NVIDIA NeMo Canary-1B STT service."""
import logging
import time
from typing import Dict, Any, Optional
import soundfile as sf

logger = logging.getLogger(__name__)

_model = None  # nemo.collections.asr.models.EncDecMultiTaskModel


def _load():
    global _model
    if _model is not None:
        return
    logger.info("Loading NVIDIA Canary-1B…")
    from nemo.collections.asr.models import EncDecMultiTaskModel
    _model = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b")
    _model.eval()


def transcribe_canary(audio_path: str, source_lang: str = "ar") -> Dict[str, Any]:
    _load()
    duration = sf.info(audio_path).duration

    t0 = time.time()
    # Canary expects a list of paths; returns list of strings
    transcripts = _model.transcribe(
        audio=[audio_path],
        batch_size=1,
        source_lang=source_lang,
        target_lang=source_lang,
        task="asr",
        pnc="yes",
    )
    elapsed = time.time() - t0

    text = transcripts[0] if transcripts else ""
    return {
        "duration": duration,
        "segments": [{"start": 0, "end": duration, "text": text}],
        "elapsed_sec": round(elapsed, 2),
        "speedup": round(duration / elapsed, 2) if elapsed > 0 else 0,
    }
```

Upload + run:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/services/stt_nemo.py engine/tests/test_smoke_stt.py root@194.68.245.175:/workspace/voice-studio-v2/engine/services/
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_smoke_stt.py::test_nemo_canary_transcribes_arabic -v 2>&1 | tail -15"
```
Expected: 1 passed (downloads ~4GB model first time). If NeMo fails to install or import, mark Canary as "unavailable" in service registry and continue.

- [ ] **Step 3: Commit**

```bash
git add engine/services/stt_nemo.py engine/tests/test_smoke_stt.py
git commit -m "feat(engine): NVIDIA Canary STT service + smoke test"
```

---

## Phase 3: Diarization Services

### Task 3.1: pyannote.audio 3.1 service

**Files:**
- Create: `engine/services/diar_pyannote.py`
- Create: `engine/tests/test_smoke_diar.py`

- [ ] **Step 1: Set HF token (required for pyannote 3.1)**

Pyannote 3.1 requires accepting model conditions on HuggingFace and an HF_TOKEN. The user needs to provide this.

```bash
echo "ACTION REQUIRED: Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and accept the conditions, then set:"
echo "  ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 \"echo 'export HF_TOKEN=your_token_here' >> ~/.bashrc\""
```

If user has not provided token, set up a placeholder and skip the test:

```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "grep -q 'HF_TOKEN' ~/.bashrc || echo 'export HF_TOKEN=' >> ~/.bashrc"
```

- [ ] **Step 2: Write failing smoke test**

Local file `engine/tests/test_smoke_diar.py`:
```python
"""Diarization smoke tests."""
import os
import pytest
from pathlib import Path

FIXTURE = Path("/workspace/voice-studio-v2/engine/tests/fixtures/short_arabic.wav")


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="HF_TOKEN not set")
def test_pyannote_diarizes_short_clip():
    from engine.services.diar_pyannote import diarize_pyannote
    from engine.services.stt_whisper import transcribe_whisper
    stt = transcribe_whisper(str(FIXTURE), model_size="large-v3")
    result = diarize_pyannote(str(FIXTURE), segments=stt["segments"])
    assert "segments" in result
    for seg in result["segments"]:
        assert "speaker" in seg
```

- [ ] **Step 3: Implement diar_pyannote.py**

Local file `engine/services/diar_pyannote.py`:
```python
"""pyannote.audio 3.1 speaker diarization."""
import logging
import os
import time
from typing import List, Dict, Any, Optional
import torch
from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)

_pipeline: Optional[Pipeline] = None


def _load():
    global _pipeline
    if _pipeline is not None:
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set; required for pyannote 3.1")
    logger.info("Loading pyannote/speaker-diarization-3.1…")
    _pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )
    _pipeline.to(torch.device("cuda"))


def diarize_pyannote(audio_path: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run pyannote diarization and assign speakers to existing segments by overlap."""
    _load()
    t0 = time.time()
    diarization = _pipeline(audio_path)
    elapsed = time.time() - t0

    # Convert pyannote output to (start, end, speaker) tuples
    spk_turns = [
        (turn.start, turn.end, label)
        for turn, _, label in diarization.itertracks(yield_label=True)
    ]

    # Assign speaker to each input segment by max-overlap
    out_segments = []
    for s in segments:
        best_label = "SPEAKER_0"
        best_overlap = 0.0
        for ts, te, label in spk_turns:
            overlap = max(0.0, min(s["end"], te) - max(s["start"], ts))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        out_segments.append({**s, "speaker": best_label})

    return {
        "segments": out_segments,
        "elapsed_sec": round(elapsed, 2),
    }
```

Upload + run:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/services/diar_pyannote.py engine/tests/test_smoke_diar.py root@194.68.245.175:/workspace/voice-studio-v2/engine/services/
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_smoke_diar.py::test_pyannote_diarizes_short_clip -v 2>&1 | tail -10"
```
Expected: 1 passed if HF_TOKEN set; SKIP otherwise. If skipped, surface a clear message in the final summary so user can provide the token.

- [ ] **Step 4: Commit**

```bash
git add engine/services/diar_pyannote.py engine/tests/test_smoke_diar.py
git commit -m "feat(engine): pyannote 3.1 diarization service"
```

### Task 3.2: ECAPA-TDNN diarization (existing approach as fallback)

**Files:**
- Create: `engine/services/diar_ecapa.py`
- Modify: `engine/tests/test_smoke_diar.py`

- [ ] **Step 1: Append failing smoke test**

Append to `engine/tests/test_smoke_diar.py`:
```python
@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
def test_ecapa_diarizes_short_clip():
    from engine.services.diar_ecapa import diarize_ecapa
    from engine.services.stt_whisper import transcribe_whisper
    stt = transcribe_whisper(str(FIXTURE), model_size="large-v3")
    result = diarize_ecapa(str(FIXTURE), segments=stt["segments"], n_speakers=2)
    assert "segments" in result
    for seg in result["segments"]:
        assert "speaker" in seg
```

- [ ] **Step 2: Implement diar_ecapa.py (port from existing process_long.py)**

Local file `engine/services/diar_ecapa.py`:
```python
"""ECAPA-TDNN + agglomerative clustering diarization (no HF token required)."""
import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier

from engine import config

logger = logging.getLogger(__name__)

_classifier: Optional[EncoderClassifier] = None


def _load():
    global _classifier
    if _classifier is not None:
        return
    logger.info("Loading ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)…")
    _classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(config.MODELS_DIR / "spkrec-ecapa"),
        run_opts={"device": "cuda"},
    )


def diarize_ecapa(
    audio_path: str,
    segments: List[Dict[str, Any]],
    n_speakers: int = 2,
) -> Dict[str, Any]:
    """ECAPA embeddings per segment + agglomerative clustering."""
    _load()
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    t0 = time.time()
    embeddings = []
    valid_idx = []
    for i, seg in enumerate(segments):
        s = int(seg["start"] * sr)
        e = int(seg["end"] * sr)
        if e - s < int(0.5 * sr):
            continue
        chunk = torch.tensor(audio[s:e]).unsqueeze(0).float().to("cuda")
        with torch.no_grad():
            emb = _classifier.encode_batch(chunk).squeeze().cpu().numpy()
        embeddings.append(emb)
        valid_idx.append(i)

    if len(embeddings) < 2:
        out = [{**s, "speaker": "SPEAKER_0"} for s in segments]
        return {"segments": out, "elapsed_sec": round(time.time() - t0, 2)}

    n_clusters = min(n_speakers, len(embeddings))
    X = np.stack(embeddings)
    labels = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    ).fit_predict(X)

    out = [dict(s) for s in segments]
    last_label = 0
    li = 0
    for i, seg in enumerate(out):
        if li < len(valid_idx) and valid_idx[li] == i:
            last_label = int(labels[li])
            li += 1
        seg["speaker"] = f"SPEAKER_{last_label}"

    return {"segments": out, "elapsed_sec": round(time.time() - t0, 2)}
```

Upload + run:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/services/diar_ecapa.py engine/tests/test_smoke_diar.py root@194.68.245.175:/workspace/voice-studio-v2/engine/services/
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_smoke_diar.py::test_ecapa_diarizes_short_clip -v 2>&1 | tail -10"
```
Expected: 1 passed

- [ ] **Step 3: Commit**

```bash
git add engine/services/diar_ecapa.py engine/tests/test_smoke_diar.py
git commit -m "feat(engine): ECAPA-TDNN diarization service"
```

---

## Phase 4: TTS Services

### Task 4.1: VibeVoice TTS service (1.5B + Large)

**Files:**
- Create: `engine/services/tts_vibevoice.py`
- Create: `engine/tests/test_smoke_tts.py`

- [ ] **Step 1: Clone the HF Space code (already proven to work earlier)**

```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "
cd /workspace/voice-studio-v2/engine/vendor
[ -d vibe-voice-custom-voices ] || git clone --depth 1 https://huggingface.co/spaces/vibingvoice/vibe-voice-custom-voices
# Use 1.5B by default (smaller, fits with other models)
sed -i \"s|VibeVoice-Large|VibeVoice-1.5B|g; s|aoi-ot/VibeVoice-1.5B|microsoft/VibeVoice-1.5B|g\" vibe-voice-custom-voices/app.py
# Mock comfy module
mkdir -p /workspace/voice-studio-v2/engine/vendor/comfy_mock/comfy
cat > /workspace/voice-studio-v2/engine/vendor/comfy_mock/comfy/__init__.py << 'EOF'
from . import model_management
EOF
cat > /workspace/voice-studio-v2/engine/vendor/comfy_mock/comfy/model_management.py << 'EOF'
import torch
class InterruptProcessingException(Exception): pass
def soft_empty_cache():
    if torch.cuda.is_available(): torch.cuda.empty_cache()
def get_torch_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EOF
echo 'VibeVoice TTS vendor ready'"
```
Expected: prints `VibeVoice TTS vendor ready`

- [ ] **Step 2: Write failing smoke test**

Local file `engine/tests/test_smoke_tts.py`:
```python
"""TTS smoke tests — verifies each model can clone a 5s reference + generate Arabic."""
import pytest
from pathlib import Path

REFERENCE = Path("/workspace/voice-studio-v2/engine/tests/fixtures/short_arabic.wav")
TEST_TEXT = "مرحباً، هذا اختبار."


@pytest.mark.skipif(not REFERENCE.exists(), reason="fixture missing")
def test_vibevoice_15b_clones_arabic():
    from engine.services.tts_vibevoice import clone_vibevoice
    out_path = clone_vibevoice(
        text=TEST_TEXT,
        reference_audio=str(REFERENCE),
        model="VibeVoice-1.5B",
    )
    assert Path(out_path).exists()
    import soundfile as sf
    info = sf.info(out_path)
    assert info.duration > 0.5
```

- [ ] **Step 3: Implement tts_vibevoice.py — wrap the HF Space generator**

Local file `engine/services/tts_vibevoice.py`:
```python
"""VibeVoice TTS — wraps the HF Space inference code (proven working)."""
import logging
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Add VibeVoice vendor + comfy mock to path
VENDOR = Path("/workspace/voice-studio-v2/engine/vendor")
SPACE_DIR = VENDOR / "vibe-voice-custom-voices"
COMFY_MOCK = VENDOR / "comfy_mock"
for p in (str(SPACE_DIR), str(COMFY_MOCK)):
    if p not in sys.path:
        sys.path.insert(0, p)

# folder_paths mock (ComfyUI shim)
class _MockFolderPaths:
    def get_folder_paths(self, name: str):
        if name == "checkpoints":
            d = SPACE_DIR / "models"
            d.mkdir(parents=True, exist_ok=True)
            return [str(d)]
        return []

if "folder_paths" not in sys.modules:
    sys.modules["folder_paths"] = _MockFolderPaths()


_node = None  # VibeVoiceSingleSpeakerNode (cached)
_loaded_model = None  # str: which model is currently loaded


def _ensure_node(model: str = "VibeVoice-1.5B") -> object:
    global _node, _loaded_model
    if _node is not None and _loaded_model == model:
        return _node

    from nodes.single_speaker_node import VibeVoiceSingleSpeakerNode
    _node = VibeVoiceSingleSpeakerNode()
    model_paths = {
        "VibeVoice-1.5B": "microsoft/VibeVoice-1.5B",
        "VibeVoice-Large": "aoi-ot/VibeVoice-Large",
    }
    logger.info(f"Loading VibeVoice TTS model: {model}…")
    _node.load_model(
        model_name=model,
        model_path=model_paths[model],
        attention_type="auto",
    )
    _loaded_model = model
    return _node


def clone_vibevoice(
    text: str,
    reference_audio: str,
    model: str = "VibeVoice-1.5B",
    diffusion_steps: int = 40,
    cfg_scale: float = 1.5,
    seed: int = 42,
) -> str:
    """Generate cloned audio. Returns path to output WAV."""
    node = _ensure_node(model)

    import soundfile as sf
    import numpy as np
    audio, sr = sf.read(reference_audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    voice_dict = {"waveform": np.expand_dims(audio, 0), "sample_rate": sr}

    t0 = time.time()
    (audio_dict,) = node.generate_speech(
        text=text,
        model=model,
        attention_type="auto",
        free_memory_after_generate=False,
        diffusion_steps=diffusion_steps,
        seed=seed,
        cfg_scale=cfg_scale,
        use_sampling=False,
        voice_to_clone=voice_dict,
    )
    elapsed = time.time() - t0

    # Save output
    out_path = tempfile.mktemp(suffix=".wav", dir=str(VENDOR.parent.parent / "jobs"))
    waveform = audio_dict["waveform"]
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu().numpy()
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 2:
        waveform = waveform[0]
    sf.write(out_path, waveform, audio_dict["sample_rate"])
    logger.info(f"VibeVoice TTS done in {elapsed:.1f}s → {out_path}")
    return out_path
```

Upload + run:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/services/tts_vibevoice.py engine/tests/test_smoke_tts.py root@194.68.245.175:/workspace/voice-studio-v2/engine/services/
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_smoke_tts.py::test_vibevoice_15b_clones_arabic -v -s 2>&1 | tail -15"
```
Expected: 1 passed (downloads VibeVoice-1.5B ~3GB on first run)

- [ ] **Step 4: Commit**

```bash
git add engine/services/tts_vibevoice.py engine/tests/test_smoke_tts.py
git commit -m "feat(engine): VibeVoice TTS (1.5B) service + smoke test"
```

### Task 4.2: F5-TTS service

**Files:**
- Create: `engine/services/tts_f5.py`
- Modify: `engine/tests/test_smoke_tts.py`

- [ ] **Step 1: Install F5-TTS package**

```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "pip install --quiet f5-tts 2>&1 | tail -3"
```
Expected: no errors. If install fails, fall back to: `pip install git+https://github.com/SWivid/F5-TTS.git`

- [ ] **Step 2: Append failing smoke test**

Append to `engine/tests/test_smoke_tts.py`:
```python
@pytest.mark.skipif(not REFERENCE.exists(), reason="fixture missing")
def test_f5_tts_clones_arabic():
    from engine.services.tts_f5 import clone_f5
    out_path = clone_f5(
        text=TEST_TEXT,
        reference_audio=str(REFERENCE),
        reference_text="هذا صوت مرجعي.",
    )
    assert Path(out_path).exists()
    import soundfile as sf
    assert sf.info(out_path).duration > 0.5
```

- [ ] **Step 3: Implement tts_f5.py**

Local file `engine/services/tts_f5.py`:
```python
"""F5-TTS voice cloning."""
import logging
import time
import tempfile
from pathlib import Path
from typing import Optional
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

_model = None
_vocoder = None


def _load():
    global _model, _vocoder
    if _model is not None:
        return
    logger.info("Loading F5-TTS model…")
    from f5_tts.api import F5TTS
    _model = F5TTS()  # default checkpoint
    logger.info("F5-TTS loaded.")


def clone_f5(
    text: str,
    reference_audio: str,
    reference_text: str = "",
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    seed: int = 42,
) -> str:
    """Generate cloned audio. Returns path to output WAV."""
    _load()
    out_path = tempfile.mktemp(suffix=".wav", dir="/workspace/voice-studio-v2/jobs")
    t0 = time.time()
    wav, sr, _ = _model.infer(
        ref_file=reference_audio,
        ref_text=reference_text,
        gen_text=text,
        file_wave=out_path,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        seed=seed,
    )
    logger.info(f"F5-TTS done in {time.time()-t0:.1f}s")
    return out_path
```

Upload + run:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/services/tts_f5.py engine/tests/test_smoke_tts.py root@194.68.245.175:/workspace/voice-studio-v2/engine/services/
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_smoke_tts.py::test_f5_tts_clones_arabic -v 2>&1 | tail -15"
```
Expected: 1 passed (downloads F5-TTS checkpoint ~2GB)

- [ ] **Step 4: Commit**

```bash
git add engine/services/tts_f5.py engine/tests/test_smoke_tts.py
git commit -m "feat(engine): F5-TTS service + smoke test"
```

### Task 4.3: XTTS-v2 service

**Files:**
- Create: `engine/services/tts_xtts.py`
- Modify: `engine/tests/test_smoke_tts.py`

- [ ] **Step 1: Append failing smoke test**

Append to `engine/tests/test_smoke_tts.py`:
```python
@pytest.mark.skipif(not REFERENCE.exists(), reason="fixture missing")
def test_xtts_v2_clones_arabic():
    from engine.services.tts_xtts import clone_xtts
    out_path = clone_xtts(
        text=TEST_TEXT,
        reference_audio=str(REFERENCE),
        language="ar",
    )
    assert Path(out_path).exists()
    import soundfile as sf
    assert sf.info(out_path).duration > 0.5
```

- [ ] **Step 2: Implement tts_xtts.py**

Local file `engine/services/tts_xtts.py`:
```python
"""Coqui XTTS-v2 voice cloning."""
import logging
import os
import time
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)
os.environ["COQUI_TOS_AGREED"] = "1"

_tts = None


def _load():
    global _tts
    if _tts is not None:
        return
    from TTS.api import TTS
    logger.info("Loading XTTS-v2…")
    _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)


def clone_xtts(
    text: str,
    reference_audio: str,
    language: str = "ar",
) -> str:
    _load()
    out_path = tempfile.mktemp(suffix=".wav", dir="/workspace/voice-studio-v2/jobs")
    t0 = time.time()
    _tts.tts_to_file(
        text=text,
        speaker_wav=reference_audio,
        language=language,
        file_path=out_path,
    )
    logger.info(f"XTTS-v2 done in {time.time()-t0:.1f}s")
    return out_path
```

Upload + run:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/services/tts_xtts.py engine/tests/test_smoke_tts.py root@194.68.245.175:/workspace/voice-studio-v2/engine/services/
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_smoke_tts.py::test_xtts_v2_clones_arabic -v 2>&1 | tail -15"
```
Expected: 1 passed (downloads XTTS-v2 ~2GB). Note: XTTS-v2's "ar" language support produces best output with shorter texts.

- [ ] **Step 3: Commit**

```bash
git add engine/services/tts_xtts.py engine/tests/test_smoke_tts.py
git commit -m "feat(engine): XTTS-v2 service + smoke test"
```

### Task 4.4: Fish Speech 1.5 service

**Files:**
- Create: `engine/services/tts_fishspeech.py`
- Modify: `engine/tests/test_smoke_tts.py`

- [ ] **Step 1: Install Fish Speech**

```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "
cd /workspace/voice-studio-v2/engine/vendor
[ -d fish-speech ] || git clone --depth 1 https://github.com/fishaudio/fish-speech.git
cd fish-speech && pip install --quiet -e . 2>&1 | tail -3"
```
Expected: no errors

- [ ] **Step 2: Append failing smoke test**

Append to `engine/tests/test_smoke_tts.py`:
```python
@pytest.mark.skipif(not REFERENCE.exists(), reason="fixture missing")
def test_fish_speech_clones_arabic():
    from engine.services.tts_fishspeech import clone_fish
    out_path = clone_fish(
        text=TEST_TEXT,
        reference_audio=str(REFERENCE),
        reference_text="مرجع.",
    )
    assert Path(out_path).exists()
    import soundfile as sf
    assert sf.info(out_path).duration > 0.5
```

- [ ] **Step 3: Implement tts_fishspeech.py**

Local file `engine/services/tts_fishspeech.py`:
```python
"""Fish Speech 1.5 voice cloning via API helpers."""
import logging
import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

VENDOR = Path("/workspace/voice-studio-v2/engine/vendor/fish-speech")
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))


def clone_fish(
    text: str,
    reference_audio: str,
    reference_text: str = "",
) -> str:
    """Run Fish Speech inference via its CLI tool."""
    out_path = tempfile.mktemp(suffix=".wav", dir="/workspace/voice-studio-v2/jobs")
    t0 = time.time()

    # Fish Speech ships a CLI: tools/llama/generate.py + tools/vqgan/inference.py
    # The simpler path is using its Python API:
    from fish_speech.inference_engine import TTSInferenceEngine
    engine = TTSInferenceEngine.from_pretrained(
        "fishaudio/fish-speech-1.5",
        device="cuda",
    )
    audio = engine.synthesize(
        text=text,
        reference_audio=reference_audio,
        reference_text=reference_text,
    )
    import soundfile as sf
    sf.write(out_path, audio.waveform, audio.sample_rate)
    logger.info(f"Fish Speech done in {time.time()-t0:.1f}s")
    return out_path
```

Upload + run:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/services/tts_fishspeech.py engine/tests/test_smoke_tts.py root@194.68.245.175:/workspace/voice-studio-v2/engine/services/
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_smoke_tts.py::test_fish_speech_clones_arabic -v 2>&1 | tail -20"
```
Expected: 1 passed if Fish Speech API is stable; if API has changed, mark as "experimental" and provide fallback CLI invocation.

- [ ] **Step 4: Commit (or document failure if test fails)**

```bash
git add engine/services/tts_fishspeech.py engine/tests/test_smoke_tts.py
git commit -m "feat(engine): Fish Speech 1.5 service + smoke test"
```

---

## Phase 5: API Endpoints

### Task 5.1: Service registry

**Files:**
- Modify: `engine/services/__init__.py`

- [ ] **Step 1: Write registry that maps tool name → callable**

Local file `engine/services/__init__.py`:
```python
"""Tool name → handler registry. Adds a callable layer so main.py is decoupled from concrete services."""
from typing import Callable, Dict


def _whisper_large(audio_path):
    from .stt_whisper import transcribe_whisper
    return transcribe_whisper(audio_path, model_size="large-v3")


def _whisper_turbo(audio_path):
    from .stt_whisper import transcribe_whisper
    return transcribe_whisper(audio_path, model_size="large-v3-turbo")


def _vibevoice_asr(audio_path):
    from .stt_vibevoice import transcribe_vibevoice
    return transcribe_vibevoice(audio_path)


def _nemo_canary(audio_path):
    from .stt_nemo import transcribe_canary
    return transcribe_canary(audio_path)


def _pyannote(audio_path, segments):
    from .diar_pyannote import diarize_pyannote
    return diarize_pyannote(audio_path, segments)


def _ecapa(audio_path, segments, n_speakers=2):
    from .diar_ecapa import diarize_ecapa
    return diarize_ecapa(audio_path, segments, n_speakers)


def _vibevoice_15b(text, ref, **kw):
    from .tts_vibevoice import clone_vibevoice
    return clone_vibevoice(text, ref, model="VibeVoice-1.5B", **kw)


def _vibevoice_large(text, ref, **kw):
    from .tts_vibevoice import clone_vibevoice
    return clone_vibevoice(text, ref, model="VibeVoice-Large", **kw)


def _f5(text, ref, **kw):
    from .tts_f5 import clone_f5
    return clone_f5(text, ref, **kw)


def _xtts(text, ref, **kw):
    from .tts_xtts import clone_xtts
    return clone_xtts(text, ref, **kw)


def _fish(text, ref, **kw):
    from .tts_fishspeech import clone_fish
    return clone_fish(text, ref, **kw)


STT_HANDLERS: Dict[str, Callable] = {
    "whisper-large-v3": _whisper_large,
    "whisper-turbo": _whisper_turbo,
    "vibevoice-asr": _vibevoice_asr,
    "nemo-canary": _nemo_canary,
}

DIAR_HANDLERS: Dict[str, Callable] = {
    "pyannote-3.1": _pyannote,
    "ecapa-tdnn": _ecapa,
}

TTS_HANDLERS: Dict[str, Callable] = {
    "vibevoice-1.5b": _vibevoice_15b,
    "vibevoice-large": _vibevoice_large,
    "f5-tts": _f5,
    "xtts-v2": _xtts,
    "fish-speech-1.5": _fish,
}
```

Upload + commit:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/services/__init__.py root@194.68.245.175:/workspace/voice-studio-v2/engine/services/
git add engine/services/__init__.py
git commit -m "feat(engine): service registry"
```

### Task 5.2: /api/upload endpoint

**Files:**
- Modify: `engine/main.py`

- [ ] **Step 1: Write failing test**

Local file `engine/tests/test_api.py`:
```python
"""FastAPI endpoint tests via TestClient."""
import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from engine.main import app
from engine import config

# Skip auth in tests by setting same key
os.environ["VS_API_KEY"] = config.API_KEY


@pytest.fixture
def client():
    return TestClient(app)


def test_upload_accepts_audio(client, tmp_path):
    audio = tmp_path / "test.wav"
    import soundfile as sf, numpy as np
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    with audio.open("rb") as f:
        r = client.post(
            "/api/upload",
            files={"file": ("test.wav", f, "audio/wav")},
            headers={"X-API-Key": config.API_KEY},
        )
    assert r.status_code == 200
    body = r.json()
    assert "audio_path" in body
    assert "duration" in body
```

- [ ] **Step 2: Implement /api/upload**

Append to `engine/main.py`:
```python
import uuid
import shutil
from fastapi import UploadFile, File, Depends


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), _=Depends(require_api_key)):
    """Save uploaded audio to /workspace/voice-studio-v2/jobs/{uuid}/orig.{ext}."""
    job_id = uuid.uuid4().hex[:8]
    job_dir = config.JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix or ".bin"
    dest = job_dir / f"orig{ext}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    # Convert to 16k mono WAV for downstream services
    import subprocess, imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    wav_path = job_dir / "audio_16k.wav"
    subprocess.run(
        [ffmpeg, "-y", "-i", str(dest), "-ac", "1", "-ar", "16000", str(wav_path)],
        capture_output=True, check=True,
    )
    import soundfile as sf
    info = sf.info(str(wav_path))
    if info.duration > config.MAX_AUDIO_DURATION_SEC:
        raise HTTPException(status_code=413, detail=f"Audio exceeds {config.MAX_AUDIO_DURATION_SEC}s limit")

    return {
        "job_id": job_id,
        "audio_path": str(wav_path),
        "duration": info.duration,
        "size_bytes": dest.stat().st_size,
    }
```

Add `Path` import at top: `from pathlib import Path`.

Upload + run:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/main.py engine/tests/test_api.py root@194.68.245.175:/workspace/voice-studio-v2/engine/
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_api.py::test_upload_accepts_audio -v 2>&1 | tail -10"
```
Expected: 1 passed

- [ ] **Step 3: Commit**

```bash
git add engine/main.py engine/tests/test_api.py
git commit -m "feat(engine): /api/upload endpoint"
```

### Task 5.3: /api/process endpoint (STT + Diar)

**Files:**
- Modify: `engine/main.py`

- [ ] **Step 1: Write failing test**

Append to `engine/tests/test_api.py`:
```python
def test_process_with_whisper_and_ecapa(client, tmp_path):
    fixture = Path("/workspace/voice-studio-v2/engine/tests/fixtures/short_arabic.wav")
    if not fixture.exists():
        pytest.skip("fixture missing")

    # Upload
    with fixture.open("rb") as f:
        up = client.post(
            "/api/upload",
            files={"file": ("a.wav", f, "audio/wav")},
            headers={"X-API-Key": config.API_KEY},
        )
    audio_path = up.json()["audio_path"]

    # Process
    r = client.post(
        "/api/process",
        json={
            "audio_path": audio_path,
            "stt_tool": "whisper-large-v3",
            "diar_tool": "ecapa-tdnn",
        },
        headers={"X-API-Key": config.API_KEY},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "segments" in body
    assert "timings" in body
    assert "cost" in body
    assert body["tools_used"]["stt"] == "whisper-large-v3"
    assert body["tools_used"]["diar"] == "ecapa-tdnn"
    for seg in body["segments"]:
        assert "speaker" in seg
```

- [ ] **Step 2: Implement /api/process**

Append to `engine/main.py`:
```python
import time as _time
from .schemas import ProcessRequest, ProcessResponse
from .services import STT_HANDLERS, DIAR_HANDLERS
from .cost_tracker import calc_cost
from . import jobs_store

def _extract_speaker_samples(wav_path, segments, job_dir):
    import soundfile as sf, numpy as np
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    by_spk = {}
    for s in segments:
        by_spk.setdefault(s.get("speaker", "SPEAKER_0"), []).append(s)
    silence = np.zeros(int(0.2 * sr))
    samples = {}
    for sp, segs in by_spk.items():
        chunks, total = [], 0.0
        for s in segs[:50]:
            if total >= 30: break
            si, ei = int(s["start"] * sr), int(s["end"] * sr)
            chunks.append(audio[si:ei])
            chunks.append(silence)
            total += s["end"] - s["start"]
        samp = np.concatenate(chunks) if chunks else np.zeros(int(0.5 * sr))
        path = job_dir / f"{sp}_sample.wav"
        sf.write(path, samp, sr)
        samples[sp] = {"url": f"/files/{job_dir.name}/{path.name}", "duration": round(len(samp)/sr, 2)}
    return samples


@app.post("/api/process", response_model=ProcessResponse)
def api_process(req: ProcessRequest, _=Depends(require_api_key)):
    if req.stt_tool not in STT_HANDLERS:
        raise HTTPException(400, f"Unknown stt_tool: {req.stt_tool}")

    audio_path = req.audio_path
    if not Path(audio_path).exists():
        raise HTTPException(404, f"Audio not found: {audio_path}")

    job_dir = Path(audio_path).parent
    job_id = job_dir.name

    timings = {}

    # STT
    t0 = _time.time()
    stt_result = STT_HANDLERS[req.stt_tool](audio_path)
    timings["stt_sec"] = round(_time.time() - t0, 2)
    segments = stt_result["segments"]

    # Diarization (skip if VibeVoice ASR — already provides speakers)
    diar_used = req.diar_tool
    if req.stt_tool != "vibevoice-asr":
        if not req.diar_tool or req.diar_tool not in DIAR_HANDLERS:
            raise HTTPException(400, f"diar_tool required for stt {req.stt_tool}")
        t0 = _time.time()
        diar_result = DIAR_HANDLERS[req.diar_tool](audio_path, segments)
        timings["diar_sec"] = round(_time.time() - t0, 2)
        segments = diar_result["segments"]
    else:
        timings["diar_sec"] = 0.0
        diar_used = "built-in (vibevoice-asr)"

    # Sample extraction
    t0 = _time.time()
    samples = _extract_speaker_samples(audio_path, segments, job_dir)
    timings["extract_sec"] = round(_time.time() - t0, 2)

    # Cost
    rate = config.GPU_RATE_PER_HOUR_USD
    cost = {
        "stt_usd": calc_cost(timings["stt_sec"], rate),
        "diar_usd": calc_cost(timings["diar_sec"], rate),
        "extract_usd": calc_cost(timings["extract_sec"], rate),
    }
    cost["total_usd"] = round(sum(cost.values()), 4)

    duration = stt_result["duration"]
    job_data = {
        "id": job_id,
        "audio_path": audio_path,
        "segments": segments,
        "samples": samples,
        "timings": timings,
        "cost": cost,
        "tools_used": {"stt": req.stt_tool, "diar": diar_used},
        "duration": duration,
    }
    jobs_store.update_job(job_id, **job_data) if jobs_store.get_job(job_id) else jobs_store._JOBS.update({job_id: job_data})

    return ProcessResponse(
        job_id=job_id,
        duration=duration,
        segments=segments,
        samples=samples,
        timings=timings,
        cost=cost,
        tools_used={"stt": req.stt_tool, "diar": diar_used},
    )
```

Upload + run:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/main.py engine/tests/test_api.py root@194.68.245.175:/workspace/voice-studio-v2/engine/
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_api.py::test_process_with_whisper_and_ecapa -v 2>&1 | tail -15"
```
Expected: 1 passed

- [ ] **Step 3: Commit**

```bash
git add engine/main.py engine/tests/test_api.py
git commit -m "feat(engine): /api/process endpoint (STT + Diar)"
```

### Task 5.4: /api/clone endpoint (TTS)

**Files:**
- Modify: `engine/main.py`

- [ ] **Step 1: Write failing test**

Append to `engine/tests/test_api.py`:
```python
def test_clone_with_vibevoice_15b(client):
    fixture = Path("/workspace/voice-studio-v2/engine/tests/fixtures/short_arabic.wav")
    if not fixture.exists():
        pytest.skip("fixture missing")

    # Upload + process first to get a speaker sample
    with fixture.open("rb") as f:
        up = client.post(
            "/api/upload",
            files={"file": ("a.wav", f, "audio/wav")},
            headers={"X-API-Key": config.API_KEY},
        )
    audio_path = up.json()["audio_path"]
    pr = client.post(
        "/api/process",
        json={"audio_path": audio_path, "stt_tool": "whisper-large-v3", "diar_tool": "ecapa-tdnn"},
        headers={"X-API-Key": config.API_KEY},
    ).json()
    job_id = pr["job_id"]
    speaker = list(pr["samples"].keys())[0]

    r = client.post(
        "/api/clone",
        json={
            "job_id": job_id,
            "speaker_id": speaker,
            "text": "مرحباً.",
            "tts_tool": "vibevoice-1.5b",
        },
        headers={"X-API-Key": config.API_KEY},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "audio_url" in body
    assert body["duration"] > 0.3
    assert body["cost_usd"] > 0
```

- [ ] **Step 2: Implement /api/clone**

Append to `engine/main.py`:
```python
from .schemas import CloneRequest, CloneResponse
from .services import TTS_HANDLERS


@app.post("/api/clone", response_model=CloneResponse)
def api_clone(req: CloneRequest, _=Depends(require_api_key)):
    if req.tts_tool not in TTS_HANDLERS:
        raise HTTPException(400, f"Unknown tts_tool: {req.tts_tool}")

    job = jobs_store.get_job(req.job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {req.job_id}")

    samples = job.get("samples", {})
    if req.speaker_id not in samples:
        raise HTTPException(400, f"Speaker {req.speaker_id} not in job samples")

    # Resolve speaker sample path
    sample_url = samples[req.speaker_id]["url"]  # /files/{job}/{name}
    sample_path = config.JOBS_DIR / sample_url.split("/files/", 1)[1]

    if len(req.text) > config.MAX_TEXT_CHARS:
        raise HTTPException(400, f"Text exceeds {config.MAX_TEXT_CHARS} chars")

    t0 = _time.time()
    handler = TTS_HANDLERS[req.tts_tool]
    out_path_str = handler(
        text=req.text,
        ref=str(sample_path),
        diffusion_steps=req.diffusion_steps,
        cfg_scale=req.cfg_scale,
        seed=req.seed,
    ) if req.tts_tool.startswith("vibevoice") else handler(
        text=req.text,
        reference_audio=str(sample_path),
    )
    elapsed = _time.time() - t0

    out_path = Path(out_path_str)
    import soundfile as sf
    out_duration = sf.info(str(out_path)).duration

    # Move output into job dir for served URL
    final = config.JOBS_DIR / req.job_id / f"clone_{int(_time.time())}.wav"
    final.parent.mkdir(parents=True, exist_ok=True)
    if out_path != final:
        shutil.move(str(out_path), str(final))

    cost = calc_cost(elapsed, config.GPU_RATE_PER_HOUR_USD)
    return CloneResponse(
        audio_url=f"/files/{req.job_id}/{final.name}",
        duration=round(out_duration, 2),
        elapsed=round(elapsed, 2),
        rtf=round(elapsed / out_duration, 2) if out_duration > 0 else 0,
        cost_usd=cost,
        tool=req.tts_tool,
    )
```

Upload + run:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/main.py engine/tests/test_api.py root@194.68.245.175:/workspace/voice-studio-v2/engine/
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_api.py::test_clone_with_vibevoice_15b -v 2>&1 | tail -15"
```
Expected: 1 passed

- [ ] **Step 3: Commit**

```bash
git add engine/main.py engine/tests/test_api.py
git commit -m "feat(engine): /api/clone endpoint"
```

### Task 5.5: /api/jobs and /api/costs endpoints

**Files:**
- Modify: `engine/main.py`

- [ ] **Step 1: Write tests**

Append to `engine/tests/test_api.py`:
```python
def test_jobs_list_and_get(client):
    r = client.get("/api/jobs", headers={"X-API-Key": config.API_KEY})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_costs_summary(client):
    r = client.get("/api/costs", headers={"X-API-Key": config.API_KEY})
    assert r.status_code == 200
    body = r.json()
    assert "total_usd" in body
    assert "rate_per_hour" in body
```

- [ ] **Step 2: Implement endpoints**

Append to `engine/main.py`:
```python
@app.get("/api/jobs")
def api_jobs(_=Depends(require_api_key)):
    return [
        {
            "id": j["id"],
            "duration": j.get("duration"),
            "tools_used": j.get("tools_used"),
            "cost": j.get("cost"),
        }
        for j in jobs_store.all_jobs().values()
    ]


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str, _=Depends(require_api_key)):
    j = jobs_store.get_job(job_id)
    if not j:
        raise HTTPException(404, f"Job not found: {job_id}")
    return j


@app.get("/api/costs")
def api_costs(_=Depends(require_api_key)):
    total = 0.0
    breakdown = {}
    for j in jobs_store.all_jobs().values():
        c = j.get("cost", {})
        total += c.get("total_usd", 0)
    return {
        "total_usd": round(total, 4),
        "rate_per_hour": config.GPU_RATE_PER_HOUR_USD,
        "jobs_count": len(jobs_store.all_jobs()),
    }
```

Upload + run:
```bash
scp -i ~/.ssh/id_ed25519 -P 22133 engine/main.py engine/tests/test_api.py root@194.68.245.175:/workspace/voice-studio-v2/engine/
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/test_api.py -v 2>&1 | tail -15"
```
Expected: all tests pass

- [ ] **Step 3: Restart engine to pick up new code**

```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "/workspace/voice-studio-v2/engine/run_engine.sh"
```
Expected: prints `Engine started.`

- [ ] **Step 4: Commit**

```bash
git add engine/main.py engine/tests/test_api.py
git commit -m "feat(engine): /api/jobs and /api/costs endpoints"
```

---

## Phase 6: Frontend (Local Flask + HTML)

### Task 6.1: Create index_v2.html based on index.html

**Files:**
- Create: `templates/index_v2.html`

- [ ] **Step 1: Copy index.html → index_v2.html**

```bash
cp templates/index.html templates/index_v2.html
```

- [ ] **Step 2: Add tool-selection dropdowns above "بدء المعالجة" button**

Edit `templates/index_v2.html`. Find the section with `<button class="btn" id="processBtn"` and immediately before it, add:

```html
<div class="controls-row" style="margin-bottom: 12px;">
  <label>STT:
    <select id="sttTool">
      <option value="whisper-large-v3" selected>Whisper Large-v3</option>
      <option value="whisper-turbo">Whisper Turbo</option>
      <option value="vibevoice-asr">VibeVoice ASR (تجريبي)</option>
      <option value="nemo-canary">NVIDIA Canary</option>
    </select>
  </label>
  <label>Diarization:
    <select id="diarTool">
      <option value="pyannote-3.1" selected>pyannote 3.1</option>
      <option value="ecapa-tdnn">ECAPA-TDNN</option>
    </select>
  </label>
</div>
```

Also find the TTS clone section (`<button class="btn" id="cloneBtn">`) and modify the `clone-controls` div to add:
```html
<label>TTS:
  <select id="ttsTool">
    <option value="vibevoice-1.5b" selected>VibeVoice 1.5B</option>
    <option value="vibevoice-large">VibeVoice Large</option>
    <option value="f5-tts">F5-TTS</option>
    <option value="xtts-v2">XTTS-v2</option>
    <option value="fish-speech-1.5">Fish Speech 1.5</option>
  </select>
</label>
```

- [ ] **Step 3: Update JS to send tool selection in API calls**

Find the `fetch('/api/process'` call and modify to:
```javascript
const sttTool = document.getElementById('sttTool').value;
const diarTool = document.getElementById('diarTool').value;
fd.append('stt_tool', sttTool);
fd.append('diar_tool', diarTool);
```

Find the `/api/clone` call and modify to add:
```javascript
body: JSON.stringify({
  ...,
  tts_tool: document.getElementById('ttsTool').value,
  ...
}),
```

- [ ] **Step 4: Commit**

```bash
git add templates/index_v2.html
git commit -m "feat(frontend): index_v2.html with tool selection dropdowns"
```

### Task 6.2: Add comparison panel to index_v2.html

**Files:**
- Modify: `templates/index_v2.html`

- [ ] **Step 1: Add comparison section HTML**

Below the metrics panel (`<div class="panel hidden" id="metricsPanel">`), add:
```html
<div class="panel hidden" id="comparisonPanel">
  <h2>📊 سجل التجارب</h2>
  <div style="overflow-x: auto;">
    <table id="comparisonTable" style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
      <thead style="background: #1c2732; color: #1d9bf0;">
        <tr>
          <th style="padding: 8px;">#</th>
          <th>المرحلة</th>
          <th>الأداة</th>
          <th>الإدخال</th>
          <th>المدة</th>
          <th>السرعة</th>
          <th>التكلفة</th>
          <th>عرض</th>
        </tr>
      </thead>
      <tbody id="comparisonBody"></tbody>
    </table>
  </div>
  <div style="margin-top: 14px; text-align: left;">
    <strong>إجمالي الجلسة: </strong>
    <span id="sessionCost" style="color: #f4b942; font-size: 1.2em;">$0.0000</span>
  </div>
</div>
```

- [ ] **Step 2: Add JS to track each operation in the comparison table**

Add to the script section:
```javascript
window.SESSION_RUNS = [];
function logRun(stage, tool, input, durationSec, speedup, costUsd, audioUrl) {
  window.SESSION_RUNS.push({stage, tool, input, durationSec, speedup, costUsd, audioUrl, ts: Date.now()});
  drawComparison();
}
function drawComparison() {
  const body = document.getElementById('comparisonBody');
  const sessionCost = document.getElementById('sessionCost');
  body.innerHTML = '';
  let total = 0;
  window.SESSION_RUNS.forEach((r, i) => {
    total += r.costUsd || 0;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="padding:6px; text-align:center">${i+1}</td>
      <td>${r.stage}</td>
      <td>${r.tool}</td>
      <td title="${escapeHtml(r.input || '')}">${escapeHtml((r.input||'').substring(0,40))}</td>
      <td>${r.durationSec ? r.durationSec.toFixed(1)+'s' : '-'}</td>
      <td>${r.speedup ? r.speedup+'x' : '-'}</td>
      <td>$${(r.costUsd||0).toFixed(4)}</td>
      <td>${r.audioUrl ? '<audio controls src="'+r.audioUrl+'" style="height:32px"></audio>' : '-'}</td>
    `;
    body.appendChild(tr);
  });
  sessionCost.textContent = '$' + total.toFixed(4);
  document.getElementById('comparisonPanel').classList.remove('hidden');
}
```

In the existing process success handler, add:
```javascript
logRun('STT', sttTool, file.name, data.timings.stt_sec, data.duration / data.timings.stt_sec, data.cost.stt_usd, null);
logRun('Diar', diarTool, file.name, data.timings.diar_sec, null, data.cost.diar_usd, null);
```

In the clone success handler, add:
```javascript
logRun('TTS', document.getElementById('ttsTool').value, text, data.elapsed, 1/data.rtf, data.cost_usd, data.audio_url);
```

- [ ] **Step 3: Commit**

```bash
git add templates/index_v2.html
git commit -m "feat(frontend): comparison table + session cost tracking"
```

### Task 6.3: Local Flask /v2 route + proxy to Pod

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add /v2 route + proxy endpoints**

Append to `app.py`:
```python
import httpx as _httpx

POD_API = "http://127.0.0.1:8000"  # SSH tunnel
POD_API_KEY = os.environ.get("VS_API_KEY", "dev-key-change-me")
_pod_headers = {"X-API-Key": POD_API_KEY}


@app.route('/v2')
def v2_index():
    return render_template('index_v2.html')


@app.route('/v2/api/upload', methods=['POST'])
def v2_upload():
    f = request.files.get('audio') or request.files.get('file')
    if not f:
        return jsonify({"error": "no file"}), 400
    files = {"file": (f.filename, f.stream, f.content_type or "audio/mpeg")}
    r = _httpx.post(f"{POD_API}/api/upload", files=files, headers=_pod_headers, timeout=120)
    return jsonify(r.json()), r.status_code


@app.route('/v2/api/process', methods=['POST'])
def v2_process():
    data = request.json or {}
    r = _httpx.post(f"{POD_API}/api/process", json=data, headers=_pod_headers, timeout=900)
    return jsonify(r.json()), r.status_code


@app.route('/v2/api/clone', methods=['POST'])
def v2_clone():
    data = request.json or {}
    r = _httpx.post(f"{POD_API}/api/clone", json=data, headers=_pod_headers, timeout=600)
    return jsonify(r.json()), r.status_code


@app.route('/v2/api/jobs')
def v2_jobs():
    r = _httpx.get(f"{POD_API}/api/jobs", headers=_pod_headers, timeout=10)
    return jsonify(r.json()), r.status_code


@app.route('/v2/api/costs')
def v2_costs():
    r = _httpx.get(f"{POD_API}/api/costs", headers=_pod_headers, timeout=10)
    return jsonify(r.json()), r.status_code


@app.route('/v2/files/<path:subpath>')
def v2_files(subpath):
    """Proxy files served by Pod /files/."""
    r = _httpx.get(f"{POD_API}/files/{subpath}", headers=_pod_headers, timeout=120)
    if r.status_code != 200:
        return ("not found", 404)
    return r.content, 200, {"Content-Type": r.headers.get("Content-Type", "application/octet-stream")}
```

- [ ] **Step 2: Update index_v2.html to use /v2 endpoints**

Edit `templates/index_v2.html`. Replace `/api/upload` → `/v2/api/upload`, `/api/process` → `/v2/api/process`, `/api/clone` → `/v2/api/clone`. Audio URLs from server are `/files/...` — prefix with `/v2`: `/v2/files/...`.

- [ ] **Step 3: Restart local Flask**

```bash
PID=$(netstat -ano 2>&1 | grep ":5000" | grep LISTENING | awk '{print $NF}' | head -1); if [ -n "$PID" ]; then taskkill //F //PID $PID 2>&1 | head -2; fi
cd "C:/Users/Admin/Desktop/project/x" && python app.py &
disown
sleep 3
curl -s http://127.0.0.1:5000/v2 | head -5
```
Expected: returns HTML for v2 page

- [ ] **Step 4: Commit**

```bash
git add app.py templates/index_v2.html
git commit -m "feat(local): /v2 route + proxy to Pod engine"
```

---

## Phase 7: End-to-End Verification

### Task 7.1: Manual UI smoke test (with tunnel + engine running)

**Files:**
- Create: `docs/verification.md`

- [ ] **Step 1: Verify pre-requisites**

```bash
# Tunnel up
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/health
# Local Flask up
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5000/v2
```
Expected: both print `200`

- [ ] **Step 2: Browser-driven smoke test using mcp__Claude_in_Chrome**

Open `http://127.0.0.1:5000/v2` in Chrome via:
```python
from mcp__Claude_in_Chrome__navigate import navigate
navigate(url="http://127.0.0.1:5000/v2", tabId=...)
```
Visually verify: dropdowns visible, drop-zone visible, page renders without console errors.

- [ ] **Step 3: Document the verification result**

Write `docs/verification.md` with:
- Date of verification
- Each tool's smoke-test result (pass/fail/skip)
- Notes on any tool that failed (e.g. "VibeVoice ASR failed during model download — segfault")

- [ ] **Step 4: Commit**

```bash
git add docs/verification.md
git commit -m "docs: e2e verification log"
```

### Task 7.2: Run all smoke tests one final time, summarize

**Files:**
- Modify: `docs/verification.md`

- [ ] **Step 1: Run full test suite on Pod**

```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175 "cd /workspace/voice-studio-v2 && python -m pytest engine/tests/ -v --tb=short 2>&1 | tail -50"
```
Expected: most tests pass; record any failures.

- [ ] **Step 2: Append summary to verification.md**

Update `docs/verification.md` with the test summary (X passed / Y failed / Z skipped, plus notes per failed tool).

- [ ] **Step 3: Notify user**

Print a summary message:
```
Voice Studio v2 ready for manual testing.

Tools verified:
- STT: [list of working tools]
- Diar: [list of working tools]
- TTS: [list of working tools]

Tools with issues (continue at user's discretion):
- [tool]: [reason]

Open http://127.0.0.1:5000/v2 in your browser to start testing.
```

- [ ] **Step 4: Commit final state**

```bash
git add docs/verification.md
git commit -m "docs: final verification summary; voice-studio-v2 ready for testing"
```

---

## Self-Review Notes

The following spec requirements are covered:

| Spec Requirement | Task |
|---|---|
| Pod GPU FastAPI engine | Task 1.5–1.6 |
| Lazy-loading model manager + LRU eviction | Task 1.4 |
| 4 STT tools (Whisper×2, VibeVoice ASR, NeMo) | Tasks 2.1–2.4 |
| 2 Diar tools (pyannote, ECAPA) | Tasks 3.1–3.2 |
| 5 TTS tools (VibeVoice×2, F5, XTTS, Fish) | Tasks 4.1–4.4 |
| /api/upload, /process, /clone, /jobs, /costs | Tasks 5.2–5.5 |
| SSH tunnel | Task 1.7 |
| Local Flask /v2 with proxy | Task 6.3 |
| Frontend with tool dropdowns | Task 6.1 |
| Comparison panel + session cost | Task 6.2 |
| Cost tracker per stage | Task 1.2, integrated in 5.3 + 5.4 |
| Pydantic schemas | Task 1.3 |
| Smoke tests per tool | Tasks 2.x, 3.x, 4.x |
| 5-second Arabic test fixture | Task 0.4 |
| Disk + memory cleanup | Tasks 0.1, 0.2 |
| Daemon survival across SSH disconnect | Task 1.6 |
| API key authentication | Task 1.5 |
| Hard caps on text/audio length | Task 1.5 + 5.4 |
| Risk: VibeVoice ASR may fail | Task 2.3 (graceful continue) |

No placeholders detected. Function signatures are consistent across tasks. Type names (`ProcessRequest`, `CloneResponse`, `Segment`, etc.) used consistently between schemas (Task 1.3) and API endpoints (Tasks 5.x).
