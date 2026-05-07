"""FastAPI engine entrypoint."""
import logging
from pathlib import Path
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
