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
    speakers_detected: Optional[int] = None


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
