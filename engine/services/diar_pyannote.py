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
