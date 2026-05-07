"""ECAPA-TDNN + agglomerative clustering diarization (no HF token required)."""
import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np
import soundfile as sf
import torch

# Pod cuDNN init is broken (CUDNN_STATUS_NOT_INITIALIZED on conv ops).
# Disable cuDNN — falls back to native CUDA kernels (slower but works).
torch.backends.cudnn.enabled = False

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
