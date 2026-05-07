"""ECAPA-TDNN + agglomerative clustering diarization (no HF token required).

Auto-detects speaker count via silhouette score when n_speakers is None.
"""
import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np
import soundfile as sf
import torch

# Disable cuDNN — Pod has cuDNN init issue
torch.backends.cudnn.enabled = False

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
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


def _pick_best_k(X: np.ndarray, max_k: int = 5) -> int:
    """Pick k in [2, min(max_k, n-1)] with highest silhouette score on cosine.

    Falls back to 1 if there are too few embeddings to cluster.
    """
    n = len(X)
    if n < 2:
        return 1
    upper = min(max_k, n - 1)
    if upper < 2:
        return 1
    best_k, best_score = 2, -1.0
    for k in range(2, upper + 1):
        try:
            labels = AgglomerativeClustering(
                n_clusters=k, metric="cosine", linkage="average"
            ).fit_predict(X)
            # Need at least 2 distinct labels to score
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels, metric="cosine")
            if score > best_score:
                best_k, best_score = k, score
        except Exception as e:
            logger.warning(f"Silhouette eval failed at k={k}: {e}")
    return best_k


def diarize_ecapa(
    audio_path: str,
    segments: List[Dict[str, Any]],
    n_speakers: Optional[int] = None,
) -> Dict[str, Any]:
    """ECAPA embeddings per segment + agglomerative clustering.

    If `n_speakers` is None, picks the best k via silhouette score (k in 2..5).
    """
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
        return {
            "segments": out,
            "speakers_detected": 1,
            "elapsed_sec": round(time.time() - t0, 2),
        }

    X = np.stack(embeddings)
    n_clusters = n_speakers if n_speakers is not None else _pick_best_k(X, max_k=5)
    n_clusters = max(1, min(n_clusters, len(embeddings)))

    if n_clusters >= 2:
        labels = AgglomerativeClustering(
            n_clusters=n_clusters, metric="cosine", linkage="average"
        ).fit_predict(X)
    else:
        labels = [0] * len(embeddings)

    out = [dict(s) for s in segments]
    last_label = 0
    li = 0
    for i, seg in enumerate(out):
        if li < len(valid_idx) and valid_idx[li] == i:
            last_label = int(labels[li])
            li += 1
        seg["speaker"] = f"SPEAKER_{last_label}"

    return {
        "segments": out,
        "speakers_detected": int(n_clusters),
        "elapsed_sec": round(time.time() - t0, 2),
    }
