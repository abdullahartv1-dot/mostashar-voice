"""In-memory + disk-backed job state."""
import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

from . import config

_JOBS: Dict[str, Dict[str, Any]] = {}


def create_job() -> str:
    job_id = uuid.uuid4().hex[:8]
    _JOBS[job_id] = {"id": job_id, "status": "pending"}
    return job_id


def _job_file(job_id: str) -> Path:
    return config.JOBS_DIR / job_id / "job.json"


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Return job from memory; if missing, try to load from disk."""
    if job_id in _JOBS:
        return _JOBS[job_id]
    fp = _job_file(job_id)
    if fp.exists():
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            _JOBS[job_id] = data
            return data
        except Exception:
            return None
    return None


def update_job(job_id: str, **fields) -> None:
    if job_id in _JOBS:
        _JOBS[job_id].update(fields)
        _persist(job_id)


def save_job(job_id: str, data: Dict[str, Any]) -> None:
    """Write the full job data to memory + disk."""
    _JOBS[job_id] = data
    _persist(job_id)


def _persist(job_id: str) -> None:
    fp = _job_file(job_id)
    fp.parent.mkdir(parents=True, exist_ok=True)
    try:
        fp.write_text(json.dumps(_JOBS[job_id], ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def all_jobs() -> Dict[str, Dict[str, Any]]:
    """List jobs from memory + on-disk discoveries."""
    out = dict(_JOBS)
    for child in config.JOBS_DIR.iterdir():
        if not child.is_dir():
            continue
        if child.name in out:
            continue
        f = child / "job.json"
        if f.exists():
            try:
                out[child.name] = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                pass
    return out
