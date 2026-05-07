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
