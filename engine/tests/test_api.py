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
    assert body["cost_usd"] >= 0


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
