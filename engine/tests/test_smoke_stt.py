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
