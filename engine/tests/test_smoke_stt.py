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


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
def test_whisper_turbo_transcribes_arabic():
    from engine.services.stt_whisper import transcribe_whisper
    result = transcribe_whisper(str(FIXTURE), model_size="large-v3-turbo")
    assert result["duration"] == pytest.approx(5.0, abs=0.5)
    assert len(result["segments"]) >= 1


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
