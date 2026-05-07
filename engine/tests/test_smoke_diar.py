"""Diarization smoke tests."""
import os
import pytest
from pathlib import Path

FIXTURE = Path("/workspace/voice-studio-v2/engine/tests/fixtures/short_arabic.wav")


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="HF_TOKEN not set")
def test_pyannote_diarizes_short_clip():
    from engine.services.diar_pyannote import diarize_pyannote
    from engine.services.stt_whisper import transcribe_whisper
    stt = transcribe_whisper(str(FIXTURE), model_size="large-v3")
    result = diarize_pyannote(str(FIXTURE), segments=stt["segments"])
    assert "segments" in result
    for seg in result["segments"]:
        assert "speaker" in seg


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
def test_ecapa_diarizes_short_clip():
    from engine.services.diar_ecapa import diarize_ecapa
    from engine.services.stt_whisper import transcribe_whisper
    stt = transcribe_whisper(str(FIXTURE), model_size="large-v3")
    result = diarize_ecapa(str(FIXTURE), segments=stt["segments"], n_speakers=2)
    assert "segments" in result
    for seg in result["segments"]:
        assert "speaker" in seg


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
def test_ecapa_auto_detect_speakers():
    from engine.services.diar_ecapa import diarize_ecapa
    from engine.services.stt_whisper import transcribe_whisper
    stt = transcribe_whisper(str(FIXTURE), model_size="large-v3")
    result = diarize_ecapa(str(FIXTURE), segments=stt["segments"], n_speakers=None)
    assert "speakers_detected" in result
    assert result["speakers_detected"] >= 1
    # All segments should have a speaker label
    for seg in result["segments"]:
        assert "speaker" in seg
