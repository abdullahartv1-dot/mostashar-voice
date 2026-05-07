"""TTS smoke tests — verifies each model can clone a 5s reference + generate Arabic."""
import pytest
from pathlib import Path

REFERENCE = Path("/workspace/voice-studio-v2/engine/tests/fixtures/short_arabic.wav")
TEST_TEXT = "مرحباً، هذا اختبار."


@pytest.mark.skipif(not REFERENCE.exists(), reason="fixture missing")
def test_vibevoice_15b_clones_arabic():
    from engine.services.tts_vibevoice import clone_vibevoice
    out_path = clone_vibevoice(
        text=TEST_TEXT,
        reference_audio=str(REFERENCE),
        model="VibeVoice-1.5B",
    )
    assert Path(out_path).exists()
    import soundfile as sf
    info = sf.info(out_path)
    assert info.duration > 0.5


@pytest.mark.skipif(not REFERENCE.exists(), reason="fixture missing")
def test_f5_tts_clones_arabic():
    from engine.services.tts_f5 import clone_f5
    out_path = clone_f5(
        text=TEST_TEXT,
        reference_audio=str(REFERENCE),
        reference_text="هذا صوت مرجعي.",
    )
    assert Path(out_path).exists()
    import soundfile as sf
    assert sf.info(out_path).duration > 0.5
