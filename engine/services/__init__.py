"""Tool name → handler registry. Adds a callable layer so main.py is decoupled from concrete services."""
from typing import Callable, Dict


def _whisper_large(audio_path):
    from .stt_whisper import transcribe_whisper
    return transcribe_whisper(audio_path, model_size="large-v3")


def _whisper_turbo(audio_path):
    from .stt_whisper import transcribe_whisper
    return transcribe_whisper(audio_path, model_size="large-v3-turbo")


def _vibevoice_asr(audio_path):
    from .stt_vibevoice import transcribe_vibevoice
    return transcribe_vibevoice(audio_path)


def _nemo_canary(audio_path):
    from .stt_nemo import transcribe_canary
    return transcribe_canary(audio_path)


def _pyannote(audio_path, segments):
    from .diar_pyannote import diarize_pyannote
    return diarize_pyannote(audio_path, segments)


def _ecapa(audio_path, segments, n_speakers=2):
    from .diar_ecapa import diarize_ecapa
    return diarize_ecapa(audio_path, segments, n_speakers)


def _vibevoice_15b(text, ref, **kw):
    from .tts_vibevoice import clone_vibevoice
    return clone_vibevoice(text, ref, model="VibeVoice-1.5B", **kw)


def _vibevoice_large(text, ref, **kw):
    from .tts_vibevoice import clone_vibevoice
    return clone_vibevoice(text, ref, model="VibeVoice-Large", **kw)


def _f5(text, ref, **kw):
    from .tts_f5 import clone_f5
    return clone_f5(text, ref, **kw)


def _xtts(text, ref, **kw):
    from .tts_xtts import clone_xtts
    return clone_xtts(text, ref, **kw)


STT_HANDLERS: Dict[str, Callable] = {
    "whisper-large-v3": _whisper_large,
    "whisper-turbo": _whisper_turbo,
    "vibevoice-asr": _vibevoice_asr,
    "nemo-canary": _nemo_canary,
}

DIAR_HANDLERS: Dict[str, Callable] = {
    "pyannote-3.1": _pyannote,
    "ecapa-tdnn": _ecapa,
}

TTS_HANDLERS: Dict[str, Callable] = {
    "vibevoice-1.5b": _vibevoice_15b,
    "vibevoice-large": _vibevoice_large,
    "f5-tts": _f5,
    "xtts-v2": _xtts,
}
