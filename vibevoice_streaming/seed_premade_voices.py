"""
Seed the voice library with high-quality premade Arabic voices.

Uses Microsoft Azure Neural voices via free `edge-tts` library to generate
~20-second reference samples for each accent, then registers them as voice
profiles in /workspace/refs/voices/.

After this runs, GET /v1/voices will list these alongside `default` and any
user-cloned voices.
"""
import os
import json
import time
import asyncio
import edge_tts
import soundfile as sf
import librosa
import numpy as np

VOICES_DIR = os.environ.get("MV_VOICES_DIR", "/workspace/refs/voices")
SAMPLE_RATE = 24000
os.makedirs(VOICES_DIR, exist_ok=True)

# Voices to seed: Microsoft Azure Neural voices for Arabic
# (voice_id_local, edge_tts_short_name, display_name, language)
PREMADE = [
    # Saudi Arabian
    ("hamed_saudi",     "ar-SA-HamedNeural",   "حامد - سعودي",     "ar"),
    ("zariyah_saudi",   "ar-SA-ZariyahNeural", "زرياء - سعودية",    "ar"),
    # Egyptian
    ("shakir_egypt",    "ar-EG-ShakirNeural",  "شاكر - مصري",      "ar"),
    ("salma_egypt",     "ar-EG-SalmaNeural",   "سلمى - مصرية",      "ar"),
    # Lebanese
    ("rami_lebanon",    "ar-LB-RamiNeural",    "رامي - لبناني",     "ar"),
    ("layla_lebanon",   "ar-LB-LaylaNeural",   "ليلى - لبنانية",     "ar"),
    # Emirati
    ("hamdan_uae",      "ar-AE-HamdanNeural",  "حمدان - إماراتي",   "ar"),
    ("fatima_uae",      "ar-AE-FatimaNeural",  "فاطمة - إماراتية",  "ar"),
    # Kuwaiti
    ("fahed_kuwait",    "ar-KW-FahedNeural",   "فهد - كويتي",       "ar"),
    ("noura_kuwait",    "ar-KW-NouraNeural",   "نورة - كويتية",     "ar"),
]

# Reference text — used to generate the audio sample. Mix of formal + conversational.
REFERENCE_TEXT = (
    "السلام عليكم ورحمة الله وبركاته. مرحباً بكم في منصة مستشار. "
    "نحن نقدم خدمات الاستشارات والتواصل الصوتي بأعلى جودة. "
    "أتمنى أن تكون تجربتكم معنا مميزة ومفيدة. شكراً لكم على ثقتكم بنا."
)


async def synthesize_voice(short_name: str, output_wav: str) -> bool:
    """Generate ~20s of speech using edge-tts and save as WAV at 24kHz."""
    mp3_tmp = output_wav + ".tmp.mp3"
    try:
        comm = edge_tts.Communicate(REFERENCE_TEXT, short_name)
        await comm.save(mp3_tmp)
        # Re-encode to mono 24kHz WAV
        audio, _sr = librosa.load(mp3_tmp, sr=SAMPLE_RATE, mono=True)
        # cap at 30s
        if len(audio) > 30 * SAMPLE_RATE:
            audio = audio[: 30 * SAMPLE_RATE]
        sf.write(output_wav, audio, SAMPLE_RATE)
        return True
    except Exception as e:
        print(f"  ✗ {short_name}: {e}")
        return False
    finally:
        if os.path.exists(mp3_tmp):
            os.remove(mp3_tmp)


async def main():
    print(f"=== Seeding {len(PREMADE)} premade voices to {VOICES_DIR} ===\n")
    success = 0
    for voice_id, short_name, display_name, lang in PREMADE:
        wav_path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
        meta_path = os.path.join(VOICES_DIR, f"{voice_id}.json")

        if os.path.exists(wav_path):
            print(f"  ⊙ {voice_id}: already exists, skipping")
            success += 1
            continue

        print(f"  → {voice_id} ({short_name}) ...", end="", flush=True)
        ok = await synthesize_voice(short_name, wav_path)
        if not ok:
            continue

        info = sf.info(wav_path)
        meta = {
            "voice_id": voice_id,
            "name": display_name,
            "language": lang,
            "created_at": int(time.time()),
            "dur_s": round(info.duration, 2),
            "source": f"edge-tts:{short_name}",
            "premade": True,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f" ✓ {info.duration:.1f}s")
        success += 1

    print(f"\n=== {success}/{len(PREMADE)} premade voices ready ===")
    print("Run server (or restart) — they auto-load from disk.")


if __name__ == "__main__":
    asyncio.run(main())
