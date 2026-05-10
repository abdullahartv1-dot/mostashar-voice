"""
Pre-process Arabic text for VibeVoice TTS.

Strategy: Mishkal for base diacritization + curated overrides for common
ambiguous words that Mishkal/VibeVoice mispronounce.

Usage:
    from engine.arabic_normalize import prepare_for_tts
    clean = prepare_for_tts("النص مولد بالذكاء الاصطناعي")
"""
import re
from typing import Optional

# Known truly-ambiguous words that VibeVoice mispronounces by default.
# RULE: be MINIMAL. Each diacritic added = different generation path = subtle voice drift.
# Only fix words that VibeVoice gets wrong with the bare form.
OVERRIDES = [
    # مولد in technical context defaults to wrong reading "مَوْلِد" (birth);
    # force "مُوَلَّد" (generated) when context is clearly technical.
    ("مولد بالكامل",  "مُوَلَّد بالكامل"),
    ("مولد بواسطة",   "مُوَلَّد بواسطة"),
    ("مولد بالحاسوب", "مُوَلَّد بالحاسوب"),
    ("مولد آلياً",    "مُوَلَّد آلياً"),
    ("مولد آليا",     "مُوَلَّد آليا"),
    ("صوت مولد",      "صوت مُوَلَّد"),
    ("نص مولد",       "نص مُوَلَّد"),

    # Hamza/taa-marbuta normalization — same pronunciation but VibeVoice reads them
    # differently (no extra diacritics added; same number of tokens).
    ("الان",   "الآن"),
    ("الاصلي", "الأصلي"),
    ("بصمه",   "بصمة"),
    ("منصه",   "منصة"),
    # NOTE: "الذكاء الاصطناعي" is intentionally NOT diacritized — VibeVoice reads
    # it correctly bare, and adding tashkeel changes voice timbre due to different
    # token sequence. Keep this list short and surgical.
]


def prepare_for_tts(text: str, use_mishkal: bool = True) -> str:
    """Normalize an Arabic input string for TTS. Returns text ready to send to VibeVoice."""
    out = text

    # 1) Run Mishkal first (broad-stroke diacritization)
    if use_mishkal:
        try:
            from mishkal.tashkeel import TashkeelClass
            vocalizer = TashkeelClass()
            out = vocalizer.tashkeel(out)
        except ImportError:
            pass

    # 2) Strip diacritics from the words we're about to override (so .replace works)
    DIACRITICS = "ًٌٍَُِّْٰٕٓٔ"
    def _strip_diacritics(s):
        return re.sub('[' + DIACRITICS + ']', '', s)
    stripped = _strip_diacritics(out)
    # Build mapping in stripped space
    final = stripped
    for src, dst in OVERRIDES:
        final = final.replace(src, dst)

    # Now reapply Mishkal's diacritics to UNCHANGED segments, keep our overrides as-is.
    # Pragmatic: if the stripped form == original stripped, return Mishkal's diacritized.
    # If overrides changed it, return our hybrid (our overrides + raw rest).
    out = final if final != stripped else out

    # 3) Cleanup whitespace + sentence boundaries
    out = re.sub(r'\.(?!\s)', '. ', out)
    out = re.sub(r'\s+', ' ', out).strip()
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    samples = [
        "مرحبا بكم في تجربة استنساخ الصوت العربي من منصة مستشار. هذا النص الذي تسمعونه الان مولد بالكامل بواسطة الذكاء الاصطناعي. لكنه يستخدم بصمة صوت المتحدث الاصلي.",
    ]
    for s in samples:
        print("IN :", s)
        print("OUT:", prepare_for_tts(s, use_mishkal=False))
        print("OUT (+mishkal):", prepare_for_tts(s, use_mishkal=True))
        print()
