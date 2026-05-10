"""Selective Arabic diacritization via Qwen 2.5-7B-Instruct.

Strategy:
- Don't diacritize the whole sentence (changes voice timbre in TTS).
- Ask Qwen to identify ONLY ambiguous words and return them as a JSON list.
- Caller replaces those words in the original text, leaves the rest untouched.

This produces TTS-friendly text: minimal token-sequence drift while fixing
mispronunciations on the words that actually need it.
"""
import json
import logging
import os
import re
import time
from typing import Optional, List, Dict
import torch

logger = logging.getLogger(__name__)

_CACHE = "/workspace/voice-studio-v2/engine/vendor/vibe-voice-custom-voices/vibevoice"
_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

_model = None
_tok = None


def _load() -> None:
    global _model, _tok
    if _model is not None:
        return
    os.environ["HF_HUB_CACHE"] = _CACHE
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading {_MODEL_NAME} for diacritization (~15GB VRAM)…")
    _tok = AutoTokenizer.from_pretrained(_MODEL_NAME, cache_dir=_CACHE)
    _model = AutoModelForCausalLM.from_pretrained(
        _MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",  # cuDNN-frontend SDPA breaks on this Pod
        cache_dir=_CACHE,
    ).eval()


SYSTEM_PROMPT = """أنت خبير في علم اللسانيات العربية، تساعد في تحضير النصوص لمحركات تحويل النص إلى صوت (TTS).

مهمتك: العثور على كل كلمة عربية يحتاج محرك TTS إلى تشكيل **هويتها الداخلية** (وزنها الصرفي) للنطق الصحيح. هذا يشمل:
1. **الكلمات متعددة القراءات**: مثل "مولد" (مَوْلِد ↔ مُوَلَّد)، "علم" (عَلِمَ ↔ عِلْم)، "كتب" (كَتَبَ ↔ كُتُب).
2. **الأسماء التي قد تُنطق خطأ**: مثل "مستشار"، "استنساخ"، "بصمة".
3. **الكلمات الفنية**: التي قد يُقرأها النموذج بلهجة خاطئة.

⚠️ **قاعدة حرجة**: شكِّل الكلمة بدون **علامة الإعراب النهائية** (آخر حرف). أي:
- ❌ "اسْتِنْسَاخَ" (بفتحة على خ — قد تكون خاطئة حسب موقع الكلمة في الجملة)
- ❌ "اسْتِنْسَاخِ" (بكسرة على خ — قد تكون خاطئة كذلك)
- ✅ "اسْتِنْسَاخ" (بدون حركة على خ — يُتركها للسياق)

هذا لأن إعراب آخر الكلمة يعتمد على بنية الجملة، وإذا أخطأ النموذج فيها أنتجت TTS كلاماً غير طبيعي. التشكيل **الداخلي** فقط يضمن نطق هوية الكلمة الصحيح.

تجاهل الكلمات الوظيفية الواضحة (في، هذا، من، إلى، الذي، إلخ).

أرجع JSON صالحاً بهذا الشكل بالضبط — بدون أي شرح:
{"replacements": [{"original": "X", "diacritized": "تشكيل_داخلي_X"}, ...]}"""


FEW_SHOT_USER = """شكِّل الكلمات التي تحتاج تشكيلاً داخلياً للـ TTS في:
"مرحبا، هذا النص مولد بالكامل بواسطة الذكاء الاصطناعي. يستخدم بصمة صوت المتحدث الأصلي من منصة مستشار."
"""

FEW_SHOT_ASSISTANT = """{"replacements": [{"original": "مولد", "diacritized": "مُوَلَّد"}, {"original": "الذكاء", "diacritized": "الذَّكاء"}, {"original": "الاصطناعي", "diacritized": "الاِصْطِنَاعِيّ"}, {"original": "بصمة", "diacritized": "بَصْمَة"}, {"original": "المتحدث", "diacritized": "المُتَحَدِّث"}, {"original": "الأصلي", "diacritized": "الأَصْلِيّ"}, {"original": "مستشار", "diacritized": "مُسْتَشَار"}]}"""


def diacritize_ambiguous(text: str, max_new_tokens: int = 400) -> List[Dict[str, str]]:
    """Return [{original, diacritized}, ...] for ambiguous words in `text`.

    The caller is expected to do `.replace(original, diacritized)` on the input.
    """
    _load()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
        {"role": "user", "content": f"شكِّل الكلمات الملتبسة في:\n\"{text}\""},
    ]
    prompt = _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tok(prompt, return_tensors="pt").to(_model.device)

    t0 = time.time()
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_tok.eos_token_id,
        )
    elapsed = time.time() - t0
    raw = _tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    logger.info(f"Qwen diacritize {elapsed:.1f}s: {raw[:200]}")

    # Find balanced top-level JSON object: scan { ... } with brace counting
    start = raw.find('{')
    if start < 0:
        logger.warning(f"No JSON object in Qwen output: {raw}")
        return []
    depth = 0
    end = -1
    for i, c in enumerate(raw[start:], start):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        logger.warning(f"Unbalanced braces: {raw}")
        return []
    try:
        data = json.loads(raw[start:end])
        return data.get("replacements", [])
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}\nslice: {raw[start:end]}")
        return []


_DIACRITICS = "ًٌٍَُِّْٰٕٓٔ"


def _strip_diacritics(s: str) -> str:
    return re.sub('[' + _DIACRITICS + ']', '', s)


def _canonical(s: str) -> str:
    """Strip diacritics + normalize hamza variants/taa-marbuta/ya for comparison.
    Used to validate Qwen's diacritized form is the SAME word as the original
    (rejects hallucinations like ذكاء→ذهاء) while ACCEPTING orthographic
    fixes like الاصلي→الأصلي."""
    s = _strip_diacritics(s)
    # Alef variants → plain alef
    s = re.sub('[إأآا]', 'ا', s)
    # Taa marbuta ↔ haa (treat as same)
    s = s.replace('ة', 'ه')
    # Ya variants
    s = s.replace('ى', 'ي')
    return s


def normalize_for_tts(text: str) -> str:
    """One-shot: detect ambiguous words and apply replacements to `text`.

    Validates each replacement: the diacritized form, with diacritics stripped,
    must equal the original — otherwise it's a hallucination (e.g. "ذكاء" →
    "ذهاء") and we reject it.
    """
    repls = diacritize_ambiguous(text)
    out = text
    for r in repls:
        orig = r.get("original")
        diac = r.get("diacritized")
        if not (orig and diac and orig != diac):
            continue
        # Validate: canonical form (no diacritics + normalized hamza/ya/ta) must match.
        # Accepts orthographic fixes (الاصلي→الأَصْلِيّ); rejects hallucinations (ذكاء→ذهاء).
        if _canonical(diac) != _canonical(orig):
            logger.warning(f"REJECTED hallucination: {orig!r} → {diac!r}")
            continue
        # Whole-word replace — avoid e.g. "نص" matching inside "منصة".
        # Arabic word boundaries: any non-Arabic-letter character (or string edge).
        # If the word doesn't start with ال, also accept ال-prefixed form in text.
        # Same Qwen "diacritized" form gets used either way (for definite article
        # cases the listener context handles ال).
        candidates = [(orig, diac)]
        if not orig.startswith('ال') and not diac.startswith('ال'):
            candidates.append(('ال' + orig, 'ال' + diac))
        replaced = False
        for src, dst in candidates:
            pattern = r'(?<![؀-ۿ])' + re.escape(src) + r'(?![؀-ۿ])'
            new_out, n = re.subn(pattern, dst, out)
            if n > 0:
                out = new_out
                replaced = True
                break
        if not replaced:
            logger.info(f"NOT FOUND in text (skipping): {orig!r}")
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    text = "مرحبا بكم في تجربة استنساخ الصوت العربي من منصة مستشار. هذا النص الذي تسمعونه الان مولد بالكامل بواسطة الذكاء الاصطناعي. لكنه يستخدم بصمة صوت المتحدث الاصلي."
    repls = diacritize_ambiguous(text)
    print("=== ambiguous words detected ===")
    for r in repls:
        print(f"  {r.get('original')!r} → {r.get('diacritized')!r}")
    print()
    out = normalize_for_tts(text)
    print("=== final TTS-ready text ===")
    print(out)
