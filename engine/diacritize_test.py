"""Test Qwen 2.5-7B-Instruct for context-aware Arabic diacritization."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ["HF_HUB_CACHE"] = "/workspace/voice-studio-v2/engine/vendor/vibe-voice-custom-voices/vibevoice"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading {MODEL}…")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="eager",
).eval()
print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

text_in = "مرحبا بكم في تجربة استنساخ الصوت العربي من منصة مستشار. هذا النص الذي تسمعونه الان مولد بالكامل بواسطة الذكاء الاصطناعي. لكنه يستخدم بصمة صوت المتحدث الاصلي."

system = "You are an expert Arabic linguist. Add full diacritics (tashkeel: fatha ◌َ, damma ◌ُ, kasra ◌ِ, sukun ◌ْ, shadda ◌ّ, tanwin) to every Arabic word based on context and grammar. Return ONLY the fully-vocalized Arabic text, nothing else."
user = f"""Examples of correct vocalization:

Input: هذا النص مولد بالحاسوب
Output: هَذَا النَّصُّ مُوَلَّدٌ بِالْحَاسُوبِ

Input: احتفلنا بمولد النبي
Output: اِحْتَفَلْنَا بِمَوْلِدِ النَّبِيِّ

Input: الذكاء الاصطناعي تقنية حديثة
Output: الذَّكَاءُ الاِصْطِنَاعِيُّ تِقْنِيَةٌ حَدِيثَةٌ

Now vocalize this text fully:

Input: {text_in}
Output:"""

messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": user},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("\n=== generating ===")
inputs = tok(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=400, do_sample=False, pad_token_id=tok.eos_token_id)
gen = tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
print("\n=== Qwen 7B-Instruct output ===")
print(gen)
print()
print("=== Manual (winning) reference ===")
print("مُوَلَّد بالكامل بواسطة الذكاء الاصطناعي")
