"""Verify Large's encoder is identical to Realtime, then transplant Large encoder weights."""
import os, sys, time
os.environ['HF_HOME'] = '/workspace/hf-cache'

import torch
import vibevoice
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference

print("[1/3] loading VibeVoice-Large (source)...", flush=True)
m_large = VibeVoiceForConditionalGenerationInference.from_pretrained(
    "aoi-ot/VibeVoice-Large",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
encoder_large_sd = m_large.model.acoustic_tokenizer.encoder.state_dict()
n_large = sum(v.numel() for v in encoder_large_sd.values())
print(f"  Large encoder: {len(encoder_large_sd)} tensors, {n_large:,} params")

print("\n[2/3] loading VibeVoice-Realtime (target)...", flush=True)
m_rt = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
    "microsoft/VibeVoice-Realtime-0.5B",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
target_sd = m_rt.model.acoustic_tokenizer.encoder.state_dict()
n_rt = sum(v.numel() for v in target_sd.values())
print(f"  Realtime encoder: {len(target_sd)} tensors, {n_rt:,} params")

print(f"\n[3/3] checking compatibility...")
mismatched = []
for k in target_sd:
    if k not in encoder_large_sd:
        mismatched.append(f"{k} MISSING in Large")
    elif encoder_large_sd[k].shape != target_sd[k].shape:
        mismatched.append(f"{k} SHAPE MISMATCH: Large {tuple(encoder_large_sd[k].shape)} vs Realtime {tuple(target_sd[k].shape)}")

if mismatched:
    print(f"  ⚠️ {len(mismatched)} mismatches:")
    for m in mismatched[:5]:
        print(f"    {m}")
    if len(mismatched) > 5:
        print(f"    ... and {len(mismatched)-5} more")
    sys.exit(1)

print(f"  ✅ Large and Realtime encoders are IDENTICAL in architecture")

# Transplant
m_rt.model.acoustic_tokenizer.encoder.load_state_dict(encoder_large_sd, strict=True)
print(f"  ✅ transplanted Large encoder weights into Realtime")

# Compare to existing patched (1.5B-encoder) Realtime — are weights different?
import os.path as p
existing = "/workspace/vv-realtime-patched"
if p.exists(p.join(existing, "config.json")):
    print(f"\n  comparing Large encoder vs 1.5B encoder (already in vv-realtime-patched):")
    m_old = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        existing, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    old_sd = m_old.model.acoustic_tokenizer.encoder.state_dict()
    diff_count = 0
    max_diff = 0
    for k in old_sd:
        if not torch.equal(old_sd[k], encoder_large_sd[k]):
            diff_count += 1
            d = (old_sd[k].float() - encoder_large_sd[k].float()).abs().max().item()
            max_diff = max(max_diff, d)
    print(f"    {diff_count}/{len(old_sd)} tensors differ between 1.5B and Large encoders, max abs diff: {max_diff:.4f}")
    del m_old
    import gc; gc.collect()

# Save patched Realtime model with Large encoder
SAVE_DIR = "/workspace/vv-realtime-large-encoder"
print(f"\n[saving to {SAVE_DIR}]")
m_rt.save_pretrained(SAVE_DIR)
print(f"✅ saved {SAVE_DIR}")
