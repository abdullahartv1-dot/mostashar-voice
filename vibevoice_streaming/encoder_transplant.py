"""ENCODER TRANSPLANT EXPERIMENT
- VibeVoice-1.5B has trained acoustic_tokenizer.encoder (344M params)
- VibeVoice-Realtime-0.5B has the SAME architecture but RANDOM weights
- Verified: same shape, same param count
- Plan: copy encoder weights from 1.5B → Realtime, then use Realtime for cloning + streaming
"""
import os, sys, time
os.environ['HF_HOME'] = '/workspace/hf-cache'

import torch
import vibevoice
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference

print("[1/3] loading VibeVoice-1.5B (source of encoder weights)...", flush=True)
m_15b = VibeVoiceForConditionalGenerationInference.from_pretrained(
    "microsoft/VibeVoice-1.5B",
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # cpu first to avoid OOM with 2 models loaded
)
encoder_sd = m_15b.model.acoustic_tokenizer.encoder.state_dict()
print(f"  extracted encoder state dict: {len(encoder_sd)} tensors")

print("\n[2/3] loading VibeVoice-Realtime-0.5B (target — encoder has random weights)...", flush=True)
m_rt = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
    "microsoft/VibeVoice-Realtime-0.5B",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
target_sd = m_rt.model.acoustic_tokenizer.encoder.state_dict()
print(f"  target encoder state dict: {len(target_sd)} tensors")

print("\n[3/3] transplanting encoder weights from 1.5B → Realtime...")
# Verify shapes match
mismatched = []
for k in target_sd:
    if k not in encoder_sd:
        mismatched.append(f"{k} MISSING in source")
    elif encoder_sd[k].shape != target_sd[k].shape:
        mismatched.append(f"{k} SHAPE MISMATCH: source {encoder_sd[k].shape} vs target {target_sd[k].shape}")

if mismatched:
    print(f"  ⚠️ {len(mismatched)} mismatches:")
    for m in mismatched[:5]:
        print(f"    {m}")
    if len(mismatched) > 5:
        print(f"    ... and {len(mismatched)-5} more")
    print("\n  ABORT — incompatible encoders")
    sys.exit(1)

# Load state dict
m_rt.model.acoustic_tokenizer.encoder.load_state_dict(encoder_sd, strict=True)
print(f"  ✅ transplanted {len(encoder_sd)} tensors successfully")

# Free 1.5B from memory
del m_15b
import gc; gc.collect()

# Save patched Realtime model
SAVE_DIR = "/workspace/vv-realtime-patched"
print(f"\n[saving patched model to {SAVE_DIR}]")
m_rt.save_pretrained(SAVE_DIR)
print(f"✅ saved {SAVE_DIR}")
print("\nNext: test creating a custom voice preset using 01.mp3")
