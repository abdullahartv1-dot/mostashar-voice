# 🔒 v1-Stable Restore Point — 2026-05-09

## What this is

A frozen, working snapshot of the VibeVoice-Large streaming TTS solution.
**This is the fallback if experimental work breaks something.**

## Verified working configuration

- **Model:** `aoi-ot/VibeVoice-Large` (8B)
- **Attention:** `flash_attention_2` (CRITICAL — sdpa silently produces gibberish on Arabic)
- **Diffusion steps:** 60
- **CFG scale:** 1.8
- **Seed:** 42
- **Chunking strategy:** sentence (.! ?)
- **Reference audio:** 30s clipped from 01.mp3

## Measured performance (2026-05-09 on RTX PRO 6000)

| Metric | Value |
|---|---|
| Server-side TTFA (1-word chunk) | **1320ms** |
| Server-side TTFA (5-word chunk) | 3192ms |
| GPU memory at rest | 18.7GB / 96GB |
| RTF (real-time factor) | ~1.0 |
| Quality | Large-level (user confirmed "ممتاز جداً") |

## Files

- `server.py` — FastAPI WebSocket streaming server
- `client.html` — Browser test client with TTFA measurement
- `setup_new_pod.sh` — One-shot Pod setup (torch 2.8 + cu128 + py3.12)
- `deploy_to_pod.sh` — Deploy after Pod restart
- `tunnel.sh` — SSH tunnel
- `smoke_clone_large.py` — Direct test script
- `smoke_clone_correct.py` — 1.5B comparison
- `large_encoder_transplant.py` — Encoder transplant (saved /workspace/vv-realtime-large-encoder)
- `encoder_transplant.py` — 1.5B → Realtime transplant

## How to restore

If experiments break things, copy this folder back over:

```bash
cp -r vibevoice_streaming_v1_stable/* vibevoice_streaming/
```

Then redeploy via `deploy_to_pod.sh` after starting a fresh Pod.

## What this DOES NOT solve

- TTFA target of 511ms (not achieved — Large architecture limit)
- Pure realtime conversation use case (need Cartesia or different approach)

## Next experiments (do NOT modify v1-stable)

1. Reduce diffusion_steps from 60 → 15 — see how much TTFA improves vs quality loss
2. Pre-cache LM prefill — save reference audio processing
3. Hybrid Realtime + Large — first chunk fast, rest premium quality
4. Custom Realtime preset — untested, encoder already transplanted
