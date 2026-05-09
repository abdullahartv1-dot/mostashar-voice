# مُسْتَشَار — Real-time Arabic Voice Cloning TTS

Self-hosted, ElevenLabs-style voice library + realtime text-to-speech with voice cloning.
Built on **VibeVoice-Large (8B)** for premium Arabic quality, with **AudioStreamer** for sub-500ms TTFA.

> 🎯 **Achieved:** Large quality cloning + realtime TTFA (~334ms) on RTX PRO 6000
> 🌍 **Stack:** FastAPI + Next.js 14 + Tailwind + shadcn/ui + WebSocket streaming

---

## 🏆 What Works

| Feature | Status | Notes |
|---|---|---|
| Voice cloning from 3-30s reference audio | ✅ | upload or record in browser |
| Voice library API (POST/GET/DELETE) | ✅ | persisted to disk |
| Realtime streaming TTS via WebSocket | ✅ | PCM16 chunks, gapless playback |
| Arabic language support | ✅ | flash-attention required |
| TTFA (Time To First Audio) | **~334ms** | warmed, sub-500ms |
| RTF (real-time factor) | ~0.5 | faster than realtime |
| Multi-voice user library | ✅ | each voice loaded once, reused |

## 📁 Repo Structure

```
vibevoice_streaming/
├── server_v4_voice_library_ui.py   ← FastAPI backend (production)
├── server_v3_voice_library.py      ← v3 (voice library API)
├── server_v2_streaming.py          ← v2 (AudioStreamer breakthrough)
├── server.py                       ← v1 baseline (sentence chunking)
├── deploy_to_pod.sh                ← One-shot Pod setup
├── setup_new_pod.sh                ← Fresh Pod from scratch
├── tunnel.sh                       ← SSH tunnel helper
├── exp1_diffusion_sweep.py         ← Diffusion steps benchmark
├── exp2_profile_overhead.py        ← Where the time goes
└── web/                            ← Next.js 14 + shadcn UI
    ├── app/
    │   ├── clone/page.tsx          ← Clone page (upload + waveform + form)
    │   ├── library/page.tsx        ← Library page (picker + list + player)
    │   └── layout.tsx
    ├── components/
    │   ├── nav-bar.tsx
    │   └── ui/                     ← shadcn components + Orb + VoicePicker
    └── lib/                        ← cn util + types
vibevoice_streaming_v1_stable/      ← Frozen v1 restore point
docs/specs/                         ← Design specs
```

## 🚀 Quick Start

### 1. Backend (Pod with GPU)

Requires CUDA-capable GPU (RTX PRO 6000 / H100 / A100 — needs ~20GB VRAM for VibeVoice-Large).

```bash
# Install deps (Ubuntu 24.04 + Python 3.12)
pip install --break-system-packages \
  'transformers==4.51.3' 'tokenizers>=0.21,<0.22' 'accelerate==1.6.0' \
  soundfile librosa fastapi 'uvicorn[standard]' python-multipart \
  diffusers peft flash-attn --no-build-isolation

# Clone vibevoice (community fork — has cloning support)
git clone --depth 1 https://github.com/vibevoice-community/VibeVoice /workspace/vv-community
cd /workspace/vv-community && pip install --break-system-packages --no-deps -e .

# Start server (downloads VibeVoice-Large on first run, ~16GB)
cd /workspace
env HF_HOME=/workspace/hf-cache \
    VV_MODEL=aoi-ot/VibeVoice-Large \
    VV_DIFF_STEPS=15 \
    VV_CFG=1.8 \
    VV_PORT=8080 \
    python server_v4_voice_library_ui.py
```

### 2. Frontend (local dev)

```bash
cd vibevoice_streaming/web
npm install
npm run dev   # http://localhost:3000
```

The Next.js dev server proxies `/voices`, `/tts/*`, `/health` to `http://127.0.0.1:8080` (the backend). Override via `BACKEND_URL` env var.

### 3. SSH tunnel (when backend is on remote Pod)

```bash
ssh -i ~/.ssh/id_ed25519 -p POD_PORT -L 8080:localhost:8080 -N -f root@POD_HOST
```

## 🔬 Key Technical Findings

1. **VibeVoice acoustic encoder is bit-identical** across 0.5B/1.5B/Large variants. Microsoft trained it once and ships the same weights everywhere. The Realtime-0.5B variant has the encoder *zeroed out* (anti-deepfake) but the architecture slot is intact.

2. **Streaming is built into Large's `model.generate()`** via the `audio_streamer` kwarg (line 657 of `modeling_vibevoice_inference.py` in the community fork). Pass an `AudioStreamer(batch_size=1)` and the model emits PCM chunks during the diffusion loop.

3. **Diffusion steps sweet spot is 15** — drops generation time by 57% vs 60 steps with no audible quality loss for cloning.

4. **flash_attention_2 is mandatory** — `sdpa` silently produces gibberish for Arabic on bf16. `eager` works but is slower.

5. **Streaming model architecture ≠ regular model architecture.** Microsoft's VibeVoice-Realtime-0.5B splits Qwen into two LMs (text + TTS), trained jointly. Cannot be retrofitted to Large without retraining.

## 🎛️ API Reference

### `POST /voices/clone` (multipart/form-data)
```
name=string
language=ar|en|multi
start_s=float (0.0)
end_s=float (≤30, ≥3 seconds of selected region)
audio=file (mp3/wav/m4a/webm)
→ {voice_id, name, language, dur_s, redirect}
```

### `GET /voices`
```json
{"voices": [{"voice_id", "name", "language", "dur_s", "preview_url", "created_at"}, ...]}
```

### `GET /voices/{voice_id}/preview` → WAV file
### `DELETE /voices/{voice_id}` → 204
### `WS /tts/stream`
Send: `{"text": "...", "voice_id": "..."}`
Receive: `{type:"meta",sample_rate:24000}` then binary PCM16LE chunks, then `{type:"ttfa",ms:N}` and `{type:"done",total_*}`

## 💰 Production Cost

| GPU | Hourly | Monthly (24/7) | Notes |
|---|---|---|---|
| RTX PRO 6000 | $1.91 | ~$1,400 | Recommended — 96GB VRAM, ~$1.10/hr saved vs H100 |
| H100 SXM | $3.01 | ~$2,200 | Slightly faster but pricier |

vs commercial alternatives:
- ElevenLabs Turbo v2.5: ~$5,000-10,000/month at scale
- Cartesia Sonic-3: ~$1,500/month at scale (cloud only, no self-host)

## 📚 Design Docs

- [docs/specs/2026-05-09-streaming-tts-design.md](docs/specs/2026-05-09-streaming-tts-design.md) — full architecture + decision log

## 🛠️ Tested With

- **Pod:** RunPod RTX PRO 6000 Blackwell (96GB) in US-NC-1
- **Host stack:** Ubuntu 24.04, Python 3.12, PyTorch 2.8.0+cu128, flash-attn 2.8.3
- **Models:** `aoi-ot/VibeVoice-Large` (8B, ~16GB), `microsoft/VibeVoice-Realtime-0.5B` (for comparison)

## 📝 License

This project's code is MIT.
The underlying VibeVoice models are MIT (per Microsoft) — but check each model card on Hugging Face.

## 🙏 Credits

- [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice) — base model
- [vibevoice-community/VibeVoice](https://github.com/vibevoice-community/VibeVoice) — community fork with cloning support intact
- [shadcn/ui](https://ui.shadcn.com/) — Tailwind component primitives
