# Voice Studio v2 — Tools Comparison Platform

**Date:** 2026-05-07
**Status:** Design approved by user, ready for implementation plan
**Author:** Claude (via brainstorming with user)

## Goal

Build a Voice Studio comparison platform on RunPod GPU (RTX A5000, $0.27/hr) that lets users select different AI tools for each pipeline stage (STT, Diarization, TTS), run the same audio through them, and compare quality, speed, and cost side-by-side. Same UI structure as `templates/index.html` but with tool selection dropdowns, all running on cloud GPU instead of local CPU + HuggingFace Spaces.

## Why

User pain points discovered through testing:
- HuggingFace Spaces voice cloning hits "GPU quota exceeded" after 3-5 generations
- Whisper Small on local CPU misreads Arabic names ("ومكل ثوم" instead of "أم كلثوم")
- VibeVoice voice cloning produces robotic accent for Saudi dialect
- No data on which tool is fastest/cheapest/best-quality for Arabic

User wants direct, audible comparison between tools — not local CPU vs HF Spaces — to make informed tooling decisions.

## Non-Goals

- Local CPU comparison (already determined CPU loses on accuracy)
- Free HuggingFace Spaces comparison (already determined to be quota-limited)
- Production deployment (this is an evaluation platform)
- Auto-selection of "best tool" (user makes the choice manually)

## User Stories

1. **Compare STT accuracy on Arabic names**: User uploads `03.mp3`, runs once with Whisper Large-v3, then again with VibeVoice ASR, compares which transcribes "أم كلثوم" correctly.

2. **Compare diarization quality**: User runs same audio with pyannote.audio 3.1 vs ECAPA-TDNN, compares which correctly separates speakers in a multi-speaker conversation.

3. **Compare voice cloning accent**: User picks a 30-second reference of SPEAKER_0, generates same Arabic text with VibeVoice-1.5B, F5-TTS, and XTTS-v2 sequentially, listens to all three to pick the most natural Saudi accent.

4. **Track total cost**: After running 5 tool combinations, user sees a cumulative cost table: "Whisper Large-v3 used 9 min on GPU = $0.040, VibeVoice ASR used 35 min = $0.158…"

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Local Machine                                      │
│  ─────────────                                      │
│  Browser (http://127.0.0.1:5000)                    │
│      ↓                                              │
│  Local Flask (templates/index_v2.html)              │
│      ↓ (proxy API calls)                            │
│  SSH Tunnel: localhost:8000 → Pod:8000              │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  RunPod GPU (RTX A5000, 24GB VRAM, $0.27/hr)        │
│  ────────────────────────────────────────────       │
│  FastAPI Engine (port 8000)                         │
│  ├─ /api/stt/{tool}        → STT processing         │
│  ├─ /api/diar/{tool}       → Diarization            │
│  ├─ /api/tts/{tool}        → Voice cloning          │
│  ├─ /api/jobs/{id}         → Job status + results   │
│  └─ /api/costs             → Cumulative cost table  │
│                                                     │
│  Model Manager (lazy load + LRU eviction)           │
│  ├─ Loads model only when tool first selected       │
│  ├─ Keeps 1-2 models in VRAM at a time              │
│  └─ Evicts oldest when VRAM pressure > 80%          │
│                                                     │
│  Models on disk (~50 GB total):                     │
│  STT:  Whisper-large-v3, Whisper-turbo,             │
│        VibeVoice-ASR-HF, NVIDIA Canary-1B           │
│  DIAR: pyannote-3.1, ECAPA-TDNN                     │
│  TTS:  VibeVoice-Large, VibeVoice-1.5B,             │
│        F5-TTS, XTTS-v2, Fish-Speech-1.5             │
└─────────────────────────────────────────────────────┘
```

### Why this split?
- Local Flask preserves existing UI investment (`index.html`)
- Pod GPU avoids cold-start cost on every request
- SSH tunnel gives free encryption + no public exposure
- FastAPI on Pod gives async + auto-docs + standard REST

## Components

### Component 1: Pod GPU Engine (`engine/`)

**Purpose:** Run AI models on demand, expose REST API.

**Files:**
- `engine/main.py` — FastAPI app + routes
- `engine/model_manager.py` — Lazy load + LRU eviction
- `engine/services/stt.py` — Whisper, VibeVoice ASR, NeMo handlers
- `engine/services/diar.py` — pyannote, ECAPA handlers
- `engine/services/tts.py` — VibeVoice TTS, F5-TTS, XTTS, Fish handlers
- `engine/cost_tracker.py` — Per-job cost calculation

**Interface:** REST API
```
POST /api/process
  body: { audio_url, stt_tool, diar_tool }
  returns: { job_id, transcript, speakers, timings, cost }

POST /api/clone
  body: { job_id, speaker_id, text, tts_tool, params }
  returns: { audio_url, duration, time, cost }

GET /api/jobs/{job_id}
GET /api/costs
GET /api/health → { gpu_mem_used, models_loaded }
```

**Why isolated like this?**
- Each service file owns one stage; no cross-service coupling
- Model manager is the only place that knows about VRAM/loading; services just request models by name
- Cost tracker is a pure function of `(seconds, gpu_rate)` — easy to test

### Component 2: Local Flask (`app.py`)

**Purpose:** Serve UI, proxy to Pod engine, persist results locally.

**Files:**
- `app.py` — extended from existing app, add `/v2` route
- `templates/index_v2.html` — extended from `index.html` with tool dropdowns
- `static/audio/jobs/{job_id}/` — saved audio outputs for offline replay

**Interface:** Browser ↔ Flask ↔ Pod
- Flask receives upload, uploads to Pod, returns `job_id`
- Subsequent calls go directly to Pod via SSH tunnel
- Results JSON cached locally so user can revisit later

### Component 3: Frontend UI (`templates/index_v2.html`)

**Purpose:** Same look-and-feel as `index.html` + tool selection.

**Changes from `index.html`:**
- Add 2 dropdowns above "بدء المعالجة" button: STT tool, Diar tool
- Add 1 dropdown next to "توليد الصوت" button: TTS tool
- Add **comparison panel** at the bottom: cumulative table of all runs
- Add **cost summary** badge in the header (running total)

**Why:** Reuses 90% of existing CSS/JS, drops in dropdowns + comparison table.

## Data Flow

### Processing flow:
```
1. User uploads file via Flask UI
2. Flask saves to /static/audio/{job_id}/orig.mp3
3. Flask uploads to Pod via /api/upload
4. User picks STT tool, Diar tool, clicks "بدء المعالجة"
5. Flask calls Pod /api/process { stt_tool, diar_tool, audio_url }
6. Pod's model_manager loads selected models (lazy)
7. STT service transcribes → segments
8. Diar service labels speakers → updates segments
9. (If VibeVoice ASR: both happen in one call, skip step 8)
10. Pod returns timings + transcript + speakers + cost
11. Flask renders timeline + transcript (same as index.html)
12. Result cached in static/audio/jobs/{job_id}/result.json
```

### Cloning flow:
```
1. User picks TTS tool, types text, picks speaker, clicks "توليد"
2. Flask sends to Pod /api/clone
3. Pod loads TTS model (if not loaded; evicts STT model if VRAM tight)
4. TTS generates audio
5. Pod returns audio file URL + timing + cost
6. Flask downloads audio to static/audio/jobs/{job_id}/clone_{n}.wav
7. Audio plays in browser; entry added to comparison table
```

## Error Handling

- **Pod connection lost** (SSH tunnel down): Flask catches, shows "Pod غير متصل" with retry button.
- **Model OOM during load**: Model manager evicts LRU model and retries; if still fails, returns 503 with clear message.
- **Inference timeout** (>10 min): Cancel job, return partial results if any, mark cost = elapsed time.
- **Disk full on Pod**: Detect via `df`, refuse new model loads, prompt to free space.
- **Audio format unsupported**: ffmpeg conversion attempted; if fails, return 400 with format guidance.

## Lazy Loading Strategy

VRAM = 24 GB. Reserved for model weights + activations.

**Memory budget per model (approx):**
- Whisper Large-v3: 3 GB
- Whisper Turbo: 1.5 GB
- VibeVoice ASR (Qwen 7B): 18 GB
- NeMo Canary: 4 GB
- pyannote 3.1: 1 GB
- ECAPA: 0.5 GB
- VibeVoice TTS Large: 8 GB
- VibeVoice 1.5B: 3 GB
- F5-TTS: 2 GB
- XTTS-v2: 2 GB
- Fish Speech: 4 GB

**Rules:**
1. Load on first request for that tool.
2. Track last-used timestamp per model.
3. Before loading, if total VRAM usage + new model > 80% × 24 GB, evict LRU model.
4. VibeVoice ASR is "exclusive" — when it's loaded, no other model fits; we evict everything else.
5. Loading times are tracked separately from inference (so cost shows "load: 30s, infer: 9 min").

## Testing Strategy

### Smoke tests (during environment setup):
For each model, run a 5-second test on `/workspace/test_clip.wav` and verify:
- Model loads without error
- Inference returns valid output
- Output passes basic shape check (transcript non-empty, audio file >1KB)

### Comparison tests:
After all tools verified, the user runs manual comparison via UI. No automated benchmarks (the user said they want manual testing).

### Failure modes that are OK:
- VibeVoice ASR may fail on full 93-min audio (timeout) — that's a useful data point in the comparison.
- Some TTS tools may produce poor Arabic — that's the comparison purpose.

## Security & Cost Controls

- SSH tunnel restricts API to localhost (no public exposure).
- API key on Pod side as defense-in-depth: `X-API-Key: <random>`.
- Hard cap on TTS text length: 2000 chars (prevent runaway cost).
- Hard cap on STT audio: 6 hours (prevent overnight runs by mistake).
- Cost summary shows live `$X.XXX` so user sees burn rate.
- Auto-stop Pod after 30 min idle (configurable, off by default for testing).

## Out of Scope (for first version)

- Multi-user (single-user platform, no auth needed for SSH tunnel)
- Database persistence (JSON files in `static/audio/jobs/` is fine)
- Real-time streaming (batch processing only)
- Mobile UI (desktop only)
- Automatic "winner" selection (user decides)

## Success Criteria

After this is built and the environment is ready, the user can:
1. Upload `03.mp3` (the 93-min Arabic audio they have).
2. Run with Whisper Large-v3 + pyannote → see results.
3. Re-run with VibeVoice ASR → see results.
4. Compare both transcripts + costs side-by-side in the comparison table.
5. Generate same text with 3-4 different TTS tools.
6. Listen to all generated audios in the browser.
7. See total session cost (e.g., "$0.45 spent so far").
8. Have data to make tooling decisions.

## Open Questions (resolved)

- ✅ Tools list: 12 tools across 3 stages (user approved).
- ✅ Comparison style: sequential (run one tool at a time, log results), not parallel.
- ✅ UI: extends index.html, doesn't replace it.
- ✅ Hosting: RunPod RTX A5000 (already provisioned).
- ✅ Cost tracking: per-stage + cumulative session.

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| VibeVoice ASR setup fails (saw cuDNN/transformers issues) | High | Mark as "experimental" tool; pipeline still works without it |
| Some models incompatible with current torch version | Medium | Pin tested versions; document conflicts |
| 24 GB VRAM insufficient for VibeVoice ASR + active session | Medium | Evict everything else when ASR loads |
| Pod disk fills with model downloads | Medium | Quota check before download; prompt to clean |
| Container disk (20GB) hits limit (it did before) | High | Force HF cache to /workspace, not /root |
