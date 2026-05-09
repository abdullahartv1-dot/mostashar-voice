# Streaming TTS for Mostashar — Design Spec

**Date:** 2026-05-09
**Author:** Claude (autonomous research session while user sleeps)
**Status:** Proposal — pending user review on wake

## Goal

Deliver Arabic voice cloning TTS with:
1. **Excellent cloning quality** (Large-level, user-confirmed in `large_oneshot.wav`)
2. **Low TTFA** for "press to listen" UX (target: ≤2 seconds, ideally ≤1.5s)
3. **Self-hosted** on user's GPU infrastructure (no per-minute API fees)

## Background — what we found

### ✅ Confirmed working
- **VibeVoice-Large + flash_attention_2 + diff=60 + cfg=1.8** = excellent Arabic cloning
- **Sentence chunking** drops TTFA from 8.7s → 3.6s with no quality loss
- **Encoder is bit-identical** across all VibeVoice variants (0.5B/1.5B/Large) — Microsoft trained it once
- **VibeVoice-Realtime-0.5B + en-Carter_man preset** = 511ms TTFA but no cloning, English voice

### ❌ Discovered roadblock — Option E is much harder than initially proposed

**User's hypothesis:** "Port streaming inference TO Large for best of both worlds"

**Reality** (verified in `modeling_vibevoice_streaming.py` vs `modeling_vibevoice.py`):

```
Large/1.5B (regular):                   Realtime-0.5B (streaming):
  language_model = FULL Qwen             language_model = LOWER N-K Qwen layers
                                         tts_language_model = UPPER K Qwen layers
  semantic_tokenizer ✓                   (REMOVED entirely)
  acoustic + semantic connectors         only acoustic_connector
```

The streaming model **splits Qwen into two LMs** — this split is engineered AT TRAINING TIME. We cannot post-hoc split Large's pretrained Qwen2-7B into top-K and bottom-(N-K) and expect coherent output. The layers were trained as a unified stack.

**Confidence:** Code-verified, not speculation. See `vibevoice_architecture_discovery.md` in memory.

## Three viable paths forward

### Path F — Sub-sentence chunking with Large (RECOMMENDED)
**Idea:** Apply finer-grained chunking than sentence-level. Split on phrase boundaries (commas, conjunctions, periods) into 3-8 word chunks. Generate each on Large, stream as ready.

**Pros:**
- Quality stays at confirmed-excellent Large level
- No architectural changes
- TTFA estimate: **1.5-2.5s** (down from 3.6s)
- Can also reduce diffusion_steps from 60 to 30-40 (further speed, acceptable quality drop)

**Cons:**
- Not as fast as Realtime's 511ms
- Tuning needed — finding chunk boundaries that don't break prosody

**TTFA components for first chunk:**
- chunks of ~5 words ≈ 1-1.5s of audio
- Generation at RTF 0.4-1.0 → 0.4-1.5s gen time
- Network + browser audio init: ~200ms
- **Total estimate: 1.5-2.5s**

### Path B' — Custom Realtime preset (Plan B if F fails to meet TTFA target)
**Idea:** Use already-transplanted encoder in Realtime. Reverse-engineer the prefill flow to create custom `.pt` voice presets from Arabic reference audio.

**Pros:**
- Inherits Realtime's 511ms TTFA
- We have the encoder transplanted
- The 25 existing presets prove the format works

**Cons:**
- Quality bottlenecked by 0.5B LM (likely B+ grade vs Large's A+)
- Reverse engineering prefill flow takes 2-4h
- User explicitly said cloning quality matters

**Status:** Saved to memory as fallback. Implement only if F fails.

### Path G — Hybrid (deploy F first, add B' later if needed)
**Idea:** Deploy F immediately for production use. Develop B' in parallel as a "live conversation" path while F handles "press to read".

**Use cases split:**
- Press-to-read in CRM (notes, task descriptions, reports) → Large + sub-sentence chunking (F)
- Live voice agent on calls → Realtime + custom preset OR Cartesia (when ready)

## Architecture for Path F (recommended for immediate implementation)

```
┌────────────────────────────────────────────────────────────────┐
│                  Browser (mostashar UI)                        │
│  Click "🔊 استمع" on text                                       │
│         ↓                                                       │
│  WebSocket /tts/stream  (text + voice_id)                      │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│              FastAPI Streaming Server (Pod)                    │
│                                                                │
│  1. Receive text                                               │
│  2. Split into sub-sentences (by ، . ! ؟ + conjunction words)  │
│  3. For each chunk:                                            │
│     a. Generate audio with VibeVoice-Large                     │
│        (cached voice prefill — re-use for same voice_id)       │
│     b. Encode WAV chunk                                        │
│     c. Send chunk via WebSocket                                │
│  4. Send EOS signal                                            │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│                  Browser Audio Pipeline                        │
│                                                                │
│  - Receive WAV chunks via WebSocket                            │
│  - Use Web Audio API: AudioBufferSourceNode queue              │
│  - Schedule playback gapless                                   │
│  - First chunk plays at TTFA, subsequent chunks scheduled      │
└────────────────────────────────────────────────────────────────┘
```

## Components

### Server side (1 file: `server.py`)
- FastAPI app with `/tts/stream` WebSocket endpoint
- Singleton VibeVoice-Large model loaded at startup
- Voice cache (Dict[voice_id → reference_audio_tensor])
- Sub-sentence chunker (regex-based: `[،.!؟,]|\sو\s|\sف\s|\sثم\s`)
- Per-chunk generator with same model (re-uses GPU memory)

### Client side (1 file: `client.html`)
- "Listen" button that opens WebSocket
- Web Audio API queue for gapless playback
- Status indicator (loading / playing / done)

### Voice management
- Initial: hardcoded voice_id="user_default" → 01.mp3 reference
- Future: voice library API (POST /voices/clone uploads audio + creates voice_id)

## Open questions to test

1. **Does sub-sentence chunking break prosody?**
   - Test: Generate "مرحبا، كيف حالك؟ أنا بخير." with sentence chunking vs sub-sentence
   - Listen for unnatural pauses or pitch breaks at sub-sentence boundaries

2. **Can diffusion_steps drop from 60 to 30 without quality loss?**
   - Test: A/B same chunk at 60 vs 30 steps
   - User likely won't hear difference but verify

3. **Real TTFA for sub-sentence chunking?**
   - Currently estimated 1.5-2.5s; need to measure
   - Critical for go/no-go decision

4. **Does Web Audio API gapless playback work with VibeVoice's audio format?**
   - 24kHz mono PCM → AudioBuffer
   - Schedule with audioContext.currentTime + previous_chunk.end
   - Cross-fade if needed

## Success criteria (per user-stated "level B" target)

- [ ] TTFA ≤ 2.5s (must) — current Large baseline is 3.6s
- [ ] TTFA ≤ 1.5s (stretch) — would match user's expectation
- [ ] Quality matches `large_oneshot.wav` (must) — that's user-confirmed excellent
- [ ] No audible gaps/artifacts at chunk boundaries (must)
- [ ] Full text of 3 sentences plays in <12s end-to-end (must) — current is 10.8s

## Implementation plan (next session)

1. Restart Pod (or use cheaper one)
2. Build `server.py` with chunking logic
3. Test with 3-sentence Arabic prompt — measure TTFA
4. Tune diffusion_steps to find quality/speed sweet spot
5. Build `client.html` for browser test
6. End-to-end measurement
7. Either deploy as production OR pivot to Path B' if TTFA target unmet

## Cost analysis

- Pod (H100): $2.99/hr running, $0.028/hr stopped
- Development: ~3-4 hours of pod time = $9-12
- Production: 1 H100 24/7 = ~$2,150/month for unlimited usage
- Alternative: A100 80GB at $1.99/hr = ~$1,433/month (might be enough for Large)

## Decision points for user (when waking)

1. **Approve Path F as primary direction?** (vs continuing E or B)
2. **Acceptable TTFA target?** 1.5s ideal, 2.5s acceptable, 3.6s reluctant accept
3. **OK to drop diffusion_steps from 60 to 30 if quality is preserved?**
4. **Production GPU choice:** H100 ($2,150/mo) vs A100 ($1,433/mo) — speed difference vs cost
