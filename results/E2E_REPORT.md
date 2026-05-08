# Voice Studio v2 — E2E Test Report

**Date:** 2026-05-07
**Audio:** 01.mp3 (21.24s Arabic interview, 0.81 MB)
**GPU:** RTX A5000 ($0.27/hr)
**Engine PID:** 40358 (Pod 194.68.245.175:22133, port 8000)
**Test script:** `scripts/e2e_test.py`
**Result file:** `results/e2e_test_1778199795.json`
**Test text (TTS):** "مرحباً بكم في تجربة استنساخ الصوت العربي. هذا الصوت تم توليده بالكامل."

## STT Comparison

| Tool             | Status          | Wall Time | Cost    | Speakers | First Segment              |
|------------------|-----------------|-----------|---------|----------|----------------------------|
| whisper-large-v3 | PASS            | 11.2s     | $0.0008 | 2        | "كنت مليونير وأنا عمري 26 سنة" (SPEAKER_0, 0.00–2.96s) |
| whisper-turbo    | PASS            | 3.5s      | $0.0002 | 2        | "كنت مليونير وأنا عمري 26 سنة" (SPEAKER_0, 0.00–2.92s) |
| nemo-canary      | PASS (en-only)  | 18.8s     | $0.0014 | 1        | " Kent millionaire."       |
| vibevoice-asr    | FAIL (known)    | 0.8s      | -       | -        | RuntimeError: model cache only has 1/8 shards (HF CDN throttling) |

Notes:
- `whisper-large-v3` and `whisper-turbo` produce equivalent Arabic transcripts; large-v3 is ~3.2x slower for ~4x cost.
- `nemo-canary` only officially supports en/de/es/fr — Arabic input gets transliterated to "Kent millionaire" (a phonetic English approximation of "كنت مليونير").
- `vibevoice-asr` now fast-fails (0.8s) with a clear cache-incomplete message instead of hanging; previously caused the engine to leak partial downloads and break disk quota.

## TTS Comparison

| Tool             | Status          | Wall Time | Cost    | Output Duration | RTF  | Audio URL |
|------------------|-----------------|-----------|---------|-----------------|------|-----------|
| vibevoice-1.5b   | PASS            | 14.0s     | $0.0010 | 6.80s           | 2.02 | `/files/b3177529/clone_1778199763.wav` |
| f5-tts           | PASS            | 9.6s      | $0.0007 | 3.57s           | 2.61 | `/files/b3177529/clone_1778199773.wav` |
| xtts-v2          | PASS            | 22.2s     | $0.0017 | 8.16s           | 2.70 | `/files/b3177529/clone_1778199795.wav` |
| vibevoice-large  | FAIL (disabled) | 0.2s      | -       | -               | -    | RuntimeError: 10-shard download throttled by HF CDN; intentionally fast-failed |

## Total Session Cost: $0.0058

(Sum of STT runs + TTS runs from the final E2E pass.)

## Bug Fixes Landed

### Bug 1 — Persist `jobs_store` to disk

`engine/jobs_store.py` is now backed by `<job_dir>/job.json` on disk. `get_job` falls back to disk if not in memory, `save_job`/`update_job` write through, and `all_jobs` discovers on-disk jobs not yet in memory. `engine/main.py:api_process` uses `save_job` instead of in-memory dict mutation.

Verification: ran `/api/process` for job `4466accf`, confirmed `job.json` written, restarted engine, and `GET /api/jobs/4466accf` returned 200 with full data — speakers, segments, samples preserved across restart.

### Bug 2 — VibeVoice ASR 500 root cause

`microsoft/VibeVoice-ASR-HF` ships 8 safetensor shards (~5GB total). On this Pod the HF CDN closes connections mid-download (CLOSE-WAIT TCP states observed); 7 of 8 shards are left as `.incomplete` files that never finish but DO consume disk quota. The model itself also pulls Qwen-7B (~15GB) which compounds the failure.

Hardened `engine/services/stt_vibevoice.py`: a new `_check_complete_cache()` runs before any download attempt. If any of the 8 shards are missing, raises a clear `RuntimeError` (now surfaced through the new try/except in `api_process` as a structured 500 response) and skips the call. This prevents the partial-download-fills-disk cascade that previously broke every other tool.

Marked as **known-experimental** — recommended path is a separate one-shot install script that downloads all 8 shards plus Qwen-7B during pod warm-up, with verification, before the engine accepts vibevoice-asr requests.

### Bug 3 — TTS HTTP 500 root causes

Three layers contributed:

1. **Lost `jobs_store` from VibeVoice ASR crash.** Fixed by Bug 1; clones can now find the job even if a previous worker died.
2. **F5-TTS internal Whisper download.** F5-TTS calls `transformers.pipeline("automatic-speech-recognition", "openai/whisper-large-v3-turbo")` to transcribe the reference audio when `ref_text` is empty. That model isn't fully available in any HF cache (config-only, no `model.safetensors`). Fixed by adding `_autotranscribe_ref()` in `engine/services/tts_f5.py` that uses our existing `faster-whisper-large-v3-turbo` to pre-transcribe and pass `ref_text` explicitly. F5-TTS then skips its broken pipeline path and runs in 1.5s of generation.
3. **NeMo `.nemo` tar-extract OOM-on-quota.** `nemo.from_pretrained("nvidia/canary-1b")` extracts a ~3GB tar to `TMPDIR`. The previous `TMPDIR=$VS_HOME/tmp` was on the constrained workspace volume and immediately hit disk quota. Fixed in `engine/run_engine.sh` by setting `TMPDIR=/tmp` (container overlay fs has 11GB free). Also reuses `/workspace/hf-cache` (10GB pre-downloaded models) as `HF_HOME` instead of re-downloading into the constrained `voice-studio-v2/.cache`.

Also added a structured try/except wrapper in `engine/main.py:api_process` and `api_clone` so any handler exception is logged with traceback AND returned as `{"detail": "STT/TTS X failed: ExcType: msg"}` instead of an opaque "Internal Server Error" — this is what made debugging the remaining 500s tractable.

## Recommendations

- **Best Arabic STT:** `whisper-large-v3` (most accurate Arabic + speaker timing).
- **Fastest STT:** `whisper-turbo` (3.5s wall on 21s audio, $0.0002/run, ~equivalent transcript).
- **Cheapest combination:** `whisper-turbo` + `ecapa-tdnn` diar + `f5-tts` clone = ~$0.0009 per round-trip on this audio length.
- **Highest-quality TTS:** subjective — listen to `vibevoice-1.5b`, `f5-tts`, `xtts-v2` outputs at the URLs above and pick.
- **VibeVoice ASR / Large:** re-enable only after a one-shot install script verifies all shards exist locally (don't let the engine try to download mid-request).

## Known Issues

- **VibeVoice ASR (microsoft/VibeVoice-ASR-HF):** HF CDN throttles the 8-shard download on this Pod. 1/8 shards present. Engine fast-fails with cache-incomplete error. Needs offline pre-download script.
- **VibeVoice Large (aoi-ot/VibeVoice-Large):** Same family of issue (10 shards, ~8GB) plus disk-quota constraint. Hard-disabled in `_ensure_node` until disk + download situation improves.
- **pyannote 3.1:** requires `HF_TOKEN` env var to enable (currently unused; `ecapa-tdnn` covers diarization for now).
- **nemo-canary:** doesn't officially support Arabic — produces an English phonetic approximation. Use it only for en/de/es/fr inputs in production.

## How to use the UI

1. Open `http://127.0.0.1:5000/v2` in a browser.
2. Drag/drop the audio file onto the upload area.
3. Pick STT (e.g. `whisper-turbo`) + Diarization tool (`ecapa-tdnn`).
4. Click "بدء المعالجة" (start processing).
5. Type Arabic text, pick a TTS engine (e.g. `vibevoice-1.5b`), click "توليد الصوت" (generate audio).
6. Listen + compare in the bottom comparison table.
7. Session cost auto-tracked in real-time in the header.
