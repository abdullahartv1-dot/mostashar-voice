# Gold Reference Voices — Mostashar Voice

This directory holds **canonical reference voices** that have been measured
to produce exceptional TTS quality through VibeVoice-Large. They are kept
checked-in as the empirical "gold standard" for comparing future references
and for retraining / regression-testing the project.

## Why this matters

VibeVoice clones the *full acoustic profile* of the reference — including
its noise floor, microphone characteristics, room reverb, and dynamic
range. A studio-grade reference produces studio-grade output; a phone
recording produces phone-quality output. The single biggest determinant
of perceived TTS quality is the cleanliness of the source.

We discovered this empirically when the user reported "the voice in
`/call` is amazing — better than anything else I've tested." Profiling
the references showed `default.wav` is an order of magnitude cleaner
than the rest of the library:

| Voice | spectral_flatness | sib/speech | HF/speech | Source |
|---|---|---|---|---|
| **`default-2026-05-09.wav`** | **0.019** ⭐ | **0.103** | 0.029 | Studio-grade human recording |
| `صالح-673cb5.wav` | 0.131 | 0.024 | 0.000 | User-uploaded mic recording |
| `hamed_saudi.wav` (edge-tts) | 0.222 | 0.038 | 0.049 | Microsoft Azure Neural → MP3 → WAV |

Lower flatness = more tonal / less noise. Higher sib ratio = sharper
sibilants (س, ش, ف, ث, خ, ح). The `default` reference wins both axes
by a large margin.

## Gold-standard thresholds

A reference voice is considered **production-grade** when it satisfies
all of the following on the 24 kHz mono WAV:

| Metric | Threshold |
|---|---|
| Duration | 15 s ≤ x ≤ 30 s |
| Peak | 0.4 ≤ peak ≤ 0.95 (avoid clipping, avoid being too quiet) |
| Spectral flatness | ≤ 0.05 (≤ 0.10 is acceptable, > 0.20 is noisy) |
| HF (>8 kHz) / speech (80-4 kHz) ratio | ≤ 0.05 |
| Sibilant (4-8 kHz) / speech ratio | ≥ 0.05 (higher = crisper consonants) |
| 50/60 Hz hum ratio | ≤ 0.01 |

## Reproducing the measurement

The numbers above come from `vibevoice_streaming/_diag_noise.py` (kept in
the repo). Run it from the pod against any reference WAV to get a
quality report.

## Files

- `default-2026-05-09.wav` — the gold reference at the moment we found it.
  Hash: `e8068b83756b7eefe8ef7492c569dd11`
- `default-2026-05-09.json` — its metadata (denoise flag etc.)

## Operating procedure

When adding a new voice to production, score it against the table above.
If any metric fails, **clean the reference first** rather than shipping
poor TTS:

1. Re-record in a quieter room with the mic 10-20 cm from the speaker
2. Run the audio through a noise reducer offline
3. Trim leading/trailing silence to 15-30 s of pure speech
4. Re-measure with `_diag_noise.py`

For premade voices generated from edge-tts (Azure Neural), accept that
quality is bounded by MP3 compression. They sound fine for casual use
but won't reach `default`-level fidelity.
