# OpenVoice v2 Sidecar — Deploy & Operations

> **Purpose.** Apply a clarity-of-articulation donor's acoustic profile
> to a user's freshly-uploaded clone reference, *once*, at clone time.
> The enhanced reference is what VibeVoice clones from on every future
> generation — so realtime conversations stay fast (no per-turn voice
> conversion) while every reply benefits from cleaner مخارج الحروف.

---

## Architecture

```
                                                ┌──────────────────────────┐
[POST /v1/voices/add] ──┐                       │  /workspace/openvoice-   │
                        │                       │  venv  (transformers     │
                        │  upload.wav (~10s)    │  pinned, OpenVoice v2)   │
                        ▼                       │                          │
┌───────────────────────────────────┐  HTTP     │  openvoice_server.py     │
│  server_v5_api.py  (port 8080)    │ ────────► │  (port 8083, GPU)        │
│  /v1/voices/add                   │  /v1/     │                          │
│    │                              │  enhance- │  ┌───────────────────┐   │
│    └─► _save_voice                │  reference│  │ ToneColorConverter│   │
│          │                        │           │  │  (cached)         │   │
│          ├─► save raw upload      │           │  └────────┬──────────┘   │
│          │   {id}.raw.wav         │           │           │              │
│          ├─► call sidecar ────────┼───────────┘           │              │
│          ▼                        │                       │              │
│       enhanced.wav                │ ◄─────────────────────┘              │
│          │                        │                                      │
│          └─► save canonical       │                                      │
│              {id}.wav             │                                      │
└───────────────────────────────────┘                                      │
                                                                           │
[every TTS / conversation reply]                                           │
        ▼                                                                  │
   loads {id}.wav  (no sidecar call) ─────────────────────────────────────┘
```

**Key isolation guarantees.**
- Sidecar lives in its own venv (different transformers / numpy versions
  from main server) so OpenVoice deps cannot regress VibeVoice serving.
- Sidecar is *optional*: if it's down or `MV_OPENVOICE_URL=""`, cloning
  still works and just uses the raw upload as the canonical reference.
- Failure of any sidecar call is logged and falls through silently —
  the user always gets a working voice, possibly without enhancement.

---

## One-time setup on the pod

```bash
# 1. Create the dedicated venv. Keep it OUTSIDE /workspace/x so a
#    `git pull` on the main repo never touches it.
python3 -m venv /workspace/openvoice-venv
source /workspace/openvoice-venv/bin/activate

# 2. Pin Python deps. OpenVoice v2 needs torch ≥ 2.0 and a recent
#    transformers; we let pip pick the latest compatible.
pip install --upgrade pip wheel
pip install \
    'torch>=2.0' 'torchaudio>=2.0' \
    librosa soundfile numpy \
    fastapi 'uvicorn[standard]' httpx \
    pydantic

# 3. Install OpenVoice itself from upstream.
pip install git+https://github.com/myshell-ai/OpenVoice@main

# 4. Download the v2 checkpoints. The repo ships them via release
#    tarballs — fetch and unpack into the configured CKPT_DIR.
mkdir -p /workspace/openvoice/checkpoints_v2
cd /workspace/openvoice/checkpoints_v2
wget -q https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip
unzip -q checkpoints_v2_0417.zip
rm checkpoints_v2_0417.zip
# After unpack, verify:
ls converter/   # → checkpoint.pth, config.json
```

---

## Runtime

### Start the sidecar

```bash
source /workspace/openvoice-venv/bin/activate
export OPENVOICE_PORT=8083
export OPENVOICE_CKPT_DIR=/workspace/openvoice/checkpoints_v2
export OPENVOICE_DONORS_DIR=/workspace/refs/voices
export OPENVOICE_DEFAULT_DONOR=hamed_saudi
nohup python /workspace/x/vibevoice_streaming/openvoice_server.py \
    > /tmp/openvoice.log 2>&1 &
echo $! > /tmp/openvoice.pid
```

### Smoke test

```bash
# Liveness — must return loaded=true within ~10 s of startup.
curl -s http://127.0.0.1:8083/health | jq

# List donors — confirms hamed_saudi.wav is discoverable.
curl -s http://127.0.0.1:8083/v1/donors | jq

# End-to-end enhance against a sample upload.
curl -s -X POST \
    -F target=@/workspace/refs/voices/saleh-XXXXXX.wav \
    -F donor_id=hamed_saudi \
    -F return_metrics=true \
    http://127.0.0.1:8083/v1/enhance-reference | jq '.metrics'
```

### Stop / restart

```bash
kill -TERM "$(cat /tmp/openvoice.pid)"
# Or, if the pid file was lost:
pkill -f openvoice_server.py
```

---

## Environment variables (main server)

Set in the main `server_v5_api.py` env so cloning auto-routes to the
sidecar:

| var | default | meaning |
|---|---|---|
| `MV_OPENVOICE_URL` | `http://127.0.0.1:8083` | sidecar base URL; empty disables enhancement |
| `MV_OPENVOICE_DONOR` | `hamed_saudi` | clarity donor used by `/v1/voices/add` |
| `MV_OPENVOICE_TIMEOUT_S` | `30` | per-clone timeout; falls back to raw on overrun |

---

## Operations

### Health monitoring

`GET /v1/health` on the main server now includes:

```json
{
  "openvoice": {
    "url": "http://127.0.0.1:8083",
    "reachable": true,
    "loaded": true,
    "detail": {
      "cached_donors": ["hamed_saudi"],
      "warmup_ms": 4231,
      "vram_gb": 0.9
    }
  }
}
```

If `reachable=false`, every `/v1/voices/add` call will silently fall
back to raw cloning — so this metric must be alerted on.

### Re-enhancing an existing voice

If a voice was created while the sidecar was down (or with the wrong
donor), re-run enhancement without re-uploading:

```bash
curl -X POST \
    -H "Authorization: Bearer $MV_API_KEY" \
    -F donor=hamed_saudi \
    https://api.example.com/v1/voices/saleh-abc123/enhance
```

This reads `{voice_id}.raw.wav` (preserved on every upload) and
overwrites the canonical `{voice_id}.wav`.

### Reverting to the raw upload

If a user dislikes the enhanced sound:

```bash
curl -X POST \
    -H "Authorization: Bearer $MV_API_KEY" \
    https://api.example.com/v1/voices/saleh-abc123/revert-to-raw
```

This restores `{voice_id}.wav` from `{voice_id}.raw.wav`.

---

## Troubleshooting

### Sidecar comes up but `loaded=false`

- Most likely the checkpoint files are missing or corrupted. Verify:
  `ls /workspace/openvoice/checkpoints_v2/converter/`
  Should contain `checkpoint.pth` (≈ 100 MB) and `config.json`.

### Cloning is slow (>20 s)

- First call after sidecar restart pays a ~5 s donor-SE-extraction tax.
  Subsequent calls reuse the cached embedding. Pre-warm by hitting
  `/v1/donors` once at startup.
- Check `nvidia-smi` — if GPU is full from VibeVoice, the sidecar
  serializes inside its own lock; wait or stop the largest TTS job.

### Output sounds wrong (worse than raw)

- The donor file matters. Try a different donor:
  `MV_OPENVOICE_DONOR=default` or pick another via the `donor` form
  field of `/v1/voices/{id}/enhance`.
- Compare metrics in `/v1/voices/{id}` — if `enhanced.spectral_flatness`
  is *higher* than `target.spectral_flatness`, the donor is noisier
  than the raw upload and you should not enhance against it.

### Disk filling up

- `{voice_id}.raw.wav` doubles disk usage per cloned voice. With ~30 s
  of 24 kHz mono = ~1.4 MB per voice, so 1 000 voices = ~1.4 GB raw +
  1.4 GB enhanced = 2.8 GB total. Acceptable but worth watching.

---

## File map

| path | purpose |
|---|---|
| `vibevoice_streaming/openvoice_server.py` | the sidecar |
| `vibevoice_streaming/server_v5_api.py` | main server, `_openvoice_enhance_reference` calls into the sidecar |
| `/workspace/openvoice-venv/` | sidecar's isolated Python environment |
| `/workspace/openvoice/checkpoints_v2/` | OpenVoice model weights |
| `/workspace/refs/voices/{id}.wav` | canonical reference (used by VibeVoice) |
| `/workspace/refs/voices/{id}.raw.wav` | original user upload (preserved for re-enhance / revert) |
| `/workspace/refs/voices/{id}.json` | meta with `enhanced` flag + before/after metrics |
| `/tmp/openvoice.log` | sidecar runtime logs |
| `/tmp/openvoice.pid` | sidecar PID for restarts |
