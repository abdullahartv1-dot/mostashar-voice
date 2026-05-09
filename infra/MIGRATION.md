# Migration & Deployment Runbook

Operational reference for two scenarios:

1. **Day-to-day code changes** → covered by GitHub Actions
   (`.github/workflows/deploy-pod.yml`). Push to `main` or
   `feat/voice-studio-v2`, the workflow deploys to the live pod in
   ~2-3 minutes.

2. **Moving to a new pod / new GPU host** → covered by
   `infra/bootstrap_new_pod.sh`. ~10-15 minutes start-to-finish on
   a fresh RunPod with a fast HF mirror.

For self-hosted (non-RunPod) Docker deployments, see
`infra/docker-compose.yml`.

---

## 1. Day-to-day deploys (CI/CD)

### One-time setup — GitHub repo secrets

Repo → **Settings → Secrets and variables → Actions → New**:

| Secret name             | Value                                                  |
|-------------------------|--------------------------------------------------------|
| `POD_HOST`              | RunPod direct-TCP host, e.g. `103.196.86.113`          |
| `POD_PORT`              | RunPod direct-TCP port, e.g. `14042`                   |
| `POD_SSH_PRIVATE_KEY`   | full contents of `~/.ssh/id_ed25519`                   |

The matching public key (`~/.ssh/id_ed25519.pub`) should already be
in `/root/.ssh/authorized_keys` on the pod. RunPod adds your account
key automatically when you provision a pod from the dashboard, so
this is normally already done.

> **Find the host + port:** RunPod console → click your pod → tab
> **Connect → SSH over exposed TCP**.

### Triggering a deploy

Three ways:

| Trigger          | When it fires                                           |
|------------------|---------------------------------------------------------|
| `git push`       | automatic on `main` or `feat/voice-studio-v2` if any of `vibevoice_streaming/{server_v5_api,gemma4_server,openvoice_server}.py` changed |
| Manual dispatch  | Actions tab → "Deploy to RunPod" → "Run workflow"       |
| Restart-only     | same as above with `restart_only: true` — useful when you want to bounce the server without uploading new code |

### What the workflow does

1. checkout the commit
2. SSH-prep with the secret key
3. SCP the three Python servers to `/workspace/`
4. python-syntax-check them on the pod (fail fast on typos)
5. kill the running server on `:8080`
6. relaunch with the canonical env vars (Whisper locked to `ar`)
7. poll `/v1/health` until `whisper_loaded=true` (up to 3 min)
8. assert the running model is `whisper-large-v3` (not turbo)

If any step fails, the workflow exits non-zero and the previous
server is *already gone* — you'll see logs in GitHub Actions UI.
Manually re-run with `restart_only: true` to recover.

### Common failures + fixes

| Symptom                                          | Cause                                                          | Fix                                              |
|--------------------------------------------------|----------------------------------------------------------------|--------------------------------------------------|
| "Permission denied (publickey)"                  | wrong / missing `POD_SSH_PRIVATE_KEY`                          | regenerate, paste full key into secret           |
| "Connection refused"                             | pod is stopped                                                 | start the pod from RunPod console                |
| "FAIL — server did not respond within 180 s"     | model load is slow / cache wiped                               | check pod logs, may need `bootstrap_new_pod.sh`  |
| "unexpected whisper model: 'whisper-large-v3-turbo'" | someone overrode `MV_WHISPER_MODEL` in env                  | update `start_server.sh` env defaults            |

---

## 2. New pod migration (Bootstrap)

When the H100 maintenance window happens, or you want to swap to a
cheaper GPU, or RunPod loses your pod:

### Step 1 — provision a fresh pod

RunPod template: anything with CUDA 12.1+ and ≥ 60 GB persistent
volume. We've been using `runpod/pytorch:2.4.0-py3.11-cuda12.1.1`.

### Step 2 — paste the bootstrap into the pod

In the pod's web terminal (or via SSH):

```bash
# pull + run the bootstrap. Authoritative branch is feat/voice-studio-v2.
curl -fsSL https://raw.githubusercontent.com/abdullahartv1-dot/mostashar-voice/feat/voice-studio-v2/infra/bootstrap_new_pod.sh \
    | bash
```

The bootstrap clones the repo, pip-installs deps, downloads the
~46 GB of HF model weights, and prints the next-step command.

### Step 3 — migrate voice references (~50 MB)

If you want the same voice library as the old pod:

```bash
# from your laptop, with both pods reachable
scp -r OLD_POD:/workspace/refs/voices NEW_POD:/workspace/refs/voices
```

### Step 4 — start the servers

```bash
ssh root@NEW_POD
bash /workspace/x/infra/start_server.sh         # main server, port 8080
bash /workspace/x/infra/start_gemma4.sh         # Gemma sidecar,  port 8082
# Optional: OpenVoice sidecar
# bash /workspace/x/vibevoice_streaming/start_openvoice.sh
```

### Step 5 — point CI/CD at the new pod

Update the `POD_HOST` + `POD_PORT` secrets in GitHub Actions to match
the new pod's connection details (Connect tab → SSH over exposed TCP).

That's it. Subsequent deploys go via the workflow.

---

## 3. Self-hosted (Docker compose)

Use this when you've outgrown RunPod or want full sovereignty.

```bash
cd infra
cp .env.example .env             # set MV_API_KEY, etc.
docker compose up -d
docker compose logs -f vibevoice
```

Required: a host with NVIDIA Container Toolkit and at least one GPU
(H100 / A100 / RTX 4090 — anything with ≥ 24 GB VRAM works for the
small variant; ≥ 60 GB needed for VibeVoice-Large + Whisper + Qwen
+ Gemma).

Stop without losing the model cache:

```bash
docker compose down                # keeps `hf-cache` and `voices` volumes
```

Wipe everything:

```bash
docker compose down -v             # also removes ~46 GB cache
```

---

## 4. File map (what lives where)

| Path                                          | Purpose                                       |
|-----------------------------------------------|-----------------------------------------------|
| `vibevoice_streaming/server_v5_api.py`        | main FastAPI server (port 8080)               |
| `vibevoice_streaming/gemma4_server.py`        | Gemma 4 sidecar (port 8082)                   |
| `vibevoice_streaming/openvoice_server.py`     | OpenVoice sidecar (port 8083, optional)       |
| `infra/bootstrap_new_pod.sh`                  | one-shot pod setup                            |
| `infra/start_server.sh`                       | launch / restart main server                  |
| `infra/start_gemma4.sh`                       | launch / restart Gemma sidecar                |
| `infra/Dockerfile.vibevoice`                  | self-hosted image for main server             |
| `infra/Dockerfile.gemma4`                     | self-hosted image for Gemma sidecar           |
| `infra/docker-compose.yml`                    | full self-hosted stack                        |
| `.github/workflows/deploy-pod.yml`            | CI auto-deploy to live pod                    |

---

## 5. Health monitoring

`GET /v1/health` returns the full system snapshot:

```bash
curl -sk https://YOUR-HOST/v1/health | jq
```

Watch fields:

- `whisper_loaded: true` + `whisper_model: "openai/whisper-large-v3"` — Arabic transcription is up
- `asr_loaded: true` — VibeVoice-ASR fallback (used by /v1/transcribe)
- `llm_loaded: true` — Qwen (legacy fallback, not used in conversation flow)
- `gemma4.loaded: true` — sidecar reachable + warmed up
- `openvoice.reachable: true` — only if you started the OpenVoice sidecar

Recent sessions:

```bash
curl -sk https://YOUR-HOST/v1/conversation/sessions?limit=10 \
     -H 'xi-api-key: YOUR_KEY' | jq
```

Per-session audio diagnostics (clipping, silence, etc):

```bash
SID=sess-XXXXXXXXXX-YYYYYYYYYYYY
curl -sk "https://YOUR-HOST/v1/conversation/sessions/$SID/log?include_audio_metrics=true" \
     -H 'xi-api-key: YOUR_KEY' | jq
```
