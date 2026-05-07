# VibeVoice ASR — install guide + research log

Tested install procedure for Microsoft's **VibeVoice ASR**
(`microsoft/VibeVoice-ASR-HF`), researched end-to-end on a Hetzner
CPU host inside a clean PyTorch Docker container.

The companion script lives at:
- `engine/scripts/install_vibevoice_asr.sh`

## TL;DR — the working install

```bash
./engine/scripts/install_vibevoice_asr.sh /workspace /workspace/hf-cache
```

That's it. Three pieces:
1. `pip install transformers==5.3.0 accelerate soundfile imageio-ffmpeg librosa`
2. `snapshot_download("microsoft/VibeVoice-ASR-HF")` (~16 GB, ~3 min on a fast pipe)
3. `from transformers import VibeVoiceAsrForConditionalGeneration`

No patches. No git clones. No editable installs. No special flags
beyond `HF_HUB_DISABLE_XET=1`.

## The big finding: ignore the `microsoft/VibeVoice` GitHub repo

The original install plan was based on the GitHub package
(`pip install -e VibeVoice/`) plus several patches — a transformers
JSON-serialization fix, a `modeling_utils` import rewrite, and a
`peft` version pin. That entire path is **wrong for the ASR
checkpoint** and silently produces a broken model.

What we observed on the Hetzner host:

- The GitHub `vibevoice.modular.modeling_vibevoice_asr` declares a
  state_dict with prefixes `model.acoustic_tokenizer.encoder.*`,
  `model.acoustic_tokenizer.decoder.*`, `model.acoustic_connector.*`.
- The **actual HF checkpoint** uses prefixes
  `acoustic_tokenizer_encoder.*`, `multi_modal_projector.*`, and has
  **no** decoder weights at all (ASR doesn't need them).
- Loading the checkpoint via the GitHub class produces:
  > Some weights of VibeVoiceASRForConditionalGeneration were not
  > initialized from the model checkpoint at microsoft/VibeVoice-ASR-HF
  > and are newly initialized: ['lm_head.weight',
  > 'model.acoustic_connector.fc1.bias', ... (>500 keys) ...]
  These keys end up randomly initialized — every Hetzner load showed
  `mean_abs=0.0e+00` for the encoder weights. The model would run
  but produce nonsense.

The HF model card says (we found this during the research, not
upfront):

> VibeVoice-ASR is available as of v5.3.0 of Transformers!
> ```
> pip install "transformers>=5.3.0"
> ```
> ```python
> from transformers import VibeVoiceAsrForConditionalGeneration
> ```

Loading via the **transformers 5.3.0 native class**:
- All 901 weight keys map cleanly. Zero "newly initialized" warnings.
- Param count: **8.33 B** (the published checkpoint is encoder-only +
  Qwen 2.5 7B; the GitHub class assumed a full 13.1 B with decoder).
- Sample weights are non-zero: encoder `mean_abs=4.5e-2`, Qwen
  `q_proj mean_abs=1.4e-2`. Both signs of a properly loaded
  checkpoint.
- `processor.apply_transcription_request` + `model.generate(max_new_tokens=5)`
  on synthetic 2-second silence produced
  `<|im_start|>assistant\n[{"` — the JSON ASR response prefix the
  model is trained to emit.

## Disk + memory requirements (measured)

| Resource | Value |
| --- | --- |
| HF cache `models--microsoft--VibeVoice-ASR-HF` | **15.53 GB** (8 safetensors shards: ~2.3 GB each except shard 8 which is 35 MB) |
| Python deps delta over `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime` | ~700 MB |
| **Disk total during install** | **~16 GB** plus the ~6 GB existing conda env |
| RAM, CPU bf16 inference | ~17 GB peak (8.33 B params × 2 bytes + activations) |
| RAM, CPU fp32 inference | ~33 GB peak — only run if you have it |
| VRAM, GPU bf16 inference | ~17 GB |

**The Pod's "20 GB quota too small for VibeVoice ASR" was correct
in spirit but conservative**: 16 GB for the model fits inside 20 GB.
The Pod-side errors (`model-00001-of-00008.safetensors does not
appear to exist`) were the symptom of a partial download — the HF
cache writes each shard as `*.safetensors.incomplete` first, and
`from_pretrained` doesn't see incomplete files. Free a couple of GB,
re-run with `resume_download=True`, and it works.

## Issues encountered (and resolutions)

### 1. WRONG INSTALL PATH — GitHub vibevoice package + transformers 4.51.3

**Symptom**: `from_pretrained` succeeds with a long warning about
hundreds of "newly initialized" weights. Inference proceeds but
produces meaningless output (random-init encoder).

**Root cause**: Architecture/key-name mismatch between the GitHub
`microsoft/VibeVoice` codebase (originally TTS-focused) and the
later-published `microsoft/VibeVoice-ASR-HF` checkpoint. Microsoft
did not update the public GitHub repo with the new ASR architecture
— they integrated it directly into HF transformers.

**Fix**: Use `transformers >= 5.3.0`, no GitHub repo, no patches.

### 2. `peft==0.13.0` rejected by `diffusers`

(Only matters if you go down the GitHub-repo path; we kept this
note for completeness in case anyone tries that route again.)

```
ImportError: peft>=0.17.0 is required for a normal functioning of
this module, but found peft==0.13.0.
```

The vibevoice GitHub repo's `pyproject.toml` lists `diffusers` as a
dep (used by `vibevoice.schedule.dpm_solver`). Modern diffusers
(>=0.30) requires peft>=0.17.0. Bumping `peft==0.17.0` resolves it
with no impact on core code (peft is only used in optional
`finetuning-asr/` scripts).

### 3. `Object of type dtype is not JSON serializable`

(Same: only matters on the wrong install path.)

`transformers/configuration_utils.py` line 934 does
`json.dumps(config_dict, indent=2, sort_keys=True)`. If config
contains a raw `torch.dtype`, it crashes. The fix is to add
`default=str`. The native `VibeVoiceAsrConfig` shipped in
`transformers==5.3.0` does not trigger this — it serializes
properly without the patch.

### 4. HF Xet backend GIL crash on concurrent shard downloads

Some Linux builds of `huggingface_hub`'s optional xet acceleration
crash under the parallelism of an 8-shard `snapshot_download`. We
set `HF_HUB_DISABLE_XET=1` in the install script, which forces the
classic HTTPS path — completely reliable, with negligible
throughput cost on a multi-Gbit pipe.

### 5. Misleading "20 GB Pod quota too small" framing

The brief estimated the model at "~46 GB" (likely an fp32 figure).
Actual on-disk size of the bf16 safetensors snapshot is **15.53
GB**, well under 20 GB. The Pod failures were partial downloads
caused by *transient* disk pressure (other workloads filling
`/workspace`) leaving `*.incomplete` shards.

## Deploy on RunPod GPU Pod

Pre-flight: confirm **at least 18 GB free** in `/workspace`:
```bash
df -h /workspace
```

Then:
```bash
ssh -i ~/.ssh/id_ed25519 -p 22133 root@194.68.245.175

# on the Pod:
cd /workspace

# pull the install script (or scp it from the project)
scp -P 22133 -i ~/.ssh/id_ed25519 \
    ./engine/scripts/install_vibevoice_asr.sh \
    root@194.68.245.175:/workspace/install_vibevoice_asr.sh

# back on the Pod:
chmod +x /workspace/install_vibevoice_asr.sh
/workspace/install_vibevoice_asr.sh /workspace /workspace/hf-cache
```

The script will install deps, download the model, and verify the
load on whatever device is available (CPU or GPU). On a GPU Pod
you can switch the verify step to GPU by editing the script's
final python block to add `device_map="auto"`.

If `snapshot_download` is interrupted, just rerun the script —
on `huggingface_hub >=0.27` partial shards resume automatically.

## Known limitations

- **GPU inference not yet verified by us.** The Hetzner research
  host has no GPU; we only verified CPU-bf16 load + 5-token
  generation. The HF model card's GPU example is the canonical
  reference. Plan to smoke-test on the Pod after deploy.
- **transformers 5.3.0 is a major version bump** from the rest of
  the project's stack. If `engine/` or other services pin
  `transformers<5`, the VibeVoice ASR service must run in its
  own Python env / container.
- **Xet disabled.** We disable HF Xet; this costs marginal
  download throughput but is reliable.
