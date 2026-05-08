"""Re-run TTS tests against an existing job_id (with SPEAKER_0 sample)."""
import json
import time
import os
import sys
from pathlib import Path
import httpx

sys.stdout.reconfigure(encoding='utf-8')

API = "http://127.0.0.1:8000"
KEY = os.environ.get("VS_API_KEY", "dev-key-change-me")
HDR = {"X-API-Key": KEY}

ROOT = Path("C:/Users/Admin/Desktop/project/x")
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AUDIO = Path("C:/Users/Admin/Downloads/01.mp3")
TEST_TEXT = "مرحباً بكم في تجربة استنساخ الصوت العربي. هذا الصوت تم توليده بالكامل."

# Step 1: upload + process to get a fresh job_id with SPEAKER_0 sample
print("=== Uploading + processing for fresh job_id ===")
with AUDIO.open("rb") as fp:
    r = httpx.post(f"{API}/api/upload", files={"file": (AUDIO.name, fp, "audio/mpeg")}, headers=HDR, timeout=120)
upload = r.json()
print(f"  job_id={upload['job_id']}")
audio_path = upload["audio_path"]

r = httpx.post(f"{API}/api/process", json={"audio_path": audio_path, "stt_tool": "whisper-turbo", "diar_tool": "ecapa-tdnn"}, headers=HDR, timeout=600)
proc = r.json()
job_id = proc["job_id"]
print(f"  process job_id={job_id}, speakers={proc.get('speakers_detected')}")

results = {"job_id": job_id, "tts_runs": []}

# Step 2: try each TTS
TTS_TOOLS = ["vibevoice-1.5b", "f5-tts", "xtts-v2", "vibevoice-large"]
for tool in TTS_TOOLS:
    print(f"\n=== TTS: {tool} ===")
    t0 = time.time()
    try:
        r = httpx.post(
            f"{API}/api/clone",
            json={
                "job_id": job_id,
                "speaker_id": "SPEAKER_0",
                "text": TEST_TEXT,
                "tts_tool": tool,
                "diffusion_steps": 40,
                "cfg_scale": 1.5,
            },
            headers=HDR,
            timeout=600,
        )
        wall = time.time() - t0
        if r.status_code == 200:
            data = r.json()
            run = {
                "tool": tool,
                "status": "PASS",
                "wall_sec": round(wall, 2),
                "elapsed": data.get("elapsed"),
                "output_duration": data.get("duration"),
                "rtf": data.get("rtf"),
                "cost_usd": data.get("cost_usd"),
                "audio_url": data.get("audio_url"),
            }
            print(f"  PASS: {data.get('elapsed', 0):.1f}s, output {data.get('duration', 0):.1f}s, ${data.get('cost_usd', 0)}, url={data.get('audio_url')}")
        else:
            run = {
                "tool": tool,
                "status": "FAIL",
                "wall_sec": round(wall, 2),
                "http_status": r.status_code,
                "error": r.text[:1000],
            }
            print(f"  FAIL HTTP {r.status_code}: {r.text[:300]}")
    except Exception as e:
        wall = time.time() - t0
        run = {"tool": tool, "status": "ERROR", "wall_sec": round(wall, 2), "error": str(e)[:500]}
        print(f"  ERROR: {e}")
    results["tts_runs"].append(run)

out = RESULTS_DIR / f"tts_only_{int(time.time())}.json"
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSaved: {out}")
