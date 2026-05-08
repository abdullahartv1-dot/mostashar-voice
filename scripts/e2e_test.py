"""E2E verification: run every tool on 01.mp3 and save results."""
import json
import time
import os
import sys
from pathlib import Path
import httpx

sys.stdout.reconfigure(encoding='utf-8')

API = "http://127.0.0.1:8000"  # via SSH tunnel
KEY = os.environ.get("VS_API_KEY", "dev-key-change-me")
HDR = {"X-API-Key": KEY}

ROOT = Path("C:/Users/Admin/Desktop/project/x")
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AUDIO = Path("C:/Users/Admin/Downloads/01.mp3")
if not AUDIO.exists():
    print(f"ERROR: {AUDIO} not found")
    sys.exit(1)

TEST_TEXT = "مرحباً بكم في تجربة استنساخ الصوت العربي. هذا الصوت تم توليده بالكامل."

results = {
    "audio_file": str(AUDIO),
    "audio_size_mb": round(AUDIO.stat().st_size / 1024 / 1024, 2),
    "test_text": TEST_TEXT,
    "rate_per_hour": 0.27,
    "stt_runs": [],
    "tts_runs": [],
    "started_at": time.time(),
}

# 1) Upload once (any STT tool can reuse this audio_path)
print("=== Uploading audio ===")
with AUDIO.open("rb") as fp:
    r = httpx.post(
        f"{API}/api/upload",
        files={"file": (AUDIO.name, fp, "audio/mpeg")},
        headers=HDR,
        timeout=120,
    )
upload = r.json()
print(f"  job_id: {upload.get('job_id')}, duration: {upload.get('duration')}s")
audio_path = upload["audio_path"]
results["upload"] = upload

# 2) Run each STT tool
STT_TOOLS = ["whisper-large-v3", "whisper-turbo", "nemo-canary", "vibevoice-asr"]
# vibevoice-asr is known-experimental (HF download throttled / 5GB model + 15GB Qwen 7B);
# cap it short so we don't hang the test session.
STT_TIMEOUTS = {"vibevoice-asr": 60}
last_successful_job_id = None
for tool in STT_TOOLS:
    print(f"\n=== STT: {tool} ===")
    t0 = time.time()
    try:
        r = httpx.post(
            f"{API}/api/process",
            json={"audio_path": audio_path, "stt_tool": tool, "diar_tool": "ecapa-tdnn"},
            headers=HDR,
            timeout=STT_TIMEOUTS.get(tool, 600),
        )
        wall = time.time() - t0
        if r.status_code == 200:
            data = r.json()
            run = {
                "tool": tool,
                "status": "PASS",
                "wall_sec": round(wall, 2),
                "engine_timings": data.get("timings"),
                "cost": data.get("cost"),
                "speakers_detected": data.get("speakers_detected"),
                "n_segments": len(data.get("segments", [])),
                "first_3_segments": data.get("segments", [])[:3],
                "job_id": data.get("job_id"),
            }
            last_successful_job_id = data.get("job_id")
            print(f"  PASS in {wall:.1f}s, {len(data.get('segments', []))} segments, ${data.get('cost', {}).get('total_usd', 0)}")
            for seg in data.get("segments", [])[:3]:
                print(f"    [{seg.get('start', 0):.2f}s] {seg.get('speaker', '?')}: {seg.get('text', '')[:80]}")
        else:
            run = {
                "tool": tool,
                "status": "FAIL",
                "wall_sec": round(wall, 2),
                "http_status": r.status_code,
                "error": r.text[:500],
            }
            print(f"  FAIL HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        wall = time.time() - t0
        run = {
            "tool": tool,
            "status": "ERROR",
            "wall_sec": round(wall, 2),
            "error": str(e)[:500],
        }
        print(f"  ERROR: {e}")
    results["stt_runs"].append(run)

# 3) Run each TTS tool against last_successful_job_id
TTS_TOOLS = ["vibevoice-1.5b", "f5-tts", "xtts-v2", "vibevoice-large"]
if last_successful_job_id:
    print(f"\nUsing job_id={last_successful_job_id} for TTS clone tests")
    for tool in TTS_TOOLS:
        print(f"\n=== TTS: {tool} ===")
        t0 = time.time()
        try:
            r = httpx.post(
                f"{API}/api/clone",
                json={
                    "job_id": last_successful_job_id,
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
                    "error": r.text[:500],
                }
                print(f"  FAIL HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            wall = time.time() - t0
            run = {
                "tool": tool,
                "status": "ERROR",
                "wall_sec": round(wall, 2),
                "error": str(e)[:500],
            }
            print(f"  ERROR: {e}")
        results["tts_runs"].append(run)

# 4) Save final summary
results["finished_at"] = time.time()
results["total_session_cost_usd"] = round(sum(
    (r.get("cost", {}).get("total_usd", 0) if r.get("status") == "PASS" else 0)
    for r in results["stt_runs"]
) + sum(
    (r.get("cost_usd", 0) if r.get("status") == "PASS" else 0)
    for r in results["tts_runs"]
), 4)

out = RESULTS_DIR / f"e2e_test_{int(time.time())}.json"
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n=== Saved to: {out} ===")

# Print compact summary
print("\n" + "="*70)
print("STT Summary")
print("="*70)
for r in results["stt_runs"]:
    status = r.get("status")
    if status == "PASS":
        print(f"  {r['tool']:25} PASS  {r['wall_sec']:6.1f}s  ${r['cost']['total_usd']}")
    else:
        print(f"  {r['tool']:25} {status}  {r['wall_sec']:6.1f}s")

print("\n" + "="*70)
print("TTS Summary")
print("="*70)
for r in results["tts_runs"]:
    status = r.get("status")
    if status == "PASS":
        print(f"  {r['tool']:25} PASS  {r['wall_sec']:6.1f}s  ${r['cost_usd']}  output: {r['output_duration']}s")
    else:
        print(f"  {r['tool']:25} {status}  {r['wall_sec']:6.1f}s")

print(f"\nTotal session cost: ${results['total_session_cost_usd']}")
