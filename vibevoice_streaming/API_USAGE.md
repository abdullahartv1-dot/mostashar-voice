# Mostashar Voice API — Usage Examples

ElevenLabs-style REST API. Consumable from **NestJS, OpenAI-style SDK, browser fetch, Python, curl** — anything that speaks HTTP.

## Auth

Set `MV_API_KEY=secret123` env var when starting the server. Then send:

```
xi-api-key: secret123
```

Or:

```
Authorization: Bearer secret123
```

If `MV_API_KEY` is unset, auth is **disabled** (development mode).

For WebSocket endpoint use `?api_key=secret123` query param (browsers can't set headers on WS).

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/health` | Server health |
| `GET` | `/v1/models` | Available models |
| `GET` | `/v1/voices` | List all voices |
| `GET` | `/v1/voices/{voice_id}` | Voice metadata |
| `GET` | `/v1/voices/{voice_id}/preview` | Reference audio (WAV) |
| `POST` | `/v1/voices/add` | Clone new voice (multipart) |
| `DELETE` | `/v1/voices/{voice_id}` | Delete voice |
| `POST` | `/v1/text-to-speech/{voice_id}` | Generate full WAV (non-streaming) |
| `POST` | `/v1/text-to-speech/{voice_id}/stream` | Stream PCM via HTTP chunked |
| `WS` | `/v1/text-to-speech/{voice_id}/ws` | Stream via WebSocket (TTFA + meta) |
| `POST` | `/v1/speech-to-text` | STT (multipart audio) — plain text |
| `POST` | `/v1/transcribe` | STT + diarization + timestamps |

Auto-generated Swagger UI: **http://YOUR_HOST/docs**

---

## cURL examples

### 1. List voices
```bash
curl http://localhost:8080/v1/voices \
  -H "xi-api-key: secret123"
```

Response:
```json
{
  "voices": [
    {
      "voice_id": "default",
      "name": "Default",
      "language": "ar",
      "dur_s": 21.24,
      "preview_url": "/v1/voices/default/preview",
      "created_at": null
    }
  ]
}
```

### 2. Clone a voice
```bash
curl -X POST http://localhost:8080/v1/voices/add \
  -H "xi-api-key: secret123" \
  -F "name=أحمد السعودي" \
  -F "language=ar" \
  -F "start_s=0" \
  -F "end_s=20" \
  -F "files=@reference.mp3"
```

Response:
```json
{ "voice_id": "احمد-السعودي-7f3a9c", "name": "أحمد السعودي", "language": "ar", "dur_s": 20.0 }
```

### 3. Generate full audio (non-streaming)
```bash
curl -X POST http://localhost:8080/v1/text-to-speech/default \
  -H "xi-api-key: secret123" \
  -H "Content-Type: application/json" \
  -d '{"text": "مرحبا بكم في منصة مستشار", "voice_settings": {"diffusion_steps": 15}}' \
  --output speech.wav
```

### 4. Streaming (HTTP chunked)
```bash
curl -X POST http://localhost:8080/v1/text-to-speech/default/stream \
  -H "xi-api-key: secret123" \
  -H "Content-Type: application/json" \
  -d '{"text": "هذا اختبار للبث المباشر", "output_format": "wav"}' \
  --output stream.wav
```

For `output_format=pcm_24000` you receive raw PCM16LE 24kHz; pipe to `aplay -r 24000 -f S16_LE` on Linux.

### 5. Speech-to-Text (simple)
```bash
curl -X POST http://localhost:8080/v1/speech-to-text \
  -H "xi-api-key: secret123" \
  -F "file=@meeting.mp3" \
  -F "language=ar"
```

Response:
```json
{ "text": "السلام عليكم. هذا اختبار للنسخ.", "language": "ar", "duration_s": 12.3 }
```

### 6. Transcribe with diarization + timestamps
```bash
curl -X POST http://localhost:8080/v1/transcribe \
  -H "xi-api-key: secret123" \
  -F "file=@interview.mp3" \
  -F "language=ar"
```

Response:
```json
{
  "text": "كنت مليونير وأنا عمري ستة وعشرين سنة. ستة وعشرين سنة. أي نعم.",
  "language": "ar",
  "duration_s": 21.24,
  "speakers_count": 2,
  "segments": [
    {"start_time": 0.0,  "end_time": 2.91, "speaker_id": 0, "text": "كنت مليونير وأنا عمري ستة وعشرين سنة."},
    {"start_time": 2.91, "end_time": 3.72, "speaker_id": 1, "text": "ستة وعشرين سنة."},
    {"start_time": 3.72, "end_time": 4.14, "speaker_id": 0, "text": "أي نعم."}
  ],
  "generation_ms": 4442
}
```

Powered by **VibeVoice-ASR** (Microsoft) — 50 languages, up to 60 min single-pass, RTF ~0.06.

---

## Node.js / NestJS examples

### Non-streaming TTS

```ts
// nestjs service
import { Injectable, HttpException } from "@nestjs/common";

@Injectable()
export class TTSService {
  private readonly baseUrl = process.env.MOSTASHAR_API_URL!  // e.g. http://pod.example:8080
  private readonly apiKey = process.env.MOSTASHAR_API_KEY!

  async synthesize(voiceId: string, text: string): Promise<Buffer> {
    const r = await fetch(`${this.baseUrl}/v1/text-to-speech/${voiceId}`, {
      method: "POST",
      headers: {
        "xi-api-key": this.apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
        voice_settings: { diffusion_steps: 15, cfg_scale: 1.8 },
      }),
    })
    if (!r.ok) throw new HttpException(`TTS failed: ${r.status}`, r.status)
    return Buffer.from(await r.arrayBuffer())
  }
}
```

### Streaming TTS (chunked WAV → HTTP response)

```ts
// nestjs controller — proxy stream to client
import { Controller, Get, Query, Res } from "@nestjs/common"
import { Response } from "express"

@Controller("speak")
export class SpeakController {
  @Get()
  async speak(@Query("voice") voice: string, @Query("text") text: string, @Res() res: Response) {
    const upstream = await fetch(
      `${process.env.MOSTASHAR_API_URL}/v1/text-to-speech/${voice}/stream`,
      {
        method: "POST",
        headers: { "xi-api-key": process.env.MOSTASHAR_API_KEY!, "Content-Type": "application/json" },
        body: JSON.stringify({ text, output_format: "wav" }),
      }
    )
    res.set("Content-Type", "audio/wav")
    if (!upstream.body) return res.status(500).end()
    const reader = upstream.body.getReader()
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      res.write(value)
    }
    res.end()
  }
}
```

### WebSocket streaming (full TTFA control)

```ts
import WebSocket from "ws"

const ws = new WebSocket(`ws://${HOST}/v1/text-to-speech/default/ws?api_key=secret123`)
ws.on("open", () => {
  ws.send(JSON.stringify({ text: "مرحبا", voice_settings: { diffusion_steps: 15 } }))
})
ws.on("message", (data, isBinary) => {
  if (!isBinary) {
    const msg = JSON.parse(data.toString())
    if (msg.type === "ttfa") console.log(`TTFA: ${msg.ms}ms`)
    if (msg.type === "done") console.log(`Done: ${msg.total_wall_ms}ms`)
  } else {
    // raw PCM16LE 24kHz audio chunk → send to player or save to file
  }
})
```

---

## Python examples

```python
import requests

# List voices
r = requests.get("http://localhost:8080/v1/voices", headers={"xi-api-key": "secret123"})
print(r.json())

# Clone
with open("ref.mp3", "rb") as f:
    r = requests.post(
        "http://localhost:8080/v1/voices/add",
        headers={"xi-api-key": "secret123"},
        data={"name": "Ahmed", "language": "ar", "start_s": 0, "end_s": 20},
        files={"files": f},
    )
voice_id = r.json()["voice_id"]

# Generate full
r = requests.post(
    f"http://localhost:8080/v1/text-to-speech/{voice_id}",
    headers={"xi-api-key": "secret123"},
    json={"text": "أهلاً وسهلاً"},
)
with open("out.wav", "wb") as f:
    f.write(r.content)

# Streaming
with requests.post(
    f"http://localhost:8080/v1/text-to-speech/{voice_id}/stream",
    headers={"xi-api-key": "secret123"},
    json={"text": "اختبار البث", "output_format": "pcm_24000"},
    stream=True,
) as r:
    with open("stream.pcm", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
```

---

## Browser fetch example

```js
// Streaming WAV — playable with native <audio>
const r = await fetch(`${API}/v1/text-to-speech/${voiceId}/stream`, {
  method: "POST",
  headers: { "xi-api-key": API_KEY, "Content-Type": "application/json" },
  body: JSON.stringify({ text: "السلام عليكم", output_format: "wav" }),
})
const url = URL.createObjectURL(await r.blob())
new Audio(url).play()
```

For low-latency streaming with Web Audio API, use the WebSocket endpoint (`/v1/text-to-speech/{voice_id}/ws`) which emits PCM chunks as the model produces them.

---

## Voice settings

```json
{
  "voice_settings": {
    "cfg_scale": 1.8,         // 1.0-3.0  (higher = more faithful to reference voice)
    "diffusion_steps": 15,    // 5-60     (lower = faster, slight quality drop)
    "seed": 42                // for reproducibility
  }
}
```

Defaults are tuned for Arabic + Large model. `diffusion_steps=15` is the sweet spot for TTFA.

---

## Output formats

| `output_format` | Returns | Use case |
|---|---|---|
| `pcm_24000` | Raw PCM16LE 24kHz mono (chunked) | Lowest overhead, browser Web Audio API |
| `wav` | WAV with header + PCM | `<audio>` tag, command-line `aplay`, save as file |

---

## Error responses

All errors return JSON:
```json
{ "detail": { "error": { "code": "voice_not_found", "message": "...", "status": 404 } } }
```

| HTTP | Reason |
|---|---|
| 400 | Bad request (e.g. text too short, audio decode failed, region <3s) |
| 401 | `invalid_api_key` |
| 403 | Forbidden (e.g. cannot delete `default` voice) |
| 404 | Voice / endpoint not found |
| 422 | Validation error (Pydantic) |
| 500 | Internal error |

---

## Performance

| Metric | Value (RTX PRO 6000) |
|---|---|
| **TTS** TTFA (warm) | ~100ms via streaming endpoint |
| **TTS** RTF | ~0.5 (faster than realtime) |
| **STT** RTF | ~0.06 (15× faster than realtime) |
| **STT** wall (20s audio) | ~1.5s |
| GPU mem (TTS only) | ~19 GB |
| GPU mem (TTS + ASR) | ~36 GB |
| Concurrent requests | 1 currently — needs queue/worker for multi-user |

For production deployment with multiple concurrent users, run multiple workers behind a load balancer or use FP8 quantization to fit more workers per GPU.
