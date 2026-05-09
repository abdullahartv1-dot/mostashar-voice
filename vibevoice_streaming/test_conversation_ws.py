"""
Smoke test for /v1/conversation/ws.

Sends a text turn (skipping ASR), receives the assistant text + TTS PCM,
prints latency stats and saves the audio to /tmp/conv_out.wav.

Run on the pod:
    python /workspace/test_conversation_ws.py
"""
import asyncio
import json
import time
import wave
import struct

import websockets


WS_URL = "ws://127.0.0.1:8080/v1/conversation/ws?api_key=mostashar-dev-key&voice_id=hamed_saudi"
SAMPLE_RATE = 24000


async def main():
    pcm_chunks = []
    transcript = ""
    response_text = ""
    ttfa_ms = 0.0
    total_ms = 0.0

    async with websockets.connect(WS_URL, max_size=None) as ws:
        # Wait for server "ready"
        first = await ws.recv()
        print(f"server: {first}")

        t0 = time.time()
        await ws.send(json.dumps({"type": "text", "content": "ما هي عاصمة المملكة العربية السعودية؟"}))

        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                pcm_chunks.append(msg)
                continue
            data = json.loads(msg)
            print(f"server: {data}")
            if data.get("type") == "transcript":
                transcript = data.get("text", "")
            elif data.get("type") == "response_text":
                response_text = data.get("text", "")
            elif data.get("type") == "turn_done":
                ttfa_ms = data.get("ttfa_ms", 0)
                total_ms = data.get("total_ms", 0)
                break
            elif data.get("type") == "error":
                print("ERROR:", data.get("message"))
                return

    pcm = b"".join(pcm_chunks)
    print()
    print("=" * 60)
    print(f"transcript     : {transcript or '(text input — no STT)'}")
    print(f"response       : {response_text}")
    print(f"TTFA           : {ttfa_ms} ms")
    print(f"total turn     : {total_ms} ms")
    print(f"audio bytes    : {len(pcm)}")
    print(f"audio duration : {len(pcm)/2/SAMPLE_RATE:.2f} s")
    print(f"wall total     : {(time.time()-t0)*1000:.0f} ms")

    if pcm:
        with wave.open("/tmp/conv_out.wav", "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)
        print("saved → /tmp/conv_out.wav")


if __name__ == "__main__":
    asyncio.run(main())
