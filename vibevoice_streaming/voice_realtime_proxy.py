"""Realtime conversation proxy — Whisper + OpenAI + ElevenLabs stack.

The 'مكالمات' tab WS endpoint /v1/realtime/ws used to bridge directly to
OpenAI's Realtime API (one model in, audio out). Per user request the
stack is now:

  Mic PCM16 → Whisper / gpt-4o-mini-transcribe (STT)
            → OpenAI Chat Completions (LLM + MCP function calling, via
              voice_openai_loop.run_voice_turn_openai)
            → ElevenLabs streaming TTS
            → Speaker PCM

We keep the same WebSocket event protocol the frontend already speaks
(user_transcript, assistant_text_delta/done, response_done, binary
PCM frames, vad_speech_started/stopped) so RealtimeClient on the
frontend doesn't need to change beyond dropping the 16→24 kHz
upsample (server now does Whisper's preferred 16 kHz directly).

VAD is driven by the frontend: it sends `{"type":"commit"}` when the
user goes silent for SILENCE_THRESHOLD_MS. The backend buffers audio
until that commit, runs the pipeline, streams the response, and
returns to listening.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import List

import numpy as np

try:
    from voice_openai_loop import run_voice_turn_openai  # type: ignore[no-redef]
    from voice_session import VoiceSession  # type: ignore[no-redef]
    from voice_elevenlabs_tts import elevenlabs_stream  # type: ignore[no-redef]
    from voice_openai_stt import openai_transcribe  # type: ignore[no-redef]
except ImportError:
    from .voice_openai_loop import run_voice_turn_openai
    from .voice_session import VoiceSession
    from .voice_elevenlabs_tts import elevenlabs_stream
    from .voice_openai_stt import openai_transcribe


# Frontend captures at 16 kHz mono PCM16 LE — same rate Whisper /
# gpt-4o-mini-transcribe expect. No resampling on either side.
INPUT_SAMPLE_RATE = 16000


async def run_realtime_session(client_ws, session: VoiceSession) -> None:
    """Handle one /v1/realtime/ws client lifetime.

    Protocol:
      • Client → Server: binary PCM16 LE @ 16 kHz, plus
          {"type":"commit"}            user finished speaking
          {"type":"cancel"}            abort in-flight response
          {"type":"text","content":..} typed message (no audio)
      • Server → Client:
          {"type":"ready"}
          {"type":"user_transcript","text":...}
          {"type":"assistant_text_delta","text":...}  // we emit full
                                                       // text in one go
          {"type":"assistant_text_done","text":...}
          {"type":"response_done"}
          {"type":"tool_call_started","tool":...}
          {"type":"tool_result","tool":...,"ok":...}
          {"type":"error","message":...}
          binary PCM16 LE @ 24 kHz (ElevenLabs output format)

    Pre-warm MCP cache so the first tool call doesn't pay setup cost.
    """
    if session.mcp_connected and not session._initialized:
        await session.ensure_mcp()
        try:
            await session.get_workflows()
        except Exception:
            pass
        try:
            await session.get_calendars()
        except Exception:
            pass

    await client_ws.send_json({"type": "ready"})
    print("[realtime] session ready (whisper+openai+elevenlabs stack)", flush=True)

    audio_buffer = bytearray()
    processing = False
    interrupt_flag = {"value": False}

    while True:
        try:
            msg = await client_ws.receive()
        except Exception:
            break
        if msg.get("type") == "websocket.disconnect":
            print("[realtime] client disconnected", flush=True)
            break

        if "bytes" in msg and msg["bytes"] is not None:
            # Drop audio while processing — the response is in flight,
            # the user's next utterance starts a new turn.
            if not processing:
                audio_buffer.extend(msg["bytes"])
            continue

        text_msg = msg.get("text")
        if not text_msg:
            continue
        try:
            data = json.loads(text_msg)
        except Exception:
            continue

        mtype = data.get("type")

        if mtype == "commit":
            if processing or not audio_buffer:
                continue
            processing = True
            interrupt_flag["value"] = False
            pcm_bytes = bytes(audio_buffer)
            audio_buffer.clear()
            try:
                await _process_turn(
                    client_ws, session, pcm_bytes, interrupt_flag,
                )
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                print(f"[realtime] turn failed: {e}", flush=True)
                try:
                    await client_ws.send_json({
                        "type": "error", "message": f"turn failed: {e}",
                    })
                except Exception:
                    pass
            finally:
                processing = False

        elif mtype == "cancel":
            audio_buffer.clear()
            interrupt_flag["value"] = True

        elif mtype == "text":
            content = (data.get("content") or "").strip()
            if not content or processing:
                continue
            processing = True
            interrupt_flag["value"] = False
            try:
                await _process_text(client_ws, session, content, interrupt_flag)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                print(f"[realtime] text turn failed: {e}", flush=True)
            finally:
                processing = False


async def _process_turn(
    client_ws,
    session: VoiceSession,
    pcm_bytes: bytes,
    interrupt_flag: dict,
) -> None:
    """Audio → text → LLM → TTS → audio."""
    # PCM16 LE → float32 [-1, 1]
    audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    if audio_i16.size < int(0.2 * INPUT_SAMPLE_RATE):
        print(f"[realtime] audio too short ({audio_i16.size / INPUT_SAMPLE_RATE:.2f}s)",
              flush=True)
        return
    audio_f32 = audio_i16.astype(np.float32) / 32768.0

    # STT
    user_text = ""
    try:
        user_text = await openai_transcribe(
            audio_f32,
            prompt="هذه محادثة عربية فصحى بلهجة سعودية.",
        )
    except Exception as e:  # noqa: BLE001
        print(f"[realtime] STT error: {e}", flush=True)

    if not user_text:
        await client_ws.send_json({
            "type": "error",
            "message": "لم أسمع كلامك بوضوح، حاول ثانية.",
        })
        return

    await client_ws.send_json({"type": "user_transcript", "text": user_text})

    if interrupt_flag["value"]:
        return

    await _run_llm_and_stream_tts(client_ws, session, user_text, interrupt_flag)


async def _process_text(
    client_ws,
    session: VoiceSession,
    user_text: str,
    interrupt_flag: dict,
) -> None:
    """Typed text input — skip STT, run LLM + TTS."""
    await client_ws.send_json({"type": "user_transcript", "text": user_text})
    await _run_llm_and_stream_tts(client_ws, session, user_text, interrupt_flag)


async def _run_llm_and_stream_tts(
    client_ws,
    session: VoiceSession,
    user_text: str,
    interrupt_flag: dict,
) -> None:
    """Shared LLM + TTS pipeline for both audio and text turns."""
    # LLM (OpenAI Chat Completions + MCP tool calling, via the same
    # loop the chat tab uses — keeps the multi-tool + retry +
    # giving-up-detection logic identical across both tabs).
    try:
        result = await run_voice_turn_openai(user_text=user_text, session=session)
    except Exception as e:  # noqa: BLE001
        print(f"[realtime] LLM error: {e}", flush=True)
        await client_ws.send_json({
            "type": "error", "message": "تعذّر توليد الرد.",
        })
        return

    # Surface tool events to the frontend (UI feedback).
    for ev in result.tool_events:
        try:
            await client_ws.send_json({
                "type": ev.get("type", "tool_event"), **ev,
            })
        except Exception:
            pass

    response_text = result.text or ""
    if not response_text:
        await client_ws.send_json({"type": "response_done"})
        return

    # Emit the full assistant text as one "delta" + a "done" so the
    # frontend shows it whether or not the user listens to the audio.
    await client_ws.send_json({
        "type": "assistant_text_delta", "text": response_text,
    })
    await client_ws.send_json({
        "type": "assistant_text_done", "text": response_text,
    })

    # TTS via ElevenLabs — streams PCM16 LE 24 kHz mono.
    chunk_count = 0
    try:
        async for chunk in elevenlabs_stream(response_text):
            if interrupt_flag["value"]:
                print("[realtime] interrupted by user", flush=True)
                break
            try:
                await client_ws.send_bytes(chunk)
                chunk_count += 1
            except Exception as e:  # noqa: BLE001
                print(f"[realtime] send_bytes failed: {e}", flush=True)
                break
    except Exception as e:  # noqa: BLE001
        print(f"[realtime] TTS error: {e}", flush=True)

    print(f"[realtime] turn complete: {chunk_count} audio chunks streamed",
          flush=True)
    try:
        await client_ws.send_json({"type": "response_done"})
    except Exception:
        pass
