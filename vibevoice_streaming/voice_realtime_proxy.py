"""OpenAI Realtime API proxy — direct audio↔audio low-latency conversation.

How this differs from the existing pipeline:

  ┌─ Current /v1/conversation/ws ────────────────────────────┐
  │  Browser → Whisper (STT) → OpenAI text (LLM) → VibeVoice │
  │  (TTS) → Browser. ~1500 ms TTFA, custom Arabic voice.    │
  └──────────────────────────────────────────────────────────┘

  ┌─ This /v1/realtime/ws ───────────────────────────────────┐
  │  Browser → OpenAI Realtime (audio in, audio out, one     │
  │  model, function calls included) → Browser. ~300 ms      │
  │  TTFA, OpenAI's voice (alloy / verse / coral / etc).     │
  │  We sit in between only to (a) hide the key, (b) inject  │
  │  the user's Moshaar MCP tools.                           │
  └──────────────────────────────────────────────────────────┘

Architecture:
  - Browser opens our WS at /v1/realtime/ws (passes mcp_url / mcp_key).
  - We open ANOTHER WS to wss://api.openai.com/v1/realtime.
  - Forward audio chunks user→OpenAI; forward audio/text deltas
    OpenAI→user.
  - Convert our 16-tool VOICE_TOOLS catalog into OpenAI Realtime's
    `session.tools` format on connect.
  - When OpenAI emits `response.function_call_arguments.done`, run the
    tool via the same MoshaarMCPClient the chat path uses, then send
    `conversation.item.create` (function_call_output) + `response.create`
    so OpenAI continues with the tool result.

Pricing (as of 2026-05): input audio $0.06/min, output audio $0.24/min,
plus text tokens. A typical 5-minute conversation ≈ $0.50-$1.

To enable: pass MV_OPENAI_KEY env var. Frontend hits /v1/realtime/ws
when the user opens the new "مكالمات" tab.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

try:
    import websockets
    from websockets.asyncio.client import connect as ws_connect  # newer API
except Exception:  # noqa: BLE001
    websockets = None  # type: ignore[assignment]
    ws_connect = None  # type: ignore[assignment]

try:
    from voice_agent_tools import VOICE_TOOLS, TOOLS_BY_NAME, needs_confirmation  # type: ignore[no-redef]
    from voice_openai_loop import _FIELD_TYPES  # reuse the schema map
    from voice_react_loop import _execute_tool, _summarize_result  # reuse MCP exec
    from voice_session import VoiceSession  # type: ignore[no-redef]
    from voice_elevenlabs_tts import elevenlabs_stream  # type: ignore[no-redef]
except ImportError:
    from .voice_agent_tools import VOICE_TOOLS, TOOLS_BY_NAME, needs_confirmation
    from .voice_openai_loop import _FIELD_TYPES
    from .voice_react_loop import _execute_tool, _summarize_result
    from .voice_session import VoiceSession
    from .voice_elevenlabs_tts import elevenlabs_stream


log = logging.getLogger("voice_realtime_proxy")

REALTIME_URL = os.environ.get(
    "MV_OPENAI_REALTIME_URL",
    "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17",
)
# OpenAI voice options: alloy / ash / ballad / coral / echo / sage / shimmer / verse
REALTIME_VOICE = os.environ.get("MV_OPENAI_REALTIME_VOICE", "shimmer")

# TTS backend for the calls tab. "openai" uses OpenAI Realtime's native
# audio output (voice from MV_OPENAI_REALTIME_VOICE). "elevenlabs" tells
# the realtime session to produce text only and pipes that text through
# ElevenLabs streaming TTS — cleaner Arabic prosody, slightly higher
# TTFA (~+500 ms because we wait for sentence boundaries before TTS).
REALTIME_TTS_BACKEND = os.environ.get(
    "MV_REALTIME_TTS_BACKEND", "openai"
).lower().strip()
if REALTIME_TTS_BACKEND not in ("openai", "elevenlabs"):
    REALTIME_TTS_BACKEND = "openai"


# Detect sentence boundaries so we can send chunks to ElevenLabs early
# instead of waiting for the full response. Latin + Arabic terminators.
import re as _re
_SENTENCE_BOUNDARY = _re.compile(r"[.!?؟।]+\s+|[.!?؟।]+$")


SYSTEM_INSTRUCTIONS_AR = """\
أنت "سارة"، مساعدة صوتية لمنصة مستشار (إدارة قضايا قانونية وأعمال).
تتكلمين العربية الفصحى المبسّطة مع لمسة سعودية. ردودك قصيرة وعملية —
جملتان كحد أقصى عادةً.

أنتِ **مبادِرة**: اعملي بنفسك حتى تنجزي الطلب بدون أن تسألي المستخدم
عن خطوات وسيطة. استخدمي الأدوات بتسلسل: استدعي أداة، اقرئي نتيجتها،
ثم استدعي الأداة التالية تلقائياً حتى تكتمل المهمة.

قواعد السلوك:
- لا تذكري للمستخدم أبداً أنك تستخدمين "أدوات" أو IDs.
- إذا طلب المستخدم "كل المواعيد" أو "كل القضايا" أو "كل المهام":
  استدعي أداة الفهرسة (list_calendars / list_cases) أولاً ثم استدعي
  أدوات التفاصيل لكل عنصر، واجمعي النتائج. **لا تطلبي توضيحاً قبل
  المحاولة الأولى.**
- مهام متعددة الخطوات: استمري حتى الإنجاز. استدعي الأدوات التالية
  مباشرة بعد كل نتيجة، لا تستأذني.
- قبل عمليات التعديل (إنشاء/تحديث/حذف/إرسال): أوصفي العملية واطلبي
  تأكيداً. لا تنفّذي حتى أسمع "نعم/تمام/وافق".
- IDs والـ tokens لا تُذكر بالصوت. استخدمي الأسماء فقط.
- التواريخ النسبية حوّليها إلى ISO 8601 UTC قبل تمريرها للأدوات.
- عند خطأ، جرّبي مرة ثانية بمعطيات مختلفة قبل ما تخبري المستخدم بفشل.
"""


def _build_realtime_tools() -> List[Dict[str, Any]]:
    """Convert our VOICE_TOOLS catalog into Realtime's tools format.

    Realtime's tool spec is the same JSON-schema-based function-calling
    format as Chat Completions, just at the top level (not nested under
    "function:"). Re-use the field-type map from voice_openai_loop.
    """
    out: List[Dict[str, Any]] = []
    for vt in VOICE_TOOLS:
        props: Dict[str, Any] = {}
        for field in vt.required + vt.optional:
            spec = _FIELD_TYPES.get(field, {"type": "string"})
            props[field] = dict(spec)
        desc_parts = [vt.purpose_ar]
        if vt.notes:
            desc_parts.append("ملاحظات: " + vt.notes)
        if vt.needs_confirmation:
            desc_parts.append("⚠️ اطلب تأكيداً شفهياً قبل التنفيذ.")
        out.append({
            "type": "function",
            "name": vt.name,
            "description": " | ".join(desc_parts)[:1024],
            "parameters": {
                "type": "object",
                "properties": props,
                "required": list(vt.required),
            },
        })
    return out


async def run_realtime_session(
    client_ws,                     # FastAPI WebSocket (user side)
    session: VoiceSession,         # holds MCP credentials
) -> None:
    """Pump events between the browser and OpenAI Realtime.

    Two concurrent tasks:
      • user→openai: forward audio chunks, mode-control messages
      • openai→user: forward audio/text deltas, handle function calls

    The user-side WS speaks plain JSON for control + binary PCM frames
    for audio. The OpenAI WS speaks JSON for everything (audio chunks
    are base64-encoded inside `input_audio_buffer.append` events).
    """
    api_key = os.environ.get("MV_OPENAI_KEY", "").strip()
    if not api_key:
        await client_ws.send_json({
            "type": "error",
            "message": "OpenAI key not configured on the pod.",
        })
        return

    if ws_connect is None:
        await client_ws.send_json({
            "type": "error",
            "message": "websockets module not installed on pod.",
        })
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    print("[realtime] connecting to OpenAI...", flush=True)
    try:
        async with ws_connect(REALTIME_URL, additional_headers=headers) as openai_ws:
            await _configure_session(openai_ws, session)
            await client_ws.send_json({"type": "ready"})

            await asyncio.gather(
                _pump_user_to_openai(client_ws, openai_ws),
                _pump_openai_to_user(client_ws, openai_ws, session),
                return_exceptions=True,
            )
    except Exception as e:  # noqa: BLE001
        print(f"[realtime] session failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        try:
            await client_ws.send_json({
                "type": "error", "message": f"realtime failed: {e}",
            })
        except Exception:
            pass


async def _configure_session(openai_ws, session: VoiceSession) -> None:
    """Send session.update to OpenAI to set our voice, language, tools."""
    # Pre-warm MCP caches so tool execution doesn't add startup delay
    if session.mcp_connected:
        await session.ensure_mcp()
        try:
            await session.get_workflows()
        except Exception:
            pass
        try:
            await session.get_calendars()
        except Exception:
            pass

    tools = _build_realtime_tools() if (session.mcp_connected and session._initialized) else []

    # When ElevenLabs handles TTS, ask OpenAI for text only. We still
    # use the same Realtime model + tool calling + server VAD, just
    # without the (now wasted) audio synthesis on OpenAI's side.
    modalities = ["text"] if REALTIME_TTS_BACKEND == "elevenlabs" else ["text", "audio"]

    session_obj: Dict[str, Any] = {
        "modalities": modalities,
        "instructions": SYSTEM_INSTRUCTIONS_AR,
        "input_audio_format": "pcm16",
        "input_audio_transcription": {
            "model": "whisper-1",
        },
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 700,
            "create_response": True,
        },
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.8,
    }
    # Only set voice + output format when OpenAI is producing audio.
    if REALTIME_TTS_BACKEND == "openai":
        session_obj["voice"] = REALTIME_VOICE
        session_obj["output_audio_format"] = "pcm16"

    update_msg = {"type": "session.update", "session": session_obj}
    await openai_ws.send(json.dumps(update_msg))
    print(
        f"[realtime] session.update sent (tools={len(tools)}, "
        f"tts={REALTIME_TTS_BACKEND}, "
        f"voice={REALTIME_VOICE if REALTIME_TTS_BACKEND == 'openai' else 'elevenlabs'})",
        flush=True,
    )


async def _pump_user_to_openai(client_ws, openai_ws) -> None:
    """Forward user audio + control events to OpenAI."""
    while True:
        try:
            msg = await client_ws.receive()
        except Exception:
            return
        if msg.get("type") == "websocket.disconnect":
            print("[realtime] client disconnected", flush=True)
            return

        if "bytes" in msg and msg["bytes"] is not None:
            # Audio chunk — base64-encode and forward to OpenAI.
            b64 = base64.b64encode(msg["bytes"]).decode("ascii")
            await openai_ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": b64,
            }))
            continue

        text = msg.get("text")
        if not text:
            continue
        try:
            data = json.loads(text)
        except Exception:
            continue

        mtype = data.get("type")
        if mtype == "commit":
            # User finished speaking — let OpenAI know to start response.
            await openai_ws.send(json.dumps({
                "type": "input_audio_buffer.commit",
            }))
            await openai_ws.send(json.dumps({"type": "response.create"}))
        elif mtype == "cancel":
            # User interrupted — cancel any in-flight response.
            await openai_ws.send(json.dumps({"type": "response.cancel"}))
        elif mtype == "text":
            # Optional: send a typed message as a conversation item.
            content = (data.get("content") or "").strip()
            if content:
                await openai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": content}],
                    },
                }))
                await openai_ws.send(json.dumps({"type": "response.create"}))


async def _stream_elevenlabs_to_client(
    client_ws, text: str, audio_count: List[int],
) -> None:
    """Stream a chunk of text through ElevenLabs and forward PCM bytes
    to the user's WS. Used when REALTIME_TTS_BACKEND=elevenlabs."""
    if not text.strip():
        return
    try:
        async for chunk in elevenlabs_stream(text):
            try:
                await client_ws.send_bytes(chunk)
                audio_count[0] += 1
                if audio_count[0] == 1 or audio_count[0] % 25 == 0:
                    print(
                        f"[realtime/elevenlabs] forwarded audio chunk "
                        f"#{audio_count[0]} ({len(chunk)} bytes)", flush=True,
                    )
            except Exception as e:  # noqa: BLE001
                print(f"[realtime/elevenlabs] send_bytes failed: {e}", flush=True)
                break
    except Exception as e:  # noqa: BLE001
        print(f"[realtime/elevenlabs] stream error: {e}", flush=True)


async def _pump_openai_to_user(
    client_ws, openai_ws, session: VoiceSession,
) -> None:
    """Forward OpenAI events + audio to the user, handle function calls."""
    pending_args: Dict[str, str] = {}  # call_id → accumulating JSON args
    event_count = 0
    # mutable counter; using a list so the closure-like access in the
    # handler doesn't accidentally rebind a local var
    _audio_chunks_sent = [0]
    # When REALTIME_TTS_BACKEND=elevenlabs, accumulate OpenAI's text
    # deltas and flush each completed sentence into ElevenLabs streaming
    # TTS so the user hears the response with low latency.
    elevenlabs_buffer: List[str] = [""]

    try:
        async for raw in openai_ws:
            event_count += 1
            try:
                ev = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode())
            except Exception:
                print(f"[realtime] non-JSON event from OpenAI: {raw[:200]}", flush=True)
                continue
            et = ev.get("type", "")
            # First-time debug: log first few events + every error/important one
            if event_count <= 3 or et in (
                "error", "session.created", "session.updated",
                "response.done", "response.audio_transcript.done",
                "input_audio_buffer.speech_started",
                "response.function_call_arguments.done",
                "conversation.item.input_audio_transcription.completed",
            ):
                print(f"[realtime] openai event #{event_count}: {et}", flush=True)
                if et == "error":
                    print(f"[realtime] error detail: {ev.get('error')}", flush=True)

            # ⚠️ EVERYTHING BELOW MUST live at 12-space indent so it's
            # inside `async for raw in openai_ws`. A previous edit left
            # this block at 8-space indent, which made it run ONCE after
            # the loop exited (using whatever `et` happened to hold last)
            # — that's why audio.delta events stopped reaching the user
            # in the calls tab.
            if et == "response.audio.delta":
                audio_b64 = ev.get("delta", "")
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    await client_ws.send_bytes(audio_bytes)
                    _audio_chunks_sent[0] += 1
                    if _audio_chunks_sent[0] == 1 or _audio_chunks_sent[0] % 25 == 0:
                        print(
                            f"[realtime] forwarded audio chunk #{_audio_chunks_sent[0]} "
                            f"({len(audio_bytes)} bytes)", flush=True,
                        )
                except Exception as e:  # noqa: BLE001
                    print(f"[realtime] send_bytes failed: {e}", flush=True)
                    import traceback; traceback.print_exc()

            elif et == "response.audio_transcript.delta":
                # OpenAI audio path — text accompanies the generated audio.
                await client_ws.send_json({
                    "type": "assistant_text_delta", "text": ev.get("delta", ""),
                })

            elif et == "response.audio_transcript.done":
                await client_ws.send_json({
                    "type": "assistant_text_done",
                    "text": ev.get("transcript", ""),
                })

            elif et == "response.text.delta":
                # Pure-text modality (REALTIME_TTS_BACKEND=elevenlabs).
                # Forward the delta to the user's transcript bubble AND
                # accumulate it in the ElevenLabs buffer. When we cross
                # a sentence boundary, flush the completed sentence(s)
                # to ElevenLabs so playback starts before the full
                # response is even done generating.
                delta = ev.get("delta", "")
                if delta:
                    await client_ws.send_json({
                        "type": "assistant_text_delta", "text": delta,
                    })
                    elevenlabs_buffer[0] += delta
                    # Cut at the last sentence boundary; keep the tail
                    # in the buffer for the next iteration.
                    buf = elevenlabs_buffer[0]
                    last_end = -1
                    for m in _SENTENCE_BOUNDARY.finditer(buf):
                        last_end = m.end()
                    if last_end > 0:
                        to_speak = buf[:last_end].strip()
                        elevenlabs_buffer[0] = buf[last_end:]
                        if to_speak:
                            # Fire and don't await — TTS runs concurrently
                            # with the next OpenAI events.
                            asyncio.create_task(
                                _stream_elevenlabs_to_client(
                                    client_ws, to_speak, _audio_chunks_sent,
                                )
                            )

            elif et == "response.text.done":
                # Flush any remaining text in the buffer through ElevenLabs.
                tail = elevenlabs_buffer[0].strip()
                elevenlabs_buffer[0] = ""
                if tail:
                    asyncio.create_task(
                        _stream_elevenlabs_to_client(
                            client_ws, tail, _audio_chunks_sent,
                        )
                    )
                await client_ws.send_json({
                    "type": "assistant_text_done",
                    "text": ev.get("text", ""),
                })

            elif et == "conversation.item.input_audio_transcription.completed":
                await client_ws.send_json({
                    "type": "user_transcript",
                    "text": ev.get("transcript", ""),
                })

            elif et == "response.function_call_arguments.delta":
                cid = ev.get("call_id", "")
                pending_args[cid] = pending_args.get(cid, "") + ev.get("delta", "")

            elif et == "response.function_call_arguments.done":
                cid = ev.get("call_id", "")
                name = ev.get("name", "")
                args_json = ev.get("arguments") or pending_args.pop(cid, "{}") or "{}"
                try:
                    args = json.loads(args_json)
                except Exception:
                    args = {}
                await _handle_tool_call(
                    openai_ws, client_ws, session, cid, name, args,
                )

            elif et == "input_audio_buffer.speech_started":
                await client_ws.send_json({"type": "vad_speech_started"})

            elif et == "input_audio_buffer.speech_stopped":
                await client_ws.send_json({"type": "vad_speech_stopped"})

            elif et == "response.done":
                await client_ws.send_json({"type": "response_done"})

            elif et == "error":
                err = ev.get("error", {})
                print(f"[realtime] openai error: {err}", flush=True)
                await client_ws.send_json({
                    "type": "error",
                    "message": err.get("message", "openai error"),
                })
    except Exception as e:  # noqa: BLE001
        print(f"[realtime] openai pump exited: {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
    print(f"[realtime] openai pump finished after {event_count} events", flush=True)


async def _handle_tool_call(
    openai_ws, client_ws,
    session: VoiceSession,
    call_id: str, name: str, args: Dict[str, Any],
) -> None:
    """Execute an MCP tool call from OpenAI and stream the result back.

    For mutation tools (needs_confirmation), we don't auto-execute —
    instead we send a synthetic tool result asking OpenAI to read its
    own description back to the user and request confirmation. (OpenAI
    Realtime doesn't have a native "pause for confirmation" primitive,
    so we model it as a conversational gate.)
    """
    print(f"[realtime] tool call {name} args={list(args.keys())}", flush=True)
    await client_ws.send_json({
        "type": "tool_call_started", "tool": name,
    })

    if name not in TOOLS_BY_NAME:
        envelope = {"ok": False, "kind": "unknown_tool",
                    "error": f"الأداة '{name}' غير متاحة."}
    elif needs_confirmation(name):
        # Don't execute — push back a "needs confirmation" result so
        # the model asks the user before re-calling.
        envelope = {
            "ok": False, "kind": "needs_confirmation",
            "error": f"العملية تحتاج تأكيد شفهي قبل التنفيذ. "
                     f"اطلب من المستخدم 'نعم' لتنفيذ: {TOOLS_BY_NAME[name].purpose_ar}",
        }
    else:
        envelope = await _execute_tool({"name": name, "arguments": args}, session)

    summary = _summarize_result(name, envelope)
    await client_ws.send_json({
        "type": "tool_result", "tool": name, "ok": envelope.get("ok", False),
    })

    # Send the result back to OpenAI as a function_call_output item, then
    # ask it to continue the response.
    await openai_ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": summary,
        },
    }))
    await openai_ws.send(json.dumps({"type": "response.create"}))
