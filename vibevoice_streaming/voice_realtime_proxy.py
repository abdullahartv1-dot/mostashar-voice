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
except ImportError:
    from .voice_agent_tools import VOICE_TOOLS, TOOLS_BY_NAME, needs_confirmation
    from .voice_openai_loop import _FIELD_TYPES
    from .voice_react_loop import _execute_tool, _summarize_result
    from .voice_session import VoiceSession


log = logging.getLogger("voice_realtime_proxy")

REALTIME_URL = os.environ.get(
    "MV_OPENAI_REALTIME_URL",
    "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17",
)
# OpenAI voice options: alloy / ash / ballad / coral / echo / sage / shimmer / verse
REALTIME_VOICE = os.environ.get("MV_OPENAI_REALTIME_VOICE", "shimmer")


SYSTEM_INSTRUCTIONS_AR = """\
أنت "سارة"، مساعدة صوتية لمنصة مستشار (إدارة قضايا قانونية وأعمال).
تتكلمين العربية الفصحى المبسّطة مع لمسة سعودية. ردودك قصيرة وعملية —
جملتان كحد أقصى عادةً.

قواعد السلوك:
- لا تذكري للمستخدم أبداً أنك تستخدمين "أدوات" أو IDs. هو لا يهتم.
- استدعِ الأدوات المتاحة عند الحاجة لمعلومة من نظام مستشار.
- قبل أي عملية تعدّل البيانات (إنشاء/تحديث/حذف/إرسال): أوصفي العملية
  شفهياً واطلبي تأكيداً. لا تنفّذي حتى أسمع "نعم/تمام/وافق".
- IDs والـ tokens لا تُذكر بالصوت. استخدمي الأسماء فقط.
- التواريخ النسبية حوّليها إلى ISO 8601 UTC قبل تمريرها للأدوات.
- عند خطأ من النظام، اشرحي السبب بكلمات بسيطة.
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

    # Realtime API session.update — only Whisper-1 is valid for the
    # nested input_audio_transcription model (the newer
    # gpt-4o-mini-transcribe is for the top-level /audio/transcriptions
    # endpoint, not embedded inside Realtime). Using the wrong model
    # makes OpenAI close the WS with an error.
    update_msg = {
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "voice": REALTIME_VOICE,
            "instructions": SYSTEM_INSTRUCTIONS_AR,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
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
        },
    }
    await openai_ws.send(json.dumps(update_msg))
    print(f"[realtime] session.update sent (tools={len(tools)}, voice={REALTIME_VOICE})",
          flush=True)


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


async def _pump_openai_to_user(
    client_ws, openai_ws, session: VoiceSession,
) -> None:
    """Forward OpenAI events + audio to the user, handle function calls."""
    pending_args: Dict[str, str] = {}  # call_id → accumulating JSON args
    event_count = 0

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

        if et == "response.audio.delta":
            # base64 pcm16 audio chunk — decode and binary-stream to user.
            audio_b64 = ev.get("delta", "")
            try:
                audio_bytes = base64.b64decode(audio_b64)
                await client_ws.send_bytes(audio_bytes)
                # Counter: log every 50 chunks so we can confirm audio is
                # actually being forwarded. If the user reports "no
                # response", check this number in /workspace/server_v5.log.
                if not hasattr(_pump_openai_to_user, "_audio_count"):
                    _pump_openai_to_user._audio_count = 0  # type: ignore[attr-defined]
                _pump_openai_to_user._audio_count += 1  # type: ignore[attr-defined]
                if _pump_openai_to_user._audio_count % 25 == 0:  # type: ignore[attr-defined]
                    print(f"[realtime] forwarded {_pump_openai_to_user._audio_count} audio chunks "  # type: ignore[attr-defined]
                          f"({len(audio_bytes)} bytes/last)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[realtime] send_bytes failed: {e}", flush=True)

        elif et == "response.audio_transcript.delta":
            await client_ws.send_json({
                "type": "assistant_text_delta", "text": ev.get("delta", ""),
            })

        elif et == "response.audio_transcript.done":
            await client_ws.send_json({
                "type": "assistant_text_done",
                "text": ev.get("transcript", ""),
            })

        elif et == "conversation.item.input_audio_transcription.completed":
            # User's audio transcribed by OpenAI — show them what was heard.
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
