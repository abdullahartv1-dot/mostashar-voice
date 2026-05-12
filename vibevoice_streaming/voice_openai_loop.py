"""OpenAI-powered voice agent loop — native function calling alternative
to the Gemma 4 ReAct loop in voice_react_loop.py.

Why this exists: Gemma 4 E4B's tool-calling requires a hand-rolled ReAct
loop with <tool>...</tool> tags. It works but it's brittle — the model
sometimes forgets the tag format, mis-summarizes tool results, or replies
in non-Arabic on edge cases. OpenAI's gpt-4o-mini supports native function
calling via the `tools` parameter, so we get:

  • Reliable structured tool invocation (no regex parsing of strings)
  • Better Arabic understanding (gpt-4o-mini is multilingual-first)
  • Larger context window (128K vs Gemma's 8K) — full conversation history
  • Cheaper than gpt-4o, faster than gpt-4 Turbo

This module re-uses voice_react_loop's _execute_tool + _summarize_result
(same MCP workarounds, same field whitelist), only the LLM-facing piece
is replaced. Set MV_LLM_BACKEND=openai to switch the WS handler to this
loop. MV_LLM_BACKEND=gemma (the default) keeps the legacy ReAct path.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

try:
    from voice_agent_prompt import build_system_prompt, classify_confirmation  # type: ignore[no-redef]
    from voice_agent_tools import (  # type: ignore[no-redef]
        TOOLS_BY_NAME,
        VOICE_TOOLS,
        needs_confirmation,
    )
    from voice_react_loop import (  # type: ignore[no-redef]
        ReactResult,
        _execute_tool,
        _summarize_result,
    )
    from voice_session import VoiceSession  # type: ignore[no-redef]
except ImportError:
    from .voice_agent_prompt import build_system_prompt, classify_confirmation
    from .voice_agent_tools import TOOLS_BY_NAME, VOICE_TOOLS, needs_confirmation
    from .voice_react_loop import ReactResult, _execute_tool, _summarize_result
    from .voice_session import VoiceSession


log = logging.getLogger("voice_openai_loop")


OPENAI_ENDPOINT = os.environ.get(
    "MV_OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions"
)
OPENAI_MODEL = os.environ.get("MV_OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_S = float(os.environ.get("MV_OPENAI_TIMEOUT_S", "30"))

MAX_ITERATIONS = 5


# ──────────────────────────────────────────────────────────────────────────
# Tool schema translation: VoiceTool → OpenAI function-calling format
# ──────────────────────────────────────────────────────────────────────────

# Map our coarse param names → OpenAI parameter specs. We don't ship the
# full Moshaar JSON schema (it's 50+ KB and we only use ~30 fields). For
# fields we know are arrays, mark them as such so OpenAI doesn't pass
# strings where arrays are required (the calendar_id bug).
_FIELD_TYPES: Dict[str, Dict[str, Any]] = {
    # IDs
    "task_id": {"type": "string", "description": "Task ULID"},
    "case_id": {"type": "string", "description": "Case ULID"},
    "client_id": {"type": "string", "description": "Client ULID"},
    "calendar_id": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Calendar ULID(s) — pass as array even for one calendar",
    },
    "chatId": {"type": "string", "description": "Chat ULID"},
    "threadId": {"type": "string", "description": "Thread (task) ULID"},
    "workflow_id": {"type": "string", "description": "Workflow template ULID"},
    "supervisor_id": {"type": "string"},
    "principal_lawyer_id": {"type": "string"},
    "specialty_id": {"type": "string"},
    "assignee_ids": {"type": "array", "items": {"type": "string"}},
    "team_member_ids": {"type": "array", "items": {"type": "string"}},
    "mentionedWorkspaceMembers": {"type": "array", "items": {"type": "string"}},
    "replyToId": {"type": "string"},
    "status_on_template_id": {"type": "string"},
    # Plain fields
    "name": {"type": "string", "description": "Display name in Arabic"},
    "search": {"type": "string", "description": "Search term in Arabic"},
    "description": {
        "type": "string",
        "description": "Plain Arabic text — the server wraps it in Lexical JSON.",
    },
    "content": {
        "type": "string",
        "description": "Plain Arabic text — server wraps it in Block Editor HTML.",
    },
    "location": {"type": "string"},
    "color": {"type": "string"},
    # Booleans
    "is_done": {"type": "boolean"},
    "is_archived": {"type": "boolean"},
    "is_completed": {"type": "boolean"},
    "is_trashed": {"type": "boolean"},
    "all_day": {"type": "boolean"},
    # Enums
    "priority": {
        "type": "string",
        "enum": ["low", "medium", "high", "urgent"],
    },
    "status": {"type": "string"},
    "type": {
        "type": "string",
        "enum": ["appointment", "call", "email"],
        "description": "Calendar session type",
    },
    "form_type": {
        "type": "string",
        "enum": ["event", "consultation", "session"],
    },
    "privacy_setting": {"type": "string", "enum": ["public", "private"]},
    "availability": {"type": "string"},
    # Dates (ISO 8601 UTC strings)
    "start_date": {"type": "string", "description": "ISO 8601 UTC, e.g. 2026-05-15T17:00:00Z"},
    "end_date": {"type": "string", "description": "ISO 8601 UTC"},
    "estimated_due_date": {"type": "string", "description": "ISO 8601 UTC"},
    # Pagination
    "page": {"type": "integer", "minimum": 1},
    "perPage": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
    # Complex (kept loose so OpenAI doesn't try to schematize fully)
    "settings": {"type": "object", "description": "Settings dict, {} is OK as default"},
    "checklist_groups": {"type": "array"},
    "attendees": {"type": "array"},
    "reminders": {"type": "array"},
    "checklist": {"type": "array"},
}


def _build_openai_tools() -> List[Dict[str, Any]]:
    """Convert our VOICE_TOOLS catalog into OpenAI function-calling format."""
    out: List[Dict[str, Any]] = []
    for vt in VOICE_TOOLS:
        props: Dict[str, Any] = {}
        for field in vt.required + vt.optional:
            spec = _FIELD_TYPES.get(field, {"type": "string"})
            props[field] = dict(spec)  # copy

        params = {
            "type": "object",
            "properties": props,
            "required": list(vt.required),
            "additionalProperties": False,
        }
        # OpenAI description gets the Arabic purpose + any extra notes,
        # plus a strong hint about confirmation behaviour.
        desc_parts = [vt.purpose_ar]
        if vt.notes:
            desc_parts.append("ملاحظات: " + vt.notes)
        if vt.needs_confirmation:
            desc_parts.append(
                "⚠️ هذي عملية تعدّل البيانات — اطلب تأكيداً شفهياً قبل التنفيذ."
            )
        out.append({
            "type": "function",
            "function": {
                "name": vt.name,
                "description": " | ".join(desc_parts)[:1024],
                "parameters": params,
            },
        })
    return out


# Cache the tool list — it never changes during a process lifetime
_OPENAI_TOOLS: Optional[List[Dict[str, Any]]] = None


def get_openai_tools() -> List[Dict[str, Any]]:
    global _OPENAI_TOOLS
    if _OPENAI_TOOLS is None:
        _OPENAI_TOOLS = _build_openai_tools()
    return _OPENAI_TOOLS


# ──────────────────────────────────────────────────────────────────────────
# History translation: our internal format → OpenAI messages
# ──────────────────────────────────────────────────────────────────────────

def _history_to_openai_messages(
    session: VoiceSession,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    """Convert VoiceSession.history into the messages list OpenAI expects.

    Our history entries look like:
      {"role": "user", "content": "..."}
      {"role": "assistant", "content": "..."}                       # plain
      {"role": "assistant", "content": "...", "tool_calls": [...]}  # tool call
      {"role": "tool", "name": "...", "content": "...",
       "tool_call_id": "..."}                                       # tool result

    OpenAI requires tool_call_id linkage between assistant tool_calls and
    tool result messages. We preserve that via the tool_call_id field.
    """
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for turn in session.history_as_messages():
        role = turn.get("role")
        if role in ("user", "assistant"):
            msg: Dict[str, Any] = {"role": role, "content": turn.get("content", "")}
            if role == "assistant" and turn.get("tool_calls"):
                msg["tool_calls"] = turn["tool_calls"]
                # OpenAI dislikes null content on assistant messages with
                # tool_calls — set empty string explicitly.
                if msg["content"] is None:
                    msg["content"] = ""
            msgs.append(msg)
        elif role == "tool":
            msgs.append({
                "role": "tool",
                "tool_call_id": turn.get("tool_call_id", ""),
                "name": turn.get("name", ""),
                "content": turn.get("content", ""),
            })
    return msgs


# ──────────────────────────────────────────────────────────────────────────
# OpenAI call
# ──────────────────────────────────────────────────────────────────────────

async def _call_openai(
    messages: List[Dict[str, Any]],
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 400,
) -> Dict[str, Any]:
    """Single OpenAI chat-completions call. Returns the first choice's message.

    Raises if OPENAI_API_KEY is unset or the API rejects the request.
    """
    api_key = os.environ.get("MV_OPENAI_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "MV_OPENAI_KEY env var is not set on the pod — "
            "OpenAI backend can't run."
        )

    payload: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,  # low — keep replies focused
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=OPENAI_TIMEOUT_S) as client:
        resp = await client.post(OPENAI_ENDPOINT, json=payload, headers=headers)

    if resp.status_code >= 400:
        log.warning("openai %s — %s", resp.status_code, resp.text[:500])
        raise RuntimeError(f"openai status {resp.status_code}")

    body = resp.json()
    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError("openai returned no choices")
    return choices[0].get("message", {})


# ──────────────────────────────────────────────────────────────────────────
# Main loop (mirrors voice_react_loop.run_voice_turn signature)
# ──────────────────────────────────────────────────────────────────────────

async def run_voice_turn_openai(
    *,
    user_text: str,
    session: VoiceSession,
    max_iterations: int = MAX_ITERATIONS,
) -> ReactResult:
    """Process one user utterance via OpenAI with native function calling.

    Mirrors voice_react_loop.run_voice_turn so the WS handler can swap
    one for the other based on the MV_LLM_BACKEND env var. Re-uses the
    same _execute_tool (with BigInt workaround) and _summarize_result
    (with per-tool field whitelist) so we don't duplicate the MCP logic.

    Args:
        user_text: From Whisper, or empty when continuing after a tool result
        session: Per-WS state with MCP credentials + history

    Returns:
        ReactResult with the final Arabic text plus tool_events for the WS.
    """
    # Pre-warm MCP + caches (best-effort, same as the Gemma path)
    if session.mcp_connected and not session._initialized:
        if await session.ensure_mcp():
            try:
                await session.get_workflows()
            except Exception:
                pass
            try:
                await session.get_calendars()
            except Exception:
                pass

    # Confirmation handling — same logic as ReAct path
    if session.pending_tool:
        verdict = classify_confirmation(user_text)
        if verdict == "yes":
            pending = session.pending_tool
            session.pending_tool = None
            envelope = await _execute_tool(pending, session)
            summary = _summarize_result(pending["name"], envelope)
            session.push_tool_result(pending["name"], summary)
            user_text = ""  # let the model continue with the tool result
        elif verdict == "no":
            session.pending_tool = None
            return ReactResult(text="تمام، ألغيت العملية.")
        else:
            session.pending_tool = None  # treat as new request

    if user_text:
        session.push_user(user_text)

    system_prompt = build_system_prompt(
        mcp_connected=session.mcp_connected and session._initialized,
        workflows=session.cached_workflows,
        cached_members=session.cached_members,
    )
    # Re-emphasize Arabic + brevity since OpenAI is otherwise multilingual.
    system_prompt += (
        "\n\nملاحظة: استخدم function calling لأي عملية تحتاج بيانات من النظام. "
        "ردك النهائي يجب أن يكون عربياً قصيراً (جملة-جملتين)."
    )

    tools = get_openai_tools() if (session.mcp_connected and session._initialized) else None

    tool_events: List[Dict[str, Any]] = []
    final_text = ""

    for iteration in range(max_iterations):
        messages = _history_to_openai_messages(session, system_prompt)
        try:
            msg = await _call_openai(messages, tools=tools, max_tokens=400)
        except Exception as e:
            log.warning("openai call failed iter=%d: %s", iteration, e)
            final_text = "اعتذار، تعذّر توليد الرد. حاول مرة أخرى."
            break

        tool_calls = msg.get("tool_calls") or []
        content = (msg.get("content") or "").strip()

        if not tool_calls:
            # Plain text response — we're done
            final_text = content or "تمام."
            session.push_assistant(final_text)
            break

        # ── Tool calls present ──
        # OpenAI may suggest multiple parallel tool calls. We currently run
        # them sequentially because some are mutations that need user
        # confirmation. If ANY require confirmation we surface the first
        # one and bail.
        first_call = tool_calls[0]
        call_id = first_call.get("id", "")
        fn = first_call.get("function", {})
        tool_name = fn.get("name", "")
        try:
            raw_args = json.loads(fn.get("arguments", "{}") or "{}")
        except (ValueError, TypeError):
            raw_args = {}

        if not tool_name or tool_name not in TOOLS_BY_NAME:
            # Hallucinated tool name — push an error result so the model can recover
            session.history.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            })
            session.history.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name or "unknown",
                "content": json.dumps(
                    {"error": "tool not in voice catalog",
                     "available": list(TOOLS_BY_NAME.keys())[:8]},
                    ensure_ascii=False,
                ),
            })
            user_text = ""
            continue

        # Confirmation gate for mutations
        if needs_confirmation(tool_name):
            confirm_msg = content or (
                f"تريد أن أُنفّذ: {TOOLS_BY_NAME[tool_name].purpose_ar}؟ "
                "قل 'نعم' للتأكيد."
            )
            session.pending_tool = {"name": tool_name, "arguments": raw_args}
            session.push_assistant(confirm_msg)
            tool_events.append({"type": "confirmation_pending", "tool": tool_name})
            return ReactResult(
                text=confirm_msg,
                pending_confirmation=True,
                tool_events=tool_events,
            )

        # Execute read-only tool — uses voice_react_loop._execute_tool so
        # the BigInt + perPage workarounds apply identically.
        log.info("openai exec %s args=%s",
                 tool_name, list((raw_args or {}).keys()))
        envelope = await _execute_tool(
            {"name": tool_name, "arguments": raw_args}, session,
        )
        summary = _summarize_result(tool_name, envelope)

        # Push assistant turn with tool_calls + the tool result, linked
        # by tool_call_id so OpenAI can match them up.
        session.history.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })
        session.history.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": tool_name,
            "content": summary,
        })
        tool_events.append({
            "type": "tool_result",
            "tool": tool_name,
            "ok": envelope.get("ok", False),
        })
        user_text = ""  # loop with the tool result in context

    else:
        final_text = "اعتذار، الطلب تطلب خطوات أكثر مما توقعت. حاول صياغة أبسط."
        session.push_assistant(final_text)

    return ReactResult(
        text=final_text,
        pending_confirmation=False,
        tool_events=tool_events,
    )
