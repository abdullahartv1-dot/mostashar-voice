"""ReAct loop — runs a voice turn that may invoke MCP tools.

This is the orchestrator that ties everything together:
  1. Build the Arabic system prompt from VOICE_TOOLS catalog
  2. Send (system + history + user_text) to Gemma 4
  3. Parse Gemma's response for <tool>...</tool> tags
  4. If present:
       - Mutations → emit a confirmation event, store pending
       - Read-only → execute MCP call, push result, loop
  5. If no tool tag → return plain text response

Designed to be called from the existing WS handler in server_v5_api.py
with minimal coupling — the only requirement is a function that calls
the Gemma sidecar with (system_prompt, history, user_text) → response.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

try:
    from voice_agent_helpers import to_block_editor_html, to_lexical_json  # type: ignore[no-redef]
    from voice_agent_prompt import (  # type: ignore[no-redef]
        build_system_prompt,
        classify_confirmation,
        parse_tool_call,
        strip_tool_call,
    )
    from voice_agent_tools import (  # type: ignore[no-redef]
        DROPDOWN_TOOLS,
        TOOLS_BY_NAME,
        needs_confirmation,
    )
    from voice_session import VoiceSession  # type: ignore[no-redef]
except ImportError:
    from .voice_agent_helpers import to_block_editor_html, to_lexical_json
    from .voice_agent_prompt import (
        build_system_prompt,
        classify_confirmation,
        parse_tool_call,
        strip_tool_call,
    )
    from .voice_agent_tools import (
        DROPDOWN_TOOLS,
        TOOLS_BY_NAME,
        needs_confirmation,
    )
    from .voice_session import VoiceSession


log = logging.getLogger("voice_react_loop")

MAX_REACT_ITERATIONS = 5


# Type alias: function that calls Gemma sidecar.
# Signature: (system_prompt, history, user_text, max_tokens) -> response_text
GemmaCallFn = Callable[[str, List[Dict[str, Any]], str, int], Awaitable[str]]


# ──────────────────────────────────────────────────────────────────────────
# Argument normalization — wrap text fields in the right format
# ──────────────────────────────────────────────────────────────────────────

# Fields that need Lexical JSON conversion (task/case/calendar description)
LEXICAL_FIELDS = {"description"}

# Fields that need Block Editor HTML conversion (message content)
HTML_FIELDS = {"content"}


def _normalize_arguments(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Convert plain-text fields to the formats Moshaar expects.

    The voice agent emits human-readable Arabic strings; this function
    wraps them in Lexical JSON or Block Editor HTML where required.

    Mutates a copy; never the original dict.
    """
    out: Dict[str, Any] = dict(args)

    # Description fields → Lexical JSON
    if "description" in out and isinstance(out["description"], str):
        out["description"] = to_lexical_json(out["description"])

    # Message content fields → Block Editor HTML
    if "content" in out and isinstance(out["content"], str):
        # If it already looks like Block Editor HTML, leave it alone
        if not out["content"].lstrip().startswith("<div class=\"bn-"):
            out["content"] = to_block_editor_html(out["content"])

    # Defaults for create_case settings (Moshaar accepts {} as minimal)
    if tool_name == "create_case" and "settings" not in out:
        out["settings"] = {}

    return out


# ──────────────────────────────────────────────────────────────────────────
# Tool execution
# ──────────────────────────────────────────────────────────────────────────

async def _execute_tool(
    tool_call: Dict[str, Any],
    session: VoiceSession,
) -> Dict[str, Any]:
    """Run one MCP tool call. Returns a uniform envelope:
        {"ok": True, "data": ...} or {"ok": False, "kind": "...", "error": "..."}
    """
    name = tool_call["name"]
    raw_args = tool_call.get("arguments", {}) or {}

    if name not in TOOLS_BY_NAME:
        return {"ok": False, "kind": "unknown_tool",
                "error": f"الأداة '{name}' غير متاحة في هذي النسخة."}

    if not session.mcp:
        return {"ok": False, "kind": "no_mcp",
                "error": "غير متصل بـ مستشار."}

    args = _normalize_arguments(name, raw_args)
    return await session.mcp.call_safe(name, args)


# ──────────────────────────────────────────────────────────────────────────
# Truncation helpers — keep MCP results small enough for the prompt
# ──────────────────────────────────────────────────────────────────────────

def _summarize_result(name: str, envelope: Dict[str, Any]) -> str:
    """Produce a compact, LLM-friendly summary of an MCP result.

    Avoids feeding raw 50KB JSON back to Gemma. We keep just enough
    structure for the LLM to compose its natural-language reply.
    """
    if not envelope.get("ok"):
        kind = envelope.get("kind", "error")
        return json.dumps(
            {"error": envelope.get("error", ""), "kind": kind},
            ensure_ascii=False,
        )

    data = envelope.get("data")
    if data is None:
        return json.dumps({"ok": True}, ensure_ascii=False)

    # Already-flat dict — pass through (capped)
    raw = json.dumps(data, ensure_ascii=False)
    if len(raw) <= 1200:
        return raw

    # Try common patterns: data.data, data.items, data.results
    if isinstance(data, dict):
        for k in ("data", "items", "results", "rows"):
            if k in data and isinstance(data[k], list):
                # Truncate list to first 8 items
                truncated = {
                    **{kk: vv for kk, vv in data.items() if kk != k},
                    k: data[k][:8],
                    "_truncated_from": len(data[k]),
                }
                return json.dumps(truncated, ensure_ascii=False)[:1200]

    return raw[:1200] + "..."


# ──────────────────────────────────────────────────────────────────────────
# Main ReAct loop
# ──────────────────────────────────────────────────────────────────────────

class ReactResult:
    """Outcome of one voice turn through the ReAct loop."""

    __slots__ = ("text", "pending_confirmation", "tool_events")

    def __init__(
        self,
        text: str,
        pending_confirmation: bool = False,
        tool_events: Optional[List[Dict[str, Any]]] = None,
    ):
        self.text = text
        self.pending_confirmation = pending_confirmation
        self.tool_events = tool_events or []


async def run_voice_turn(
    *,
    user_text: str,
    session: VoiceSession,
    gemma_call: GemmaCallFn,
    max_iterations: int = MAX_REACT_ITERATIONS,
) -> ReactResult:
    """Process one user utterance, possibly invoking MCP tools.

    Args:
        user_text: The text from Whisper (or direct text message)
        session: Per-WS VoiceSession holding MCP client + history
        gemma_call: Async function that calls the Gemma sidecar.
            Signature: (system_prompt, history, user_text, max_tokens) -> str

    Returns:
        ReactResult with the final response text, plus event metadata.
    """
    # Build system prompt — includes MCP catalog if connected
    if session.mcp_connected:
        # Try to ensure MCP + pre-warm caches (best-effort)
        if not session._initialized:
            ok = await session.ensure_mcp()
            if ok:
                # Pre-fetch workflows once; helps Gemma resolve them later
                try:
                    await session.get_workflows()
                except Exception:
                    pass

    system_prompt = build_system_prompt(
        mcp_connected=session.mcp_connected and session._initialized,
        workflows=session.cached_workflows,
        cached_members=session.cached_members,
    )

    # Handle a pending confirmation first
    if session.pending_tool:
        verdict = classify_confirmation(user_text)
        if verdict == "yes":
            pending = session.pending_tool
            session.pending_tool = None
            envelope = await _execute_tool(pending, session)
            summary = _summarize_result(pending["name"], envelope)
            session.push_tool_result(pending["name"], summary)
            # Re-run loop with no new user input (just the tool result)
            user_text = ""  # continue with tool result in history
        elif verdict == "no":
            session.pending_tool = None
            return ReactResult(text="تمام، ألغيت العملية.")
        # else: treat as new request, drop pending
        else:
            session.pending_tool = None

    # Push the user text to history (if non-empty)
    if user_text:
        session.push_user(user_text)

    # ReAct iterations
    tool_events: List[Dict[str, Any]] = []
    final_text = ""

    for iteration in range(max_iterations):
        history_msgs = session.history_as_messages()
        try:
            gemma_resp = await gemma_call(
                system_prompt,
                history_msgs,
                user_text if iteration == 0 else "",  # first iter sends user text
                400,  # max_tokens — generous to allow tool args
            )
        except Exception as e:
            log.warning("gemma call failed in iteration %d: %s", iteration, e)
            final_text = "اعتذار، تعذّر توليد الرد. حاول مرة أخرى."
            break

        tool_call = parse_tool_call(gemma_resp)

        if not tool_call:
            # No tool call → plain text response, done
            final_text = strip_tool_call(gemma_resp) or gemma_resp
            session.push_assistant(final_text)
            break

        # Check confirmation gate
        if needs_confirmation(tool_call["name"]):
            # Build a description-only message; Gemma should have included it
            confirm_msg = strip_tool_call(gemma_resp)
            if not confirm_msg:
                # Fallback: synthesize a generic confirmation
                tool = TOOLS_BY_NAME[tool_call["name"]]
                confirm_msg = f"تريد أن أُنفّذ: {tool.purpose_ar}؟ قل 'نعم' للتأكيد."
            session.pending_tool = tool_call
            session.push_assistant(confirm_msg)
            tool_events.append({
                "type": "confirmation_pending",
                "tool": tool_call["name"],
            })
            return ReactResult(
                text=confirm_msg,
                pending_confirmation=True,
                tool_events=tool_events,
            )

        # Execute read-only tool
        log.info("executing tool %s with args %s",
                 tool_call["name"], list((tool_call.get("arguments") or {}).keys()))
        envelope = await _execute_tool(tool_call, session)
        summary = _summarize_result(tool_call["name"], envelope)

        # Push assistant's tool-call-containing message + the tool result
        session.push_assistant(gemma_resp)
        session.push_tool_result(tool_call["name"], summary)
        tool_events.append({
            "type": "tool_result",
            "tool": tool_call["name"],
            "ok": envelope.get("ok", False),
        })

        # Loop — Gemma will now see the tool result and either respond or call another
        user_text = ""

    else:
        # Hit iteration cap
        final_text = "اعتذار، الطلب تطلب خطوات أكثر مما توقعت. حاول صياغة أبسط."
        session.push_assistant(final_text)

    return ReactResult(
        text=final_text,
        pending_confirmation=False,
        tool_events=tool_events,
    )
