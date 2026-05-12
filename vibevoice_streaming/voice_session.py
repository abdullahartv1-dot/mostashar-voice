"""Voice session — per-WebSocket-connection state for the voice agent.

Each open `/v1/conversation/ws` connection gets a `VoiceSession` instance
that holds:

  • the user's MCP credentials (in memory only, never persisted)
  • cached dropdown data (workflows, members, clients) so we don't refetch
  • conversation history for the LLM
  • pending tool call awaiting confirmation
  • the MCP client itself

The session lives for the duration of the WebSocket and is torn down on
disconnect — the MCP key is released from memory and the httpx client is
closed.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Deque, Dict, List, Optional

# Allow this module to be imported both as part of a package (when
# the WS server is launched via `python -m vibevoice_streaming.server_v5_api`)
# and standalone (when launched as a plain script — the deploy pattern).
try:
    from moshaar_mcp import MoshaarMCPClient, MoshaarToolSpec  # type: ignore[no-redef]
    from voice_agent_helpers import to_lexical_json, to_block_editor_html  # type: ignore[no-redef]
    from voice_agent_tools import VOICE_TOOLS  # type: ignore[no-redef]
except ImportError:
    from .moshaar_mcp import MoshaarMCPClient, MoshaarToolSpec
    from .voice_agent_helpers import to_lexical_json, to_block_editor_html
    from .voice_agent_tools import VOICE_TOOLS


MAX_HISTORY = 16  # keep last 8 user/assistant turns


class VoiceSession:
    """In-memory state for one voice conversation."""

    def __init__(
        self,
        *,
        mcp_url: Optional[str] = None,
        mcp_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        self.user_id = user_id
        self.mcp_url = mcp_url
        self.mcp_key = mcp_key
        self.mcp: Optional[MoshaarMCPClient] = None

        self.history: Deque[Dict[str, Any]] = deque(maxlen=MAX_HISTORY)
        self.pending_tool: Optional[Dict[str, Any]] = None

        # Cached dropdown data
        self.cached_workflows: Optional[List[Dict[str, Any]]] = None
        self.cached_members: Optional[List[Dict[str, Any]]] = None
        self.cached_calendars: Optional[List[Dict[str, Any]]] = None

        self._init_lock = asyncio.Lock()
        self._initialized = False

    # ───────────────────────────────────────────────────────────────────
    # Lifecycle
    # ───────────────────────────────────────────────────────────────────

    @property
    def mcp_connected(self) -> bool:
        return bool(self.mcp_url and self.mcp_key)

    async def ensure_mcp(self) -> bool:
        """Create the MCP client + do initial handshake if not already done.

        Returns True on success, False if no key configured or init failed.
        Safe to call multiple times.
        """
        if not self.mcp_connected:
            return False

        async with self._init_lock:
            if self._initialized:
                return True
            if self.mcp is None:
                self.mcp = MoshaarMCPClient(self.mcp_url, self.mcp_key)
            try:
                await self.mcp.initialize()
                self._initialized = True
                return True
            except Exception:
                # Caller can retry on next turn if user wants
                return False

    async def aclose(self) -> None:
        """Drop credentials + close MCP client."""
        if self.mcp:
            try:
                await self.mcp.aclose()
            except Exception:
                pass
        self.mcp = None
        self.mcp_key = None  # zero out
        self.pending_tool = None
        self.history.clear()

    # ───────────────────────────────────────────────────────────────────
    # Dropdown caching — lazy-fetch the things every voice command needs
    # ───────────────────────────────────────────────────────────────────

    async def get_workflows(self, force: bool = False) -> List[Dict[str, Any]]:
        """Cached workflows lookup."""
        if self.cached_workflows is not None and not force:
            return self.cached_workflows
        if not await self.ensure_mcp():
            return []
        result = await self.mcp.call_safe("get_workflows", {"perPage": 20})
        if not result.get("ok"):
            return []
        data = result.get("data", {})
        # Moshaar wraps as {"success": true, "data": {"data": [...], ...}}
        workflows = _extract_list(data, "workflows") or _extract_list(data, "data") or []
        self.cached_workflows = workflows
        return workflows

    async def get_members(self, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """Workspace members lookup (not cached if a search is provided)."""
        if search is None and self.cached_members is not None:
            return self.cached_members
        if not await self.ensure_mcp():
            return []
        args: Dict[str, Any] = {"perPage": 30}
        if search:
            args["search"] = search
        result = await self.mcp.call_safe("get_workspace_members", args)
        if not result.get("ok"):
            return []
        members = _extract_list(result["data"], "members") or _extract_list(result["data"], "data") or []
        if search is None:
            self.cached_members = members
        return members

    # ───────────────────────────────────────────────────────────────────
    # History management
    # ───────────────────────────────────────────────────────────────────

    def push_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def push_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    def push_tool_result(self, name: str, payload: str) -> None:
        # Cap payload to avoid blowing the prompt
        snippet = payload[:1500]
        self.history.append({
            "role": "tool",
            "name": name,
            "content": snippet,
        })

    def history_as_messages(self) -> List[Dict[str, str]]:
        return list(self.history)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _extract_list(envelope: Any, primary_key: str) -> Optional[List[Dict[str, Any]]]:
    """Try to extract a list from Moshaar's nested envelopes.

    Moshaar returns:
        {"success": true, "data": [...]}           # simple
        {"success": true, "data": {"data": [...]}} # paginated
        {"success": true, "data": {"items": [...]}} # alternate
    """
    if isinstance(envelope, list):
        return envelope
    if not isinstance(envelope, dict):
        return None
    candidate = envelope.get(primary_key)
    if isinstance(candidate, list):
        return candidate
    inner = envelope.get("data")
    if isinstance(inner, list):
        return inner
    if isinstance(inner, dict):
        for k in (primary_key, "data", "items", "results"):
            v = inner.get(k)
            if isinstance(v, list):
                return v
    return None
