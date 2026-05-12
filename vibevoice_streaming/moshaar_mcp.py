"""Moshaar MCP client — async HTTP/JSON-RPC client for the Moshaar workspace MCP.

The Moshaar platform exposes 59 tools at `<endpoint>/v2/mcp/master` using the
standard MCP JSON-RPC over HTTP protocol. Authentication is via the
`x-mcp-api-key` HTTP header (each user has their own key, scoped to their
workspace).

This client is designed for the voice agent: it's lightweight, async, and
intended for one-instance-per-WebSocket-session (i.e. per active call).

Key design points:
  • per-session — never shared between users; key + endpoint passed at construction
  • async — uses httpx.AsyncClient with sensible timeouts
  • typed errors — raises MCPError subclasses so caller can humanize messages
  • no persistence — nothing written to disk, nothing logged with the key
  • single-request mode — no streaming/SSE needed for the tool calls we use

For the V1 voice-agent integration, this client is only instantiated once
the WebSocket handshake provides `x-moshaar-mcp-url` and `x-moshaar-mcp-key`
headers. Lifetime equals the WebSocket lifetime.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx


DEFAULT_TIMEOUT_S = 30.0
DEFAULT_ENDPOINT = "https://api.moshaar.com/v2/mcp/master"


class MCPError(Exception):
    """Base class for all MCP failures."""


class MCPAuthError(MCPError):
    """401/403 — invalid or expired API key."""


class MCPNetworkError(MCPError):
    """Network failure (timeout, DNS, connection refused)."""


class MCPProtocolError(MCPError):
    """JSON-RPC envelope error or malformed response."""


class MCPToolError(MCPError):
    """The tool ran on Moshaar but returned an error — e.g. validation failure."""

    def __init__(self, message: str, code: Optional[int] = None, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data


@dataclass
class MoshaarToolSpec:
    """Minimal description of an MCP tool, as returned by `tools/list`."""

    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)

    @property
    def required(self) -> List[str]:
        return list(self.input_schema.get("required", []))

    @property
    def properties(self) -> Dict[str, Any]:
        return dict(self.input_schema.get("properties", {}))


class MoshaarMCPClient:
    """Async MCP client for one Moshaar workspace.

    Usage::

        async with MoshaarMCPClient(url, key) as mcp:
            tools = await mcp.list_tools()
            result = await mcp.call("list_tasks", {"is_done": False})

    Or with manual close::

        mcp = MoshaarMCPClient(url, key)
        try:
            ...
        finally:
            await mcp.aclose()
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        api_key: str = "",
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
    ):
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError(f"endpoint must be http(s): {endpoint!r}")
        if not api_key:
            raise ValueError("api_key is required (per-user MCP key)")

        self.endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._req_id = 0
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "x-mcp-api-key": api_key,
            },
        )

    async def __aenter__(self) -> "MoshaarMCPClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # ───────────────────────────────────────────────────────────────────────
    # Core JSON-RPC plumbing
    # ───────────────────────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    async def _rpc(self, method: str, params: Dict[str, Any] | None = None) -> Any:
        """Send one JSON-RPC request; return the `result` field on success.

        Raises one of MCPError's subclasses on any failure.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params or {},
        }
        try:
            resp = await self._client.post(self.endpoint, json=payload)
        except httpx.TimeoutException as e:
            raise MCPNetworkError(f"timeout calling {method}") from e
        except httpx.HTTPError as e:
            raise MCPNetworkError(f"network error calling {method}: {e}") from e

        if resp.status_code in (401, 403):
            raise MCPAuthError(f"auth rejected (status {resp.status_code})")
        if resp.status_code >= 500:
            raise MCPNetworkError(f"server error {resp.status_code} on {method}")
        if resp.status_code >= 400:
            raise MCPProtocolError(
                f"unexpected status {resp.status_code} on {method}: {resp.text[:200]}"
            )

        try:
            body = resp.json()
        except ValueError as e:
            raise MCPProtocolError(f"invalid JSON in response: {resp.text[:200]}") from e

        if "error" in body:
            err = body["error"]
            raise MCPToolError(
                err.get("message", "tool error"),
                code=err.get("code"),
                data=err.get("data"),
            )

        if "result" not in body:
            raise MCPProtocolError(f"missing 'result' in response: {body!r}")

        return body["result"]

    # ───────────────────────────────────────────────────────────────────────
    # Public API — what the voice agent calls
    # ───────────────────────────────────────────────────────────────────────

    async def initialize(self) -> Dict[str, Any]:
        """Initial MCP handshake — confirms protocol + server info."""
        return await self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "moshaar-voice-agent", "version": "1.0.0"},
        })

    async def list_tools(self) -> List[MoshaarToolSpec]:
        """List all 59 tools the workspace exposes."""
        result = await self._rpc("tools/list", {})
        return [
            MoshaarToolSpec(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            )
            for t in result.get("tools", [])
        ]

    async def call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool.  Returns parsed JSON result.

        On success, the MCP envelope is::

            {"content": [{"type": "text", "text": "...json..."}]}

        We parse the inner JSON if it looks like JSON, otherwise return the
        raw text.  Many Moshaar tools wrap their output as
        `{"success": true, "data": {...}}` — caller can inspect.
        """
        result = await self._rpc("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })

        # The MCP spec returns: {"content": [{"type": "text", "text": "<json>"}]}
        content = result.get("content", [])
        if not content:
            return None

        out = []
        for item in content:
            if item.get("type") != "text":
                out.append(item)
                continue
            text = item.get("text", "")
            try:
                out.append(json.loads(text))
            except (ValueError, TypeError):
                out.append(text)
        return out[0] if len(out) == 1 else out

    async def call_safe(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Like `call`, but catches all MCPErrors and returns a uniform envelope.

        Useful for the ReAct loop — the LLM gets back a JSON either way:
          {"ok": true, "data": ...}
          {"ok": false, "error": "...", "kind": "auth"|"network"|"tool"|"protocol"}
        """
        try:
            data = await self.call(tool_name, arguments)
            return {"ok": True, "data": data}
        except MCPAuthError as e:
            return {"ok": False, "kind": "auth", "error": str(e)}
        except MCPNetworkError as e:
            return {"ok": False, "kind": "network", "error": str(e)}
        except MCPToolError as e:
            return {"ok": False, "kind": "tool", "error": str(e), "code": e.code}
        except MCPProtocolError as e:
            return {"ok": False, "kind": "protocol", "error": str(e)}
        except Exception as e:  # noqa: BLE001  — last-resort safety net
            return {"ok": False, "kind": "unknown", "error": str(e)}


# ──────────────────────────────────────────────────────────────────────────
# Standalone smoke test (run as: `python -m vibevoice_streaming.moshaar_mcp`)
# ──────────────────────────────────────────────────────────────────────────
async def _smoke_test():
    import os
    import sys

    key = os.environ.get("MOSHAAR_MCP_KEY", "").strip()
    if not key:
        print("ERR: set MOSHAAR_MCP_KEY env var to run smoke test")
        sys.exit(1)

    url = os.environ.get("MOSHAAR_MCP_URL", DEFAULT_ENDPOINT)
    print(f"Connecting to {url} ...")

    async with MoshaarMCPClient(url, key) as mcp:
        info = await mcp.initialize()
        print(f"[OK] initialize: server={info.get('serverInfo', {}).get('name')}")

        tools = await mcp.list_tools()
        print(f"[OK] tools/list: {len(tools)} tools")
        for t in tools[:3]:
            print(f"     - {t.name}: {t.description[:60]}")

        # Quick call: aggregate_cases_by_status (no args)
        agg = await mcp.call_safe("aggregate_cases_by_status", {})
        print(f"[OK] aggregate_cases_by_status: {json.dumps(agg, ensure_ascii=False)[:200]}")


if __name__ == "__main__":
    asyncio.run(_smoke_test())
