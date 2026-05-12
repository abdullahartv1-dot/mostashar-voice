"""Voice-agent helpers — wrappers for Moshaar's content formats.

Moshaar's API uses two opinionated content formats that the voice agent
should never have to think about:

  • **Lexical JSON** — for `description` fields on tasks, cases, and
    calendar sessions. This is Meta's Lexical rich-text-editor wire format,
    a verbose tree of nested nodes.

  • **Block Editor HTML** — for `content` on chat messages and thread
    messages. This is the BlockNote editor's HTML format with custom
    `data-node-type` attributes.

The voice agent emits plain Arabic text in its tool calls; this module
converts those strings to the correct wire format before sending.
"""

from __future__ import annotations

import html
import json
import uuid
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────
# Lexical JSON — for task / case / calendar-session descriptions
# ──────────────────────────────────────────────────────────────────────────

def to_lexical(text: str, direction: str = "rtl") -> Dict[str, Any]:
    """Wrap plain text in Lexical Editor JSON format.

    Splits on blank lines (\\n\\n) to make paragraphs.  Each paragraph is
    a Lexical paragraph node with a single text node child.

    Args:
        text: Plain text from the user (typically Arabic, RTL)
        direction: "rtl" (default) or "ltr"

    Returns:
        Lexical document dict. JSON-dump this before sending to Moshaar:
            json.dumps(to_lexical(text), ensure_ascii=False)
    """
    paragraphs = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
    children: List[Dict[str, Any]] = []

    for p in paragraphs:
        children.append({
            "type": "paragraph",
            "version": 1,
            "direction": direction,
            "format": "",
            "indent": 0,
            "textFormat": 0,
            "textStyle": "",
            "children": [{
                "type": "text",
                "version": 1,
                "text": p,
                "detail": 0,
                "format": 0,
                "mode": "normal",
                "style": "",
            }],
        })

    # If no paragraphs, emit an empty paragraph so the editor doesn't crash
    if not children:
        children = [{
            "type": "paragraph",
            "version": 1,
            "direction": direction,
            "format": "",
            "indent": 0,
            "textFormat": 0,
            "textStyle": "",
            "children": [],
        }]

    return {
        "root": {
            "type": "root",
            "version": 1,
            "direction": direction,
            "format": "",
            "indent": 0,
            "children": children,
        }
    }


def to_lexical_json(text: str, direction: str = "rtl") -> str:
    """Like `to_lexical` but returns a serialized JSON string ready to send."""
    return json.dumps(to_lexical(text, direction), ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────
# Block Editor HTML — for chat / thread messages
# ──────────────────────────────────────────────────────────────────────────

def _bn_block(content_html: str, block_id: Optional[str] = None) -> str:
    """Wrap a single content node in the Block Editor block envelope."""
    bid = block_id or str(uuid.uuid4())
    return (
        f'<div class="bn-block-outer" data-node-type="blockOuter" data-id="{bid}">'
        f'<div class="bn-block" data-node-type="blockContainer" data-id="{bid}">'
        f'{content_html}'
        f'</div></div>'
    )


def to_block_editor_html(text: str) -> str:
    """Convert plain text to Block Editor HTML.

    Each blank-line-separated paragraph becomes its own `paragraph` block.
    HTML special characters in the text are escaped.

    Args:
        text: Plain text (Arabic supported — UTF-8 throughout)

    Returns:
        HTML string ready for the `content` field of send_message /
        send_thread_message.
    """
    paragraphs = [p.strip() for p in (text or "").split("\n\n") if p.strip()]

    if not paragraphs:
        # Empty message — emit one empty paragraph so the API accepts it
        paragraphs = [""]

    blocks: List[str] = []
    for p in paragraphs:
        safe = html.escape(p, quote=False)
        block_content = (
            f'<div class="bn-block-content" data-content-type="paragraph">'
            f'<p class="bn-inline-content">{safe}</p>'
            f'</div>'
        )
        blocks.append(_bn_block(block_content))

    return (
        '<div class="bn-block-group" data-node-type="blockGroup">'
        + "".join(blocks)
        + '</div>'
    )


# ──────────────────────────────────────────────────────────────────────────
# Date / time helpers — Arabic relative dates → ISO 8601
# ──────────────────────────────────────────────────────────────────────────

def now_iso() -> str:
    """Current UTC time in ISO 8601, suitable for the system prompt."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today_iso() -> str:
    """Current UTC date as YYYY-MM-DD."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
