# Voice Agent ↔ Moshaar MCP Integration — Design Spec

**Date:** 2026-05-12
**Author:** Abdullah Al-Amri + Claude
**Status:** Draft (awaiting implementation approval)
**Scope:** Add MCP-aware tool-calling to the existing voice agent so users can manage Moshaar (tasks, cases, calendar, chats, blocknotes) by voice.

---

## 1. Goal

A user calls Sara in the live `/call` UI and says:
> "سارة، اعرض مهامي النشطة، وأنشئ مهمة لمراجعة عقد الموكل أحمد قبل يوم الخميس."

Sara should:

1. Understand intent
2. Resolve "أحمد" → `client_id` via `get_clients(search="أحمد")`
3. Call `get_workflows()` → pick a workflow_id (cached)
4. Confirm verbally: *"تأكيد: أنشئ مهمة 'مراجعة عقد الموكل أحمد' للموكل أحمد، الاستحقاق الأربعاء الساعة 5 العصر؟"*
5. On "نعم": call `create_task(...)` and `list_tasks(is_done=false)`
6. Reply with the natural summary plus the new task

All without touching a keyboard.

## 2. Non-Goals (V1)

- ❌ Yjs-encoded blocknote content (deferred to V2 — voice creates empty blocknotes only, content goes into thread messages)
- ❌ Uploading new files / images (MCP doesn't expose `upload_file`)
- ❌ Workflow editing (`update_task_workflow` with `compare_status_templates`) — too complex for voice
- ❌ Case-side bulk edits — admin-only operations
- ❌ Cross-tenant data — every voice session is single-user

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ Browser (voice-moshaar-frontend)                              │
│                                                                │
│  /settings → User enters:                                     │
│    • Moshaar MCP endpoint (default: api.moshaar.com/v2/...)   │
│    • Moshaar MCP API key (mcp_sk_...)                         │
│  ↓ stored in localStorage (NOT sent unless WS open)           │
│                                                                │
│  /call (existing UI) → opens WebSocket with extra headers:    │
│    x-moshaar-mcp-url: <user URL>                              │
│    x-moshaar-mcp-key: <user key>                              │
└──────────────────┬───────────────────────────────────────────┘
                   │ WS /v1/conversation/ws
                   ▼
┌──────────────────────────────────────────────────────────────┐
│ Voice Backend (server_v5_api.py)                              │
│                                                                │
│  PER-SESSION STATE (in-memory, never persisted):              │
│    • whisper STT pipeline                                     │
│    • gemma4_url (sidecar)                                     │
│    • moshaar_mcp_url + moshaar_mcp_key (from headers)         │
│    • MoshaarMCPClient instance (lazy-init)                    │
│    • conversation_history (last 10 turns)                     │
│    • cached: workflows, calendars, workspace_members          │
│                                                                │
│  TURN FLOW:                                                    │
│   audio in → Whisper → text                                   │
│   text + system_prompt + tools_catalog + history              │
│        ↓                                                        │
│   POST /v1/chat to gemma4_server.py                           │
│        ↓                                                        │
│   parse response:                                              │
│     ├─ if <tool>...</tool> tag → ReAct loop:                  │
│     │   1. parse JSON                                         │
│     │   2. if "needs_confirmation": speak confirmation        │
│     │   3. else: execute MCP call                             │
│     │   4. feed result back to Gemma                          │
│     │   5. repeat up to 5 iterations                          │
│     └─ else: text response                                    │
│        ↓                                                        │
│   TTS (cloned Sara voice) → audio out                         │
└──────────────────┬───────────────────────────────────────────┘
                   │ HTTP POST (per tool call)
                   ▼
┌──────────────────────────────────────────────────────────────┐
│ User's Moshaar Workspace (api.moshaar.com/v2/mcp/master)       │
│  Authenticated with user's own mcp_sk_*                       │
│  All MCP calls scoped to user's permissions                   │
└──────────────────────────────────────────────────────────────┘
```

## 4. The 16 Tools — V1 Catalog

Curated subset of the 59 MCP tools. Each given an Arabic gloss the agent uses internally.

| # | Tool | Voice purpose (Arabic) | Confirmation needed? |
|---|------|------------------------|----------------------|
| 1 | `get_workflows` | يعرض workflows المتاحة (مخفي — يُستدعى تلقائياً) | لا |
| 2 | `get_workspace_members` | يحلّ اسم موظف إلى ID (مخفي) | لا |
| 3 | `get_clients` | يبحث عن عميل بالاسم (مخفي) | لا |
| 4 | `list_tasks` | "وش مهامي اليوم؟" / "ايش عندي بعد؟" | لا (قراءة فقط) |
| 5 | `get_task` | "اقرأ لي تفاصيل المهمة" | لا |
| 6 | `create_task` | "أنشئ مهمة..." | **نعم — قبل التنفيذ** |
| 7 | `update_task` | "غيّر استحقاق المهمة" / "خلّصها" | **نعم — للحذف والتغييرات الكبرى** |
| 8 | `delete_task` | "احذف المهمة" | **نعم — حذف صريح** |
| 9 | `list_cases` | "اعرض قضاياي" | لا |
| 10 | `get_case` | "اقرأ قضية X" | لا |
| 11 | `create_case` | "افتح قضية جديدة..." | **نعم** |
| 12 | `aggregate_cases_by_status` | "كم قضية معلقة؟" | لا |
| 13 | `list_calendar_sessions` | "وش جدولي؟" | لا |
| 14 | `create_calendar_session` | "احجز موعد..." | **نعم** |
| 15 | `send_message` | "أرسل رسالة لـ..." | **نعم — قبل الإرسال** |
| 16 | `send_thread_message` | "علّق على المهمة..." / "أضف ملاحظة على القضية..." | **نعم — قبل الإرسال** |

### Tool subset rationale

- **High-frequency** ops: tasks (1-8), cases (9-12), calendar (13-14)
- **Dropdown helpers** (1-3) are pre-fetched once per session and cached
- **Communication** ops (15-16) are powerful but require explicit confirmation
- **17th candidate dropped**: `create_blocknote` — voice can't generate Yjs content; instead, agent suggests *"أنشئ ملاحظة كـ thread message على المهمة"* which uses `send_thread_message` (already in catalog)

## 5. Per-User MCP Key Handling

### Why per-user?
- Multiple testers will share the voice service
- Each tester has their own Moshaar workspace + permissions
- Hardcoded master key = wrong tenant data, security exposure

### Design

**Frontend (voice-moshaar-frontend):**
- New page: `/settings/mcp` (Arabic UI: "إعدادات MCP")
- Two fields:
  - `MCP Endpoint URL` (default: `https://api.moshaar.com/v2/mcp/master`)
  - `MCP API Key` (paste field, masked input, "Test connection" button)
- "Test" button: ping the MCP server with `tools/list`, show green check if 200 + tools returned
- Stored in `localStorage` under `moshaar_mcp_*` keys
- Never sent to our server EXCEPT when establishing a WebSocket call

**Backend (server_v5_api.py):**
- WebSocket handshake reads `x-moshaar-mcp-url` and `x-moshaar-mcp-key` headers
- Headers stored in `SessionLog.mcp_config` (memory only, never persisted to disk)
- On WebSocket close: `MoshaarMCPClient.aclose()` and config dropped
- If headers missing: agent operates in "chat-only" mode (no MCP calls, just conversational)

**Security properties:**
- Master key never touches our server
- User keys live in volatile memory for the call duration only
- HTTPS terminates at our LB; key visible only to our process
- Per-user audit trail in Moshaar (their MCP key, their actions)

## 6. Lexical JSON Wrapper

Moshaar's `description` fields (task, case, calendar session) expect Lexical JSON. Voice agent never generates this directly — uses a Python helper.

```python
def to_lexical(text: str, direction: str = "rtl") -> dict:
    """
    Wrap a plain string in Lexical Editor JSON format.

    Voice agent passes Arabic text; this produces the verbose JSON the
    Moshaar frontend renders. Splits on \n\n to make paragraphs.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    children = []
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
            }]
        })
    return {
        "root": {
            "type": "root",
            "version": 1,
            "direction": direction,
            "format": "",
            "indent": 0,
            "children": children or [{
                # empty doc fallback
                "type": "paragraph",
                "version": 1,
                "direction": direction,
                "format": "", "indent": 0, "textFormat": 0, "textStyle": "",
                "children": [],
            }]
        }
    }
```

Used internally; the JSON gets `json.dumps()`-ed before being sent in the `description` field.

## 7. Block Editor HTML Wrapper

For chat messages (`send_message`, `send_thread_message`), Moshaar expects Block Editor HTML.

```python
import uuid

def to_block_editor_html(text: str) -> str:
    """Wrap plain text in Block Editor HTML for Moshaar messages."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    blocks = []
    for p in paragraphs:
        bid = str(uuid.uuid4())
        # Escape HTML special chars
        safe = (p.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;"))
        blocks.append(
            f'<div class="bn-block-outer" data-node-type="blockOuter" data-id="{bid}">'
            f'<div class="bn-block" data-node-type="blockContainer" data-id="{bid}">'
            f'<div class="bn-block-content" data-content-type="paragraph">'
            f'<p class="bn-inline-content">{safe}</p>'
            f'</div></div></div>'
        )
    return (
        '<div class="bn-block-group" data-node-type="blockGroup">'
        + "".join(blocks)
        + '</div>'
    )
```

## 8. ReAct Prompting Pattern

Gemma 4 doesn't have native function calling, so we use ReAct prompting.

### System Prompt Template

```text
أنت سارة، مساعدة صوتية لمنصة موسحار (إدارة قضايا قانونية وأعمال).
أنت تتكلمين العربية الفصحى مع لكنة سعودية خفيفة. ردودك قصيرة وعملية
(جملتين كحد أقصى، تناسب الصوت).

—————————————————————————————————————————
السلوكيات الأساسية:

1. لا تذكري الأدوات أو الـ JSON أبداً للمستخدم. هو لا يهتم.
2. عند الحاجة لمعلومة من النظام أو لتنفيذ عملية، أخرجي tool call
   بهذا الشكل الدقيق فقط:
   <tool>{"name":"...","arguments":{...}}</tool>
3. بعد كل tool call، انتظري النتيجة. لا تختلقي بيانات.
4. قبل أي عملية كتابة (create_*, update_*, delete_*, send_*):
   - أعطي تأكيداً منطوقاً واضحاً
   - انتظري "نعم" / "تأكيد" / "وافق" / "تمام"
   - عند الرفض، لا تنفّذي
5. الأرقام والمعرّفات (IDs) لا تُذكر بالصوت. استخدمي الأسماء فقط.
6. عند البحث عن شخص (موظف/عميل)، استخدمي اسم العائلة + الاسم الأول.
   إذا تطابق أكثر من شخص، اطلبي توضيحاً.
7. التواريخ النسبية ("بكرا"، "الأحد القادم") حوّليها إلى ISO 8601.
   اليوم: {current_date_iso}

—————————————————————————————————————————
الأدوات المتاحة لك (16 أداة):

[لكل أداة: اسم، استخدام، حقول مطلوبة، نموذج]

TOOL: list_tasks
USE: عرض مهام المستخدم (نشطة افتراضياً).
REQUIRED: [] (كلها اختيارية)
OPTIONAL: search, status, priority, assignee_id, case_id, is_done,
          is_archived, perPage (default 10)
EXAMPLE: <tool>{"name":"list_tasks","arguments":{"is_done":false,"perPage":5}}</tool>

TOOL: create_task
USE: إنشاء مهمة جديدة.
REQUIRED: name, workflow_id
OPTIONAL: description (سيتم لفّه في Lexical تلقائياً)، priority،
          start_date, estimated_due_date, assignee_ids, supervisor_id,
          case_id, client_id, checklist_groups
CONFIRMATION: مطلوب قبل التنفيذ.
EXAMPLE: <tool>{"name":"create_task","arguments":{"name":"مراجعة عقد",
         "workflow_id":"01H...","estimated_due_date":"2026-05-15T17:00:00Z",
         "priority":"high","description":"مراجعة بنود العقد قبل التوقيع"}}</tool>

[... 14 more tools, same format ...]

—————————————————————————————————————————
سياق الجلسة الحالية:
- مستخدم: {user_name}
- workspace: {workspace_name}
- آخر workflow استخدمتيه: {last_workflow_id_or_null}
- workflows متاحة: {workflow_summary}

—————————————————————————————————————————
أبدئي بترحيب قصير وانتظري طلب المستخدم.
```

### Turn Loop

```python
async def voice_turn(user_text: str, session: VoiceSession) -> str:
    """One conversational turn — may invoke multiple tool calls."""
    session.history.append({"role": "user", "content": user_text})

    for iteration in range(5):  # max 5 tool calls per turn
        # Call Gemma
        prompt = build_prompt(session.system_prompt, session.history, session.tool_catalog)
        gemma_response = await gemma_chat(prompt)

        # Parse for tool calls
        tool_call = parse_tool_tag(gemma_response)
        if not tool_call:
            # Plain reply — done
            session.history.append({"role": "assistant", "content": gemma_response})
            return gemma_response

        # Check confirmation gate
        if tool_call["name"] in MUTATION_TOOLS and not tool_call.get("_confirmed"):
            # Build confirmation prompt and return to user
            confirmation = build_confirmation_message(tool_call)
            session.pending_tool = tool_call
            session.history.append({"role": "assistant", "content": confirmation})
            return confirmation

        # Execute MCP call
        try:
            result = await session.mcp.call(
                tool_call["name"], tool_call["arguments"]
            )
        except MCPError as e:
            result = {"error": str(e)}

        # Feed result back
        session.history.append({"role": "assistant", "content": gemma_response})
        session.history.append({
            "role": "tool",
            "name": tool_call["name"],
            "content": json.dumps(result, ensure_ascii=False)[:2000],  # cap
        })
        # Continue loop — Gemma may now respond or call another tool

    # Hit iteration cap — return whatever Gemma said last
    return "اعتذار، الطلب يحتاج وقت أطول. هل تريد المحاولة مرة أخرى؟"
```

### Confirmation Handling

When user responds to a pending tool call:

```python
async def handle_confirmation_response(user_text: str, session: VoiceSession):
    if not session.pending_tool:
        return None
    affirmatives = ["نعم", "أيوه", "تمام", "وافق", "أكد", "موافق", "تأكيد"]
    negatives = ["لا", "ألغ", "إلغاء", "ما أبي", "ولا تكمل"]
    text_lower = user_text.strip().lower()

    if any(a in text_lower for a in affirmatives):
        session.pending_tool["_confirmed"] = True
        return await execute_tool_call(session.pending_tool, session)
    elif any(n in text_lower for n in negatives):
        session.pending_tool = None
        return "تمام، ألغيت العملية."
    else:
        # Treat as new request, drop pending
        session.pending_tool = None
        return None  # caller proceeds with normal turn
```

## 9. Error Handling

| Error case | Behavior |
|------------|----------|
| MCP unreachable (network error) | Say *"المنصة غير متاحة الآن، حاول بعد لحظات."* + log + return |
| MCP auth error (401/403) | Say *"مفتاحك غير صحيح. راجع إعدادات MCP."* |
| MCP returns error in JSON-RPC `error` field | Say a humanized version of the error message |
| Gemma generates invalid JSON in `<tool>` | Retry once with strict instruction; if fails, say *"حاول بصياغة مختلفة"* |
| Tool not in V1 catalog | Treat as no-op + say *"هذي العملية مش متاحة بعد."* |
| Confirmation timeout (no user response 30s) | Auto-cancel + say *"ألغيت طلب التأكيد بعد التأخر."* |

## 10. File Structure

```
vibevoice_streaming/
├── moshaar_mcp.py                 # NEW — MCP client (httpx + JSON-RPC)
├── voice_agent_prompt.py          # NEW — system prompt builder
├── voice_agent_tools.py           # NEW — tool catalog + Arabic glosses
├── voice_agent_helpers.py         # NEW — to_lexical(), to_block_editor_html()
├── gemma4_server.py               # MODIFIED — adds /v1/chat-with-tools endpoint
├── server_v5_api.py               # MODIFIED — WS handler accepts MCP headers,
│                                              wires VoiceSession to MCP client
└── voice_session.py               # NEW — per-WS session state container

docs/superpowers/specs/
└── 2026-05-12-voice-agent-mcp-integration.md   # this file

infra/
└── .env.example                   # MODIFIED — no new server-side vars
                                   #             (MCP key is per-user, not env)

voice-moshaar-frontend/
├── src/pages/settings.tsx         # NEW — MCP settings page
├── src/hooks/useMoshaarMCP.ts     # NEW — localStorage adapter
└── src/pages/call.tsx             # MODIFIED — sends MCP headers on WS open
```

## 11. API Surface Changes

### Backend

**Modified WebSocket: `/v1/conversation/ws`**

Now accepts headers:
- `x-moshaar-mcp-url` (optional — defaults to `https://api.moshaar.com/v2/mcp/master`)
- `x-moshaar-mcp-key` (optional — if missing, agent runs in chat-only mode)

Sends new event types in addition to existing transcript / audio events:
- `event: tool_call` — JSON `{name, arguments, requires_confirmation}`
- `event: tool_result` — JSON `{name, result_summary}` (truncated for UI)
- `event: confirmation_pending` — JSON `{action_summary_arabic, action_summary_english}`

**Modified Gemma sidecar: `/v1/chat` (`gemma4_server.py`)**

Now optionally accepts:
- `system_prompt` (string) — full system prompt with tools catalog
- `tool_catalog_hint` (boolean, default false) — whether to bias toward tool use

### Frontend

**New Page: `/settings/mcp`**
- Fields: endpoint URL, API key
- Buttons: Save, Test, Clear
- On save: store in localStorage
- On test: ping `tools/list` directly from browser (works because MCP is HTTP)

**Modified Page: `/call`**
- On open WS: pass MCP headers from localStorage
- New UI: small "MCP connected ✓ / disconnected ✗" indicator
- New UI: confirmation modal when agent requests confirmation (with TTS voice still asking the same)

## 12. Testing Strategy

### Unit tests (`tests/voice_agent/`)

- `test_to_lexical.py` — Arabic text, multi-paragraph, empty
- `test_to_block_editor_html.py` — HTML escaping, multi-paragraph
- `test_parse_tool_tag.py` — well-formed, malformed, no tag, multiple tags
- `test_confirmation_gate.py` — affirmatives/negatives/ambiguous in Arabic
- `test_moshaar_mcp_client.py` — mocked HTTP responses

### Integration tests (`tests/voice_agent_integration/`)

Requires a sandbox Moshaar workspace + test API key (loaded from env):
- Full WS turn: "اعرض مهامي" → list_tasks → response
- Confirmation flow: "أنشئ مهمة X" → confirmation → "نعم" → create_task
- Cancellation: "أنشئ مهمة X" → "لا" → no MCP call made
- ID resolution: "أرسل لـ سعد رسالة" → get_workspace_members(search=سعد) → send_message
- Error path: bad key → 401 → graceful error message

### E2E (manual, weekly)

Three scenarios on staging:
1. **Task creation flow** — voice → confirm → MCP create → UI verify
2. **Calendar booking** — voice → confirm → session created on Moshaar
3. **Multi-step** — "Show pending cases, then book a call with the first client"

## 13. Implementation Plan (8 Tasks, ~6-8 hours)

| # | Task | Estimate |
|---|------|----------|
| 1 | `moshaar_mcp.py` — async httpx MCP client with timeout, retries, error parsing | 45 min |
| 2 | `voice_agent_helpers.py` — to_lexical + to_block_editor_html + unit tests | 30 min |
| 3 | `voice_agent_tools.py` — 16-tool catalog with Arabic glosses + confirmation flags | 30 min |
| 4 | `voice_agent_prompt.py` — system prompt builder (dynamic with session context) | 45 min |
| 5 | `voice_session.py` — VoiceSession class + state container | 30 min |
| 6 | `gemma4_server.py` — extend `/v1/chat` to accept system_prompt + parse tool calls | 60 min |
| 7 | `server_v5_api.py` — WS header parsing, session wiring, ReAct loop integration | 90 min |
| 8 | Frontend `/settings/mcp` page + `/call` header injection | 90 min |

**Total: ~6-8 hours of focused work**

## 14. Open Questions / Risks

1. **Gemma 4's tool-call accuracy** — does it reliably emit `<tool>...</tool>` after the system prompt instructions? Will need eval.
2. **Latency budget** — current TTFA ~250ms (TTS only). Tool call adds 1 Gemma round-trip + 1 MCP round-trip = ~600-1500ms extra. Acceptable for non-streaming responses.
3. **Confirmation UX over voice** — Arabic affirmative detection is fuzzy ("ايوة"، "تمام"، "ماشي"). Need broad keyword set + maybe fall back to "say yes or no".
4. **Multi-tenant audit trail** — should we log MCP calls (sanitized) for debugging? Yes — log tool name + args (with PII scrubbed) + status, never the raw key.
5. **Date parsing** — "بعد بكرا"، "نهاية الأسبوع"، "الإثنين القادم" all need Gemma to compute ISO 8601 correctly. We give it `{current_date_iso}` in the prompt; eval on common phrasings.

## 15. Rollout

- **Phase 1:** Behind feature flag `VOICE_MCP_ENABLED=true` env var. Internal testing.
- **Phase 2:** Open to first 3 testers (with their own MCP keys). Monitor:
  - tool success rate (target >90%)
  - false confirmation rate (target <5%)
  - turn latency p95 (target <2.5s)
- **Phase 3:** Enable for all users with valid MCP config.

## 16. Reference

- Moshaar MCP spec: `https://api.moshaar.com/v2/mcp/master` (call `get_mcp_guidance` for live docs)
- Full tool catalog: `docs/moshaar-tools.json` (59 tools, generated 2026-05-12)
- This spec replaces the earlier brainstorm in chat sessions 2026-05-11.

---

**Next steps after approval:**
1. Write the implementation plan in `docs/superpowers/plans/2026-05-12-voice-agent-mcp-implementation.md`
2. Begin Task 1 (`moshaar_mcp.py`) using TDD
