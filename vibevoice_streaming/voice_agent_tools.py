"""Voice-agent tool catalog — the 16 Moshaar MCP tools we expose to Gemma.

Of the 59 tools Moshaar's MCP exposes, V1 uses 16 curated ones — the
high-frequency operations that make sense over voice. See
`docs/superpowers/specs/2026-05-12-voice-agent-mcp-integration.md` for
the design rationale and the full catalog.

Each entry includes:
  • the tool name (snake_case as Moshaar exposes it)
  • a short Arabic gloss the agent uses internally
  • whether confirmation is required before execution
  • a compact JSON-schema-like spec the prompt builder will paginate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class VoiceTool:
    name: str
    purpose_ar: str          # one-line Arabic purpose for the system prompt
    required: List[str]       # required argument names (subset for voice)
    optional: List[str]       # optional argument names (subset for voice)
    needs_confirmation: bool  # True if mutation — agent must confirm first
    example_arabic_intent: str  # an example user utterance that triggers this
    notes: str = ""           # extra hints (e.g. dropdown dependencies)


# ──────────────────────────────────────────────────────────────────────────
# The 16-tool V1 catalog
# ──────────────────────────────────────────────────────────────────────────

VOICE_TOOLS: List[VoiceTool] = [
    # ── Dropdowns (used by the agent transparently — never confirmation) ──
    VoiceTool(
        name="get_workflows",
        purpose_ar="استعرض أنواع workflows المتاحة لإنشاء مهمة (تُستدعى تلقائياً).",
        required=[],
        optional=["search", "page", "perPage"],
        needs_confirmation=False,
        example_arabic_intent="(غير ظاهر للمستخدم)",
        notes="استدعِ هذي قبل create_task للحصول على workflow_id. خزّن النتيجة في الذاكرة.",
    ),
    VoiceTool(
        name="list_calendars",
        purpose_ar="اعرض جميع التقاويم. يُستدعى قبل أي عملية تقويم لجلب calendar_id.",
        required=[],
        optional=["search", "type", "perPage"],
        needs_confirmation=False,
        example_arabic_intent="(تلقائي قبل list_calendar_sessions أو create_calendar_session)",
        notes=(
            "النوع primary = تقويم المستخدم الشخصي، default = تقويم المساحة المشتركة.\n"
            "خزّن النتيجة في الذاكرة كي لا تكرّر النداء."
        ),
    ),
    VoiceTool(
        name="get_workspace_members",
        purpose_ar="ابحث عن موظف بالاسم لتحويله إلى ID.",
        required=[],
        optional=["search", "page", "perPage"],
        needs_confirmation=False,
        example_arabic_intent="(تلقائي عند ذكر اسم موظف)",
        notes="استخدم 'search' بالاسم العربي.",
    ),
    VoiceTool(
        name="get_clients",
        purpose_ar="ابحث عن عميل بالاسم لتحويله إلى client_id.",
        required=[],
        optional=["search", "page", "perPage"],
        needs_confirmation=False,
        example_arabic_intent="(تلقائي عند ذكر اسم عميل)",
        notes="إذا تطابق أكثر من عميل، اطلب من المستخدم توضيح الاسم.",
    ),

    # ── Read-only queries ──
    VoiceTool(
        name="list_tasks",
        purpose_ar="اعرض مهام المستخدم. الافتراضي: 5 مهام نشطة.",
        required=[],
        optional=["search", "is_done", "is_archived", "priority", "case_id",
                  "client_id", "assignee_id", "perPage"],
        needs_confirmation=False,
        example_arabic_intent="ايش مهامي اليوم؟ / اعرض المهام النشطة",
        notes="استخدم perPage=5 افتراضياً. is_done=false للنشطة.",
    ),
    VoiceTool(
        name="get_task",
        purpose_ar="اقرأ تفاصيل مهمة معينة.",
        required=["task_id"],
        optional=[],
        needs_confirmation=False,
        example_arabic_intent="اقرأ لي تفاصيل المهمة الأولى",
        notes="task_id يأتي من نتائج list_tasks.",
    ),
    VoiceTool(
        name="list_cases",
        purpose_ar="اعرض قضايا/مشاريع المستخدم.",
        required=[],
        optional=["search", "status", "priority", "client_id", "is_archived", "perPage"],
        needs_confirmation=False,
        example_arabic_intent="اعرض قضاياي / كم قضية معي؟",
    ),
    VoiceTool(
        name="get_case",
        purpose_ar="اقرأ تفاصيل قضية معينة.",
        required=["case_id"],
        optional=[],
        needs_confirmation=False,
        example_arabic_intent="اقرأ قضية الموكل أحمد",
    ),
    VoiceTool(
        name="aggregate_cases_by_status",
        purpose_ar="إحصائيات: كم قضية في كل حالة.",
        required=[],
        optional=[],
        needs_confirmation=False,
        example_arabic_intent="كم قضية في الانتظار؟ / إحصائيات القضايا",
    ),
    VoiceTool(
        name="list_calendar_sessions",
        purpose_ar="اعرض المواعيد/الجلسات. يجب تحديد calendar_id لتجنّب خطأ الخادم.",
        required=["calendar_id"],
        optional=["search", "type", "form_type", "start_date", "end_date", "perPage", "is_completed"],
        needs_confirmation=False,
        example_arabic_intent="ايش جدولي اليوم؟ / مواعيدي بكرا",
        notes=(
            "calendar_id لازم يكون array من ids: [\"01K...\"], لا تمرّره كنص.\n"
            "خادم مستشار فيه خلل serialize BigInt إذا استعلمنا كل الجلسات دفعة واحدة، "
            "فنحن نُجبر تحديد التقويم. استدعِ list_calendars أولاً والتقط الـ primary "
            "للمستخدم (أو حسب الاسم).\n"
            "start_date / end_date تنسيق ISO 8601 UTC. perPage<=4 آمن للتقاويم "
            "التي بها سجلّات فاسدة. للنتائج الأكبر، استخدم start_date+end_date لتضييق "
            "النطاق."
        ),
    ),

    # ── Mutations (CONFIRMATION REQUIRED) ──
    VoiceTool(
        name="create_task",
        purpose_ar="أنشئ مهمة جديدة. يحتاج workflow_id من get_workflows أولاً.",
        required=["name", "workflow_id"],
        optional=["description", "priority", "case_id", "client_id",
                  "supervisor_id", "assignee_ids", "start_date",
                  "estimated_due_date", "checklist_groups"],
        needs_confirmation=True,
        example_arabic_intent="أنشئ مهمة لمراجعة عقد أحمد قبل الخميس",
        notes=("description لازم Lexical JSON. الـ wrapper يحوّل النص العربي تلقائياً.\n"
               "estimated_due_date تنسيق ISO 8601: 2026-05-15T17:00:00Z"),
    ),
    VoiceTool(
        name="update_task",
        purpose_ar="حدّث مهمة (تغيير حالة، تاريخ، أولوية، اسم...).",
        required=["task_id"],
        optional=["name", "description", "priority", "estimated_due_date",
                  "start_date", "status_on_template_id", "case_id", "client_id",
                  "supervisor_id", "assignee_ids", "is_archived"],
        needs_confirmation=True,
        example_arabic_intent="غيّر استحقاق مهمة العقد للأحد / أرشف المهمة",
    ),
    VoiceTool(
        name="delete_task",
        purpose_ar="احذف مهمة (soft delete — يمكن استرجاعها).",
        required=["task_id"],
        optional=[],
        needs_confirmation=True,
        example_arabic_intent="احذف المهمة الأخيرة",
        notes="تأكيد إضافي مطلوب — هذي عملية حذف.",
    ),
    VoiceTool(
        name="create_case",
        purpose_ar="افتح قضية/مشروع جديد.",
        required=["name", "priority", "status", "settings"],
        optional=["description", "start_date", "end_date", "team_member_ids",
                  "client_id", "principal_lawyer_id", "specialty_id"],
        needs_confirmation=True,
        example_arabic_intent="افتح قضية جديدة لـ سعد بأولوية عاجلة",
        notes="settings يمكن أن يكون {} كافتراضي.",
    ),
    VoiceTool(
        name="create_calendar_session",
        purpose_ar="احجز موعد/جلسة. يحتاج calendar_id من list_calendars أولاً.",
        required=["calendar_id", "name", "type", "form_type",
                  "start_date", "end_date", "privacy_setting"],
        optional=["client_id", "case_id", "all_day", "location", "availability",
                  "color", "description", "attendees", "reminders"],
        needs_confirmation=True,
        example_arabic_intent="احجز موعد مع أحمد بكرا الساعة 3",
        notes=("type: appointment | call | email\n"
               "form_type: event | consultation (لا session للأعمال)\n"
               "privacy_setting: public | private\n"
               "التواريخ ISO 8601 UTC."),
    ),
    VoiceTool(
        name="send_message",
        purpose_ar="أرسل رسالة في محادثة موجودة.",
        required=["chatId", "content"],
        optional=["mentionedWorkspaceMembers", "replyToId"],
        needs_confirmation=True,
        example_arabic_intent="أرسل لـ سعد رسالة: 'الاجتماع تأخر ساعة'",
        notes="content يجب أن يكون Block Editor HTML — الـ wrapper يحوّل.",
    ),
    VoiceTool(
        name="send_thread_message",
        purpose_ar="علّق على مهمة عبر thread (= التعليقات في النظام).",
        required=["threadId", "content"],
        optional=["mentionedWorkspaceMembers", "replyToId",
                  "case_id", "task_id", "client_id", "checklist"],
        needs_confirmation=True,
        example_arabic_intent="علّق على المهمة 'تمت المراجعة الأولى'",
        notes=("content = Block Editor HTML.\n"
               "task_id لربط الـ thread بمهمة.\n"
               "checklist: array من {title, is_completed} لإضافة عناصر مهمة."),
    ),
]


# Fast lookup by name
TOOLS_BY_NAME: Dict[str, VoiceTool] = {t.name: t for t in VOICE_TOOLS}


# Just the names that require confirmation
MUTATION_TOOLS: List[str] = [t.name for t in VOICE_TOOLS if t.needs_confirmation]


# Tools that the agent calls transparently (no confirmation, used to resolve IDs)
DROPDOWN_TOOLS: List[str] = [
    "get_workflows",
    "list_calendars",
    "get_workspace_members",
    "get_clients",
]


def get_tool(name: str) -> VoiceTool | None:
    """Return the VoiceTool definition or None if not in the V1 catalog."""
    return TOOLS_BY_NAME.get(name)


def needs_confirmation(tool_name: str) -> bool:
    """True if the tool is a mutation that must be confirmed first."""
    t = TOOLS_BY_NAME.get(tool_name)
    return bool(t and t.needs_confirmation)
