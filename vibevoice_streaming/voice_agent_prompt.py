"""System prompt builder for the voice agent.

Generates the Arabic system prompt that teaches Gemma 4 the ReAct pattern
for tool calling. The prompt is built dynamically so we can inject:

  • current date / time (for relative-date parsing)
  • cached workflow + member summaries (so agent can reference them)
  • whether MCP is connected (different mode if no key)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

try:
    from voice_agent_helpers import now_iso, today_iso  # type: ignore[no-redef]
    from voice_agent_tools import VOICE_TOOLS, VoiceTool  # type: ignore[no-redef]
except ImportError:
    from .voice_agent_helpers import now_iso, today_iso
    from .voice_agent_tools import VOICE_TOOLS, VoiceTool


# ──────────────────────────────────────────────────────────────────────────
# Persona & behavior — the unchanging part of the prompt
# ──────────────────────────────────────────────────────────────────────────

PERSONA = """\
أنت "سارة"، مساعدة صوتية لمنصة مستشار (إدارة قضايا قانونية وأعمال).
تتكلمين العربية الفصحى المبسّطة مع لمسة سعودية. ردودك للمستخدم
قصيرة وعملية — جملتان كحد أقصى عادةً، لأنها ستُنطق بالصوت.

أنتِ مساعدة **مبادِرة**: اعملي بنفسك حتى تنجزي طلب المستخدم بدون
أن تسأليه عن خطوات وسيطة. استخدمي الأدوات المتاحة بتسلسل، اقرئي
النتائج، ثم استدعِ الأداة التالية تلقائياً حتى تكتمل المهمة."""


BEHAVIOR_RULES = """\
السلوكيات الأساسية (التزمي بها بدقة):

١. لا تذكري للمستخدم أبداً أنك تستخدمين "أدوات" أو JSON أو IDs. هو لا يهتم.

٢. **كوني مبادِرة، لا تطلبي توضيحات قبل التنفيذ**. إذا طلب المستخدم
   "كل المواعيد" أو "كل القضايا" أو "كل المهام":
     - استدعي أدوات الفهرسة (list_calendars, list_cases, ...) أولاً
     - ثم استدعي أدوات التفاصيل (list_calendar_sessions لكل تقويم) بنفسك
     - **اجمعي النتائج**، ولخّصي للمستخدم في النهاية
   لا تقولي أبداً "أحتاج أن أعرف أي تقويم" أو "أي قضية" قبل ما تجرّبي
   الفهرسة. فقط بعد ما تكون النتائج غامضة فعلاً (10 موكلين بنفس الاسم
   مثلاً) اسألي عن التوضيح.

٣. **مهام متعددة الخطوات: استمري حتى الإنجاز**. لا تتوقفي بعد أول
   استدعاء أداة إذا الطلب يحتاج خطوات أكثر. استدعي الأداة التالية
   فوراً. لا تقولي "أتقدر أعمل لك x بعدين؟" — جرّبيها مباشرة.

٤. قبل أي عملية تعدّل البيانات (create_*, update_*, delete_*, send_*):
     - أوصفي العملية بصوت عربي طبيعي وتأكدي شفهياً.
     - انتظري "نعم" أو "تمام" أو "وافق" قبل التنفيذ.
     - عند سماع "لا" أو "إلغاء" — لا تنفّذي وقولي "تمام، ألغيت".

٥. IDs والـ tokens لا تُذكر بالصوت. استخدمي الأسماء فقط.

٦. التواريخ النسبية ("بكرا"، "نهاية الأسبوع"، "الإثنين القادم") حوّليها إلى
   ISO 8601 UTC قبل وضعها في tool call.

٧. عند نجاح عملية، أعطي ملخصاً شفهياً قصيراً يذكر النتيجة الرئيسية فقط.
   لو النتائج كثيرة، اذكري العدد الإجمالي + أبرز 3-5 عناصر.

٨. عند خطأ من النظام، اشرحي السبب بكلمات بسيطة دون مصطلحات تقنية،
   ولا تستسلمي عند خطأ أول — جرّبي مرة ثانية بمعطيات مختلفة قبل ما تخبري
   المستخدم بفشل.
"""


# ──────────────────────────────────────────────────────────────────────────
# Tool catalog rendering — converts VoiceTool list to prompt text
# ──────────────────────────────────────────────────────────────────────────

def _render_tool(t: VoiceTool) -> str:
    confirm_marker = " ⚠️ تأكيد مطلوب" if t.needs_confirmation else ""
    req = ", ".join(t.required) if t.required else "—"
    opt = ", ".join(t.optional) if t.optional else "—"
    notes_block = f"\n   ملاحظات: {t.notes}" if t.notes else ""
    return (
        f"• {t.name}{confirm_marker}\n"
        f"   الغرض: {t.purpose_ar}\n"
        f"   حقول مطلوبة: {req}\n"
        f"   حقول اختيارية: {opt}\n"
        f"   مثال متى تُستخدم: {t.example_arabic_intent}"
        f"{notes_block}"
    )


def render_tool_catalog(tools: Optional[List[VoiceTool]] = None) -> str:
    """Render the 16-tool catalog in Arabic for the system prompt."""
    if tools is None:
        tools = VOICE_TOOLS
    body = "\n\n".join(_render_tool(t) for t in tools)
    return (
        "الأدوات المتاحة لك (16 أداة مستشار):\n\n"
        + body
        + "\n\n— نهاية القائمة —"
    )


# ──────────────────────────────────────────────────────────────────────────
# Context block — injected at the bottom of the system prompt
# ──────────────────────────────────────────────────────────────────────────

def render_context(
    *,
    user_name: Optional[str] = None,
    workspace_name: Optional[str] = None,
    workflows: Optional[List[Dict[str, Any]]] = None,
    cached_members: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render the dynamic session context."""
    parts = ["سياق الجلسة الحالية:"]
    parts.append(f"• الوقت الحالي UTC: {now_iso()}")
    parts.append(f"• تاريخ اليوم: {today_iso()}")
    if user_name:
        parts.append(f"• المستخدم: {user_name}")
    if workspace_name:
        parts.append(f"• اسم workspace: {workspace_name}")

    if workflows:
        wf_lines = [f"   - {wf.get('name', '?')}" for wf in workflows[:5]]
        parts.append("• workflows متاحة:")
        parts.extend(wf_lines)

    if cached_members:
        names = [m.get("name", "?") for m in cached_members[:8]]
        parts.append(f"• أعضاء الفريق المعروفون: {'، '.join(names)}")

    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# Full prompt assembly
# ──────────────────────────────────────────────────────────────────────────

def build_system_prompt(
    *,
    mcp_connected: bool,
    user_name: Optional[str] = None,
    workspace_name: Optional[str] = None,
    workflows: Optional[List[Dict[str, Any]]] = None,
    cached_members: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the full Arabic system prompt for the voice agent.

    Args:
        mcp_connected: True if user has provided a valid MCP key (Moshaar
            access enabled). False → chat-only mode, no tools advertised.
        user_name: Optional — for personalization
        workspace_name: Optional — name of the user's Moshaar workspace
        workflows: Optional — cached `get_workflows()` result
        cached_members: Optional — cached `get_workspace_members()` result

    Returns:
        Full prompt text ready to send as the `system_prompt` field.
    """
    sections = [PERSONA, "", BEHAVIOR_RULES, ""]

    if mcp_connected:
        sections.append(render_tool_catalog())
        sections.append("")
        sections.append(render_context(
            user_name=user_name,
            workspace_name=workspace_name,
            workflows=workflows,
            cached_members=cached_members,
        ))
        sections.append("")
        sections.append(
            "ابدئي الجلسة بترحيب قصير (جملة واحدة) ثم انتظري طلب المستخدم. "
            "لو سألك المستخدم 'ايش تقدرين تسوين؟'، اذكري 3 إلى 4 أمثلة فقط."
        )
    else:
        sections.append(
            "⚠️ تنبيه: مفتاح مستشار غير مُعدّ. أنتِ تعملين في وضع 'محادثة فقط' — \n"
            "لا تستطيعين الوصول إلى المهام أو القضايا أو التقويم. إذا حاول المستخدم \n"
            "طلب عملية، وجّهيه إلى زر 'إعدادات MCP' لإضافة المفتاح."
        )

    return "\n".join(sections)


# ──────────────────────────────────────────────────────────────────────────
# Tool-call parsing — extracts <tool>...</tool> from Gemma's output
# ──────────────────────────────────────────────────────────────────────────

import re

_TOOL_TAG_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)


def parse_tool_call(gemma_output: str) -> Optional[Dict[str, Any]]:
    """Extract the first <tool>...</tool> JSON from Gemma's response.

    Returns:
        Dict with 'name' and 'arguments' keys, or None if no valid tool tag.
    """
    match = _TOOL_TAG_RE.search(gemma_output or "")
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    if "name" not in obj or "arguments" not in obj:
        return None
    if not isinstance(obj.get("arguments"), dict):
        return None
    return obj


def strip_tool_call(gemma_output: str) -> str:
    """Return Gemma's output with all <tool>...</tool> blocks removed."""
    return _TOOL_TAG_RE.sub("", gemma_output or "").strip()


# ──────────────────────────────────────────────────────────────────────────
# Confirmation classification — recognize Arabic affirmatives / negatives
# ──────────────────────────────────────────────────────────────────────────

AFFIRMATIVE_KEYWORDS = [
    "نعم", "ايوه", "ايوا", "أيوه", "أيوا",
    "تمام", "ماشي", "موافق", "وافق", "اوكي", "أوكي",
    "أكد", "تأكيد", "أكيد", "اكيد",
    "صح", "زبط",
]


NEGATIVE_KEYWORDS = [
    "لا", "لأ", "ما أبي", "ما ابي", "ما ابغى",
    "ألغ", "ألغي", "الغ", "الغي", "إلغاء", "الغاء",
    "ما يصير", "لا تكمل",
    "ولا تكمل", "خلاص لا",
]


def classify_confirmation(user_text: str) -> str:
    """Classify a confirmation response.

    Returns:
        'yes' | 'no' | 'unclear'
    """
    if not user_text:
        return "unclear"
    text = user_text.strip().lower()

    # Check negatives first (since "لا" can be substring of other words)
    if any(neg in text for neg in NEGATIVE_KEYWORDS):
        # but not if it's clearly an affirmative containing a "لا" by accident
        if not any(aff == text.strip() or text.startswith(aff + " ") for aff in AFFIRMATIVE_KEYWORDS):
            return "no"
    if any(aff in text for aff in AFFIRMATIVE_KEYWORDS):
        return "yes"
    return "unclear"
