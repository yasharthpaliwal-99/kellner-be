"""
Structured assistant replies for specific tool flows: [SPEAK]...[/SPEAK] + [SHOW]...[/SHOW].

Only one format appendix is injected per turn (token-efficient). Default turns use the
legacy LLM stream + TTS path unchanged.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

ResponseMode = Literal["none", "bill", "order_confirmation", "recommendations"]

# Injected only when mode != "none", after tool phase, before the streaming completion.
# [SHOW] must contain a single JSON object (no markdown fences). Frontend also gets
# `assistant_reply_mode` + `assistant_structured` over the WebSocket for typed payloads.
_FORMAT_APPENDIX: dict[str, str] = {
    "bill": (
        "For THIS reply only, return exactly two tags and nothing else (no text outside tags).\n"
        "[SPEAK] One short sentence (<=25 words) about bill/payment. [/SPEAK]\n"
        "[SHOW] A single JSON object only, no markdown code fences, matching this shape exactly:\n"
        '{"items":[{"name":"<string>","quantity":<number>,"price":<number or null>}],"total":<number or null>}\n'
        "Use tool results only; use null if unknown. Numbers for quantity/price/total. [/SHOW]"
    ),
    "order_confirmation": (
        "For THIS reply only, return exactly two tags and nothing else (no text outside tags).\n"
        "[SPEAK] One short confirmation line (<=25 words). [/SPEAK]\n"
        "[SHOW] A single JSON object only, no markdown code fences, matching this shape exactly:\n"
        '{"items":[{"name":"<string>","quantity":<number>}]}\n'
        "Use tool results only. [/SHOW]"
    ),
    "recommendations": (
        "For THIS reply only, return exactly two tags and nothing else (no text outside tags).\n"
        "[SPEAK] One short warm reaction (<=25 words). Do not read every dish name or price aloud. [/SPEAK]\n"
        "[SHOW] A single JSON object only, no markdown code fences, matching this shape exactly:\n"
        '{"recommendation_focus":"<string>","items":[{"name":"<string>","quantity":<number>,"price":<number or null>}]}\n'
        "recommendation_focus summarizes the guest ask (e.g. spicy, vegetarian). "
        "Items should match the on-screen recommendation cards. Use null for unknown price. [/SHOW]"
    ),
}


def parse_show_payload(show: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Parse the SHOW block as JSON. Strips optional ```json fences if the model adds them.
    Returns (dict_or_None, canonical_string_for_history) — history string is JSON if parse ok.
    """
    raw = (show or "").strip()
    if not raw:
        return None, ""
    s = raw
    if s.startswith("```"):
        lines = s.split("\n")
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        return None, raw
    if not isinstance(data, dict):
        return None, raw
    canonical = json.dumps(data, ensure_ascii=False)
    return data, canonical


def detect_response_mode(
    tools_used: List[str],
    menu_recommendation_count: int,
) -> ResponseMode:
    """
    Pick at most one structured mode. Priority: bill > order_confirmation > recommendations.
    """
    s = set(tools_used or [])
    if "bring_the_bill" in s or "review_and_feedback" in s:
        return "bill"
    if "get_current_order" in s or "place_order" in s or "modify_order" in s:
        return "order_confirmation"
    if "get_menu_items" in s and menu_recommendation_count > 0:
        return "recommendations"
    return "none"


def append_format_instruction(messages: List[dict], mode: ResponseMode) -> List[dict]:
    """Return a shallow copy of messages with a one-turn format system message appended."""
    if mode == "none":
        return messages
    appendix = _FORMAT_APPENDIX.get(mode)
    if not appendix:
        return messages
    out = list(messages)
    out.append(
        {
            "role": "system",
            "content": "OUTPUT FORMAT (this assistant message only — follow exactly):\n" + appendix,
        }
    )
    return out


def parse_speak_show(text: str) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Extract spoken (TTS) and display text from a tagged reply.
    Returns (spoken, show, ok). If ok is False, caller should fall back to legacy behaviour.
    """
    if not (text or "").strip():
        return None, None, False
    sp = re.search(r"\[SPEAK\](.*?)\[/SPEAK\]", text, re.DOTALL | re.IGNORECASE)
    sh = re.search(r"\[SHOW\](.*?)\[/SHOW\]", text, re.DOTALL | re.IGNORECASE)
    if not sp or not sh:
        return None, None, False
    spoken = (sp.group(1) or "").strip()
    show = (sh.group(1) or "").strip()
    if not spoken and not show:
        return None, None, False
    return spoken or None, show or None, True


def assistant_history_content(spoken_ok: bool, show: Optional[str], full_raw: str) -> str:
    """What to store in conversation history for the assistant turn."""
    if spoken_ok and (show or "").strip():
        return (show or "").strip()
    return full_raw.strip()


def history_after_show(
    spoken_ok: bool,
    show_raw: Optional[str],
    full_raw: str,
    payload: Optional[Dict[str, Any]],
) -> str:
    """Prefer stable JSON for history when SHOW parsed as an object."""
    if spoken_ok and payload is not None:
        return json.dumps(payload, ensure_ascii=False)
    return assistant_history_content(spoken_ok, show_raw, full_raw)
