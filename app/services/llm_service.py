import json
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from openai import AsyncAzureOpenAI, AzureOpenAI

from app.config import config
from app.services.response_format import detect_response_mode, parse_speak_show
from app.services.session_context import get_session
from app.services.tool_executor import ToolExecutor

# Tool rounds only choose/execute tools — keep completions tiny to avoid slow prose.
TOOL_ROUND_MAX_TOKENS = 128
# GPT-5 models spend completion budget on internal reasoning tokens before tool/text output.
GPT5_TOOL_ROUND_MAX_TOKENS = 512
# [SPEAK]+[SHOW] JSON fits in ~120–180 tokens; avoids 400-token menu essays after tools.
FORMAT_STREAM_MAX_TOKENS = 180
GPT5_FORMAT_STREAM_MAX_TOKENS = 400
# Legacy plain stream when mode=none but tools ran (no structured tags).
PLAIN_STREAM_MAX_TOKENS = 200
GPT5_PLAIN_STREAM_MAX_TOKENS = 400


def _deployment_name_lower() -> str:
    return (config.AZURE_OPENAI_DEPLOYMENT_NAME or "").lower()


def completion_limit_kwargs(limit: int) -> Dict[str, int]:
    """GPT-5 deployments use max_completion_tokens; gpt-4o and older use max_tokens."""
    if _deployment_name_lower().startswith("gpt-5"):
        return {"max_completion_tokens": limit}
    return {"max_tokens": limit}


def chat_temperature_kwargs(temperature: float) -> Dict[str, float]:
    """GPT-5 chat only supports the default temperature; omit the parameter."""
    if _deployment_name_lower().startswith("gpt-5"):
        return {}
    return {"temperature": temperature}


def gpt5_model_kwargs() -> Dict[str, Any]:
    """Extra chat params for GPT-5 deployments (reasoning budget + minimal reasoning effort)."""
    if not _deployment_name_lower().startswith("gpt-5"):
        return {}
    return {"reasoning_effort": "minimal"}


def tool_round_limit() -> int:
    if _deployment_name_lower().startswith("gpt-5"):
        return GPT5_TOOL_ROUND_MAX_TOKENS
    return TOOL_ROUND_MAX_TOKENS


def format_stream_limit() -> int:
    if _deployment_name_lower().startswith("gpt-5"):
        return GPT5_FORMAT_STREAM_MAX_TOKENS
    return FORMAT_STREAM_MAX_TOKENS


def plain_stream_limit() -> int:
    if _deployment_name_lower().startswith("gpt-5"):
        return GPT5_PLAIN_STREAM_MAX_TOKENS
    return PLAIN_STREAM_MAX_TOKENS


def _recommendations_from_get_menu_json(result_json: str) -> List[Dict[str, Any]]:
    """Parse get_menu_items tool output into UI-friendly rows (only when search succeeded)."""
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict) or data.get("error"):
        return []
    raw = data.get("items")
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        name = it.get("name")
        if not name:
            continue
        desc = (it.get("description") or "").strip()
        cuisine = (it.get("cuisine_type") or "").strip()
        info_parts = [p for p in [cuisine, desc[:280] if desc else ""] if p]
        info = " · ".join(info_parts) if info_parts else ""
        price = it.get("price")
        if price is not None:
            try:
                price = float(price)
            except (TypeError, ValueError):
                price = None
        image = it.get("image")
        image_url = str(image).strip() if image is not None else ""
        out.append(
            {
                "name": str(name),
                "price": price,
                "info": info,
                "image": image_url or None,
            }
        )
    return out

_SYSTEM_PROMPT = """You are a professional restaurant waiter at a dine-in restaurant (not a hotel). Be warm, concise, and helpful.
- Do not mention room service, hotel stays, or front-desk services. Help with the menu, ordering, bill, and table requests only.
- Do not ask for the guest's name, age, or other personal details. If the guest volunteers a name you may use it naturally; never prompt for name or age.
- Use tools to fetch real menu data before answering menu questions.
- Before adding items, call get_current_order if unsure what is already on the ticket. Do NOT call place_order again for a dish that is already on the order unless the guest clearly asks to add more (e.g. "another", "one more", "add a second", "extra portion"). If they only mention a dish again in passing (e.g. "I love the tiramisu") without asking to add more, acknowledge it — do not place_order.
- When the guest confirms NEW items to add, call place_order with exact dish names from the menu (list of strings) and table_number if they gave one. Repeating a name in the same list increases quantity for that dish (e.g. two entries "Tiramisu" means two). The tool response includes order_id and line_id per line.
- To change quantity or remove a line, use modify_order with action set_quantity or remove_item; pass line_id from the last order snapshot or dish_name matching the item. Status: draft = taking order, then confirmed, then completed — use modify_order action set_status with new_status.
- cancel_order is still a stub if asked.
- When the guest asks for the bill/check, call bring_the_bill first (marks bill_requested=true), then call get_bill_breakdown to fetch deterministic line items and totals.
- For bill responses, never invent numbers. Use get_bill_breakdown tool values exactly (items, subtotal, service charge, GST, grand total).
- After sharing bill info, ask one short follow-up question for feedback about the food/experience.
- Rating persistence already uses review_and_feedback. Ask for rating once per order: check get_proactive_checklist; if rating_asked_at is missing, ask now and call mark_rating_asked.
- For proactive flow, avoid repeating the same category suggestions. Use get_proactive_checklist before suggesting sweets/drinks/other add-ons; when you suggest one, call update_proactive_checklist with the matching flag.
- For review_and_feedback: NEVER invent a rating or paraphrase praise. overall_rating MUST be the exact number the guest stated (1–5). feedback_text MUST be their actual words about the meal (or a faithful short quote), including complaints — do not substitute generic positive text. If they did not give a rating or comment yet, omit those fields or only set bill_requested.
- After get_menu_items (or any recommendation flow), the guest sees full dish cards on screen (name, price, description). **Do not** describe every item or read prices aloud. Give a brief spoken reaction (~1–2 sentences, roughly **30 words or fewer** for the whole verbal part of your reply) and offer to help them choose or add to the order.
- The full text of your reply may still be longer for on-screen reading if needed, but assume **only the first ~30 words are spoken aloud** — put the most important spoken message first, then any extra detail is display-only.
- No markdown or bullet lists in replies; plain sentences.
- If the Guest profile below includes a name, you may greet them by name once; never ask for name or age."""

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_order",
            "description": (
                "Return the guest's current order for this session (line items, quantities, subtotal, status). "
                "Call before place_order when the guest might already have items on the ticket, to avoid duplicate orders."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Optional Mongo order id; omit to use the active session order.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_menu_items",
            "description": (
                "Semantically search the menu for dishes relevant to the guest's request. "
                "Pass the guest's query as-is — e.g. 'spicy vegetarian mains', 'light dessert', "
                "'something with chicken'. Returns the 9 most relevant available dishes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of what the guest is looking for.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_item_availability",
            "description": "Check whether a specific menu item is available today.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Name of the menu item to check.",
                    }
                },
                "required": ["item_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "Place a food order for the guest.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of item names to order.",
                    },
                    "table_number": {
                        "type": "integer",
                        "description": "Table number for the order.",
                    },
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_order",
            "description": (
                "Modify the guest's order in MongoDB. action set_quantity: set quantity (needs line_id or dish_name, quantity>=1). "
                "action remove_item: delete a line (line_id or dish_name). "
                "action set_status: new_status draft (taking order), confirmed, or completed. "
                "order_id optional — defaults to the active session order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set_quantity", "remove_item", "set_status"],
                        "description": "set_quantity | remove_item | set_status",
                    },
                    "order_id": {
                        "type": "string",
                        "description": "Mongo order id; omit to use the current session order.",
                    },
                    "line_id": {
                        "type": "string",
                        "description": "line_id from place_order / modify_order response (for set_quantity or remove_item).",
                    },
                    "dish_name": {
                        "type": "string",
                        "description": "Menu item name if line_id unknown (matches one line).",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Required for set_quantity.",
                    },
                    "new_status": {
                        "type": "string",
                        "enum": ["draft", "confirmed", "completed", "taking_order"],
                        "description": "Required for set_status. draft = still taking order.",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bring_the_bill",
            "description": (
                "Mark that the guest asked for the bill/check on the active order. "
                "Sets bill_requested=true and billing timestamp in Mongo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Optional Mongo order id; omit to use current session order.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_bill_breakdown",
            "description": (
                "Return deterministic bill fields for the active order: item lines with quantity, unit_price, line_total, "
                "plus subtotal, service_charge_amount, gst_amount, and grand_total."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Optional Mongo order id; omit to use current session order.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_proactive_checklist",
            "description": (
                "Return proactive follow-up state for the order: checklist flags (sweets/drinks/others) "
                "and rating_asked_at/rating_received_at timestamps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Optional Mongo order id; omit to use current session order.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_proactive_checklist",
            "description": (
                "Mark proactive suggestion categories as completed when they are actually suggested."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sweets_suggested": {
                        "type": "boolean",
                        "description": "Set true when sweets/dessert suggestion was made.",
                    },
                    "drinks_suggested": {
                        "type": "boolean",
                        "description": "Set true when drinks/beverage suggestion was made.",
                    },
                    "others_suggested": {
                        "type": "boolean",
                        "description": "Set true when other add-on/upsell suggestion was made.",
                    },
                    "order_id": {
                        "type": "string",
                        "description": "Optional Mongo order id; omit to use current session order.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_rating_asked",
            "description": (
                "Record that the rating/feedback question has already been asked for this order "
                "so it is not repeated again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Optional Mongo order id; omit to use current session order.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "review_and_feedback",
            "description": (
                "Record bill request and/or post-meal feedback on the guest's order document. "
                "CRITICAL: overall_rating MUST be the integer the guest actually said (1–5). "
                "feedback_text MUST be their real words (or exact quote), including negative feedback — never fabricate. "
                "Use when they ask for the bill (bill_requested=true) and/or give ratings or comments. "
                "Merges into the same Mongo order (review + billing fields)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bill_requested": {
                        "type": "boolean",
                        "description": "True when the guest asks for the check / bill.",
                    },
                    "overall_rating": {
                        "type": "integer",
                        "description": "1–5 only. Must match what the guest said (e.g. 'three stars' → 3). Do not default to 5.",
                    },
                    "feedback_text": {
                        "type": "string",
                        "description": "Verbatim or faithful summary of what the guest said about the meal. Include complaints if any.",
                    },
                    "item_feedback": {
                        "type": "array",
                        "description": "Optional per-item ratings.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "line_id": {"type": "string"},
                                "dish_name": {"type": "string"},
                                "rating": {"type": "integer", "description": "1–5"},
                                "comment": {"type": "string"},
                            },
                        },
                    },
                    "order_id": {
                        "type": "string",
                        "description": "Mongo order id; omit for the current session order.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel an existing order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                },
                "required": ["order_id"],
            },
        },
    },
]


def _extract_tts_segments(buffer: str) -> Tuple[List[str], str]:
    """Pull leading fragments ending at sentence punctuation; leave remainder in buffer."""
    out: List[str] = []
    while buffer:
        end = -1
        for punct in [".", "?", "!", "।"]:
            idx = buffer.find(punct)
            if idx != -1 and (end == -1 or idx < end):
                end = idx
        if end == -1:
            break
        seg = buffer[: end + 1].strip()
        buffer = buffer[end + 1 :].lstrip()
        if seg:
            out.append(seg)
    return out, buffer


def _maybe_force_flush(buffer: str, max_chars: int = 140) -> Tuple[Optional[str], str]:
    """If buffer is very long, flush at last space to unblock TTS overlap."""
    if len(buffer) < max_chars:
        return None, buffer
    cut = buffer.rfind(" ")
    if cut < 40:
        return None, buffer
    chunk = buffer[:cut].strip()
    rest = buffer[cut:].lstrip()
    return chunk if chunk else None, rest


class LLMService:
    _customer_cache: Dict[tuple, dict] = {}

    @classmethod
    def clear_customer_cache(cls, customer_id: int, hotel_id: int) -> None:
        cls._customer_cache.pop((customer_id, hotel_id), None)

    def __init__(self) -> None:
        if not config.AZURE_OPENAI_ENDPOINT or not config.AZURE_OPENAI_API_KEY:
            raise ValueError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env")
        if not config.AZURE_OPENAI_DEPLOYMENT_NAME:
            raise ValueError("Set AZURE_OPENAI_DEPLOYMENT_NAME in .env")

        self._client = AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
        self._aclient = AsyncAzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
        self._executor = ToolExecutor()

    def _get_customer_profile(self, customer_id: int, hotel_id: int) -> dict:
        if customer_id <= 0:
            return {"found": False}
        key = (customer_id, hotel_id)
        if key not in LLMService._customer_cache:
            raw = self._executor.run(
                "find_user_preference",
                {"customer_id": customer_id, "hotel_id": hotel_id},
            )
            LLMService._customer_cache[key] = json.loads(raw)
        return LLMService._customer_cache[key]

    def resolve_tools(
        self,
        messages: list,
        on_first_tool_call: Optional[Callable[[], None]] = None,
        timing_events: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[list, Optional[str], List[Dict[str, Any]], bool, List[str]]:
        """First LLM phase: tools only, non-streaming.

        Returns (messages, direct_answer_or_None, menu_recommendations, tools_called, tools_used_names).
        """
        menu_recommendations: List[Dict[str, Any]] = []
        tools_used: List[str] = []
        tools_called = False
        call_num = 0
        while True:
            call_num += 1
            t = time.perf_counter()
            response = self._client.chat.completions.create(
                model=config.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
                **chat_temperature_kwargs(0.2),
                **gpt5_model_kwargs(),
                **completion_limit_kwargs(tool_round_limit()),
            )
            llm_dt = time.perf_counter() - t
            print(f"  [LLM] call #{call_num} (tools): {llm_dt:.2f}s")
            msg = response.choices[0].message
            if not msg.tool_calls:
                answer = (msg.content or "").strip()
                usage = getattr(response, "usage", None)
                in_tok = getattr(usage, "prompt_tokens", None) if usage else None
                out_tok = getattr(usage, "completion_tokens", None) if usage else None
                total_tok = getattr(usage, "total_tokens", None) if usage else None
                if tools_called:
                    mode = detect_response_mode(tools_used, len(menu_recommendations))
                    if mode != "none":
                        _, _, ok_tag = parse_speak_show(answer)
                        if ok_tag:
                            print(
                                f"  [LLM] tools done — tagged direct ({len(answer)} chars)"
                            )
                        else:
                            print(
                                f"  [LLM] tools done — defer format stream"
                                f" (mode={mode}, skipped {len(answer)} char prose)"
                            )
                            answer = None
                    else:
                        print(f"  [LLM] no tool calls — direct ({len(answer)} chars)")
                else:
                    print(f"  [LLM] no tool calls — direct ({len(answer)} chars)")
                if timing_events is not None:
                    timing_events.append(
                        {
                            "event": "llm_tools_call",
                            "call": call_num,
                            "seconds": round(llm_dt, 4),
                            "has_tool_calls": False,
                            "answer_chars": len(answer or ""),
                            "prompt_tokens": in_tok,
                            "completion_tokens": out_tok,
                            "total_tokens": total_tok,
                            "deferred_format_stream": tools_called
                            and answer is None
                            and detect_response_mode(
                                tools_used, len(menu_recommendations)
                            )
                            != "none",
                        }
                    )
                return messages, answer if answer else None, menu_recommendations, tools_called, tools_used
            tool_names = [tc.function.name for tc in msg.tool_calls]
            print(f"  [LLM] tool calls: {tool_names}")
            if timing_events is not None:
                usage = getattr(response, "usage", None)
                timing_events.append(
                    {
                        "event": "llm_tools_call",
                        "call": call_num,
                        "seconds": round(llm_dt, 4),
                        "has_tool_calls": True,
                        "tools": tool_names,
                        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
                    }
                )
            if not tools_called:
                tools_called = True
                if on_first_tool_call is not None:
                    try:
                        on_first_tool_call()
                    except Exception:
                        pass
            messages.append(msg)
            for tc in msg.tool_calls:
                tools_used.append(tc.function.name)
                args = json.loads(tc.function.arguments or "{}")
                t_tool = time.perf_counter()
                result = self._executor.run(tc.function.name, args)
                tool_dt = time.perf_counter() - t_tool
                print(f"  [TOOL] {tc.function.name}({args}): {tool_dt:.2f}s")
                if timing_events is not None:
                    timing_events.append(
                        {
                            "event": "tool_exec",
                            "name": tc.function.name,
                            "seconds": round(tool_dt, 4),
                        }
                    )
                if tc.function.name == "get_menu_items":
                    menu_recommendations = _recommendations_from_get_menu_json(result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            # Menu search is one-shot; skip the slow post-tool prose call entirely.
            if detect_response_mode(tools_used, len(menu_recommendations)) == "recommendations":
                print(
                    "  [LLM] tools done — skip prose call; "
                    "format stream next (mode=recommendations)"
                )
                if timing_events is not None:
                    timing_events.append(
                        {
                            "event": "tools_phase_complete",
                            "mode": "recommendations",
                            "tools": list(tools_used),
                            "deferred_format_stream": True,
                        }
                    )
                return messages, None, menu_recommendations, tools_called, tools_used

    async def astream_tts_segments_after_tools(
        self,
        messages: list,
        direct_answer: Optional[str],
        timing_events: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[str]:
        """
        Second LLM phase: if direct_answer set (model finished in tool phase), segment for TTS.
        Otherwise stream completion and yield TTS-sized segments as tokens arrive.
        """
        if direct_answer is not None:
            t0 = time.perf_counter()
            nseg = 0
            for s in self._segments_from_plain(direct_answer):
                nseg += 1
                yield s
            if timing_events is not None:
                timing_events.append(
                    {
                        "event": "llm_stream",
                        "mode": "direct",
                        "seconds": round(time.perf_counter() - t0, 4),
                        "segments_yielded": nseg,
                    }
                )
            return

        t = time.perf_counter()
        stream_usage = None
        stream = await self._aclient.chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **chat_temperature_kwargs(0.7),
            **gpt5_model_kwargs(),
            **completion_limit_kwargs(plain_stream_limit()),
        )
        first = True
        buffer = ""
        async for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                stream_usage = chunk.usage
            if first:
                first_dt = time.perf_counter() - t
                print(f"  [LLM] stream (response) first chunk: {first_dt:.2f}s")
                if timing_events is not None:
                    timing_events.append(
                        {
                            "event": "llm_stream_first_chunk",
                            "seconds": round(first_dt, 4),
                        }
                    )
                first = False
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            buffer += delta
            segs, buffer = _extract_tts_segments(buffer)
            for s in segs:
                yield s
            forced, buffer = _maybe_force_flush(buffer)
            if forced:
                yield forced
        if buffer.strip():
            yield buffer.strip()
        if timing_events is not None:
            timing_events.append(
                {
                    "event": "llm_stream",
                    "mode": "stream",
                    "seconds": round(time.perf_counter() - t, 4),
                    "prompt_tokens": getattr(stream_usage, "prompt_tokens", None),
                    "completion_tokens": getattr(stream_usage, "completion_tokens", None),
                    "total_tokens": getattr(stream_usage, "total_tokens", None),
                }
            )

    async def astream_raw_text_after_tools(
        self,
        messages: list,
        direct_answer: Optional[str],
        timing_events: Optional[List[Dict[str, Any]]] = None,
        *,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Second LLM phase without sentence splitting — raw delta.content chunks only.
        Use when the model must emit [SPEAK]/[SHOW] tags without breaking them across yields.
        """
        if direct_answer is not None:
            t0 = time.perf_counter()
            yield direct_answer
            if timing_events is not None:
                timing_events.append(
                    {
                        "event": "llm_stream",
                        "mode": "direct_raw",
                        "seconds": round(time.perf_counter() - t0, 4),
                        "chunks_yielded": 1,
                    }
                )
            return

        limit = max_tokens if max_tokens is not None else format_stream_limit()
        t = time.perf_counter()
        print(f"  [LLM] stream (raw format) max_tokens={limit}")
        stream_usage = None
        stream = await self._aclient.chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **chat_temperature_kwargs(0.2),
            **gpt5_model_kwargs(),
            **completion_limit_kwargs(limit),
        )
        first = True
        n_chunks = 0
        async for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                stream_usage = chunk.usage
            if first:
                first_dt = time.perf_counter() - t
                print(f"  [LLM] stream (raw format) first chunk: {first_dt:.2f}s")
                if timing_events is not None:
                    timing_events.append(
                        {
                            "event": "llm_stream_first_chunk",
                            "seconds": round(first_dt, 4),
                            "mode": "raw",
                        }
                    )
                first = False
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                n_chunks += 1
                yield delta
        if timing_events is not None:
            timing_events.append(
                {
                    "event": "llm_stream",
                    "mode": "raw",
                    "seconds": round(time.perf_counter() - t, 4),
                    "chunks_yielded": n_chunks,
                    "prompt_tokens": getattr(stream_usage, "prompt_tokens", None),
                    "completion_tokens": getattr(stream_usage, "completion_tokens", None),
                    "total_tokens": getattr(stream_usage, "total_tokens", None),
                }
            )

    def _segments_from_plain(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        segs, rest = _extract_tts_segments(text)
        for s in segs:
            yield s
        buf = rest
        while buf:
            forced, buf = _maybe_force_flush(buf)
            if forced:
                yield forced
            else:
                break
        if buf.strip():
            yield buf.strip()

    def _build_messages(self, user_query: str, history: list, context: List[Dict[str, Any]]) -> list:
        sess = get_session()
        cid = int(sess.customer_id) if sess is not None else int(config.DEFAULT_CUSTOMER_ID)
        hid = int(sess.hotel_id) if sess is not None else int(config.DEFAULT_HOTEL_ID)
        profile = self._get_customer_profile(cid, hid)
        profile_text = ""
        if profile.get("found"):
            p = profile
            nm_stripped = (p.get("name") or "").strip()
            ag = p.get("age")
            if not nm_stripped:
                profile_text = (
                    "\n\nGuest profile — returning guest (face recognized). Do not ask for name or age. "
                    "Focus on menu and ordering."
                )
            else:
                profile_text = (
                    f"\n\nGuest profile — name: {nm_stripped} (use for greeting only; do not ask for age). "
                    f"dietary: {p.get('dietary_preferences')}, "
                    f"allergens: {p.get('allergens')}, "
                    f"favourite dishes: {p.get('favorite_dishes')}, "
                    f"visit count: {p.get('visit_count')}, "
                    f"notes: {p.get('notes')}."
                )
        elif cid <= 0:
            profile_text = (
                "\n\n**Guest identity:** Anonymous table session. Do not ask for name or age. "
                "Help with menu, ordering, and bill only."
            )

        lines = []
        for item in context:
            if isinstance(item, dict) and "text" in item:
                lines.append(f"- {item['text']}")
            else:
                lines.append(f"- {item}")
        context_text = "\n".join(lines) if lines else "(no additional context)"

        lang = (getattr(sess, "agent_language", None) or "en") if sess else "en"
        if str(lang).lower() not in ("en", "hinglish"):
            lang = "en"
        if str(lang).lower() == "hinglish":
            lang_line = (
                "\n\n**Language (mandatory):** This restaurant uses **Hinglish** for this device — "
                "natural Hindi–English mix (urban Indian restaurant style). Mix both in the same "
                "sentences when it sounds natural; Roman script is fine for Hindi words unless "
                "Devanagari reads better for a specific word. Keep dish names in English/Latin when "
                "usual. Same tool-calling rules. Put the key spoken line in the first ~30 words "
                "(see spoken-length rule above)."
            )
        else:
            lang_line = "\n\n**Language (mandatory):** Reply in English only."

        messages = [
            {
                "role": "system",
                "content": (
                    f"{_SYSTEM_PROMPT}{profile_text}{lang_line}\n\n"
                    f"Knowledge context:\n{context_text}"
                ),
            }
        ]
        messages.extend(history[-10:])
        messages.append({"role": "user", "content": user_query})
        return messages
