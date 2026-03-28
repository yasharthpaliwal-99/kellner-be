import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from openai import AsyncAzureOpenAI, AzureOpenAI

from app.config import config
from app.services.tool_executor import ToolExecutor

_SYSTEM_PROMPT = """You are a professional restaurant waiter. Be warm, concise, and helpful.
- Use tools to fetch real menu data before answering menu questions.
- If a guest asks to place, modify, or cancel an order, use the appropriate tool.
- Keep replies short enough to speak aloud comfortably — no lists or markdown."""

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_menu_items",
            "description": (
                "Semantically search the menu for dishes relevant to the guest's request. "
                "Pass the guest's query as-is — e.g. 'spicy vegetarian mains', 'light dessert', "
                "'something with chicken'. Returns the 6 most relevant available dishes."
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
            "description": "Modify an existing order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "changes": {
                        "type": "string",
                        "description": "Description of changes to make.",
                    },
                },
                "required": ["order_id", "changes"],
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
        key = (customer_id, hotel_id)
        if key not in LLMService._customer_cache:
            raw = self._executor.run(
                "find_user_preference",
                {"customer_id": customer_id, "hotel_id": hotel_id},
            )
            LLMService._customer_cache[key] = json.loads(raw)
        return LLMService._customer_cache[key]

    def resolve_tools(self, messages: list) -> Tuple[list, Optional[str]]:
        """First LLM phase: tools only, non-streaming. Returns (messages_incl_tool_results, direct_answer_or_None)."""
        call_num = 0
        while True:
            call_num += 1
            t = time.perf_counter()
            response = self._client.chat.completions.create(
                model=config.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
                max_tokens=400,
                temperature=0.7,
            )
            print(f"  [LLM] call #{call_num} (tools): {time.perf_counter() - t:.2f}s")
            msg = response.choices[0].message
            if not msg.tool_calls:
                answer = (msg.content or "").strip()
                print(f"  [LLM] no tool calls — direct ({len(answer)} chars)")
                return messages, answer if answer else None
            tool_names = [tc.function.name for tc in msg.tool_calls]
            print(f"  [LLM] tool calls: {tool_names}")
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                t_tool = time.perf_counter()
                result = self._executor.run(tc.function.name, args)
                print(f"  [TOOL] {tc.function.name}({args}): {time.perf_counter() - t_tool:.2f}s")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

    async def astream_tts_segments_after_tools(
        self,
        messages: list,
        direct_answer: Optional[str],
    ) -> AsyncIterator[str]:
        """
        Second LLM phase: if direct_answer set (model finished in tool phase), segment for TTS.
        Otherwise stream completion and yield TTS-sized segments as tokens arrive.
        """
        if direct_answer is not None:
            for s in self._segments_from_plain(direct_answer):
                yield s
            return

        t = time.perf_counter()
        stream = await self._aclient.chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            max_tokens=400,
            temperature=0.7,
            stream=True,
        )
        first = True
        buffer = ""
        async for chunk in stream:
            if first:
                print(f"  [LLM] stream (response) first chunk: {time.perf_counter() - t:.2f}s")
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

    def stream_response(self, user_query: str, history: list, context: list):
        """Sync iterator for legacy batch clients (e.g. test scripts)."""
        messages = self._build_messages(user_query, history, context)
        try:
            messages, direct = self.resolve_tools(messages)
        except Exception as e:
            if "431" in str(e) or "tool" in str(e).lower():
                messages = self._build_messages(user_query, history, context)
                direct = None
            else:
                raise
        if direct is not None:
            for part in self._segments_from_plain(direct):
                yield part
        else:
            stream = self._client.chat.completions.create(
                model=config.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=messages,
                max_tokens=400,
                temperature=0.7,
                stream=True,
            )
            buffer = ""
            for chunk in stream:
                if not chunk.choices:
                    continue
                buffer += chunk.choices[0].delta.content or ""
                segs, buffer = _extract_tts_segments(buffer)
                for s in segs:
                    yield s
                forced, buffer = _maybe_force_flush(buffer)
                if forced:
                    yield forced
            if buffer.strip():
                yield buffer.strip()

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
        profile = self._get_customer_profile(1, 1)
        profile_text = ""
        if profile.get("found"):
            p = profile
            profile_text = (
                f"\n\nGuest profile — name: {p.get('name')}, "
                f"dietary: {p.get('dietary_preferences')}, "
                f"allergens: {p.get('allergens')}, "
                f"favourite dishes: {p.get('favorite_dishes')}, "
                f"visit count: {p.get('visit_count')}, "
                f"notes: {p.get('notes')}."
            )

        lines = []
        for item in context:
            if isinstance(item, dict) and "text" in item:
                lines.append(f"- {item['text']}")
            else:
                lines.append(f"- {item}")
        context_text = "\n".join(lines) if lines else "(no additional context)"

        messages = [
            {
                "role": "system",
                "content": f"{_SYSTEM_PROMPT}{profile_text}\n\nKnowledge context:\n{context_text}",
            }
        ]
        messages.extend(history[-10:])
        messages.append({"role": "user", "content": user_query})
        return messages
