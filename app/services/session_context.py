"""Per-connection voice session (hotel, customer, session id, Mongo order id)."""
from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConversationSession:
    session_id: str
    hotel_id: int
    customer_id: int
    order_id: Optional[str] = None  # Mongo ObjectId hex string after first place_order
    # Mongo `hotels` doc: agent_language | … → "en" | "hinglish" (legacy hi/hindi → hinglish)
    agent_language: str = "en"
    # Filled by place_order; flushed over WS before "done" (same object — safe across threads).
    pending_order_suggestions: List[Dict[str, Any]] = field(default_factory=list)


_session_var: ContextVar[Optional[ConversationSession]] = ContextVar(
    "kellner_conversation_session", default=None
)


def attach_session(session: ConversationSession) -> Token:
    return _session_var.set(session)


def reset_session(token: Token) -> None:
    _session_var.reset(token)


def get_session() -> Optional[ConversationSession]:
    return _session_var.get()
