"""Per-connection voice session (hotel, customer, session id, Mongo order id)."""
from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConversationSession:
    session_id: str
    hotel_id: int
    customer_id: int
    order_id: Optional[str] = None  # Mongo ObjectId hex string after first place_order
    # Mongo `hotels` doc: agent_language | … → "en" | "hinglish" (legacy hi/hindi → hinglish)
    agent_language: str = "en"


_session_var: ContextVar[Optional[ConversationSession]] = ContextVar(
    "kellner_conversation_session", default=None
)


def attach_session(session: ConversationSession) -> Token:
    return _session_var.set(session)


def reset_session(token: Token) -> None:
    _session_var.reset(token)


def get_session() -> Optional[ConversationSession]:
    return _session_var.get()
