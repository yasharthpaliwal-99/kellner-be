"""Long-lived hotel device sessions (login once, reuse session_id)."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from app.db.mongo import get_mongo_client
from app.config import config


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _hotels_collection():
    client = get_mongo_client()
    if client is None:
        return None
    return client[config.MONGODB_DB_NAME]["hotels"]


def _sessions_collection():
    client = get_mongo_client()
    if client is None:
        return None
    return client[config.MONGODB_DB_NAME]["device_sessions"]


def password_hash(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def password_verify(password: str, stored: str) -> bool:
    try:
        algo, salt_hex, digest_hex = stored.split("$", 2)
        if algo != "scrypt":
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except Exception:
        return False
    got = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return hmac.compare_digest(got, expected)


def ensure_device_auth_indexes() -> None:
    hotels = _hotels_collection()
    if hotels is not None:
        hotels.create_index([("hotel_id", 1)], name="hotel_id_unique", unique=True)
    sessions = _sessions_collection()
    if sessions is not None:
        sessions.create_index([("session_id", 1)], name="session_id_unique", unique=True)
        sessions.create_index([("hotel_id", 1), ("role", 1)], name="hotel_role")


def normalize_agent_language(raw: Any) -> str:
    """Hotel DB values → en | hinglish (default en). Legacy hi/hindi map to hinglish."""
    s = str(raw or "").strip().lower().replace("_", "-")
    if not s:
        return "en"
    if s in ("hinglish", "hing", "hi-en", "hindi-english", "hing-lish"):
        return "hinglish"
    if s in ("hi", "hin", "hindi", "hi-in"):
        return "hinglish"
    if s.startswith("hi-"):
        return "hinglish"
    if s in ("en", "eng", "english", "en-in", "en-us", "en-gb"):
        return "en"
    if s.startswith("en-"):
        return "en"
    return "en"


def agent_language_from_hotel_doc(hotel: Optional[Dict[str, Any]]) -> str:
    """
    Same `hotels` document as login credentials (`password_hash`, `hotel_id`, …).
    Add **`agent_language`**: `"en"` or `"hinglish"` there (`hi`/`hindi` are treated as hinglish).
    At login we already load this doc for password check; language is read from it once.
    Older docs: falls back to `language`, `locale`, `preferred_language` if needed.
    """
    if not hotel:
        return "en"
    for key in ("agent_language", "language", "locale", "preferred_language"):
        v = hotel.get(key)
        if v is not None and str(v).strip():
            return normalize_agent_language(v)
    return "en"


def _agent_language_from_hotel_id(hotel_id: Any) -> str:
    """Mongo round-trip only when session doc has no agent_language yet (legacy)."""
    hotels = _hotels_collection()
    if hotels is None:
        return "en"
    variants = _hotel_id_variants(hotel_id)
    if not variants:
        return "en"
    hotel = hotels.find_one({"hotel_id": {"$in": variants}})
    return agent_language_from_hotel_doc(hotel)


def agent_language_for_session(doc: Optional[Dict[str, Any]]) -> str:
    """
    Copy of hotel `agent_language` written at login onto `device_sessions`.
    WebSocket only sends `session_id` (not password), so language lives on the session
    snapshot — no “passing through” handlers; just read this field after validate.
    """
    if not doc:
        return "en"
    return normalize_agent_language(doc.get("agent_language"))


def _hotel_id_variants(hotel_id: Any) -> list[Any]:
    raw = str(hotel_id).strip()
    if not raw:
        return []
    out: list[Any] = [raw]
    try:
        n = int(raw)
        out.extend([n, str(n)])
    except ValueError:
        pass
    seen = set()
    uniq: list[Any] = []
    for v in out:
        k = (type(v).__name__, repr(v))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(v)
    return uniq


def login_hotel_device(
    *,
    hotel_id: Any,
    password: str,
    role: str,
    table_number: Optional[int] = None,
    device_id: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    hotels = _hotels_collection()
    sessions = _sessions_collection()
    if hotels is None or sessions is None:
        return None, "MongoDB is not configured (set MONGODB_URI in .env)."
    variants = _hotel_id_variants(hotel_id)
    if not variants:
        return None, "hotel_id is required."
    hotel = hotels.find_one({"hotel_id": {"$in": variants}})
    if not hotel:
        return None, "Invalid credentials."
    if not password_verify(password, str(hotel.get("password_hash") or "")):
        return None, "Invalid credentials."

    sid = secrets.token_urlsafe(32)
    now = _utcnow()
    # Same `hotel` dict as credential check — read language here, persist on session for WS.
    agent_lang = agent_language_from_hotel_doc(hotel)
    sess = {
        "session_id": sid,
        "hotel_id": hotel.get("hotel_id"),
        "role": role,
        "table_number": table_number,
        "device_id": device_id,
        "revoked": False,
        "created_at": now,
        "last_seen_at": now,
        "agent_language": agent_lang,
    }
    sessions.insert_one(sess)
    return sess, None


def validate_device_session(session_id: str, role: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    sessions = _sessions_collection()
    if sessions is None:
        return None, "MongoDB is not configured (set MONGODB_URI in .env)."
    sid = (session_id or "").strip()
    if not sid:
        return None, "Missing session_id."
    doc = sessions.find_one({"session_id": sid, "revoked": False})
    if not doc:
        return None, "Invalid device session."
    if role and str(doc.get("role") or "") != role:
        return None, "Session role mismatch."
    now = _utcnow()
    to_set: Dict[str, Any] = {"last_seen_at": now}
    raw_lang = doc.get("agent_language")
    if raw_lang is None or not str(raw_lang).strip():
        to_set["agent_language"] = _agent_language_from_hotel_id(doc.get("hotel_id"))
    sessions.update_one({"_id": doc["_id"]}, {"$set": to_set})
    merged = dict(doc)
    merged.update(to_set)
    return merged, None
