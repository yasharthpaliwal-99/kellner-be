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
    sess = {
        "session_id": sid,
        "hotel_id": hotel.get("hotel_id"),
        "role": role,
        "table_number": table_number,
        "device_id": device_id,
        "revoked": False,
        "created_at": now,
        "last_seen_at": now,
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
    sessions.update_one({"_id": doc["_id"]}, {"$set": {"last_seen_at": _utcnow()}})
    return doc, None
