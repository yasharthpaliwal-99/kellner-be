"""Lazy MongoDB client for orders (sync / pymongo)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from app.config import config

if TYPE_CHECKING:
    from pymongo.collection import Collection
    from pymongo.mongo_client import MongoClient

_client: Optional["MongoClient"] = None


def get_mongo_client() -> Optional["MongoClient"]:
    global _client
    if not config.MONGODB_URI:
        return None
    if _client is None:
        from pymongo import MongoClient

        _client = MongoClient(
            config.MONGODB_URI,
            serverSelectionTimeoutMS=8000,
            appname="kellner",
        )
    return _client


def get_orders_collection() -> Optional["Collection[Any]"]:
    client = get_mongo_client()
    if client is None:
        return None
    db = client[config.MONGODB_DB_NAME]
    return db[config.MONGODB_ORDERS_COLLECTION]


def ensure_order_indexes() -> None:
    col = get_orders_collection()
    if col is None:
        return
    col.create_index(
        [("hotel_id", 1), ("session_id", 1), ("status", 1)],
        name="hotel_session_status",
    )
    col.create_index([("hotel_id", 1), ("created_at", -1)], name="hotel_created")
