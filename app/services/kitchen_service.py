"""Kitchen dashboard: read orders; table writes requests + per-dish spice on order docs."""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from bson import ObjectId

from app.db.mongo import get_orders_collection

SpiceLevel = Literal["mild", "low", "medium", "high"]
SPICE_LEVELS: Tuple[str, ...] = ("mild", "low", "medium", "high")

# Kitchen-written progress on line_items[].dish_status (table polls GET /kitchen same as spice).
DishStatus = Literal["queued", "preparing", "cooking", "arriving", "ready", "served"]
DISH_STATUSES: Tuple[str, ...] = ("queued", "preparing", "cooking", "arriving", "ready", "served")


def _utc_day_bounds(date_str: str) -> Tuple[datetime, datetime]:
    """Inclusive start, exclusive end for created_at filter (UTC)."""
    d = datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end


def _parse_date_filter(
    date_param: Optional[str],
) -> Tuple[Optional[Tuple[datetime, datetime]], Optional[str]]:
    """
    Returns ((start, end) UTC bounds for created_at, error). None if no date filter.
    """
    if date_param is None or not str(date_param).strip():
        return None, None
    s = str(date_param).strip()
    try:
        return (_utc_day_bounds(s), None)
    except ValueError:
        return None, "date must be a valid YYYY-MM-DD (filters created_at to that UTC calendar day)."


def hotel_id_query_variants(hotel_id_param: str) -> List[Any]:
    """
    Build values for $in so documents match whether hotel_id was stored as int or string.
    """
    s = (hotel_id_param or "").strip()
    if not s:
        return []
    variants: List[Any] = [s]
    try:
        n = int(s, 10)
        variants.append(n)
        if str(n) != s:
            variants.append(str(n))
    except ValueError:
        try:
            n = int(float(s))
            variants.append(n)
            variants.append(str(n))
        except ValueError:
            pass
    seen: set = set()
    out: List[Any] = []
    for v in variants:
        key = (type(v).__name__, repr(v))
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, float):
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return obj
    return str(obj)


def serialize_order(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Same fields as Mongo document; JSON-safe (ObjectId/datetime serialized)."""
    return _json_safe(dict(doc))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _oid(order_id: str) -> Optional[ObjectId]:
    try:
        return ObjectId(str(order_id).strip())
    except Exception:
        return None


def _load_order(col: Any, hotel_id_param: str, order_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    oid = _oid(order_id)
    if oid is None:
        return None, "Invalid order_id."
    variants = hotel_id_query_variants(hotel_id_param)
    if not variants:
        return None, "hotel_id is required."
    doc = col.find_one({"_id": oid, "hotel_id": {"$in": variants}})
    if not doc:
        return None, "Order not found for this hotel_id."
    return doc, None


def resolve_order_id_by_table(hotel_id_param: str, table_number: int) -> Tuple[Optional[str], Optional[str]]:
    """Latest draft order for this hotel and table (by `updated_at`)."""
    col = get_orders_collection()
    variants = hotel_id_query_variants(hotel_id_param)
    if not variants:
        return None, "hotel_id is required."
    if col is None:
        return None, "MongoDB is not configured (set MONGODB_URI in .env)."
    doc = col.find_one(
        {
            "hotel_id": {"$in": variants},
            "status": "draft",
            "$or": [
                {"table_number": table_number},
                {"table_number": str(table_number)},
            ],
        },
        sort=[("updated_at", -1)],
    )
    if not doc:
        return None, "No draft order for this hotel and table_number."
    return str(doc["_id"]), None


def resolve_order_id_by_table_kitchen(hotel_id_param: str, table_number: int) -> Tuple[Optional[str], Optional[str]]:
    """Latest draft or confirmed order for this hotel + table (kitchen updates prep after order may be confirmed)."""
    col = get_orders_collection()
    variants = hotel_id_query_variants(hotel_id_param)
    if not variants:
        return None, "hotel_id is required."
    if col is None:
        return None, "MongoDB is not configured (set MONGODB_URI in .env)."
    doc = col.find_one(
        {
            "hotel_id": {"$in": variants},
            "status": {"$in": ["draft", "confirmed"]},
            "$or": [
                {"table_number": table_number},
                {"table_number": str(table_number)},
            ],
        },
        sort=[("updated_at", -1)],
    )
    if not doc:
        return None, "No active order for this hotel and table_number."
    return str(doc["_id"]), None


def _normalize_dish_name(name: Any) -> str:
    s = str(name or "").strip().lower()
    return " ".join(s.split()) if s else ""


def update_order_table_ops(
    hotel_id_param: str,
    table_number: int,
    *,
    request_text: Optional[str] = None,
    dish_name: Optional[str] = None,
    spice_level: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Table UI: push service note to `requests[]` and/or set `spice_level` on lines matching dish_name.
    Kitchen reads both via GET /kitchen.
    """
    text = (request_text or "").strip()
    spice = (spice_level or "").strip().lower() if spice_level is not None else ""
    dish_key = _normalize_dish_name(dish_name)

    if not text and not spice:
        return None, "Provide request_text and/or dish_name with spice_level."
    if spice and not dish_key:
        return None, "dish_name is required when setting spice_level."
    if spice and spice not in SPICE_LEVELS:
        return None, f"spice_level must be one of: {', '.join(SPICE_LEVELS)}."

    col = get_orders_collection()
    if col is None:
        return None, "MongoDB is not configured (set MONGODB_URI in .env)."

    hid = (hotel_id_param or "").strip()
    if not hid:
        return None, "hotel_id is required."

    oid_str, err = resolve_order_id_by_table(hid, int(table_number))
    if err or not oid_str:
        return None, err or "Could not resolve order."

    doc, err = _load_order(col, hid, oid_str)
    if err or not doc:
        return None, err

    now = _utcnow()
    set_fields: Dict[str, Any] = {"updated_at": now}
    push: Dict[str, Any] = {}

    if text:
        push["requests"] = {"id": str(uuid.uuid4()), "text": text, "created_at": now}

    if spice:
        lines: List[Dict[str, Any]] = [dict(x) for x in (doc.get("line_items") or [])]
        updated = 0
        for i, li in enumerate(lines):
            if _normalize_dish_name(li.get("name")) == dish_key:
                lines[i] = {**lines[i], "spice_level": spice}
                updated += 1
        if updated == 0:
            return None, "dish_name not found on this order."
        set_fields["line_items"] = lines

    variants = hotel_id_query_variants(hid)
    update: Dict[str, Any] = {"$set": set_fields, "$inc": {"version": 1}}
    if push:
        update["$push"] = push

    res = col.update_one({"_id": doc["_id"], "hotel_id": {"$in": variants}}, update)
    if res.matched_count == 0:
        return None, "Order not found for this hotel_id."

    doc2, err2 = _load_order(col, hid, oid_str)
    if err2 or not doc2:
        return None, err2
    return serialize_order(doc2), None


def update_kitchen_line_dish_status(
    hotel_id_param: str,
    table_number: int,
    dish_name: str,
    dish_status: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Kitchen: set dish_status on all line_items rows matching dish_name (same name-normalization as spice)."""
    st = (dish_status or "").strip().lower()
    if st not in DISH_STATUSES:
        return None, f"dish_status must be one of: {', '.join(DISH_STATUSES)}."
    key = _normalize_dish_name(dish_name)
    if not key:
        return None, "dish_name is required."

    col = get_orders_collection()
    if col is None:
        return None, "MongoDB is not configured (set MONGODB_URI in .env)."
    hid = (hotel_id_param or "").strip()
    if not hid:
        return None, "hotel_id is required."

    oid_str, err = resolve_order_id_by_table_kitchen(hid, int(table_number))
    if err or not oid_str:
        return None, err or "Could not resolve order."

    doc, err = _load_order(col, hid, oid_str)
    if err or not doc:
        return None, err

    lines: List[Dict[str, Any]] = [dict(x) for x in (doc.get("line_items") or [])]
    updated = 0
    for i, li in enumerate(lines):
        if _normalize_dish_name(li.get("name")) == key:
            lines[i] = {**lines[i], "dish_status": st}
            updated += 1
    if updated == 0:
        return None, "dish_name not found on this order."

    now = _utcnow()
    variants = hotel_id_query_variants(hid)
    res = col.update_one(
        {"_id": doc["_id"], "hotel_id": {"$in": variants}},
        {"$set": {"line_items": lines, "updated_at": now}, "$inc": {"version": 1}},
    )
    if res.matched_count == 0:
        return None, "Order not found for this hotel_id."

    doc2, err2 = _load_order(col, hid, oid_str)
    if err2 or not doc2:
        return None, err2
    return serialize_order(doc2), None


def fetch_kitchen_snapshot(
    hotel_id_param: str,
    date_param: Optional[str] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns ({ orders, stats, hotel_id, filter }, error_message).
    Orders are exact collection documents (JSON-encoded); filter reduces payload when date is set.
    """
    col = get_orders_collection()
    if col is None:
        return None, "MongoDB is not configured (set MONGODB_URI in .env)."

    variants = hotel_id_query_variants(hotel_id_param)
    if not variants:
        return None, "hotel_id is required."

    bounds, date_err = _parse_date_filter(date_param)
    if date_err:
        return None, date_err

    q: Dict[str, Any] = {"hotel_id": {"$in": variants}}
    filter_meta: Optional[Dict[str, Any]] = None
    if bounds is not None:
        start, end = bounds
        q["created_at"] = {"$gte": start, "$lt": end}
        filter_meta = {
            "date": str(date_param).strip(),
            "field": "created_at",
            "timezone": "UTC",
            "gte": start.isoformat(),
            "lt": end.isoformat(),
        }

    cursor = col.find(q).sort("updated_at", -1)

    orders: List[Dict[str, Any]] = []
    by_status: Dict[str, int] = {"draft": 0, "confirmed": 0, "completed": 0, "other": 0}
    subtotal_sum = 0.0
    bill_requested_count = 0

    for doc in cursor:
        raw = doc.get("status")
        st = (str(raw).strip().lower() if raw is not None else "") or "other"
        if st in by_status:
            by_status[st] += 1
        else:
            by_status["other"] += 1

        subtotal_sum += float(doc.get("subtotal") or 0)

        billing = doc.get("billing") or {}
        if isinstance(billing, dict) and billing.get("bill_requested_at") is not None:
            bill_requested_count += 1

        orders.append(serialize_order(doc))

    stats: Dict[str, Any] = {
        "total_orders": len(orders),
        "by_status": by_status,
        "subtotal_sum": round(subtotal_sum, 2),
        "bill_requested_count": bill_requested_count,
    }

    return (
        {
            "hotel_id": hotel_id_param.strip(),
            "filter": filter_meta,
            "orders": orders,
            "stats": stats,
        },
        None,
    )
