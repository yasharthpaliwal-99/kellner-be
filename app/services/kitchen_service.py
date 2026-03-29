"""Read-only kitchen dashboard data from Mongo orders by hotel_id."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from bson import ObjectId

from app.db.mongo import get_orders_collection


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
