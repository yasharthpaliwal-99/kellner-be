"""
Draft orders in MongoDB: create or append line items validated against Postgres menu.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from bson import ObjectId
from bson.errors import InvalidId

from app.db.mongo import get_orders_collection
from app.db.pool import get_pool
from app.services.session_context import get_session


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# draft = taking order; confirmed / completed as lifecycle
STATUS_ALIASES = {
    "taking_order": "draft",
    "takingorder": "draft",
    "in_progress": "draft",
}


def _normalize_order_status(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    if s in STATUS_ALIASES:
        s = STATUS_ALIASES[s]
    if s in ("draft", "confirmed", "completed"):
        return s
    return None


def _recalc_subtotal(lines: List[Dict[str, Any]]) -> float:
    return round(sum(float(x.get("line_total") or 0) for x in lines), 2)


def _load_order_for_session(
    col: Any,
    sess: Any,
    order_id: Optional[str],
    *,
    statuses: Optional[List[str]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    hotel_id = sess.hotel_id
    st = statuses if statuses is not None else ["draft", "confirmed"]
    if order_id and str(order_id).strip():
        try:
            oid = ObjectId(str(order_id).strip())
        except (InvalidId, TypeError):
            return None, "Invalid order_id."
        doc = col.find_one({"_id": oid, "hotel_id": hotel_id})
        return (doc, None) if doc else (None, "Order not found.")
    if getattr(sess, "order_id", None):
        try:
            oid = ObjectId(str(sess.order_id).strip())
            doc = col.find_one({"_id": oid, "hotel_id": hotel_id})
            if doc:
                return doc, None
        except (InvalidId, TypeError, ValueError):
            pass
    cur = col.find(
        {
            "hotel_id": hotel_id,
            "session_id": sess.session_id,
            "status": {"$in": st},
        },
    ).sort("updated_at", -1).limit(1)
    try:
        doc = cur.next()
    except StopIteration:
        doc = None
    return (doc, None) if doc else (None, "No active order for this session.")


def _lookup_dishes_by_name(
    hotel_id: int, names: List[str]
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Resolve menu names (case-insensitive trim) to dish_id, name, price.
    Returns (resolved_by_normalized_key -> row, unknown_names).
    """
    pool = get_pool()
    conn = pool.getconn()
    resolved: Dict[str, Dict[str, Any]] = {}
    unknown: List[str] = []
    try:
        with conn.cursor() as cur:
            for raw in names:
                n = (raw or "").strip()
                if not n:
                    continue
                key = n.lower()
                if key in resolved:
                    continue
                cur.execute(
                    """
                    SELECT dish_id, name, price
                    FROM menu_items
                    WHERE hotel_id = %s AND available = true
                      AND LOWER(TRIM(name)) = LOWER(TRIM(%s))
                    LIMIT 1
                    """,
                    (hotel_id, n),
                )
                row = cur.fetchone()
                if row is None:
                    unknown.append(n)
                else:
                    price = float(row[2]) if row[2] is not None else 0.0
                    resolved[key] = {
                        "dish_id": int(row[0]),
                        "name": row[1],
                        "unit_price": price,
                    }
    finally:
        pool.putconn(conn)
    return resolved, unknown


def _aggregate_lines(
    items: List[str], resolved: Dict[str, Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], float]:
    """Merge duplicate dish names into quantities; build line_items + subtotal."""
    counts: Dict[int, Dict[str, Any]] = {}
    order_names: List[str] = []
    for raw in items:
        n = (raw or "").strip()
        if not n:
            continue
        key = n.lower()
        if key not in resolved:
            continue
        r = resolved[key]
        did = r["dish_id"]
        order_names.append(n)
        if did not in counts:
            counts[did] = {**r, "quantity": 0}
        counts[did]["quantity"] += 1

    line_items: List[Dict[str, Any]] = []
    subtotal = 0.0
    for did, row in counts.items():
        qty = row["quantity"]
        unit = row["unit_price"]
        line_total = round(unit * qty, 2)
        subtotal = round(subtotal + line_total, 2)
        line_items.append(
            {
                "line_id": uuid.uuid4().hex[:12],
                "dish_id": did,
                "name": row["name"],
                "quantity": qty,
                "unit_price": unit,
                "line_total": line_total,
            }
        )
    return line_items, subtotal


def get_current_order_snapshot(order_id: Optional[str] = None) -> Dict[str, Any]:
    """Return the session's active order so the model can avoid duplicate place_order calls."""
    col = get_orders_collection()
    if col is None:
        return {"ok": False, "error": "MongoDB is not configured (set MONGODB_URI in .env)."}
    sess = get_session()
    if sess is None:
        return {"ok": False, "error": "No active conversation session."}
    doc, err = _load_order_for_session(
        col,
        sess,
        order_id,
        statuses=["draft", "confirmed", "completed"],
    )
    if err or not doc:
        return {"ok": False, "error": err or "No order found."}
    rev = doc.get("review") or {}
    return {
        "ok": True,
        "order_id": str(doc["_id"]),
        "status": doc.get("status"),
        "table_number": doc.get("table_number"),
        "line_items": doc.get("line_items") or [],
        "subtotal": float(doc.get("subtotal") or 0),
        "currency": doc.get("currency") or "USD",
        "has_saved_review": bool(
            rev.get("overall_rating") is not None
            or (rev.get("feedback_text") or "").strip()
            or (rev.get("item_ratings") or [])
        ),
    }


def place_order(
    items: List[str],
    table_number: Optional[int] = None,
) -> Dict[str, Any]:
    col = get_orders_collection()
    if col is None:
        return {
            "ok": False,
            "error": "MongoDB is not configured (set MONGODB_URI in .env).",
        }

    sess = get_session()
    if sess is None:
        return {"ok": False, "error": "No active conversation session."}

    if not items:
        return {"ok": False, "error": "No items in order."}

    hotel_id = sess.hotel_id
    customer_id = sess.customer_id
    session_id = sess.session_id

    resolved, unknown = _lookup_dishes_by_name(hotel_id, items)
    line_items, added_subtotal = _aggregate_lines(items, resolved)

    if not line_items:
        return {
            "ok": False,
            "error": "No matching menu items for this hotel.",
            "unknown_items": unknown,
        }

    now = _utcnow()
    event = {
        "at": now,
        "type": "place_order",
        "detail": {"items_requested": items, "unknown_items": unknown},
    }

    existing = col.find_one(
        {
            "hotel_id": hotel_id,
            "session_id": session_id,
            "status": "draft",
        }
    )

    if existing is None:
        doc = {
            "hotel_id": hotel_id,
            "customer_id": customer_id,
            "session_id": session_id,
            "status": "draft",
            "table_number": table_number,
            "currency": "USD",
            "line_items": line_items,
            "subtotal": added_subtotal,
            "source": "voice",
            "events": [event],
            "created_at": now,
            "updated_at": now,
            "version": 1,
        }
        ins = col.insert_one(doc)
        oid = str(ins.inserted_id)
        sess.order_id = oid
        return {
            "ok": True,
            "order_id": oid,
            "status": "draft",
            "line_items": line_items,
            "subtotal": added_subtotal,
            "unknown_items": unknown or None,
            "message": "Order created.",
        }

    # Append to existing draft: merge line items by dish_id
    old_lines: List[Dict[str, Any]] = list(existing.get("line_items") or [])
    by_dish: Dict[int, Dict[str, Any]] = {}
    for li in old_lines:
        did = int(li["dish_id"])
        by_dish[did] = dict(li)

    for li in line_items:
        did = int(li["dish_id"])
        if did in by_dish:
            q0 = int(by_dish[did]["quantity"])
            q1 = int(li["quantity"])
            unit = float(by_dish[did]["unit_price"])
            new_q = q0 + q1
            by_dish[did]["quantity"] = new_q
            by_dish[did]["line_total"] = round(unit * new_q, 2)
        else:
            by_dish[did] = li

    merged = list(by_dish.values())
    new_subtotal = round(sum(float(x["line_total"]) for x in merged), 2)
    new_version = int(existing.get("version") or 1) + 1

    update_fields: Dict[str, Any] = {
        "line_items": merged,
        "subtotal": float(new_subtotal),
        "updated_at": now,
        "version": new_version,
    }
    if table_number is not None:
        update_fields["table_number"] = table_number

    col.update_one(
        {"_id": existing["_id"]},
        {
            "$set": update_fields,
            "$push": {"events": event},
        },
    )
    oid = str(existing["_id"])
    sess.order_id = oid
    return {
        "ok": True,
        "order_id": oid,
        "status": "draft",
        "line_items": merged,
        "subtotal": float(new_subtotal),
        "unknown_items": unknown or None,
        "message": "Order updated (items added to existing draft).",
    }


def modify_order(
    action: str,
    order_id: Optional[str] = None,
    line_id: Optional[str] = None,
    dish_name: Optional[str] = None,
    quantity: Optional[int] = None,
    new_status: Optional[str] = None,
) -> Dict[str, Any]:
    col = get_orders_collection()
    if col is None:
        return {"ok": False, "error": "MongoDB is not configured (set MONGODB_URI in .env)."}

    sess = get_session()
    if sess is None:
        return {"ok": False, "error": "No active conversation session."}

    act = (action or "").strip().lower().replace("-", "_")
    if act not in ("set_quantity", "remove_item", "set_status"):
        return {
            "ok": False,
            "error": "action must be set_quantity, remove_item, or set_status.",
        }

    doc, err = _load_order_for_session(col, sess, order_id)
    if err or not doc:
        return {"ok": False, "error": err or "Order not found."}

    if doc.get("status") == "completed":
        return {"ok": False, "error": "This order is completed and cannot be changed."}

    now = _utcnow()
    lines: List[Dict[str, Any]] = [dict(x) for x in (doc.get("line_items") or [])]
    event_detail: Dict[str, Any] = {"action": act}

    def _find_line_index() -> int:
        if line_id and str(line_id).strip():
            lid = str(line_id).strip()
            for i, li in enumerate(lines):
                if str(li.get("line_id", "")) == lid:
                    return i
        if dish_name and str(dish_name).strip():
            dn = str(dish_name).strip().lower()
            for i, li in enumerate(lines):
                if str(li.get("name", "")).strip().lower() == dn:
                    return i
        return -1

    if act == "set_status":
        st = _normalize_order_status(new_status)
        if not st:
            return {
                "ok": False,
                "error": "new_status must be draft (taking order), confirmed, or completed.",
            }
        cur = doc.get("status") or "draft"
        allowed = True
        if st == "draft" and cur in ("confirmed", "completed"):
            allowed = False
        if st == "confirmed" and cur == "completed":
            allowed = False
        if not allowed:
            return {
                "ok": False,
                "error": f"Cannot change status from {cur} to {st}.",
            }
        event_detail["from_status"] = cur
        event_detail["to_status"] = st
        new_version = int(doc.get("version") or 1) + 1
        col.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {"status": st, "updated_at": now, "version": new_version},
                "$push": {
                    "events": {
                        "at": now,
                        "type": "modify_order",
                        "detail": event_detail,
                    }
                },
            },
        )
        sess.order_id = str(doc["_id"])
        return {
            "ok": True,
            "order_id": str(doc["_id"]),
            "status": st,
            "line_items": lines,
            "subtotal": float(doc.get("subtotal") or 0),
            "message": f"Order status set to {st}.",
        }

    # Line edits: not when only status would apply — block if no lines for quantity/remove
    idx = _find_line_index()
    if idx < 0:
        return {
            "ok": False,
            "error": "Line not found. Pass line_id from the order or dish_name matching the menu item.",
        }

    if act == "set_quantity":
        if quantity is None:
            return {"ok": False, "error": "quantity is required for set_quantity."}
        try:
            q = int(quantity)
        except (TypeError, ValueError):
            return {"ok": False, "error": "quantity must be an integer."}
        if q < 1:
            return {"ok": False, "error": "quantity must be at least 1; use remove_item to delete a line."}
        unit = float(lines[idx].get("unit_price") or 0)
        lines[idx]["quantity"] = q
        lines[idx]["line_total"] = round(unit * q, 2)
        event_detail["line_id"] = lines[idx].get("line_id")
        event_detail["quantity"] = q

    elif act == "remove_item":
        removed = lines.pop(idx)
        event_detail["removed"] = removed

    new_subtotal = _recalc_subtotal(lines)
    new_version = int(doc.get("version") or 1) + 1

    col.update_one(
        {"_id": doc["_id"]},
        {
            "$set": {
                "line_items": lines,
                "subtotal": float(new_subtotal),
                "updated_at": now,
                "version": new_version,
            },
            "$push": {
                "events": {
                    "at": now,
                    "type": "modify_order",
                    "detail": event_detail,
                }
            },
        },
    )
    sess.order_id = str(doc["_id"])
    return {
        "ok": True,
        "order_id": str(doc["_id"]),
        "status": doc.get("status", "draft"),
        "line_items": lines,
        "subtotal": float(new_subtotal),
        "message": "Order updated.",
    }


def review_and_feedback(
    bill_requested: bool = False,
    overall_rating: Optional[int] = None,
    feedback_text: Optional[str] = None,
    item_feedback: Optional[List[Dict[str, Any]]] = None,
    order_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Attach billing + review fields to the session order document (draft / confirmed / completed).
    """
    col = get_orders_collection()
    if col is None:
        return {"ok": False, "error": "MongoDB is not configured (set MONGODB_URI in .env)."}

    sess = get_session()
    if sess is None:
        return {"ok": False, "error": "No active conversation session."}

    has_review = overall_rating is not None or (
        feedback_text and str(feedback_text).strip()
    )
    has_items = bool(item_feedback and len(item_feedback) > 0)
    if not bill_requested and not has_review and not has_items:
        return {
            "ok": False,
            "error": "Provide bill_requested, overall_rating, feedback_text, and/or item_feedback.",
        }

    doc, err = _load_order_for_session(
        col,
        sess,
        order_id,
        statuses=["draft", "confirmed", "completed"],
    )
    if err or not doc:
        return {"ok": False, "error": err or "Order not found."}

    now = _utcnow()
    review = dict(doc.get("review") or {})
    if overall_rating is not None:
        try:
            r = int(overall_rating)
        except (TypeError, ValueError):
            return {"ok": False, "error": "overall_rating must be an integer 1–5."}
        if r < 1 or r > 5:
            return {"ok": False, "error": "overall_rating must be between 1 and 5."}
        review["overall_rating"] = r
    if feedback_text and str(feedback_text).strip():
        review["feedback_text"] = str(feedback_text).strip()
    if has_items:
        norm_items: List[Dict[str, Any]] = []
        for row in item_feedback:
            if not isinstance(row, dict):
                continue
            entry: Dict[str, Any] = {}
            if row.get("line_id") is not None:
                entry["line_id"] = str(row["line_id"]).strip()
            if row.get("dish_name") is not None:
                entry["dish_name"] = str(row["dish_name"]).strip()
            if row.get("rating") is not None:
                try:
                    ir = int(row["rating"])
                except (TypeError, ValueError):
                    continue
                if 1 <= ir <= 5:
                    entry["rating"] = ir
            if row.get("comment") is not None:
                c = str(row["comment"]).strip()
                if c:
                    entry["comment"] = c
            if entry.get("rating") is not None and (entry.get("line_id") or entry.get("dish_name")):
                norm_items.append(entry)
        if norm_items:
            review["item_ratings"] = norm_items
        elif item_feedback and len(item_feedback) > 0:
            return {
                "ok": False,
                "error": "Each item_feedback row needs rating 1–5 and dish_name or line_id.",
            }
    if review.get("overall_rating") or review.get("feedback_text") or review.get("item_ratings"):
        if not review.get("submitted_at"):
            review["submitted_at"] = now
        review["updated_at"] = now

    billing = dict(doc.get("billing") or {})
    if bill_requested:
        billing["bill_requested_at"] = now

    new_version = int(doc.get("version") or 1) + 1
    set_fields: Dict[str, Any] = {
        "review": review,
        "billing": billing,
        "updated_at": now,
        "version": new_version,
    }

    event_detail: Dict[str, Any] = {
        "bill_requested": bill_requested,
        "has_overall_rating": overall_rating is not None,
        "has_feedback_text": bool(feedback_text and str(feedback_text).strip()),
        "item_feedback_count": len(review.get("item_ratings") or []),
    }

    col.update_one(
        {"_id": doc["_id"]},
        {
            "$set": set_fields,
            "$push": {
                "events": {
                    "at": now,
                    "type": "review_and_feedback",
                    "detail": event_detail,
                }
            },
        },
    )
    sess.order_id = str(doc["_id"])
    return {
        "ok": True,
        "order_id": str(doc["_id"]),
        "review": review,
        "billing": billing,
        "message": "Saved billing / feedback on this order.",
    }


def bring_the_bill(order_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Mark the session order as bill requested.
    """
    col = get_orders_collection()
    if col is None:
        return {"ok": False, "error": "MongoDB is not configured (set MONGODB_URI in .env)."}

    sess = get_session()
    if sess is None:
        return {"ok": False, "error": "No active conversation session."}

    doc, err = _load_order_for_session(
        col,
        sess,
        order_id,
        statuses=["draft", "confirmed", "completed"],
    )
    if err or not doc:
        return {"ok": False, "error": err or "Order not found."}

    now = _utcnow()
    billing = dict(doc.get("billing") or {})
    billing["bill_requested_at"] = now
    new_version = int(doc.get("version") or 1) + 1

    col.update_one(
        {"_id": doc["_id"]},
        {
            "$set": {
                "bill_requested": True,
                "billing": billing,
                "updated_at": now,
                "version": new_version,
            },
            "$push": {
                "events": {
                    "at": now,
                    "type": "bring_the_bill",
                    "detail": {"bill_requested": True},
                }
            },
        },
    )
    sess.order_id = str(doc["_id"])
    return {
        "ok": True,
        "order_id": str(doc["_id"]),
        "bill_requested": True,
        "billing": billing,
        "message": "Bill has been requested.",
    }
