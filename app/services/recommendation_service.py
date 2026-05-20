"""
Display-only order suggestions from menu_items.best_pair_with (dish_id list).

Triggered after each successful place_order for the dish(es) just added.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from app.db.pool import get_pool

logger = logging.getLogger(__name__)

# Do not run pairing rails when the guest only added these course types.
_SKIP_TRIGGER_COURSE_TYPES = frozenset(
    {
        "beverage",
        "beverages",
        "drink",
        "drinks",
        "dessert",
        "desserts",
        "sweet",
        "sweets",
    }
)

_MAX_SUGGESTIONS = 6


def _parse_pair_ids(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        out: List[int] = []
        for x in raw:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out
    s = str(raw).strip()
    if not s or s.lower() in ("null", "none"):
        return []
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    out = []
    for part in s.split(","):
        p = part.strip().strip('"')
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError:
            continue
    return out


def _norm_course(raw: Any) -> str:
    return (str(raw or "").strip().lower().replace(" ", "_"))


def _card_info(name: str, cuisine: Optional[str], description: Optional[str]) -> str:
    desc = (description or "").strip()
    cuisine_s = (cuisine or "").strip()
    parts = [p for p in [cuisine_s, desc[:280] if desc else ""] if p]
    return " · ".join(parts) if parts else ""


def build_order_suggestions(
    hotel_id: int,
    newly_added_dish_ids: List[int],
    order_line_items: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Pairings for dishes added in this place_order only.
    Excludes dishes already on the ticket. Returns None if nothing to show.
    """
    added = [int(x) for x in newly_added_dish_ids if x is not None]
    if not added:
        return None

    on_ticket: Set[int] = set()
    for li in order_line_items or []:
        try:
            on_ticket.add(int(li["dish_id"]))
        except (KeyError, TypeError, ValueError):
            continue

    try:
        pool = get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT dish_id, name, course_type, best_pair_with
                    FROM menu_items
                    WHERE hotel_id = %s AND dish_id = ANY(%s)
                    """,
                    (hotel_id, added),
                )
                trigger_rows = cur.fetchall()
        finally:
            pool.putconn(conn)
    except Exception as exc:
        logger.warning(
            "order_suggestions: menu lookup failed hotel_id=%s dish_ids=%s: %s",
            hotel_id,
            added,
            exc,
        )
        return None

    if not trigger_rows:
        logger.info("order_suggestions: no menu rows for dish_ids=%s", added)
        return None

    triggered_by: List[Dict[str, Any]] = []
    pair_ids: List[int] = []
    seen_pair: Set[int] = set()

    for dish_id, name, course_type, best_pair_with in trigger_rows:
        if _norm_course(course_type) in _SKIP_TRIGGER_COURSE_TYPES:
            continue
        triggered_by.append({"dish_id": int(dish_id), "name": name})
        for pid in _parse_pair_ids(best_pair_with):
            if pid in on_ticket or pid in seen_pair:
                continue
            seen_pair.add(pid)
            pair_ids.append(pid)

    if not pair_ids or not triggered_by:
        logger.info(
            "order_suggestions: no pairings trigger_dish_ids=%s "
            "(empty best_pair_with, skipped course_type, or all pairs already on ticket)",
            added,
        )
        return None

    try:
        pool = get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT dish_id, name, price, image, description, cuisine_type
                    FROM menu_items
                    WHERE hotel_id = %s
                      AND available = true
                      AND dish_id = ANY(%s)
                    ORDER BY dish_id
                    """,
                    (hotel_id, pair_ids),
                )
                rows = cur.fetchall()
        finally:
            pool.putconn(conn)
    except Exception as exc:
        logger.warning("order_suggestions: paired dish lookup failed: %s", exc)
        return None

    items: List[Dict[str, Any]] = []
    for r in rows:
        did, nm, price, image, description, cuisine = r
        if int(did) in on_ticket:
            continue
        p = price
        if p is not None and isinstance(p, Decimal):
            p = float(p)
        elif p is not None:
            try:
                p = float(p)
            except (TypeError, ValueError):
                p = None
        img = (str(image).strip() if image else None) or None
        items.append(
            {
                "dish_id": int(did),
                "name": nm,
                "price": p,
                "image": img,
                "info": _card_info(nm, cuisine, description),
            }
        )
        if len(items) >= _MAX_SUGGESTIONS:
            break

    if not items:
        return None

    if len(triggered_by) == 1:
        title = f"Pairs well with {triggered_by[0]['name']}"
    else:
        title = "Pairs well with your order"

    return {
        "title": title,
        "triggered_by": triggered_by,
        "items": items,
    }
