"""Postgres menu_items read/update for staff (availability + spotlight flags)."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, NamedTuple, Tuple

from app.db.pool import get_pool


class MenuItemUpdate(NamedTuple):
    dish_id: int
    available: bool
    chef_special: bool
    todays_special: bool
    must_try: bool


def _menu_row_dict(
    row: tuple,
    *,
    with_spotlights: bool = False,
    with_image: bool = True,
) -> Dict[str, Any]:
    price = row[2]
    if price is not None and isinstance(price, Decimal):
        price = float(price)
    out: Dict[str, Any] = {
        "dish_id": int(row[0]),
        "name": row[1],
        "price": price,
        "available": bool(row[3]),
    }
    idx = 4
    if with_image:
        out["image"] = row[idx]
        idx += 1
    if with_spotlights:
        out["chef_special"] = bool(row[idx])
        out["todays_special"] = bool(row[idx + 1])
        out["must_try"] = bool(row[idx + 2])
    return out


def _same_hotel(session_hotel: Any, request_hotel: Any) -> bool:
    a = str(session_hotel).strip()
    b = str(request_hotel).strip()
    if a == b:
        return True
    try:
        return int(a) == int(b)
    except ValueError:
        return False


def fetch_menu_rows(hotel_id: int) -> List[Dict[str, Any]]:
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT dish_id, name, price, available, image,
                       chef_special, todays_special, must_try
                FROM menu_items
                WHERE hotel_id = %s
                ORDER BY dish_id ASC
                """,
                (hotel_id,),
            )
            rows = cur.fetchall()
        return [_menu_row_dict(r, with_spotlights=True, with_image=True) for r in rows]
    finally:
        pool.putconn(conn)


def apply_menu_item_updates(
    hotel_id: int, updates: List[MenuItemUpdate]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Update availability and spotlight flags for each dish_id scoped to hotel_id.
    Returns (updated_rows, failures) where failures are {dish_id, reason}.
    """
    pool = get_pool()
    conn = pool.getconn()
    updated: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            for item in updates:
                cur.execute(
                    """
                    UPDATE menu_items
                    SET available = %s,
                        chef_special = %s,
                        todays_special = %s,
                        must_try = %s
                    WHERE dish_id = %s AND hotel_id = %s
                    RETURNING dish_id, name, price, available, image,
                              chef_special, todays_special, must_try
                    """,
                    (
                        item.available,
                        item.chef_special,
                        item.todays_special,
                        item.must_try,
                        item.dish_id,
                        hotel_id,
                    ),
                )
                row = cur.fetchone()
                if row is None:
                    failures.append({"dish_id": item.dish_id, "reason": "not_found_or_wrong_hotel"})
                    continue
                updated.append(_menu_row_dict(row, with_spotlights=True, with_image=True))
        conn.commit()
        return updated, failures
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = True
        pool.putconn(conn)


# Back-compat alias
apply_availability_updates = apply_menu_item_updates


_SPOTLIGHT_RAILS = (
    ("chef_special", "Chef's Special"),
    ("todays_special", "Today's Special"),
    ("must_try", "Must Try"),
)


def fetch_spotlight_rails(hotel_id: int, *, limit_per_rail: int = 12) -> List[Dict[str, Any]]:
    """Available dishes grouped for guest home screen (read-only)."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT dish_id, name, price, available, image,
                       chef_special, todays_special, must_try
                FROM menu_items
                WHERE hotel_id = %s
                  AND available = true
                  AND (chef_special OR todays_special OR must_try)
                ORDER BY dish_id ASC
                """,
                (hotel_id,),
            )
            rows = cur.fetchall()
    finally:
        pool.putconn(conn)

    buckets: Dict[str, List[Dict[str, Any]]] = {key: [] for key, _ in _SPOTLIGHT_RAILS}
    for r in rows:
        item = _menu_row_dict(r, with_spotlights=True, with_image=True)
        if item["chef_special"]:
            buckets["chef_special"].append(item)
        if item["todays_special"]:
            buckets["todays_special"].append(item)
        if item["must_try"]:
            buckets["must_try"].append(item)

    rails: List[Dict[str, Any]] = []
    for rail_id, title in _SPOTLIGHT_RAILS:
        items = buckets[rail_id][:limit_per_rail]
        rails.append({"id": rail_id, "title": title, "items": items})
    return rails


def update_menu_image_url(hotel_id: int, dish_id: int, image_url: str) -> Dict[str, Any] | None:
    """
    Update menu_items.image for a specific dish scoped to the hotel.
    Returns updated row summary or None if dish not found for hotel.
    """
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE menu_items
                SET image = %s
                WHERE dish_id = %s AND hotel_id = %s
                RETURNING dish_id, name, price, available, image
                """,
                (image_url, dish_id, hotel_id),
            )
            row = cur.fetchone()
        conn.commit()
        if row is None:
            return None
        price = row[2]
        if price is not None and isinstance(price, Decimal):
            price = float(price)
        return {
            "dish_id": int(row[0]),
            "name": row[1],
            "price": price,
            "available": bool(row[3]),
            "image": row[4],
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
