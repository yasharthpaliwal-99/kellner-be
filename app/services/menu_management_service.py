"""Postgres menu_items read/update for staff (availability)."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Tuple

from app.db.pool import get_pool


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
                SELECT dish_id, name, price, available
                FROM menu_items
                WHERE hotel_id = %s
                ORDER BY dish_id ASC
                """,
                (hotel_id,),
            )
            rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            price = r[2]
            if price is not None and isinstance(price, Decimal):
                price = float(price)
            out.append(
                {
                    "dish_id": int(r[0]),
                    "name": r[1],
                    "price": price,
                    "available": bool(r[3]),
                }
            )
        return out
    finally:
        pool.putconn(conn)


def apply_availability_updates(
    hotel_id: int, updates: List[Tuple[int, bool]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    For each (dish_id, available), update row if it belongs to hotel_id.
    Returns (updated_rows, failures) where failures are {dish_id, reason}.
    """
    pool = get_pool()
    conn = pool.getconn()
    updated: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            for dish_id, available in updates:
                cur.execute(
                    """
                    UPDATE menu_items
                    SET available = %s
                    WHERE dish_id = %s AND hotel_id = %s
                    RETURNING dish_id, name, price, available
                    """,
                    (available, dish_id, hotel_id),
                )
                row = cur.fetchone()
                if row is None:
                    failures.append({"dish_id": dish_id, "reason": "not_found_or_wrong_hotel"})
                    continue
                price = row[2]
                if price is not None and isinstance(price, Decimal):
                    price = float(price)
                updated.append(
                    {
                        "dish_id": int(row[0]),
                        "name": row[1],
                        "price": price,
                        "available": bool(row[3]),
                    }
                )
        conn.commit()
        return updated, failures
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = True
        pool.putconn(conn)
