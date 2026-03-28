"""
Tool execution layer.
_tool_get_menu_items and _tool_check_item_availability query the real
PostgreSQL database via the shared connection pool.
Order tools are stubbed until Phase 2.
"""
import json
from typing import Any, Dict

from app.db.pool import get_pool
from app.services.embedding_service import embed


class ToolExecutor:
    def run(self, name: str, arguments: Dict[str, Any]) -> str:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return json.dumps({"error": f"Tool '{name}' is not wired yet."})
        return handler(arguments)

    def _tool_get_menu_items(self, args: Dict[str, Any]) -> str:
        query = args.get("query") or args.get("category") or "food"
        pool = get_pool()
        conn = pool.getconn()
        try:
            query_vector = embed(query)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT name, cuisine_type, description, price, ingredients, allergens
                    FROM menu_items
                    WHERE available = true
                    ORDER BY embedding <=> %s::vector
                    LIMIT 6
                    """,
                    (query_vector,),
                )
                rows = cur.fetchall()
            items = [
                {
                    "name": r[0],
                    "cuisine_type": r[1],
                    "description": r[2],
                    "price": float(r[3]) if r[3] is not None else None,
                    "ingredients": r[4],
                    "allergens": r[5],
                }
                for r in rows
            ]
            return json.dumps({"items": items})
        except Exception as e:
            return json.dumps({"error": str(e)})
        finally:
            pool.putconn(conn)

    def _tool_check_item_availability(self, args: Dict[str, Any]) -> str:
        item_name = args.get("item_name", "")
        pool = get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT name, available
                    FROM menu_items
                    WHERE LOWER(name) = LOWER(%s)
                    LIMIT 1
                    """,
                    (item_name,),
                )
                row = cur.fetchone()
            if row is None:
                return json.dumps({"item": item_name, "available": False, "note": "Item not found on menu."})
            return json.dumps({"item": row[0], "available": row[1]})
        except Exception as e:
            return json.dumps({"error": str(e)})
        finally:
            pool.putconn(conn)

    def _tool_find_user_preference(self, args: Dict[str, Any]) -> str:
        customer_id = args.get("customer_id")
        hotel_id = args.get("hotel_id")
        if not customer_id or not hotel_id:
            return json.dumps({"error": "Both customer_id and hotel_id are required."})

        pool = get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT name, dietary_preferences, allergens, preferred_cuisines,
                           favorite_dishes, disliked_ingredients, visit_count,
                           last_visit, total_spend, notes
                    FROM customers
                    WHERE customer_id = %s AND hotel_id = %s
                    LIMIT 1
                    """,
                    (customer_id, hotel_id),
                )
                row = cur.fetchone()

            if row is None:
                return json.dumps({"found": False, "message": "Customer not found."})

            return json.dumps({
                "found": True,
                "name": row[0],
                "dietary_preferences": row[1],
                "allergens": row[2],
                "preferred_cuisines": row[3],
                "favorite_dishes": row[4],
                "disliked_ingredients": row[5],
                "visit_count": row[6],
                "last_visit": str(row[7]) if row[7] else None,
                "total_spend": float(row[8]) if row[8] else 0,
                "notes": row[9],
            })
        except Exception as e:
            return json.dumps({"error": str(e)})
        finally:
            pool.putconn(conn)

    def _tool_place_order(self, args: Dict[str, Any]) -> str:
        # Phase 2: POST /api/orders
        return json.dumps({"status": "not_implemented", "message": "Order placement coming in Phase 2."})

    def _tool_modify_order(self, args: Dict[str, Any]) -> str:
        # Phase 2: PATCH /api/orders/{order_id}
        return json.dumps({"status": "not_implemented", "message": "Order modification coming in Phase 2."})

    def _tool_cancel_order(self, args: Dict[str, Any]) -> str:
        # Phase 2: DELETE /api/orders/{order_id}
        return json.dumps({"status": "not_implemented", "message": "Order cancellation coming in Phase 2."})
