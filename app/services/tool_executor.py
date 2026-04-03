"""
Tool execution layer.
_tool_get_menu_items and _tool_check_item_availability query the real
PostgreSQL database via the shared connection pool.
place_order persists draft orders to MongoDB (session-scoped).
"""
import json
from datetime import date, datetime
from typing import Any, Dict

from app.db.pool import get_pool
from app.services.embedding_service import embed
from app.services.order_service import get_current_order_snapshot
from app.services.order_service import bring_the_bill as persist_bring_the_bill
from app.services.order_service import modify_order as persist_modify_order
from app.services.order_service import place_order as persist_place_order
from app.services.order_service import review_and_feedback as persist_review_and_feedback
from app.services.face_local_service import update_guest_profile
from app.services.session_context import get_session


def _dumps(obj: Any) -> str:
    """JSON for LLM tool results; datetimes from Mongo/review must be serializable."""

    def _default(o: Any) -> Any:
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return str(o)

    return json.dumps(obj, default=_default)


class ToolExecutor:
    def run(self, name: str, arguments: Dict[str, Any]) -> str:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return _dumps({"error": f"Tool '{name}' is not wired yet."})
        return handler(arguments)

    def _tool_get_current_order(self, args: Dict[str, Any]) -> str:
        oid = args.get("order_id")
        if oid is not None:
            oid = str(oid).strip() or None
        result = get_current_order_snapshot(order_id=oid)
        return _dumps(result)

    def _tool_get_menu_items(self, args: Dict[str, Any]) -> str:
        query = args.get("query") or args.get("category") or "food"
        pool = get_pool()
        conn = pool.getconn()
        try:
            query_vector = embed(query)
            sess = get_session()
            with conn.cursor() as cur:
                if sess is not None:
                    cur.execute(
                        """
                        SELECT name, cuisine_type, description, price, ingredients, allergens
                        FROM menu_items
                        WHERE available = true AND hotel_id = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT 6
                        """,
                        (sess.hotel_id, query_vector),
                    )
                else:
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
            return _dumps({"items": items})
        except Exception as e:
            return _dumps({"error": str(e)})
        finally:
            pool.putconn(conn)

    def _tool_check_item_availability(self, args: Dict[str, Any]) -> str:
        item_name = args.get("item_name", "")
        pool = get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                sess = get_session()
                if sess is not None:
                    cur.execute(
                        """
                        SELECT name, available
                        FROM menu_items
                        WHERE hotel_id = %s AND LOWER(TRIM(name)) = LOWER(TRIM(%s))
                        LIMIT 1
                        """,
                        (sess.hotel_id, item_name),
                    )
                else:
                    cur.execute(
                        """
                        SELECT name, available
                        FROM menu_items
                        WHERE LOWER(TRIM(name)) = LOWER(TRIM(%s))
                        LIMIT 1
                        """,
                        (item_name,),
                    )
                row = cur.fetchone()
            if row is None:
                return _dumps({"item": item_name, "available": False, "note": "Item not found on menu."})
            return _dumps({"item": row[0], "available": row[1]})
        except Exception as e:
            return _dumps({"error": str(e)})
        finally:
            pool.putconn(conn)

    def _tool_find_user_preference(self, args: Dict[str, Any]) -> str:
        customer_id = args.get("customer_id")
        hotel_id = args.get("hotel_id")
        if not customer_id or not hotel_id:
            return _dumps({"error": "Both customer_id and hotel_id are required."})

        pool = get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT name, dietary_preferences, allergens, preferred_cuisines,
                           favorite_dishes, disliked_ingredients, visit_count,
                           last_visit, total_spend, notes, age
                    FROM customers
                    WHERE customer_id = %s AND hotel_id = %s
                    LIMIT 1
                    """,
                    (customer_id, hotel_id),
                )
                row = cur.fetchone()

            if row is None:
                return _dumps({"found": False, "message": "Customer not found."})

            return _dumps({
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
                "age": row[10],
            })
        except Exception as e:
            return _dumps({"error": str(e)})
        finally:
            pool.putconn(conn)

    def _tool_place_order(self, args: Dict[str, Any]) -> str:
        raw = args.get("items")
        if not isinstance(raw, list):
            return _dumps({"ok": False, "error": "items must be a list of dish name strings."})
        items = [str(x).strip() for x in raw if str(x).strip()]
        tn = args.get("table_number")
        table_number = None
        if tn is not None and str(tn).strip() != "":
            try:
                table_number = int(tn)
            except (TypeError, ValueError):
                table_number = None
        result = persist_place_order(items, table_number=table_number)
        return _dumps(result)

    def _tool_review_and_feedback(self, args: Dict[str, Any]) -> str:
        br = args.get("bill_requested")
        bill_requested = bool(br) if br is not None else False
        ov = args.get("overall_rating")
        overall_rating = None
        if ov is not None and str(ov).strip() != "":
            try:
                overall_rating = int(ov)
            except (TypeError, ValueError):
                try:
                    overall_rating = int(float(ov))
                except (TypeError, ValueError):
                    overall_rating = None
        ft = args.get("feedback_text")
        feedback_text = (str(ft).strip() if ft is not None else None) or None
        raw_items = args.get("item_feedback")
        item_feedback = raw_items if isinstance(raw_items, list) else None
        oid = args.get("order_id")
        if oid is not None:
            oid = str(oid).strip() or None
        result = persist_review_and_feedback(
            bill_requested=bill_requested,
            overall_rating=overall_rating,
            feedback_text=feedback_text,
            item_feedback=item_feedback,
            order_id=oid,
        )
        return _dumps(result)

    def _tool_bring_the_bill(self, args: Dict[str, Any]) -> str:
        oid = args.get("order_id")
        if oid is not None:
            oid = str(oid).strip() or None
        result = persist_bring_the_bill(order_id=oid)
        return _dumps(result)

    def _tool_modify_order(self, args: Dict[str, Any]) -> str:
        action = args.get("action")
        oid = args.get("order_id")
        if oid is not None:
            oid = str(oid).strip() or None
        line_id = args.get("line_id")
        if line_id is not None:
            line_id = str(line_id).strip() or None
        dish_name = args.get("dish_name")
        if dish_name is not None:
            dish_name = str(dish_name).strip() or None
        qty = args.get("quantity")
        qv = None
        if qty is not None and str(qty).strip() != "":
            try:
                qv = int(qty)
            except (TypeError, ValueError):
                try:
                    qv = int(float(qty))
                except (TypeError, ValueError):
                    qv = None
        new_status = args.get("new_status")
        result = persist_modify_order(
            str(action) if action is not None else "",
            order_id=oid,
            line_id=line_id,
            dish_name=dish_name,
            quantity=qv,
            new_status=str(new_status) if new_status is not None else None,
        )
        return _dumps(result)

    def _tool_cancel_order(self, args: Dict[str, Any]) -> str:
        # Phase 2: DELETE /api/orders/{order_id}
        return _dumps({"status": "not_implemented", "message": "Order cancellation coming in Phase 2."})

    def _tool_update_guest_profile(self, args: Dict[str, Any]) -> str:
        name = (args.get("name") or "").strip()
        if not name:
            return _dumps({"ok": False, "error": "name is required."})
        age_raw = args.get("age")
        age = None
        if age_raw is not None and str(age_raw).strip() != "":
            try:
                age = int(age_raw)
            except (TypeError, ValueError):
                try:
                    age = int(float(age_raw))
                except (TypeError, ValueError):
                    age = None
        sess = get_session()
        if sess is None:
            return _dumps({"ok": False, "error": "No active session."})
        result = update_guest_profile(
            customer_id=int(sess.customer_id),
            hotel_id=int(sess.hotel_id),
            name=name,
            age=age,
        )
        if result.get("ok"):
            from app.services.llm_service import LLMService

            LLMService.clear_customer_cache(int(sess.customer_id), int(sess.hotel_id))
        return _dumps(result)
