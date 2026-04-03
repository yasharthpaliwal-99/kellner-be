"""Enroll / identify guests using local embeddings + pgvector."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.config import config

logger = logging.getLogger(__name__)
from app.db.pool import get_pool
from app.services.face_local_ml import embedding_from_image_bytes


def _vec_to_sql_literal(emb: np.ndarray) -> str:
    return "[" + ",".join(str(float(x)) for x in emb.flatten().tolist()) + "]"


def identify_from_embedding(emb: np.ndarray, hotel_id: int) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    """
    Nearest-neighbor by cosine distance; accept if dist <= FACE_MATCH_MAX_DISTANCE.
    Returns (customer_id, distance, error_message).
    """
    pool = get_pool()
    conn = pool.getconn()
    try:
        lit = _vec_to_sql_literal(emb)
        max_d = float(getattr(config, "FACE_MATCH_MAX_DISTANCE", 0.45))
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT customer_id, embedding <=> %s::vector AS dist
                FROM face_embeddings
                WHERE hotel_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT 1
                """,
                (lit, hotel_id, lit),
            )
            row = cur.fetchone()
        if not row:
            logger.info("face_identify_no_rows hotel_id=%s (no embeddings yet)", hotel_id)
            return None, None, None
        cid, dist = int(row[0]), float(row[1])
        if dist <= max_d:
            logger.info(
                "face_identify_match hotel_id=%s customer_id=%s cosine_distance=%.4f threshold=%.4f",
                hotel_id,
                cid,
                dist,
                max_d,
            )
            return cid, dist, None
        logger.info(
            "face_identify_near_miss hotel_id=%s nearest_customer_id=%s cosine_distance=%.4f threshold=%.4f",
            hotel_id,
            cid,
            dist,
            max_d,
        )
        return None, dist, None
    except Exception as e:
        logger.warning("face_identify_error hotel_id=%s error=%s", hotel_id, e)
        return None, None, str(e)
    finally:
        pool.putconn(conn)


def enroll_customer_with_embedding(emb: np.ndarray, hotel_id: int) -> Tuple[Optional[int], Optional[str]]:
    """
    Create customer row (name/age null), store embedding.
    Returns (customer_id, error_message).
    """
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO customers (
                    hotel_id, name, email, phone,
                    dietary_preferences, allergens, preferred_cuisines,
                    favorite_dishes, disliked_ingredients,
                    visit_count, total_spend, notes, age
                )
                VALUES (
                    %s, NULL, NULL, NULL,
                    ARRAY[]::TEXT[], ARRAY[]::TEXT[], ARRAY[]::TEXT[],
                    ARRAY[]::TEXT[], ARRAY[]::TEXT[],
                    0, 0, 'face-local enroll', NULL
                )
                RETURNING customer_id
                """,
                (hotel_id,),
            )
            row = cur.fetchone()
            if not row:
                return None, "Failed to create customer."
            cid = int(row[0])
            lit = _vec_to_sql_literal(emb)
            cur.execute(
                """
                INSERT INTO face_embeddings (customer_id, hotel_id, embedding)
                VALUES (%s, %s, %s::vector)
                """,
                (cid, hotel_id, lit),
            )
        conn.commit()
        logger.info("face_enroll_new hotel_id=%s customer_id=%s", hotel_id, cid)
        return cid, None
    except Exception as e:
        conn.rollback()
        logger.warning("face_enroll_error hotel_id=%s error=%s", hotel_id, e)
        return None, str(e)
    finally:
        pool.putconn(conn)


def recognise_me(
    image_bytes: bytes, hotel_id: int
) -> Tuple[Optional[int], bool, Optional[float], bool, Optional[str]]:
    """
    One-shot: match existing face, or create customer + embedding.
    Returns (customer_id, matched, distance_or_nearest_miss, created_new, error_message).
    """
    try:
        emb = embedding_from_image_bytes(image_bytes)
    except Exception as e:
        logger.info("face_embedding_failed hotel_id=%s error=%s", hotel_id, e)
        return None, False, None, False, str(e)

    cid, dist, err = identify_from_embedding(emb, hotel_id)
    if err:
        return None, False, None, False, err
    if cid is not None:
        return cid, True, dist, False, None

    eid, eerr = enroll_customer_with_embedding(emb, hotel_id)
    if eerr:
        return None, False, dist, False, eerr
    return eid, False, dist, True, None


def customer_belongs_to_hotel(customer_id: int, hotel_id: int) -> bool:
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM customers WHERE customer_id = %s AND hotel_id = %s LIMIT 1",
                (customer_id, hotel_id),
            )
            return cur.fetchone() is not None
    finally:
        pool.putconn(conn)


def update_guest_profile(customer_id: int, hotel_id: int, name: str, age: Optional[int]) -> Dict[str, Any]:
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE customers
                SET name = %s, age = %s
                WHERE customer_id = %s AND hotel_id = %s
                RETURNING customer_id
                """,
                (name.strip(), age, customer_id, hotel_id),
            )
            row = cur.fetchone()
        conn.commit()
        if not row:
            return {"ok": False, "error": "Customer not found or hotel mismatch."}
        return {"ok": True, "customer_id": customer_id}
    except Exception as e:
        conn.rollback()
        return {"ok": False, "error": str(e)}
    finally:
        pool.putconn(conn)
