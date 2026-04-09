"""
Per-turn latency / LLM metrics.

Create `transactions` yourself: run the SQL below once in psql / Azure Query Editor
(same conventions as menu_items: public, INTEGER ids, TIMESTAMP DEFAULT NOW()).
The app does not auto-create this table.

    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id         UUID PRIMARY KEY,
        session_id             TEXT NOT NULL,
        customer_id            INTEGER NOT NULL,
        hotel_id                 INTEGER NOT NULL,
        turn_id                  INTEGER NOT NULL,
        date_created             TIMESTAMP DEFAULT NOW(),
        stt_seconds              DOUBLE PRECISION,
        retrieval_seconds        DOUBLE PRECISION,
        llm_tools_wall_seconds   DOUBLE PRECISION,
        llm_stream_wait_seconds  DOUBLE PRECISION,
        tts_seconds              DOUBLE PRECISION,
        first_audio_seconds      DOUBLE PRECISION,
        turn_total_seconds       DOUBLE PRECISION NOT NULL,
        tools_called             BOOLEAN,
        recommendation_count     INTEGER,
        transcript_chars         INTEGER,
        assistant_chars          INTEGER,
        had_error                BOOLEAN NOT NULL DEFAULT false,
        events                   JSONB NOT NULL DEFAULT '[]'::jsonb
    );
    CREATE INDEX IF NOT EXISTS idx_transactions_session_customer_date
        ON transactions (session_id, customer_id, date_created DESC);
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from psycopg2.extras import Json

from app.db.pool import get_pool

logger = logging.getLogger(__name__)


def insert_transaction_row(row: Dict[str, Any]) -> None:
    """Best-effort insert; skips if PG is not configured or insert fails."""
    try:
        pool = get_pool()
    except ValueError:
        logger.debug("transactions: PostgreSQL not configured, skip")
        return
    conn = pool.getconn()
    try:
        params = dict(row)
        params["events"] = Json(params.get("events") or [])
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO transactions (
                    transaction_id, session_id, customer_id, hotel_id, turn_id,
                    stt_seconds, retrieval_seconds, llm_tools_wall_seconds,
                    llm_stream_wait_seconds, tts_seconds, first_audio_seconds,
                    turn_total_seconds, tools_called, recommendation_count,
                    transcript_chars, assistant_chars, had_error, events
                ) VALUES (
                    CAST(%(transaction_id)s AS uuid), %(session_id)s, %(customer_id)s, %(hotel_id)s,
                    %(turn_id)s, %(stt_seconds)s, %(retrieval_seconds)s, %(llm_tools_wall_seconds)s,
                    %(llm_stream_wait_seconds)s, %(tts_seconds)s, %(first_audio_seconds)s,
                    %(turn_total_seconds)s, %(tools_called)s, %(recommendation_count)s,
                    %(transcript_chars)s, %(assistant_chars)s, %(had_error)s, %(events)s
                )
                """,
                params,
            )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.warning("transactions: insert failed: %s", exc)
    finally:
        pool.putconn(conn)
