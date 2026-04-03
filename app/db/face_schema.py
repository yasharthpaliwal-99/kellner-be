"""PostgreSQL schema for local face embeddings (InsightFace + pgvector)."""
from __future__ import annotations

from app.config import config
from app.db.pool import get_pool


def ensure_face_schema() -> None:
    """Idempotent: customers.name nullable, age column, face_embeddings table."""
    try:
        pool = get_pool()
    except ValueError:
        return
    conn = pool.getconn()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                DO $$
                BEGIN
                  IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'customers'
                      AND column_name = 'name' AND is_nullable = 'NO'
                  ) THEN
                    ALTER TABLE customers ALTER COLUMN name DROP NOT NULL;
                  END IF;
                END $$;
                """
            )
            cur.execute(
                """
                ALTER TABLE customers
                ADD COLUMN IF NOT EXISTS age INTEGER;
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id              SERIAL PRIMARY KEY,
                    customer_id     INTEGER NOT NULL UNIQUE
                        REFERENCES customers(customer_id) ON DELETE CASCADE,
                    hotel_id        INTEGER NOT NULL,
                    embedding       vector(512) NOT NULL,
                    created_at      TIMESTAMP DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_face_embeddings_hotel
                ON face_embeddings(hotel_id);
                """
            )
    finally:
        pool.putconn(conn)
