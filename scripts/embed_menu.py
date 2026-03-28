"""
One-time script: adds embedding column to menu_items and populates it.
Run once (safe to re-run — skips rows that already have embeddings):
    python scripts/embed_menu.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.config import config
from app.services.embedding_service import embed

ADD_COLUMN = """
ALTER TABLE menu_items
ADD COLUMN IF NOT EXISTS embedding vector(1536);
"""

FETCH_ROWS = """
SELECT dish_id, name, description
FROM menu_items
WHERE embedding IS NULL;
"""

UPDATE_EMBEDDING = """
UPDATE menu_items SET embedding = %s WHERE dish_id = %s;
"""

CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS menu_items_embedding_idx
ON menu_items USING hnsw (embedding vector_cosine_ops);
"""


def main():
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(
        host=config.PGSQL_ENDPOINT,
        dbname=config.PGSQL_DB_NAME,
        user=config.PGSQL_ADMIN_USERNAME,
        password=config.PGSQL_ADMIN_PASSWORD,
        port=5432,
        sslmode="require",
    )
    conn.autocommit = True

    with conn.cursor() as cur:
        print("Adding embedding column (if not exists)...")
        cur.execute(ADD_COLUMN)

        cur.execute(FETCH_ROWS)
        rows = cur.fetchall()
        print(f"{len(rows)} rows need embedding...")

        for dish_id, name, description in rows:
            text = f"{name}. {description or ''}"
            vector = embed(text)
            cur.execute(UPDATE_EMBEDDING, (vector, dish_id))
            print(f"  Embedded: {name}")

        print("Creating HNSW index...")
        cur.execute(CREATE_INDEX)

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
