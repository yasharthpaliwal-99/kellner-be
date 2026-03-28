"""
One-time PostgreSQL bootstrap for Kellner (Azure Flexible Server + pgvector).

Creates:
  - extension vector (allow-list VECTOR in Azure Portal first)
  - menu_items (semantic search via embed_menu.py + tool_executor)
  - customers (profiles for find_user_preference)

Run once:
    python scripts/setup_db.py

Then (after menu rows exist):
    python scripts/embed_menu.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.config import config

CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

CREATE_MENU_TABLE = """
CREATE TABLE IF NOT EXISTS menu_items (
    dish_id       SERIAL PRIMARY KEY,
    hotel_id      INTEGER NOT NULL,
    name          TEXT NOT NULL,
    price         NUMERIC(10, 2),
    ingredients   TEXT[],
    cuisine_type  TEXT,
    description   TEXT,
    allergens     TEXT[],
    available     BOOLEAN DEFAULT true,
    date_created  TIMESTAMP DEFAULT NOW()
);
"""

CREATE_CUSTOMERS_TABLE = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id          SERIAL PRIMARY KEY,
    hotel_id             INTEGER NOT NULL,
    name                 TEXT NOT NULL,
    email                TEXT UNIQUE,
    phone                TEXT,
    dietary_preferences  TEXT[],
    allergens            TEXT[],
    preferred_cuisines   TEXT[],
    favorite_dishes      TEXT[],
    disliked_ingredients TEXT[],
    visit_count          INTEGER DEFAULT 0,
    last_visit           TIMESTAMP,
    total_spend          NUMERIC(10, 2) DEFAULT 0,
    notes                TEXT,
    date_created         TIMESTAMP DEFAULT NOW()
);
"""

MENU_SEED = [
    (1, "Grilled Salmon", 24.99, ["Atlantic salmon", "lemon butter", "capers", "seasonal vegetables"], "Continental", "Pan-seared Atlantic salmon fillet with lemon butter sauce and seasonal vegetables", ["fish", "dairy"]),
    (1, "Chicken Tikka", 18.99, ["chicken", "yogurt", "spices", "basmati rice", "mint chutney"], "Indian", "Tandoor-marinated chicken served with basmati rice and mint chutney", ["dairy", "gluten"]),
    (1, "Mushroom Risotto", 17.99, ["arborio rice", "wild mushrooms", "parmesan", "truffle oil", "butter"], "Italian", "Creamy arborio rice with wild mushrooms, parmesan and truffle oil", ["dairy", "gluten"]),
    (1, "Margherita Pizza", 15.99, ["pizza dough", "tomato sauce", "mozzarella", "fresh basil"], "Italian", "Classic pizza with tomato base, fresh mozzarella and basil", ["gluten", "dairy"]),
    (1, "Caesar Salad", 12.99, ["romaine lettuce", "croutons", "parmesan", "caesar dressing", "egg"], "Continental", "Crisp romaine lettuce with croutons, parmesan and house caesar dressing", ["gluten", "dairy", "egg"]),
    (1, "Bruschetta", 8.99, ["sourdough", "tomatoes", "basil", "garlic", "olive oil"], "Italian", "Toasted sourdough topped with fresh tomatoes, basil and olive oil", ["gluten"]),
    (1, "Garlic Bread", 6.99, ["sourdough", "roasted garlic", "butter", "parsley"], "Continental", "Toasted sourdough with roasted garlic butter and herbs", ["gluten", "dairy"]),
    (1, "Tiramisu", 8.99, ["mascarpone", "espresso", "ladyfingers", "cocoa", "egg"], "Italian", "Classic Italian dessert with mascarpone cream and espresso-soaked ladyfingers", ["dairy", "gluten", "egg"]),
    (1, "Mango Sorbet", 7.99, ["fresh mango", "sugar", "lime juice"], "Continental", "House-made fresh mango sorbet with mint garnish", []),
    (1, "Chocolate Lava Cake", 9.99, ["dark chocolate", "butter", "egg", "flour", "sugar"], "Continental", "Warm dark chocolate cake with molten centre, served with vanilla ice cream", ["dairy", "gluten", "egg"]),
    (1, "Sparkling Water", 3.99, ["water"], "Beverages", "Still or sparkling mineral water, 500ml", []),
    (1, "Fresh Lemonade", 4.99, ["lemon", "sugar", "mint", "water"], "Beverages", "House-made lemonade with fresh mint and ice", []),
    (1, "House Red Wine", 9.99, ["red wine"], "Beverages", "Glass of our selected house red wine, 175ml", ["sulphites"]),
]

INSERT_MENU = """
INSERT INTO menu_items
    (hotel_id, name, price, ingredients, cuisine_type, description, allergens)
SELECT %s, %s, %s, %s, %s, %s, %s
WHERE NOT EXISTS (
    SELECT 1 FROM menu_items m
    WHERE m.hotel_id = %s AND LOWER(TRIM(m.name)) = LOWER(TRIM(%s))
);
"""

CUSTOMER_SEED = [
    (1, "Aarav Sharma", "aarav@example.com", "+1-555-0101", ["vegetarian"], ["dairy"], ["Indian", "Italian"], ["Mushroom Risotto", "Bruschetta"], ["meat"], 12, 320.50, "Prefers window seat"),
    (1, "Priya Mehta", "priya@example.com", "+1-555-0102", ["vegan"], [], ["Continental", "Italian"], ["Caesar Salad", "Mango Sorbet"], ["dairy", "egg"], 5, 110.00, "Birthday on April 3"),
    (1, "James Walker", "james@example.com", "+1-555-0103", [], ["nuts"], ["Continental"], ["Grilled Salmon", "Tiramisu"], [], 20, 780.00, "VIP — frequent guest"),
    (1, "Sara Nguyen", "sara@example.com", "+1-555-0104", ["gluten-free"], ["gluten"], ["Indian"], ["Chicken Tikka"], ["wheat", "barley"], 3, 55.00, None),
    (1, "Carlos Rivera", "carlos@example.com", "+1-555-0105", [], ["shellfish"], ["Italian", "Continental"], ["Margherita Pizza", "House Red Wine"], ["shellfish"], 8, 245.00, "Allergic to shellfish — flag always"),
]

INSERT_CUSTOMER = """
INSERT INTO customers
    (hotel_id, name, email, phone, dietary_preferences, allergens,
     preferred_cuisines, favorite_dishes, disliked_ingredients,
     visit_count, total_spend, notes)
VALUES
    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (email) DO NOTHING;
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
        print("Enabling pgvector extension…")
        cur.execute(CREATE_EXTENSION)

        print("Creating menu_items…")
        cur.execute(CREATE_MENU_TABLE)

        print("Creating customers…")
        cur.execute(CREATE_CUSTOMERS_TABLE)

        print("Seeding menu…")
        for row in MENU_SEED:
            hid, name, price, ing, cuisine, desc, allerg = row
            cur.execute(
                INSERT_MENU,
                (hid, name, price, ing, cuisine, desc, allerg, hid, name),
            )

        print("Seeding customers…")
        for row in CUSTOMER_SEED:
            cur.execute(INSERT_CUSTOMER, row)

    conn.close()
    print(f"Done. Menu rows: {len(MENU_SEED)}, customers: {len(CUSTOMER_SEED)}.")
    print("Next: python scripts/embed_menu.py")


if __name__ == "__main__":
    main()
