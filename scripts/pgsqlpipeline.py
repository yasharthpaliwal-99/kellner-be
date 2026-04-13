import psycopg2

#DB config
conn = psycopg2.connect(
    host="pgsqlkellner.postgres.database.azure.com",
    dbname="postgres",
    user="kellnerpgsql",
    password="California@2027",
    port=5432,
    sslmode="require"
)
PGSQL_ENDPOINT=pgsqlkellner.postgres.database.azure.com
PGSQL_DB_NAME=postgres
PGSQL_ADMIN_USERNAME=kellnerpgsql
PGSQL_ADMIN_PASSWORD=California@2027
cursor = conn.cursor()

# 📂 CSV file path
file_path = "/Users/yasharthpaliwal/Downloads/CCCMenu - Sheet2.csv"

# 🚀 COPY command
with open(file_path, "r") as f:
    cursor.copy_expert(
        """
        COPY menu_items(
        dish_id, hotel_id, name, price, ingredients, cuisine_type, description,
        allergens, available, date_created, embedding, taste_profile, spice_level,
        texture, is_veg, is_vegan, course_type, best_pair_with, is_signature,
        prep_time, customizable, custom_options, tags, is_combo
        )
        FROM STDIN WITH CSV HEADER
        """,
        f
    )

conn.commit()
cursor.close()
conn.close()

print("Append Done")