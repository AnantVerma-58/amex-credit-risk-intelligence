
import os
from pathlib import Path

import pymysql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

SOURCE_TABLE = "staging_monthly_raw_v2"
FEATURE_TABLE = "customer_features_v2"
HISTORY_TABLE = "customer_history_summary"

# Encoded numeric values that should be treated as categories,
# not averaged as if they were continuous measurements.
CATEGORICAL_FEATURES = {
    "B_30", "B_38", "D_114", "D_116", "D_117",
    "D_120", "D_126", "D_66", "D_68",
}

conn = pymysql.connect(
    host=os.environ["DB_HOST"],
    port=int(os.environ["DB_PORT"]),
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    database=os.environ["DB_NAME"],
    charset="utf8mb4",
    autocommit=True,
)

try:
    with conn.cursor() as cur:
        # Do not overwrite an existing feature table accidentally.
        cur.execute("""
            SELECT COUNT(*)
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = %s
        """, (FEATURE_TABLE,))
        if cur.fetchone()[0]:
            raise RuntimeError(
                f"{FEATURE_TABLE} already exists. Inspect it before rebuilding."
            )

        # Discover the source columns from MariaDB.
        cur.execute("""
            SELECT COLUMN_NAME, DATA_TYPE
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION
        """, (SOURCE_TABLE,))
        source_columns = cur.fetchall()

        if len(source_columns) != 190:
            raise RuntimeError(
                f"Expected 190 source columns; found {len(source_columns)}"
            )

        feature_columns = [
            (name, dtype)
            for name, dtype in source_columns
            if name not in {"customer_ID", "S_2"}
        ]

        select_expressions = ["s.customer_ID"]

        for name, dtype in feature_columns:
            quoted = f"`{name}`"

            # Latest observed value, using the customer's final statement date.
            select_expressions.append(
                f"MAX(CASE WHEN s.S_2 = h.last_statement_date "
                f"THEN s.{quoted} END) AS `latest__{name}`"
            )

            # Average only genuinely numeric, non-categorical fields.
            if (
                dtype in {"double", "float", "decimal", "int",
                          "bigint", "smallint", "mediumint", "tinyint"}
                and name not in CATEGORICAL_FEATURES
            ):
                select_expressions.append(
                    f"AVG(s.{quoted}) AS `mean__{name}`"
                )

        select_sql = ",\n    ".join(select_expressions)

        create_sql = f"""
            CREATE TABLE `{FEATURE_TABLE}` ENGINE=InnoDB AS
            SELECT
                {select_sql}
            FROM `{SOURCE_TABLE}` s
            INNER JOIN `{HISTORY_TABLE}` h
                ON h.customer_ID = s.customer_ID
            GROUP BY s.customer_ID
        """

        print("Building customer-level features from MariaDB.")
        print(f"Source table: {SOURCE_TABLE}")
        print(f"Output table: {FEATURE_TABLE}")
        print("This query may take several minutes.")

        cur.execute(create_sql)

        print("Feature table created.")

        cur.execute(f"""
            ALTER TABLE `{FEATURE_TABLE}`
            ADD PRIMARY KEY (customer_ID)
        """)
        print("Primary key created.")

        cur.execute(f"SELECT COUNT(*) FROM `{FEATURE_TABLE}`")
        customer_count = cur.fetchone()[0]

        cur.execute(f"SHOW COLUMNS FROM `{FEATURE_TABLE}`")
        column_count = len(cur.fetchall())

        print(f"Customer rows: {customer_count:,}")
        print(f"Total columns including customer_ID: {column_count:,}")

        if customer_count != 458_913:
            raise RuntimeError(
                f"Expected 458,913 customers; found {customer_count:,}"
            )

        print("Feature-table row-count validation passed.")

finally:
    conn.close()
