
import os
from pathlib import Path

import pymysql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

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
        # Add a unique key to enforce one row per customer per statement date.
        cur.execute("""
            SELECT COUNT(*)
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'staging_monthly_raw'
              AND INDEX_NAME = 'uq_customer_statement_date'
        """)
        index_exists = cur.fetchone()[0] > 0

        if not index_exists:
            print("Creating unique customer/date index; this may take a while...")
            cur.execute("""
                ALTER TABLE staging_monthly_raw
                ADD UNIQUE KEY uq_customer_statement_date (customer_ID, S_2)
            """)
            print("Unique index created.")

        # Confirm that every labeled customer appears in the monthly data.
        cur.execute("""
            SELECT COUNT(*)
            FROM customer_labels l
            LEFT JOIN (
                SELECT DISTINCT customer_ID
                FROM staging_monthly_raw
            ) m ON m.customer_ID = l.customer_ID
            WHERE m.customer_ID IS NULL
        """)
        labels_without_monthly_data = cur.fetchone()[0]

        # Find monthly customers without a label.
        cur.execute("""
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT customer_ID
                FROM staging_monthly_raw
            ) m
            LEFT JOIN customer_labels l
              ON l.customer_ID = m.customer_ID
            WHERE l.customer_ID IS NULL
        """)
        monthly_customers_without_labels = cur.fetchone()[0]

        # Inspect actual D_63 categories and their frequencies.
        cur.execute("""
            SELECT D_63, COUNT(*) AS records
            FROM staging_monthly_raw
            GROUP BY D_63
            ORDER BY records DESC
        """)
        d63_values = cur.fetchall()

        print("\nRelationship validation")
        print("-" * 40)
        print("Labels without monthly data:", labels_without_monthly_data)
        print("Monthly customers without labels:", monthly_customers_without_labels)

        print("\nActual D_63 values:")
        for value, count in d63_values:
            print(f"  {value!r}: {count:,}")

        if labels_without_monthly_data != 0:
            raise RuntimeError("Some labeled customers have no monthly records.")

        if monthly_customers_without_labels != 0:
            raise RuntimeError("Some monthly customers have no labels.")

        print("\nDatabase relationship validation passed.")
finally:
    conn.close()
