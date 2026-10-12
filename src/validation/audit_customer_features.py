
import os
from pathlib import Path

import pymysql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

TABLE = "customer_features_v1"
EXPECTED_CUSTOMERS = 458_913

conn = pymysql.connect(
    host=os.environ["DB_HOST"],
    port=int(os.environ["DB_PORT"]),
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    database=os.environ["DB_NAME"],
    charset="utf8mb4",
)

try:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COLUMN_NAME, DATA_TYPE
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION
        """, (TABLE,))
        columns = cur.fetchall()

        if not columns:
            raise RuntimeError(f"Table not found: {TABLE}")

        feature_names = [
            name for name, _ in columns if name != "customer_ID"
        ]

        # A single aggregate query profiles all feature columns.
        expressions = []
        for name in feature_names:
            q = f"`{name}`"
            expressions.append(f"SUM({q} IS NULL) AS `{name}`")

        sql = f"""
            SELECT COUNT(*) AS row_count,
                   {", ".join(expressions)}
            FROM `{TABLE}`
        """
        cur.execute(sql)
        result = cur.fetchone()

        row_count = result[0]
        null_counts = dict(zip(feature_names, result[1:]))

        print("Feature table audit")
        print("-" * 45)
        print(f"Rows: {row_count:,}")
        print(f"Columns including customer_ID: {len(columns):,}")
        print(f"Feature columns: {len(feature_names):,}")

        if row_count != EXPECTED_CUSTOMERS:
            raise RuntimeError(
                f"Expected {EXPECTED_CUSTOMERS:,} rows, found {row_count:,}"
            )

        # Confirm every feature customer has exactly one label.
        cur.execute("""
            SELECT
                COUNT(*) AS matched_rows,
                SUM(l.customer_ID IS NULL) AS missing_labels
            FROM customer_features_v1 f
            LEFT JOIN customer_labels l
                ON l.customer_ID = f.customer_ID
        """)
        matched_rows, missing_labels = cur.fetchone()

        print(f"Feature rows checked against labels: {matched_rows:,}")
        print(f"Feature customers without labels: {missing_labels:,}")

        # Summarize feature missingness.

        ranked = sorted(
            [
                (name, int(count), 100.0 * int(count) / row_count)
                for name, count in null_counts.items()
            ],
            key=lambda x: x[2],
            reverse=True,
        )

        all_null = [
            name for name, count in null_counts.items()
            if count == row_count
        ]

        print("\nTop 20 features by missing percentage:")
        for name, count, pct in ranked[:20]:
            print(f"{name:24s} {count:>10,} missing ({pct:6.2f}%)")

        print(f"\nAll-null feature columns: {len(all_null)}")
        for name in all_null:
            print(" ", name)

        if missing_labels != 0:
            raise RuntimeError("Some feature customers have no labels.")

        print("\nFeature audit passed.")
finally:
    conn.close()
