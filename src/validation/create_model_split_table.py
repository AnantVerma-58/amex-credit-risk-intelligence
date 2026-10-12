
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
        cur.execute("""
            SELECT COUNT(*)
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'model_customer_split'
        """)

        if cur.fetchone()[0]:
            raise RuntimeError(
                "model_customer_split already exists. "
                "Inspect it instead of overwriting the assignments."
            )

        print("Creating stratified customer splits in MariaDB...")

        cur.execute("""
            CREATE TABLE model_customer_split ENGINE=InnoDB AS
            SELECT
                customer_ID,
                target,
                CASE
                    WHEN rn <= FLOOR(class_count * 0.70)
                        THEN 'train'
                    WHEN rn <= FLOOR(class_count * 0.85)
                        THEN 'validation'
                    ELSE 'test'
                END AS split
            FROM (
                SELECT
                    customer_ID,
                    target,
                    ROW_NUMBER() OVER (
                        PARTITION BY target
                        ORDER BY CRC32(customer_ID), customer_ID
                    ) AS rn,
                    COUNT(*) OVER (
                        PARTITION BY target
                    ) AS class_count
                FROM customer_labels
            ) ranked
        """)

        cur.execute("""
            ALTER TABLE model_customer_split
            ADD PRIMARY KEY (customer_ID),
            ADD INDEX idx_model_split (split)
        """)

        cur.execute("""
            SELECT
                split,
                target,
                COUNT(*) AS customers
            FROM model_customer_split
            GROUP BY split, target
            ORDER BY
                FIELD(split, 'train', 'validation', 'test'),
                target
        """)

        print("\nSplit distribution:")
        for split, target, count in cur.fetchall():
            print(f"{split:10s} target={target} customers={count:,}")

        cur.execute("SELECT COUNT(*) FROM model_customer_split")
        total = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(*)
            FROM model_customer_split s
            LEFT JOIN customer_features_v2 f
                ON f.customer_ID = s.customer_ID
            WHERE f.customer_ID IS NULL
        """)
        missing_features = cur.fetchone()[0]

        if total != 458_913:
            raise RuntimeError(f"Expected 458,913 assignments, got {total:,}")

        if missing_features:
            raise RuntimeError(
                f"{missing_features:,} customers have no feature row."
            )

        print(f"\nTotal assignments: {total:,}")
        print(f"Customers without features: {missing_features:,}")
        print("Split assignment validation passed.")

finally:
    conn.close()
