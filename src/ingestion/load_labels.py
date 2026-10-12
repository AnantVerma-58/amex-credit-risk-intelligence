
import csv
import os
from pathlib import Path

import pymysql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "raw" / "train_labels.csv"
EXPECTED_ROWS = 458_913

load_dotenv(ROOT / ".env")

with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
    header = next(csv.reader(f))

if header != ["customer_ID", "target"]:
    raise ValueError(f"Unexpected label header: {header}")

conn = pymysql.connect(
    host=os.environ["DB_HOST"],
    port=int(os.environ["DB_PORT"]),
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    database=os.environ["DB_NAME"],
    charset="utf8mb4",
    local_infile=True,
    autocommit=True,
)

try:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS customer_labels (
                customer_ID VARCHAR(64) NOT NULL,
                target TINYINT UNSIGNED NOT NULL,
                PRIMARY KEY (customer_ID),
                CONSTRAINT chk_customer_target CHECK (target IN (0, 1))
            ) ENGINE=InnoDB
              DEFAULT CHARSET=utf8mb4
              COLLATE=utf8mb4_unicode_ci
        """)

        cur.execute("SELECT COUNT(*) FROM customer_labels")
        existing = cur.fetchone()[0]
        if existing:
            raise RuntimeError(
                f"customer_labels already contains {existing:,} rows. "
                "Refusing to append; inspect before retrying."
            )

        sql = r"""
            LOAD DATA LOCAL INFILE %s
            INTO TABLE customer_labels
            CHARACTER SET utf8mb4
            FIELDS TERMINATED BY ','
            OPTIONALLY ENCLOSED BY '"'
            ESCAPED BY '"'
            LINES TERMINATED BY '\n'
            IGNORE 1 LINES
            (@customer_ID, @target)
            SET
                customer_ID = NULLIF(TRIM(@customer_ID), ''),
                target = NULLIF(TRIM(@target), '')
        """

        print(f"Loading labels from {CSV_PATH}")
        cur.execute(sql, (str(CSV_PATH),))
        print("Rows reported affected:", f"{cur.rowcount:,}")

        cur.execute("SELECT COUNT(*) FROM customer_labels")
        row_count = cur.fetchone()[0]

        cur.execute("""
            SELECT target, COUNT(*)
            FROM customer_labels
            GROUP BY target
            ORDER BY target
        """)
        distribution = cur.fetchall()

        cur.execute("SHOW COUNT(*) WARNINGS")
        warning_count = cur.fetchone()[0]

        print("Label rows:", f"{row_count:,}")
        print("Target distribution:")
        for target, count in distribution:
            print(f"  {target}: {count:,}")
        print("Import warnings:", warning_count)

        if warning_count:
            cur.execute("SHOW WARNINGS LIMIT 20")
            for warning in cur.fetchall():
                print("WARNING:", warning)

        if row_count != EXPECTED_ROWS:
            raise RuntimeError(
                f"Expected {EXPECTED_ROWS:,} labels, found {row_count:,}"
            )

        if distribution != ((0, 340085), (1, 118828)):
            raise RuntimeError(
                f"Unexpected target distribution: {distribution}"
            )

        if warning_count:
            raise RuntimeError("Import warnings found; inspect before proceeding.")

        print("Label import validation passed.")
finally:
    conn.close()
