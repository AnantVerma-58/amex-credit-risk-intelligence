
import csv
import os
from pathlib import Path

import pymysql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "raw" / "train_data.csv"
SOURCE_TABLE = "staging_monthly_raw"
TABLE = "staging_monthly_raw_v2"
EXPECTED_ROWS = 5_531_451

load_dotenv(ROOT / ".env")

with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
    columns = next(csv.reader(f))

if len(columns) != 190 or len(set(columns)) != 190:
    raise ValueError("CSV header validation failed")

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
        cur.execute(f"SHOW TABLES LIKE '{TABLE}'")
        exists = cur.fetchone() is not None

        if not exists:
            cur.execute(f"CREATE TABLE `{TABLE}` LIKE `{SOURCE_TABLE}`")

        cur.execute(f"SELECT COUNT(*) FROM `{TABLE}`")
        existing_rows = cur.fetchone()[0]
        if existing_rows:
            raise RuntimeError(
                f"{TABLE} contains {existing_rows:,} rows. "
                "Do not append or truncate automatically; inspect first."
            )

        # Drop the copied unique index during bulk loading.
        cur.execute("""
            SELECT COUNT(*)
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = %s
              AND INDEX_NAME = 'uq_customer_statement_date'
        """, (TABLE,))
        if cur.fetchone()[0]:
            cur.execute(f"""
                ALTER TABLE `{TABLE}`
                DROP INDEX uq_customer_statement_date
            """)

        # Read every CSV field into a temporary variable first.
        # This allows blank fields to be converted to SQL NULL.
        user_variables = ", ".join(f"@v_{c}" for c in columns)

        assignments = []
        for c in columns:
            raw = f"NULLIF(TRIM(@v_{c}), '')"

            if c == "S_2":
                assignments.append(
                    f"`S_2` = STR_TO_DATE({raw}, '%%Y-%%m-%%d')"
                )
            else:
                # Works for numeric fields and preserves NULL for
                # empty categorical/string fields.
                assignments.append(f"`{c}` = {raw}")

        sql = f"""
            LOAD DATA LOCAL INFILE %s
            INTO TABLE `{TABLE}`
            CHARACTER SET utf8mb4
            FIELDS TERMINATED BY ','
            OPTIONALLY ENCLOSED BY '"'
            ESCAPED BY '"'
            LINES TERMINATED BY '\\n'
            IGNORE 1 LINES
            ({user_variables})
            SET {", ".join(assignments)}
        """

        print(f"Loading corrected data into {TABLE}...")
        cur.execute(sql, (str(CSV_PATH),))

        # Read warnings immediately after LOAD DATA.
        cur.execute("SHOW COUNT(*) WARNINGS")
        warnings = cur.fetchone()[0]
        print("Import warnings:", warnings)

        if warnings:
            cur.execute("SHOW WARNINGS LIMIT 20")
            for warning in cur.fetchall():
                print("WARNING:", warning)

        cur.execute(f"""
            SELECT
                COUNT(*),
                SUM(customer_ID IS NULL),
                SUM(S_2 IS NULL),
                SUM(P_2 IS NULL),
                SUM(D_42 IS NULL),
                SUM(D_64 IS NULL)
            FROM `{TABLE}`
        """)
        rows, null_ids, null_dates, p2_nulls, d42_nulls, d64_nulls = (
            cur.fetchone()
        )

        print(f"Rows: {rows:,}")
        print(f"Null customer IDs: {null_ids:,}")
        print(f"Null statement dates: {null_dates:,}")
        print(f"P_2 nulls: {p2_nulls:,}")
        print(f"D_42 nulls: {d42_nulls:,}")
        print(f"D_64 nulls: {d64_nulls:,}")

        expected = (
            EXPECTED_ROWS,
            0,
            0,
            45_985,
            4_740_137,
            217_442,
        )
        actual = (
            rows,
            null_ids,
            null_dates,
            p2_nulls,
            d42_nulls,
            d64_nulls,
        )

        if actual != expected or warnings:
            raise RuntimeError(
                "Corrected-load validation failed. "
                "Keep the old table and inspect the results."
            )

        print("Corrected missingness checks passed.")

        # Enforce one record per customer and statement date.
        cur.execute(f"""
            ALTER TABLE `{TABLE}`
            ADD UNIQUE KEY uq_customer_statement_date
                (customer_ID, S_2)
        """)
        print("Unique customer/date index created.")

finally:
    conn.close()
