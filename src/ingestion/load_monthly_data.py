
import csv
import os
from pathlib import Path

import pymysql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "raw" / "train_data.csv"
TABLE_NAME = "staging_monthly_raw"
EXPECTED_ROWS = 5_531_451

load_dotenv(ROOT / ".env")

with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
    columns = next(csv.reader(f))

if len(columns) != 190 or len(set(columns)) != len(columns):
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
        cur.execute(f"SELECT COUNT(*) FROM `{TABLE_NAME}`")
        existing_rows = cur.fetchone()[0]

        if existing_rows != 0:
            raise RuntimeError(
                f"{TABLE_NAME} already contains {existing_rows:,} rows. "
                "Refusing to append. Inspect the table before retrying."
            )

        # Build the column list from the validated CSV header.
        column_list = ", ".join(f"`{col}`" for col in columns)

        # LOCAL INFILE reads the source file on this machine.
        # Empty strings become NULL; the statement date is parsed explicitly.
        sql = f"""
        LOAD DATA LOCAL INFILE %s
        INTO TABLE `{TABLE_NAME}`
        CHARACTER SET utf8mb4
        FIELDS TERMINATED BY ','
        OPTIONALLY ENCLOSED BY '"'
        ESCAPED BY '"'
        LINES TERMINATED BY '\\n'
        IGNORE 1 LINES
        ({column_list})
        SET
            `customer_ID` = NULLIF(TRIM(`customer_ID`), ''),
            `S_2` = STR_TO_DATE(TRIM(`S_2`), '%%Y-%%m-%%d')
        """

        print(f"Loading: {CSV_PATH}")
        print(f"Expected rows: {EXPECTED_ROWS:,}")
        print("This may take a while; do not interrupt unless necessary.")

        cur.execute(sql, (str(CSV_PATH),))
        print("Rows reported affected by LOAD DATA:", f"{cur.rowcount:,}")

        cur.execute(f"SELECT COUNT(*) FROM `{TABLE_NAME}`")
        actual_rows = cur.fetchone()[0]
        print("Rows currently in staging table:", f"{actual_rows:,}")

        cur.execute("SHOW COUNT(*) WARNINGS")
        warning_count = cur.fetchone()[0]
        print("Import warning count:", warning_count)

        if warning_count:
            cur.execute("SHOW WARNINGS LIMIT 20")
            for warning in cur.fetchall():
                print("WARNING:", warning)

        if actual_rows != EXPECTED_ROWS:
            raise RuntimeError(
                f"Row-count mismatch: expected {EXPECTED_ROWS:,}, "
                f"found {actual_rows:,}. Investigate before proceeding."
            )

        print("Row-count validation passed.")
finally:
    conn.close()
