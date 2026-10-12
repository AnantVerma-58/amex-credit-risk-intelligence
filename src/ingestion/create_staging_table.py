
import csv
import os
from pathlib import Path

import pymysql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "raw" / "train_data.csv"
TABLE_NAME = "staging_monthly_raw"

# Competition categorical fields. Keep their original encoded values
# in staging; categorical treatment happens later in feature engineering.
STRING_FEATURES = {"D_63", "D_64"}

load_dotenv(ROOT / ".env")

required_env = [
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
]
missing_env = [key for key in required_env if not os.getenv(key)]
if missing_env:
    raise RuntimeError(f"Missing .env settings: {missing_env}")

with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
    columns = next(csv.reader(f))

if len(columns) != 190 or len(set(columns)) != len(columns):
    raise ValueError("CSV header failed validation")

if columns[:2] != ["customer_ID", "S_2"]:
    raise ValueError("Unexpected first two CSV columns")

# All names originate from the validated source header.
definitions = []
for col in columns:
    if col == "customer_ID":
        definitions.append("`customer_ID` VARCHAR(64) NULL")
    elif col == "S_2":
        definitions.append("`S_2` DATE NULL")
    elif col in STRING_FEATURES:
        definitions.append(f"`{col}` VARCHAR(32) NULL")
    else:
        definitions.append(f"`{col}` DOUBLE NULL")

create_sql = f"""
CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (
    {", ".join(definitions)}
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
"""

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
        cur.execute(create_sql)
        cur.execute(f"SHOW COLUMNS FROM `{TABLE_NAME}`")
        actual_columns = cur.fetchall()

    if len(actual_columns) != len(columns):
        raise RuntimeError(
            f"Expected {len(columns)} columns, found {len(actual_columns)}"
        )

    print(f"Table verified: {TABLE_NAME}")
    print(f"Columns: {len(actual_columns)}")
    print("Engine: InnoDB")
    print("No source rows have been loaded yet.")
finally:
    conn.close()
