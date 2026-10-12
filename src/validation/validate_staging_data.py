
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
)

try:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) AS row_count,
                COUNT(DISTINCT customer_ID) AS customer_count,
                SUM(customer_ID IS NULL) AS null_customer_ids,
                SUM(S_2 IS NULL) AS null_statement_dates,
                MIN(S_2) AS min_statement_date,
                MAX(S_2) AS max_statement_date
            FROM staging_monthly_raw
        """)
        row = cur.fetchone()

        names = [item[0] for item in cur.description]
        result = dict(zip(names, row))

        print("Staging data validation")
        print("-" * 40)
        for key, value in result.items():
            print(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")

        cur.execute("""
            SELECT COUNT(*)
            FROM staging_monthly_raw
            WHERE D_63 IS NOT NULL
              AND D_63 NOT IN ('CR', 'CO', 'CL', 'XZ', 'XM')
        """)
        print("D_63 values outside common categories:", cur.fetchone()[0])

        cur.execute("""
            SELECT COUNT(*)
            FROM staging_monthly_raw
            WHERE S_2 < '2017-03-01'
               OR S_2 > '2018-03-31'
        """)
        print("Dates outside audited range:", cur.fetchone()[0])

        assert result["row_count"] == 5_531_451, "Row-count mismatch"
        assert result["null_customer_ids"] == 0, "Null customer IDs found"
        assert result["null_statement_dates"] == 0, "Null statement dates found"
        assert str(result["min_statement_date"]) == "2017-03-01"
        assert str(result["max_statement_date"]) == "2018-03-31"

        print("\nCore staging checks passed.")
finally:
    conn.close()
