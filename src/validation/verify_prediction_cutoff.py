from pathlib import Path
import json
import os
from decimal import Decimal
from datetime import date, datetime

import pymysql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

CUTOFF = "2018-03-31"
OUT_DIR = ROOT / "reports" / "temporal_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)



def json_safe(value):
    """Convert database values to JSON-compatible Python values."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def main():
    conn = pymysql.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_NAME"],
        cursorclass=pymysql.cursors.DictCursor,
    )

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_rows,
                    COUNT(DISTINCT customer_ID) AS customers,
                    MIN(S_2) AS earliest_statement,
                    MAX(S_2) AS latest_statement,
                    SUM(S_2 > %s) AS rows_after_cutoff,
                    SUM(S_2 = %s) AS rows_on_cutoff,
                    SUM(S_2 IS NULL) AS null_statement_dates
                FROM staging_monthly_raw_v2
                """,
                (CUTOFF, CUTOFF),
            )
            result = cur.fetchone()

            cur.execute(
                """
                SELECT
                    DATE(S_2) AS last_statement_date,
                    COUNT(*) AS customers
                FROM staging_monthly_raw_v2 s
                INNER JOIN (
                    SELECT customer_ID, MAX(S_2) AS last_date
                    FROM staging_monthly_raw_v2
                    GROUP BY customer_ID
                ) h
                    ON s.customer_ID = h.customer_ID
                   AND s.S_2 = h.last_date
                GROUP BY DATE(S_2)
                ORDER BY DATE(S_2)
                """
            )
            endpoint_dates = cur.fetchall()

        result["cutoff"] = CUTOFF
        result["cutoff_passed"] = (
            result["rows_after_cutoff"] == 0
            and result["null_statement_dates"] == 0
        )

        with (OUT_DIR / "prediction_cutoff_audit.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "summary": {
                        k: json_safe(v) for k, v in result.items()
                    },
                    "customer_last_statement_dates": [
                        {
                            k: json_safe(v) for k, v in row.items()
                        }
                        for row in endpoint_dates
                    ],
                },
                f,
                indent=2,
            )

        print("Prediction cutoff audit")
        print("=" * 45)
        for key, value in result.items():
            print(f"{key}: {value}")

        print("\nCustomer last-statement dates:")
        for row in endpoint_dates:
            print(f"  {row['last_statement_date']}: {row['customers']:,}")

        print("\nSaved: reports/temporal_audit/prediction_cutoff_audit.json")

    finally:
        conn.close()


if __name__ == "__main__":
    main()