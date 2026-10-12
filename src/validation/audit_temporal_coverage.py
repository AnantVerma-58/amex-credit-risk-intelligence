from pathlib import Path
import json
import os

import pandas as pd
import pymysql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

OUT_DIR = ROOT / "reports" / "temporal_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_connection():
    required = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing .env settings: {missing}")

    return pymysql.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_NAME"],
        cursorclass=pymysql.cursors.Cursor,
        connect_timeout=10,
    )


def query_df(conn, sql):
    """Fetch query results using a regular tuple cursor."""
    with conn.cursor() as cursor:
        cursor.execute(sql)
        columns = [item[0] for item in cursor.description]
        rows = cursor.fetchall()

    return pd.DataFrame(rows, columns=columns)


def main():
    conn = get_connection()

    try:
        print("Checking customer history table schema...")
        schema = query_df(conn, "SHOW COLUMNS FROM customer_history_summary")
        print(schema[["Field", "Type"]].to_string(index=False))

        history_columns = set(schema["Field"].astype(str))
        required_history_columns = {
            "customer_ID",
            "first_statement_date",
            "last_statement_date",
            "statement_count",
        }

        missing = required_history_columns - history_columns
        if missing:
            raise RuntimeError(
                "The history table does not contain the expected columns: "
                f"{sorted(missing)}. Review the schema printed above."
            )

        print("\nAuditing monthly statement coverage...")

        monthly = query_df(
            conn,
            """
            SELECT
                DATE_FORMAT(S_2, '%Y-%m') AS statement_month,
                COUNT(*) AS statement_rows,
                COUNT(DISTINCT customer_ID) AS distinct_customers
            FROM staging_monthly_raw_v2
            GROUP BY DATE_FORMAT(S_2, '%Y-%m')
            ORDER BY statement_month
            """,
        )

        monthly.to_csv(
            OUT_DIR / "monthly_statement_coverage.csv", index=False
        )

        print("\nStatement coverage by month:")
        print(monthly.to_string(index=False))

        print("\nAuditing customer history endpoints...")

        history = query_df(
            conn,
            """
            SELECT
                h.customer_ID,
                h.first_statement_date,
                h.last_statement_date,
                h.statement_count,
                l.target
            FROM customer_history_summary AS h
            INNER JOIN customer_labels AS l
                ON h.customer_ID = l.customer_ID
            """,
        )

        # Convert database dates only after fetching the actual result rows.
        history["first_statement_date"] = pd.to_datetime(
            history["first_statement_date"], errors="raise"
        )
        history["last_statement_date"] = pd.to_datetime(
            history["last_statement_date"], errors="raise"
        )

        history["last_statement_month"] = (
            history["last_statement_date"].dt.to_period("M").astype(str)
        )

        endpoint_summary = (
            history.groupby("last_statement_month")
            .agg(
                customers=("customer_ID", "nunique"),
                mean_statement_count=("statement_count", "mean"),
                target_1_rate=("target", "mean"),
            )
            .reset_index()
        )

        endpoint_summary.to_csv(
            OUT_DIR / "last_statement_month_summary.csv", index=False
        )

        print("\nCustomer counts by last statement month:")
        print(endpoint_summary.round(4).to_string(index=False))

        overall = {
            "customers_with_labels_and_history": int(len(history)),
            "earliest_first_statement": str(
                history["first_statement_date"].min().date()
            ),
            "latest_last_statement": str(
                history["last_statement_date"].max().date()
            ),
            "median_statement_count": float(
                history["statement_count"].median()
            ),
            "minimum_statement_count": int(
                history["statement_count"].min()
            ),
            "maximum_statement_count": int(
                history["statement_count"].max()
            ),
            "target_1_rate_in_joined_data": float(
                history["target"].mean()
            ),
        }

        with (OUT_DIR / "temporal_coverage_summary.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(overall, f, indent=2)

        print("\nOverall summary:")
        for key, value in overall.items():
            print(f"{key}: {value}")

        print("\nSaved reports:")
        print(OUT_DIR / "monthly_statement_coverage.csv")
        print(OUT_DIR / "last_statement_month_summary.csv")
        print(OUT_DIR / "temporal_coverage_summary.json")

    finally:
        conn.close()


if __name__ == "__main__":
    main()