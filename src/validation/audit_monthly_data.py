
from pathlib import Path
from collections import Counter
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "raw" / "train_data.csv"
REPORT_PATH = ROOT / "reports" / "monthly_data_audit.json"

CHUNK_SIZE = 20_000

def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(CSV_PATH)

    expected_columns = None
    total_rows = 0
    invalid_ids = 0
    invalid_dates = 0
    missing_by_column = Counter()
    min_date = None
    max_date = None

    # Exact duplicate-key tracking. This set can consume memory,
    # so monitor RAM. If necessary, we'll move key checking to SQL.
    seen_keys = set()
    duplicate_keys = 0

    for chunk in pd.read_csv(
        CSV_PATH,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        if expected_columns is None:
            expected_columns = chunk.columns.tolist()
            if not {"customer_ID", "S_2"}.issubset(expected_columns):
                raise ValueError("Missing customer_ID or S_2")

        if chunk.columns.tolist() != expected_columns:
            raise ValueError("Column mismatch between chunks")

        total_rows += len(chunk)
        missing_by_column.update(
            chunk.isna().sum().to_dict()
        )

        invalid_ids += int(
            chunk["customer_ID"].isna().sum()
        )

        dates = pd.to_datetime(
            chunk["S_2"], errors="coerce"
        )
        invalid_dates += int(dates.isna().sum())

        valid_dates = dates.dropna()
        if not valid_dates.empty:
            chunk_min = valid_dates.min()
            chunk_max = valid_dates.max()

            min_date = (
                chunk_min if min_date is None
                else min(min_date, chunk_min)
            )
            max_date = (
                chunk_max if max_date is None
                else max(max_date, chunk_max)
            )

        # This is exact but potentially memory-intensive.
        # We'll provide a disk-backed alternative if needed.
        for customer_id, statement_date in zip(
            chunk["customer_ID"],
            chunk["S_2"],
        ):
            if pd.isna(customer_id) or pd.isna(statement_date):
                continue

            key = (customer_id, statement_date)
            if key in seen_keys:
                duplicate_keys += 1
            else:
                seen_keys.add(key)

        print(
            f"Processed {total_rows:,} rows; "
            f"distinct customer/date keys tracked: "
            f"{len(seen_keys):,}"
        )

    report = {
        "file": CSV_PATH.name,
        "file_size_gb": round(
            CSV_PATH.stat().st_size / 1e9, 3
        ),
        "rows": total_rows,
        "columns": len(expected_columns or []),
        "distinct_customer_date_keys": len(seen_keys),
        "duplicate_customer_date_rows": duplicate_keys,
        "missing_customer_ids": invalid_ids,
        "invalid_statement_dates": invalid_dates,
        "min_statement_date": (
            str(min_date.date()) if min_date is not None else None
        ),
        "max_statement_date": (
            str(max_date.date()) if max_date is not None else None
        ),
        "missing_counts": dict(missing_by_column),
        "missing_percent": {
            col: round(count * 100 / total_rows, 3)
            for col, count in missing_by_column.items()
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nAudit report: {REPORT_PATH}")
    print(json.dumps({
        key: value for key, value in report.items()
        if key != "missing_counts" and key != "missing_percent"
    }, indent=2))

if __name__ == "__main__":
    main()
