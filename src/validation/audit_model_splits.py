
from pathlib import Path
from collections import Counter

import pyarrow.parquet as pq
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "features" / "splits"

EXPECTED_ROWS = {
    "train": 321_238,
    "validation": 68_837,
    "test": 68_838,
}

EXPECTED_COLUMNS = 367  # target + customer_ID + 365 features


def audit_split(split_name, expected_rows):
    folder = SPLIT_DIR / split_name
    files = sorted(folder.glob("part-*.parquet"))

    if not files:
        raise RuntimeError(f"No Parquet files found in {folder}")

    total_rows = 0
    target_counts = Counter()
    null_counts = Counter()
    reference_schema = None
    column_names = None

    for file in files:
        parquet_file = pq.ParquetFile(file)
        schema = parquet_file.schema_arrow

        if reference_schema is None:
            reference_schema = schema
            column_names = schema.names
        elif not schema.equals(reference_schema, check_metadata=False):
            raise RuntimeError(
                f"Inconsistent schema found in {file.name}"
            )

        table = parquet_file.read()
        rows = table.num_rows
        total_rows += rows

        if "target" not in table.column_names:
            raise RuntimeError(f"Missing target in {file.name}")

        if "customer_ID" not in table.column_names:
            raise RuntimeError(f"Missing customer_ID in {file.name}")

        targets = table.column("target").to_pylist()
        target_counts.update(targets)

        for name in table.column_names:
            null_counts[name] += table.column(name).null_count

        del table

    print(f"\n{'=' * 55}")
    print(f"Split: {split_name}")
    print(f"{'=' * 55}")
    print(f"Files: {len(files)}")
    print(f"Rows: {total_rows:,}")
    print(f"Columns: {len(column_names)}")
    print(f"Target counts: {dict(sorted(target_counts.items()))}")

    if total_rows != expected_rows:
        raise RuntimeError(
            f"{split_name}: expected {expected_rows:,} rows, "
            f"found {total_rows:,}"
        )

    if len(column_names) != EXPECTED_COLUMNS:
        raise RuntimeError(
            f"Expected {EXPECTED_COLUMNS} columns, "
            f"found {len(column_names)}"
        )

    if set(target_counts) != {0, 1}:
        raise RuntimeError(
            f"Unexpected target values: {dict(target_counts)}"
        )

    if null_counts["customer_ID"] > 0:
        raise RuntimeError("Missing customer IDs detected")

    if null_counts["target"] > 0:
        raise RuntimeError("Missing targets detected")

    missingness = pd.Series(
        {
            name: null_counts[name] / total_rows * 100
            for name in column_names
            if name not in ("customer_ID", "target")
        },
        name="missing_pct",
    ).sort_values(ascending=False)

    print("\nTop 15 features by missingness:")
    print(missingness.head(15).to_string(float_format=lambda x: f"{x:.2f}%"))

    all_null = missingness[missingness >= 100]

    if not all_null.empty:
        print("\nWARNING: All-null features:")
        print(all_null.to_string())

    print("\nSchema:")
    print(reference_schema)

    return total_rows


def main():
    total = 0

    for split_name, expected_rows in EXPECTED_ROWS.items():
        total += audit_split(split_name, expected_rows)

    print(f"\n{'=' * 55}")
    print(f"TOTAL AUDITED ROWS: {total:,}")
    print("All partition row counts and schemas passed.")
    print("Review the missingness reports before modeling.")


if __name__ == "__main__":
    main()
