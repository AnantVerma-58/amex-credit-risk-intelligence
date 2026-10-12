
from pathlib import Path
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pymysql
from pymysql.cursors import SSCursor
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "features" / "splits"
BATCH_SIZE = 5_000
SPLITS = ("train", "validation", "test")

load_dotenv(ROOT / ".env")


def connect():
    return pymysql.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_NAME"],
        charset="utf8mb4",
        cursorclass=SSCursor,
        autocommit=True,
    )


def build_arrow_schema(columns):
    fields = []

    for name in columns:
        if name == "customer_ID":
            dtype = pa.string()
        elif name == "target":
            dtype = pa.int8()
        elif name.endswith("__D_63") or name.endswith("__D_64"):
            dtype = pa.string()
        else:
            dtype = pa.float32()

        fields.append(pa.field(name, dtype))

    return pa.schema(fields)


def export_split(conn, split_name):
    output_dir = OUTPUT_DIR / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_parts = list(output_dir.glob("part-*.parquet"))
    if existing_parts:
        raise RuntimeError(
            f"{output_dir} already contains Parquet parts. "
            "Move or remove them before rerunning this export."
        )

    query = """
        SELECT s.target, f.*
        FROM model_customer_split AS s
        JOIN customer_features_v2 AS f
            ON f.customer_ID = s.customer_ID
        WHERE s.split = %s
        ORDER BY f.customer_ID
    """

    cursor = conn.cursor()
    part_number = 0
    exported_rows = 0

    try:
        print(f"\nExporting {split_name}...")

        cursor.execute(query, (split_name,))
        columns = [item[0] for item in cursor.description]
        schema = build_arrow_schema(columns)

        while True:
            rows = cursor.fetchmany(BATCH_SIZE)

            if not rows:
                break

            frame = pd.DataFrame.from_records(
                rows,
                columns=columns,
            )

            arrays = [
                pa.array(
                    frame[column],
                    type=schema.field(column).type,
                    from_pandas=True,
                    safe=False,
                )
                for column in columns
            ]

            table = pa.Table.from_arrays(arrays, schema=schema)

            output_file = (
                output_dir / f"part-{part_number:05d}.parquet"
            )

            pq.write_table(
                table,
                output_file,
                compression="snappy",
            )

            exported_rows += len(frame)
            part_number += 1

            if part_number % 10 == 0:
                print(
                    f"  Exported {exported_rows:,} customers "
                    f"across {part_number:,} files"
                )

            del table, arrays, frame, rows

    finally:
        cursor.close()

    print(
        f"Finished {split_name}: "
        f"{exported_rows:,} customers, "
        f"{part_number:,} Parquet files"
    )

    return exported_rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect()

    try:
        totals = {}

        for split_name in SPLITS:
            totals[split_name] = export_split(conn, split_name)

        expected = {
            "train": 321_238,
            "validation": 68_837,
            "test": 68_838,
        }

        print("\nExport validation:")
        for split_name in SPLITS:
            actual = totals[split_name]
            print(
                f"  {split_name:10s}: "
                f"{actual:,} rows "
                f"(expected {expected[split_name]:,})"
            )

            if actual != expected[split_name]:
                raise RuntimeError(
                    f"{split_name} row count mismatch"
                )

        print("\nAll split exports passed row-count validation.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
