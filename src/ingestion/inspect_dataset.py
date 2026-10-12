
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = RAW / "train_data.csv"
LABEL_PATH = RAW / "train_labels.csv"

def inspect_monthly_file(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    sample = pd.read_csv(path, nrows=2000)

    result = {
        "file": path.name,
        "size_gb": round(path.stat().st_size / 1e9, 3),
        "sample_rows": len(sample),
        "sample_columns": len(sample.columns),
        "columns": sample.columns.tolist(),
        "sample_dtypes": {
            col: str(dtype)
            for col, dtype in sample.dtypes.items()
        },
        "sample_missing_pct": {
            col: round(float(pct * 100), 2)
            for col, pct in sample.isna().mean().items()
        },
    }

    if "customer_ID" in sample:
        result["sample_distinct_customers"] = int(
            sample["customer_ID"].nunique()
        )

    if "S_2" in sample:
        dates = pd.to_datetime(sample["S_2"], errors="coerce")
        result["sample_date_min"] = (
            str(dates.min()) if dates.notna().any() else None
        )
        result["sample_date_max"] = (
            str(dates.max()) if dates.notna().any() else None
        )
        result["sample_invalid_dates"] = int(dates.isna().sum())

    return result

def inspect_labels(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    labels = pd.read_csv(path)

    required = {"customer_ID", "target"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")

    return {
        "file": path.name,
        "rows": len(labels),
        "columns": labels.columns.tolist(),
        "duplicate_customer_ids": int(
            labels["customer_ID"].duplicated().sum()
        ),
        "missing_customer_ids": int(labels["customer_ID"].isna().sum()),
        "missing_targets": int(labels["target"].isna().sum()),
        "target_counts": {
            str(k): int(v)
            for k, v in labels["target"].value_counts(
                dropna=False
            ).items()
        },
        "target_values": sorted(
            labels["target"].dropna().unique().tolist()
        ),
    }

def main():
    inventory = {
        "monthly_data": inspect_monthly_file(TRAIN_PATH),
        "labels": inspect_labels(LABEL_PATH),
    }

    output = REPORTS / "data_inventory.json"
    output.write_text(
        json.dumps(inventory, indent=2, default=str),
        encoding="utf-8",
    )

    print(json.dumps(inventory, indent=2, default=str))
    print(f"\nSaved inventory to: {output}")

if __name__ == "__main__":
    main()
