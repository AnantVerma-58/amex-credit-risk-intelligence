from pathlib import Path
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import pymysql
from dotenv import load_dotenv
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODEL_PATH = ROOT / "models/experiments/remove_over_99pct_missing.txt"
VALIDATION_DIR = ROOT / "data/features/splits/validation"
OUT_DIR = ROOT / "reports/model_robustness"

QUERY_BATCH_SIZE = 1000

CATEGORICAL_FEATURES = {
    "latest__D_63", "latest__D_64", "latest__D_66",
    "latest__D_68", "latest__B_30", "latest__B_38",
    "latest__D_114", "latest__D_116", "latest__D_117",
    "latest__D_120", "latest__D_126",
}


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
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
    )


def load_history_metadata(customer_ids):
    """Fetch history metadata only for the requested validation customers."""
    conn = get_connection()
    records = []

    try:
        ids = list(customer_ids)

        with conn.cursor() as cur:
            for start in range(0, len(ids), QUERY_BATCH_SIZE):
                batch = ids[start:start + QUERY_BATCH_SIZE]
                placeholders = ",".join(["%s"] * len(batch))

                cur.execute(
                    f"""
                    SELECT
                        customer_ID,
                        statement_count,
                        last_statement_date
                    FROM customer_history_summary
                    WHERE customer_ID IN ({placeholders})
                    """,
                    batch,
                )
                records.extend(cur.fetchall())

                print(
                    f"\rHistory metadata fetched: "
                    f"{min(start + len(batch), len(ids)):,}/{len(ids):,}",
                    end="",
                    flush=True,
                )

        print()

    finally:
        conn.close()

    metadata = pd.DataFrame(
        records,
        columns=["customer_ID", "statement_count", "last_statement_date"],
    )

    if not metadata.empty:
        metadata["customer_ID"] = metadata["customer_ID"].astype(str)
        metadata["last_statement_date"] = pd.to_datetime(
            metadata["last_statement_date"], errors="raise"
        )

    return metadata


def calculate_metrics(group, group_name, dimension):
    y = group["target"].to_numpy(dtype=np.int8)
    scores = group["predicted_score"].to_numpy(dtype=float)

    result = {
        "dimension": dimension,
        "group": str(group_name),
        "customers": int(len(group)),
        "positive_count": int(y.sum()),
        "negative_count": int((y == 0).sum()),
        "positive_rate": float(y.mean()) if len(y) else np.nan,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "brier_score": np.nan,
    }

    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, scores))
        result["average_precision"] = float(
            average_precision_score(y, scores)
        )

    if len(y):
        result["brier_score"] = float(brier_score_loss(y, scores))

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading saved model...")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    features = model.feature_name()
    print(f"Model features: {len(features)}")

    print("Loading validation partition...")
    df = pd.read_parquet(VALIDATION_DIR)
    df["customer_ID"] = df["customer_ID"].astype(str)

    missing_features = set(features) - set(df.columns)
    if missing_features:
        raise ValueError(
            f"Missing model features: {sorted(missing_features)}"
        )

    X = df.loc[:, features].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")

    print("Scoring validation customers...")
    df["predicted_score"] = model.predict(X, num_threads=4)
    del X

    print("\nFetching statement-history metadata from MariaDB...")
    metadata = load_history_metadata(df["customer_ID"].unique())

    if metadata["customer_ID"].duplicated().any():
        raise ValueError("Duplicate customer IDs in history metadata.")

    df = df.merge(
        metadata,
        on="customer_ID",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int((df["_merge"] != "both").sum())
    print(f"Validation customers missing history metadata: {unmatched:,}")

    if unmatched:
        raise ValueError(
            "Some validation customers lack history metadata. "
            "Investigate before interpreting subgroup results."
        )

    df.drop(columns=["_merge"], inplace=True)

    if df["statement_count"].isna().any():
        raise ValueError("Missing statement counts detected.")
    if df["last_statement_date"].isna().any():
        raise ValueError("Missing last statement dates detected.")

    df["history_group"] = pd.cut(
        df["statement_count"],
        bins=[0, 3, 6, 9, 13],
        labels=["1-3 statements", "4-6 statements",
                "7-9 statements", "10-13 statements"],
        include_lowest=True,
    )

    day = df["last_statement_date"].dt.day
    df["statement_date_group"] = pd.cut(
        day,
        bins=[0, 10, 20, 31],
        labels=["Mar 1-10", "Mar 11-20", "Mar 21-31"],
        include_lowest=True,
    )

    results = []

    for group_name, group in df.groupby(
        "history_group", observed=True
    ):
        results.append(
            calculate_metrics(group, group_name, "history_length")
        )

    for group_name, group in df.groupby(
        "statement_date_group", observed=True
    ):
        results.append(
            calculate_metrics(group, group_name, "last_statement_date")
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT_DIR / "validation_robustness.csv", index=False)

    # Save counts separately so subgroup coverage is easy to review.
    history_counts = (
        df.groupby("history_group", observed=True)
        .agg(
            customers=("customer_ID", "size"),
            min_statements=("statement_count", "min"),
            max_statements=("statement_count", "max"),
        )
        .reset_index()
    )
    history_counts.to_csv(
        OUT_DIR / "history_group_counts.csv", index=False
    )

    date_counts = (
        df.groupby("statement_date_group", observed=True)
        .agg(
            customers=("customer_ID", "size"),
            earliest_date=("last_statement_date", "min"),
            latest_date=("last_statement_date", "max"),
        )
        .reset_index()
    )
    date_counts.to_csv(
        OUT_DIR / "statement_date_group_counts.csv", index=False
    )

    print("\nValidation robustness results:")
    print(result_df.round(5).to_string(index=False))

    print("\nHistory group counts:")
    print(history_counts.to_string(index=False))

    print("\nStatement-date group counts:")
    print(date_counts.to_string(index=False))

    print("\nSaved artifacts:")
    for path in sorted(OUT_DIR.iterdir()):
        print(f"  {path.name} ({path.stat().st_size:,} bytes)")

    print(
        "\nInterpretation caution: subgroup metrics reflect the "
        "competition-sampled validation data. These comparisons are "
        "diagnostic and do not establish fairness or production stability."
    )


if __name__ == "__main__":
    main()
