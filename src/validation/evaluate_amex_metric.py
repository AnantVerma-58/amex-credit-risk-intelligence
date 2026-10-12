
from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "lightgbm_baseline.txt"
VALID_DIR = ROOT / "data" / "features" / "splits" / "validation"
REPORT_DIR = ROOT / "reports"

CATEGORICAL_SOURCE_FEATURES = {
    "B_30", "B_38", "D_66", "D_68",
    "D_114", "D_116", "D_117", "D_120", "D_126",
}


def is_categorical_feature(col):
    source_name = col.rsplit("__", 1)[-1]
    return (
        col.endswith("__D_63")
        or col.endswith("__D_64")
        or source_name in CATEGORICAL_SOURCE_FEATURES
    )


def gini_coefficient(y_true, y_score):
    order = np.argsort(-y_score, kind="mergesort")
    sorted_y = y_true[order]

    if sorted_y.sum() == 0:
        raise ValueError("Gini requires at least one positive label")

    n = len(sorted_y)
    cumulative_positives = np.cumsum(sorted_y)

    return (
        cumulative_positives.sum() / sorted_y.sum()
        - (n + 1) / 2
    ) / n


def amex_metric(y_true, y_score):
    # Normalized Gini
    normalized_gini = (
        gini_coefficient(y_true, y_score)
        / gini_coefficient(y_true, y_true.astype(float))
    )

    # Fraction of all positive labels captured in the highest-risk 4%.
    top_count = max(1, int(0.04 * len(y_true)))
    top_indices = np.argsort(-y_score, kind="mergesort")[:top_count]
    defaults_captured = y_true[top_indices].sum() / y_true.sum()

    score = 0.5 * (normalized_gini + defaults_captured)

    return normalized_gini, defaults_captured, score


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    print("Loading saved LightGBM model...")
    booster = lgb.Booster(model_file=str(MODEL_PATH))

    print("Loading validation partition...")
    valid = pd.read_parquet(VALID_DIR)

    y_valid = valid["target"].to_numpy(dtype=np.int8)
    X_valid = valid.drop(columns=["target", "customer_ID"])

    categorical_cols = [
        col for col in X_valid.columns
        if is_categorical_feature(col)
    ]

    # Reuse the categorical levels saved with the trained model.
    saved_categories = getattr(booster, "pandas_categorical", None)

    if saved_categories is None or len(saved_categories) != len(categorical_cols):
        raise RuntimeError(
            "Could not recover the model's categorical mappings. "
            "Do not evaluate until the training category mappings "
            "have been recovered."
        )

    for col, categories in zip(categorical_cols, saved_categories):
        X_valid[col] = pd.Categorical(
            X_valid[col],
            categories=categories,
        )

    print("Generating validation predictions...")
    predictions = booster.predict(
        X_valid,
        num_threads=4,
    )

    roc_auc = roc_auc_score(y_valid, predictions)
    average_precision = average_precision_score(y_valid, predictions)
    normalized_gini, captured, score = amex_metric(
        y_valid.astype(float),
        predictions,
    )

    print("\n===== VALIDATION METRICS =====")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"Average Precision: {average_precision:.6f}")
    print(f"Normalized Gini: {normalized_gini:.6f}")
    print(f"Defaults captured in top 4%: {captured:.4%}")
    print(f"AMEX metric: {score:.6f}")

    # Save feature importance for review.
    importance = pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)

    importance_path = REPORT_DIR / "lightgbm_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    print("\nTop 20 features by gain:")
    print(importance.head(20).to_string(index=False))

    metrics = {
        "model": "LightGBM baseline",
        "partition": "validation",
        "rows": int(len(y_valid)),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "normalized_gini": float(normalized_gini),
        "defaults_captured_top_4pct": float(captured),
        "amex_metric": float(score),
        "top_4pct_customer_count": int(max(1, int(0.04 * len(y_valid)))),
        "note": (
            "Random customer-level split; full-history summary features; "
            "not an out-of-time early-warning evaluation."
        ),
    }

    metrics_path = REPORT_DIR / "amex_validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved metrics: {metrics_path.relative_to(ROOT)}")
    print(f"Saved feature importance: {importance_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
