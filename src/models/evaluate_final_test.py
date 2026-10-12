from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models/experiments/remove_over_99pct_missing.txt"
)
TEST_DIR = PROJECT_ROOT / "data/features/splits/test"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "final_test_metrics.json"

# These features were treated as categorical during training.
CATEGORICAL_FEATURES = {
    "latest__D_63",
    "latest__D_64",
    "latest__D_66",
    "latest__D_68",
    "latest__B_30",
    "latest__B_38",
    "latest__D_114",
    "latest__D_116",
    "latest__D_117",
    "latest__D_120",
    "latest__D_126",
}


def amex_metrics(y_true, scores):
    """Calculate the weighted AMEX competition metric."""
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)

    if len(y_true) != len(scores):
        raise ValueError("Labels and scores have different lengths.")

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        raise ValueError("Both target classes must be present.")

    def weighted_gini(order):
        labels_sorted = y_true[order]
        weights = np.where(labels_sorted == 0, 20.0, 1.0)

        weight_random = np.cumsum(weights / weights.sum())
        total_positive_weight = np.sum(labels_sorted * weights)
        cumulative_positive_weight = np.cumsum(
            labels_sorted * weights
        )
        lorentz = cumulative_positive_weight / total_positive_weight

        return np.sum((lorentz - weight_random) * weights)

    prediction_order = np.argsort(-scores, kind="mergesort")
    perfect_order = np.argsort(-y_true, kind="mergesort")

    normalized_gini = (
        weighted_gini(prediction_order)
        / weighted_gini(perfect_order)
    )

    labels_sorted = y_true[prediction_order]
    weights_sorted = np.where(labels_sorted == 0, 20, 1)

    cutoff = int(0.04 * weights_sorted.sum())
    within_cutoff = np.cumsum(weights_sorted) <= cutoff

    defaults_captured = (
        labels_sorted[within_cutoff].sum() / y_true.sum()
    )

    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(
            average_precision_score(y_true, scores)
        ),
        "log_loss": float(log_loss(y_true, scores)),
        "normalized_gini": float(normalized_gini),
        "defaults_captured_top_4pct": float(defaults_captured),
        "amex_metric": float(
            0.5 * (normalized_gini + defaults_captured)
        ),
        "customers_in_weighted_top_4pct": int(
            within_cutoff.sum()
        ),
    }


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    test_files = sorted(TEST_DIR.glob("*.parquet"))
    if not test_files:
        raise FileNotFoundError(
            f"No Parquet files found in {TEST_DIR}"
        )

    print("Loading saved LightGBM model...")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    feature_names = model.feature_name()

    print(f"Model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Expected features: {len(feature_names)}")
    print(f"Test Parquet files: {len(test_files)}")

    missing_categoricals = (
        CATEGORICAL_FEATURES - set(feature_names)
    )
    if missing_categoricals:
        raise ValueError(
            "Categorical features missing from model: "
            f"{sorted(missing_categoricals)}"
        )

    all_labels = []
    all_scores = []
    total_rows = 0

    print("\nScoring test data in file-sized chunks...")

    for i, file_path in enumerate(test_files, start=1):
        df = pd.read_parquet(file_path)

        if "target" not in df.columns:
            raise ValueError(f"'target' column missing in {file_path}")

        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"Missing model features in {file_path}: "
                f"{sorted(missing_features)}"
            )

        y = df["target"].to_numpy(dtype=np.int8)
        X = df.loc[:, feature_names].copy()

        # Restore categorical dtypes used when training the model.
        for col in CATEGORICAL_FEATURES:
            X[col] = X[col].astype("category")

        predictions = model.predict(X, num_threads=4)

        if not np.isfinite(predictions).all():
            raise ValueError(
                f"Non-finite predictions found in {file_path}"
            )

        all_labels.append(y)
        all_scores.append(np.asarray(predictions))
        total_rows += len(df)

        print(
            f"[{i:02d}/{len(test_files):02d}] "
            f"{file_path.name}: {len(df):,} rows"
        )

        del X, y, predictions, df

    y_test = np.concatenate(all_labels)
    y_scores = np.concatenate(all_scores)

    results = amex_metrics(y_test, y_scores)

    results.update({
        "evaluation": "final_test",
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "feature_count": len(feature_names),
        "test_rows": int(total_rows),
        "target_0_count": int((y_test == 0).sum()),
        "target_1_count": int((y_test == 1).sum()),
        "target_1_fraction": float(y_test.mean()),
        "test_files": len(test_files),
        "random_seed": 42,
        "metric_note": (
            "AMEX weighted metric: target=0 receives weight 20; "
            "target=1 receives weight 1. Test set was not used "
            "to select features or tune model parameters."
        ),
    })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print(f"\nSaved report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
