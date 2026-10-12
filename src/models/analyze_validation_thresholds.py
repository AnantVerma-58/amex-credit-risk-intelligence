from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/experiments/remove_over_99pct_missing.txt"
VALIDATION_DIR = ROOT / "data/features/splits/validation"
OUT_PATH = ROOT / "reports/model_interpretability/validation_thresholds.csv"

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]

def main():
    print("Loading model and validation data...")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    df = pd.read_parquet(VALIDATION_DIR)

    features = model.feature_name()
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")

    X = df.loc[:, features].copy()
    categorical = {
        "latest__D_63", "latest__D_64", "latest__D_66",
        "latest__D_68", "latest__B_30", "latest__B_38",
        "latest__D_114", "latest__D_116", "latest__D_117",
        "latest__D_120", "latest__D_126",
    }
    for col in categorical:
        if col in X.columns:
            X[col] = X[col].astype("category")

    y = df["target"].to_numpy(dtype=np.int8)
    scores = model.predict(X, num_threads=4)

    rows = []
    for threshold in THRESHOLDS:
        pred = (scores >= threshold).astype(np.int8)
        tn, fp, fn, tp = confusion_matrix(
            y, pred, labels=[0, 1]
        ).ravel()

        rows.append({
            "threshold": threshold,
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "flagged_customers": int(pred.sum()),
            "flagged_pct": float(pred.mean() * 100),
        })

    results = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_PATH, index=False)

    print("\nValidation threshold comparison:")
    print(results.round(4).to_string(index=False))
    print(f"\nSaved: {OUT_PATH.relative_to(ROOT)}")
    print(
        "\nCaution: these thresholds reflect the competition's sampled "
        "class distribution and are not production decision thresholds."
    )

if __name__ == "__main__":
    main()