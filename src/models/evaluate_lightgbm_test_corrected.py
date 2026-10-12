from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.models.amex_metrics import amex_metrics

MODEL_PATH = ROOT / "models/experiments/remove_over_99pct_missing.txt"
TEST_DIR = ROOT / "data/features/splits/test"
OUT_PATH = ROOT / "reports/model_comparison/lightgbm_test_metrics_corrected.json"

CATEGORICAL_FEATURES = {
    "latest__D_63", "latest__D_64", "latest__D_66",
    "latest__D_68", "latest__B_30", "latest__B_38",
    "latest__D_114", "latest__D_116", "latest__D_117",
    "latest__D_120", "latest__D_126",
}


def main():
    model = lgb.Booster(model_file=str(MODEL_PATH))
    features = model.feature_name()

    print("Loading held-out test partition...")
    test = pd.read_parquet(TEST_DIR)
    y = test["target"].to_numpy(dtype=np.int8)

    X = test.loc[:, features].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")

    print("Scoring LightGBM...")
    probabilities = np.clip(
        model.predict(X, num_threads=4), 1e-7, 1 - 1e-7
    )

    metrics = {
        "model": "LightGBM",
        "customers": int(len(y)),
        "positive_count": int(y.sum()),
        "positive_rate_sampled": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(
            average_precision_score(y, probabilities)
        ),
        "log_loss": float(log_loss(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "feature_count": len(features),
        "evaluation_partition": "held-out test",
        "test_used_for_model_selection": False,
    }
    metrics.update(amex_metrics(y, probabilities))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== CORRECTED LIGHTGBM TEST METRICS ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
