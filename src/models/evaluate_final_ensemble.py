from pathlib import Path
import json
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import catboost
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

ROOT = PROJECT_ROOT
TEST_DIR = ROOT / "data/features/splits/test"
LIGHTGBM_PATH = ROOT / "models/experiments/remove_over_99pct_missing.txt"
CATBOOST_PATH = ROOT / "models/catboost_baseline.cbm"

OUT_DIR = ROOT / "reports/model_comparison"
OUT_JSON = OUT_DIR / "final_ensemble_test_metrics.json"

LIGHTGBM_WEIGHT = 0.70
CATBOOST_WEIGHT = 0.30

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


def calculate_metrics(y, probabilities):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)

    result = {
        "customers": int(len(y)),
        "positive_count": int(np.sum(y)),
        "positive_rate_sampled": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
    }
    result.update(amex_metrics(y, p))
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading frozen LightGBM and CatBoost models...")
    lgb_model = lgb.Booster(model_file=str(LIGHTGBM_PATH))
    cat_model = catboost.CatBoostClassifier()
    cat_model.load_model(str(CATBOOST_PATH))

    features = lgb_model.feature_name()
    cat_features = [c for c in features if c in CATEGORICAL_FEATURES]

    print(f"Shared features: {len(features)}")
    print(f"Frozen blend weights: LightGBM={LIGHTGBM_WEIGHT:.2f}, "
          f"CatBoost={CATBOOST_WEIGHT:.2f}")

    print("\nLoading held-out test partition...")
    test = pd.read_parquet(TEST_DIR)

    if test["customer_ID"].duplicated().any():
        raise ValueError("Duplicate customer IDs in test partition.")

    y = test["target"].to_numpy(dtype=np.int8)
    missing = set(features) - set(test.columns)
    if missing:
        raise ValueError(f"Test partition missing features: {sorted(missing)}")

    X_lgb = test.loc[:, features].copy()
    for col in cat_features:
        X_lgb[col] = X_lgb[col].astype("category")

    X_cat = test.loc[:, features].copy()
    for col in cat_features:
        X_cat[col] = (
            X_cat[col].astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    print("Scoring test data with LightGBM...")
    start = time.perf_counter()
    p_lgb = lgb_model.predict(X_lgb, num_threads=4)
    lgb_seconds = time.perf_counter() - start

    print("Scoring test data with CatBoost...")
    start = time.perf_counter()
    p_cat = cat_model.predict_proba(X_cat)[:, 1]
    cat_seconds = time.perf_counter() - start

    p_blend = LIGHTGBM_WEIGHT * p_lgb + CATBOOST_WEIGHT * p_cat

    metrics = calculate_metrics(y, p_blend)
    metrics.update({
        "model": "Frozen LightGBM-CatBoost ensemble",
        "lightgbm_weight": LIGHTGBM_WEIGHT,
        "catboost_weight": CATBOOST_WEIGHT,
        "lightgbm_model": str(LIGHTGBM_PATH.relative_to(ROOT)),
        "catboost_model": str(CATBOOST_PATH.relative_to(ROOT)),
        "feature_count": len(features),
        "lightgbm_scoring_seconds": round(lgb_seconds, 3),
        "catboost_scoring_seconds": round(cat_seconds, 3),
        "evaluation_partition": "held-out test",
        "selection_partition": "validation",
        "test_used_for_model_selection": False,
    })

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== FINAL HELD-OUT ENSEMBLE TEST METRICS ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved report: {OUT_JSON}")
    print(
        "\nThe blend weights were selected on validation and kept fixed. "
        "Do not tune them using these test results."
    )


if __name__ == "__main__":
    main()
