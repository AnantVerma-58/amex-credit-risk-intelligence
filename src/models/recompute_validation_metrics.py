from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.models.amex_metrics import amex_metrics

REPORT_DIR = ROOT / "reports/model_comparison"
PREDICTIONS = REPORT_DIR / "validation_predictions.parquet"


def evaluate(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    result = {
        "customers": len(y),
        "positive_count": int(np.sum(y)),
        "positive_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
    }
    result.update(amex_metrics(y, p))
    return result


def main():
    df = pd.read_parquet(PREDICTIONS)
    y = df["target"].to_numpy(dtype=np.int8)

    model_columns = {
        "Logistic Regression": "logistic_regression",
        "LightGBM": "lightgbm",
        "CatBoost": "catboost",
    }

    predictions = {}
    rows = []

    for name, col in model_columns.items():
        predictions[name] = df[col].to_numpy(dtype=float)
        row = evaluate(y, predictions[name])
        row["model"] = name
        rows.append(row)

    for weight_lgb in np.round(np.arange(0, 1.01, 0.1), 2):
        weight_cat = 1.0 - weight_lgb
        p = (
            weight_lgb * predictions["LightGBM"]
            + weight_cat * predictions["CatBoost"]
        )
        row = evaluate(y, p)
        row["model"] = (
            "CatBoost only" if weight_lgb == 0 else
            "LightGBM only" if weight_lgb == 1 else
            f"Blend {weight_lgb:.1f}/{weight_cat:.1f}"
        )
        row["lightgbm_weight"] = weight_lgb
        row["catboost_weight"] = weight_cat
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(
        REPORT_DIR / "validation_metrics_recomputed.csv",
        index=False,
    )

    print("Recomputed validation metrics, sorted by AMEX:")
    print(
        result.sort_values("amex_metric", ascending=False)
        .round(6)
        .to_string(index=False)
    )
    print(
        f"\nSaved: {REPORT_DIR / 'validation_metrics_recomputed.csv'}"
    )


if __name__ == "__main__":
    main()
