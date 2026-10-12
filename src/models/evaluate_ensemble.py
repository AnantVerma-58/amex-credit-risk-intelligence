from pathlib import Path
import sys

# Make the project root importable during direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.models.amex_metrics import amex_metrics

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = (
    ROOT / "reports/model_comparison/validation_predictions.parquet"
)
OUTPUT = ROOT / "reports/model_comparison/ensemble_comparison.csv"


def evaluate(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    result = {
        "roc_auc": roc_auc_score(y, p),
        "average_precision": average_precision_score(y, p),
        "log_loss": log_loss(y, p),
        "brier_score": brier_score_loss(y, p),
    }
    result.update(amex_metrics(y, p))
    return result


def main():
    df = pd.read_parquet(PREDICTIONS)
    y = df["target"].to_numpy(dtype=np.int8)

    rows = []
    for weight_lgb in np.round(np.arange(0, 1.01, 0.1), 2):
        weight_cat = 1.0 - weight_lgb

        p = (
            weight_lgb * df["lightgbm"].to_numpy()
            + weight_cat * df["catboost"].to_numpy()
        )

        row = evaluate(y, p)
        row["lightgbm_weight"] = weight_lgb
        row["catboost_weight"] = weight_cat
        row["model"] = (
            "CatBoost only" if weight_lgb == 0 else
            "LightGBM only" if weight_lgb == 1 else
            "Blended ensemble"
        )
        rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        "amex_metric", ascending=False
    )
    results.to_csv(OUTPUT, index=False)

    print("Validation ensemble comparison (sorted by AMEX metric):")
    print(results.round(6).to_string(index=False))
    print(f"\nSaved: {OUTPUT}")
    print(
        "\nThese blend weights are exploratory and selected on validation. "
        "Do not treat the best validation blend as an unbiased final result."
    )


if __name__ == "__main__":
    main()
