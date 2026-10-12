from pathlib import Path
import json

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/experiments/remove_over_99pct_missing.txt"
VALIDATION_DIR = ROOT / "data/features/splits/validation"
OUT_DIR = ROOT / "reports/model_calibration"

N_BINS = 10
SEED = 42

CATEGORICAL_FEATURES = {
    "latest__D_63", "latest__D_64", "latest__D_66",
    "latest__D_68", "latest__B_30", "latest__B_38",
    "latest__D_114", "latest__D_116", "latest__D_117",
    "latest__D_120", "latest__D_126",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading saved model...")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    features = model.feature_name()

    print("Loading validation partition...")
    df = pd.read_parquet(VALIDATION_DIR)

    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")

    X = df.loc[:, features].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")

    y = df["target"].to_numpy(dtype=np.int8)
    scores = np.asarray(model.predict(X, num_threads=4), dtype=float)

    if not np.isfinite(scores).all():
        raise ValueError("Predictions contain non-finite values.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected probability predictions in [0, 1].")

    brier = brier_score_loss(y, scores)
    loss = log_loss(y, scores, labels=[0, 1])

    # Equal-frequency bins: each bin has roughly the same number of rows.
    reliability = pd.DataFrame({
        "target": y,
        "predicted_score": scores,
    })
    reliability["bin"] = pd.qcut(
        reliability["predicted_score"],
        q=N_BINS,
        labels=False,
        duplicates="drop",
    )

    table = (
        reliability.groupby("bin", observed=True)
        .agg(
            customer_count=("target", "size"),
            mean_predicted_score=("predicted_score", "mean"),
            observed_positive_rate=("target", "mean"),
            minimum_score=("predicted_score", "min"),
            maximum_score=("predicted_score", "max"),
        )
        .reset_index()
    )
    table["calibration_gap"] = (
        table["observed_positive_rate"] - table["mean_predicted_score"]
    )

    table.to_csv(OUT_DIR / "reliability_table.csv", index=False)

    # Reliability curve.
    observed, predicted = calibration_curve(
        y, scores, n_bins=N_BINS, strategy="quantile"
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(
        predicted, observed, marker="o",
        label="Model (validation)"
    )
    ax.plot(
        [0, 1], [0, 1], linestyle="--",
        label="Perfect calibration"
    )
    ax.set_xlabel("Mean predicted score in bin")
    ax.set_ylabel("Observed positive-label rate")
    ax.set_title("LightGBM Reliability Curve — Validation Set")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "reliability_curve.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    metrics = {
        "evaluation": "validation_probability_calibration",
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "validation_rows": int(len(y)),
        "feature_count": int(len(features)),
        "observed_positive_rate": float(y.mean()),
        "mean_predicted_score": float(scores.mean()),
        "brier_score": float(brier),
        "log_loss": float(loss),
        "reliability_bins": int(len(table)),
        "binning": "quantile / approximately equal-frequency",
        "random_seed": SEED,
        "interpretation_note": (
            "These calibration metrics reflect the competition-sampled "
            "validation distribution. The negative class was sampled, "
            "so scores are not automatically real-world default "
            "probabilities. Calibration against a representative "
            "portfolio population may be required."
        ),
    }

    with (OUT_DIR / "calibration_metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    print("\nCalibration metrics")
    print("=" * 55)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print("\nReliability table")
    print(table.round(5).to_string(index=False))

    print("\nSaved artifacts:")
    for path in sorted(OUT_DIR.iterdir()):
        print(f"  {path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()