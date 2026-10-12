from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models/experiments/remove_over_99pct_missing.txt"
VALIDATION_DIR = ROOT / "data/features/splits/validation"
OUT_DIR = ROOT / "reports/model_interpretability"

SAMPLE_SIZE = 5000
SEED = 42

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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading saved model...")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    feature_names = model.feature_name()
    print(f"Model features: {len(feature_names)}")

    print("Loading validation partition...")
    df = pd.read_parquet(VALIDATION_DIR)

    missing = set(feature_names) - set(df.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")

    # Preserve the exact feature order used by the model.
    X = df.loc[:, feature_names].copy()

    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")

    # 1. Native LightGBM feature importance.
    print("Calculating LightGBM feature importance...")
    importance = pd.DataFrame({
        "feature": feature_names,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    })

    importance = importance.sort_values(
        "gain", ascending=False
    ).reset_index(drop=True)

    importance.insert(0, "gain_rank", np.arange(1, len(importance) + 1))
    importance.to_csv(
        OUT_DIR / "lightgbm_feature_importance.csv", index=False
    )

    print("\nTop 20 features by gain:")
    print(importance[["gain_rank", "feature", "gain", "split"]].head(20).to_string(index=False))

    # SHAP runs on a reproducible sample of validation rows.
    sample_size = min(SAMPLE_SIZE, len(X))
    X_sample = X.sample(
        n=sample_size, random_state=SEED
    ).copy()

    print(f"\nCalculating SHAP values for {sample_size:,} validation rows...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Some SHAP versions return a list for binary classification.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    # Handle Explanation objects from newer SHAP versions.
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    shap_values = np.asarray(shap_values)

    # Ensure a 2D matrix for binary-class feature attributions.
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    if shap_values.shape != X_sample.shape:
        raise ValueError(
            f"Unexpected SHAP shape {shap_values.shape}; "
            f"expected {X_sample.shape}"
        )

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_importance = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values(
        "mean_abs_shap", ascending=False
    ).reset_index(drop=True)

    shap_importance.insert(
        0, "shap_rank", np.arange(1, len(shap_importance) + 1)
    )

    shap_importance.to_csv(
        OUT_DIR / "shap_feature_importance.csv", index=False
    )

    print("\nTop 20 features by mean absolute SHAP value:")
    print(shap_importance.head(20).to_string(index=False))

    print("\nCreating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=25,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "shap_summary.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close("all")

    # Save metadata for reproducibility.
    metadata = {
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "validation_rows": int(len(df)),
        "shap_sample_rows": int(sample_size),
        "feature_count": int(len(feature_names)),
        "random_seed": SEED,
        "explanation_note": (
            "SHAP describes model attribution, not causality. "
            "SHAP values are calculated on a validation sample."
        ),
    }

    import json
    with (OUT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nInterpretability artifacts saved in:")
    print(OUT_DIR.relative_to(ROOT))
    for path in sorted(OUT_DIR.iterdir()):
        print(f"  {path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()