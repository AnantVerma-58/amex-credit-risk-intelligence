from pathlib import Path
import json

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models/experiments/remove_over_99pct_missing.txt"
VALIDATION_DIR = ROOT / "data/features/splits/validation"
OUT_DIR = ROOT / "reports/model_interpretability/dependence"

SAMPLE_SIZE = 5000
SEED = 42

SELECTED_FEATURES = [
    "latest__P_2",
    "latest__B_1",
    "latest__B_4",
    "latest__D_39",
]

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

    print("Loading model...")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    feature_names = model.feature_name()

    missing_selected = set(SELECTED_FEATURES) - set(feature_names)
    if missing_selected:
        raise ValueError(
            f"Selected features are absent from the model: "
            f"{sorted(missing_selected)}"
        )

    print("Loading validation partition...")
    df = pd.read_parquet(VALIDATION_DIR)

    missing = set(feature_names) - set(df.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")

    X = df.loc[:, feature_names].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")

    X_sample = X.sample(
        n=min(SAMPLE_SIZE, len(X)),
        random_state=SEED,
    ).copy()

    print(f"SHAP sample rows: {len(X_sample):,}")
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_sample)

    if isinstance(values, list):
        values = values[-1]
    if hasattr(values, "values"):
        values = values.values

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]

    if values.shape != X_sample.shape:
        raise ValueError(
            f"Unexpected SHAP shape {values.shape}; "
            f"expected {X_sample.shape}"
        )

    shap_df = pd.DataFrame(
        values, columns=feature_names, index=X_sample.index
    )

    rows = []

    for feature in SELECTED_FEATURES:
        x = pd.to_numeric(X_sample[feature], errors="coerce")
        s = pd.to_numeric(shap_df[feature], errors="coerce")

        valid = x.notna() & s.notna()
        xv = x.loc[valid]
        sv = s.loc[valid]

        missing_pct = float(x.isna().mean() * 100)

        if len(xv) >= 3 and xv.nunique() > 1:
            corr = spearmanr(xv, sv)
            spearman_rho = float(corr.statistic)
            spearman_p = float(corr.pvalue)
        else:
            spearman_rho = None
            spearman_p = None

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.scatter(
            xv,
            sv,
            s=13,
            alpha=0.35,
            edgecolors="none",
        )
        ax.axhline(0, linestyle="--", linewidth=1)

        ax.set_title(
            f"SHAP dependence: {feature}\n"
            f"Validation sample n={len(X_sample):,}; "
            f"missing={missing_pct:.2f}%"
        )
        ax.set_xlabel(f"{feature} value")
        ax.set_ylabel(
            "SHAP contribution to model output\n"
            "(positive values push output upward)"
        )
        ax.grid(True, alpha=0.2)
        fig.tight_layout()

        output_path = OUT_DIR / f"{feature}_dependence.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        rows.append({
            "feature": feature,
            "sample_rows": int(len(X_sample)),
            "non_missing_rows": int(valid.sum()),
            "missing_pct": missing_pct,
            "feature_unique_values_non_missing": int(xv.nunique()),
            "feature_min": float(xv.min()) if len(xv) else None,
            "feature_median": float(xv.median()) if len(xv) else None,
            "feature_max": float(xv.max()) if len(xv) else None,
            "mean_shap": float(s.mean()),
            "mean_abs_shap": float(s.abs().mean()),
            "shap_min": float(s.min()),
            "shap_max": float(s.max()),
            "spearman_feature_vs_shap": spearman_rho,
            "spearman_p_value": spearman_p,
            "plot_file": output_path.name,
        })

        print(
            f"{feature}: valid={int(valid.sum()):,}, "
            f"missing={missing_pct:.2f}%, "
            f"mean_abs_SHAP={s.abs().mean():.5f}, "
            f"Spearman={spearman_rho}"
        )

    stats = pd.DataFrame(rows)
    stats.to_csv(OUT_DIR / "dependence_statistics.csv", index=False)

    metadata = {
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "validation_sample_rows": int(len(X_sample)),
        "random_seed": SEED,
        "selected_features": SELECTED_FEATURES,
        "shap_output_note": (
            "TreeExplainer default raw output for binary LightGBM is "
            "typically log-odds. Positive SHAP contributions increase "
            "the model output; they do not establish causation."
        ),
    }

    with (OUT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved dependence artifacts:")
    for path in sorted(OUT_DIR.iterdir()):
        print(f"  {path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()