
from pathlib import Path
import sys

import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LIGHTGBM_MODEL = (
    PROJECT_ROOT / "models/experiments/remove_over_99pct_missing.txt"
)
CATBOOST_MODEL = PROJECT_ROOT / "models/catboost_baseline.cbm"
OUTPUT_DIR = PROJECT_ROOT / "reports/explainability"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # LightGBM: split importance and gain importance
    if not LIGHTGBM_MODEL.exists():
        raise FileNotFoundError(f"Missing model: {LIGHTGBM_MODEL}")

    lgb_model = lgb.Booster(model_file=str(LIGHTGBM_MODEL))
    lgb_names = lgb_model.feature_name()

    lgb_df = pd.DataFrame({
        "feature": lgb_names,
        "split_importance": lgb_model.feature_importance(
            importance_type="split"
        ),
        "gain_importance": lgb_model.feature_importance(
            importance_type="gain"
        ),
    })

    lgb_df["gain_importance_pct"] = (
        100 * lgb_df["gain_importance"]
        / max(lgb_df["gain_importance"].sum(), 1e-12)
    )
    lgb_df = lgb_df.sort_values(
        "gain_importance", ascending=False
    )
    lgb_df.to_csv(
        OUTPUT_DIR / "lightgbm_feature_importance.csv",
        index=False,
    )

    # CatBoost: PredictionValuesChange importance
    if not CATBOOST_MODEL.exists():
        raise FileNotFoundError(f"Missing model: {CATBOOST_MODEL}")

    cat_model = CatBoostClassifier()
    cat_model.load_model(str(CATBOOST_MODEL))

    cat_names = cat_model.feature_names_
    cat_values = cat_model.get_feature_importance(
        type="PredictionValuesChange"
    )

    cat_df = pd.DataFrame({
        "feature": cat_names,
        "importance": cat_values,
    })
    cat_df["importance_pct"] = (
        100 * cat_df["importance"]
        / max(cat_df["importance"].sum(), 1e-12)
    )
    cat_df = cat_df.sort_values(
        "importance", ascending=False
    )
    cat_df.to_csv(
        OUTPUT_DIR / "catboost_feature_importance.csv",
        index=False,
    )

    print(f"LightGBM features: {len(lgb_df)}")
    print(f"CatBoost features: {len(cat_df)}")
    print("\nTop 15 LightGBM features:")
    print(lgb_df.head(15).to_string(index=False))
    print("\nTop 15 CatBoost features:")
    print(cat_df.head(15).to_string(index=False))
    print(f"\nReports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
