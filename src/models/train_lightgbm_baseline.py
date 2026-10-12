
from pathlib import Path
import json
import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "features" / "splits"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# These are the numeric-encoded categorical source variables.
CATEGORICAL_SOURCE_FEATURES = {
    "B_30", "B_38", "D_66", "D_68",
    "D_114", "D_116", "D_117", "D_120", "D_126",
}


def load_partition(name):
    path = SPLIT_DIR / name
    print(f"Loading {name} partition...")
    df = pd.read_parquet(path)

    if "target" not in df or "customer_ID" not in df:
        raise ValueError(f"{name} is missing target or customer_ID")

    print(f"  Shape: {df.shape}")
    return df


def main():
    train = load_partition("train")
    valid = load_partition("validation")

    # Keep IDs separate from model inputs.
    y_train = train["target"].astype("int8")
    y_valid = valid["target"].astype("int8")

    X_train = train.drop(columns=["target", "customer_ID"])
    X_valid = valid.drop(columns=["target", "customer_ID"])

    del train, valid
    gc.collect()

    if list(X_train.columns) != list(X_valid.columns):
        raise ValueError("Train and validation feature columns differ")

    categorical_cols = [
        col for col in X_train.columns
        if (
            col.endswith("__D_63")
            or col.endswith("__D_64")
            or col.rsplit("__", 1)[-1] in CATEGORICAL_SOURCE_FEATURES
        )
    ]

    # Ensure each categorical feature uses the same category levels
    # in training and validation.
    for col in categorical_cols:
        combined_categories = pd.Index(
            pd.concat(
                [X_train[col], X_valid[col]],
                ignore_index=True,
            ).dropna().unique()
        )

        X_train[col] = pd.Categorical(
            X_train[col],
            categories=combined_categories,
        )
        X_valid[col] = pd.Categorical(
            X_valid[col],
            categories=combined_categories,
        )

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_valid):,}")
    print(f"Feature count: {X_train.shape[1]}")
    print(f"Categorical features: {categorical_cols}")

    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=4,
        verbosity=-1,
    )

    print("\nTraining LightGBM baseline...")

    model.fit(
        X_train,
        y_train,
        categorical_feature=categorical_cols,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                verbose=True,
            ),
            lgb.log_evaluation(period=50),
        ],
    )

    probabilities = model.predict_proba(
        X_valid,
        num_iteration=model.best_iteration_,
    )[:, 1]

    roc_auc = roc_auc_score(y_valid, probabilities)
    pr_auc = average_precision_score(y_valid, probabilities)

    report = {
        "model": "LightGBM baseline",
        "random_seed": SEED,
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_valid)),
        "feature_count": int(X_train.shape[1]),
        "categorical_features": categorical_cols,
        "best_iteration": int(model.best_iteration_ or 0),
        "validation_roc_auc": float(roc_auc),
        "validation_average_precision": float(pr_auc),
        "validation_target_rate": float(y_valid.mean()),
        "note": (
            "Random customer-level split with full-history summary "
            "features; not an out-of-time early-warning evaluation."
        ),
    }

    model_path = MODEL_DIR / "lightgbm_baseline.txt"
    report_path = REPORT_DIR / "lightgbm_baseline_metrics.json"

    model.booster_.save_model(str(model_path))
    report_path.write_text(json.dumps(report, indent=2))

    print("\n===== VALIDATION RESULTS =====")
    print(f"Best iteration: {model.best_iteration_}")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"PR-AUC / Average Precision: {pr_auc:.6f}")
    print(f"Validation target rate: {y_valid.mean():.4%}")
    print(f"\nSaved model: {model_path.relative_to(ROOT)}")
    print(f"Saved metrics: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
