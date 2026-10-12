from pathlib import Path
import sys

# Ensure the project root is importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import json
import time
import warnings

import catboost
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.models.amex_metrics import amex_metrics

ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = ROOT / "data/features/splits/train"
VALIDATION_DIR = ROOT / "data/features/splits/validation"

LIGHTGBM_PATH = ROOT / "models/experiments/remove_over_99pct_missing.txt"
LOGISTIC_PATH = ROOT / "models/logistic_regression_baseline.joblib"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports/model_comparison"

CATEGORICAL_FEATURES = [
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
]


def calculate_metrics(y_true, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)

    result = {
        "customers": int(len(y_true)),
        "positive_count": int(np.sum(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "log_loss": float(log_loss(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }
    result.update(amex_metrics(y_true, probabilities))
    return result


def prepare_catboost_frame(frame, features, categorical):
    X = frame.loc[:, features].copy()

    # CatBoost requires categorical values to be strings or integers,
    # not floating-point values or NaN.
    for col in categorical:
        X[col] = (
            X[col]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    return X


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"CatBoost version: {catboost.__version__}")
    print("Loading saved LightGBM model to establish feature schema...")

    lightgbm_model = lgb.Booster(model_file=str(LIGHTGBM_PATH))
    features = lightgbm_model.feature_name()

    categorical = [c for c in CATEGORICAL_FEATURES if c in features]

    print(f"Features: {len(features)}")
    print(f"Categorical features: {len(categorical)}")

    print("Loading training and validation partitions...")
    train = pd.read_parquet(TRAIN_DIR)
    validation = pd.read_parquet(VALIDATION_DIR)

    if train["customer_ID"].duplicated().any():
        raise ValueError("Duplicate customer IDs in training data.")
    if validation["customer_ID"].duplicated().any():
        raise ValueError("Duplicate customer IDs in validation data.")

    y_train = train["target"].to_numpy(dtype=np.int8)
    y_val = validation["target"].to_numpy(dtype=np.int8)

    X_train = prepare_catboost_frame(train, features, categorical)
    X_val = prepare_catboost_frame(validation, features, categorical)

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=5.0,
        random_seed=42,
        thread_count=4,
        allow_writing_files=False,
        verbose=100,
    )

    print("\nTraining CatBoost with validation early stopping...")
    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        cat_features=categorical,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=100,
    )

    training_seconds = time.perf_counter() - start

    print("\nScoring validation customers...")
    start = time.perf_counter()
    catboost_probabilities = model.predict_proba(X_val)[:, 1]
    scoring_seconds = time.perf_counter() - start

    catboost_metrics = calculate_metrics(y_val, catboost_probabilities)
    catboost_metrics.update({
        "model": "CatBoost",
        "training_seconds": round(training_seconds, 3),
        "validation_scoring_seconds": round(scoring_seconds, 3),
        "best_iteration": int(model.get_best_iteration()),
        "iterations_configured": 2000,
    })

    model_path = MODEL_DIR / "catboost_baseline.cbm"
    model.save_model(str(model_path))

    with open(
        REPORT_DIR / "catboost_metrics.json", "w", encoding="utf-8"
    ) as f:
        json.dump(catboost_metrics, f, indent=2)

    # Recalculate all three models on exactly the same validation rows.
    print("\nRecomputing same-run metrics for Logistic Regression and LightGBM...")

    logistic_pipeline = joblib.load(LOGISTIC_PATH)
    logistic_probabilities = logistic_pipeline.predict_proba(
        X_val
    )[:, 1]

    # LightGBM requires categorical columns to be pandas category dtype.
    X_val_lgb = validation.loc[:, features].copy()
    for col in categorical:
        X_val_lgb[col] = X_val_lgb[col].astype("category")

    lightgbm_probabilities = lightgbm_model.predict(
        X_val_lgb, num_threads=4
    )

    results = []
    for name, probabilities in [
        ("Logistic Regression", logistic_probabilities),
        ("LightGBM", lightgbm_probabilities),
        ("CatBoost", catboost_probabilities),
    ]:
        metrics = calculate_metrics(y_val, probabilities)
        metrics["model"] = name
        results.append(metrics)

    comparison = pd.DataFrame(results)
    comparison["training_seconds"] = [
        None,
        None,  # LightGBM training was performed in an earlier experiment.
        round(training_seconds, 3),
    ]
    comparison["validation_scoring_seconds"] = [
        None,
        None,
        round(scoring_seconds, 3),
    ]
    comparison["best_iteration"] = [
        None,
        743,  # Previously recorded validation-selected LightGBM iteration.
        int(model.get_best_iteration()),
    ]

    columns = [
        "model",
        "customers",
        "positive_count",
        "positive_rate",
        "roc_auc",
        "average_precision",
        "weighted_normalized_gini",
        "weighted_top4_default_capture",
        "amex_metric",
        "log_loss",
        "brier_score",
        "training_seconds",
        "validation_scoring_seconds",
        "best_iteration",
    ]

    comparison = comparison[columns]
    comparison.to_csv(
        REPORT_DIR / "validation_model_comparison.csv",
        index=False,
    )

    predictions = pd.DataFrame({
        "customer_ID": validation["customer_ID"].astype(str),
        "target": y_val,
        "logistic_regression": logistic_probabilities,
        "lightgbm": lightgbm_probabilities,
        "catboost": catboost_probabilities,
    })
    predictions.to_parquet(
        REPORT_DIR / "validation_predictions.parquet",
        index=False,
    )

    print("\n=== THREE-MODEL VALIDATION COMPARISON ===")
    print(comparison.round(6).to_string(index=False))

    print(f"\nSaved CatBoost model: {model_path}")
    print(f"Saved metrics: {REPORT_DIR / 'catboost_metrics.json'}")
    print(
        "Saved comparison: "
        f"{REPORT_DIR / 'validation_model_comparison.csv'}"
    )
    print(
        "Saved validation predictions: "
        f"{REPORT_DIR / 'validation_predictions.parquet'}"
    )
    print("\nThe held-out test partition was not loaded or evaluated.")


if __name__ == "__main__":
    main()
