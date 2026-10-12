from pathlib import Path
import sys

# Ensure the project root is importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import inspect
import json
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.amex_metrics import amex_metrics

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data/features/splits/train"
VALIDATION_DIR = ROOT / "data/features/splits/validation"
LIGHTGBM_PATH = ROOT / "models/experiments/remove_over_99pct_missing.txt"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports/model_comparison"

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

# Previous recorded validation AMEX score for the selected LightGBM model.
EXPECTED_LIGHTGBM_VALIDATION_AMEX = 0.794493


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
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
    }
    result.update(amex_metrics(y_true, probabilities))
    return result


def make_one_hot_encoder():
    # Supports both newer and older scikit-learn releases.
    parameters = inspect.signature(OneHotEncoder).parameters
    kwargs = {"handle_unknown": "ignore", "dtype": np.float32}

    if "sparse_output" in parameters:
        kwargs["sparse_output"] = True
    else:
        kwargs["sparse"] = True

    return OneHotEncoder(**kwargs)


def prepare_categorical_columns(frame, columns):
    for col in columns:
        # Treat numeric-coded categories as categories, not continuous values.
        frame[col] = (
            frame[col]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )
    return frame


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"scikit-learn version: {sklearn.__version__}")
    print("Loading saved LightGBM model...")
    lightgbm_model = lgb.Booster(model_file=str(LIGHTGBM_PATH))
    features = lightgbm_model.feature_name()
    categorical = [c for c in features if c in CATEGORICAL_FEATURES]
    numeric = [c for c in features if c not in categorical]

    print(f"Shared feature count: {len(features)}")
    print(f"Categorical features: {len(categorical)}")
    print(f"Numeric features: {len(numeric)}")

    print("Loading train and validation Parquet partitions...")
    train = pd.read_parquet(TRAIN_DIR)
    validation = pd.read_parquet(VALIDATION_DIR)

    missing_train = set(features) - set(train.columns)
    missing_val = set(features) - set(validation.columns)
    if missing_train or missing_val:
        raise ValueError(
            f"Missing train features: {sorted(missing_train)}; "
            f"missing validation features: {sorted(missing_val)}"
        )

    y_train = train["target"].to_numpy(dtype=np.int8)
    y_val = validation["target"].to_numpy(dtype=np.int8)

    if train["customer_ID"].duplicated().any():
        raise ValueError("Duplicate customer IDs in training partition.")
    if validation["customer_ID"].duplicated().any():
        raise ValueError("Duplicate customer IDs in validation partition.")

    print(
        f"Train rows: {len(train):,}; "
        f"validation rows: {len(validation):,}"
    )

    # Evaluate the existing LightGBM model first, without retraining it.
    print("\nScoring saved LightGBM model on validation...")
    X_val_lgb = validation.loc[:, features].copy()
    for col in categorical:
        X_val_lgb[col] = X_val_lgb[col].astype("category")

    lgb_start = time.perf_counter()
    lgb_probabilities = lightgbm_model.predict(
        X_val_lgb, num_threads=4
    )
    lgb_seconds = time.perf_counter() - lgb_start
    lgb_metrics = calculate_metrics(y_val, lgb_probabilities)
    lgb_metrics["model"] = "LightGBM (saved candidate)"
    lgb_metrics["training_seconds"] = None
    lgb_metrics["validation_scoring_seconds"] = round(lgb_seconds, 3)

    print("Saved LightGBM validation metrics:")
    print(json.dumps(lgb_metrics, indent=2))

    previous = EXPECTED_LIGHTGBM_VALIDATION_AMEX
    current = lgb_metrics["amex_metric"]
    if abs(current - previous) > 0.01:
        warnings.warn(
            "The shared metric differs from the previously recorded "
            f"LightGBM validation score ({previous:.6f} vs {current:.6f}). "
            "Verify metric parity before interpreting model comparisons."
        )

    # Prepare independent copies for the Logistic Regression pipeline.
    print("\nPreparing Logistic Regression inputs...")
    X_train = train.loc[:, features].copy()
    X_val = validation.loc[:, features].copy()

    del X_val_lgb

    X_train = prepare_categorical_columns(X_train, categorical)
    X_val = prepare_categorical_columns(X_val, categorical)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                ),
            ),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="saga",
                    max_iter=300,
                    tol=1e-3,
                    random_state=42,
                    n_jobs=4,
                ),
            ),
        ]
    )

    print("\nTraining Logistic Regression...")
    print("This may take several minutes on the full training partition.")
    train_start = time.perf_counter()
    logistic_pipeline.fit(X_train, y_train)
    training_seconds = time.perf_counter() - train_start

    print("Scoring Logistic Regression validation partition...")
    scoring_start = time.perf_counter()
    logistic_probabilities = logistic_pipeline.predict_proba(X_val)[:, 1]
    scoring_seconds = time.perf_counter() - scoring_start

    logistic_metrics = calculate_metrics(y_val, logistic_probabilities)
    logistic_metrics["model"] = "Logistic Regression"
    logistic_metrics["training_seconds"] = round(training_seconds, 3)
    logistic_metrics["validation_scoring_seconds"] = round(
        scoring_seconds, 3
    )
    logistic_metrics["iterations_used"] = int(
        logistic_pipeline.named_steps["classifier"].n_iter_[0]
    )
    logistic_metrics["max_iterations"] = 300

    if logistic_metrics["iterations_used"] >= 300:
        warnings.warn(
            "Logistic Regression reached max_iter=300. "
            "Check convergence before treating its result as final."
        )

    model_path = MODEL_DIR / "logistic_regression_baseline.joblib"
    joblib.dump(logistic_pipeline, model_path, compress=3)

    results = [lgb_metrics, logistic_metrics]
    comparison = pd.DataFrame(results)

    preferred_columns = [
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
    ]
    comparison = comparison[preferred_columns]
    comparison.to_csv(
        REPORT_DIR / "validation_model_comparison.csv",
        index=False,
    )

    with open(
        REPORT_DIR / "logistic_regression_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(logistic_metrics, f, indent=2)

    print("\n=== VALIDATION MODEL COMPARISON ===")
    print(comparison.round(6).to_string(index=False))
    print(f"\nSaved Logistic Regression model: {model_path}")
    print(
        "Saved comparison: "
        f"{REPORT_DIR / 'validation_model_comparison.csv'}"
    )
    print(
        "Saved Logistic Regression metrics: "
        f"{REPORT_DIR / 'logistic_regression_metrics.json'}"
    )
    print(
        "\nNo test data was loaded or used. The comparison is based "
        "on the validation partition only."
    )


if __name__ == "__main__":
    main()
