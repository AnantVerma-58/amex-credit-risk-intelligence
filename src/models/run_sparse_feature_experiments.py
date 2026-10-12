
from pathlib import Path
import gc
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "features" / "splits"
REPORT_DIR = ROOT / "reports" / "experiments"
MODEL_DIR = ROOT / "models" / "experiments"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_CONFIG = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 100,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": 4,
    "verbosity": -1,
}

CATEGORICAL_SOURCE_FEATURES = {
    "B_30", "B_38", "D_66", "D_68",
    "D_114", "D_116", "D_117", "D_120", "D_126",
}


def is_categorical_feature(col):
    source_name = col.rsplit("__", 1)[-1]
    return (
        col.endswith("__D_63")
        or col.endswith("__D_64")
        or source_name in CATEGORICAL_SOURCE_FEATURES
    )


def amex_metrics(y_true, scores):
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)

    if len(y_true) != len(scores):
        raise ValueError("Labels and scores have different lengths")

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        raise ValueError("Both target classes must be present")

    # Official AMEX weighting:
    # non-default (target=0) has weight 20; default (target=1) has weight 1.
    def weighted_gini(order):
        labels_sorted = y_true[order]
        weights = np.where(labels_sorted == 0, 20.0, 1.0)

        weight_random = np.cumsum(weights / weights.sum())
        total_positive_weight = np.sum(labels_sorted * weights)
        cumulative_positive_weight = np.cumsum(
            labels_sorted * weights
        )
        lorentz = cumulative_positive_weight / total_positive_weight

        return np.sum((lorentz - weight_random) * weights)

    # Rank by model predictions.
    prediction_order = np.argsort(-scores, kind="mergesort")

    # Rank by true labels for the perfect-ranking reference.
    perfect_order = np.argsort(-y_true, kind="mergesort")

    normalized_gini = (
        weighted_gini(prediction_order)
        / weighted_gini(perfect_order)
    )

    # Official weighted top-4% cutoff.
    labels_sorted = y_true[prediction_order]
    weights_sorted = np.where(labels_sorted == 0, 20, 1)
    cutoff = int(0.04 * weights_sorted.sum())

    within_cutoff = np.cumsum(weights_sorted) <= cutoff
    defaults_captured = (
        labels_sorted[within_cutoff].sum() / y_true.sum()
    )

    roc_auc = roc_auc_score(y_true, scores)
    average_precision = average_precision_score(y_true, scores)

    return {
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "normalized_gini": float(normalized_gini),
        "defaults_captured_top_4pct": float(defaults_captured),
        "amex_metric": float(
            0.5 * (normalized_gini + defaults_captured)
        ),
        "customers_in_weighted_top_4pct": int(within_cutoff.sum()),
    }




def main():
    print("Loading training and validation partitions...")

    train = pd.read_parquet(SPLIT_DIR / "train")
    valid = pd.read_parquet(SPLIT_DIR / "validation")

    y_train = train["target"].astype("int8")
    y_valid = valid["target"].astype("int8")

    # Calculate feature missingness on TRAINING data only.
    feature_cols = [
        col for col in train.columns
        if col not in ("target", "customer_ID")
    ]

    missing_pct = (
        train[feature_cols].isna().mean() * 100
    )

    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(valid):,}")
    print(f"Total features before filtering: {len(feature_cols)}")

    X_train_all = train[feature_cols].copy()
    X_valid_all = valid[feature_cols].copy()

    del train, valid
    gc.collect()

    # Use consistent category vocabularies across train/validation.
    # Categories are derived from feature values, not target labels.
    categorical_cols_all = [
        col for col in feature_cols if is_categorical_feature(col)
    ]

    for col in categorical_cols_all:
        categories = pd.Index(
            pd.concat(
                [X_train_all[col], X_valid_all[col]],
                ignore_index=True,
            ).dropna().unique()
        )

        X_train_all[col] = pd.Categorical(
            X_train_all[col], categories=categories
        )
        X_valid_all[col] = pd.Categorical(
            X_valid_all[col], categories=categories
        )

    experiments = [
        ("baseline_all_features", None),
        ("remove_over_99pct_missing", 99.0),
        ("remove_over_95pct_missing", 95.0),
    ]

    results = []

    for experiment_name, threshold in experiments:
        if threshold is None:
            selected = feature_cols
        else:
            selected = [
                col for col in feature_cols
                if missing_pct[col] <= threshold
            ]

        removed = [col for col in feature_cols if col not in selected]

        X_train = X_train_all[selected]
        X_valid = X_valid_all[selected]

        categorical_cols = [
            col for col in selected
            if col in categorical_cols_all
        ]

        print("\n" + "=" * 65)
        print(f"Experiment: {experiment_name}")
        print(f"Missingness threshold: {threshold}")
        print(f"Features retained: {len(selected)}")
        print(f"Features removed: {len(removed)}")
        print(f"Categorical features retained: {len(categorical_cols)}")
        print("Training model...")

        model = lgb.LGBMClassifier(**BATCH_CONFIG)

        model.fit(
            X_train,
            y_train,
            categorical_feature=categorical_cols,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(100, verbose=True),
                lgb.log_evaluation(100),
            ],
        )

        scores = model.predict_proba(
            X_valid,
            num_iteration=model.best_iteration_,
        )[:, 1]

        metrics = amex_metrics(y_valid.to_numpy(), scores)
        metrics.update({
            "experiment": experiment_name,
            "missingness_threshold_pct": threshold,
            "feature_count": len(selected),
            "removed_feature_count": len(removed),
            "best_iteration": int(model.best_iteration_ or 0),
            "train_rows": int(len(y_train)),
            "validation_rows": int(len(y_valid)),
            "random_seed": SEED,
        })

        print("\nValidation metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")

        model_path = MODEL_DIR / f"{experiment_name}.txt"
        model.booster_.save_model(str(model_path))

        metrics["model_path"] = str(model_path.relative_to(ROOT))
        metrics["removed_features"] = removed
        metrics["selected_features"] = selected
        results.append(metrics)

        del model, scores, X_train, X_valid
        gc.collect()

    # Save full results and a compact comparison table.
    results_path = REPORT_DIR / "sparse_feature_experiments.json"
    results_path.write_text(json.dumps(results, indent=2))

    comparison_cols = [
        "experiment",
        "feature_count",
        "best_iteration",
        "roc_auc",
        "average_precision",
        "normalized_gini",
        "defaults_captured_top_4pct",
        "amex_metric",
    ]

    comparison = pd.DataFrame(results)[comparison_cols]
    comparison_path = REPORT_DIR / "sparse_feature_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPARISON")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved detailed results: {results_path.relative_to(ROOT)}")
    print(f"Saved comparison: {comparison_path.relative_to(ROOT)}")
    print("Test partition was not loaded or used.")


if __name__ == "__main__":
    main()
