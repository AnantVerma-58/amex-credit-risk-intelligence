# AmEx Credit Risk Intelligence Platform

An end-to-end credit-risk analytics and machine-learning project built
using the American Express Default Prediction dataset.

The project covers data ingestion, quality validation, customer-level
feature engineering, model benchmarking, risk ranking, model evaluation,
calibration, interpretability, and SQL-based portfolio analytics.

## Project objectives

- Build a reproducible customer-level credit-risk modelling pipeline.
- Compare Logistic Regression, LightGBM, and CatBoost.
- Evaluate a validation-selected LightGBM–CatBoost ensemble.
- Implement AMEX competition-style evaluation metrics.
- Analyze feature importance, calibration, and model robustness.
- Document limitations relevant to credit-risk decisioning.

## Selected results

The following results are from the saved held-out test reports.

| Metric | LightGBM | 70/30 ensemble |
|---|---:|---:|
| ROC-AUC | 0.9608 | 0.9612 |
| Average precision | 0.8970 | 0.8983 |
| Weighted normalized Gini | 0.9216 | 0.9224 |
| Top-4 default capture | 65.77% | 66.22% |
| AMEX metric | 0.7896 | 0.7923 |
| Log loss | 0.2196 | 0.2183 |
| Brier score | 0.0685 | 0.0681 |

The ensemble uses 70% LightGBM and 30% CatBoost prediction scores.
Its blend weights were selected using validation results, not the
held-out test set.

See `reports/model_card.md` and the saved model-comparison reports for
additional evaluation details.

## Architecture

```text
AMEX source data
      |
      v
Data ingestion and staging
      |
      v
Data quality and relationship validation
      |
      v
Customer-level feature engineering
      |
      v
Customer-level train / validation / test split
      |
      v
Logistic Regression | LightGBM | CatBoost
      |
      v
Validation-based ensemble selection
      |
      v
Held-out evaluation
      |
      +--> Calibration analysis
      +--> Feature importance and SHAP
      +--> Robustness and temporal audits
      +--> SQL analytics and reports