# AMEX Credit Risk Intelligence Platform — Model Card

## 1. Model overview

- Model: LightGBM binary classifier
- Model artifact: models/experiments/remove_over_99pct_missing.txt
- Task: Rank customers by their likelihood of the positive competition label.
- Feature count: 353
- Feature construction: Customer-level latest-statement features and
  historical means.
- Validation approach: Customer-level stratified train/validation/test split.
- Selected model: The model excluding features with more than 99% missingness.
- Test set: Held out from feature-selection and model-selection decisions.

This is a competition-aligned benchmark, not a production-approved
credit decision system.

## 2. Dataset and target limitations

- Source statement records: 5,531,451
- Customers: 458,913
- Statement period: 2017-03-01 to 2018-03-31
- Labels: 458,913 customers
- Competition-sampled positive-label rate: approximately 25.89%.

The competition dataset downsampled the negative class. The observed
positive-label rate must not be interpreted as the actual portfolio
default rate.

The official competition target is tied to each customer's own latest
statement and subsequent performance window. This project does not
currently have labels for multiple historical prediction snapshots
per customer, so it does not establish a common-date chronological
backtest.

## 3. Final held-out test performance

Source: reports/final_test_metrics.json

- Test customers: 68,838
- ROC-AUC: 0.960778
- Average Precision: 0.897018
- Log loss: 0.219594
- Weighted normalized Gini: 0.921555
- Weighted top-4% default capture: 0.657672
- Official-style AMEX metric: 0.789613

The weighted AMEX metric uses weight 20 for target=0 and weight 1
for target=1, consistent with the competition's negative-class
sampling convention.

The test results are reserved for final evaluation. Do not use this
partition for further feature selection or tuning.

## 4. Feature selection experiment

Source: reports/experiments/sparse_feature_comparison.csv

| Experiment | Features | Validation ROC-AUC | Validation AMEX |
|---|---:|---:|---:|
| All features | 365 | 0.961312 | 0.792437 |
| Remove >99% missing | 353 | 0.961151 | 0.794493 |
| Remove >95% missing | 344 | 0.961041 | 0.791914 |

The 353-feature model was selected based on validation AMEX performance.
The improvement over the all-feature baseline was modest.

## 5. Interpretability

Sources:
- reports/model_interpretability/lightgbm_feature_importance.csv
- reports/model_interpretability/shap_feature_importance.csv
- reports/model_interpretability/model_insights_report.md

The strongest global feature by both model gain and mean absolute SHAP
was latest__P_2.

Other influential features included latest__B_1, latest__B_4,
latest__D_39, mean__P_2, mean__S_3, latest__B_2, and mean__D_42.

Feature importance and SHAP values describe model behavior. They do
not establish causal relationships or explain a customer's outcome
with certainty.

## 6. Threshold analysis

Source: reports/model_interpretability/validation_thresholds.csv

A threshold of 0.40 achieved the highest F1 score among the tested
thresholds:

- Precision: 0.7780
- Recall: 0.8615
- F1: 0.8176
- Specificity: 0.9141
- Customers flagged: 19,736 of 68,837 (28.67%)

This is a provisional validation threshold, not an approved production
threshold. Threshold selection requires business costs, operational
capacity, representative data, and policy review.

## 7. Probability calibration

Source: reports/model_calibration/calibration_metrics.json

Validation results:
- Brier score: 0.068036
- Log loss: 0.218884
- Mean predicted score: 0.257737
- Observed positive-label rate: 0.258931

Calibration measured on the competition-sampled dataset does not
establish real-world default-probability calibration. No population
correction has been applied.

## 8. Robustness by customer history

Source: reports/model_robustness/validation_robustness.csv

| History length | Customers | ROC-AUC | Average Precision |
|---|---:|---:|---:|
| 1-3 statements | 2,504 | 0.90414 | 0.83051 |
| 4-6 statements | 2,208 | 0.91681 | 0.87987 |
| 7-9 statements | 2,650 | 0.94042 | 0.92464 |
| 10-13 statements | 61,475 | 0.96462 | 0.89982 |

Performance is weaker for customers with shorter histories. Investigate
this group further before making operational claims.

## 9. Robustness by latest statement date

Source: reports/model_robustness/validation_robustness.csv

| Latest statement | Customers | ROC-AUC | Average Precision |
|---|---:|---:|---:|
| March 1-10, 2018 | 20,360 | 0.95891 | 0.89457 |
| March 11-20, 2018 | 23,263 | 0.96194 | 0.90090 |
| March 21-31, 2018 | 25,214 | 0.96221 | 0.89778 |

Performance was similar across these three within-month groups.
This does not demonstrate stability across future months or economic
regimes.

## 10. Known limitations and required future work

1. Evaluate performance uncertainty for short-history subgroups.
2. Validate calibration using data representative of the intended
   deployment population.
3. Define business costs and operational capacity before selecting
   a decision threshold.
4. Obtain appropriately aligned historical labels for a chronological
   backtest.
5. Evaluate subgroup performance, fairness, drift, and stability.
6. Review feature availability and leakage risks for the intended
   prediction time.
7. Establish model monitoring, reproducibility, and retraining criteria.
8. Do not use this benchmark alone to make real-world credit decisions.

## 11. Supporting artifacts

- Final test metrics: reports/final_test_metrics.json
- Feature experiments: reports/experiments/
- Interpretability: reports/model_interpretability/
- Calibration: reports/model_calibration/
- Temporal audits: reports/temporal_audit/
- Robustness: reports/model_robustness/
- Baseline model: models/lightgbm_baseline.txt
- Selected model: models/experiments/remove_over_99pct_missing.txt

## 12. Multi-model comparison and final ensemble

### Models evaluated

- Logistic Regression: regularized linear baseline.
- LightGBM: gradient-boosted tree benchmark.
- CatBoost: gradient-boosted trees with native categorical handling.
- LightGBM-CatBoost ensemble: fixed 70% LightGBM and 30% CatBoost
  probability blend.

All models were compared using the same customer-level training and
validation partitions and 353 selected features. Ensemble weights
were selected using validation results only.

### Corrected validation comparison

Source: reports/model_comparison/validation_metrics_recomputed.csv

| Model | Validation ROC-AUC | Average Precision | AMEX metric |
|---|---:|---:|---:|
| Logistic Regression | 0.957279 | 0.887814 | 0.776903 |
| LightGBM | 0.961151 | 0.897899 | 0.794493 |
| CatBoost | 0.961342 | 0.898277 | 0.791178 |
| LightGBM-CatBoost 70/30 blend | 0.961623 | 0.899085 | 0.795498 |

The ensemble was selected as the leading candidate based on validation
AMEX performance. The tested blend weights were not selected using
the held-out test set.

### Corrected held-out test comparison

Sources:
- reports/model_comparison/lightgbm_test_metrics_corrected.json
- reports/model_comparison/final_ensemble_test_metrics.json

| Metric | LightGBM | 70/30 ensemble |
|---|---:|---:|
| Customers | 68,838 | 68,838 |
| ROC-AUC | 0.960778 | 0.961221 |
| Average Precision | 0.897018 | 0.898309 |
| Weighted normalized Gini | 0.921555 | 0.922441 |
| Weighted top-4% default capture | 0.657672 | 0.662160 |
| AMEX metric | 0.789613 | 0.792301 |
| Log loss | 0.219594 | 0.218316 |
| Brier score | 0.068515 | 0.068066 |

The frozen ensemble improved the test AMEX metric by approximately
0.002687 and weighted top-4% default capture by approximately
0.449 percentage points compared with LightGBM alone.

These test results are for final reporting only. They must not be used
to change blend weights, tune hyperparameters, or select additional
features.

### Metric implementation

The shared implementation is in src/models/amex_metrics.py.
It calculates weighted normalized Gini, weighted top-4% default
capture, and their mean.

The competition-style metric assigns weight 20 to target=0 and
weight 1 to target=1. The top-4% cutoff is based on cumulative sample
weight, not raw row count.

Unit tests are maintained in tests/test_amex_metrics.py. The metric
implementation and reports should be kept version-consistent; if the
metric code changes, regenerate affected comparison reports.

### Final status and limitations

The 70/30 LightGBM-CatBoost ensemble is the leading candidate in this
benchmark. It is not approved for production credit decisions.

The test data comes from a competition-sampled population. Its
positive-label rate does not represent the real portfolio default
rate, and the observed Brier score and log loss do not establish
real-world probability calibration.

Further work should include calibration on representative data,
a common-date chronological backtest with appropriately aligned
historical labels, subgroup and stability analysis, and a documented
business decision policy before production use.
