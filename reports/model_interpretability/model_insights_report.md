# AMEX Credit Risk Model — Interpretability Report

## 1. Purpose

This report summarizes the feature-importance analysis of the selected
LightGBM model. Native LightGBM gain importance and SHAP values were
calculated using the saved 353-feature model. SHAP analysis used a
reproducible sample of 5,000 validation rows with random seed 42.

## 2. Model evaluated

- Model: `models/experiments/remove_over_99pct_missing.txt`
- Algorithm: LightGBM binary classifier
- Input features: 353
- Data used for SHAP: validation partition
- SHAP sample size: 5,000
- SHAP random seed: 42

## 3. Key finding

`latest__P_2` ranks first by both LightGBM gain and mean absolute SHAP
value. Its mean absolute SHAP value is approximately 1.0433, compared
with 0.2237 for the second-ranked feature, `latest__B_1`.

This indicates that `latest__P_2` has substantially greater average
attribution magnitude than the other features in this SHAP sample.
It does not establish that the feature causes default.

## 4. Top 10 features by mean absolute SHAP

| Rank | Feature | Mean absolute SHAP |
|---:|---|---:|
| 1 | `latest__P_2` | 1.043259 |
| 2 | `latest__B_1` | 0.223696 |
| 3 | `latest__B_4` | 0.169586 |
| 4 | `latest__D_39` | 0.152590 |
| 5 | `mean__P_2` | 0.139757 |
| 6 | `mean__S_3` | 0.120870 |
| 7 | `latest__B_2` | 0.120452 |
| 8 | `mean__D_42` | 0.103240 |
| 9 | `latest__B_5` | 0.097890 |
| 10 | `latest__B_11` | 0.096181 |

## 5. Interpretation of the importance measures

- **LightGBM gain:** measures the model's accumulated improvement
  attributable to splits using a feature.
- **Mean absolute SHAP:** measures the average magnitude of a feature's
  attribution across the sampled validation observations.
- **SHAP direction:** mean absolute SHAP alone does not show whether
  high or low feature values push predictions toward the positive class.
  Review the SHAP summary plot before drawing directional conclusions.

The rankings need not be identical. The measures capture different
aspects of model behavior, and correlated features can share predictive
information.

## 6. Artifacts

- `lightgbm_feature_importance.csv`
- `shap_feature_importance.csv`
- `shap_summary.png`
- `metadata.json`

## 7. Limitations

1. Feature names are anonymized identifiers. Do not assign business
   meanings without a verified feature dictionary.
2. Feature attribution is not causal evidence.
3. The SHAP analysis describes a sample of validation observations,
   not every possible customer or future population.
4. The model uses latest-value and full-history mean features. The
   results should not be described as historical early-warning
   performance until the prediction cutoff and feature availability
   have been explicitly defined.
5. This report does not establish fairness, calibration, regulatory
   suitability, or production readiness.

## 8. Conclusion

`latest__P_2` is the dominant feature in the current model's gain and
SHAP analyses. The next interpretability task is to examine SHAP
direction and dependence, verify feature definitions where possible,
and assess whether the model's behavior is stable across customer
groups and data partitions.