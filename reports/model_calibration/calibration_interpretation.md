# Probability Calibration — Interpretation

## Evaluation setup
- Model: remove_over_99pct_missing.txt
- Dataset: validation partition
- Customers: 68,837
- Features: 353
- Reliability bins: 10, approximately equal-frequency

## Results
- Brier score: 0.068036
- Log loss: 0.218884
- Observed positive-label rate: 25.8931%
- Mean predicted score: 25.7737%

## Findings
The mean predicted score is close to the observed positive-label
rate on this validation partition. The reliability table shows
reasonably close agreement in several score bands, with larger
differences in some middle bands.

These results describe the competition-sampled validation data.
They do not establish calibration to a real credit-card portfolio.

## Limitations
- Negative-class sampling affects the observed class distribution.
- Scores should not automatically be interpreted as real-world
  default probabilities.
- No calibration transformation was applied.
- This analysis used validation data, not the held-out test set.

## Artifacts
- calibration_metrics.json
- reliability_table.csv
- reliability_curve.png
- calibration_interpretation.md