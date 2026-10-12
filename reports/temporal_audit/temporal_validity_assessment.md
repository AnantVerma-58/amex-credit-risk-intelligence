# Temporal Validity Assessment

## 1. Dataset coverage

- Statement records: 5,531,451
- Labeled customers: 458,913
- First statement date: 2017-03-01
- Last statement date: 2018-03-31
- Statement dates after 2018-03-31: 0
- Missing statement dates: 0
- Statements per customer: 1 to 13

## 2. Target definition

The AMEX Default Prediction competition defines the target using an
18-month performance window after the customer's latest statement.
A default event is defined by failure to pay the due amount within
120 days after that statement date.

Reference:
https://www.kaggle.com/competitions/amex-default-prediction/data

## 3. Feature timing

The current feature builder creates one feature row per customer.
Latest-value features use the customer's latest available statement.
Mean features aggregate the customer's available statement history.

These features are aligned to each customer's own latest statement,
rather than to one common prediction date.

## 4. Assessment

The statement-date boundary audit passed for 2018-03-31:
- No statement rows occur after the proposed boundary.
- No statement dates are null.

This verifies the observed feature-date boundary, but it is not a
complete production leakage audit.

The current task is a competition-aligned customer-level benchmark.
It should not be described as a common-date chronological backtest:
customers have different latest statement dates, and the supplied
labels provide one outcome per customer rather than labels for
multiple historical snapshots.

## 5. Limitations and next steps

1. Preserve the competition benchmark and its final test report.
2. For a true chronological backtest, obtain or construct outcome
   labels aligned to fixed historical prediction dates.
3. For production-style probability calibration, use a representative
   population with an appropriate outcome definition and observation
   window.
4. Do not interpret the competition-sampled positive-label rate as a
   real-world portfolio default rate.
5. Before deployment, perform additional data leakage, stability,
   fairness, calibration, and operational validation.

## 6. Audit artifacts

- monthly_statement_coverage.csv
- last_statement_month_summary.csv
- temporal_coverage_summary.json
- prediction_cutoff_audit.json
