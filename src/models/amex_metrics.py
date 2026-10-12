"""Competition-style AMEX metric with weighted normalized Gini."""

import numpy as np


def amex_metrics(y_true, y_score):
    """
    Compute weighted normalized Gini, weighted top-4% default capture,
    and the combined AMEX metric.

    target=0 has weight 20; target=1 has weight 1.
    Higher scores mean greater likelihood of target=1.

    The top-4% cutoff is based on cumulative sample weight, not raw
    row count. This matches the competition-style weighted definition.
    """
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    score = np.asarray(y_score, dtype=np.float64).reshape(-1)

    if len(y) == 0 or len(y) != len(score):
        raise ValueError("Labels and scores must have equal, nonzero length.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary (0 or 1).")
    if not np.isfinite(score).all():
        raise ValueError("Scores must all be finite.")
    if not np.any(y == 1):
        raise ValueError("At least one positive label is required.")
    if not np.any(y == 0):
        raise ValueError("At least one negative label is required.")

    weights = np.where(y == 0, 20.0, 1.0)
    total_weight = weights.sum()
    total_positive_weight = np.sum(y * weights)

    def weighted_gini(order):
        ordered_y = y[order]
        ordered_weights = weights[order]

        cumulative_weight = (
            np.cumsum(ordered_weights) / total_weight
        )
        cumulative_positive = (
            np.cumsum(ordered_y * ordered_weights)
            / total_positive_weight
        )

        # Integrate the difference using each row's sample weight.
        return float(
            np.sum(
                (cumulative_positive - cumulative_weight)
                * ordered_weights
            )
        )

    # Stable ordering makes ties deterministic.
    prediction_order = np.argsort(-score, kind="mergesort")
    ideal_order = np.argsort(-y, kind="mergesort")

    ideal_gini = weighted_gini(ideal_order)
    if not np.isfinite(ideal_gini) or ideal_gini <= 0:
        raise ValueError("Unable to calculate normalized weighted Gini.")

    normalized_gini = (
        weighted_gini(prediction_order) / ideal_gini
    )

    ordered_y = y[prediction_order]
    ordered_weights = weights[prediction_order]
    cumulative_weight_fraction = (
        np.cumsum(ordered_weights) / total_weight
    )

    # The official-style cutoff includes rows whose cumulative weight
    # is at or below 4% of total sample weight.
    selected = cumulative_weight_fraction <= 0.04
    top4_capture = float(
        ordered_y[selected].sum() / y.sum()
    )

    combined = 0.5 * (normalized_gini + top4_capture)

    return {
        "weighted_normalized_gini": float(normalized_gini),
        "weighted_top4_default_capture": top4_capture,
        "amex_metric": float(combined),
    }
