import numpy as np
import pytest

from src.models.amex_metrics import amex_metrics


def reference_metric(y, score):
    """Small independent reference calculation for regression tests."""
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    weights = np.where(y == 0, 20.0, 1.0)
    total_weight = weights.sum()
    total_positive_weight = (y * weights).sum()

    def gini(order):
        w = weights[order]
        target = y[order]
        cw = np.cumsum(w) / total_weight
        cp = np.cumsum(target * w) / total_positive_weight
        return np.sum((cp - cw) * w)

    order = np.argsort(-score, kind="mergesort")
    ideal_order = np.argsort(-y, kind="mergesort")
    normalized_gini = gini(order) / gini(ideal_order)

    ordered_weights = weights[order]
    ordered_y = y[order]
    cumulative_fraction = np.cumsum(ordered_weights) / total_weight
    top4 = ordered_y[cumulative_fraction <= 0.04].sum() / y.sum()

    return normalized_gini, top4, 0.5 * (normalized_gini + top4)


def test_perfect_ranking_has_normalized_gini_one():
    y = np.array([0, 0, 1, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])

    result = amex_metrics(y, score)

    assert result["weighted_normalized_gini"] == pytest.approx(1.0)
    assert result["amex_metric"] == pytest.approx(
        0.5 * (
            result["weighted_normalized_gini"]
            + result["weighted_top4_default_capture"]
        )
    )


def test_reversed_ranking_has_negative_gini():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

    result = amex_metrics(y, score)
    assert result["weighted_normalized_gini"] < 0


def test_matches_reference_implementation():
    y = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 0])
    score = np.array([0.8, 0.95, 0.2, 0.7, 0.6, 0.1, 0.4, 0.3, 0.9, 0.5])

    expected_gini, expected_top4, expected_amex = reference_metric(y, score)
    actual = amex_metrics(y, score)

    assert actual["weighted_normalized_gini"] == pytest.approx(expected_gini)
    assert actual["weighted_top4_default_capture"] == pytest.approx(expected_top4)
    assert actual["amex_metric"] == pytest.approx(expected_amex)


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        amex_metrics([0, 1], [0.1])

    with pytest.raises(ValueError):
        amex_metrics([0, 2], [0.1, 0.9])

    with pytest.raises(ValueError):
        amex_metrics([0, 1], [0.1, np.nan])

    with pytest.raises(ValueError):
        amex_metrics([0, 0], [0.1, 0.2])

    with pytest.raises(ValueError):
        amex_metrics([1, 1], [0.1, 0.2])


def test_metric_outputs_are_finite():
    result = amex_metrics(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.6, 0.9],
    )
    assert all(np.isfinite(value) for value in result.values())
