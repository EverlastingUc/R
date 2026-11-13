"""Checks for numerical stability and floating-point precision."""

import math

import numpy as np
import pytest

from nn.numerics import float_precision_example, logsumexp, stable_softmax


def test_softmax_matches_reference_for_large_logits() -> None:
    logits = np.array([1000.0, 1001.0, 999.0], dtype=np.float64)
    expected = np.exp(np.array([-1.0, 0.0, -2.0]))
    expected /= expected.sum()
    np.testing.assert_allclose(stable_softmax(logits), expected, atol=1e-15)


def test_softmax_is_finite_and_sums_to_one_for_each_row() -> None:
    logits = np.array([[1000.0, 1001.0], [-1000.0, -1001.0]])
    probabilities = stable_softmax(logits, axis=1)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0])


def test_logsumexp_matches_hand_calculation_for_large_values() -> None:
    assert logsumexp([1000.0, 1000.0]) == pytest.approx(1000.0 + math.log(2.0))


def test_logsumexp_supports_axes_and_keepdims() -> None:
    values = np.array([[0.0, 0.0], [1.0, 1.0]])
    result = logsumexp(values, axis=1, keepdims=True)
    np.testing.assert_allclose(result, [[math.log(2.0)], [1.0 + math.log(2.0)]])


def test_logsumexp_handles_large_negative_values() -> None:
    result = logsumexp([-1000.0, -1001.0])
    assert math.isfinite(result)
    assert result == pytest.approx(-1000.0 + math.log1p(math.exp(-1.0)))


def test_float32_loses_small_increment_that_float64_keeps() -> None:
    result = float_precision_example()
    assert result["float32"] == 0.0
    assert result["float64"] == 1.0


def test_stable_softmax_rejects_non_finite_logits() -> None:
    with pytest.raises(ValueError, match="finite"):
        stable_softmax([1.0, float("inf")])


def test_logsumexp_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        logsumexp([])
