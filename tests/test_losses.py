"""Hand-calculated checks for regression and classification losses."""

import math

import numpy as np
import pytest

from nn.losses import (
    binary_cross_entropy,
    categorical_cross_entropy,
    categorical_cross_entropy_from_logits,
    categorical_cross_entropy_logits_gradient,
    mean_squared_error,
    mean_squared_error_gradient,
)


def test_mean_squared_error_matches_hand_calculation() -> None:
    predictions = [[1.0, 3.0], [2.0, 4.0]]
    targets = [[0.0, 1.0], [2.0, 2.0]]
    assert mean_squared_error(predictions, targets) == pytest.approx(2.25)


def test_mean_squared_error_gradient_matches_hand_calculation() -> None:
    gradient = mean_squared_error_gradient([[1.0, 3.0]], [[0.0, 1.0]])
    np.testing.assert_allclose(gradient, [[1.0, 2.0]])


def test_binary_cross_entropy_matches_hand_calculation() -> None:
    result = binary_cross_entropy([[0.8], [0.25]], [[1.0], [0.0]])
    expected = (-math.log(0.8) - math.log(0.75)) / 2.0
    assert result == pytest.approx(expected)


def test_categorical_cross_entropy_matches_hand_calculation() -> None:
    probabilities = [[0.7, 0.3], [0.2, 0.8]]
    targets = [[1.0, 0.0], [0.0, 1.0]]
    expected = (-math.log(0.7) - math.log(0.8)) / 2.0
    assert categorical_cross_entropy(probabilities, targets) == pytest.approx(expected)


def test_cross_entropy_from_logits_is_stable_for_large_values() -> None:
    logits = np.array([[1000.0, 1001.0, 999.0]])
    targets = np.array([[0.0, 1.0, 0.0]])
    result = categorical_cross_entropy_from_logits(logits, targets)
    expected = math.log(1.0 + math.exp(-1.0) + math.exp(-2.0))
    assert result == pytest.approx(expected)


def test_cross_entropy_logits_gradient_has_expected_shape_and_values() -> None:
    logits = np.array([[0.0, 0.0]])
    targets = np.array([[1.0, 0.0]])
    gradient = categorical_cross_entropy_logits_gradient(logits, targets)
    np.testing.assert_allclose(gradient, [[-0.5, 0.5]])


def test_rejects_mismatched_prediction_and_target_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        mean_squared_error([[0.0, 1.0]], [[0.0]])


def test_rejects_non_one_hot_categorical_targets() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        categorical_cross_entropy([[0.5, 0.5]], [[1.0, 1.0]])
