"""Tests for calculus functions used by a small neural neuron."""

import math

import numpy as np
import pytest

from nn.calculus import (
    mean_squared_error,
    mean_squared_error_gradient,
    sigmoid,
    sigmoid_derivative,
    sigmoid_neuron_mse_gradients,
    sigmoid_neuron_mse_loss,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 0.5), (math.log(3.0), 0.75), (-math.log(3.0), 0.25)],
)
def test_sigmoid_matches_hand_calculated_values(value: float, expected: float) -> None:
    assert sigmoid(value) == pytest.approx(expected, abs=1e-12)


def test_sigmoid_stays_finite_for_large_inputs() -> None:
    assert sigmoid(-1000.0) == 0.0
    assert sigmoid(1000.0) == 1.0


def test_sigmoid_derivative_matches_hand_calculated_values() -> None:
    assert sigmoid_derivative(0.0) == pytest.approx(0.25)
    assert sigmoid_derivative(math.log(3.0)) == pytest.approx(0.1875)


def test_mse_matches_hand_calculation() -> None:
    assert mean_squared_error([0.0, 1.0], [0.0, 0.0]) == pytest.approx(0.5)


def test_mse_gradient_matches_hand_calculation() -> None:
    np.testing.assert_allclose(
        mean_squared_error_gradient([0.0, 1.0], [0.0, 0.0]), [0.0, 1.0]
    )


def test_sigmoid_neuron_gradients_match_finite_differences() -> None:
    inputs = np.array([1.0, 2.0], dtype=np.float64)
    weights = np.array([0.2, -0.3], dtype=np.float64)
    bias = 0.1
    target = 1.0
    weight_gradient, bias_gradient = sigmoid_neuron_mse_gradients(
        inputs, weights, bias, target
    )

    step = 1e-6
    numerical_weights = np.empty_like(weights)
    for index in range(weights.size):
        positive_weights = weights.copy()
        negative_weights = weights.copy()
        positive_weights[index] += step
        negative_weights[index] -= step
        numerical_weights[index] = (
            sigmoid_neuron_mse_loss(inputs, positive_weights, bias, target)
            - sigmoid_neuron_mse_loss(inputs, negative_weights, bias, target)
        ) / (2.0 * step)
    numerical_bias = (
        sigmoid_neuron_mse_loss(inputs, weights, bias + step, target)
        - sigmoid_neuron_mse_loss(inputs, weights, bias - step, target)
    ) / (2.0 * step)

    np.testing.assert_allclose(weight_gradient, numerical_weights, rtol=1e-6, atol=1e-8)
    assert bias_gradient == pytest.approx(numerical_bias, rel=1e-6, abs=1e-8)


def test_mse_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        mean_squared_error([1.0, 2.0], [1.0])
