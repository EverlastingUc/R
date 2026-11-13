"""Tests for the basic gradient-descent training step."""

import numpy as np
import pytest

from nn.losses import mean_squared_error
from nn.network import FeedForwardNetwork
from nn.training import gradient_descent_step


def make_regression_network() -> FeedForwardNetwork:
    network = FeedForwardNetwork([1, 2, 2, 1], ["identity"] * 3, seed=17)
    network.layers[0].weights = np.array([[0.5, -0.3]], dtype=np.float64)
    network.layers[0].bias = np.zeros((1, 2), dtype=np.float64)
    network.layers[1].weights = np.array([[0.8, 0.0], [0.0, 0.6]], dtype=np.float64)
    network.layers[1].bias = np.zeros((1, 2), dtype=np.float64)
    network.layers[2].weights = np.array([[0.7], [0.4]], dtype=np.float64)
    network.layers[2].bias = np.zeros((1, 1), dtype=np.float64)
    return network


def test_gradient_descent_step_reduces_regression_loss() -> None:
    network = make_regression_network()
    inputs = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    targets = 2.0 * inputs
    loss_before = mean_squared_error(network.forward(inputs), targets)

    reported_loss = gradient_descent_step(network, inputs, targets, 0.01)

    loss_after = mean_squared_error(network.forward(inputs), targets)
    assert reported_loss == pytest.approx(loss_before)
    assert loss_after < loss_before
    assert np.isfinite(loss_after)


def test_invalid_learning_rate_does_not_change_parameters() -> None:
    network = make_regression_network()
    original = [(layer.weights.copy(), layer.bias.copy()) for layer in network.layers]
    inputs = np.array([[1.0]], dtype=np.float64)
    targets = np.array([[2.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="learning_rate"):
        gradient_descent_step(network, inputs, targets, 0.0)

    for layer, (weights, bias) in zip(network.layers, original, strict=True):
        np.testing.assert_array_equal(layer.weights, weights)
        np.testing.assert_array_equal(layer.bias, bias)
