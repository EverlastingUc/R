"""Gradient checks for multilayer backpropagation."""

import numpy as np
import pytest

from nn.backprop import backprop_mse
from nn.gradcheck import gradient_check
from nn.network import FeedForwardNetwork


def make_network(activation: str) -> FeedForwardNetwork:
    network = FeedForwardNetwork(
        [2, 3, 2, 2],
        [activation, activation, activation],
        seed=33,
    )
    random = np.random.default_rng(33)
    for layer in network.layers:
        layer.weights = random.uniform(
            0.05, 0.2, size=(layer.input_size, layer.neuron_count)
        )
        layer.bias = np.full((1, layer.neuron_count), 0.1, dtype=np.float64)
    return network


@pytest.mark.parametrize(
    "activation", ["identity", "sigmoid", "tanh", "relu", "leaky_relu", "softmax"]
)
def test_gradient_check_passes_for_every_activation(activation: str) -> None:
    network = make_network(activation)
    inputs = np.array([[0.2, 0.7], [0.6, 0.1]], dtype=np.float64)
    targets = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float64)

    result = gradient_check(network, inputs, targets)

    assert result.passed
    assert result.max_relative_error <= result.tolerance
    assert result.parameter_count == sum(
        layer.weights.size + layer.bias.size for layer in network.layers
    )


def test_backprop_returns_finite_gradients_for_each_layer() -> None:
    network = make_network("tanh")
    inputs = np.array([[0.2, 0.7], [0.6, 0.1]], dtype=np.float64)
    targets = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float64)
    result = backprop_mse(network, inputs, targets)

    assert len(result.weight_gradients) == len(network.layers)
    assert len(result.bias_gradients) == len(network.layers)
    assert all(np.isfinite(value).all() for value in result.weight_gradients)
    assert all(np.isfinite(value).all() for value in result.bias_gradients)


def test_gradient_check_restores_network_parameters() -> None:
    network = make_network("sigmoid")
    original_weights = [layer.weights.copy() for layer in network.layers]
    original_biases = [layer.bias.copy() for layer in network.layers]
    inputs = np.array([[0.2, 0.7]], dtype=np.float64)
    targets = np.array([[0.1, 0.9]], dtype=np.float64)

    gradient_check(network, inputs, targets)

    for layer, weights, bias in zip(
        network.layers, original_weights, original_biases, strict=True
    ):
        np.testing.assert_array_equal(layer.weights, weights)
        np.testing.assert_array_equal(layer.bias, bias)
