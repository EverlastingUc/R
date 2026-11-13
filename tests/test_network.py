"""Tests for a feedforward network with multiple hidden layers."""

import numpy as np
import pytest

from nn.network import FeedForwardNetwork


def test_network_with_two_hidden_layers_runs_a_batch() -> None:
    network = FeedForwardNetwork(
        layer_sizes=[2, 4, 3, 1],
        activations=["tanh", "relu", "sigmoid"],
        seed=123,
    )
    inputs = np.ones((5, 2), dtype=np.float64)
    output, caches = network.forward_with_cache(inputs)

    assert len(network.layers) == 3
    assert output.shape == (5, 1)
    assert len(caches) == 3
    assert all(np.isfinite(output))
    assert caches[0].preactivation.shape == (5, 4)
    assert caches[1].preactivation.shape == (5, 3)


def test_network_initialization_is_reproducible() -> None:
    first = FeedForwardNetwork([2, 3, 2, 1], ["relu", "tanh", "sigmoid"], seed=8)
    second = FeedForwardNetwork([2, 3, 2, 1], ["relu", "tanh", "sigmoid"], seed=8)
    for first_layer, second_layer in zip(first.layers, second.layers, strict=True):
        np.testing.assert_array_equal(first_layer.weights, second_layer.weights)
        np.testing.assert_array_equal(first_layer.bias, second_layer.bias)


def test_network_requires_at_least_two_hidden_layers() -> None:
    with pytest.raises(ValueError, match="two hidden layers"):
        FeedForwardNetwork([2, 3, 1], ["relu", "sigmoid"])


def test_network_requires_one_activation_per_connection() -> None:
    with pytest.raises(ValueError, match="one activation per layer"):
        FeedForwardNetwork([2, 3, 2, 1], ["relu", "sigmoid"])
