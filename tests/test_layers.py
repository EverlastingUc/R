"""Tests for configurable dense layers."""

import numpy as np
import pytest

from nn.layers import DenseLayer


def test_dense_layer_matches_hand_calculated_outputs() -> None:
    layer = DenseLayer(
        input_size=2,
        neuron_count=3,
        activation="identity",
        weights=[[1.0, -1.0, 2.0], [0.5, 2.0, -0.5]],
        bias=[0.1, 0.0, -1.0],
    )
    inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.array([[2.1, 3.0, 0.0], [5.1, 5.0, 3.0]])
    np.testing.assert_allclose(layer.forward(inputs), expected, atol=1e-12)


def test_dense_layer_output_shape_uses_configured_neuron_count() -> None:
    layer = DenseLayer(input_size=3, neuron_count=5, seed=13)
    output = layer.forward(np.zeros((7, 3), dtype=np.float64))
    assert output.shape == (7, 5)


def test_dense_layer_initialization_is_reproducible() -> None:
    first = DenseLayer(input_size=3, neuron_count=4, seed=19)
    second = DenseLayer(input_size=3, neuron_count=4, seed=19)
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.bias, second.bias)


def test_dense_layer_rejects_wrong_input_feature_count() -> None:
    layer = DenseLayer(input_size=2, neuron_count=3)
    with pytest.raises(ValueError, match="feature count"):
        layer.forward(np.ones((4, 1)))


def test_dense_layer_rejects_invalid_weight_shape() -> None:
    with pytest.raises(ValueError, match="weights must have shape"):
        DenseLayer(
            input_size=2,
            neuron_count=3,
            weights=[[1.0, 2.0]],
            bias=[0.0, 0.0, 0.0],
        )


def test_dense_layer_rejects_single_missing_parameter_array() -> None:
    with pytest.raises(ValueError, match="provide both"):
        DenseLayer(input_size=2, neuron_count=3, weights=np.ones((2, 3)))
