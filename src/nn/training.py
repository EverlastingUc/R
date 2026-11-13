"""Small, readable training utilities for NumPy feedforward networks."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from nn.backprop import backprop_mse
from nn.layers import DenseLayer
from nn.network import FeedForwardNetwork


def gradient_descent_step(
    network: FeedForwardNetwork,
    inputs: ArrayLike,
    targets: ArrayLike,
    learning_rate: float,
) -> float:
    """Apply one full-batch gradient-descent update to a dense network.

    ``inputs`` has shape ``(batch_size, input_size)`` and ``targets`` has the
    same shape as the network output. Parameters are updated by
    ``parameter -= learning_rate * gradient``. Returns the loss before update.
    """
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")

    result = backprop_mse(network, inputs, targets)
    parameter_updates: list[tuple[DenseLayer, np.ndarray, np.ndarray]] = []
    for layer, weight_gradient, bias_gradient in zip(
        network.layers,
        result.weight_gradients,
        result.bias_gradients,
        strict=True,
    ):
        # Stage every update first so invalid gradients cannot partially mutate
        # a network.
        new_weights = layer.weights - learning_rate * weight_gradient
        new_bias = layer.bias - learning_rate * bias_gradient
        if not np.isfinite(new_weights).all() or not np.isfinite(new_bias).all():
            raise ValueError("gradient update produced non-finite parameters")
        parameter_updates.append((layer, new_weights, new_bias))

    for layer, new_weights, new_bias in parameter_updates:
        layer.weights[...] = new_weights
        layer.bias[...] = new_bias

    return result.loss
