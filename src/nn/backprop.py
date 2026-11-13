"""Backpropagation for mean squared error in dense feedforward networks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nn.activations import activation_derivative
from nn.losses import mean_squared_error, mean_squared_error_gradient
from nn.network import FeedForwardNetwork

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BackpropResult:
    """Loss and parameter gradients ordered from first layer to last."""

    loss: float
    weight_gradients: list[FloatArray]
    bias_gradients: list[FloatArray]


def _softmax_backward(gradient: FloatArray, probabilities: FloatArray) -> FloatArray:
    """Apply a row-wise softmax Jacobian transpose without building the matrix."""
    projection = np.sum(gradient * probabilities, axis=1, keepdims=True)
    return probabilities * (gradient - projection)


def backprop_mse(
    network: FeedForwardNetwork, inputs: ArrayLike, targets: ArrayLike
) -> BackpropResult:
    """Calculate MSE gradients for every dense-layer weight and bias.

    Inputs have shape ``(batch_size, input_size)``; targets must match the
    output shape. Returned weight/bias gradients match their parameter shapes.
    """
    output, caches = network.forward_with_cache(inputs)
    target_values = np.asarray(targets, dtype=np.float64)
    if target_values.shape != output.shape:
        raise ValueError("targets must have the same shape as network outputs")
    if not np.isfinite(target_values).all():
        raise ValueError("targets must be finite")

    loss = mean_squared_error(output, target_values)
    activation_gradient = mean_squared_error_gradient(output, target_values)
    weight_gradients: list[FloatArray] = []
    bias_gradients: list[FloatArray] = []

    for layer, cache in reversed(list(zip(network.layers, caches, strict=True))):
        if layer.activation == "softmax":
            preactivation_gradient = _softmax_backward(
                activation_gradient, cache.output
            )
        else:
            derivative = activation_derivative(
                layer.activation,
                cache.preactivation,
                negative_slope=layer.negative_slope,
            )
            preactivation_gradient = activation_gradient * derivative

        weight_gradients.append(cache.inputs.T @ preactivation_gradient)
        bias_gradients.append(np.sum(preactivation_gradient, axis=0, keepdims=True))
        activation_gradient = preactivation_gradient @ layer.weights.T

    weight_gradients.reverse()
    bias_gradients.reverse()
    return BackpropResult(
        loss=loss,
        weight_gradients=weight_gradients,
        bias_gradients=bias_gradients,
    )
