"""Calculus helpers for the neural-network learning examples."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def sigmoid(value: float) -> float:
    """Return the sigmoid of a finite scalar using a stable calculation."""
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def sigmoid_derivative(value: float) -> float:
    """Return ``sigmoid(value) * (1 - sigmoid(value))``."""
    activation = sigmoid(value)
    return activation * (1.0 - activation)


def _validated_vector_pair(
    predictions: ArrayLike, targets: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Convert two same-shaped non-empty vectors to finite float64 arrays."""
    prediction_values = np.asarray(predictions, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.float64)
    if prediction_values.ndim != 1 or target_values.ndim != 1:
        raise ValueError("predictions and targets must be one-dimensional")
    if prediction_values.shape != target_values.shape:
        raise ValueError("predictions and targets must have the same shape")
    if prediction_values.size == 0:
        raise ValueError("predictions and targets must be non-empty")
    if not np.isfinite(prediction_values).all() or not np.isfinite(target_values).all():
        raise ValueError("predictions and targets must be finite")
    return prediction_values, target_values


def mean_squared_error(predictions: ArrayLike, targets: ArrayLike) -> float:
    """Return mean squared error for two vectors shaped ``(samples,)``."""
    prediction_values, target_values = _validated_vector_pair(predictions, targets)
    return float(np.mean((prediction_values - target_values) ** 2))


def mean_squared_error_gradient(
    predictions: ArrayLike, targets: ArrayLike
) -> FloatArray:
    """Return MSE derivatives with respect to predictions, shape ``(samples,)``."""
    prediction_values, target_values = _validated_vector_pair(predictions, targets)
    return 2.0 * (prediction_values - target_values) / prediction_values.size


def _validated_neuron(
    inputs: ArrayLike, weights: ArrayLike, bias: float, target: float
) -> tuple[FloatArray, FloatArray]:
    """Check a single sigmoid neuron's feature vectors and scalar values."""
    input_values = np.asarray(inputs, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    if input_values.ndim != 1 or weight_values.ndim != 1:
        raise ValueError("inputs and weights must be one-dimensional")
    if input_values.size == 0 or input_values.shape != weight_values.shape:
        raise ValueError("inputs and weights must be non-empty and have the same shape")
    if not np.isfinite(input_values).all() or not np.isfinite(weight_values).all():
        raise ValueError("inputs and weights must be finite")
    if not math.isfinite(bias) or not math.isfinite(target):
        raise ValueError("bias and target must be finite")
    return input_values, weight_values


def sigmoid_neuron_mse_loss(
    inputs: ArrayLike, weights: ArrayLike, bias: float, target: float
) -> float:
    """Return squared error for one sigmoid neuron and one target value."""
    input_values, weight_values = _validated_neuron(inputs, weights, bias, target)
    logit = float(input_values @ weight_values + bias)
    prediction = sigmoid(logit)
    return (prediction - target) ** 2


def sigmoid_neuron_mse_gradients(
    inputs: ArrayLike, weights: ArrayLike, bias: float, target: float
) -> tuple[FloatArray, float]:
    """Return squared-error gradients for weights ``(features,)`` and bias.

    Chain rule: for ``L=(p-y)^2`` and ``p=sigmoid(z)``,
    ``dL/dz = 2(p-y) * p(1-p)``; then ``dL/dw = x * dL/dz`` and
    ``dL/db = dL/dz``.
    """
    input_values, weight_values = _validated_neuron(inputs, weights, bias, target)
    logit = float(input_values @ weight_values + bias)
    prediction = sigmoid(logit)
    loss_gradient = 2.0 * (prediction - target)
    logit_gradient = loss_gradient * prediction * (1.0 - prediction)
    return logit_gradient * input_values, float(logit_gradient)
