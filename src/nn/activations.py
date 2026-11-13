"""Activation functions and derivatives for NumPy neural-network layers."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nn.numerics import stable_softmax

FloatArray = NDArray[np.float64]
ActivationName = Literal["identity", "sigmoid", "tanh", "relu", "leaky_relu", "softmax"]


def _validated_values(values: ArrayLike) -> FloatArray:
    """Convert activation inputs to a non-empty finite float64 array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.size == 0:
        raise ValueError("activation input must have at least one value")
    if not np.isfinite(array).all():
        raise ValueError("activation input must be finite")
    return array


def _sigmoid_array(values: FloatArray) -> FloatArray:
    """Apply sigmoid elementwise without overflowing."""
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponentials = np.exp(values[~positive])
    result[~positive] = exponentials / (1.0 + exponentials)
    return result


def apply_activation(
    name: ActivationName,
    values: ArrayLike,
    axis: int = -1,
    negative_slope: float = 0.01,
) -> FloatArray:
    """Apply an activation to a tensor, preserving its shape.

    Softmax is normalized along ``axis``. Other activations operate
    elementwise. ``negative_slope`` is used only by leaky ReLU.
    """
    array = _validated_values(values)
    if name == "identity":
        return array.copy()
    if name == "sigmoid":
        return _sigmoid_array(array)
    if name == "tanh":
        return np.tanh(array)
    if name == "relu":
        return np.maximum(array, 0.0)
    if name == "leaky_relu":
        if not np.isfinite(negative_slope) or negative_slope <= 0.0:
            raise ValueError("negative_slope must be positive and finite")
        return np.where(array >= 0.0, array, negative_slope * array)
    if name == "softmax":
        return stable_softmax(array, axis=axis)
    raise ValueError(f"unknown activation: {name}")


def activation_derivative(
    name: ActivationName,
    preactivation: ArrayLike,
    negative_slope: float = 0.01,
) -> FloatArray:
    """Return elementwise derivatives with respect to preactivation values."""
    values = _validated_values(preactivation)
    if name == "identity":
        return np.ones_like(values)
    if name == "sigmoid":
        output = _sigmoid_array(values)
        return output * (1.0 - output)
    if name == "tanh":
        output = np.tanh(values)
        return 1.0 - output**2
    if name == "relu":
        return (values > 0.0).astype(np.float64)
    if name == "leaky_relu":
        if not np.isfinite(negative_slope) or negative_slope <= 0.0:
            raise ValueError("negative_slope must be positive and finite")
        return np.where(values >= 0.0, 1.0, negative_slope)
    if name == "softmax":
        raise ValueError("softmax has a matrix derivative; use softmax_jacobian")
    raise ValueError(f"unknown activation: {name}")


def softmax_jacobian(logits: ArrayLike) -> FloatArray:
    """Return the softmax Jacobian for one vector shaped ``(classes,)``."""
    values = _validated_values(logits)
    if values.ndim != 1:
        raise ValueError("softmax_jacobian expects a one-dimensional logits vector")
    probabilities = stable_softmax(values)
    return np.diag(probabilities) - np.outer(probabilities, probabilities)
