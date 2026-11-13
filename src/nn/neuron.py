"""A single artificial neuron implemented with NumPy."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

Activation = Literal["identity", "sigmoid"]


def artificial_neuron(
    inputs: ArrayLike,
    weights: ArrayLike,
    bias: float = 0.0,
    activation: Activation = "sigmoid",
) -> float:
    """Calculate one neuron's output from one input vector.

    Args:
        inputs: One-dimensional input vector with shape ``(features,)``.
        weights: One-dimensional weight vector with the same shape.
        bias: A finite number added to the weighted sum.
        activation: Either ``"identity"`` (return the weighted sum) or
            ``"sigmoid"`` (squash it to the range 0–1).

    Returns:
        A Python float containing the neuron's output.

    The neuron first computes ``z = sum(inputs * weights) + bias``. For
    sigmoid activation it returns ``1 / (1 + exp(-z))`` using a stable
    equivalent calculation for negative values of ``z``.
    """
    input_values = np.asarray(inputs, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)

    if input_values.ndim != 1 or weight_values.ndim != 1:
        raise ValueError("inputs and weights must be one-dimensional vectors")
    if input_values.size == 0:
        raise ValueError("inputs and weights must be non-empty")
    if input_values.shape != weight_values.shape:
        raise ValueError("inputs and weights must have the same length")
    if not np.isfinite(input_values).all() or not np.isfinite(weight_values).all():
        raise ValueError("inputs and weights must contain only finite values")
    if not math.isfinite(float(bias)):
        raise ValueError("bias must be finite")
    if activation not in ("identity", "sigmoid"):
        raise ValueError("activation must be 'identity' or 'sigmoid'")

    # Formula: z = x·w + b. The guard turns arithmetic overflow into a clear
    # input error rather than allowing an infinite neuron output to escape.
    with np.errstate(over="ignore", invalid="ignore"):
        weighted_sum = float(np.dot(input_values, weight_values) + bias)
    if not math.isfinite(weighted_sum):
        raise ValueError("weighted sum must be finite")

    if activation == "identity":
        return weighted_sum
    if weighted_sum >= 0.0:
        return 1.0 / (1.0 + math.exp(-weighted_sum))
    exponent = math.exp(weighted_sum)
    return exponent / (1.0 + exponent)
