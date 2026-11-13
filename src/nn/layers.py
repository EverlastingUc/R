"""Configurable dense layers built from NumPy operations."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nn.activations import ActivationName, apply_activation

FloatArray = NDArray[np.float64]


class DenseLayer:
    """A fully connected layer with configurable neurons and activation.

    Weights have shape ``(input_size, neuron_count)`` and biases have shape
    ``(1, neuron_count)``. Inputs to ``forward`` have shape
    ``(batch_size, input_size)``.
    """

    def __init__(
        self,
        input_size: int,
        neuron_count: int,
        activation: ActivationName = "relu",
        seed: int = 0,
        weights: ArrayLike | None = None,
        bias: ArrayLike | None = None,
        negative_slope: float = 0.01,
    ) -> None:
        """Create a layer with Xavier-uniform weights or supplied parameters."""
        if input_size < 1 or neuron_count < 1:
            raise ValueError("input_size and neuron_count must be positive")
        if (weights is None) != (bias is None):
            raise ValueError("provide both weights and bias, or neither")
        self.input_size = input_size
        self.neuron_count = neuron_count
        self.activation = activation
        self.negative_slope = negative_slope

        if weights is None:
            limit = math.sqrt(6.0 / (input_size + neuron_count))
            random = np.random.default_rng(seed)
            self.weights = random.uniform(
                -limit, limit, size=(input_size, neuron_count)
            )
            self.bias = np.zeros((1, neuron_count), dtype=np.float64)
        else:
            self.weights = np.asarray(weights, dtype=np.float64)
            bias_values = np.asarray(bias, dtype=np.float64)
            if self.weights.shape != (input_size, neuron_count):
                raise ValueError("weights must have shape (input_size, neuron_count)")
            if bias_values.shape == (neuron_count,):
                bias_values = bias_values.reshape(1, neuron_count)
            if bias_values.shape != (1, neuron_count):
                raise ValueError("bias must have shape (1, neuron_count)")
            self.bias = bias_values.copy()

        if not np.isfinite(self.weights).all() or not np.isfinite(self.bias).all():
            raise ValueError("weights and bias must be finite")

    def forward_with_cache(
        self, inputs: ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return activation output, input batch, and preactivation values."""
        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.input_size:
            raise ValueError(f"input feature count must be {self.input_size}")
        if values.shape[0] == 0 or not np.isfinite(values).all():
            raise ValueError("inputs must be non-empty and finite")
        preactivation = values @ self.weights + self.bias
        output = apply_activation(
            self.activation,
            preactivation,
            negative_slope=self.negative_slope,
        )
        return output, values, preactivation

    def forward(self, inputs: ArrayLike) -> FloatArray:
        """Calculate the layer output with shape ``(batch_size, neuron_count)``."""
        output, _, _ = self.forward_with_cache(inputs)
        return output
