"""Feedforward networks assembled from configurable dense layers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nn.activations import ActivationName
from nn.layers import DenseLayer

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LayerCache:
    """Values needed later to calculate gradients for one layer."""

    inputs: FloatArray
    preactivation: FloatArray
    output: FloatArray


class FeedForwardNetwork:
    """A feedforward network with at least two hidden layers.

    ``layer_sizes`` lists the input size followed by each dense layer's neuron
    count. ``activations`` has one name for each connection between sizes.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activations: Sequence[ActivationName],
        seed: int = 0,
    ) -> None:
        """Create deterministic dense layers from sizes and activations."""
        if len(layer_sizes) < 4:
            raise ValueError("network must contain at least two hidden layers")
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("provide one activation per layer connection")
        if any(size < 1 for size in layer_sizes):
            raise ValueError("all layer sizes must be positive")

        self.layer_sizes = tuple(layer_sizes)
        self.activations = tuple(activations)
        self.layers = [
            DenseLayer(
                input_size=input_size,
                neuron_count=output_size,
                activation=activation,
                seed=seed + index,
            )
            for index, (input_size, output_size, activation) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:], activations, strict=True)
            )
        ]

    def forward_with_cache(
        self, inputs: ArrayLike
    ) -> tuple[FloatArray, list[LayerCache]]:
        """Return the network output and a cache for each dense layer."""
        values = np.asarray(inputs, dtype=np.float64)
        caches: list[LayerCache] = []
        for layer in self.layers:
            output, layer_inputs, preactivation = layer.forward_with_cache(values)
            caches.append(
                LayerCache(
                    inputs=layer_inputs,
                    preactivation=preactivation,
                    output=output,
                )
            )
            values = output
        return values, caches

    def forward(self, inputs: ArrayLike) -> FloatArray:
        """Calculate outputs with shape ``(batch_size, output_size)``."""
        output, _ = self.forward_with_cache(inputs)
        return output
