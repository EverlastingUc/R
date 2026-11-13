"""A small binary classifier built with NumPy and backpropagation."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class EntropyNetwork:
    """A one-hidden-layer neural network for binary classification.

    Inputs have shape ``(batch_size, input_size)`` and targets have shape
    ``(batch_size, 1)``. The network uses sigmoid activations and binary
    cross-entropy loss. Training uses full-batch gradient descent.
    """

    def __init__(self, input_size: int, hidden_size: int, seed: int = 0) -> None:
        """Create a network with reproducible Xavier-uniform weights."""
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be positive")

        random = np.random.default_rng(seed)
        first_limit = math.sqrt(6.0 / (input_size + hidden_size))
        second_limit = math.sqrt(6.0 / (hidden_size + 1))

        self.weights_1: FloatArray = random.uniform(
            -first_limit, first_limit, size=(input_size, hidden_size)
        )
        self.bias_1: FloatArray = np.zeros((1, hidden_size), dtype=np.float64)
        self.weights_2: FloatArray = random.uniform(
            -second_limit, second_limit, size=(hidden_size, 1)
        )
        self.bias_2: FloatArray = np.zeros((1, 1), dtype=np.float64)
        self.input_size = input_size

    @staticmethod
    def _sigmoid(values: FloatArray) -> FloatArray:
        """Compute sigmoid without overflowing for large-magnitude inputs."""
        result = np.empty_like(values)
        positive = values >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exponentials = np.exp(values[~positive])
        result[~positive] = exponentials / (1.0 + exponentials)
        return result

    def _validate_inputs(self, inputs: FloatArray) -> FloatArray:
        """Convert inputs to float64 and check their batch and feature axes."""
        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.input_size:
            raise ValueError(f"inputs must have shape (batch_size, {self.input_size})")
        if values.shape[0] == 0 or not np.isfinite(values).all():
            raise ValueError("inputs must be non-empty and finite")
        return values

    def _validate_targets(self, targets: FloatArray, batch_size: int) -> FloatArray:
        """Convert binary targets to float64 and validate their shape and values."""
        values = np.asarray(targets, dtype=np.float64)
        if values.shape != (batch_size, 1):
            raise ValueError("targets must have shape (batch_size, 1)")
        if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
            raise ValueError("targets must contain only finite 0 or 1 values")
        return values

    def predict_proba(self, inputs: FloatArray) -> FloatArray:
        """Return positive-class probabilities with shape ``(batch_size, 1)``."""
        values = self._validate_inputs(inputs)
        hidden = self._sigmoid(values @ self.weights_1 + self.bias_1)
        logits = hidden @ self.weights_2 + self.bias_2
        return self._sigmoid(logits)

    def predict(self, inputs: FloatArray) -> NDArray[np.int64]:
        """Return class predictions (0 or 1) with shape ``(batch_size, 1)``."""
        return (self.predict_proba(inputs) >= 0.5).astype(np.int64)

    def fit(
        self,
        inputs: FloatArray,
        targets: FloatArray,
        epochs: int = 20_000,
        learning_rate: float = 1.0,
    ) -> FloatArray:
        """Train on a full batch and return one binary cross-entropy per epoch.

        The output-layer derivative simplifies to ``(probabilities - targets)``
        for sigmoid activation with binary cross-entropy. Each gradient is
        averaged over the batch before the parameters are updated.
        """
        values = self._validate_inputs(inputs)
        labels = self._validate_targets(targets, values.shape[0])
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive finite number")

        sample_count = values.shape[0]
        losses = np.empty(epochs, dtype=np.float64)

        for epoch in range(epochs):
            hidden_logits = values @ self.weights_1 + self.bias_1
            hidden = self._sigmoid(hidden_logits)
            output_logits = hidden @ self.weights_2 + self.bias_2
            probabilities = self._sigmoid(output_logits)

            # Stable BCE-with-logits: max(z, 0) - z*y + log(1 + exp(-|z|)).
            losses[epoch] = np.mean(
                np.maximum(output_logits, 0.0)
                - output_logits * labels
                + np.log1p(np.exp(-np.abs(output_logits)))
            )

            output_gradient = (probabilities - labels) / sample_count
            weights_2_gradient = hidden.T @ output_gradient
            bias_2_gradient = np.sum(output_gradient, axis=0, keepdims=True)

            hidden_gradient = output_gradient @ self.weights_2.T
            hidden_delta = hidden_gradient * hidden * (1.0 - hidden)
            weights_1_gradient = values.T @ hidden_delta
            bias_1_gradient = np.sum(hidden_delta, axis=0, keepdims=True)

            self.weights_2 -= learning_rate * weights_2_gradient
            self.bias_2 -= learning_rate * bias_2_gradient
            self.weights_1 -= learning_rate * weights_1_gradient
            self.bias_1 -= learning_rate * bias_1_gradient

        return losses


def main() -> None:
    """Train the network on XOR and print its probabilities and predictions."""
    inputs = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64
    )
    targets = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float64)

    network = EntropyNetwork(input_size=2, hidden_size=4, seed=7)
    losses = network.fit(inputs, targets)

    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss:   {losses[-1]:.6f}")
    print("Probabilities:")
    print(network.predict_proba(inputs))
    print("Predictions:")
    print(network.predict(inputs))


if __name__ == "__main__":
    main()
