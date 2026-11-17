"""Mini-batch classification training for Entropy feedforward networks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nn.activations import activation_derivative
from nn.backprop import BackpropResult
from nn.losses import categorical_cross_entropy_from_logits
from nn.network import FeedForwardNetwork
from nn.numerics import stable_softmax

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class StatefulOptimizer(Protocol):
    """The optimizer operation required by the classification trainer."""

    def step(
        self, parameters: Sequence[FloatArray], gradients: Sequence[FloatArray]
    ) -> None:
        """Update parameters in place."""


@dataclass(frozen=True)
class ClassificationEpochMetrics:
    """Full-split classification metrics measured after one training epoch."""

    epoch: int
    training_loss: float
    validation_loss: float
    training_accuracy: float
    validation_accuracy: float


def _labels_to_one_hot(
    labels: ArrayLike, class_count: int, name: str
) -> tuple[IntArray, FloatArray]:
    """Validate integer class IDs and convert them to one-hot rows."""
    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} labels must be a non-empty one-dimensional array")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"{name} labels must contain integer class IDs")
    class_ids = values.astype(np.int64, copy=False)
    if np.any(class_ids < 0) or np.any(class_ids >= class_count):
        raise ValueError(
            f"{name} labels must contain class IDs in [0, {class_count - 1}]"
        )
    return class_ids, np.eye(class_count, dtype=np.float64)[class_ids]


def backprop_categorical_cross_entropy(
    network: FeedForwardNetwork, inputs: ArrayLike, one_hot_targets: ArrayLike
) -> BackpropResult:
    """Calculate softmax-cross-entropy gradients for every dense layer."""
    if not network.layers or network.layers[-1].activation != "softmax":
        raise ValueError("classification output layer must use softmax")

    output, caches = network.forward_with_cache(inputs)
    targets = np.asarray(one_hot_targets, dtype=np.float64)
    if targets.shape != output.shape:
        raise ValueError("one-hot targets must match the network output shape")
    if not np.isfinite(targets).all():
        raise ValueError("one-hot targets must be finite")
    if np.any((targets < 0.0) | (targets > 1.0)) or not np.allclose(
        targets.sum(axis=1), 1.0, atol=1e-12
    ):
        raise ValueError("each target row must be a valid one-hot distribution")

    logits = caches[-1].preactivation
    loss = categorical_cross_entropy_from_logits(logits, targets)
    gradient = (output - targets) / output.shape[0]
    weight_gradients: list[FloatArray] = []
    bias_gradients: list[FloatArray] = []

    for index in range(len(network.layers) - 1, -1, -1):
        layer = network.layers[index]
        cache = caches[index]
        if index != len(network.layers) - 1:
            if layer.activation == "softmax":
                projection = np.sum(gradient * cache.output, axis=1, keepdims=True)
                gradient = cache.output * (gradient - projection)
            else:
                gradient = gradient * activation_derivative(
                    layer.activation,
                    cache.preactivation,
                    negative_slope=layer.negative_slope,
                )
        weight_gradients.append(cache.inputs.T @ gradient)
        bias_gradients.append(np.sum(gradient, axis=0, keepdims=True))
        if index > 0:
            gradient = gradient @ layer.weights.T

    weight_gradients.reverse()
    bias_gradients.reverse()
    return BackpropResult(loss, weight_gradients, bias_gradients)


class ClassificationTrainer:
    """Train a softmax classifier with the project's stateful optimizers."""

    def __init__(
        self,
        network: FeedForwardNetwork,
        optimizer: StatefulOptimizer,
        batch_size: int = 32,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        """Create a deterministic mini-batch classifier trainer."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if len(network.layers) < 3 or network.layers[-1].activation != "softmax":
            raise ValueError(
                "classifier requires at least two hidden layers and a softmax output"
            )
        self.network = network
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = bool(shuffle)
        self.completed_epochs = 0
        self._random = np.random.default_rng(seed)
        self.class_count = network.layer_sizes[-1]

    def _dataset(
        self, inputs: ArrayLike, labels: ArrayLike, name: str
    ) -> tuple[FloatArray, IntArray, FloatArray]:
        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError(f"{name} inputs must be a non-empty two-dimensional array")
        if values.shape[1] != self.network.layer_sizes[0]:
            raise ValueError(f"{name} feature count does not match the network")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} inputs must be finite")
        class_ids, one_hot = _labels_to_one_hot(labels, self.class_count, name)
        if len(class_ids) != len(values):
            raise ValueError(f"{name} inputs and labels must have matching rows")
        return values, class_ids, one_hot

    def _evaluate(
        self, inputs: FloatArray, labels: IntArray, one_hot: FloatArray
    ) -> tuple[float, float]:
        _, caches = self.network.forward_with_cache(inputs)
        logits = caches[-1].preactivation
        loss = categorical_cross_entropy_from_logits(logits, one_hot)
        predictions = np.argmax(stable_softmax(logits, axis=1), axis=1)
        return loss, float(np.mean(predictions == labels))

    def fit(
        self,
        train_inputs: ArrayLike,
        train_labels: ArrayLike,
        validation_inputs: ArrayLike,
        validation_labels: ArrayLike,
        epochs: int,
    ) -> list[ClassificationEpochMetrics]:
        """Train for a fixed number of epochs and report held-out metrics."""
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        train_x, train_y, train_one_hot = self._dataset(
            train_inputs, train_labels, "training"
        )
        validation_x, validation_y, validation_one_hot = self._dataset(
            validation_inputs, validation_labels, "validation"
        )

        history: list[ClassificationEpochMetrics] = []
        for _ in range(epochs):
            order = np.arange(len(train_x))
            if self.shuffle:
                self._random.shuffle(order)
            for start in range(0, len(order), self.batch_size):
                indices = order[start : start + self.batch_size]
                result = backprop_categorical_cross_entropy(
                    self.network, train_x[indices], train_one_hot[indices]
                )
                parameters: list[FloatArray] = []
                gradients: list[FloatArray] = []
                for layer, weight_gradient, bias_gradient in zip(
                    self.network.layers,
                    result.weight_gradients,
                    result.bias_gradients,
                    strict=True,
                ):
                    parameters.extend((layer.weights, layer.bias))
                    gradients.extend((weight_gradient, bias_gradient))
                self.optimizer.step(parameters, gradients)

            self.completed_epochs += 1
            training_loss, training_accuracy = self._evaluate(
                train_x, train_y, train_one_hot
            )
            validation_loss, validation_accuracy = self._evaluate(
                validation_x, validation_y, validation_one_hot
            )
            if not np.isfinite([training_loss, validation_loss]).all():
                raise ValueError("classification training produced a non-finite loss")
            history.append(
                ClassificationEpochMetrics(
                    epoch=self.completed_epochs,
                    training_loss=training_loss,
                    validation_loss=validation_loss,
                    training_accuracy=training_accuracy,
                    validation_accuracy=validation_accuracy,
                )
            )
        return history

    def predict_proba(self, inputs: ArrayLike) -> FloatArray:
        """Return one softmax probability row per input sample."""
        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.network.layer_sizes[0]:
            raise ValueError("inputs must match the network feature shape")
        if values.shape[0] == 0 or not np.isfinite(values).all():
            raise ValueError("inputs must be non-empty and finite")
        return self.network.forward(values)

    def predict(self, inputs: ArrayLike) -> IntArray:
        """Return the most likely integer class ID for each input sample."""
        return np.argmax(self.predict_proba(inputs), axis=1).astype(np.int64)
