"""Loss functions for regression and classification models."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nn.numerics import logsumexp, stable_softmax

FloatArray = NDArray[np.float64]


def _validated_pair(
    predictions: ArrayLike, targets: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Convert same-shaped non-empty finite prediction arrays to float64."""
    prediction_values = np.asarray(predictions, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.float64)
    if prediction_values.shape != target_values.shape:
        raise ValueError("predictions and targets must have the same shape")
    if prediction_values.size == 0:
        raise ValueError("predictions and targets must be non-empty")
    if not np.isfinite(prediction_values).all() or not np.isfinite(target_values).all():
        raise ValueError("predictions and targets must be finite")
    return prediction_values, target_values


def mean_squared_error(predictions: ArrayLike, targets: ArrayLike) -> float:
    """Return the mean squared error over all elements in matching arrays."""
    prediction_values, target_values = _validated_pair(predictions, targets)
    return float(np.mean((prediction_values - target_values) ** 2))


def mean_squared_error_gradient(
    predictions: ArrayLike, targets: ArrayLike
) -> FloatArray:
    """Return MSE derivatives with respect to predictions, same input shape."""
    prediction_values, target_values = _validated_pair(predictions, targets)
    return 2.0 * (prediction_values - target_values) / prediction_values.size


def binary_cross_entropy(
    probabilities: ArrayLike, targets: ArrayLike, epsilon: float = 1e-12
) -> float:
    """Return mean binary cross-entropy for probabilities and 0/1 targets."""
    probability_values, target_values = _validated_pair(probabilities, targets)
    if np.any((probability_values < 0.0) | (probability_values > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    if not np.isin(target_values, (0.0, 1.0)).all():
        raise ValueError("binary targets must contain only 0 or 1")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be between 0 and 0.5")
    clipped = np.clip(probability_values, epsilon, 1.0 - epsilon)
    losses = -(
        target_values * np.log(clipped) + (1.0 - target_values) * np.log1p(-clipped)
    )
    return float(np.mean(losses))


def _validated_categorical(
    probabilities: ArrayLike, targets: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Check class probability and one-hot target matrices shaped ``(batch, classes)``."""
    probability_values, target_values = _validated_pair(probabilities, targets)
    if probability_values.ndim != 2 or probability_values.shape[0] == 0:
        raise ValueError("categorical values must have shape (batch, classes)")
    if np.any((probability_values < 0.0) | (probability_values > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    if np.any((target_values < 0.0) | (target_values > 1.0)):
        raise ValueError("categorical targets must be in [0, 1]")
    if not np.allclose(np.sum(probability_values, axis=1), 1.0, atol=1e-12):
        raise ValueError("each probability row must sum to one")
    if not np.allclose(np.sum(target_values, axis=1), 1.0, atol=1e-12):
        raise ValueError("each one-hot target row must sum to one")
    return probability_values, target_values


def categorical_cross_entropy(
    probabilities: ArrayLike, targets: ArrayLike, epsilon: float = 1e-12
) -> float:
    """Return mean categorical cross-entropy for one-hot targets."""
    probability_values, target_values = _validated_categorical(probabilities, targets)
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    clipped = np.clip(probability_values, epsilon, 1.0)
    return float(-np.mean(np.sum(target_values * np.log(clipped), axis=1)))


def categorical_cross_entropy_from_logits(
    logits: ArrayLike, targets: ArrayLike
) -> float:
    """Return stable mean categorical cross-entropy from logits and one-hot labels."""
    logit_values = np.asarray(logits, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.float64)
    if logit_values.ndim != 2 or logit_values.shape != target_values.shape:
        raise ValueError("logits and targets must have the same (batch, classes) shape")
    if logit_values.shape[0] == 0 or logit_values.shape[1] == 0:
        raise ValueError("logits and targets must be non-empty")
    if not np.isfinite(logit_values).all() or not np.isfinite(target_values).all():
        raise ValueError("logits and targets must be finite")
    if np.any((target_values < 0.0) | (target_values > 1.0)):
        raise ValueError("categorical targets must be in [0, 1]")
    if not np.allclose(np.sum(target_values, axis=1), 1.0, atol=1e-12):
        raise ValueError("each one-hot target row must sum to one")
    normalizers = logsumexp(logit_values, axis=1)
    per_example = normalizers - np.sum(target_values * logit_values, axis=1)
    return float(np.mean(per_example))


def categorical_cross_entropy_logits_gradient(
    logits: ArrayLike, targets: ArrayLike
) -> FloatArray:
    """Return the mean-loss gradient with respect to softmax logits."""
    logit_values = np.asarray(logits, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.float64)
    categorical_cross_entropy_from_logits(logit_values, target_values)
    return (stable_softmax(logit_values, axis=1) - target_values) / logit_values.shape[
        0
    ]
