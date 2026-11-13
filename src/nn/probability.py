"""Probability and sampling helpers for the foundations checkpoints."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def normalize_probabilities(weights: ArrayLike) -> FloatArray:
    """Scale non-negative weights shaped ``(categories,)`` to sum to one."""
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional vector")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("weights must be finite and non-negative")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("weights must have a positive total")
    return values / total


def _validated_distribution(probabilities: ArrayLike) -> FloatArray:
    """Return probabilities after checking a non-empty normalized vector."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional vector")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isclose(np.sum(values), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("probabilities must sum to one")
    return values


def expected_value(values: ArrayLike, probabilities: ArrayLike) -> float:
    """Calculate ``sum(value * probability)`` for equally sized vectors."""
    value_array = np.asarray(values, dtype=np.float64)
    probability_array = _validated_distribution(probabilities)
    if value_array.shape != probability_array.shape:
        raise ValueError("values and probabilities must have the same shape")
    if not np.isfinite(value_array).all():
        raise ValueError("values must be finite")
    return float(np.sum(value_array * probability_array))


def variance(values: ArrayLike, probabilities: ArrayLike) -> float:
    """Calculate the probability-weighted population variance."""
    value_array = np.asarray(values, dtype=np.float64)
    probability_array = _validated_distribution(probabilities)
    if value_array.shape != probability_array.shape:
        raise ValueError("values and probabilities must have the same shape")
    if not np.isfinite(value_array).all():
        raise ValueError("values must be finite")
    mean = float(np.sum(value_array * probability_array))
    return float(np.sum(probability_array * (value_array - mean) ** 2))


def entropy_bits(probabilities: ArrayLike) -> float:
    """Calculate Shannon entropy in bits for a normalized distribution."""
    values = _validated_distribution(probabilities)
    positive = values > 0.0
    return float(-np.sum(values[positive] * np.log2(values[positive])))


def conditional_probability(
    joint_probabilities: ArrayLike, outcome_index: int, condition_index: int
) -> float:
    """Return ``P(row=outcome | column=condition)`` from a joint table.

    The joint table has shape ``(outcomes, conditions)`` and must sum to one.
    """
    joint = np.asarray(joint_probabilities, dtype=np.float64)
    if joint.ndim != 2 or joint.shape[0] == 0 or joint.shape[1] == 0:
        raise ValueError(
            "joint probabilities must be a non-empty two-dimensional table"
        )
    if not np.isfinite(joint).all() or np.any(joint < 0.0):
        raise ValueError("joint probabilities must be finite and non-negative")
    if not np.isclose(np.sum(joint), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("joint probabilities must sum to one")
    if not 0 <= outcome_index < joint.shape[0]:
        raise ValueError("outcome_index is outside the joint table")
    if not 0 <= condition_index < joint.shape[1]:
        raise ValueError("condition_index is outside the joint table")
    condition_probability = float(np.sum(joint[:, condition_index]))
    if condition_probability == 0.0:
        raise ValueError("condition has zero probability")
    return float(joint[outcome_index, condition_index] / condition_probability)


def sample_categorical(
    probabilities: ArrayLike, sample_count: int, seed: int
) -> IntArray:
    """Draw category indices with shape ``(sample_count,)`` reproducibly."""
    distribution = _validated_distribution(probabilities)
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    return np.random.default_rng(seed).choice(
        distribution.size, size=sample_count, p=distribution
    )


def empirical_distribution(samples: ArrayLike, category_count: int) -> FloatArray:
    """Return observed category frequencies for integer sample indices."""
    values = np.asarray(samples)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("samples must be a non-empty one-dimensional array")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("samples must contain integer category indices")
    if category_count < 1 or np.any(values < 0) or np.any(values >= category_count):
        raise ValueError("sample index is outside the requested categories")
    counts = np.bincount(values.astype(np.int64), minlength=category_count)
    return counts.astype(np.float64) / values.size
