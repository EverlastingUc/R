"""Numerically stable operations used by later network checkpoints."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def stable_softmax(logits: ArrayLike, axis: int = -1) -> FloatArray:
    """Return stable softmax probabilities along ``axis`` of an input tensor."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 0 or values.size == 0:
        raise ValueError("logits must be a non-empty array with at least one axis")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")

    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = values - maximum
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def logsumexp(
    values: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> float | FloatArray:
    """Compute log(sum(exp(values))) without overflowing for large values."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.size == 0:
        raise ValueError("values must be a non-empty array with at least one axis")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")

    maximum = np.max(array, axis=axis, keepdims=True)
    shifted_sum = np.sum(np.exp(array - maximum), axis=axis, keepdims=True)
    result = np.log(shifted_sum) + maximum
    if keepdims:
        return result
    if axis is None:
        return float(result.reshape(()))
    return np.squeeze(result, axis=axis)


def float_precision_example() -> dict[str, float]:
    """Show how float32 and float64 handle ``large + 1 - large``."""
    large_float32 = np.float32(1e8)
    result_float32 = float((large_float32 + np.float32(1.0)) - large_float32)
    large_float64 = np.float64(1e8)
    result_float64 = float((large_float64 + np.float64(1.0)) - large_float64)
    return {"float32": result_float32, "float64": result_float64}
