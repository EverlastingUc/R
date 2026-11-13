"""Gradient-descent helpers for known scalar and vector objectives."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Objective = Callable[[FloatArray], float]
Gradient = Callable[[FloatArray], ArrayLike]


@dataclass(frozen=True)
class GradientDescentResult:
    """Final parameter vector and objective values, including the start."""

    parameters: FloatArray
    loss_history: FloatArray


def gradient_descent(
    objective: Objective,
    gradient: Gradient,
    initial_parameters: ArrayLike,
    learning_rate: float,
    steps: int,
) -> GradientDescentResult:
    """Minimize an objective by repeated ``parameters -= rate * gradient``.

    Parameters must have shape ``(n,)``. The returned loss history has shape
    ``(steps + 1,)`` and includes the initial objective value.
    """
    parameters = np.asarray(initial_parameters, dtype=np.float64)
    if parameters.ndim != 1 or parameters.size == 0:
        raise ValueError("initial_parameters must be a non-empty vector")
    if not np.isfinite(parameters).all():
        raise ValueError("initial_parameters must be finite")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive finite")
    if steps < 1:
        raise ValueError("steps must be positive")

    parameters = parameters.copy()
    losses = np.empty(steps + 1, dtype=np.float64)
    losses[0] = float(objective(parameters))
    if not math.isfinite(losses[0]):
        raise ValueError("objective must return a finite value")

    for step in range(steps):
        gradient_values = np.asarray(gradient(parameters), dtype=np.float64)
        if gradient_values.shape != parameters.shape:
            raise ValueError("gradient must have the same shape as parameters")
        if not np.isfinite(gradient_values).all():
            raise ValueError("gradient must be finite")
        parameters -= learning_rate * gradient_values
        losses[step + 1] = float(objective(parameters))
        if not math.isfinite(losses[step + 1]):
            raise ValueError("objective became non-finite during optimization")

    return GradientDescentResult(parameters=parameters, loss_history=losses)
