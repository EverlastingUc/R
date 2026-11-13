"""Centered finite-difference gradient checking for Entropy networks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from nn.backprop import backprop_mse
from nn.losses import mean_squared_error
from nn.network import FeedForwardNetwork


@dataclass(frozen=True)
class GradientCheckResult:
    """Summary of analytic-versus-numeric gradient comparison."""

    passed: bool
    max_relative_error: float
    tolerance: float
    parameter_count: int


def gradient_check(
    network: FeedForwardNetwork,
    inputs: ArrayLike,
    targets: ArrayLike,
    epsilon: float = 1e-6,
    tolerance: float = 1e-5,
) -> GradientCheckResult:
    """Compare every weight/bias gradient with centered finite differences.

    Relative error is ``|analytic - numeric| / max(|analytic|, |numeric|, 1e-12)``.
    Parameters are restored after each perturbation. The check uses MSE because
    its derivative is implemented independently in the backpropagation path.
    """
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")

    input_values = np.asarray(inputs, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.float64)
    analytic = backprop_mse(network, input_values, target_values)
    parameter_arrays = [
        parameter
        for layer in network.layers
        for parameter in (layer.weights, layer.bias)
    ]
    gradient_arrays = [
        gradient
        for weight_gradient, bias_gradient in zip(
            analytic.weight_gradients, analytic.bias_gradients, strict=True
        )
        for gradient in (weight_gradient, bias_gradient)
    ]

    max_relative_error = 0.0
    parameter_count = 0
    for parameters, analytic_gradient in zip(
        parameter_arrays, gradient_arrays, strict=True
    ):
        for index in np.ndindex(parameters.shape):
            original = float(parameters[index])
            try:
                parameters[index] = original + epsilon
                positive_loss = mean_squared_error(
                    network.forward(input_values), target_values
                )
                parameters[index] = original - epsilon
                negative_loss = mean_squared_error(
                    network.forward(input_values), target_values
                )
            finally:
                parameters[index] = original

            numerical_gradient = (positive_loss - negative_loss) / (2.0 * epsilon)
            analytical_gradient = float(analytic_gradient[index])
            denominator = max(abs(analytical_gradient), abs(numerical_gradient), 1e-12)
            relative_error = abs(analytical_gradient - numerical_gradient) / denominator
            max_relative_error = max(max_relative_error, relative_error)
            parameter_count += 1

    return GradientCheckResult(
        passed=max_relative_error <= tolerance,
        max_relative_error=max_relative_error,
        tolerance=tolerance,
        parameter_count=parameter_count,
    )
