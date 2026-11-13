"""Output and derivative checks for neural-network activations."""

import numpy as np
import pytest

from nn.activations import (
    activation_derivative,
    apply_activation,
    softmax_jacobian,
)


@pytest.mark.parametrize(
    ("name", "values", "expected"),
    [
        ("sigmoid", [0.0], [0.5]),
        ("tanh", [0.0], [0.0]),
        ("relu", [-1.0, 0.0, 2.0], [0.0, 0.0, 2.0]),
        ("leaky_relu", [-2.0, 0.0, 2.0], [-0.02, 0.0, 2.0]),
        ("identity", [-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]),
    ],
)
def test_elementwise_activation_outputs(name, values, expected) -> None:
    np.testing.assert_allclose(apply_activation(name, values), expected, atol=1e-12)


def test_softmax_outputs_match_stable_reference_and_sum_to_one() -> None:
    logits = np.array([[1000.0, 1001.0, 999.0], [0.0, 0.0, 0.0]])
    probabilities = apply_activation("softmax", logits)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0])
    assert probabilities[0, 1] > probabilities[0, 0] > probabilities[0, 2]


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "relu", "leaky_relu"])
def test_elementwise_derivatives_match_finite_differences(name: str) -> None:
    values = np.array([-1.3, 0.4, 1.7], dtype=np.float64)
    step = 1e-6
    numeric = np.empty_like(values)
    for index in range(values.size):
        positive = values.copy()
        negative = values.copy()
        positive[index] += step
        negative[index] -= step
        numeric[index] = (
            apply_activation(name, positive)[index]
            - apply_activation(name, negative)[index]
        ) / (2.0 * step)
    np.testing.assert_allclose(
        activation_derivative(name, values), numeric, rtol=1e-6, atol=1e-8
    )


def test_softmax_jacobian_matches_finite_differences() -> None:
    logits = np.array([0.2, -0.4, 1.1], dtype=np.float64)
    analytic = softmax_jacobian(logits)
    step = 1e-6
    numeric = np.empty_like(analytic)
    for input_index in range(logits.size):
        positive = logits.copy()
        negative = logits.copy()
        positive[input_index] += step
        negative[input_index] -= step
        numeric[:, input_index] = (
            apply_activation("softmax", positive)
            - apply_activation("softmax", negative)
        ) / (2.0 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)


def test_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="activation"):
        apply_activation("unknown", [1.0])


def test_softmax_elementwise_derivative_requires_jacobian() -> None:
    with pytest.raises(ValueError, match="softmax_jacobian"):
        activation_derivative("softmax", [0.0, 1.0])
